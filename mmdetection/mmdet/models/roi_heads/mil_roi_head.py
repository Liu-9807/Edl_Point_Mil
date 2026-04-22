import torch
from mmcv.ops import batched_nms

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import scale_boxes
from mmengine.structures import InstanceData

from mmdet.models.task_modules import build_sampler, build_prior_generator
from .standard_roi_head import StandardRoIHead



@MODELS.register_module()
class MILRoIHead(StandardRoIHead):
    """MIL RoI head for point-based EDL."""
    def __init__(self, proposal_generator, **kwargs):
        super(MILRoIHead, self).__init__(**kwargs)
        # 1. 构建
        self.proposal_generator = build_prior_generator(proposal_generator)

        # 添加用于累积整个epoch统计数据的列表
        self.epoch_logits_ins = []
        self.epoch_ins_labels = []
        # 添加用于跟踪当前epoch的属性
        self.current_epoch = 0

    # -------------------------
    # MMDetection 3.x 必需接口
    # -------------------------
    def loss(self, x, batch_data_samples, **kwargs):
        """适配新接口，解包 batch_data_samples 后调用旧的 forward_train。"""
        img_metas = [s.metainfo for s in batch_data_samples]

        # 解包 GT
        gt_bboxes = [s.gt_instances.bboxes for s in batch_data_samples]
        gt_labels = [s.gt_instances.labels for s in batch_data_samples]
        gt_bboxes_ignore = [
            getattr(getattr(s, 'ignored_instances', None), 'bboxes', None)
            for s in batch_data_samples
        ]
        # 可选点标注
        gt_points = [
            getattr(getattr(s, 'gt_instances', None), 'points', None)
            for s in batch_data_samples
        ]

        # # 尽量获取原始图像 tensor，供可视化/缓存使用
        # img = kwargs.get('batch_inputs', kwargs.get('imgs', None))

        return self.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_points=gt_points,
            **kwargs)

    def predict(self, x, batch_data_samples, rescale: bool = True, **kwargs):
        """
        推理逻辑：
        1. 接收主要输入的点 (prompts)。
        2. 在点周围生成密集的候选框 (Bag)。
        3. 前向传播获取 EDL Evidence。
        4. 挑选最佳框。
        """
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        
        # 结果列表
        results_list = []

        for i, img_meta in enumerate(batch_img_metas):
            # 获取该图的用户输入点
            # 假设 batch_data_samples 中携带了 gt_instances 作为提示点 (prompts)
            # 在实际推理中，这里可能来自用户的点击交互
            data_sample = batch_data_samples[i]
            
            if hasattr(data_sample, 'gt_instances') and hasattr(data_sample.gt_instances, 'points'):
                prompts = data_sample.gt_instances.points # [N_points, 2]
            else:
                # 如果没有提供点，回退到全图滑动窗口或报错，这里假设必须有点
                empty_res = DetDataSample()
                empty_res.set_metainfo(img_meta)
                empty_res.pred_instances = InstanceData(
                    bboxes=torch.empty((0, 4), device=x[0].device),
                    labels=torch.empty((0,), dtype=torch.long, device=x[0].device),
                    scores=torch.empty((0,), device=x[0].device),
                )
                results_list.append(empty_res)
                continue

            # --- 步骤 1: 密集采样 (Jittered Proposals) ---
            # 针对每个点，生成 K 个候选框
            proposals, keep_indices = self._generate_jittered_proposals(
                prompts, img_meta['img_shape']
            )
            # proposals: [Total_N, 4], keep_indices: 用于记录哪个框属于哪个点
            
            if proposals.shape[0] == 0:
                empty_res = DetDataSample()
                empty_res.set_metainfo(img_meta)
                empty_res.pred_instances = InstanceData(
                    bboxes=torch.empty((0, 4), device=x[0].device),
                    labels=torch.empty((0,), dtype=torch.long, device=x[0].device),
                    scores=torch.empty((0,), device=x[0].device),
                )
                results_list.append(empty_res)
                continue

            # 构建 RoIs
            rois = bbox2roi([proposals]) # 这里的 batch_idx 都是 0，因为我们在循环里单张处理
            
            # --- 步骤 2: 前向推理 ---
            # trick: 为了复用 _bbox_forward，重新组装 rois 的 batch_idx
            # 这里的 x 是整个 batch 的特征，我们需要取当前第 i 张图的特征
            # 并增加一个维度 [1, C, H, W]
            feature_i = [f[i:i+1] for f in x] 
            
            # 运行 Head
            # bag_score 是整包得分（没用），ins_score 是我们要的
            _, ins_score = self._bbox_forward(feature_i, rois)

            # 读取 EDLHead 在推理阶段缓存的 RoI 2D mask，并基于阈值分割细化框。
            mask_2d = getattr(self.bbox_head, '_last_infer_mask_2d', None)
            
            # 处理 ins_score 可能是 (init, enhanced) 的情况
            if isinstance(ins_score, tuple):
                ins_score = ins_score[1] # 取 Enhanced Alpha

            # --- 步骤 3: 解码 EDL 输出 ---
            # Alpha -> Belief (Probability)
            # alpha = [N_proposals, Num_Classes]
            # S = sum(alpha, dim=1)
            # belief = (alpha - 1) / S
            # uncertainty = K / S
            
            S = torch.sum(ins_score, dim=1, keepdim=True)
            probs = ins_score / S # 或者用 (alpha - 1) / S，看你训练时的设定，通常推理直接归一化 Alpha 即可近似概率
            
            # [修改] 仅针对正类得分 (假设 Index 0 为背景) 进行判断
            # score_thr 可以通过 kwargs 传入，默认 0.05
            cfg_score_thr = None
            cfg_nms = None
            cfg_max_per_img = 100
            cfg_postprocess_strategy = 'nms'
            cfg_weighted_iou_thr = 0.55
            cfg_weighted_score_type = 'max'
            cfg_score_mode = 'max_class'
            cfg_per_point_topk = 1
            cfg_min_alpha_sum = 0.0
            test_cfg = getattr(self, 'test_cfg', None)
            rcnn_cfg = None
            if test_cfg is not None:
                if isinstance(test_cfg, dict):
                    rcnn_cfg = test_cfg.get('rcnn', None)
                else:
                    rcnn_cfg = getattr(test_cfg, 'rcnn', None)
            if rcnn_cfg is not None:
                cfg_score_thr = rcnn_cfg.get('score_thr', None)
                cfg_nms = rcnn_cfg.get('nms', None)
                cfg_max_per_img = rcnn_cfg.get('max_per_img', cfg_max_per_img)
                cfg_postprocess_strategy = rcnn_cfg.get('postprocess_strategy', cfg_postprocess_strategy)
                cfg_weighted_iou_thr = rcnn_cfg.get('weighted_iou_thr', cfg_weighted_iou_thr)
                cfg_weighted_score_type = rcnn_cfg.get('weighted_score_type', cfg_weighted_score_type)
                cfg_score_mode = rcnn_cfg.get('score_mode', cfg_score_mode)
                cfg_per_point_topk = rcnn_cfg.get('per_point_topk', cfg_per_point_topk)
                cfg_min_alpha_sum = rcnn_cfg.get('min_alpha_sum', cfg_min_alpha_sum)
            score_thr = kwargs.get('score_thr', cfg_score_thr if cfg_score_thr is not None else 0.05)
            mask_thr = kwargs.get('mask_thr', rcnn_cfg.get('mask_thr', 0.5) if rcnn_cfg is not None else 0.5)
            mask_min_area = kwargs.get('mask_min_area', rcnn_cfg.get('mask_min_area', 4) if rcnn_cfg is not None else 4)
            mask_fallback_to_proposal = kwargs.get(
                'mask_fallback_to_proposal',
                rcnn_cfg.get('mask_fallback_to_proposal', True) if rcnn_cfg is not None else True)
            postprocess_strategy = kwargs.get('postprocess_strategy', cfg_postprocess_strategy)
            weighted_iou_thr = kwargs.get('weighted_iou_thr', cfg_weighted_iou_thr)
            weighted_score_type = kwargs.get('weighted_score_type', cfg_weighted_score_type)
            score_mode = kwargs.get('score_mode', cfg_score_mode)
            per_point_topk = kwargs.get('per_point_topk', cfg_per_point_topk)
            min_alpha_sum = kwargs.get('min_alpha_sum', cfg_min_alpha_sum)

            refined_bboxes, mask_valid = self._refine_boxes_with_mask(
                proposals=proposals,
                mask_2d=mask_2d,
                img_shape=img_meta['img_shape'],
                mask_thr=mask_thr,
                mask_min_area=mask_min_area,
                fallback_to_proposal=mask_fallback_to_proposal)

            # 避免缓存跨图残留。
            if hasattr(self.bbox_head, '_last_infer_mask_2d'):
                del self.bbox_head._last_infer_mask_2d

            if probs.shape[1] > 1 and score_mode == 'exclude_class0':
                # 兼容旧逻辑：若 class 0 为背景，仅在正类中取最大分。
                pos_probs = probs[:, 1:]
                scores, tmp_labels = torch.max(pos_probs, dim=1)
                labels = tmp_labels + 1
            else:
                # 默认：在全部类别上取最大分。适配 mmdet 常规数据集(不显式包含背景类)。
                scores, labels = torch.max(probs, dim=1)
            
            # --- 步骤 4: 筛选策略 ---
            # 修改：依据特定的确定性阈值筛选各参考点对应所有达标的框，若无达标则输出空
            
            final_bboxes = []
            final_labels = []
            final_scores = []
            
            num_points = prompts.shape[0]
            
            for pt_idx in range(num_points):
                # 找到属于当前点的所有 proposal 的索引
                mask = (keep_indices == pt_idx)
                if not mask.any():
                    continue
                
                # 获取这组框的分数、标签和坐标
                subset_scores = scores[mask]     # [K]
                subset_labels = labels[mask]     # [K]
                subset_bboxes = refined_bboxes[mask]  # [K, 4]
                subset_mask_valid = mask_valid[mask]  # [K]
                subset_alpha_sum = S[mask].squeeze(1)  # [K]
                
                # 筛选：保留分数超过阈值的框
                valid_mask = (subset_scores > score_thr) & subset_mask_valid
                if min_alpha_sum > 0:
                    valid_mask = valid_mask & (subset_alpha_sum >= min_alpha_sum)
                
                if valid_mask.any():
                    valid_bboxes = subset_bboxes[valid_mask]
                    valid_labels = subset_labels[valid_mask]
                    valid_scores = subset_scores[valid_mask]

                    # 每个提示点只保留 Top-K 候选，抑制点级别冗余。
                    if per_point_topk is not None and int(per_point_topk) > 0 and valid_scores.numel() > int(per_point_topk):
                        topk_scores, topk_inds = torch.topk(valid_scores, k=int(per_point_topk))
                        valid_bboxes = valid_bboxes[topk_inds]
                        valid_labels = valid_labels[topk_inds]
                        valid_scores = topk_scores

                    final_bboxes.append(valid_bboxes)
                    final_labels.append(valid_labels)
                    final_scores.append(valid_scores)
                # 若没有通过阈值的，该点不贡献任何框

            if len(final_bboxes) > 0:
                # 使用 cat 拼接不同数量的框 (注意原代码是 stack，这里要改为 cat)
                final_bboxes = torch.cat(final_bboxes, dim=0)
                final_labels = torch.cat(final_labels, dim=0)
                final_scores = torch.cat(final_scores, dim=0)

                final_bboxes, final_scores, final_labels = self._postprocess_detections(
                    bboxes=final_bboxes,
                    scores=final_scores,
                    labels=final_labels,
                    nms_cfg=cfg_nms,
                    max_per_img=cfg_max_per_img,
                    strategy=postprocess_strategy,
                    weighted_iou_thr=weighted_iou_thr,
                    weighted_score_type=weighted_score_type)

                if final_bboxes.size(0) > cfg_max_per_img:
                    topk_scores, topk_inds = torch.topk(final_scores, k=cfg_max_per_img)
                    final_bboxes = final_bboxes[topk_inds]
                    final_labels = final_labels[topk_inds]
                    final_scores = topk_scores
                
                # Rescale 回原图尺寸
                if rescale and final_bboxes.numel() > 0:
                    sf = img_meta.get('scale_factor', None)
                    if sf is not None:
                        if isinstance(sf, torch.Tensor):
                            sf = sf.detach().cpu().tolist()
                        # MMDet meta 中 scale_factor 常见为 tuple/list: (w_scale, h_scale)
                        # 或 (w_scale, h_scale, w_scale, h_scale)。统一取前两个维度。
                        if isinstance(sf, (list, tuple)) and len(sf) >= 2:
                            w_scale, h_scale = float(sf[0]), float(sf[1])
                            inv_sf = (1.0 / w_scale, 1.0 / h_scale)
                            final_bboxes = scale_boxes(final_bboxes, inv_sf)
                
                # 封装结果
                res = DetDataSample()
                res.set_metainfo(img_meta)
                # results.bboxes, results.scores, results.labels
                inst = InstanceData()
                inst.bboxes = final_bboxes
                inst.labels = final_labels
                inst.scores = final_scores
                res.pred_instances = inst
                results_list.append(res)
            else:
                empty_res = DetDataSample()
                empty_res.set_metainfo(img_meta)
                empty_res.pred_instances = InstanceData(
                    bboxes=torch.empty((0, 4), device=x[0].device),
                    labels=torch.empty((0,), dtype=torch.long, device=x[0].device),
                    scores=torch.empty((0,), device=x[0].device),
                )
                results_list.append(empty_res)

        return results_list

    def _postprocess_detections(self,
                                bboxes,
                                scores,
                                labels,
                                nms_cfg=None,
                                max_per_img=100,
                                strategy='nms',
                                weighted_iou_thr=0.55,
                                weighted_score_type='max'):
        """Post-process detections with configurable strategy.

        Args:
            strategy: 'nms' | 'weighted' | 'none'.
        """
        if bboxes.numel() == 0:
            return bboxes, scores, labels

        if strategy in ('weighted', 'weighted_nms'):
            fused_bboxes, fused_scores, fused_labels = self._weighted_box_fusion(
                bboxes=bboxes,
                scores=scores,
                labels=labels,
                iou_thr=weighted_iou_thr,
                max_per_img=max_per_img,
                score_type=weighted_score_type)

            if strategy == 'weighted':
                return fused_bboxes, fused_scores, fused_labels

            # weighted_nms: 先融合再做一次 NMS，进一步移除近邻冗余框。
            if nms_cfg is not None and fused_bboxes.numel() > 0:
                dets, keep = batched_nms(fused_bboxes, fused_scores, fused_labels, nms_cfg)
                keep = keep[:max_per_img]
                out_bboxes = dets[:len(keep), :4]
                out_scores = dets[:len(keep), 4]
                out_labels = fused_labels[keep]
                return out_bboxes, out_scores, out_labels
            return fused_bboxes, fused_scores, fused_labels

        if strategy == 'none':
            return bboxes, scores, labels

        # Default strategy: NMS
        if nms_cfg is not None:
            dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
            keep = keep[:max_per_img]
            out_bboxes = dets[:len(keep), :4]
            out_scores = dets[:len(keep), 4]
            out_labels = labels[keep]
            return out_bboxes, out_scores, out_labels

        return bboxes, scores, labels

    def _bbox_iou(self, boxes1, boxes2):
        """Compute IoU matrix between boxes1 and boxes2."""
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return torch.zeros((boxes1.size(0), boxes2.size(0)), device=boxes1.device, dtype=boxes1.dtype)

        x1 = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

        inter_w = (x2 - x1).clamp(min=0)
        inter_h = (y2 - y1).clamp(min=0)
        inter = inter_w * inter_h

        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
        union = area1[:, None] + area2[None, :] - inter

        return inter / union.clamp(min=1e-6)

    def _weighted_box_fusion(self,
                             bboxes,
                             scores,
                             labels,
                             iou_thr=0.55,
                             max_per_img=100,
                             score_type='max'):
        """Greedy weighted box fusion per class."""
        unique_labels = labels.unique(sorted=True)
        fused_bboxes = []
        fused_scores = []
        fused_labels = []

        for cls in unique_labels:
            cls_mask = labels == cls
            cls_boxes = bboxes[cls_mask]
            cls_scores = scores[cls_mask]

            if cls_boxes.numel() == 0:
                continue

            order = torch.argsort(cls_scores, descending=True)
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            while cls_boxes.size(0) > 0:
                ref_box = cls_boxes[0:1]
                ious = self._bbox_iou(ref_box, cls_boxes).squeeze(0)
                cluster_mask = ious >= iou_thr

                cluster_boxes = cls_boxes[cluster_mask]
                cluster_scores = cls_scores[cluster_mask]

                weights = cluster_scores / cluster_scores.sum().clamp(min=1e-6)
                merged_box = (cluster_boxes * weights[:, None]).sum(dim=0)

                if score_type == 'avg':
                    merged_score = cluster_scores.mean()
                else:
                    merged_score = cluster_scores.max()

                fused_bboxes.append(merged_box)
                fused_scores.append(merged_score)
                fused_labels.append(cls)

                keep_mask = ~cluster_mask
                cls_boxes = cls_boxes[keep_mask]
                cls_scores = cls_scores[keep_mask]

        if len(fused_bboxes) == 0:
            return (
                torch.empty((0, 4), device=bboxes.device, dtype=bboxes.dtype),
                torch.empty((0,), device=scores.device, dtype=scores.dtype),
                torch.empty((0,), device=labels.device, dtype=labels.dtype)
            )

        fused_bboxes = torch.stack(fused_bboxes, dim=0)
        fused_scores = torch.stack(fused_scores, dim=0)
        fused_labels = torch.stack(fused_labels, dim=0)

        order = torch.argsort(fused_scores, descending=True)
        order = order[:max_per_img]
        return fused_bboxes[order], fused_scores[order], fused_labels[order]

    def _refine_boxes_with_mask(self,
                                proposals,
                                mask_2d,
                                img_shape,
                                mask_thr=0.5,
                                mask_min_area=4,
                                fallback_to_proposal=True):
        """Use thresholded RoI mask to localize object and refine each proposal box."""
        num_props = proposals.size(0)
        if num_props == 0:
            return proposals, torch.empty((0,), dtype=torch.bool, device=proposals.device)

        refined = proposals.clone()
        valid = torch.ones((num_props,), dtype=torch.bool, device=proposals.device)

        if mask_2d is None or not isinstance(mask_2d, torch.Tensor) or mask_2d.size(0) != num_props:
            return refined, valid

        if mask_2d.device != proposals.device:
            mask_2d = mask_2d.to(proposals.device)

        img_h, img_w = img_shape[:2]
        eps = 1e-6

        for idx in range(num_props):
            roi = proposals[idx]
            roi_mask = mask_2d[idx]
            if roi_mask.dim() == 3:
                roi_mask = roi_mask.mean(dim=0)

            bin_mask = roi_mask > mask_thr
            if int(bin_mask.sum().item()) < int(mask_min_area):
                if fallback_to_proposal:
                    refined[idx] = roi
                else:
                    valid[idx] = False
                continue

            ys, xs = torch.where(bin_mask)
            if ys.numel() == 0 or xs.numel() == 0:
                if fallback_to_proposal:
                    refined[idx] = roi
                else:
                    valid[idx] = False
                continue

            roi_w = torch.clamp(roi[2] - roi[0], min=eps)
            roi_h = torch.clamp(roi[3] - roi[1], min=eps)
            m_h = roi_mask.size(0)
            m_w = roi_mask.size(1)

            local_x1 = xs.min().to(dtype=roi.dtype) / float(m_w) * roi_w
            local_y1 = ys.min().to(dtype=roi.dtype) / float(m_h) * roi_h
            local_x2 = (xs.max().to(dtype=roi.dtype) + 1.0) / float(m_w) * roi_w
            local_y2 = (ys.max().to(dtype=roi.dtype) + 1.0) / float(m_h) * roi_h

            x1 = torch.clamp(roi[0] + local_x1, min=0.0, max=float(img_w))
            y1 = torch.clamp(roi[1] + local_y1, min=0.0, max=float(img_h))
            x2 = torch.clamp(roi[0] + local_x2, min=0.0, max=float(img_w))
            y2 = torch.clamp(roi[1] + local_y2, min=0.0, max=float(img_h))

            if (x2 - x1) <= 1 or (y2 - y1) <= 1:
                if fallback_to_proposal:
                    refined[idx] = roi
                else:
                    valid[idx] = False
                continue

            refined[idx] = torch.stack([x1, y1, x2, y2])

        return refined, valid

    def _generate_jittered_proposals(self, points, img_shape):
        """
        核心生成器：解决点在“末端”的问题。
        对于每个点，生成多尺度、多偏移的框。
        """
        h, w = img_shape[:2]
        proposals = []
        point_indices = []
        
        # 预设尺度 (Scales) 和 比例 (Ratios)
        # 假设物体大小从 32 到 128 像素不等
        base_scales = [32, 64, 128, 256] 
        ratios = [0.5, 1.0, 2.0]
        
        # 关键：偏移系数。
        # (0,0) = 点在中心
        # (-0.5, -0.5) = 点在框的右下角 (框向左上偏)
        # (0.5, 0.5) = 点在框的左上角
        # (-0.5, 0.5) = 点在框的右上角
        # (0.5, -0.5) = 点在框的左下角
        anchor_offsets = [
            (0, 0),         # Center
            (-0.4, -0.4),   # Top-Left Shift
            (0.4, 0.4),     # Bottom-Right Shift
            (-0.4, 0.4),    # Top-Right Shift
            (0.4, -0.4)     # Bottom-Left Shift
        ]

        for idx, pt in enumerate(points):
            px, py = pt[0], pt[1]
            
            for scale in base_scales:
                for ratio in ratios:
                    # 计算宽和高
                    h_box = scale * torch.sqrt(torch.tensor(ratio))
                    w_box = scale / torch.sqrt(torch.tensor(ratio))
                    
                    for off_x, off_y in anchor_offsets:
                        # 计算中心点 center_x, center_y 基于 offset
                        # 如果 off_x = 0.5, 说明点在左边，中心应该在点右边 (+0.5 * w)
                        cx = px + off_x * w_box
                        cy = py + off_y * h_box
                        
                        x1 = cx - w_box / 2
                        y1 = cy - h_box / 2
                        x2 = cx + w_box / 2
                        y2 = cy + h_box / 2
                        
                        # Clip
                        x1 = max(0, min(w, x1))
                        y1 = max(0, min(h, y1))
                        x2 = max(0, min(w, x2))
                        y2 = max(0, min(h, y2))
                        
                        if (x2 - x1) > 5 and (y2 - y1) > 5:
                            proposals.append([x1, y1, x2, y2])
                            point_indices.append(idx)

        if not proposals:
            return torch.empty((0, 4), device=points.device), torch.empty(0, device=points.device)
            
        return (
            torch.tensor(proposals, device=points.device, dtype=points.dtype),
            torch.tensor(point_indices, device=points.device, dtype=torch.long),
        )

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_sampler = None
        self.bbox_assigner = None

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        bbox_feats = self.bbox_roi_extractor(x, rois.to(x[0].device))
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        # 将 rois 也传递给 bbox_head
        bag_score, ins_score = self.bbox_head(bbox_feats, rois)
        return bag_score, ins_score

    def _calculate_mil_accuracy(self, bag_score, bag_label, ins_score, ins_label):
        """
        计算 MIL 训练过程中的准确率指标以及混淆矩阵统计 (TP, TN, FP, FN)。
        处理 EDL 可能返回的 tuple 输出，通常取最后一个元素(alpha或probs)进行评估。
        """
        metrics = {}
        
        def calculate_confusion_metrics(pred, target, prefix):
            # 这是一个二分类或多分类问题。对于多分类，这里统计的是"预测正确"与否的宏观统计，
            # 或者如果是二分类(0背景, 1前景)，我们可以更精确地计算。
            # 这里的实现假设：
            # 1. 如果是多分类，将 Class 0 视为负类(背景/健康)，Class >= 1 视为正类(缺陷)。
            # 2. 如果不需要区分具体的正类类别，统称为 Positive。
            
            # 将预测和标签转换为 0(Negative) 和 1(Positive)
            # 注意：需根据实际业务逻辑调整。这里假设 label=0 是负样本，label>0 是正样本。
            pred_binary = (pred > 0).long()
            target_binary = (target > 0).long()
            
            # 计算总数
            total = target.numel()
            if total == 0:
                return {}

            tp = ((pred_binary == 1) & (target_binary == 1)).float().sum()
            tn = ((pred_binary == 0) & (target_binary == 0)).float().sum()
            fp = ((pred_binary == 1) & (target_binary == 0)).float().sum()
            fn = ((pred_binary == 0) & (target_binary == 1)).float().sum()
            
            # 计算百分比
            return {
                f'{prefix}_tp_pct': (tp / total) * 100,
                f'{prefix}_tn_pct': (tn / total) * 100,
                f'{prefix}_fp_pct': (fp / total) * 100,
                f'{prefix}_fn_pct': (fn / total) * 100
            }

        # --- Bag Level Accuracy & Stats ---
        if bag_score is not None and bag_label is not None:
            # 处理可能的 Tuple 输出 (如 EDL 返回 evidence, alpha)
            if isinstance(bag_score, (tuple, list)):
                # 假设 Logits/Alpha 位于最后 (参考 accumulation 逻辑中取 [1] 或类似)
                val_bag = bag_score[-1] 
            else:
                val_bag = bag_score
            
            with torch.no_grad():
                pred_bag = torch.argmax(val_bag, dim=1)
                acc_bag = (pred_bag == bag_label).float().mean() * 100
                metrics['acc_bag'] = acc_bag
                
                # 计算 Bag 级别的 TP/TN/FP/FN
                metrics.update(calculate_confusion_metrics(pred_bag, bag_label, 'bag'))

        # --- Instance Level Accuracy & Stats ---
        if ins_score is not None and ins_label is not None:
            if isinstance(ins_score, (tuple, list)):
                val_ins = ins_score[1]  # 增强后的 alpha 在第二位
            else:
                val_ins = ins_score
                
            with torch.no_grad():
                pred_ins = torch.argmax(val_ins, dim=1)
                acc_ins = (pred_ins == ins_label).float().mean() * 100
                metrics['acc_instance'] = acc_ins
                
                # 计算 Instance 级别的 TP/TN/FP/FN
                metrics.update(calculate_confusion_metrics(pred_ins, ins_label, 'ins'))
                
        return metrics


    def forward_train(self, x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs):
        
        # # 从 kwargs 中获取 epoch_num，如果不存在则默认为 self.current_epoch
        epoch_num = kwargs.get('epoch_num', self.current_epoch)
        
        # 从 kwargs 中获取 img
        img = kwargs.get('img')
        gt_points = kwargs.get('gt_points')

        batch_bag_bboxes = []
        batch_instance_labels = []
        batch_bag_labels = []
        for i, img_meta in enumerate(img_metas):
            # 2. 调用 generator
            _, _, pseudo_bboxes, pseudo_labels, bag_label = self.proposal_generator(
                img_meta, 
                gt_points=gt_points[i] if gt_points else None
            )
            
            # 3. 后续进行 MIL Bag 的组装 (依然保留 MIL 特有的 Bag 逻辑)
            batch_bag_bboxes.append(pseudo_bboxes)
            batch_instance_labels.append(pseudo_labels)
            batch_bag_labels.append(torch.full((1,), bag_label, dtype=torch.long, device=pseudo_bboxes.device))

        # [新增] 仅在此时暂存数据用于 Hook 可视化，用完即删，避免显存泄漏
        if getattr(self, 'debug_proposal_vis', False):
            self._last_proposal_debug = {
                'img_metas': img_metas,
                'batch_bag_bboxes': batch_bag_bboxes, # list of tensors
                'batch_instance_labels': batch_instance_labels, # <--- 新增这行
                'gt_points': gt_points,
                'batch_bag_labels': batch_bag_labels
            }
        # 2. bbox2roi可以直接处理bbox的列表
        rois = bbox2roi(batch_bag_bboxes)


        
        # _bbox_forward calls self.bbox_head.forward() and gets the results
        bag_score, ins_score = self._bbox_forward(x, rois)

        # 4. 将标签列表拼接成一个Tensor以计算loss
        bag_labels = torch.cat(batch_bag_labels)
        ins_labels = torch.cat(batch_instance_labels)

        # [修改开始] --- 积累数据供 Epoch Hook 使用 ---
        # 确保只在训练模式下积累，且 ins_score 存在
        if ins_score is not None:
            # ins_score 可能是 Tuple (init_alpha, enhanced_alpha)，如果是，取增强后的
            if isinstance(ins_score, tuple):
                cur_ins_scores = ins_score[1] # enhanced
            else:
                cur_ins_scores = ins_score
            
            # 必须 detach 并转 CPU，否则显存会炸
            self.epoch_logits_ins.append(cur_ins_scores.detach().cpu())
            # ins_labels 是这一步产生的伪标签或者真实标签
            self.epoch_ins_labels.append(ins_labels.detach().cpu())
        # [修改结束] ---

        # Pass the results and other labels to the loss function
        losses = self.bbox_head.loss(
            cls_score=(bag_score, ins_score),
            bag_label=bag_labels,
            ins_labels=ins_labels,
            epoch_num=epoch_num
        )

        # [新增] 计算准确率指标并加入损失字典
        # 这样这些指标就会作为 log_vars 自动显示在训练日志中
        acc_metrics = self._calculate_mil_accuracy(bag_score, bag_labels, ins_score, ins_labels)
        losses.update(acc_metrics)

        return losses
