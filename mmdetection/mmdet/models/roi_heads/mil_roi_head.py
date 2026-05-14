import torch
from mmcv.ops import batched_nms

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import scale_boxes
from mmengine.structures import InstanceData

from mmdet.models.task_modules import build_sampler, build_prior_generator
from mmdet.models.utils.mil_jittered_proposals import generate_jittered_proposals
from .standard_roi_head import StandardRoIHead



@MODELS.register_module()
class MILRoIHead(StandardRoIHead):
    """MIL RoI head for point-based EDL.

    Args:
        mil_repeat_fpn_for_bags (bool): If True, expand FPN batch by repeating each
            image's features once per MIL bag and run a single ``_bbox_forward`` (Path A:
            higher FPN VRAM, one autograd segment). If False (default), one forward per
            bag with sliced features (Path B: lower FPN VRAM, multiple segments).
    """
    def __init__(self, proposal_generator,
                 infer_base_scales=None, infer_ratios=None, infer_anchor_offsets=None,
                 mil_repeat_fpn_for_bags=False,
                 **kwargs):
        super(MILRoIHead, self).__init__(**kwargs)
        # 1. 构建
        self.proposal_generator = build_prior_generator(proposal_generator)

        # 推理阶段候选框生成参数
        default_scales = [256]
        default_ratios = [0.5, 1.0, 2.0]
        default_offsets = [(0, 0), (-0.4, -0.4), (0.4, 0.4), (-0.4, 0.4), (0.4, -0.4)]
        self.infer_base_scales = infer_base_scales if infer_base_scales is not None else default_scales
        self.infer_ratios = infer_ratios if infer_ratios is not None else default_ratios
        self.infer_anchor_offsets = infer_anchor_offsets if infer_anchor_offsets is not None else default_offsets
        self.mil_repeat_fpn_for_bags = bool(mil_repeat_fpn_for_bags)

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
        debug_infer_vis = getattr(self, 'debug_infer_vis', False)
        
        # 结果列表
        results_list = []

        for i, img_meta in enumerate(batch_img_metas):
            # 获取该图的用户输入点
            # 假设 batch_data_samples 中携带了 gt_instances 作为提示点 (prompts)
            # 在实际推理中，这里可能来自用户的点击交互
            data_sample = batch_data_samples[i]

            test_cfg = getattr(self, 'test_cfg', None)
            rcnn_cfg = None
            if test_cfg is not None:
                if isinstance(test_cfg, dict):
                    rcnn_cfg = test_cfg.get('rcnn', None)
                else:
                    rcnn_cfg = getattr(test_cfg, 'rcnn', None)
            debug_mask_refine = False
            debug_mask_refine_max_rois = 20
            debug_proposal_scores = False
            debug_proposal_scores_max_rois = 20
            if rcnn_cfg is not None:
                debug_mask_refine = rcnn_cfg.get('debug_mask_refine', debug_mask_refine)
                debug_mask_refine_max_rois = rcnn_cfg.get(
                    'debug_mask_refine_max_rois', debug_mask_refine_max_rois)
                debug_proposal_scores = rcnn_cfg.get('debug_proposal_scores', debug_proposal_scores)
                debug_proposal_scores_max_rois = rcnn_cfg.get(
                    'debug_proposal_scores_max_rois', debug_proposal_scores_max_rois)
            
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
                if debug_infer_vis or debug_mask_refine or debug_proposal_scores:
                    empty_res.set_field(
                        dict(
                            points=None,
                            proposals=None,
                            refined_bboxes=None,
                            final_bboxes=None,
                            mask_refine_debug=[],
                            proposal_score_debug=[],
                        ),
                        'mil_debug',
                        dtype=dict)
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
                if debug_infer_vis or debug_mask_refine or debug_proposal_scores:
                    empty_res.set_field(
                        dict(
                            points=prompts,
                            proposals=proposals,
                            refined_bboxes=None,
                            final_bboxes=None,
                            mask_refine_debug=[],
                            proposal_score_debug=[],
                        ),
                        'mil_debug',
                        dtype=dict)
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
            _, ins_score, _ = self._bbox_forward(feature_i, rois)

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
            cfg_mask_refine_mode = 'fixed'
            cfg_mask_thr_quantile = 0.80
            cfg_mask_thr_min = 0.45
            cfg_mask_thr_max = 0.85
            cfg_mask_area_min_ratio = 0.02
            cfg_mask_area_max_ratio = 0.75
            cfg_mask_quantile_retry_delta = 0.10
            cfg_debug_mask_refine = False
            cfg_debug_mask_refine_max_rois = 20
            cfg_debug_proposal_scores = False
            cfg_debug_proposal_scores_max_rois = 20
            cfg_allow_empty_results = False
            cfg_empty_fallback = 'top1_per_point'
            cfg_empty_fallback_min_score = 0.0
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
                cfg_mask_refine_mode = rcnn_cfg.get('mask_refine_mode', cfg_mask_refine_mode)
                cfg_mask_thr_quantile = rcnn_cfg.get('mask_thr_quantile', cfg_mask_thr_quantile)
                cfg_mask_thr_min = rcnn_cfg.get('mask_thr_min', cfg_mask_thr_min)
                cfg_mask_thr_max = rcnn_cfg.get('mask_thr_max', cfg_mask_thr_max)
                cfg_mask_area_min_ratio = rcnn_cfg.get('mask_area_min_ratio', cfg_mask_area_min_ratio)
                cfg_mask_area_max_ratio = rcnn_cfg.get('mask_area_max_ratio', cfg_mask_area_max_ratio)
                cfg_mask_quantile_retry_delta = rcnn_cfg.get(
                    'mask_quantile_retry_delta', cfg_mask_quantile_retry_delta)
                cfg_debug_mask_refine = rcnn_cfg.get('debug_mask_refine', cfg_debug_mask_refine)
                cfg_debug_mask_refine_max_rois = rcnn_cfg.get(
                    'debug_mask_refine_max_rois', cfg_debug_mask_refine_max_rois)
                cfg_debug_proposal_scores = rcnn_cfg.get(
                    'debug_proposal_scores', cfg_debug_proposal_scores)
                cfg_debug_proposal_scores_max_rois = rcnn_cfg.get(
                    'debug_proposal_scores_max_rois', cfg_debug_proposal_scores_max_rois)
                cfg_allow_empty_results = rcnn_cfg.get(
                    'allow_empty_results', cfg_allow_empty_results)
                cfg_empty_fallback = rcnn_cfg.get('empty_fallback', cfg_empty_fallback)
                cfg_empty_fallback_min_score = rcnn_cfg.get(
                    'empty_fallback_min_score', cfg_empty_fallback_min_score)
            score_thr = kwargs.get('score_thr', cfg_score_thr if cfg_score_thr is not None else 0.05)
            mask_thr = kwargs.get('mask_thr', rcnn_cfg.get('mask_thr', 0.5) if rcnn_cfg is not None else 0.5)
            mask_min_area = kwargs.get('mask_min_area', rcnn_cfg.get('mask_min_area', 4) if rcnn_cfg is not None else 4)
            mask_fallback_to_proposal = kwargs.get(
                'mask_fallback_to_proposal',
                rcnn_cfg.get('mask_fallback_to_proposal', True) if rcnn_cfg is not None else True)
            mask_refine_mode = kwargs.get('mask_refine_mode', cfg_mask_refine_mode)
            mask_thr_quantile = kwargs.get('mask_thr_quantile', cfg_mask_thr_quantile)
            mask_thr_min = kwargs.get('mask_thr_min', cfg_mask_thr_min)
            mask_thr_max = kwargs.get('mask_thr_max', cfg_mask_thr_max)
            mask_area_min_ratio = kwargs.get('mask_area_min_ratio', cfg_mask_area_min_ratio)
            mask_area_max_ratio = kwargs.get('mask_area_max_ratio', cfg_mask_area_max_ratio)
            mask_quantile_retry_delta = kwargs.get(
                'mask_quantile_retry_delta', cfg_mask_quantile_retry_delta)
            postprocess_strategy = kwargs.get('postprocess_strategy', cfg_postprocess_strategy)
            weighted_iou_thr = kwargs.get('weighted_iou_thr', cfg_weighted_iou_thr)
            weighted_score_type = kwargs.get('weighted_score_type', cfg_weighted_score_type)
            score_mode = kwargs.get('score_mode', cfg_score_mode)
            per_point_topk = kwargs.get('per_point_topk', cfg_per_point_topk)
            min_alpha_sum = kwargs.get('min_alpha_sum', cfg_min_alpha_sum)
            debug_mask_refine = kwargs.get('debug_mask_refine', cfg_debug_mask_refine)
            debug_mask_refine_max_rois = kwargs.get(
                'debug_mask_refine_max_rois', cfg_debug_mask_refine_max_rois)
            debug_proposal_scores = kwargs.get(
                'debug_proposal_scores', cfg_debug_proposal_scores)
            debug_proposal_scores_max_rois = kwargs.get(
                'debug_proposal_scores_max_rois', cfg_debug_proposal_scores_max_rois)
            allow_empty_results = kwargs.get('allow_empty_results', cfg_allow_empty_results)
            empty_fallback = kwargs.get('empty_fallback', cfg_empty_fallback)
            empty_fallback_min_score = kwargs.get(
                'empty_fallback_min_score', cfg_empty_fallback_min_score)

            refined_bboxes, mask_valid, mask_refine_debug = self._refine_boxes_with_mask(
                proposals=proposals,
                mask_2d=mask_2d,
                img_shape=img_meta['img_shape'],
                mask_thr=mask_thr,
                mask_min_area=mask_min_area,
                fallback_to_proposal=mask_fallback_to_proposal,
                mask_refine_mode=mask_refine_mode,
                mask_thr_quantile=mask_thr_quantile,
                mask_thr_min=mask_thr_min,
                mask_thr_max=mask_thr_max,
                mask_area_min_ratio=mask_area_min_ratio,
                mask_area_max_ratio=mask_area_max_ratio,
                mask_quantile_retry_delta=mask_quantile_retry_delta,
                debug_mask_refine=debug_mask_refine,
                debug_max_rois=debug_mask_refine_max_rois)

            if mask_refine_debug:
                for item in mask_refine_debug:
                    proposal_idx = int(item.get('index', -1))
                    if proposal_idx >= 0 and proposal_idx < keep_indices.numel():
                        item['point_index'] = int(keep_indices[proposal_idx].item())

            # 避免缓存跨图残留。
            if hasattr(self.bbox_head, '_last_infer_mask_2d'):
                del self.bbox_head._last_infer_mask_2d

            if probs.shape[1] > 1 and score_mode == 'exclude_class0':
                # 兼容旧逻辑：若 class 0 为背景，仅在正类中取最大分。
                pos_probs = probs[:, 1:]
                scores, tmp_labels = torch.max(pos_probs, dim=1)
                # 输出到评估时需要前景类 0-based 索引
                labels = tmp_labels
            else:
                # 默认：在全部类别上取最大分。适配 mmdet 常规数据集(不显式包含背景类)。
                scores, labels = torch.max(probs, dim=1)
            
            num_points = prompts.shape[0]
            proposal_score_debug = None
            if debug_proposal_scores:
                proposal_score_debug = []
                per_point_max = int(debug_proposal_scores_max_rois) if debug_proposal_scores_max_rois is not None else 0
                for pt_idx in range(num_points):
                    pt_mask = (keep_indices == pt_idx)
                    if not pt_mask.any():
                        continue
                    subset_indices = torch.where(pt_mask)[0]
                    subset_scores = scores[pt_mask]
                    subset_labels = labels[pt_mask]
                    subset_bboxes = proposals[pt_mask]
                    if subset_scores.numel() == 0:
                        continue
                    if per_point_max > 0 and subset_scores.numel() > per_point_max:
                        top_scores, top_inds = torch.topk(subset_scores, k=per_point_max)
                        subset_bboxes = subset_bboxes[top_inds]
                        subset_labels = subset_labels[top_inds]
                        subset_scores = top_scores
                        subset_indices = subset_indices[top_inds]
                    top_idx = int(torch.argmax(subset_scores).item())
                    proposal_score_debug.append({
                        'point_index': int(pt_idx),
                        'point': prompts[pt_idx].detach().cpu(),
                        'bboxes': subset_bboxes.detach().cpu(),
                        'scores': subset_scores.detach().cpu(),
                        'labels': subset_labels.detach().cpu(),
                        'proposal_indices': subset_indices.detach().cpu(),
                        'top_index': top_idx,
                    })

            # --- 步骤 4: 筛选策略 ---
            # 修改：依据特定的确定性阈值筛选各参考点对应所有达标的框，若无达标则输出空
            
            final_bboxes = []
            final_labels = []
            final_scores = []
            
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

            # 全图为空时的兜底：避免整套评估出现“whole dataset is empty”。
            if len(final_bboxes) == 0 and not allow_empty_results:
                fallback_mode = str(empty_fallback).lower()
                fallback_min_score = float(empty_fallback_min_score)

                if fallback_mode in ('top1_per_point', 'top1', 'per_point'):
                    for pt_idx in range(num_points):
                        mask = (keep_indices == pt_idx)
                        if not mask.any():
                            continue

                        subset_scores = scores[mask]
                        subset_labels = labels[mask]
                        subset_bboxes = refined_bboxes[mask]
                        subset_mask_valid = mask_valid[mask]

                        if subset_scores.numel() == 0:
                            continue

                        # 优先从 mask 有效候选中取；若全无效则退回全部候选。
                        if subset_mask_valid.any():
                            cand_scores = subset_scores[subset_mask_valid]
                            cand_labels = subset_labels[subset_mask_valid]
                            cand_bboxes = subset_bboxes[subset_mask_valid]
                        else:
                            cand_scores = subset_scores
                            cand_labels = subset_labels
                            cand_bboxes = subset_bboxes

                        if cand_scores.numel() == 0:
                            continue

                        top_idx = int(torch.argmax(cand_scores).item())
                        top_score = cand_scores[top_idx]
                        if top_score < fallback_min_score:
                            continue

                        final_bboxes.append(cand_bboxes[top_idx:top_idx + 1])
                        final_labels.append(cand_labels[top_idx:top_idx + 1])
                        final_scores.append(top_score.view(1))

                elif fallback_mode in ('top1_global', 'global'):
                    if scores.numel() > 0:
                        if mask_valid.any():
                            cand_scores = scores[mask_valid]
                            cand_labels = labels[mask_valid]
                            cand_bboxes = refined_bboxes[mask_valid]
                        else:
                            cand_scores = scores
                            cand_labels = labels
                            cand_bboxes = refined_bboxes

                        if cand_scores.numel() > 0:
                            top_idx = int(torch.argmax(cand_scores).item())
                            top_score = cand_scores[top_idx]
                            if top_score >= fallback_min_score:
                                final_bboxes.append(cand_bboxes[top_idx:top_idx + 1])
                                final_labels.append(cand_labels[top_idx:top_idx + 1])
                                final_scores.append(top_score.view(1))

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

                final_bboxes_debug = final_bboxes
                
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
                if debug_infer_vis or debug_mask_refine or debug_proposal_scores:
                    res.set_field(
                        dict(
                            points=prompts,
                            proposals=proposals,
                            refined_bboxes=refined_bboxes,
                            final_bboxes=final_bboxes_debug,
                            mask_refine_debug=mask_refine_debug,
                            proposal_score_debug=proposal_score_debug,
                        ),
                        'mil_debug',
                        dtype=dict)
                results_list.append(res)
            else:
                empty_res = DetDataSample()
                empty_res.set_metainfo(img_meta)
                empty_res.pred_instances = InstanceData(
                    bboxes=torch.empty((0, 4), device=x[0].device),
                    labels=torch.empty((0,), dtype=torch.long, device=x[0].device),
                    scores=torch.empty((0,), device=x[0].device),
                )
                if debug_infer_vis or debug_mask_refine or debug_proposal_scores:
                    empty_res.set_field(
                        dict(
                            points=prompts,
                            proposals=proposals,
                            refined_bboxes=refined_bboxes,
                            final_bboxes=None,
                            mask_refine_debug=mask_refine_debug,
                            proposal_score_debug=proposal_score_debug,
                        ),
                        'mil_debug',
                        dtype=dict)
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
                                fallback_to_proposal=True,
                                mask_refine_mode='fixed',
                                mask_thr_quantile=0.80,
                                mask_thr_min=0.45,
                                mask_thr_max=0.85,
                                mask_area_min_ratio=0.02,
                                mask_area_max_ratio=0.75,
                                mask_quantile_retry_delta=0.10,
                                debug_mask_refine=False,
                                debug_max_rois=0):
        """Use thresholded RoI mask to localize object and refine each proposal box."""
        num_props = proposals.size(0)
        if num_props == 0:
            return proposals, torch.empty((0,), dtype=torch.bool, device=proposals.device), []

        refined = proposals.clone()
        valid = torch.ones((num_props,), dtype=torch.bool, device=proposals.device)
        debug_items = []
        debug_limit = int(debug_max_rois) if debug_max_rois is not None else 0
        collect_debug = bool(debug_mask_refine)

        if mask_2d is None or not isinstance(mask_2d, torch.Tensor) or mask_2d.size(0) != num_props:
            return refined, valid, debug_items if collect_debug else None

        if mask_2d.device != proposals.device:
            mask_2d = mask_2d.to(proposals.device)

        img_h, img_w = img_shape[:2]
        eps = 1e-6
        mode = str(mask_refine_mode).lower()

        def _append_debug(index, roi_box, refined_box, roi_mask, bin_mask, thr_value,
                          mask_area_value, area_min_value, area_max_value, valid_flag, quantile_value):
            if not collect_debug:
                return
            if debug_limit > 0 and len(debug_items) >= debug_limit:
                return
            debug_items.append({
                'index': int(index),
                'proposal_bbox': roi_box.detach().cpu(),
                'refined_bbox': refined_box.detach().cpu(),
                'roi_mask': roi_mask.detach().cpu(),
                'bin_mask': bin_mask.detach().cpu(),
                'mask_thr_used': float(thr_value.detach().cpu().item()),
                'mask_area': int(mask_area_value),
                'area_min': int(area_min_value) if area_min_value is not None else None,
                'area_max': int(area_max_value) if area_max_value is not None else None,
                'is_valid_area': bool(valid_flag),
                'mask_refine_mode': str(mode),
                'mask_quantile_used': float(quantile_value) if quantile_value is not None else None,
            })

        for idx in range(num_props):
            roi = proposals[idx]
            roi_mask = mask_2d[idx]
            if roi_mask.dim() == 3:
                roi_mask = roi_mask.mean(dim=0)

            if mode == 'quantile':
                m_h = int(roi_mask.size(0))
                m_w = int(roi_mask.size(1))
                roi_area = max(1, m_h * m_w)

                q = float(mask_thr_quantile)
                q = min(1.0, max(0.0, q))
                q_delta = max(0.0, float(mask_quantile_retry_delta))
                thr_min = float(min(mask_thr_min, mask_thr_max))
                thr_max = float(max(mask_thr_min, mask_thr_max))

                min_area_dyn = max(int(mask_min_area), int(float(mask_area_min_ratio) * roi_area))
                max_area_dyn = int(float(mask_area_max_ratio) * roi_area)
                max_area_dyn = max(min_area_dyn, max_area_dyn)

                flat_mask = roi_mask.reshape(-1)

                def _threshold_by_quantile(quantile_value):
                    thr_dyn = torch.quantile(flat_mask, quantile_value)
                    thr_dyn = torch.clamp(thr_dyn, min=thr_min, max=thr_max)
                    return roi_mask > thr_dyn

                bin_mask = _threshold_by_quantile(q)
                thr_used = torch.quantile(flat_mask, q)
                thr_used = torch.clamp(thr_used, min=thr_min, max=thr_max)
                mask_area = int(bin_mask.sum().item())

                retry_q = None
                if mask_area < min_area_dyn:
                    retry_q = max(0.0, q - q_delta)
                elif mask_area > max_area_dyn:
                    retry_q = min(1.0, q + q_delta)

                if retry_q is not None and retry_q != q:
                    bin_mask = _threshold_by_quantile(retry_q)
                    thr_used = torch.quantile(flat_mask, retry_q)
                    thr_used = torch.clamp(thr_used, min=thr_min, max=thr_max)
                    mask_area = int(bin_mask.sum().item())

                is_valid_area = (mask_area >= min_area_dyn) and (mask_area <= max_area_dyn)
                area_min = min_area_dyn
                area_max = max_area_dyn
                quantile_used = retry_q if retry_q is not None else q
            else:
                bin_mask = roi_mask > mask_thr
                thr_used = torch.tensor(float(mask_thr), device=roi_mask.device, dtype=roi_mask.dtype)
                is_valid_area = int(bin_mask.sum().item()) >= int(mask_min_area)
                mask_area = int(bin_mask.sum().item())
                area_min = int(mask_min_area)
                area_max = None
                quantile_used = None

            if not is_valid_area:
                if fallback_to_proposal:
                    refined[idx] = roi
                else:
                    valid[idx] = False
                _append_debug(
                    idx, roi, refined[idx], roi_mask, bin_mask, thr_used,
                    mask_area, area_min, area_max, is_valid_area, quantile_used)
                continue

            ys, xs = torch.where(bin_mask)
            if ys.numel() == 0 or xs.numel() == 0:
                if fallback_to_proposal:
                    refined[idx] = roi
                else:
                    valid[idx] = False
                _append_debug(
                    idx, roi, refined[idx], roi_mask, bin_mask, thr_used,
                    mask_area, area_min, area_max, is_valid_area, quantile_used)
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
                _append_debug(
                    idx, roi, refined[idx], roi_mask, bin_mask, thr_used,
                    mask_area, area_min, area_max, is_valid_area, quantile_used)
                continue

            refined[idx] = torch.stack([x1, y1, x2, y2])
            _append_debug(
                idx, roi, refined[idx], roi_mask, bin_mask, thr_used,
                mask_area, area_min, area_max, is_valid_area, quantile_used)

        return refined, valid, debug_items if collect_debug else None

    def _generate_jittered_proposals(self, points, img_shape):
        '''
        Core generator: solves the "end point" problem.
        For each point, generates multi-scale, multi-offset boxes.
        '''
        return generate_jittered_proposals(
            points,
            img_shape,
            self.infer_base_scales,
            self.infer_ratios,
            self.infer_anchor_offsets,
            min_side=5.0,
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
        # 将 rois 也传递给 bbox_head；返回 (bag_alpha, ins_score, bag_logits)
        out = self.bbox_head(bbox_feats, rois)
        if len(out) == 3:
            return out[0], out[1], out[2]
        # 兼容非 EDLHead 的两返回值 bbox head
        bag_score, ins_score = out
        return bag_score, ins_score, None

    def _merge_bag_forward_outputs(self, outs):
        """Concatenate per-bag ``_bbox_forward`` outputs in bag order."""
        if not outs:
            dev = next(self.bbox_head.parameters()).device
            nc = int(self.bbox_head.num_classes)
            z = torch.zeros((0, nc), device=dev, dtype=torch.float32)
            return z, z, z
        bag_score = torch.cat([o[0] for o in outs], dim=0)
        ins_parts = [o[1] for o in outs]
        ins0 = ins_parts[0]
        if isinstance(ins0, tuple):
            if len(ins0) >= 4:
                init_all = torch.cat([p[0] for p in ins_parts], dim=0)
                enh_all = torch.cat([p[1] for p in ins_parts], dim=0)
                batch_init = []
                batch_enh = []
                for p in ins_parts:
                    batch_init.extend(p[2])
                    batch_enh.extend(p[3])
                ins_score = (init_all, enh_all, batch_init, batch_enh)
            elif len(ins0) == 2:
                ins_score = (
                    torch.cat([p[0] for p in ins_parts], dim=0),
                    torch.cat([p[1] for p in ins_parts], dim=0),
                )
            else:
                raise TypeError(
                    f'Unsupported ins_score tuple length {len(ins0)} in bag merge')
        else:
            ins_score = torch.cat(ins_parts, dim=0)
        logits_parts = [o[2] for o in outs if len(o) > 2]
        if logits_parts and all(x is not None for x in logits_parts):
            bag_logits = torch.cat(logits_parts, dim=0)
        else:
            bag_logits = None
        return bag_score, ins_score, bag_logits

    def _expand_fpn_features_for_bags(self, x, num_bags_per_image):
        """Repeat each image's FPN maps so batch dim equals total bags (Path A)."""
        out_levels = []
        for feat in x:
            chunks = []
            for i, ni in enumerate(num_bags_per_image):
                if ni < 1:
                    continue
                chunks.append(feat[i:i + 1].repeat(ni, 1, 1, 1))
            if chunks:
                out_levels.append(torch.cat(chunks, dim=0))
            else:
                out_levels.append(feat)
        return out_levels

    def _select_eval_score(self, score):
        if isinstance(score, (tuple, list)):
            if len(score) > 1 and isinstance(score[1], torch.Tensor):
                return score[1]
            for item in reversed(score):
                if isinstance(item, torch.Tensor):
                    return item
            return None
        return score

    def _calculate_mil_accuracy(self, bag_score, bag_label, ins_score, ins_label):
        """Calculate MIL logging metrics for binary and multi-class labels."""
        metrics = {}

        def calculate_metrics(scores, target, prefix, acc_key):
            if scores is None or target is None or target.numel() == 0:
                return {}
            if scores.dim() == 1:
                scores = scores.unsqueeze(0)
            target = target.reshape(-1).long()
            if scores.size(0) != target.numel():
                n = min(scores.size(0), target.numel())
                scores = scores[:n]
                target = target[:n]
            if target.numel() == 0:
                return {}

            pred = torch.argmax(scores, dim=1)
            exact_acc = (pred == target).float().mean() * 100
            result = {acc_key: exact_acc}

            pred_binary = (pred > 0)
            target_binary = (target > 0)
            total = max(target.numel(), 1)
            tp = (pred_binary & target_binary).float().sum()
            tn = ((~pred_binary) & (~target_binary)).float().sum()
            fp = (pred_binary & (~target_binary)).float().sum()
            fn = ((~pred_binary) & target_binary).float().sum()
            eps = torch.finfo(scores.dtype).eps if scores.is_floating_point() else 1e-6
            precision = tp / torch.clamp(tp + fp, min=eps)
            recall = tp / torch.clamp(tp + fn, min=eps)
            f1 = (2 * precision * recall) / torch.clamp(precision + recall, min=eps)

            # Keep the original tp/tn/fp/fn percentage keys for binary runs.
            result.update({
                f'{prefix}_tp_pct': (tp / total) * 100,
                f'{prefix}_tn_pct': (tn / total) * 100,
                f'{prefix}_fp_pct': (fp / total) * 100,
                f'{prefix}_fn_pct': (fn / total) * 100,
                f'{prefix}_binary_acc': (pred_binary == target_binary).float().mean() * 100,
                f'{prefix}_precision': precision * 100,
                f'{prefix}_recall': recall * 100,
                f'{prefix}_f1': f1 * 100,
            })

            if target_binary.any():
                result[f'{prefix}_fg_acc'] = (
                    pred[target_binary] == target[target_binary]).float().mean() * 100

            if scores.size(1) > 2 and scores.size(1) >= 5:
                top5 = scores.topk(5, dim=1).indices
                result[f'{prefix}_top5_acc'] = (
                    top5 == target[:, None]).any(dim=1).float().mean() * 100

            return result

        with torch.no_grad():
            val_bag = self._select_eval_score(bag_score)
            if val_bag is not None and bag_label is not None:
                metrics.update(calculate_metrics(val_bag, bag_label, 'bag', 'acc_bag'))

            val_ins = self._select_eval_score(ins_score)
            if val_ins is not None and ins_label is not None:
                metrics.update(calculate_metrics(val_ins, ins_label, 'ins', 'acc_instance'))

        return metrics


    def forward_train(self, x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs):
        
        # # 从 kwargs 中获取 epoch_num，如果不存在则默认为 self.current_epoch
        epoch_num = kwargs.get('epoch_num', self.current_epoch)
        if not getattr(self.bbox_head, 'save_debug_info', False):
            if hasattr(self, '_mil_evidence_debug'):
                del self._mil_evidence_debug
        if not getattr(self.bbox_head, 'save_mask_debug', False):
            if hasattr(self, '_mil_mask_debug'):
                del self._mil_mask_debug

        # 从 kwargs 中获取 img
        img = kwargs.get('img')
        gt_points = kwargs.get('gt_points')

        batch_bag_bboxes = []
        batch_instance_labels = []
        batch_pseudo_point_ids = []
        batch_bag_labels = []
        bag_class_rows = []
        bag_to_img = []
        num_bags_per_image = []
        num_cls = int(self.bbox_head.num_classes)
        device0 = x[0].device

        for i, img_meta in enumerate(img_metas):
            bags = self.proposal_generator.forward_bags(
                img_meta,
                gt_points=gt_points[i] if gt_points else None,
                gt_bboxes=gt_bboxes[i] if gt_bboxes else None,
                gt_labels=gt_labels[i] if gt_labels else None,
                device=device0,
            )
            num_bags_per_image.append(len(bags))
            for b in bags:
                pseudo_bboxes = b['pseudo_bboxes']
                batch_bag_bboxes.append(pseudo_bboxes)
                batch_instance_labels.append(b['pseudo_labels'])
                pids = b.get('pseudo_point_ids', None)
                if pids is None or pids.numel() != pseudo_bboxes.size(0):
                    pids = torch.full(
                        (pseudo_bboxes.size(0), ),
                        -1,
                        dtype=torch.long,
                        device=pseudo_bboxes.device)
                batch_pseudo_point_ids.append(pids)
                batch_bag_labels.append(b['bag_label'].to(device0))
                bag_to_img.append(i)

                fg = int(b['fg_class_idx'])
                bl = int(b['bag_label'].view(-1)[0].item())
                row = torch.zeros(num_cls, device=device0, dtype=torch.float32)
                if fg == -2:
                    if bl == 0:
                        row[0] = 1.0
                    else:
                        gl = gt_labels[i] if i < len(gt_labels) else None
                        if gl is None:
                            row[0] = 1.0
                        elif isinstance(gl, torch.Tensor):
                            if gl.numel() == 0:
                                row[0] = 1.0
                            else:
                                for lab in torch.unique(gl.long()).tolist():
                                    idx = int(lab) + 1
                                    if 0 <= idx < num_cls:
                                        row[idx] = 1.0
                                if float(row.sum()) == 0.0:
                                    row[0] = 1.0
                        else:
                            row[0] = 1.0
                elif bl == 0 or fg < 0:
                    row[0] = 1.0
                else:
                    idx = fg + 1
                    if 0 <= idx < num_cls:
                        row[idx] = 1.0
                    else:
                        row[0] = 1.0
                bag_class_rows.append(row)

        bag_class_target = torch.stack(bag_class_rows, dim=0)

        batch_bag_bboxes = [bbox.to(device0) for bbox in batch_bag_bboxes]
        batch_instance_labels = [label.to(device0) for label in batch_instance_labels]
        batch_pseudo_point_ids = [
            pid.to(device0) for pid in batch_pseudo_point_ids
        ]
        batch_bag_labels = [label.to(device0) for label in batch_bag_labels]

        # [新增] 仅在此时暂存数据用于 Hook 可视化，用完即删，避免显存泄漏
        if getattr(self, 'debug_proposal_vis', False):
            self._last_proposal_debug = {
                'img_metas': img_metas,
                'batch_bag_bboxes': batch_bag_bboxes,
                'batch_instance_labels': batch_instance_labels,
                'gt_points': gt_points,
                'gt_bboxes': gt_bboxes,
                'gt_labels': gt_labels,
                'batch_bag_labels': batch_bag_labels,
                'bag_to_img': bag_to_img,
                'num_bags_per_image': num_bags_per_image,
            }

        # Path A: repeat FPN batch, single head forward. Path B: slice per bag.
        collect_mask_dbg = getattr(self.bbox_head, 'save_mask_debug', False)
        mask_dbg_chunks = [] if collect_mask_dbg else None

        if self.mil_repeat_fpn_for_bags:
            x_exp = self._expand_fpn_features_for_bags(x, num_bags_per_image)
            rois = bbox2roi(batch_bag_bboxes)
            bag_score, ins_score, bag_logits = self._bbox_forward(x_exp, rois)
        else:
            fwd_outs = []
            for bag_i, (img_i, pb) in enumerate(zip(bag_to_img, batch_bag_bboxes)):
                xi = [feat[img_i:img_i + 1] for feat in x]
                rois_m = bbox2roi([pb])
                fwd_outs.append(self._bbox_forward(xi, rois_m))
                if collect_mask_dbg:
                    ld = getattr(self.bbox_head, '_last_mask_debug_data', None)
                    if ld is not None:
                        r = ld['rois'].clone()
                        if r.numel() > 0:
                            r[:, 0] = float(bag_i)
                        pid_b = batch_pseudo_point_ids[bag_i]
                        n = int(pb.size(0))
                        if pid_b.numel() != n:
                            pid_b = torch.full(
                                (n, ), -1, dtype=torch.long, device=pb.device)
                        mask_dbg_chunks.append(
                            (r.detach().cpu(), ld['mask_2d'].detach().cpu(),
                             ld['ins_output'].detach().cpu(),
                             pid_b.detach().cpu()))
            bag_score, ins_score, bag_logits = self._merge_bag_forward_outputs(fwd_outs)

        # 4. 将标签列表拼接成一个Tensor以计算loss
        bag_labels = torch.cat(batch_bag_labels)
        ins_labels = torch.cat(batch_instance_labels)

        if collect_mask_dbg:
            if self.mil_repeat_fpn_for_bags:
                ld = getattr(self.bbox_head, '_last_mask_debug_data', None)
                if ld is not None and ld['mask_2d'].numel() > 0:
                    rois_cat_m = ld['rois'].detach().cpu()
                    mask_cat_m = ld['mask_2d'].detach().cpu()
                    ins_cat_m = ld['ins_output'].detach().cpu()
                    pid_rows = []
                    for bag_i, pid in enumerate(batch_pseudo_point_ids):
                        n = int(batch_bag_bboxes[bag_i].size(0))
                        if pid.numel() != n:
                            pid = torch.full(
                                (n, ), -1, dtype=torch.long, device=device0)
                        pid_rows.append(pid.detach().cpu())
                    pid_cat_m = torch.cat(
                        pid_rows, dim=0) if pid_rows else torch.empty(
                            (0, ), dtype=torch.long)
                else:
                    rois_cat_m = torch.empty((0, 5), dtype=torch.float32)
                    mask_cat_m = torch.empty((0, ), dtype=torch.float32)
                    ins_cat_m = torch.empty((0, num_cls), dtype=torch.float32)
                    pid_cat_m = torch.empty((0, ), dtype=torch.long)
            else:
                if mask_dbg_chunks:
                    rois_cat_m = torch.cat([c[0] for c in mask_dbg_chunks], dim=0)
                    mask_cat_m = torch.cat([c[1] for c in mask_dbg_chunks], dim=0)
                    ins_cat_m = torch.cat([c[2] for c in mask_dbg_chunks], dim=0)
                    pid_cat_m = torch.cat([c[3] for c in mask_dbg_chunks], dim=0)
                else:
                    rois_cat_m = torch.empty((0, 5), dtype=torch.float32)
                    mask_cat_m = torch.empty((0, ), dtype=torch.float32)
                    ins_cat_m = torch.empty((0, num_cls), dtype=torch.float32)
                    pid_cat_m = torch.empty((0, ), dtype=torch.long)
            self._mil_mask_debug = {
                'rois': rois_cat_m,
                'mask_2d': mask_cat_m,
                'ins_output': ins_cat_m,
                'ins_labels': ins_labels.detach().cpu(),
                'pseudo_point_ids': pid_cat_m,
                'bag_to_img': list(bag_to_img),
                'num_classes': num_cls,
            }
            self.bbox_head.save_mask_debug = False
            if hasattr(self.bbox_head, '_last_mask_debug_data'):
                del self.bbox_head._last_mask_debug_data

        if getattr(self.bbox_head, 'save_debug_info', False):
            if isinstance(ins_score, tuple) and len(ins_score) > 1 and isinstance(
                    ins_score[1], torch.Tensor):
                cur_ins_scores = ins_score[1]
            else:
                cur_ins_scores = ins_score
            rois_list = []
            pid_list = []
            for bag_i, (pb, pid) in enumerate(
                    zip(batch_bag_bboxes, batch_pseudo_point_ids)):
                n = int(pb.size(0))
                if n == 0:
                    continue
                img_inds = pb.new_full((n, 1), bag_i, dtype=pb.dtype)
                rois_list.append(torch.cat([img_inds, pb], dim=-1))
                if pid.numel() != n:
                    pid = torch.full(
                        (n, ), -1, dtype=torch.long, device=pb.device)
                pid_list.append(pid)
            if rois_list:
                rois_cat = torch.cat(rois_list, dim=0)
                pid_cat = torch.cat(pid_list, dim=0)
            else:
                zdev = device0
                rois_cat = torch.empty((0, 5), device=zdev, dtype=torch.float32)
                pid_cat = torch.empty((0, ), dtype=torch.long, device=zdev)
            self._mil_evidence_debug = {
                'rois': rois_cat.detach().cpu(),
                'ins_output': cur_ins_scores.detach().cpu(),
                'ins_labels': ins_labels.detach().cpu(),
                'bag_labels': bag_labels.detach().cpu(),
                'bag_to_img': list(bag_to_img),
                'pseudo_point_ids': pid_cat.detach().cpu(),
                'num_classes': num_cls,
            }
            self.bbox_head.save_debug_info = False
            if hasattr(self.bbox_head, '_last_debug_data'):
                del self.bbox_head._last_debug_data

        if getattr(self, 'debug_multiclass_analysis', False):
            cur_ins_scores = self._select_eval_score(ins_score)
            rois_dbg_list = []
            for img_i, pb in zip(bag_to_img, batch_bag_bboxes):
                r = bbox2roi([pb])
                r = r.clone()
                r[:, 0] = int(img_i)
                rois_dbg_list.append(r)
            rois_dbg = torch.cat(rois_dbg_list, dim=0) if rois_dbg_list else torch.empty(
                (0, 5))
            self._last_multiclass_debug = {
                'img_metas': img_metas,
                'rois': rois_dbg.detach().cpu(),
                'ins_scores': None if cur_ins_scores is None else cur_ins_scores.detach().cpu(),
                'ins_labels': ins_labels.detach().cpu(),
                'bag_labels': bag_labels.detach().cpu(),
                'batch_instance_labels': [label.detach().cpu() for label in batch_instance_labels],
                'batch_bag_bboxes': [bbox.detach().cpu() for bbox in batch_bag_bboxes],
            }

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
            cls_score=(bag_score, ins_score, bag_logits),
            bag_label=bag_labels,
            ins_labels=ins_labels,
            epoch_num=epoch_num,
            bag_class_target=bag_class_target,
        )

        # [新增] 计算准确率指标并加入损失字典
        # 这样这些指标就会作为 log_vars 自动显示在训练日志中
        acc_metrics = self._calculate_mil_accuracy(bag_score, bag_labels, ins_score, ins_labels)
        losses.update(acc_metrics)

        return losses
