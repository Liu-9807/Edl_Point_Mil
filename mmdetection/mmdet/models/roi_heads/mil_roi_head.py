import os
import torch
import numpy as np

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import get_box_tensor, scale_boxes

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
                results_list.append(DetDataSample()) 
                continue

            # --- 步骤 1: 密集采样 (Jittered Proposals) ---
            # 针对每个点，生成 K 个候选框
            proposals, keep_indices = self._generate_jittered_proposals(
                prompts, img_meta['img_shape']
            )
            # proposals: [Total_N, 4], keep_indices: 用于记录哪个框属于哪个点
            
            if proposals.shape[0] == 0:
                results_list.append(DetDataSample())
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
            score_thr = kwargs.get('score_thr', 0.64)

            if probs.shape[1] > 1:
                # 取除了第0列以外的部分，计算正类的最大分和对应类别
                pos_probs = probs[:, 1:] 
                scores, tmp_labels = torch.max(pos_probs, dim=1)
                labels = tmp_labels + 1 # 还原回原始 label index (1-based, 假设 0 是背景)
            else:
                # 极端情况：如果不含背景类
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
                subset_bboxes = proposals[mask]  # [K, 4]
                
                # 筛选：保留分数超过阈值的框
                valid_mask = subset_scores > score_thr
                
                if valid_mask.any():
                    final_bboxes.append(subset_bboxes[valid_mask])
                    final_labels.append(subset_labels[valid_mask])
                    final_scores.append(subset_scores[valid_mask])
                # 若没有通过阈值的，该点不贡献任何框

            if len(final_bboxes) > 0:
                # 使用 cat 拼接不同数量的框 (注意原代码是 stack，这里要改为 cat)
                final_bboxes = torch.cat(final_bboxes, dim=0)
                final_labels = torch.cat(final_labels, dim=0)
                final_scores = torch.cat(final_scores, dim=0)
                
                # Rescale 回原图尺寸
                if rescale:
                    final_bboxes = scale_boxes(final_bboxes, img_meta['scale_factor'])
                
                # 封装结果
                res = DetDataSample()
                res.set_metainfo(img_meta)
                # results.bboxes, results.scores, results.labels
                from mmdet.structures.bbox import HorizontalBoxes
                from mmengine.structures import InstanceData
                
                inst = InstanceData()
                inst.bboxes = final_bboxes
                inst.labels = final_labels
                inst.scores = final_scores
                res.pred_instances = inst
                results_list.append(res)
            else:
                results_list.append(DetDataSample())

        return results_list

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
            
        return torch.tensor(proposals, device=points.device), torch.tensor(point_indices, device=points.device)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_sampler = None
        self.bbox_assigner = None

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        bbox_feats = self.bbox_roi_extractor(x, rois.cuda())
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
