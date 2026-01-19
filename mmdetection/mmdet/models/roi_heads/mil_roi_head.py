import os
import torch
import numpy as np

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
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
        """简化预测接口；如需真实推理，可在此补充 simple_test 流程。"""
        # 当前只用于训练任务，推理返回空结果占位
        # TODO 应当补充 simple_test 逻辑
        return batch_data_samples

    def _forward(self, x, batch_data_samples=None, **kwargs):
        """tensor 模式前向；默认走 predict 占位实现。"""
        return self.predict(x, batch_data_samples, **kwargs)

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

        return losses
