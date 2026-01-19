import torch
import numpy as np
from mmengine.model import BaseModule
from mmdet.registry import TASK_UTILS

@TASK_UTILS.register_module()
class PointPseudoBoxGenerator(BaseModule):
    """根据点生成伪正样本框和背景负样本框的生成器。
    
    Args:
        box_sizes (list[list[int]]): 预定义的候选框尺寸列表 [[w, h], ...]。
        box_offset (int): 生成正样本时的随机中心偏移范围。
        num_neg_samples (int): 每张图生成的负样本候选数量。
    """
    def __init__(self,
                 box_sizes,
                 box_offset,
                 num_neg_samples,
                 pos_bag_prob=0.5,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.box_sizes = np.array(box_sizes)
        self.box_offset = box_offset
        self.num_neg_samples = num_neg_samples
        self.pos_bag_prob = pos_bag_prob

    def forward(self, img_meta, gt_points=None, gt_bboxes=None, num_pos_samples=8):
        """生成单张图像的候选框。"""
        # 1. 统一获取点坐标
        points = self._get_points_from_gt(gt_bboxes, gt_points)
        
        # 2. 生成正样本伪框
        pos_bboxes = self._get_syn_bboxs(img_meta, points, num_pos_samples)
        
        # --- 修改开始：动态计算所需的负样本数量以保持平衡 ---
        num_pos_generated = pos_bboxes.size(0)
        if num_pos_generated > 0:
            # 如果有正样本，通过生成等量的负样本来实现 1:1 平衡
            current_num_neg_req = num_pos_generated
        else:
            # 如果没有正样本（空图），则使用默认配置的数量
            current_num_neg_req = self.num_neg_samples
        # --- 修改结束 ---

        # 3. 生成互斥的负样本框 (传入动态计算的数量)
        neg_bboxes = self._generate_negative_samples(img_meta, pos_bboxes, num_neg_required=current_num_neg_req)

        # 4. 整合正负样本框及其标签
        # 注意：此处 _merge_pos_neg_bboxes 可能会随机丢弃正样本变成负包
        # 但在正包的情况下，正负样本数量已经是平衡的了
        pseudo_bboxes, pseudo_labels, bag_label = self._merge_pos_neg_bboxes(pos_bboxes, neg_bboxes)
        
        return pos_bboxes, neg_bboxes, pseudo_bboxes, pseudo_labels, bag_label

    def _merge_pos_neg_bboxes(self, pos_bboxes, neg_bboxes):
        # 1. 创建标签张量 (1: 正样本, 0: 负样本)
        # 确保使用与 bbox 相同的 device
        device = pos_bboxes.device if pos_bboxes.numel() > 0 else neg_bboxes.device
        
        # 新增：根据概率随机丢弃正样本，从而构建负包
        # 即使输入包含正样本，也有 (1 - pos_bag_prob) 的概率被强制视为负包
        if pos_bboxes.numel() > 0:
            if torch.rand(1, device=device) > self.pos_bag_prob:
                # 丢弃正样本，强制转为负包
                pos_bboxes = torch.empty((0, 4), device=device, dtype=pos_bboxes.dtype)

        pos_labels = torch.ones(pos_bboxes.size(0), dtype=torch.long, device=device)
        neg_labels = torch.zeros(neg_bboxes.size(0), dtype=torch.long, device=device)

        # 2. 基础拼接逻辑
        if pos_bboxes.numel() == 0:
            pseudo_bboxes = neg_bboxes
            pseudo_labels = neg_labels
            bag_label = 0
        elif neg_bboxes.numel() == 0:
            pseudo_bboxes = pos_bboxes
            pseudo_labels = pos_labels
            bag_label = 1
        else:
            pseudo_bboxes = torch.cat([pos_bboxes, neg_bboxes], dim=0)
            pseudo_labels = torch.cat([pos_labels, neg_labels], dim=0)
            bag_label = 1

        # 3. 执行打乱 (Shuffle)
        # 使用同一个 perm 索引同时打乱 bbox 和 label，保证对应关系
        if pseudo_bboxes.size(0) > 0:
            perm = torch.randperm(pseudo_bboxes.size(0), device=pseudo_bboxes.device)
            pseudo_bboxes = pseudo_bboxes[perm]
            pseudo_labels = pseudo_labels[perm]

        return pseudo_bboxes, pseudo_labels, bag_label

    def _get_points_from_gt(self, gt_bboxes=None, gt_points=None):
        points = []
        if gt_points is not None and len(gt_points) > 0:
            points = gt_points
        elif gt_bboxes is not None and len(gt_bboxes) > 0:
            gt_bboxes_tensor = gt_bboxes
            cx = (gt_bboxes_tensor[:, 0] + gt_bboxes_tensor[:, 2]) / 2
            cy = (gt_bboxes_tensor[:, 1] + gt_bboxes_tensor[:, 3]) / 2
            points = torch.stack((cx, cy), dim=1)
        return points

    def _get_syn_bboxs(self, img_meta, points, num_samples):
        if isinstance(points, torch.Tensor):
            if points.dim() > 2:
                points = points.squeeze(0)
        
        # 确保在 GPU 上处理，避免 CPU/GPU 切换带来的性能损耗
        if points is None or len(points) == 0:
            device = points.device if points is not None else 'cpu'
            return torch.empty((0, 4), dtype=torch.float32, device=device)

        h, w = img_meta['img_shape'][:2]
        device = points.device
        num_points = points.size(0)
        
        # 准备尺寸 Tensor
        box_sizes_tensor = torch.tensor(self.box_sizes, device=device, dtype=torch.float32)
        
        # 随机选择尺寸索引
        rand_indices = torch.randint(0, len(self.box_sizes), (num_points, num_samples), device=device)
        chosen_sizes = box_sizes_tensor[rand_indices] # (N, num_samples, 2)

        # 随机偏移 (-offset, offset)
        offsets = torch.rand(num_points, num_samples, 2, device=device) * (2 * self.box_offset) - self.box_offset
        
        # 扩展中心点并应用偏移
        centers = points.unsqueeze(1) + offsets # (N, num_samples, 2)
        
        # 计算坐标 (x1, y1, x2, y2)
        # 保持原逻辑：以中心点向四周扩散 size 大小 (宽高为 2*size)
        # 新增：使用 clamp 限制在图像边界内
        x1 = torch.clamp(centers[..., 0] - chosen_sizes[..., 0], min=0)
        y1 = torch.clamp(centers[..., 1] - chosen_sizes[..., 1], min=0)
        x2 = torch.clamp(centers[..., 0] + chosen_sizes[..., 0], max=w)
        y2 = torch.clamp(centers[..., 1] + chosen_sizes[..., 1], max=h)
        
        # 重塑为 (N_total, 4)
        syn_boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(-1, 4)
        
        return syn_boxes

    def _generate_negative_samples(self, img_meta, syn_bboxes, num_neg_required=None):
        # 局部引入以避免循环依赖
        from mmdet.structures.bbox import bbox_overlaps
        
        # 如果未传入具体数量，使用初始化时的默认值
        target_num_samples = num_neg_required if num_neg_required is not None else self.num_neg_samples

        h, w = img_meta['img_shape'][:2]
        # 确保使用与正样本相同的 device
        device = syn_bboxes.device if syn_bboxes.numel() > 0 else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        box_sizes_tensor = torch.tensor(self.box_sizes, device=device, dtype=torch.float32)
        
        # 增加候选池大小以提高不重叠采样的成功率 (基于目标数量动态调整)
        pool_size = max(target_num_samples * 10, 1000)
        
        # 1. 批量生成随机候选框
        rand_indices = torch.randint(0, len(self.box_sizes), (pool_size,), device=device)
        sizes = box_sizes_tensor[rand_indices] 

        # 随机生成左上角点
        x1 = torch.rand(pool_size, device=device) * (w - sizes[:, 0])
        y1 = torch.rand(pool_size, device=device) * (h - sizes[:, 1])
        x2 = torch.clamp(x1 + sizes[:, 0], max=w)
        y2 = torch.clamp(y1 + sizes[:, 1], max=h)
        
        candidates = torch.stack([x1, y1, x2, y2], dim=1)

        # 2. 过滤掉与正样本重叠的框
        if syn_bboxes.numel() > 0:
            ious = bbox_overlaps(candidates, syn_bboxes)
            max_ious, _ = ious.max(dim=1)
            valid_mask = max_ious == 0 
            neg_bboxes = candidates[valid_mask]
        else:
            neg_bboxes = candidates

        # 3. 采样最终结果 (使用 target_num_samples)
        num_valid = neg_bboxes.size(0)
        if num_valid >= target_num_samples:
            perm = torch.randperm(num_valid, device=device)[:target_num_samples]
            neg_bboxes = neg_bboxes[perm]
        else:
            # Fallback: 补足数量
            if num_valid > 0:
                needed = target_num_samples - num_valid
                fill_indices = torch.randint(0, num_valid, (needed,), device=device)
                neg_bboxes = torch.cat([neg_bboxes, neg_bboxes[fill_indices]], dim=0)
            else:
                neg_bboxes = candidates[:target_num_samples]

        return neg_bboxes