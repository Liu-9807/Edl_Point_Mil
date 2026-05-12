import torch
import numpy as np
from mmengine.model import BaseModule
from mmdet.registry import TASK_UTILS
from mmdet.models.utils.mil_jittered_proposals import generate_jittered_proposals

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
                 num_pos_samples=8,
                 pos_bag_prob=0.5,
                 sample_coordinate_mode='input',
                 class_box_sizes=None,
                 class_box_size_mode='absolute',
                 negative_size_source='global',
                 box_offset_mode='absolute',
                 box_offset_ratio=0.1,
                 size_jitter=0.15,
                 min_input_box_size=4.0,
                 train_use_jitter=False,
                 train_jitter_use_class_sizes=False,
                 train_infer_base_scales=None,
                 train_infer_ratios=None,
                 train_infer_anchor_offsets=None,
                 max_jitter_pos_per_bag=512,
                 neg_iou_chunk_rows=512,
                 neg_iou_max_ref_boxes=8192,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.box_sizes = np.array(box_sizes)
        self.box_offset = box_offset
        self.num_neg_samples = num_neg_samples
        self.num_pos_samples = num_pos_samples
        self.pos_bag_prob = pos_bag_prob
        self.sample_coordinate_mode = sample_coordinate_mode
        self.class_box_size_mode = class_box_size_mode
        self.negative_size_source = negative_size_source
        self.box_offset_mode = box_offset_mode
        self.box_offset_ratio = float(box_offset_ratio)
        self.size_jitter = float(size_jitter)
        self.min_input_box_size = float(min_input_box_size)
        self.class_box_sizes = (
            None if class_box_sizes is None
            else np.asarray(class_box_sizes, dtype=np.float32))

        if self.sample_coordinate_mode not in ('input', 'original'):
            raise ValueError(
                "sample_coordinate_mode must be 'input' or 'original', "
                f'but got {self.sample_coordinate_mode!r}.')
        if self.class_box_size_mode not in ('absolute', 'ratio'):
            raise ValueError(
                "class_box_size_mode must be 'absolute' or 'ratio', "
                f'but got {self.class_box_size_mode!r}.')
        if self.negative_size_source not in ('global', 'class_prior'):
            raise ValueError(
                "negative_size_source must be 'global' or 'class_prior', "
                f'but got {self.negative_size_source!r}.')
        if self.box_offset_mode not in ('absolute', 'size_ratio'):
            raise ValueError(
                "box_offset_mode must be 'absolute' or 'size_ratio', "
                f'but got {self.box_offset_mode!r}.')
        if self.class_box_sizes is not None and self.class_box_sizes.ndim != 3:
            raise ValueError(
                'class_box_sizes must have shape [num_classes, num_sizes, 2].')

        self.train_use_jitter = bool(train_use_jitter)
        self.train_jitter_use_class_sizes = bool(train_jitter_use_class_sizes)
        _def_offsets = [
            (0, 0), (-0.3, -0.3), (0.3, 0.3), (-0.3, 0.3), (0.3, -0.3)]
        self.train_infer_base_scales = (
            list(train_infer_base_scales)
            if train_infer_base_scales is not None else [32, 64, 128, 256])
        self.train_infer_ratios = (
            list(train_infer_ratios)
            if train_infer_ratios is not None else [0.5, 1.0, 2.0])
        self.train_infer_anchor_offsets = (
            [tuple(float(x) for x in o) for o in train_infer_anchor_offsets]
            if train_infer_anchor_offsets is not None else _def_offsets)
        # Cap jitter positives per class-bag before pos_bag_prob (0 = disabled).
        self.max_jitter_pos_per_bag = int(max_jitter_pos_per_bag)
        self.neg_iou_chunk_rows = int(neg_iou_chunk_rows)
        self.neg_iou_max_ref_boxes = int(neg_iou_max_ref_boxes)

    def forward(self,
                img_meta,
                gt_points=None,
                gt_bboxes=None,
                gt_labels=None,
                num_pos_samples=None,
                device=None):
        """生成单张图像的候选框。"""
        if num_pos_samples is None:
            num_pos_samples = self.num_pos_samples
        # 1. 统一获取点坐标
        points = self._get_points_from_gt(gt_bboxes, gt_points, device=device)
        
        # 2. 生成正样本伪框
        # 保留一份全量的正样本框，用于后续生成负样本时计算IoU排除区域
        # 即使当前包被判定为负包，这些区域也是包含目标的，不能作为背景
        if self.sample_coordinate_mode == 'original':
            sample_points = self._get_original_points(
                img_meta, points, device=device)
            all_pos_bboxes_orig = self._get_syn_bboxs_original(
                img_meta,
                sample_points,
                num_pos_samples,
                gt_labels=gt_labels,
                device=device)
            all_pos_bboxes = self._project_boxes_to_input(
                all_pos_bboxes_orig, img_meta)
        else:
            sample_points = points
            all_pos_bboxes_orig = None
            all_pos_bboxes = self._get_syn_bboxs(
                img_meta, points, num_pos_samples, device=device)
        pos_bboxes = all_pos_bboxes
        pos_labels = self._get_pos_labels(
            gt_labels, sample_points, num_pos_samples, all_pos_bboxes, device=device)
        
        # --- 修改开始：动态计算总实例数及正负样本配比 ---
        # 确定设备
        if isinstance(sample_points, torch.Tensor):
            device = sample_points.device
        elif pos_bboxes.numel() > 0:
            device = pos_bboxes.device
        else:
            device = torch.device('cpu')
        
        # 3. 确定是否为正包 (概率丢弃正样本，使其转为负包)
        # 将此逻辑从 _merge_pos_neg_bboxes 移至此处，以便后续计算负样本需求量
        if pos_bboxes.size(0) > 0:
            if torch.rand(1, device=device) > self.pos_bag_prob:
                # 丢弃正样本，强制转为负包
                pos_bboxes = torch.empty((0, 4), device=device, dtype=pos_bboxes.dtype)
                pos_labels = torch.empty((0,), device=device, dtype=torch.long)

        # 与当前 pos 框逐行对齐的全局点行号（与 gt_points / sample_points 第一维一致）
        num_pts = int(sample_points.size(0)) if isinstance(
            sample_points, torch.Tensor) else 0
        if pos_bboxes.numel() > 0 and num_pts > 0:
            pos_point_ids = torch.arange(
                num_pts, device=pos_bboxes.device,
                dtype=torch.long).repeat_interleave(int(num_pos_samples))
            if pos_point_ids.numel() > pos_bboxes.size(0):
                pos_point_ids = pos_point_ids[:pos_bboxes.size(0)]
        else:
            pos_point_ids = None

        # 4. 计算当前图像的目标总实例数 (固定基数 + 随机扰动)
        # 使用 self.num_neg_samples 作为基准包大小
        base_count = self.num_neg_samples
        # 定义随机扰动范围，例如 +/- 20%
        delta = int(base_count * 0.2)
        if delta > 0:
            random_diff = torch.randint(-delta, delta + 1, (1,), device=device).item()
        else:
            random_diff = 0
        target_bag_size = max(base_count + random_diff, 1) # 保证至少有1个实例

        # 5. 根据目标包大小调整正样本并计算所需负样本数
        num_pos = pos_bboxes.size(0)
        
        if num_pos >= target_bag_size:
            # 如果正样本数量超过目标包大小，随机采样截断，且不再需要负样本
            # 这种情况一般较少见，除非点非常多
            perm = torch.randperm(num_pos, device=device)[:target_bag_size]
            pos_bboxes = pos_bboxes[perm]
            pos_labels = pos_labels[perm]
            if pos_point_ids is not None and pos_point_ids.numel() == num_pos:
                pos_point_ids = pos_point_ids[perm]
            num_neg_req = 0
        else:
            # 正样本不足以填充包，剩余位置由负样本填补
            # 如果 pos_bboxes 为空 (负包)，则 num_neg_req == target_bag_size
            num_neg_req = target_bag_size - num_pos
        # --- 修改结束 ---

        # 6. 生成互斥的负样本框 (传入计算后的数量)
        # 使用 all_pos_bboxes 而非 pos_bboxes，确保即使是负包，生成的背景框也不覆盖目标
        if self.sample_coordinate_mode == 'original':
            neg_bboxes = self._generate_negative_samples_original(
                img_meta,
                all_pos_bboxes_orig,
                num_neg_required=num_neg_req,
                device=device,
            )
        else:
            neg_bboxes = self._generate_negative_samples(
                img_meta,
                all_pos_bboxes,
                num_neg_required=num_neg_req,
                device=device,
            )

        # 整合正负样本框及其标签
        pseudo_bboxes, pseudo_labels, bag_label, pseudo_point_ids = (
            self._merge_pos_neg_bboxes(
                pos_bboxes, neg_bboxes, pos_labels=pos_labels,
                pos_point_ids=pos_point_ids))

        return (pos_bboxes, neg_bboxes, pseudo_bboxes, pseudo_labels, bag_label,
                pseudo_point_ids)

    def forward_bags(self,
                     img_meta,
                     gt_points=None,
                     gt_bboxes=None,
                     gt_labels=None,
                     num_pos_samples=None,
                     device=None):
        """Per-image multi-bag generation: one MIL bag per GT class (single-class pos).

        Each dict in the returned list contains:
            - ``pseudo_bboxes`` / ``pseudo_labels`` / ``bag_label`` (same semantics as
              :meth:`forward`).
            - ``pseudo_point_ids`` (LongTensor): per-instance index into ``gt_points``
              row for positive proposals; ``-1`` for background proposals.
            - ``fg_class_idx`` (int): dataset 0-based class id for a positive bag;
              ``-1`` for a pure-negative bag; ``-2`` means legacy multi-class target
              (caller should build ``bag_class_target`` from full ``gt_labels``).
        """
        if num_pos_samples is None:
            num_pos_samples = self.num_pos_samples

        points = self._get_points_from_gt(gt_bboxes, gt_points, device=device)
        if points.numel() == 0:
            return self._forward_bags_single_from_forward(
                img_meta, gt_points, gt_bboxes, gt_labels,
                num_pos_samples=num_pos_samples, device=device, fg_class_idx=-1)

        lab = self._labels_for_points(gt_labels, int(points.size(0)), points.device)
        if lab is None:
            return self._forward_bags_single_from_forward(
                img_meta, gt_points, gt_bboxes, gt_labels,
                num_pos_samples=num_pos_samples, device=device, fg_class_idx=-2)

        if self.sample_coordinate_mode == 'original':
            sample_points = self._get_original_points(
                img_meta, points, device=device)
            full_excl_orig, full_pid, full_pos_input = (
                self._full_positive_jitter_or_legacy(
                    img_meta, sample_points, num_pos_samples, lab,
                    device=device))
        else:
            sample_points = points
            full_excl_orig = None
            full_pos_input, full_pid = self._full_positive_jitter_or_legacy_input(
                img_meta, sample_points, num_pos_samples, lab, device=device)

        device = full_pos_input.device
        uniq = torch.unique(lab.long()).tolist()
        bags = []
        for cls in uniq:
            cls = int(cls)
            cls_mask = lab == cls
            pt_idx = torch.where(cls_mask)[0]
            pos_point_ids = None
            if self.train_use_jitter:
                if full_pid is None:
                    raise RuntimeError('full_pid is required for jitter bags.')
                sel = torch.isin(full_pid, pt_idx)
                pos_bboxes = full_pos_input[sel]
                if pos_bboxes.numel() > 0:
                    pos_labels = torch.full(
                        (pos_bboxes.size(0),),
                        cls + 1,
                        dtype=torch.long,
                        device=device)
                    pos_point_ids = full_pid[sel].long()
                else:
                    pos_labels = torch.empty((0,), dtype=torch.long, device=device)
            else:
                pts_c = sample_points[cls_mask]
                labs_c = lab[cls_mask]
                if self.sample_coordinate_mode == 'original':
                    pos_orig = self._get_syn_bboxs_original(
                        img_meta,
                        pts_c,
                        num_pos_samples,
                        gt_labels=labs_c,
                        device=device)
                    pos_bboxes = self._project_boxes_to_input(pos_orig, img_meta)
                else:
                    pos_bboxes = self._get_syn_bboxs(
                        img_meta, pts_c, num_pos_samples, device=device)
                if pos_bboxes.numel() > 0:
                    pos_labels = torch.full(
                        (pos_bboxes.size(0),),
                        cls + 1,
                        dtype=torch.long,
                        device=device)
                    pos_point_ids = pt_idx.repeat_interleave(
                        int(num_pos_samples)).long()
                else:
                    pos_labels = torch.empty((0,), dtype=torch.long, device=device)

            cap = self.max_jitter_pos_per_bag
            if cap > 0 and pos_bboxes.size(0) > cap:
                perm = torch.randperm(pos_bboxes.size(0), device=device)[:cap]
                pos_bboxes = pos_bboxes[perm]
                pos_labels = pos_labels[perm]
                if pos_point_ids is not None:
                    pos_point_ids = pos_point_ids[perm]

            if pos_bboxes.size(0) > 0:
                if torch.rand(1, device=device) > self.pos_bag_prob:
                    pos_bboxes = torch.empty(
                        (0, 4), device=device, dtype=pos_bboxes.dtype)
                    pos_labels = torch.empty((0,), device=device, dtype=torch.long)
                    pos_point_ids = None

            target_bag_size = self._sample_target_bag_size(device)
            num_pos = pos_bboxes.size(0)
            if num_pos >= target_bag_size:
                perm = torch.randperm(num_pos, device=device)[:target_bag_size]
                pos_bboxes = pos_bboxes[perm]
                pos_labels = pos_labels[perm]
                if pos_point_ids is not None and pos_point_ids.numel() == num_pos:
                    pos_point_ids = pos_point_ids[perm]
                num_neg_req = 0
            else:
                num_neg_req = target_bag_size - num_pos

            if self.sample_coordinate_mode == 'original':
                neg_bboxes = self._generate_negative_samples_original(
                    img_meta,
                    full_excl_orig,
                    num_neg_required=num_neg_req,
                    device=device,
                )
            else:
                neg_bboxes = self._generate_negative_samples(
                    img_meta,
                    full_pos_input,
                    num_neg_required=num_neg_req,
                    device=device,
                )

            pseudo_bboxes, pseudo_labels, bag_label, pseudo_point_ids = (
                self._merge_pos_neg_bboxes(
                    pos_bboxes, neg_bboxes, pos_labels=pos_labels,
                    pos_point_ids=pos_point_ids))
            bl = int(bag_label) if not isinstance(bag_label, torch.Tensor) else int(
                bag_label.view(-1)[0].item())
            fg = cls if bl == 1 else -1
            bags.append({
                'pseudo_bboxes': pseudo_bboxes,
                'pseudo_labels': pseudo_labels,
                'pseudo_point_ids': pseudo_point_ids,
                'bag_label':
                torch.as_tensor([bl], dtype=torch.long, device=pseudo_bboxes.device),
                'fg_class_idx': fg,
            })
        return bags

    def _forward_bags_single_from_forward(self,
                                          img_meta,
                                          gt_points,
                                          gt_bboxes,
                                          gt_labels,
                                          num_pos_samples=None,
                                          device=None,
                                          fg_class_idx=-2):
        tup = self.forward(
            img_meta,
            gt_points=gt_points,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            num_pos_samples=num_pos_samples,
            device=device)
        _, _, pseudo_bboxes, pseudo_labels, bag_label, pseudo_point_ids = tup
        bl = int(bag_label) if not isinstance(bag_label, torch.Tensor) else int(
            bag_label.view(-1)[0].item())
        return [{
            'pseudo_bboxes': pseudo_bboxes,
            'pseudo_labels': pseudo_labels,
            'pseudo_point_ids': pseudo_point_ids,
            'bag_label':
            torch.as_tensor([bl], dtype=torch.long, device=pseudo_bboxes.device),
            'fg_class_idx': fg_class_idx,
        }]

    def _sample_target_bag_size(self, device):
        base_count = self.num_neg_samples
        delta = int(base_count * 0.2)
        if delta > 0:
            random_diff = torch.randint(
                -delta, delta + 1, (1,), device=device).item()
        else:
            random_diff = 0
        return max(base_count + random_diff, 1)

    def _max_iou_each_candidate_vs_ref(self, candidates, ref_boxes):
        """Per-row max IoU vs ref; chunked to cap peak memory (pool x N_ref)."""
        from mmdet.structures.bbox import bbox_overlaps

        device = candidates.device
        dtype = candidates.dtype
        if ref_boxes is None or ref_boxes.numel() == 0:
            return torch.zeros(candidates.size(0), device=device, dtype=dtype)
        ref = ref_boxes
        cap = self.neg_iou_max_ref_boxes
        if cap > 0 and ref.size(0) > cap:
            perm = torch.randperm(ref.size(0), device=ref.device)[:cap]
            ref = ref[perm]
        chunk = max(1, self.neg_iou_chunk_rows)
        n_c = candidates.size(0)
        parts = []
        for start in range(0, n_c, chunk):
            end = min(start + chunk, n_c)
            sub = candidates[start:end]
            ious = bbox_overlaps(sub, ref)
            parts.append(ious.max(dim=1).values)
        return torch.cat(parts, dim=0)

    def _full_positive_jitter_or_legacy(self, img_meta, sample_points,
                                        num_pos_samples, lab, device):
        """Returns (full_excl_orig, full_pid, full_pos_input)."""
        if self.train_use_jitter:
            ori_shape = self._shape_hw(img_meta, 'ori_shape')
            if ori_shape is None:
                ori_shape = self._shape_hw(img_meta, 'img_shape')
            h, w = ori_shape
            prop_o, full_pid = generate_jittered_proposals(
                sample_points, (h, w), self.train_infer_base_scales,
                self.train_infer_ratios, self.train_infer_anchor_offsets)
            if (self.train_jitter_use_class_sizes and self.class_box_sizes is not None):
                prop_o = self._rescale_jitter_boxes_with_class_sizes(
                    prop_o,
                    full_pid,
                    lab,
                    h=float(h),
                    w=float(w),
                    device=prop_o.device)
            full_pos_input = self._project_boxes_to_input(prop_o, img_meta)
            return prop_o, full_pid, full_pos_input

        full_excl_orig = self._get_syn_bboxs_original(
            img_meta,
            sample_points,
            num_pos_samples,
            gt_labels=lab,
            device=device)
        full_pos_input = self._project_boxes_to_input(
            full_excl_orig, img_meta)
        return full_excl_orig, None, full_pos_input

    def _full_positive_jitter_or_legacy_input(self, img_meta, sample_points,
                                              num_pos_samples, lab, device):
        if self.train_use_jitter:
            h, w = img_meta['img_shape'][:2]
            hf, wf = float(h), float(w)
            full_pos_input, full_pid = generate_jittered_proposals(
                sample_points, (h, w), self.train_infer_base_scales,
                self.train_infer_ratios, self.train_infer_anchor_offsets)
            if (self.train_jitter_use_class_sizes and self.class_box_sizes is not None):
                ori_shape = self._shape_hw(img_meta, 'ori_shape')
                if (self.class_box_size_mode == 'absolute' and ori_shape is not None
                        and (abs(float(ori_shape[0]) - hf) > 1e-3
                             or abs(float(ori_shape[1]) - wf) > 1e-3)):
                    ph, pw = float(ori_shape[0]), float(ori_shape[1])
                else:
                    ph, pw = hf, wf
                full_pos_input = self._rescale_jitter_boxes_with_class_sizes(
                    full_pos_input,
                    full_pid,
                    lab,
                    h=hf,
                    w=wf,
                    device=full_pos_input.device,
                    prior_h=ph,
                    prior_w=pw)
            return full_pos_input, full_pid
        box = self._get_syn_bboxs(
            img_meta, sample_points, num_pos_samples, device=device)
        return box, None

    def _merge_pos_neg_bboxes(self, pos_bboxes, neg_bboxes, pos_labels=None,
                              pos_point_ids=None):
        # 1. 创建标签张量 (1: 正样本, 0: 负样本)
        # 确保使用与 bbox 相同的 device
        device = pos_bboxes.device if pos_bboxes.numel() > 0 else neg_bboxes.device

        if pos_labels is None:
            pos_labels = torch.ones(
                pos_bboxes.size(0), dtype=torch.long, device=device)
        else:
            pos_labels = pos_labels.to(device=device, dtype=torch.long)
        neg_labels = torch.zeros(neg_bboxes.size(0), dtype=torch.long, device=device)
        neg_point_ids = torch.full(
            (neg_bboxes.size(0),), -1, dtype=torch.long, device=device)

        if pos_point_ids is not None and pos_point_ids.numel() > 0:
            pos_point_ids = pos_point_ids.to(device=device, dtype=torch.long)
            if pos_point_ids.numel() != pos_bboxes.size(0):
                pos_point_ids = None

        # 2. 基础拼接逻辑
        if pos_bboxes.numel() == 0:
            pseudo_bboxes = neg_bboxes
            pseudo_labels = neg_labels
            pseudo_point_ids = neg_point_ids
            bag_label = 0
        elif neg_bboxes.numel() == 0:
            pseudo_bboxes = pos_bboxes
            pseudo_labels = pos_labels
            bag_label = 1
            if pos_point_ids is not None and pos_point_ids.numel() == pos_bboxes.size(
                    0):
                pseudo_point_ids = pos_point_ids
            else:
                pseudo_point_ids = torch.full(
                    (pos_bboxes.size(0),),
                    -1,
                    dtype=torch.long,
                    device=device)
        else:
            pseudo_bboxes = torch.cat([pos_bboxes, neg_bboxes], dim=0)
            pseudo_labels = torch.cat([pos_labels, neg_labels], dim=0)
            bag_label = 1
            if pos_point_ids is not None and pos_point_ids.numel() == pos_bboxes.size(
                    0):
                pseudo_point_ids = torch.cat([pos_point_ids, neg_point_ids], dim=0)
            else:
                pseudo_point_ids = torch.cat([
                    torch.full(
                        (pos_bboxes.size(0),),
                        -1,
                        dtype=torch.long,
                        device=device), neg_point_ids
                ],
                                               dim=0)

        # 3. 执行打乱 (Shuffle)
        if pseudo_bboxes.size(0) > 0:
            perm = torch.randperm(
                pseudo_bboxes.size(0), device=pseudo_bboxes.device)
            pseudo_bboxes = pseudo_bboxes[perm]
            pseudo_labels = pseudo_labels[perm]
            pseudo_point_ids = pseudo_point_ids[perm]

        return pseudo_bboxes, pseudo_labels, bag_label, pseudo_point_ids

    def _get_pos_labels(self, gt_labels, points, num_samples, pos_bboxes, device=None):
        if pos_bboxes.numel() == 0:
            target_device = device or pos_bboxes.device
            return torch.empty((0,), dtype=torch.long, device=target_device)

        target_device = device or pos_bboxes.device
        num_points = int(points.size(0)) if isinstance(points, torch.Tensor) else 0
        if gt_labels is None or num_points == 0:
            return torch.ones((pos_bboxes.size(0),), dtype=torch.long, device=target_device)

        if isinstance(gt_labels, torch.Tensor):
            labels = gt_labels.to(device=target_device, dtype=torch.long).view(-1)
        else:
            labels = torch.as_tensor(gt_labels, dtype=torch.long, device=target_device).view(-1)

        if labels.numel() != num_points:
            return torch.ones((pos_bboxes.size(0),), dtype=torch.long, device=target_device)

        # Class 0 is reserved for background in PointMIL; foreground dataset
        # labels are shifted to 1..C for multi-class instance supervision.
        labels = labels + 1
        return labels.repeat_interleave(int(num_samples))

    def _get_points_from_gt(self, gt_bboxes=None, gt_points=None, device=None):
        # Empty tensor (not Python []) so callers can use .numel() / .device.
        # When gt_points / gt_bboxes are shape (0, *) tensors, len(.) == 0 and
        # neither branch below runs; list fallback used to crash in forward_bags.
        _fallback_device = device if device is not None else torch.device('cpu')
        points = torch.empty((0, 2), dtype=torch.float32, device=_fallback_device)
        if gt_points is not None and len(gt_points) > 0:
            if isinstance(gt_points, torch.Tensor):
                points = gt_points.to(device) if device is not None else gt_points
            else:
                points = torch.stack([
                    pt.to(device) if isinstance(pt, torch.Tensor) and device is not None
                    else pt if isinstance(pt, torch.Tensor)
                    else torch.tensor(pt, dtype=torch.float32, device=device)
                    for pt in gt_points
                ], dim=0)
        elif gt_bboxes is not None and len(gt_bboxes) > 0:
            gt_bboxes_tensor = gt_bboxes.to(device) if device is not None else gt_bboxes
            cx = (gt_bboxes_tensor[:, 0] + gt_bboxes_tensor[:, 2]) / 2
            cy = (gt_bboxes_tensor[:, 1] + gt_bboxes_tensor[:, 3]) / 2
            points = torch.stack((cx, cy), dim=1)
        return points

    def _as_point_tensor(self, points, device=None):
        if points is None:
            target_device = device or torch.device('cpu')
            return torch.empty((0, 2), dtype=torch.float32, device=target_device)
        if isinstance(points, torch.Tensor):
            points = points.to(device) if device is not None else points
            if points.dim() > 2:
                points = points.squeeze(0)
            return points.to(dtype=torch.float32)
        if isinstance(points, (list, tuple)) and len(points) == 0:
            target_device = device or torch.device('cpu')
            return torch.empty((0, 2), dtype=torch.float32, device=target_device)
        return torch.as_tensor(points, dtype=torch.float32, device=device).reshape(-1, 2)

    @staticmethod
    def _shape_hw(img_meta, key):
        shape = img_meta.get(key, None)
        if shape is None:
            return None
        return float(shape[0]), float(shape[1])

    def _scale_factor_tensor(self, img_meta, device, dtype=torch.float32):
        scale_factor = img_meta.get('scale_factor', None)
        if scale_factor is None:
            return torch.ones((4,), dtype=dtype, device=device)
        if isinstance(scale_factor, torch.Tensor):
            sf = scale_factor.to(device=device, dtype=dtype).flatten()
        else:
            sf = torch.as_tensor(scale_factor, dtype=dtype, device=device).flatten()
        if sf.numel() == 1:
            sf = sf.repeat(4)
        elif sf.numel() == 2:
            sf = sf.repeat(2)
        elif sf.numel() >= 4:
            sf = sf[:4]
        else:
            sf = torch.ones((4,), dtype=dtype, device=device)
        return sf

    def _get_original_points(self, img_meta, input_points, device=None):
        ori_points = img_meta.get('ori_gt_points', None)
        if ori_points is not None:
            return self._as_point_tensor(ori_points, device=device)

        points = self._as_point_tensor(input_points, device=device).clone()
        if points.numel() == 0:
            return points

        img_shape = self._shape_hw(img_meta, 'img_shape')
        if img_shape is not None and img_meta.get('flip', False):
            h, w = img_shape
            direction = img_meta.get('flip_direction', 'horizontal')
            if direction == 'horizontal':
                points[:, 0] = float(w) - points[:, 0]
            elif direction == 'vertical':
                points[:, 1] = float(h) - points[:, 1]

        sf = self._scale_factor_tensor(
            img_meta, points.device, dtype=points.dtype)
        points[:, 0] = points[:, 0] / torch.clamp(sf[0], min=1e-6)
        points[:, 1] = points[:, 1] / torch.clamp(sf[1], min=1e-6)

        ori_shape = self._shape_hw(img_meta, 'ori_shape')
        if ori_shape is not None:
            h, w = ori_shape
            points[:, 0] = torch.clamp(points[:, 0], min=0, max=float(w))
            points[:, 1] = torch.clamp(points[:, 1], min=0, max=float(h))
        return points

    def _project_boxes_to_input(self, boxes, img_meta):
        if boxes is None or boxes.numel() == 0:
            device = boxes.device if isinstance(boxes, torch.Tensor) else torch.device('cpu')
            return torch.empty((0, 4), dtype=torch.float32, device=device)

        boxes = boxes.clone()
        sf = self._scale_factor_tensor(
            img_meta, boxes.device, dtype=boxes.dtype)
        boxes = boxes * sf

        img_shape = self._shape_hw(img_meta, 'img_shape')
        if img_shape is None:
            return boxes
        h, w = img_shape

        if img_meta.get('flip', False):
            direction = img_meta.get('flip_direction', 'horizontal')
            if direction == 'horizontal':
                x1 = float(w) - boxes[:, 2].clone()
                x2 = float(w) - boxes[:, 0].clone()
                boxes[:, 0] = x1
                boxes[:, 2] = x2
            elif direction == 'vertical':
                y1 = float(h) - boxes[:, 3].clone()
                y2 = float(h) - boxes[:, 1].clone()
                boxes[:, 1] = y1
                boxes[:, 3] = y2

        boxes[:, 0::2] = torch.clamp(boxes[:, 0::2], min=0, max=float(w))
        boxes[:, 1::2] = torch.clamp(boxes[:, 1::2], min=0, max=float(h))
        return self._ensure_min_input_box_size(boxes, h=float(h), w=float(w))

    def _ensure_min_input_box_size(self, boxes, h, w):
        if boxes.numel() == 0 or self.min_input_box_size <= 0:
            return boxes
        min_size = float(self.min_input_box_size)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        need_w = widths < min_size
        need_h = heights < min_size
        if not (need_w.any() or need_h.any()):
            return boxes

        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        new_w = torch.clamp(widths, min=min_size, max=max(float(w), 1.0))
        new_h = torch.clamp(heights, min=min_size, max=max(float(h), 1.0))
        boxes[:, 0] = torch.clamp(cx - new_w * 0.5, min=0, max=float(w))
        boxes[:, 1] = torch.clamp(cy - new_h * 0.5, min=0, max=float(h))
        boxes[:, 2] = torch.clamp(cx + new_w * 0.5, min=0, max=float(w))
        boxes[:, 3] = torch.clamp(cy + new_h * 0.5, min=0, max=float(h))
        return boxes

    def _labels_for_points(self, gt_labels, num_points, device):
        if gt_labels is None or num_points <= 0:
            return None
        if isinstance(gt_labels, torch.Tensor):
            labels = gt_labels.to(device=device, dtype=torch.long).view(-1)
        else:
            labels = torch.as_tensor(
                gt_labels, dtype=torch.long, device=device).view(-1)
        if labels.numel() != num_points:
            return None
        return labels

    def _global_box_sizes_tensor(self, device, h, w):
        sizes = torch.as_tensor(
            self.box_sizes, dtype=torch.float32, device=device).reshape(-1, 2)
        if self.class_box_size_mode == 'ratio' and self.sample_coordinate_mode == 'original':
            scale = torch.tensor([float(w), float(h)], dtype=torch.float32, device=device)
            sizes = sizes * scale
        sizes[:, 0] = torch.clamp(sizes[:, 0], min=1.0, max=max(float(w), 1.0))
        sizes[:, 1] = torch.clamp(sizes[:, 1], min=1.0, max=max(float(h), 1.0))
        return sizes

    def _class_box_sizes_tensor(self, device, h, w):
        if self.class_box_sizes is None:
            return None
        sizes = torch.as_tensor(
            self.class_box_sizes, dtype=torch.float32, device=device)
        if self.class_box_size_mode == 'ratio':
            scale = torch.tensor([float(w), float(h)], dtype=torch.float32, device=device)
            sizes = sizes * scale
        sizes[..., 0] = torch.clamp(sizes[..., 0], min=1.0, max=max(float(w), 1.0))
        sizes[..., 1] = torch.clamp(sizes[..., 1], min=1.0, max=max(float(h), 1.0))
        return sizes

    def _apply_size_jitter(self, sizes):
        if self.size_jitter <= 0:
            return sizes
        low = max(0.0, 1.0 - self.size_jitter)
        high = 1.0 + self.size_jitter
        jitter = torch.rand_like(sizes) * (high - low) + low
        return sizes * jitter

    def _sample_point_class_sizes(self, labels, num_points, num_samples, h, w, device):
        global_sizes = self._global_box_sizes_tensor(device, h=h, w=w)
        class_sizes = self._class_box_sizes_tensor(device, h=h, w=w)
        if class_sizes is None or labels is None or num_points == 0:
            rand = torch.randint(
                0, global_sizes.size(0), (num_points, num_samples), device=device)
            sizes = global_sizes[rand]
            return self._apply_size_jitter(sizes)

        sizes_per_point = []
        num_classes, num_sizes = class_sizes.shape[:2]
        for label in labels.tolist():
            if 0 <= int(label) < num_classes:
                pool = class_sizes[int(label)]
            else:
                pool = global_sizes
            rand = torch.randint(0, pool.size(0), (num_samples,), device=device)
            sizes_per_point.append(pool[rand])
        sizes = torch.stack(sizes_per_point, dim=0)
        return self._apply_size_jitter(sizes)

    def _sample_negative_sizes(self, num_samples, h, w, device):
        use_class_prior = (
            self.negative_size_source == 'class_prior'
            and self.class_box_sizes is not None)
        if use_class_prior:
            sizes_pool = self._class_box_sizes_tensor(device, h=h, w=w).reshape(-1, 2)
        else:
            sizes_pool = self._global_box_sizes_tensor(device, h=h, w=w)
        rand = torch.randint(0, sizes_pool.size(0), (num_samples,), device=device)
        sizes = sizes_pool[rand]
        return self._apply_size_jitter(sizes)

    def _rescale_jitter_boxes_with_class_sizes(self,
                                                proposals_xyxy,
                                                point_indices,
                                                labels,
                                                h,
                                                w,
                                                device,
                                                prior_h=None,
                                                prior_w=None):
        """Keep jitter centers; replace (W,H) with class/global priors.

        ``(h, w)`` match the coordinate frame of ``proposals_xyxy`` (clamp space).
        When ``prior_h`` / ``prior_w`` differ (e.g. ori priors, input boxes), class
        priors are sampled in prior space then scaled into clamp space; invalid
        class ids use ``box_sizes`` in clamp space (same as global pool there).
        """
        if proposals_xyxy.numel() == 0:
            return proposals_xyxy
        if prior_h is None:
            prior_h = h
        if prior_w is None:
            prior_w = w
        dtype = proposals_xyxy.dtype
        M = proposals_xyxy.size(0)
        pid = point_indices.long()
        hf, wf = float(h), float(w)
        phf, pwf = float(prior_h), float(prior_w)

        global_out = self._global_box_sizes_tensor(device, h=hf, w=wf)
        use_scale = (abs(phf - hf) > 1e-3) or (abs(pwf - wf) > 1e-3)
        scale_wh = None
        if use_scale:
            sh = hf / phf if phf > 1e-6 else 1.0
            sw = wf / pwf if pwf > 1e-6 else 1.0
            scale_wh = proposals_xyxy.new_tensor([sw, sh])

        class_prior = self._class_box_sizes_tensor(device, h=phf, w=pwf)

        if class_prior is None or labels is None or labels.numel() == 0:
            rand = torch.randint(0, global_out.size(0), (M,), device=device)
            chosen = global_out[rand]
        else:
            num_classes = class_prior.size(0)
            K = class_prior.size(1)
            cls_ids = labels[pid]
            valid = (cls_ids >= 0) & (cls_ids < num_classes)
            rand_k = torch.randint(0, K, (M,), device=device)
            clamped = cls_ids.clamp(0, num_classes - 1)
            gathered = class_prior[clamped, rand_k]
            if use_scale and scale_wh is not None:
                gathered = gathered * scale_wh
                gathered[:, 0] = torch.clamp(
                    gathered[:, 0], min=1.0, max=max(wf, 1.0))
                gathered[:, 1] = torch.clamp(
                    gathered[:, 1], min=1.0, max=max(hf, 1.0))
            rand_g = torch.randint(0, global_out.size(0), (M,), device=device)
            fallback = global_out[rand_g]
            chosen = torch.where(valid.unsqueeze(1), gathered, fallback)

        chosen = self._apply_size_jitter(chosen)
        cx = (proposals_xyxy[:, 0] + proposals_xyxy[:, 2]) * 0.5
        cy = (proposals_xyxy[:, 1] + proposals_xyxy[:, 3]) * 0.5
        x1 = torch.clamp(cx - chosen[:, 0] * 0.5, min=0.0, max=wf)
        y1 = torch.clamp(cy - chosen[:, 1] * 0.5, min=0.0, max=hf)
        x2 = torch.clamp(cx + chosen[:, 0] * 0.5, min=0.0, max=wf)
        y2 = torch.clamp(cy + chosen[:, 1] * 0.5, min=0.0, max=hf)
        return torch.stack([x1, y1, x2, y2], dim=-1).to(dtype=dtype)

    def _sample_offsets(self, sizes, device):
        if self.box_offset_mode == 'size_ratio':
            max_offsets = sizes * float(self.box_offset_ratio)
            return (torch.rand_like(sizes) * 2.0 - 1.0) * max_offsets
        return (
            torch.rand(sizes.shape, dtype=sizes.dtype, device=device)
            * (2 * self.box_offset) - self.box_offset)

    def _get_syn_bboxs_original(self, img_meta, points, num_samples,
                                gt_labels=None, device=None):
        points = self._as_point_tensor(points, device=device)
        device = points.device
        if points.numel() == 0:
            return torch.empty((0, 4), dtype=torch.float32, device=device)

        ori_shape = self._shape_hw(img_meta, 'ori_shape')
        if ori_shape is None:
            ori_shape = self._shape_hw(img_meta, 'img_shape')
        h, w = ori_shape
        num_points = points.size(0)
        labels = self._labels_for_points(gt_labels, num_points, device)
        chosen_sizes = self._sample_point_class_sizes(
            labels, num_points, num_samples, h=h, w=w, device=device)
        offsets = self._sample_offsets(chosen_sizes, device=device)
        centers = points.unsqueeze(1) + offsets

        x1 = torch.clamp(centers[..., 0] - chosen_sizes[..., 0] / 2, min=0, max=w)
        y1 = torch.clamp(centers[..., 1] - chosen_sizes[..., 1] / 2, min=0, max=h)
        x2 = torch.clamp(centers[..., 0] + chosen_sizes[..., 0] / 2, min=0, max=w)
        y2 = torch.clamp(centers[..., 1] + chosen_sizes[..., 1] / 2, min=0, max=h)
        return torch.stack([x1, y1, x2, y2], dim=-1).view(-1, 4)

    def _get_syn_bboxs(self, img_meta, points, num_samples, device=None):
        target_device = device
        if target_device is None and isinstance(points, torch.Tensor):
            target_device = points.device

        if points is None:
            target_device = target_device or torch.device('cpu')
            return torch.empty((0, 4), dtype=torch.float32, device=target_device)

        if isinstance(points, (list, tuple)):
            if len(points) == 0:
                target_device = target_device or torch.device('cpu')
                return torch.empty((0, 4), dtype=torch.float32, device=target_device)

            points = torch.stack([
                pt.to(target_device) if isinstance(pt, torch.Tensor) and target_device is not None
                else pt if isinstance(pt, torch.Tensor)
                else torch.tensor(pt, dtype=torch.float32, device=target_device)
                for pt in points
            ], dim=0)

        if target_device is None:
            target_device = points.device

        if isinstance(points, torch.Tensor) and points.device != target_device:
            points = points.to(target_device)

        if isinstance(points, torch.Tensor) and points.dim() > 2:
            points = points.squeeze(0)
        
        # 确保在 GPU 上处理，避免 CPU/GPU 切换带来的性能损耗
        if points.numel() == 0:
            return torch.empty((0, 4), dtype=torch.float32, device=target_device)

        h, w = img_meta['img_shape'][:2]
        device = target_device
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
        # 修改：同时限制 min 和 max，防止越界导致 x1 > x2
        x1 = torch.clamp(centers[..., 0] - chosen_sizes[..., 0] / 2, min=0, max=w)
        y1 = torch.clamp(centers[..., 1] - chosen_sizes[..., 1] / 2, min=0, max=h)
        x2 = torch.clamp(centers[..., 0] + chosen_sizes[..., 0] / 2, min=0, max=w)
        y2 = torch.clamp(centers[..., 1] + chosen_sizes[..., 1] / 2, min=0, max=h)
        
        # 重塑为 (N_total, 4)
        syn_boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(-1, 4)
        
        return syn_boxes

    def _generate_negative_samples_original(self, img_meta, syn_bboxes_orig,
                                            num_neg_required=None, device=None):
        target_num_samples = (
            num_neg_required if num_neg_required is not None else self.num_neg_samples)
        if target_num_samples <= 0:
            target_device = (
                device or
                (syn_bboxes_orig.device if syn_bboxes_orig is not None and syn_bboxes_orig.numel() > 0
                 else torch.device('cpu')))
            return torch.empty((0, 4), dtype=torch.float32, device=target_device)

        if device is None:
            device = (
                syn_bboxes_orig.device
                if syn_bboxes_orig is not None and syn_bboxes_orig.numel() > 0
                else torch.device('cpu'))

        ori_shape = self._shape_hw(img_meta, 'ori_shape')
        if ori_shape is None:
            ori_shape = self._shape_hw(img_meta, 'img_shape')
        h, w = ori_shape

        pool_size = max(target_num_samples * 10, 1000)
        sizes = self._sample_negative_sizes(
            pool_size, h=h, w=w, device=device)

        max_x1 = torch.clamp(
            torch.as_tensor(float(w), device=device) - sizes[:, 0], min=0.0)
        max_y1 = torch.clamp(
            torch.as_tensor(float(h), device=device) - sizes[:, 1], min=0.0)
        x1 = torch.rand(pool_size, device=device) * max_x1
        y1 = torch.rand(pool_size, device=device) * max_y1
        x2 = torch.clamp(x1 + sizes[:, 0], max=w)
        y2 = torch.clamp(y1 + sizes[:, 1], max=h)
        candidates_orig = torch.stack([x1, y1, x2, y2], dim=1)

        if syn_bboxes_orig is not None and syn_bboxes_orig.numel() > 0:
            max_ious = self._max_iou_each_candidate_vs_ref(
                candidates_orig, syn_bboxes_orig)
            neg_bboxes_orig = candidates_orig[max_ious == 0]
        else:
            neg_bboxes_orig = candidates_orig

        num_valid = neg_bboxes_orig.size(0)
        if num_valid >= target_num_samples:
            perm = torch.randperm(num_valid, device=device)[:target_num_samples]
            neg_bboxes_orig = neg_bboxes_orig[perm]
        elif num_valid > 0:
            needed = target_num_samples - num_valid
            fill_indices = torch.randint(0, num_valid, (needed,), device=device)
            neg_bboxes_orig = torch.cat(
                [neg_bboxes_orig, neg_bboxes_orig[fill_indices]], dim=0)
        else:
            neg_bboxes_orig = candidates_orig[:target_num_samples]

        return self._project_boxes_to_input(neg_bboxes_orig, img_meta)

    def _generate_negative_samples(self, img_meta, syn_bboxes, num_neg_required=None, device=None):
        # 如果未传入具体数量，使用初始化时的默认值
        target_num_samples = num_neg_required if num_neg_required is not None else self.num_neg_samples

        h, w = img_meta['img_shape'][:2]
        # 确保负样本和正样本/特征图使用同一设备，避免 bbox2roi 拼接失败
        if device is None:
            device = syn_bboxes.device if syn_bboxes.numel() > 0 else torch.device('cpu')

        box_sizes_tensor = torch.tensor(self.box_sizes, device=device, dtype=torch.float32)
        
        # 增加候选池大小以提高不重叠采样的成功率 (基于目标数量动态调整)
        pool_size = max(target_num_samples * 10, 1000)
        
        # 1. 批量生成随机候选框
        rand_indices = torch.randint(0, len(self.box_sizes), (pool_size,), device=device)
        sizes = box_sizes_tensor[rand_indices]
        sizes[:, 0] = torch.clamp(sizes[:, 0], min=1.0, max=max(float(w), 1.0))
        sizes[:, 1] = torch.clamp(sizes[:, 1], min=1.0, max=max(float(h), 1.0))

        # 随机生成左上角点
        max_x1 = torch.clamp(torch.as_tensor(float(w), device=device) - sizes[:, 0], min=0.0)
        max_y1 = torch.clamp(torch.as_tensor(float(h), device=device) - sizes[:, 1], min=0.0)
        x1 = torch.rand(pool_size, device=device) * max_x1
        y1 = torch.rand(pool_size, device=device) * max_y1
        x2 = torch.clamp(x1 + sizes[:, 0], max=w)
        y2 = torch.clamp(y1 + sizes[:, 1], max=h)
        
        candidates = torch.stack([x1, y1, x2, y2], dim=1)

        # 2. 过滤掉与正样本重叠的框
        if syn_bboxes.numel() > 0:
            max_ious = self._max_iou_each_candidate_vs_ref(candidates, syn_bboxes)
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
