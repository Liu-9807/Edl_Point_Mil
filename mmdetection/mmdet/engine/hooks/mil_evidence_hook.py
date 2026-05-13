import os
import os.path as osp
import random

import mmcv
import numpy as np
import torch
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


def _apply_scale_flip_boxes_points(bboxes, points, img_meta):
    """Map boxes and points from input-space to original image space (CPU tensors)."""
    if bboxes is None:
        return None, points
    b = bboxes.clone()
    p = points.clone() if points is not None else None

    scale_factor = img_meta.get('scale_factor', None)
    if scale_factor is not None:
        if not isinstance(scale_factor, torch.Tensor):
            sf = torch.tensor(scale_factor, device=b.device, dtype=b.dtype)
        else:
            sf = scale_factor.to(device=b.device, dtype=b.dtype)
        if sf.numel() >= 2:
            if sf.numel() == 2:
                sf_bbox = sf.repeat(2)
                sf_point = sf
            else:
                sf_bbox = sf
                sf_point = sf[:2]
            b = b / sf_bbox
            if p is not None:
                p = p / sf_point

    if img_meta.get('flip', False):
        img_h, img_w = img_meta['ori_shape'][:2]
        if img_meta.get('flip_direction', 'horizontal') == 'horizontal':
            bx1 = img_w - b[:, 2]
            bx2 = img_w - b[:, 0]
            b[:, 0] = bx1
            b[:, 2] = bx2
            if p is not None:
                p[:, 0] = img_w - p[:, 0]
    return b, p


@HOOKS.register_module()
class MILEvidenceHook(Hook):
    """Mixed-bag EDL evidence + optional instance-mask panel (one stacked figure).

    When ``bbox_head.use_instance_mask`` is True and ``combine_mask_vis`` is True,
    ``save_mask_debug`` is enabled on the same cadence as evidence; mask rows match
    the same ``chosen`` bag and ``local_idx`` as the evidence panel.
    """

    def __init__(self,
                 interval=100,
                 out_dir=None,
                 n_per_side=None,
                 max_instances=None,
                 global_max_side=720,
                 positive_class_ids=None,
                 background_label=0,
                 patch_barh=True,
                 combine_mask_vis=True,
                 epoch_snapshot_interval=1,
                 num_samples=3,
                 seed=42):
        self.interval = interval
        self._out_dir = out_dir
        self.out_dir = None
        if n_per_side is not None:
            self.n_per_side = int(n_per_side)
        elif max_instances is not None:
            self.n_per_side = int(max_instances)
        else:
            self.n_per_side = 3
        self.global_max_side = int(global_max_side)
        self.positive_class_ids = None if positive_class_ids is None else tuple(
            int(x) for x in positive_class_ids)
        self.background_label = int(background_label)
        self.patch_barh = bool(patch_barh)
        self.combine_mask_vis = bool(combine_mask_vis)
        self.epoch_snapshot_interval = int(epoch_snapshot_interval)
        self.num_samples = int(num_samples)
        self._rng = random.Random(seed)
        self._epoch_candidates = []
        self._seen_candidates = 0

    def before_run(self, runner):
        if self._out_dir is None:
            self.out_dir = os.path.join(runner.work_dir, runner.timestamp,
                                        'vis_data/instance_bag_vis')
        else:
            self.out_dir = self._out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def before_train_epoch(self, runner):
        self._epoch_candidates = []
        self._seen_candidates = 0

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        if not self.every_n_train_iters(runner, self.interval):
            return
        model = runner.model.module if hasattr(runner.model,
                                               'module') else runner.model
        if not hasattr(model, 'roi_head') or not hasattr(model.roi_head, 'bbox_head'):
            return
        bh = model.roi_head.bbox_head
        bh.save_debug_info = True
        if self.combine_mask_vis and getattr(bh, 'use_instance_mask', False):
            bh.save_mask_debug = True

    def _score_for_pos_ranking(self, alpha_row):
        """Higher = more foreground-like under current class-id convention."""
        s = float(alpha_row.sum().clamp(min=1e-6))
        a = alpha_row / s
        if self.positive_class_ids is None:
            if a.numel() <= 1:
                return float(a.sum())
            return float(a[1:].sum())
        acc = torch.zeros((), device=a.device, dtype=a.dtype)
        for c in self.positive_class_ids:
            if 0 <= int(c) < a.numel():
                acc = acc + a[int(c)]
        return float(acc)

    def _score_for_neg_ranking(self, alpha_row):
        s = float(alpha_row.sum().clamp(min=1e-6))
        a = alpha_row / s
        bi = self.background_label
        if 0 <= bi < a.numel():
            return float(a[bi])
        return float(a[0]) if a.numel() > 0 else 0.0

    def _pick_local_indices(self, lab, ins_b):
        pos_pool = torch.where(lab > 0)[0]
        neg_pool = torch.where(lab == 0)[0]
        n = self.n_per_side
        pos_idx = []
        neg_idx = []
        if pos_pool.numel() > 0:
            scores = torch.tensor(
                [self._score_for_pos_ranking(ins_b[i]) for i in pos_pool],
                device=ins_b.device)
            order = torch.argsort(scores, descending=True)
            take = min(n, pos_pool.numel())
            pos_idx = pos_pool[order[:take]].tolist()
        if neg_pool.numel() > 0:
            scores = torch.tensor(
                [self._score_for_neg_ranking(ins_b[i]) for i in neg_pool],
                device=ins_b.device)
            order = torch.argsort(scores, descending=True)
            take = min(n, neg_pool.numel())
            neg_idx = neg_pool[order[:take]].tolist()
        if len(pos_idx) == 0 and len(neg_idx) == 0:
            return None
        local = torch.tensor(pos_idx + neg_idx, dtype=torch.long, device=lab.device)
        is_pos = torch.tensor([True] * len(pos_idx) + [False] * len(neg_idx),
                              dtype=torch.bool,
                              device=lab.device)
        return local, is_pos

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self.every_n_train_iters(runner, self.interval):
            return

        model = runner.model.module if hasattr(runner.model,
                                               'module') else runner.model
        roi_head = getattr(model, 'roi_head', None)
        if roi_head is None or not hasattr(roi_head, 'bbox_head'):
            return

        try:
            dbg = getattr(roi_head, '_mil_evidence_debug', None)
            if dbg is None:
                return

            visualizer = runner.visualizer
            if not hasattr(visualizer, 'draw_mixed_bag_evidence_panel'):
                return

            rois = dbg['rois']
            ins_output = dbg['ins_output'].float()
            ins_labels = dbg['ins_labels']
            bag_labels = dbg['bag_labels'].view(-1)
            bag_to_img = dbg['bag_to_img']
            pseudo_point_ids = dbg['pseudo_point_ids']

            num_bags = int(bag_labels.numel())
            chosen = None
            for bag_idx in range(num_bags):
                bl = int(bag_labels[bag_idx].item())
                mask = rois[:, 0].long() == bag_idx
                if not mask.any():
                    continue
                lab = ins_labels[mask]
                if bl == 1 and (lab > 0).any() and (lab == 0).any():
                    chosen = bag_idx
                    break
            if chosen is None:
                return

            mask = rois[:, 0].long() == chosen
            lab = ins_labels[mask]
            ins_b = ins_output[mask]
            pid_b = pseudo_point_ids[mask]
            roi_boxes = rois[mask, 1:5].float()

            pick = self._pick_local_indices(lab, ins_b)
            if pick is None:
                return
            local_idx, is_pos_t = pick

            boxes_t = roi_boxes[local_idx].detach().cpu().float()
            alphas_t = ins_b[local_idx].detach().cpu().float()
            pid_sel = pid_b[local_idx].detach().cpu()

            img_id = int(bag_to_img[chosen])
            data_samples = data_batch.get('data_samples', None) if isinstance(
                data_batch, dict) else None
            if not data_samples:
                return
            sample = data_samples[img_id]
            img_meta = sample.metainfo
            img_path = img_meta.get('img_path', None)
            if img_path is None or not osp.exists(img_path):
                return

            raw_image = mmcv.imread(img_path, channel_order='rgb')

            gt_pts = None
            gt_plab = None
            if hasattr(sample, 'gt_instances') and hasattr(sample.gt_instances,
                                                           'points'):
                gt_pts = sample.gt_instances.points
                if gt_pts is not None:
                    gt_pts = gt_pts.detach().cpu().float()
                if gt_pts is not None and hasattr(sample.gt_instances, 'labels'):
                    gl = sample.gt_instances.labels
                    if gl is not None and gl.numel() > 0:
                        gt_plab = gl.detach().cpu()

            b_ori, p_ori = _apply_scale_flip_boxes_points(boxes_t, gt_pts, img_meta)
            boxes_np = b_ori.numpy()
            alphas_np = alphas_t.numpy()
            is_pos_np = is_pos_t.detach().cpu().numpy()
            ref_xy = np.full((boxes_np.shape[0], 2), np.nan, dtype=np.float64)
            if p_ori is not None:
                p_np = p_ori.cpu().numpy()
                for i in range(boxes_np.shape[0]):
                    if not bool(is_pos_np[i]):
                        continue
                    pid = int(pid_sel[i].item())
                    if 0 <= pid < p_np.shape[0]:
                        ref_xy[i] = p_np[pid]

            gt_all = None
            gt_all_lab = None
            if p_ori is not None:
                gt_all = p_ori.cpu().numpy()
                if gt_plab is not None:
                    gt_all_lab = gt_plab.detach().cpu().numpy()

            mask_ready = False
            m_sel = None
            bh = roi_head.bbox_head
            if (self.combine_mask_vis and getattr(bh, 'use_instance_mask', False)
                    and hasattr(visualizer, 'draw_mixed_bag_evidence_mask_unified_panel')):
                mdbg = getattr(roi_head, '_mil_mask_debug', None)
                if mdbg is not None and mdbg.get('mask_2d') is not None:
                    mrois = mdbg['rois']
                    mm = mrois[:, 0].long() == int(chosen)
                    if mm.any():
                        idx_cpu = local_idx.detach().cpu().long()
                        mask_block = mdbg['mask_2d'][mm]
                        n_rows = int(mask_block.shape[0])
                        if (idx_cpu.numel() > 0 and int(idx_cpu.min()) >= 0
                                and int(idx_cpu.max()) < n_rows):
                            m_sel = mask_block[idx_cpu].clone()
                            if m_sel.numel() > 0:
                                mask_ready = True

            if (mask_ready and m_sel is not None
                    and hasattr(visualizer, 'draw_mixed_bag_evidence_mask_unified_panel')):
                out_img = visualizer.draw_mixed_bag_evidence_mask_unified_panel(
                    raw_image,
                    boxes_np,
                    alphas_np,
                    m_sel,
                    is_pos_np,
                    ref_xy,
                    gt_all,
                    global_max_side=self.global_max_side,
                    gt_point_labels=gt_all_lab,
                    patch_barh=self.patch_barh,
                )
            else:
                out_img = visualizer.draw_mixed_bag_evidence_panel(
                    raw_image,
                    boxes_np,
                    alphas_np,
                    is_pos_np,
                    ref_xy,
                    gt_all,
                    global_max_side=self.global_max_side,
                    gt_point_labels=gt_all_lab,
                    patch_barh=self.patch_barh,
                )

            visualizer.add_image(
                'mil_debug/instance_bag_vis', out_img, step=runner.iter)
            if self.out_dir:
                save_path = osp.join(self.out_dir,
                                     f'iter_{runner.iter}_bag_vis.png')
                mmcv.imwrite(out_img[..., ::-1], save_path)

            candidate_name = f'iter_{runner.iter}_bag_img{img_id}_bag{chosen}'
            self._update_reservoir(candidate_name, out_img)
        finally:
            if hasattr(roi_head, 'bbox_head'):
                roi_head.bbox_head.save_debug_info = False
                roi_head.bbox_head.save_mask_debug = False
            if hasattr(roi_head, '_mil_evidence_debug'):
                del roi_head._mil_evidence_debug
            if hasattr(roi_head, '_mil_mask_debug'):
                del roi_head._mil_mask_debug

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.epoch_snapshot_interval):
            return
        if len(self._epoch_candidates) == 0:
            return
        visualizer = runner.visualizer
        for idx, (name, img) in enumerate(self._epoch_candidates):
            tag = f'mil_epoch/bag_vis_{idx}'
            visualizer.add_image(tag, img, step=runner.epoch)
        self._epoch_candidates = []
        self._seen_candidates = 0

    def _update_reservoir(self, name, vis_img):
        self._seen_candidates += 1
        if len(self._epoch_candidates) < self.num_samples:
            self._epoch_candidates.append((name, vis_img))
            return
        replace_pos = self._rng.randint(0, self._seen_candidates - 1)
        if replace_pos < self.num_samples:
            self._epoch_candidates[replace_pos] = (name, vis_img)
