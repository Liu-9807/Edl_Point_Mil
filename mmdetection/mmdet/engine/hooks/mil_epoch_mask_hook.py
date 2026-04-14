import os
import os.path as osp
import random

import mmcv
import torch
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class MILEpochMaskHook(Hook):
    """Visualize 2D mask strength for randomly sampled training samples each epoch."""

    def __init__(self,
                 interval=1,
                 num_samples=3,
                 instances_per_sample=4,
                 collect_interval=20,
                 out_dir=None,
                 seed=42):
        self.interval = interval
        self.num_samples = num_samples
        self.instances_per_sample = instances_per_sample
        self.collect_interval = collect_interval
        self._out_dir = out_dir
        self.out_dir = None

        self._rng = random.Random(seed)
        self._epoch_candidates = []
        self._seen_candidates = 0

    def before_run(self, runner):
        if self._out_dir is None:
            self.out_dir = osp.join(
                runner.work_dir, runner.timestamp, 'vis_data/epoch_mask_strength')
        else:
            self.out_dir = self._out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def before_train_epoch(self, runner):
        self._epoch_candidates = []
        self._seen_candidates = 0

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        if not self.every_n_train_iters(runner, self.collect_interval):
            return

        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        roi_head = getattr(model, 'roi_head', None)
        if roi_head is None or not hasattr(roi_head, 'bbox_head'):
            return
        roi_head.bbox_head.save_mask_debug = True

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self.every_n_train_iters(runner, self.collect_interval):
            return

        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        roi_head = getattr(model, 'roi_head', None)
        if roi_head is None or not hasattr(roi_head, 'bbox_head'):
            return

        bbox_head = roi_head.bbox_head
        debug_data = getattr(bbox_head, '_last_mask_debug_data', None)

        bbox_head.save_mask_debug = False
        if debug_data is None:
            return

        visualizer = runner.visualizer
        if not hasattr(visualizer, 'draw_instance_mask_strength'):
            if hasattr(bbox_head, '_last_mask_debug_data'):
                del bbox_head._last_mask_debug_data
            return

        data_samples = data_batch.get('data_samples', None) if isinstance(data_batch, dict) else None
        if not data_samples:
            if hasattr(bbox_head, '_last_mask_debug_data'):
                del bbox_head._last_mask_debug_data
            return

        batch_id = self._rng.randrange(len(data_samples))
        data_sample = data_samples[batch_id]
        img_path = data_sample.metainfo.get('img_path', None)
        if img_path is None or not osp.exists(img_path):
            if hasattr(bbox_head, '_last_mask_debug_data'):
                del bbox_head._last_mask_debug_data
            return

        rois = debug_data['rois']
        mask_2d = debug_data['mask_2d']
        ins_output = debug_data['ins_output']

        bag_mask = rois[:, 0] == batch_id
        if bag_mask.sum().item() == 0:
            if hasattr(bbox_head, '_last_mask_debug_data'):
                del bbox_head._last_mask_debug_data
            return

        bag_rois = rois[bag_mask, 1:].clone()
        bag_mask_2d = mask_2d[bag_mask]
        bag_scores = ins_output[bag_mask]

        sf = data_sample.metainfo.get('scale_factor', None)
        if sf is not None:
            if not isinstance(sf, torch.Tensor):
                sf = torch.tensor(sf, dtype=bag_rois.dtype)
            sf = sf.to(dtype=bag_rois.dtype)
            if sf.numel() >= 2:
                if sf.numel() == 2:
                    sf_bbox = sf.repeat(2)
                else:
                    sf_bbox = sf[:4]
                bag_rois[:, 0] /= sf_bbox[0]
                bag_rois[:, 1] /= sf_bbox[1]
                bag_rois[:, 2] /= sf_bbox[2]
                bag_rois[:, 3] /= sf_bbox[3]

        num_inst = bag_rois.shape[0]
        choose_k = min(self.instances_per_sample, num_inst)
        sample_indices = self._rng.sample(range(num_inst), k=choose_k)

        raw_image = mmcv.imread(img_path, channel_order='rgb')
        vis_img = visualizer.draw_instance_mask_strength(
            image=raw_image,
            bboxes=bag_rois,
            mask_2d=bag_mask_2d,
            instance_scores=bag_scores,
            sample_indices=sample_indices,
            epoch_num=runner.epoch,
            iter_num=runner.iter)

        candidate_name = f'ep{runner.epoch}_it{runner.iter}_b{batch_id}'
        self._update_reservoir(candidate_name, vis_img)

        if hasattr(bbox_head, '_last_mask_debug_data'):
            del bbox_head._last_mask_debug_data

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return

        if len(self._epoch_candidates) == 0:
            return

        visualizer = runner.visualizer
        for idx, (name, img) in enumerate(self._epoch_candidates):
            tag = f'mil_epoch/mask_strength_{idx}'
            visualizer.add_image(tag, img, step=runner.epoch)
            if self.out_dir is not None:
                out_path = osp.join(self.out_dir, f'{name}.png')
                mmcv.imwrite(img[..., ::-1], out_path)

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
