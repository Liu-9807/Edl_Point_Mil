import os
import os.path as osp

import mmcv
from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class MILMaskRefineVisHook(Hook):
    """Visualize per-ROI mask refinement debug outputs."""

    def __init__(self,
                 interval=50,
                 out_dir=None,
                 show=False,
                 max_items=20,
                 max_points=5,
                 max_proposals_per_point=20):
        self.interval = interval
        self.out_dir = out_dir
        self.show = show
        self.max_items = max_items
        self.max_points = max_points
        self.max_proposals_per_point = max_proposals_per_point
        self._vis_dir = None

    def before_run(self, runner):
        self._init_vis_dir(runner)

    def before_test(self, runner):
        self._init_vis_dir(runner)

    def _init_vis_dir(self, runner):
        if self.out_dir:
            self._vis_dir = self.out_dir
        else:
            self._vis_dir = osp.join(runner.work_dir, runner.timestamp, 'vis_mask_refine')
        if runner.rank == 0:
            os.makedirs(self._vis_dir, exist_ok=True)

    def before_val_iter(self, runner, batch_idx, data_batch=None):
        if not self.every_n_inner_iters(batch_idx, self.interval):
            return

    def before_test_iter(self, runner, batch_idx, data_batch=None):
        if not self.every_n_inner_iters(batch_idx, self.interval):
            return

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self.every_n_inner_iters(batch_idx, self.interval):
            return
        self._draw_batch(runner, batch_idx, data_batch, outputs, stage='val')

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self.every_n_inner_iters(batch_idx, self.interval):
            return
        self._draw_batch(runner, batch_idx, data_batch, outputs, stage='test')

    def _draw_batch(self, runner, batch_idx, data_batch, outputs, stage='test'):
        visualizer = runner.visualizer
        if visualizer is None or outputs is None:
            return
        has_combined = hasattr(visualizer, 'draw_proposal_mask_refine_debug')
        if not has_combined:
            return

        data_samples = data_batch.get('data_samples', []) if data_batch else []

        for i, pred_sample in enumerate(outputs):
            debug = getattr(pred_sample, 'mil_debug', None)
            if debug is None:
                continue

            mask_refine_items = debug.get('mask_refine_debug', None)
            proposal_score_items = debug.get('proposal_score_debug', None)
            if not mask_refine_items or not proposal_score_items:
                continue

            gt_sample = data_samples[i] if i < len(data_samples) else None
            img_path = getattr(gt_sample, 'img_path', None) if gt_sample is not None else getattr(pred_sample, 'img_path', None)
            if not img_path or not osp.exists(img_path):
                continue

            img = mmcv.imread(img_path, channel_order='rgb')
            name_prefix = f'{stage}_iter{batch_idx}_s{i}'
            combined_items = visualizer.draw_proposal_mask_refine_debug(
                image=img,
                score_items=proposal_score_items,
                mask_items=mask_refine_items,
                max_points=self.max_points,
                max_items=self.max_proposals_per_point)

            if self._vis_dir and runner.rank == 0:
                for j, vis_img in enumerate(combined_items):
                    out_file = osp.join(self._vis_dir, f'{name_prefix}_point{j}.jpg')
                    mmcv.imwrite(vis_img[..., ::-1], out_file)

            if runner.rank == 0:
                for j, vis_img in enumerate(combined_items):
                    visualizer.add_image(
                        f'{stage}_proposal_mask_refine/{name_prefix}_point{j}',
                        vis_img,
                        step=runner.iter)

            break

    @staticmethod
    def every_n_inner_iters(batch_idx, interval):
        if interval <= 0:
            return False
        return (batch_idx + 1) % interval == 0
