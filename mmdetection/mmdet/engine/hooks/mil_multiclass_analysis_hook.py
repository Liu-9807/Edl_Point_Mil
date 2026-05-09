import os
import os.path as osp

import mmcv
import torch
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class MILMultiClassAnalysisHook(Hook):
    """Visualize class-aware MIL instance predictions during training.

    The hook is opt-in. It toggles a debug flag on ``MILRoIHead`` only for
    selected iterations, so existing binary experiments do not allocate or
    cache extra tensors unless this hook is explicitly configured.
    """

    def __init__(self,
                 interval=100,
                 max_instances=4096,
                 topk=(1, 5),
                 background_label=0,
                 out_dir=None):
        self.interval = int(interval)
        self.max_instances = int(max_instances)
        self.topk = tuple(int(k) for k in topk if int(k) > 0)
        self.background_label = int(background_label)
        self._out_dir = out_dir
        self.out_dir = None

    def before_run(self, runner):
        if self._out_dir is None:
            self.out_dir = osp.join(
                runner.work_dir, runner.timestamp, 'vis_data/mil_multiclass')
        else:
            self.out_dir = self._out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, 'module') else model

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        if not self.every_n_train_iters(runner, self.interval):
            return
        model = self._unwrap_model(runner.model)
        roi_head = getattr(model, 'roi_head', None)
        if roi_head is not None:
            roi_head.debug_multiclass_analysis = True

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self.every_n_train_iters(runner, self.interval):
            return

        model = self._unwrap_model(runner.model)
        roi_head = getattr(model, 'roi_head', None)
        if roi_head is None:
            return

        debug_data = getattr(roi_head, '_last_multiclass_debug', None)
        roi_head.debug_multiclass_analysis = False
        if debug_data is None:
            return

        try:
            scores = debug_data.get('ins_scores', None)
            labels = debug_data.get('ins_labels', None)
            if scores is None or labels is None or labels.numel() == 0:
                return

            scores = scores.detach().cpu()
            labels = labels.detach().cpu().long().reshape(-1)
            if scores.dim() == 1:
                scores = scores.unsqueeze(0)
            if scores.size(0) != labels.numel():
                n = min(scores.size(0), labels.numel())
                scores = scores[:n]
                labels = labels[:n]
            if labels.numel() == 0:
                return

            if self.max_instances > 0 and labels.numel() > self.max_instances:
                perm = torch.randperm(labels.numel())[:self.max_instances]
                scores = scores[perm]
                labels = labels[perm]

            visualizer = runner.visualizer
            if not hasattr(visualizer, 'draw_multiclass_instance_analysis'):
                return

            class_names = None
            dataset_meta = getattr(visualizer, 'dataset_meta', None)
            if isinstance(dataset_meta, dict):
                class_names = dataset_meta.get('classes', None)

            vis_img = visualizer.draw_multiclass_instance_analysis(
                instance_scores=scores,
                instance_labels=labels,
                class_names=class_names,
                topk=self.topk,
                background_label=self.background_label,
                title=f'MIL multi-class analysis | iter {runner.iter}')

            visualizer.add_image(
                'mil_debug/multiclass_instance_analysis',
                vis_img,
                step=runner.iter)

            if self.out_dir is not None:
                out_path = osp.join(self.out_dir, f'iter_{runner.iter}.png')
                mmcv.imwrite(vis_img[..., ::-1], out_path)
        finally:
            if hasattr(roi_head, '_last_multiclass_debug'):
                del roi_head._last_multiclass_debug
