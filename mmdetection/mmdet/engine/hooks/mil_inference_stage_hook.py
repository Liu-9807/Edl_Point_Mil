import os
import os.path as osp

import mmcv
import torch
from mmengine.structures import InstanceData
from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class MILInferenceStageVisHook(Hook):
    """Visualize inference stages: points, proposals, refined, final."""

    def __init__(self,
                 interval=50,
                 out_dir=None,
                 show=False,
                 max_proposals=200,
                 max_refined=200):
        self.interval = interval
        self.out_dir = out_dir
        self.show = show
        self.max_proposals = max_proposals
        self.max_refined = max_refined
        self._vis_dir = None
        self._gt_cache = None

    def before_run(self, runner):
        if self.out_dir:
            self._vis_dir = self.out_dir
        else:
            self._vis_dir = osp.join(runner.work_dir, runner.timestamp, 'vis_infer')

        if runner.rank == 0:
            os.makedirs(self._vis_dir, exist_ok=True)

    def before_test(self, runner):
        if self.out_dir:
            self._vis_dir = self.out_dir
        else:
            self._vis_dir = osp.join(runner.work_dir, runner.timestamp, 'vis_infer')

        if runner.rank == 0:
            os.makedirs(self._vis_dir, exist_ok=True)

        self._build_gt_cache(runner)

    def _build_gt_cache(self, runner):
        """Build a fast lookup from image path to GT annotations for test overlay."""
        self._gt_cache = {}
        dataset = getattr(getattr(runner, 'test_dataloader', None), 'dataset', None)
        if dataset is None or not hasattr(dataset, 'data_list'):
            return

        for data_info in dataset.data_list:
            img_path = data_info.get('img_path', None)
            instances = data_info.get('instances', [])
            if img_path is None:
                continue
            self._gt_cache[osp.normpath(img_path)] = instances

    @staticmethod
    def _build_gt_instances(instances):
        """Convert list-of-dict instance annotations into InstanceData."""
        gt_instances = InstanceData()

        if not instances:
            gt_instances.bboxes = torch.zeros((0, 4), dtype=torch.float32)
            gt_instances.labels = torch.zeros((0, ), dtype=torch.long)
            gt_instances.points = torch.zeros((0, 2), dtype=torch.float32)
            return gt_instances

        bboxes = []
        labels = []
        points = []
        for ins in instances:
            bbox = ins.get('bbox', [0, 0, 0, 0])
            # Ensure bbox is a list/array with exactly 4 elements
            if isinstance(bbox, (list, tuple)):
                bbox = list(bbox)
            elif hasattr(bbox, 'tolist'):  # numpy array
                bbox = bbox.tolist()
            else:
                bbox = [0, 0, 0, 0]
            
            if len(bbox) == 4:
                bboxes.append(bbox)
            else:
                # Skip malformed bboxes
                bboxes.append([0, 0, 0, 0])
            
            labels.append(ins.get('bbox_label', 0))
            if 'point' in ins and ins['point'] is not None:
                points.append(ins['point'])

        gt_instances.bboxes = torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
        gt_instances.labels = torch.tensor(labels, dtype=torch.long)
        if len(points) > 0:
            gt_instances.points = torch.tensor(points, dtype=torch.float32).reshape(-1, 2)
        else:
            gt_instances.points = torch.zeros((len(bboxes), 2), dtype=torch.float32)
        return gt_instances

    def before_val_iter(self, runner, batch_idx, data_batch=None):
        if not self.every_n_inner_iters(batch_idx, self.interval):
            return
        self._enable_debug_cache(runner)

    def before_test_iter(self, runner, batch_idx, data_batch=None):
        if not self.every_n_inner_iters(batch_idx, self.interval):
            return
        self._enable_debug_cache(runner)

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self.every_n_inner_iters(batch_idx, self.interval):
            return
        self._draw_batch(runner, batch_idx, data_batch, outputs, stage='val')
        self._disable_debug_cache(runner)

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self.every_n_inner_iters(batch_idx, self.interval):
            return
        self._draw_batch(runner, batch_idx, data_batch, outputs, stage='test')
        self._disable_debug_cache(runner)

    def _enable_debug_cache(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        roi_head = getattr(model, 'roi_head', None)
        if roi_head is not None:
            roi_head.debug_infer_vis = True

    def _disable_debug_cache(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        roi_head = getattr(model, 'roi_head', None)
        if roi_head is not None:
            roi_head.debug_infer_vis = False

    def _draw_batch(self, runner, batch_idx, data_batch, outputs, stage='test'):
        visualizer = runner.visualizer
        if visualizer is None or outputs is None:
            return
        if not hasattr(visualizer, 'draw_mil_inference_stages'):
            return

        data_samples = data_batch.get('data_samples', []) if data_batch else []

        for i, pred_sample in enumerate(outputs):
            debug = getattr(pred_sample, 'mil_debug', None)
            if debug is None:
                continue

            gt_sample = data_samples[i] if i < len(data_samples) else None
            img_path = getattr(gt_sample, 'img_path', None) if gt_sample is not None else getattr(pred_sample, 'img_path', None)
            if not img_path or not osp.exists(img_path):
                continue

            gt_instances = getattr(gt_sample, 'gt_instances', None) if gt_sample is not None else None
            if gt_instances is None and self._gt_cache is not None:
                cached = self._gt_cache.get(osp.normpath(img_path), [])
                if cached:
                    gt_instances = self._build_gt_instances(cached)

            img = mmcv.imread(img_path, channel_order='rgb')
            vis_img = visualizer.draw_mil_inference_stages(
                image=img,
                gt_instances=gt_instances,
                points=debug.get('points', None),
                proposals=debug.get('proposals', None),
                refined_bboxes=debug.get('refined_bboxes', None),
                final_bboxes=debug.get('final_bboxes', None),
                data_sample=gt_sample if gt_sample is not None else pred_sample,
                max_proposals=self.max_proposals,
                max_refined=self.max_refined)

            name = f'{stage}_iter{batch_idx}_s{i}'

            if self._vis_dir and runner.rank == 0:
                out_file = osp.join(self._vis_dir, f'{name}.jpg')
                mmcv.imwrite(vis_img[..., ::-1], out_file)

            if runner.rank == 0:
                visualizer.add_image(f'{stage}_vis/{name}', vis_img, step=runner.iter)

            break

    @staticmethod
    def every_n_inner_iters(batch_idx, interval):
        if interval <= 0:
            return False
        return (batch_idx + 1) % interval == 0
