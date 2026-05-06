import os.path as osp
import torch
import mmcv
import numpy as np
from mmengine.hooks import Hook
from mmengine.structures import InstanceData
from mmdet.registry import HOOKS
import os

@HOOKS.register_module()
class MILInferenceVisHook(Hook):
    """
    在 Validation 过程中可视化 'Point -> Box' 的推理结果。
    """
    def __init__(self, interval=50, out_dir=None, show=False):
        self.interval = interval
        self.out_dir = out_dir
        self.show = show
        self._vis_dir = None
        self._gt_cache = None

    def before_run(self, runner):
        if self.out_dir:
            self._vis_dir = self.out_dir
        else:
            self._vis_dir = osp.join(runner.work_dir, runner.timestamp, 'vis_data/val_inference')
        
        # 仅主进程创建目录
        if runner.rank == 0:
            os.makedirs(self._vis_dir, exist_ok=True)

    def before_test(self, runner):
        if self.out_dir:
            self._vis_dir = self.out_dir
        else:
            self._vis_dir = osp.join(runner.work_dir, runner.timestamp, 'vis_data/test_inference')

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
            bboxes.append(ins.get('bbox', [0, 0, 0, 0]))
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

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """
        Validation 阶段的推理可视化。
        使用统一的 draw_mil_inference_result 方法，mode='point_box'。
        
        Args:
            data_batch: dict containing 'data_samples' (List[DetDataSample]) with GT info
            outputs: List[DetDataSample] containing pred_instances
        """
        if (batch_idx + 1) % self.interval != 0:
            return

        visualizer = runner.visualizer
        if not hasattr(visualizer, 'draw_mil_inference_result'):
            return

        gt_samples = data_batch.get('data_samples', [])
        pred_samples = outputs if outputs is not None else []
        
        for i, (gt_sample, pred_sample) in enumerate(zip(gt_samples, pred_samples)):
            # 1. 读取图像
            img_path = getattr(gt_sample, 'img_path', None)
            if not img_path or not osp.exists(img_path):
                continue

            img = mmcv.imread(img_path, channel_order='rgb')

            # 2. 提取 GT 和 Pred
            gt_instances = getattr(gt_sample, 'gt_instances', None)
            pred_instances = getattr(pred_sample, 'pred_instances', None)

            # 3. 统一的可视化调用：mode='point_box' 用于 Val
            try:
                vis_img = visualizer.draw_mil_inference_result(
                    image=img,
                    pred_instances=pred_instances,
                    gt_instances=gt_instances,
                    data_sample=gt_sample,
                    mode='point_box'  # 显式使用 Val 模式
                )
            except Exception as e:
                runner.logger.warning(f"Failed to visualize val iter {batch_idx} sample {i}: {e}")
                continue

            # 4. 保存结果
            name = f"val_iter{batch_idx}_s{i}"
            
            # 记录到 Tensorboard
            visualizer.add_image(f'val_vis/{name}', vis_img, step=runner.iter)

            # 本地保存
            if self._vis_dir:
                out_file = osp.join(self._vis_dir, f"{name}.jpg")
                mmcv.imwrite(vis_img[..., ::-1], out_file)  # RGB -> BGR

            # 每个 iter 仅保存第一张图像，控制输出量
            break

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """
        Test 阶段的推理可视化。
        使用统一的 draw_mil_inference_result 方法，mode='box_box' 或 'point_box'。
        
        Args:
            outputs: List[DetDataSample] containing pred_instances
        """
        if (batch_idx + 1) % self.interval != 0:
            return

        visualizer = runner.visualizer
        if visualizer is None or outputs is None:
            return

        if not hasattr(visualizer, 'draw_mil_inference_result'):
            return

        for i, pred_sample in enumerate(outputs):
            img_path = getattr(pred_sample, 'img_path', None)
            if not img_path or not osp.exists(img_path):
                continue

            img = mmcv.imread(img_path, channel_order='rgb')
            pred_sample_cpu = pred_sample.cpu()

            # 尝试从预测样本或缓存中获取 GT
            gt_instances = None
            has_gt_in_pred = hasattr(pred_sample_cpu, 'gt_instances') and len(pred_sample_cpu.gt_instances) > 0
            
            if has_gt_in_pred:
                gt_instances = pred_sample_cpu.gt_instances
            elif self._gt_cache is not None:
                # 从缓存恢复 GT
                cached = self._gt_cache.get(osp.normpath(img_path), [])
                if cached:
                    gt_instances = self._build_gt_instances(cached)

            # 选择可视化模式
            # 如果有 GT points，使用 'point_box' 模式；否则使用 'box_box' 模式
            mode = 'point_box' if (gt_instances is not None and hasattr(gt_instances, 'points') and len(gt_instances.points) > 0) else 'box_box'

            # 统一的可视化调用
            try:
                vis_img = visualizer.draw_mil_inference_result(
                    image=img,
                    pred_instances=pred_sample_cpu.pred_instances,
                    gt_instances=gt_instances,
                    data_sample=pred_sample_cpu,
                    mode=mode  # 根据 GT 可用性自动选择模式
                )
            except Exception as e:
                runner.logger.warning(f"Failed to visualize test iter {batch_idx} sample {i}: {e}")
                continue

            # 保存结果
            name = f'test_iter{batch_idx}_s{i}'
            
            if self._vis_dir:
                out_file = osp.join(self._vis_dir, f'{name}.jpg')
                mmcv.imwrite(vis_img[..., ::-1], out_file)  # RGB -> BGR
            
            # 记录到可视化后端
            visualizer.add_image(f'test_vis/{name}', vis_img, step=runner.iter)

            # 每个 iter 仅保存第一张图像
            break