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
        Args:
            data_batch: dict containing 'data_samples' (List[DetDataSample]) which has GT info.
            outputs: List[DetDataSample] containing pred_instances.
        """
        # 控制频率
        if (batch_idx + 1) % self.interval != 0:
            return

        visualizer = runner.visualizer
        if not hasattr(visualizer, 'draw_mil_inference'):
            return

        # 获取输入和输出列表
        # 注意: data_batch 是经过 DataPreprocessor 的还是原始的？
        # 在 MMEngine 中，Hook 接收到的 data_batch 通常是 DataLoader 出来的原始内容，或者处理后的。
        # 安全起见，从 data_batch['data_samples'] 获取 GT，从 outputs 获取 Pred
        
        gt_samples = data_batch['data_samples']
        pred_samples = outputs
        
        for i, (gt_sample, pred_sample) in enumerate(zip(gt_samples, pred_samples)):
            
            # 1. 准备图片
            img_path = gt_sample.img_path
            if img_path and osp.exists(img_path):
                img = mmcv.imread(img_path, channel_order='rgb')
            else:
                continue

            # 2. 提取信息
            # GT Points
            gt_instances = None
            if hasattr(gt_sample, 'gt_instances'):
                gt_instances = gt_sample.gt_instances
            
            # Predict Boxes
            pred_instances = pred_sample.pred_instances

            # 3. 绘制
            # 新增 data_sample=gt_sample 作为参数传递
            vis_img = visualizer.draw_mil_inference(
                image=img,
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                data_sample=gt_sample  # <--- 在这里传入包含 scale_factor 的 data_sample
            )

            # 4. 保存与记录
            # 文件名：val_batch0_sample0.jpg
            name = f"val_iter{batch_idx}_s{i}"
            
            # 添加到 Tensorboard / WandB
            visualizer.add_image(f'val_vis/{name}', vis_img, step=runner.iter)

            # 保存到本地
            if self._vis_dir:
                out_file = osp.join(self._vis_dir, f"{name}.jpg")
                mmcv.imwrite(vis_img[..., ::-1], out_file) # RGB -> BGR
            
            # 为了避免生成太多图，如果是 batch 处理，每个 iter 只画一张图就 break
            break

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Draw GT and prediction overlay during test iterations."""
        if (batch_idx + 1) % self.interval != 0:
            return

        visualizer = runner.visualizer
        if visualizer is None:
            return

        if outputs is None:
            return

        for i, pred_sample in enumerate(outputs):
            img_path = getattr(pred_sample, 'img_path', None)
            if not img_path or not osp.exists(img_path):
                continue

            img = mmcv.imread(img_path, channel_order='rgb')
            draw_sample = pred_sample.cpu()

            # If test pipeline doesn't carry GT, recover it from dataset cache.
            has_gt = hasattr(draw_sample, 'gt_instances') and len(draw_sample.gt_instances) > 0
            if (not has_gt) and self._gt_cache is not None:
                cached = self._gt_cache.get(osp.normpath(img_path), [])
                draw_sample.gt_instances = self._build_gt_instances(cached)

            out_file = None
            if self._vis_dir:
                name = f'test_iter{batch_idx}_s{i}'
                out_file = osp.join(self._vis_dir, f'{name}.jpg')
            else:
                name = 'test_overlay'

            visualizer.add_datasample(
                name=name,
                image=img,
                data_sample=draw_sample,
                draw_gt=True,
                draw_pred=True,
                show=self.show,
                pred_score_thr=0.0,
                out_file=out_file,
                step=runner.iter)

            # Keep one image per iter to control output volume.
            break