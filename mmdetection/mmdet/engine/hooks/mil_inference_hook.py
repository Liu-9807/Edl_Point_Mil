import os.path as osp
import torch
import mmcv
import numpy as np
from mmengine.hooks import Hook
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

    def before_run(self, runner):
        if self.out_dir:
            self._vis_dir = self.out_dir
        else:
            self._vis_dir = osp.join(runner.work_dir, runner.timestamp, 'vis_data/val_inference')
        
        # 仅主进程创建目录
        if runner.rank == 0:
            os.makedirs(self._vis_dir, exist_ok=True)

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
            vis_img = visualizer.draw_mil_inference(
                img,
                pred_instances,
                gt_instances
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