import torch
import numpy as np
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import os

@HOOKS.register_module()
class MILEpochScatterHook(Hook):
    """
    在每个 Epoch 结束时，收集全量的实例级 Evidence 并绘制 2D 分布散点图。
    """
    def __init__(self, interval=1, out_dir=None):
        self.interval = interval
        self._out_dir = out_dir
        self.out_dir = None

    def before_run(self, runner):
        """在训练开始前设置输出目录"""
        if self._out_dir is None:
            self.out_dir = os.path.join(runner.work_dir, runner.timestamp, 'vis_data/epoch_scatter')
        else:
            self.out_dir = self._out_dir
        
        os.makedirs(self.out_dir, exist_ok=True)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """
        确保数据被收集。这里作为一个 safeguard，或者真正的收集逻辑也可以放在这里。
        但为了遵循你的代码结构，收集逻辑建议放在 MILRoIHead.forward_train 中。
        """
        pass

    def after_train_epoch(self, runner):
        """Epoch 结束时的绘图逻辑"""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        roi_head = getattr(model, 'roi_head', None)
        if roi_head is not None:
            # Keep EDL annealing epoch in sync with runner progress.
            roi_head.current_epoch = runner.epoch + 1

        # 检查是否满足 interval
        if not self.every_n_epochs(runner, self.interval):
            return

        if not roi_head:
            return

        # 1. 获取并拼接数据
        # 只要你在 MILRoIHead 中维护了 self.epoch_logits_ins 列表
        if not hasattr(roi_head, 'epoch_logits_ins') or len(roi_head.epoch_logits_ins) == 0:
            return

        try:
            # 拼接 Tensor [Total_N, C]
            all_scores = torch.cat(roi_head.epoch_logits_ins, dim=0).cpu().numpy()
            all_labels = torch.cat(roi_head.epoch_ins_labels, dim=0).cpu().numpy()
            

            # 确保不小于0 (虽然理论上 alpha >= 1)
            all_evidence = np.maximum(0, all_scores - 1.0)

            # 2. 调用可视化
            visualizer = runner.visualizer
            if hasattr(visualizer, 'draw_evidence_scatter'):
                scatter_img = visualizer.draw_evidence_scatter(
                    all_evidence,
                    all_labels,
                    epoch_num=runner.epoch
                )
                
                # 3. 记录
                visualizer.add_image('mil_epoch/evidence_scatter', scatter_img, step=runner.epoch)
                if self.out_dir is not None:
                    import mmcv
                    out_path = os.path.join(self.out_dir, f'evidence_scatter_epoch_{runner.epoch}.png')
                    mmcv.imwrite(scatter_img, out_path)
        except Exception as e:
            runner.logger.warning(f"Failed to draw epoch scatter: {e}")
        
        finally:
            # 4. 极为重要：清空缓存，防止显存/内存爆炸
            roi_head.epoch_logits_ins = []
            roi_head.epoch_ins_labels = []