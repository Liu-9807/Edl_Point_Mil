import os
import os.path as osp
import torch
import numpy as np
import mmcv
from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class MILEvidenceHook(Hook):
    """
    可视化 EDL Head 推理后的实例图像块与证据值的对应关系。
    """
    def __init__(self, interval=100, out_dir=None, max_instances=10):
        self.interval = interval
        self._out_dir = out_dir
        self.out_dir = None
        self.max_instances = max_instances

    def before_run(self, runner):
        """在训练开始前设置输出目录"""
        if self._out_dir is None:
            self.out_dir = os.path.join(runner.work_dir, runner.timestamp,'vis_data/instance_evidence_vis')
        else:
            self.out_dir = self._out_dir
        
        os.makedirs(self.out_dir, exist_ok=True)

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        if self.every_n_train_iters(runner, self.interval):
            if hasattr(runner.model, 'module'):
                model = runner.model.module
            else:
                model = runner.model
            
            # 定位到 EDLHead
            # 假设结构: Model -> RoIHead -> BBoxHead (EDLHead)
            if hasattr(model, 'roi_head') and hasattr(model.roi_head, 'bbox_head'):
                model.roi_head.bbox_head.save_debug_info = True

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self.every_n_train_iters(runner, self.interval):
            model = runner.model
            if hasattr(model, 'module'):
                model = model.module
            
            bbox_head = model.roi_head.bbox_head
            
            # 1. 获取中间数据
            if not hasattr(bbox_head, '_last_debug_data'):
                return
            
            debug_data = bbox_head._last_debug_data
            
            rois = debug_data['rois'] # [N, 5] (batch_ind, x1, y1, x2, y2)
            ins_output = debug_data['ins_output'] # [N, NumClasses]
            
            # 2. 获取图像元数据 (用于坐标反变换)
            # data_batch 是一个 dict，通常包含 'data_samples'
            # data_samples 是一个 list of DetDataSample
            data_samples = data_batch['data_samples']
            
            visualizer = runner.visualizer
            if not hasattr(visualizer, 'draw_instance_evidence'):
                 print("Visualizer does not support draw_instance_evidence")
                 bbox_head.save_debug_info = False
                 return

            # 3. 按 Bag (Image) 循环处理
            # 假设只画 Batch 中的第一张图来节省时间
            batch_id = 0
            img_meta = data_samples[batch_id].metainfo
            img_path = img_meta.get('img_path', None)
            
            if img_path and osp.exists(img_path):
                # 读取原图
                raw_image = mmcv.imread(img_path, channel_order='rgb')
                
                # 获取该 Batch 的 RoI
                mask = rois[:, 0] == batch_id
                bag_rois = rois[mask, 1:] # [K, 4]
                bag_scores = ins_output[mask] # [K, C]
                
                # --- 坐标反变换关键逻辑 ---
                # RoI 是基于 data_batch['inputs'] 的，也就是经过 Resize/Pad 后的
                # img_meta['scale_factor'] 通常是 (w_scale, h_scale)
                # 原图坐标 = 当前坐标 / scale
                
                scale_factor = img_meta.get('scale_factor', None)
                if scale_factor is not None:
                     # scale_factor 是 (w_scale, h_scale)
                     # bag_rois: x1, y1, x2, y2
                     bag_rois[:, 0] /= scale_factor[0]
                     bag_rois[:, 1] /= scale_factor[1]
                     bag_rois[:, 2] /= scale_factor[0]
                     bag_rois[:, 3] /= scale_factor[1]
                
                # 4. 调用 Visualizer
                res_img = visualizer.draw_instance_evidence(
                    raw_image,
                    bag_rois,
                    bag_scores,
                    max_instances=self.max_instances
                )
                
                # 5. 写入 Tensorboard / 本地
                visualizer.add_image('mil_debug/instance_evidence', res_img, step=runner.iter)
                
                if self.out_dir:
                    save_path = osp.join(self.out_dir, f'iter_{runner.iter}_ev_vis.png')
                    mmcv.imwrite(res_img[..., ::-1], save_path) # RGB -> BGR

            # 清理
            bbox_head.save_debug_info = False
            del bbox_head._last_debug_data