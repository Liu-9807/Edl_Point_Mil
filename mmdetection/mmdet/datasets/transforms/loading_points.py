import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms import PackDetInputs
from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadPointAnnotations(BaseTransform):
    """自定义转换：从 instances 中提取 point 字段"""
    def transform(self, results: dict) -> dict:
        instances = results.get('instances', [])
        gt_points = []
        for instance in instances:
            # 获取 point，默认为 [0, 0] 以防报错
            gt_points.append(instance.get('point', [0.0, 0.0]))
        
        # 转换为 numpy 数组并存入 results，供后续 pipeline 使用
        results['gt_points'] = np.array(gt_points, dtype=np.float32).reshape(-1, 2)
        return results

@TRANSFORMS.register_module()
class PackPointDetInputs(PackDetInputs):
    """自定义打包：将 gt_points 打包进 data_sample.gt_instances"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 关键步骤：添加映射关系
        # 将 pipeline 里的 'gt_points' 字段映射到 InstanceData 的 'points' 属性
        self.mapping_table['gt_points'] = 'points'

@TRANSFORMS.register_module()
class ResizePoints(BaseTransform):
    """
    专门跟随 Standard Resize/Flip 操作去调整 gt_points 的坐标。
    必须放在 Resize, RandomFlip, Pad 之后。
    """
    def transform(self, results: dict) -> dict:
        if 'gt_points' not in results:
            return results
            
        points = results['gt_points']
        
        # 1. 处理 Resize (由于 Resize 记录了 scale_factor)
        scale_factor = results.get('scale_factor', None)
        if scale_factor is not None:
            # scale_factor 可能是 (w_scale, h_scale) 或者 (w, h, w, h)
            # points 是 (N, 2) -> (x, y)
            points[:, 0] *= scale_factor[0]
            points[:, 1] *= scale_factor[1]

        # 2. 处理 RandomFlip
        if results.get('flip', False):
            img_shape = results['img_shape'] # (h, w) after resize
            h, w = img_shape[:2]
            direction = results.get('flip_direction', 'horizontal')
            
            if direction == 'horizontal':
                # x' = w - 1 - x (或者 w - x，取决于具体定义，通常 w - x 足够近似)
                points[:, 0] = w - points[:, 0]
            elif direction == 'vertical':
                points[:, 1] = h - points[:, 1]

        # 更新回 results
        results['gt_points'] = points
        return results