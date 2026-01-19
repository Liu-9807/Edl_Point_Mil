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