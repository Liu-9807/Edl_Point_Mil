import numpy as np
from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class PointMilMetric(BaseMetric):
    """Custom metric for wind turbine detection with point annotations."""
    
    rule = 'greater'  # 越大越好
    
    def __init__(self, 
                 iou_thr: float = 0.5,
                 collect_device: str = 'cpu',
                 prefix: str = 'point_mil_',
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thr = iou_thr
    
    def process(self, data_batch, data_samples):
        """Process one batch of data samples."""
        # for data_sample in data_samples:
        #     # 提取预测结果和 GT
        #     pred_instances = data_sample.pred_instances
        #     gt_instances = data_sample.gt_instances
            
        #     # 简化实现：计算 IoU
        #     # 实际应根据你的需求补充完整的评估逻辑
        #     self.results.append({
        #         'pred_bboxes': pred_instances.bboxes.cpu().numpy(),
        #         'pred_scores': pred_instances.scores.cpu().numpy(),
        #         'gt_bboxes': gt_instances.bboxes.cpu().numpy(),
        #     })
        pass
    
    def compute_metrics(self, results: list) -> dict:
        """Compute metrics from processed results."""
        # 简单的准确度计算（可根据需求扩展）
        tp, fp, fn = 0, 0, 0
        
        for result in results:
            pred_bboxes = result['pred_bboxes']
            gt_bboxes = result['gt_bboxes']
            
            matched = set()
            for pred_bbox in pred_bboxes:
                for gt_idx, gt_bbox in enumerate(gt_bboxes):
                    if gt_idx not in matched:
                        iou = self._compute_iou(pred_bbox, gt_bbox)
                        if iou > self.iou_thr:
                            tp += 1
                            matched.add(gt_idx)
                            break
                else:
                    fp += 1
            
            fn += len(gt_bboxes) - len(matched)
        
        # 计算 precision, recall, f1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            f'{self.prefix}precision': precision,
            f'{self.prefix}recall': recall,
            f'{self.prefix}f1': f1,
        }
    
    @staticmethod
    def _compute_iou(bbox1, bbox2):
        """计算两个 bbox 的 IoU (格式: [x1, y1, x2, y2])."""
        x1_min, y1_min, x1_max, y1_max = bbox1[:4]
        x2_min, y2_min, x2_max, y2_max = bbox2[:4]
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0