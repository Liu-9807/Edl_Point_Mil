import numpy as np
from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class PointMilMetric(BaseMetric):
    """Custom metric for wind turbine detection with point annotations."""
    
    rule = 'greater'  # 越大越好
    
    def __init__(self, 
                 iou_thr: float = 0.5,
                 ap_iou_thrs: tuple = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
                 use_eval_gt_from_meta: bool = True,
                 collect_device: str = 'cpu',
                 prefix: str = 'point_mil_',
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thr = iou_thr
        self.ap_iou_thrs = tuple(float(v) for v in ap_iou_thrs)
        self.use_eval_gt_from_meta = use_eval_gt_from_meta
    
    def process(self, data_batch, data_samples):
        """Process one batch of data samples."""
        batch_gt_samples = None
        if isinstance(data_batch, dict):
            batch_gt_samples = data_batch.get('data_samples', None)

        for idx, data_sample in enumerate(data_samples):
            if isinstance(data_sample, dict):
                pred_instances = data_sample.get('pred_instances', None)
                gt_instances = data_sample.get('gt_instances', None)
            else:
                pred_instances = getattr(data_sample, 'pred_instances', None)
                gt_instances = getattr(data_sample, 'gt_instances', None)

            if gt_instances is None and batch_gt_samples is not None and idx < len(batch_gt_samples):
                batch_gt = batch_gt_samples[idx]
                if isinstance(batch_gt, dict):
                    gt_instances = batch_gt.get('gt_instances', None)
                else:
                    gt_instances = getattr(batch_gt, 'gt_instances', None)

            if pred_instances is None or gt_instances is None:
                continue

            pred_bboxes = pred_instances['bboxes'] if isinstance(pred_instances, dict) else pred_instances.bboxes
            gt_bboxes = gt_instances['bboxes'] if isinstance(gt_instances, dict) else gt_instances.bboxes

            # Prefer dataset-provided eval GT (e.g. YOLO box labels) from metainfo.
            if self.use_eval_gt_from_meta:
                eval_gt_bboxes = None
                if isinstance(data_sample, dict):
                    metainfo = data_sample.get('metainfo', {})
                    eval_gt_bboxes = metainfo.get('eval_gt_bboxes', None)
                else:
                    metainfo = getattr(data_sample, 'metainfo', {})
                    if isinstance(metainfo, dict):
                        eval_gt_bboxes = metainfo.get('eval_gt_bboxes', None)

                if eval_gt_bboxes is not None:
                    gt_bboxes = eval_gt_bboxes

            if hasattr(pred_bboxes, 'tensor'):
                pred_bboxes = pred_bboxes.tensor
            if hasattr(gt_bboxes, 'tensor'):
                gt_bboxes = gt_bboxes.tensor

            if hasattr(gt_bboxes, 'detach'):
                gt_bboxes_np = gt_bboxes.detach().cpu().numpy()
            else:
                gt_bboxes_np = np.asarray(gt_bboxes, dtype=np.float32).reshape(-1, 4)

            pred_scores = pred_instances.get('scores', None) if isinstance(pred_instances, dict) else getattr(pred_instances, 'scores', None)
            if pred_scores is None:
                pred_scores = np.ones((pred_bboxes.shape[0], ), dtype=np.float32)
            else:
                pred_scores = pred_scores.detach().cpu().numpy()

            pred_labels = pred_instances.get('labels', None) if isinstance(pred_instances, dict) else getattr(pred_instances, 'labels', None)
            if pred_labels is None:
                pred_labels = np.zeros((pred_bboxes.shape[0], ), dtype=np.int64)
            elif hasattr(pred_labels, 'detach'):
                pred_labels = pred_labels.detach().cpu().numpy()
            else:
                pred_labels = np.asarray(pred_labels, dtype=np.int64)

            gt_labels = gt_instances.get('labels', None) if isinstance(gt_instances, dict) else getattr(gt_instances, 'labels', None)
            if gt_labels is None:
                gt_labels = np.zeros((gt_bboxes_np.shape[0], ), dtype=np.int64)
            elif hasattr(gt_labels, 'detach'):
                gt_labels = gt_labels.detach().cpu().numpy()
            else:
                gt_labels = np.asarray(gt_labels, dtype=np.int64)

            self.results.append({
                'pred_bboxes': pred_bboxes.detach().cpu().numpy(),
                'pred_scores': pred_scores,
                'pred_labels': pred_labels,
                'gt_bboxes': gt_bboxes_np,
                'gt_labels': gt_labels,
            })
    
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

        ap_by_thr = {}
        for thr in self.ap_iou_thrs:
            ap_by_thr[thr] = self._compute_ap_at_thr(results, thr)

        map_5095 = float(np.mean(list(ap_by_thr.values()))) if ap_by_thr else 0.0
        ap50 = float(ap_by_thr.get(0.5, 0.0))
        
        return {
            f'{self.prefix}precision': precision,
            f'{self.prefix}recall': recall,
            f'{self.prefix}f1': f1,
            f'{self.prefix}AP50': ap50,
            f'{self.prefix}mAP': map_5095,
        }

    def _compute_ap_at_thr(self, results: list, iou_thr: float) -> float:
        """Compute class-agnostic AP at a single IoU threshold."""
        gt_count = 0
        preds = []

        for img_idx, result in enumerate(results):
            gt_bboxes = np.asarray(result['gt_bboxes'], dtype=np.float32).reshape(-1, 4)
            pred_bboxes = np.asarray(result['pred_bboxes'], dtype=np.float32).reshape(-1, 4)
            pred_scores = np.asarray(result['pred_scores'], dtype=np.float32).reshape(-1)

            gt_count += gt_bboxes.shape[0]
            for bbox, score in zip(pred_bboxes, pred_scores):
                preds.append((float(score), img_idx, bbox))

        if gt_count == 0:
            return 0.0
        if len(preds) == 0:
            return 0.0

        preds.sort(key=lambda x: x[0], reverse=True)
        matched = {
            img_idx: np.zeros(np.asarray(results[img_idx]['gt_bboxes']).reshape(-1, 4).shape[0], dtype=bool)
            for img_idx in range(len(results))
        }

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for i, (_, img_idx, pred_bbox) in enumerate(preds):
            gt_bboxes = np.asarray(results[img_idx]['gt_bboxes'], dtype=np.float32).reshape(-1, 4)
            if gt_bboxes.shape[0] == 0:
                fp[i] = 1
                continue

            ious = self._compute_iou_vectorized(pred_bbox, gt_bboxes)
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])

            if best_iou >= iou_thr and not matched[img_idx][best_idx]:
                tp[i] = 1
                matched[img_idx][best_idx] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(gt_count, 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

        return self._area_under_pr_curve(recalls, precisions)

    @staticmethod
    def _area_under_pr_curve(recalls: np.ndarray, precisions: np.ndarray) -> float:
        """Compute AP by integrating precision-recall curve with monotonic precision."""
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        return float(ap)

    @staticmethod
    def _compute_iou_vectorized(bbox: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """Compute IoU between one bbox and N bboxes."""
        x1 = np.maximum(bbox[0], bboxes[:, 0])
        y1 = np.maximum(bbox[1], bboxes[:, 1])
        x2 = np.minimum(bbox[2], bboxes[:, 2])
        y2 = np.minimum(bbox[3], bboxes[:, 3])

        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        inter = inter_w * inter_h

        area1 = np.maximum(0.0, bbox[2] - bbox[0]) * np.maximum(0.0, bbox[3] - bbox[1])
        area2 = np.maximum(0.0, bboxes[:, 2] - bboxes[:, 0]) * np.maximum(0.0, bboxes[:, 3] - bboxes[:, 1])
        union = area1 + area2 - inter

        iou = np.zeros_like(union, dtype=np.float32)
        valid = union > 0
        iou[valid] = inter[valid] / union[valid]
        return iou
    
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