from typing import Any, Dict, Optional, Sequence

import torch
from mmengine.evaluator import BaseMetric

from mmdet.registry import METRICS


@METRICS.register_module()
class MILTrainingMetric(BaseMetric):
    """Monitor MIL bag/instance classification during training.

    The metric is background-aware by default, where ``background_label`` is
    treated as negative and every label greater than it is foreground. This
    keeps the original binary PointMIL task valid while also supporting
    multi-class instance labels such as DOTA's ``0 + 18`` label space.
    """

    default_prefix = 'train_mil'
    rule = 'greater'
    init_value_map = {'acc_bag': 0, 'acc_instance': 0}

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = 'train_mil',
                 num_classes: Optional[int] = None,
                 background_label: int = 0,
                 topk: Sequence[int] = (1, 5),
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes = None if num_classes is None else int(num_classes)
        self.background_label = int(background_label)
        self.topk = tuple(sorted({int(k) for k in topk if int(k) > 0}))

    @staticmethod
    def _get_field(sample: Any, name: str):
        if isinstance(sample, dict):
            return sample.get(name, None)
        if hasattr(sample, name):
            return getattr(sample, name)
        if hasattr(sample, 'get'):
            try:
                return sample.get(name, None)
            except TypeError:
                return None
        return None

    @staticmethod
    def _select_score(score):
        if isinstance(score, (tuple, list)):
            # EDLHead returns (init_alpha, enhanced_alpha, ...); use enhanced
            # alpha when present, otherwise fall back to the last tensor.
            if len(score) > 1 and isinstance(score[1], torch.Tensor):
                return score[1]
            for item in reversed(score):
                if isinstance(item, torch.Tensor):
                    return item
            return None
        return score

    @staticmethod
    def _to_cpu_tensor(value):
        if value is None:
            return None
        value = MILTrainingMetric._select_score(value)
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        return value.detach().cpu()

    @staticmethod
    def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
        if scores.ndim == 1:
            return scores.unsqueeze(0)
        return scores

    @staticmethod
    def _normalize_targets(targets: torch.Tensor) -> torch.Tensor:
        return targets.reshape(-1).long()

    def process(self, data_batch, data_samples) -> None:
        """Collect bag/instance scores from a batch."""
        for sample in data_samples:
            result: Dict[str, torch.Tensor] = {}

            bag_score = self._to_cpu_tensor(self._get_field(sample, 'bag_score'))
            bag_label = self._to_cpu_tensor(self._get_field(sample, 'bag_label'))
            if bag_score is not None and bag_label is not None:
                result['bag_score'] = self._normalize_scores(bag_score)
                result['bag_label'] = self._normalize_targets(bag_label)

            ins_score = self._to_cpu_tensor(self._get_field(sample, 'ins_score'))
            ins_label = self._to_cpu_tensor(self._get_field(sample, 'ins_label'))
            if ins_score is not None and ins_label is not None:
                result['ins_score'] = self._normalize_scores(ins_score)
                result['ins_label'] = self._normalize_targets(ins_label)

            if result:
                self.results.append(result)

    def _infer_num_classes(self, *scores) -> int:
        if self.num_classes is not None:
            return self.num_classes
        max_classes = 0
        for score in scores:
            if score is not None and score.numel() > 0:
                max_classes = max(max_classes, int(score.shape[1]))
        return max_classes

    def _classification_metrics(self, scores: torch.Tensor,
                                targets: torch.Tensor,
                                prefix: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if scores.numel() == 0 or targets.numel() == 0:
            return metrics

        scores = self._normalize_scores(scores)
        targets = self._normalize_targets(targets)
        if scores.size(0) != targets.numel():
            n = min(scores.size(0), targets.numel())
            scores = scores[:n]
            targets = targets[:n]
        if targets.numel() == 0:
            return metrics

        preds = scores.argmax(dim=1)
        exact = (preds == targets).float().mean() * 100
        metrics[f'acc_{prefix}'] = float(exact.item())

        pred_fg = preds != self.background_label
        target_fg = targets != self.background_label
        binary_correct = (pred_fg == target_fg).float().mean() * 100
        metrics[f'{prefix}_binary_acc'] = float(binary_correct.item())

        tp = (pred_fg & target_fg).float().sum()
        tn = ((~pred_fg) & (~target_fg)).float().sum()
        fp = (pred_fg & (~target_fg)).float().sum()
        fn = ((~pred_fg) & target_fg).float().sum()
        total = max(float(targets.numel()), 1.0)
        eps = torch.finfo(torch.float32).eps
        precision = tp / torch.clamp(tp + fp, min=eps)
        recall = tp / torch.clamp(tp + fn, min=eps)
        f1 = (2 * precision * recall) / torch.clamp(
            precision + recall, min=eps)

        metrics[f'{prefix}_tp_pct'] = float((tp / total * 100).item())
        metrics[f'{prefix}_tn_pct'] = float((tn / total * 100).item())
        metrics[f'{prefix}_fp_pct'] = float((fp / total * 100).item())
        metrics[f'{prefix}_fn_pct'] = float((fn / total * 100).item())
        metrics[f'{prefix}_precision'] = float((precision * 100).item())
        metrics[f'{prefix}_recall'] = float((recall * 100).item())
        metrics[f'{prefix}_f1'] = float((f1 * 100).item())

        if target_fg.any():
            fg_exact = (preds[target_fg] == targets[target_fg]).float().mean()
            metrics[f'{prefix}_fg_acc'] = float((fg_exact * 100).item())

        num_classes = self._infer_num_classes(scores)
        for k in self.topk:
            if num_classes <= 2 or k <= 1 or k > scores.size(1):
                continue
            topk_preds = scores.topk(k, dim=1).indices
            topk_acc = (topk_preds == targets[:, None]).any(dim=1).float().mean()
            metrics[f'{prefix}_top{k}_acc'] = float((topk_acc * 100).item())

        return metrics

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute bag-level and instance-level MIL classification metrics."""
        metrics: Dict[str, float] = {}

        bag_scores = [r['bag_score'] for r in results if 'bag_score' in r]
        bag_targets = [r['bag_label'] for r in results if 'bag_label' in r]
        if bag_scores and bag_targets:
            metrics.update(
                self._classification_metrics(
                    torch.cat(bag_scores, dim=0),
                    torch.cat(bag_targets, dim=0),
                    prefix='bag'))

        ins_scores = [r['ins_score'] for r in results if 'ins_score' in r]
        ins_targets = [r['ins_label'] for r in results if 'ins_label' in r]
        if ins_scores and ins_targets:
            metrics.update(
                self._classification_metrics(
                    torch.cat(ins_scores, dim=0),
                    torch.cat(ins_targets, dim=0),
                    prefix='instance'))

        return metrics
