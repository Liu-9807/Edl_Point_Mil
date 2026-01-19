import torch
from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class MILTrainingMetric(BaseMetric):
    """
    用于监控 MIL 训练过程中的分类准确率。
    
    与 PointMilMetric 的区别：
    - PointMilMetric: 评估最终检测框的定位精度（推理端）
    - MILTrainingMetric: 监控多实例分类的学习效果（训练端）
    """
    
    rule = 'greater'
    init_value_map = {'acc_bag': 0, 'acc_instance': 0}
    collect_device = 'cpu'
    
    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: str = 'train_mil_',
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.bag_preds = []
        self.bag_targets = []
        self.ins_preds = []
        self.ins_targets = []
    
    def process(self, data_batch, data_samples):
        """
        处理一个批次。
        
        data_samples 应包含：
        - bag_score: [B, num_classes] 包级别预测
        - bag_label: [B] 包级别真实标签
        - ins_score: [N, num_classes] 实例级别预测
        - ins_label: [N] 实例级别真实标签
        """
        for sample in data_samples:
            # 包级别准确率
            if hasattr(sample, 'bag_score') and hasattr(sample, 'bag_label'):
                self.bag_preds.append(sample.bag_score.detach().cpu())
                self.bag_targets.append(sample.bag_label.detach().cpu())
            
            # 实例级别准确率
            if hasattr(sample, 'ins_score') and hasattr(sample, 'ins_label'):
                self.ins_preds.append(sample.ins_score.detach().cpu())
                self.ins_targets.append(sample.ins_label.detach().cpu())
    
    def compute_metrics(self, results: list = None) -> dict:
        """计算包级别和实例级别的准确率。"""
        metrics = {}
        
        # 包级别准确率
        if self.bag_preds:
            bag_preds = torch.cat(self.bag_preds, dim=0)
            bag_targets = torch.cat(self.bag_targets, dim=0)
            pred_labels = torch.argmax(bag_preds, dim=1)
            acc_bag = (pred_labels == bag_targets).float().mean().item() * 100
            metrics[f'{self.prefix}acc_bag'] = acc_bag
        
        # 实例级别准确率
        if self.ins_preds:
            ins_preds = torch.cat(self.ins_preds, dim=0)
            ins_targets = torch.cat(self.ins_targets, dim=0)
            pred_labels = torch.argmax(ins_preds, dim=1)
            acc_instance = (pred_labels == ins_targets).float().mean().item() * 100
            metrics[f'{self.prefix}acc_instance'] = acc_instance
        
        return metrics