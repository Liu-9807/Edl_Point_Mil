from mmengine.model import BaseModule
from torch.amp import autocast
from mmdet.registry import MODELS
import torch
import torch.nn as nn


def auto_fp16(apply_to=None, out_fp32=False):
    """Decorator to enable automatic mixed precision (AMP)."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with autocast():
                return func(*args, **kwargs)
        return wrapper
    return decorator


class MeanPooling(BaseModule):
    """Mean pooling layer for MIL."""
    def __init__(self, init_cfg=None):
        super(MeanPooling, self).__init__(init_cfg)

    def forward(self, x):
        return torch.mean(x, dim=0, keepdim=True)

class MaxPooling(BaseModule):
    """Max pooling layer for MIL."""
    def __init__(self, init_cfg=None):
        super(MaxPooling, self).__init__(init_cfg)

    def forward(self, x):
        return torch.max(x, dim=0, keepdim=True)[0]

def build_mil_pooler(cfg):
    """Build MIL pooler."""
    return MODELS.build(cfg)


@MODELS.register_module()
class Attention_Pooling(BaseModule):
    def __init__(self, in_dim=1024, hid_dim=512, init_cfg=None):
        super(Attention_Pooling, self).__init__(init_cfg)
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, x, ret_attn=True):
        A_ = self.attention(x)
        A_ = torch.transpose(A_, 1, 0)
        score = nn.functional.softmax(A_, dim=1)
        out = torch.matmul(score, x)
        if ret_attn:
            return out, score
        return out


@MODELS.register_module()
class Gated_Attention_Pooling(BaseModule):
    def __init__(self, in_dim, hid_dim, dropout=0.5, init_cfg=None):
        super(Gated_Attention_Pooling, self).__init__(init_cfg)
        self.fc1 = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.Tanh(), nn.Dropout(dropout))
        self.score_fc = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.fc2 = nn.Linear(hid_dim, 1)

    def forward(self, x, ret_attn=False):
        emb = self.fc1(x)
        scr = self.score_fc(x)
        new_emb = emb.mul(scr)
        A_ = self.fc2(new_emb)
        A_ = torch.transpose(A_, 1, 0)
        score = nn.functional.softmax(A_, dim=1)
        out = torch.matmul(score, x)
        if ret_attn:
            return out, score
        return out


@MODELS.register_module()
class MILHead(BaseModule):
    """Multiple Instance Learning BBox Head.

    This head implements the generic MIL paradigm:
    Score = g(pool(f(x_k)))

    Args:
        embedder (dict): Config for the instance embedder `f`.
        pooler (dict): Config for the permutation-invariant pooler `pool`.
        classifier (dict): Config for the classifier `g`.
        loss_mil (dict): Config of MIL loss.
        init_cfg (dict, optional): Initialization config dict. Defaults to None.
    """
    def __init__(self,
                 embedder,
                 pooler,
                 classifier,
                 loss_mil=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 init_cfg=None):
        super(MILHead, self).__init__(init_cfg)
        
        if isinstance(embedder, dict):
            self.instance_embedder = MODELS.build(embedder)
        else:
            self.instance_embedder = embedder
        self.pooler = build_mil_pooler(pooler)
        if isinstance(classifier, dict):
            self.classifier = MODELS.build(classifier)
        else:
            self.classifier = classifier
        
        self.loss_mil = MODELS.build(loss_mil)

    @auto_fp16()
    def forward(self, x):
        """
        Args:
            x (Tensor): Input features, shape (N, C, H, W),
                        where N is the number of instances in a bag.
        """
        # Flatten features from (N, C, H, W) to (N, C*H*W)
        x = x.view(x.size(0), -1)
        
        # 1. Instance Embedding: f(x_k)
        instance_embeddings = self.instance_embedder(x)
        
        # 2. Pooling: pool(f(x_k))
        bag_embedding = self.pooler(instance_embeddings)
        
        # 3. Classification: g(...)
        bag_score, ins_score = self.classifier(bag_embedding)

        return bag_score, ins_score

    # -------------------------
    # MMDetection 3.x 兼容接口
    # -------------------------
    def predict(self, x, **kwargs):
        """推理占位：直接复用 forward 输出。"""
        return self.forward(x)

    def _forward(self, x, **kwargs):
        """tensor 模式前向：直接复用 forward。"""
        return self.forward(x)

    def loss(self,
             cls_score,
             bag_label,
             **kwargs):
        """Compute loss."""
        losses = dict()
        losses['loss_mil'] = self.loss_mil(cls_score, bag_label)
        return losses