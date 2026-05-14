from mmengine.config import read_base

with read_base():
    from .edl_point_mil_r50_fpn_1x import *

# Bag-level softmax + soft-target CE; no instance-level loss_aux.
model['roi_head']['bbox_head'].update(
    dict(
        bag_loss_type='softmax_ce',
        loss_edl=dict(type='SoftTargetCrossEntropyLoss', loss_weight=1.0),
        loss_aux=dict(
            type='EDLLoss',
            loss_type='mse',
            loss_weight=0.0,
            branch='instance',
            annealing_step=2),
    ))
