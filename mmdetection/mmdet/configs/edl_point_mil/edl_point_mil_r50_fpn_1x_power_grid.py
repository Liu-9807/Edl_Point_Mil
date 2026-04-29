from mmengine.config import read_base

with read_base():
    from .._base_.datasets.power_grid_dataset import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *

default_scope = 'mmdet'

model = dict(
    type='PointMIL',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    roi_head=dict(
        type='MILRoIHead',
        proposal_generator=dict(
            type='PointPseudoBoxGenerator',
            box_sizes=[[128, 128], [64, 64]],
            box_offset=10,
            num_neg_samples=50),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='EDLHead',
            num_classes=2,
            use_instance_mask=True,
            ins_enhance=True,
            loss_edl=dict(
                type='EDLLoss',
                loss_type='mse',
                loss_weight=1.0,
                branch='bag',
                annealing_step=2),
            loss_aux=dict(
                type='EDLLoss',
                loss_type='mse',
                loss_weight=0.5,
                branch='instance',
                annealing_step=2),
            edl_evidence_func='softplus',
        ),
        train_cfg=dict(rcnn=None),
        test_cfg=dict(
            rcnn=dict(
                score_thr=0.65,
                score_mode='max_class',
                per_point_topk=1,
                min_alpha_sum=3.0,
                mask_thr=0.65,
                mask_min_area=4,
                mask_fallback_to_proposal=False,
                postprocess_strategy='weighted_nms',
                weighted_iou_thr=0.6,
                weighted_score_type='max',
                nms=dict(type='nms', iou_threshold=0.45),
                max_per_img=50))),
                )

visualizer = dict(
    type='MILVisualizer',
    draw_gt_pred_overlay=True,
    gt_overlay_color='deepskyblue',
    pred_overlay_color='lime',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
    name='visualizer')

custom_hooks = [
    dict(type='MILProposalHook', interval=50),
    dict(type='MILEvidenceHook', interval=50),
    dict(type='MILEpochScatterHook', interval=1),
    dict(type='MILEpochMaskHook', interval=1, num_samples=3, instances_per_sample=4, collect_interval=20),
    dict(type='MILInferenceVisHook', interval=1),
]

randomness = dict(seed=42, deterministic=False)
