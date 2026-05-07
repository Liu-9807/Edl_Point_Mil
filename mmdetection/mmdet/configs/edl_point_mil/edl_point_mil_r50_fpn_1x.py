from mmengine.config import read_base

with read_base():
    from .._base_.datasets.Youyu_jiangyong_dataset import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *

default_scope = 'mmdet'

# model settings
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
        num_outs=5),
    roi_head=dict(
        type='MILRoIHead',
        infer_base_scales=[256],
        infer_ratios=[1.0],
        infer_anchor_offsets=[
            (0, 0),
            (-0.4, -0.4),
            (0.4, 0.4),
            (-0.4, 0.4),
            (0.4, -0.4)
        ],
        proposal_generator = dict(
            type='PointPseudoBoxGenerator',
            box_sizes=[[128, 128], [256, 256]],
            box_offset=5,
            num_neg_samples=50
        ),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='EDLHead',
            num_classes=1,
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

        train_cfg=dict(
            rcnn=None  # rcnn的设置现在由roi_head的train_cfg管理
        ),
        test_cfg=dict(
            rcnn=dict(
            score_thr=0.66,
            score_mode='max_class',
            per_point_topk=1,
            min_alpha_sum=0.0,
            mask_thr=0.65,
                mask_min_area=4,
            mask_refine_mode='quantile',
            mask_fallback_to_proposal=False,
            debug_mask_refine=True,
            debug_mask_refine_max_rois=20,
            debug_proposal_scores=True,
            debug_proposal_scores_max_rois=20,
            postprocess_strategy='weighted_nms',
            weighted_iou_thr=0.6,
                weighted_score_type='max',
            nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=50)
        )),
)

visualizer = dict(
    type='MILVisualizer',  # <--- 使用自定义的可视化器
    draw_gt_pred_overlay=True,
    gt_overlay_color='deepskyblue',
    pred_overlay_color='lime',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
    name='visualizer'
)

custom_hooks = [
    dict(type='MILProposalHook', interval=50),
    dict(type='MILEvidenceHook', interval=50),
    dict(type='MILEpochScatterHook', interval=1),
    dict(type='MILEpochMaskHook', interval=1, num_samples=3, instances_per_sample=4, collect_interval=20),
    dict(type='MILInferenceStageVisHook', interval=500),
    dict(type='MILMaskRefineVisHook', interval=500, max_items=20, max_points=5, max_proposals_per_point=20),
]

randomness = dict(seed=42, deterministic=False)




