from mmengine.config import read_base

with read_base():
    from .._base_.datasets.dota_point_mil_dataset import *
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
        infer_base_scales=[32, 64, 128, 256],
        infer_ratios=[0.5, 1.0, 2.0],
        infer_anchor_offsets=[
            (0, 0),
            (-0.3, -0.3),
            (0.3, 0.3),
            (-0.3, 0.3),
            (0.3, -0.3),
        ],
        proposal_generator=dict(
            type='PointPseudoBoxGenerator',
            box_sizes=[[16, 16], [32, 32], [64, 64], [128, 128], [256, 256]],
            box_offset=8,
            num_pos_samples=16,
            num_neg_samples=64,
            pos_bag_prob=0.5,
            sample_coordinate_mode='original',
            class_box_size_mode='absolute',
            negative_size_source='class_prior',
            box_offset_mode='size_ratio',
            box_offset_ratio=0.1,
            size_jitter=0.15,
            min_input_box_size=4.0,
            class_box_sizes=[
                [[47, 50], [76, 73], [119, 114]],
                [[50, 51], [68, 67], [116, 112]],
                [[20, 19], [31, 30], [55, 51]],
                [[53, 53], [132, 146], [192, 239]],
                [[10, 9], [15, 15], [23, 23]],
                [[24, 22], [39, 39], [68, 63]],
                [[22, 19], [34, 32], [48, 48]],
                [[47, 90], [89, 111], [127, 195]],
                [[55, 62], [110, 125], [152, 221]],
                [[13, 12], [20, 20], [36, 35]],
                [[84, 82], [158, 164], [353, 387]],
                [[21, 20], [34, 34], [54, 53]],
                [[46, 46], [80, 97], [149, 195]],
                [[31, 30], [42, 40], [53, 51]],
                [[23, 63], [26, 69], [40, 83]],
                [[54, 55], [70, 69], [107, 96]],
                [[257, 168], [416, 285], [637, 439]],
                [[21, 21], [25, 24], [29, 27]],
            ]),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='EDLHead',
            num_classes=19,
            use_instance_mask=True,
            # WSDDN-style dual projections (class x detection) + spatial softmax bag agg.
            use_wsddn_dual_branch=True,
            ins_enhance=True,
            instance_loss_mode='ii',
            loss_edl=dict(
                type='EDLLoss',
                loss_type='mse',
                loss_weight=1.0,
                branch='bag',
                annealing_step=1000),
            loss_aux=dict(
                type='EDLLoss',
                loss_type='mse',
                loss_weight=1.0,
                branch='instance',
                separate='II',
                aggregate='mean',
                annealing_step=1000),
            edl_evidence_func='softplus'),
        train_cfg=dict(rcnn=None),
        test_cfg=dict(
            rcnn=dict(
                score_thr=0.05,
                score_mode='exclude_class0',
                per_point_topk=1,
                min_alpha_sum=0.0,
                mask_thr=0.5,
                mask_min_area=4,
                mask_refine_mode='quantile',
                mask_fallback_to_proposal=True,
                allow_empty_results=False,
                empty_fallback='top1_per_point',
                empty_fallback_min_score=0.0,
                postprocess_strategy='weighted_nms',
                weighted_iou_thr=0.5,
                weighted_score_type='max',
                nms=dict(type='nms', iou_threshold=0.45),
                max_per_img=300))))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='MILVisualizer',
    draw_gt_pred_overlay=True,
    gt_overlay_color='deepskyblue',
    pred_overlay_color='lime',
    vis_backends=vis_backends,
    name='visualizer')

custom_hooks = [
    dict(type='MILProposalHook', interval=200),
    dict(type='MILEvidenceHook', interval=200, max_instances=1),
    dict(
        type='MILEpochMaskHook',
        interval=1,
        num_samples=3,
        instances_per_sample=6,
        pos_instances_per_sample=3,
        neg_instances_per_sample=3,
        positive_class_ids=list(range(1, 19)),
        collect_interval=100),
    dict(
        type='MILMultiClassAnalysisHook',
        interval=100,
        max_instances=4096,
        topk=(1, 5),
        background_label=0)
]

randomness = dict(seed=42, deterministic=False)
