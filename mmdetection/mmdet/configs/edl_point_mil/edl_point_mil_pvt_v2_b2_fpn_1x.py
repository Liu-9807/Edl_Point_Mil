# PVTv2-B2 + FPN for PointMIL (EDL)
#
# Compatibility with ResNet+FPN pipeline (PointMIL.extract_feat: backbone -> neck -> roi_head):
# - Backbone returns a tuple of 4 NCHW maps (out_indices 0..3).
# - FPN neck.in_channels must match per-stage width:
#     embed_dims_i = embed_dims * num_heads[i]  with defaults embed_dims=64, num_heads=[1,2,5,8]
#     => [64, 128, 320, 512] (same as configs/pvt/retinanet_pvtv2-b2_fpn_1x_coco.py).
# - featmap_strides=[4,8,16,32] matches default strides (4,2,2,2) for four stages vs input.
# - init_cfg Pretrained loads official PVTv2-B2 cls weights; convert_weights=True (default) runs
#   pvt_convert() to match this implementation (see mmdet/models/backbones/pvt.py).
# - frozen_stages=4 freezes all four PVT stages (each stage: patch_embed + blocks + norm).

from mmengine.config import read_base

with read_base():
    from .._base_.datasets.Youyu_jiangyong_dataset import *
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
        type='PyramidVisionTransformerV2',
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.1,
        convert_weights=True,
        frozen_stages=4,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        type='MILRoIHead',
        infer_base_scales=[256],
        infer_ratios=[1.0],
        infer_anchor_offsets=[
            (0, 0),
            (-0.2, -0.2),
            (0.2, 0.2),
            (-0.2, 0.2),
            (0.2, -0.2),
            (0.3, 0.3),
            (-0.3, -0.3),
            (0.3, -0.3),
            (-0.3, 0.3)
        ],
        proposal_generator=dict(
            type='PointPseudoBoxGenerator',
            box_sizes=[[128, 128], [256, 256]],
            box_offset=25,
            mil_bag_size_base=50,
            use_jittered_positive_proposals=True,
            jitter_base_scales=[128, 256],
            jitter_aspect_ratios=[1.0],
            jitter_center_offsets=[
                (0, 0),
                (-0.2, -0.2),
                (0.2, 0.2),
                (-0.2, 0.2),
                (0.2, -0.2),
                (0.3, 0.3),
                (-0.3, -0.3),
                (0.3, -0.3),
                (-0.3, 0.3),
            ],
        ),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='EDLHead',
            num_classes=2,
            use_instance_mask=True,
            use_wsddn_dual_branch=False,
            ins_enhance=True,
            loss_edl=dict(
                type='EDLLoss',
                loss_type='mse',
                loss_weight=1.0,
                branch='bag',
                annealing_step=10),
            loss_aux=dict(
                type='EDLLoss',
                loss_type='mse',
                loss_weight=0.5,
                branch='instance',
                annealing_step=2),
            edl_evidence_func='softplus',
        ),

        train_cfg=dict(
            rcnn=None
        ),
        test_cfg=dict(
            rcnn=dict(
                score_thr=0.05,
                score_mode='exclude_class0',
                per_point_topk=1,
                min_alpha_sum=0.0,
                mask_thr=0.65,
                mask_min_area=4,
                mask_refine_mode='quantile',
                mask_fallback_to_proposal=True,
                allow_empty_results=False,
                empty_fallback='top1_per_point',
                empty_fallback_min_score=0.0,
                debug_mask_refine=True,
                debug_mask_refine_max_rois=20,
                debug_proposal_scores=True,
                debug_proposal_scores_max_rois=20,
                postprocess_strategy='weighted',
                weighted_iou_thr=0.1,
                weighted_score_type='avg',
                nms=dict(type='nms', iou_threshold=0.45),
                max_per_img=50)
        )),
)

visualizer = dict(
    type='MILVisualizer',
    draw_gt_pred_overlay=True,
    gt_overlay_color='deepskyblue',
    pred_overlay_color='lime',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
    name='visualizer'
)

custom_hooks = [
    dict(type='MILProposalHook', interval=50),
    dict(
        type='MILEvidenceHook',
        interval=50,
        n_per_side=5,
        global_max_side=720,
        patch_barh=True,
        combine_mask_vis=True,
        epoch_snapshot_interval=1,
        num_samples=5),
    dict(type='MILEpochScatterHook', interval=1),
    dict(type='MILInferenceStageVisHook', interval=10),
    dict(type='MILMaskRefineVisHook', interval=10, max_items=20, max_points=10, max_proposals_per_point=20),
]

randomness = dict(seed=42, deterministic=False)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=35., norm_type=2),
)
