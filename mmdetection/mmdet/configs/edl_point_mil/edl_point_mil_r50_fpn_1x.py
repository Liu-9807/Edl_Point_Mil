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
        proposal_generator = dict(
            type='PointPseudoBoxGenerator',
            box_sizes=[[128, 128], [256, 256]], 
            box_offset=5,
            num_neg_samples=35
        ),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='EDLHead',
            num_cls=1,
            pooler=dict(
                type='Gated_Attention_Pooling',
                in_dim=1024,
                hid_dim=128,
            ),
            ins_enhance=True,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=3.0),
            loss_edl=dict(
                type='EDLLoss',
                loss_type='mse',
                loss_weight=0.5,
                annealing_step=2),
            loss_aux=dict(
                type='EDLLoss',
                loss_type='mse',
                loss_weight=0.5,
                annealing_step=2),
            edl_evidence_func='relu',
        ),

        train_cfg=dict(
            rcnn=None  # rcnn的设置现在由roi_head的train_cfg管理
        ),
        test_cfg=dict(
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)
        )),
)

visualizer = dict(
    type='MILVisualizer',  # <--- 使用自定义的可视化器
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
    name='visualizer'
)

custom_hooks = [
    dict(type='MILProposalHook', interval=10, out_dir='work_dirs/debug_vis'),
    dict(type='MILEvidenceHook', interval=10, out_dir='work_dirs/vis_evidence'),
    dict(type='MILEpochScatterHook', interval=1, out_dir='work_dirs/epoch_scatter')
    ]

    # # 模型训练和测试的设置
    # train_cfg=dict(
    #     rcnn=None  # rcnn的设置现在由roi_head的train_cfg管理
    # ),
    # test_cfg=dict(
    #     rcnn=dict(
    #         score_thr=0.05,
    #         nms=dict(type='nms', iou_threshold=0.5),
    #         max_per_img=100)
    # ))



# # 注册自定义的 Hook
# custom_hooks = [
#     dict(type='MILRoIHeadEpochHook')
# ]




