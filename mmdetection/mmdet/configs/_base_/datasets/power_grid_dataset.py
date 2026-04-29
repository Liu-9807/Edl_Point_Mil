# dataset settings

dataset_type = 'PowerGridPointCocoDataset'
data_root = '/home/user/Dataset/Power_grid_dataset/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadPointAnnotations'),
    dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='ResizePoints'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PackPointDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadPointAnnotations'),
    dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
    dict(type='ResizePoints'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(
        type='PackPointDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction',
            'eval_gt_bboxes', 'eval_gt_labels', 'eval_gt_source'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=24,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='coco/instances_train.json',
        data_prefix=dict(img=''),
        geojson_dir='geojsons',
        image_root='images',
        point_to_bbox_size=20,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        backend_args=None))

optim_warper = dict(
    type='OptimWrapper',
    accumulation_steps=4,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=24,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='coco/instances_test.json',
        data_prefix=dict(img=''),
        geojson_dir='geojsons',
        image_root='images',
        point_to_bbox_size=20,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=test_pipeline,
        backend_args=None,
        test_mode=True))

test_dataloader = val_dataloader

train_evaluator = dict(
    type='MILTrainingMetric',
    prefix='train_mil_',
    collect_device='cpu')

val_evaluator = dict(
    type='PointMilMetric',
    iou_thr=0.5,
    use_eval_gt_from_meta=True,
    prefix='val_detection_',
    collect_device='cpu')

test_evaluator = val_evaluator
