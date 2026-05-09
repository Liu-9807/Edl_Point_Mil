# dataset settings

dataset_type = 'DotaPointMilDataset'
data_root = '/home/user/Dataset/DOTA-v2.0/'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadPointAnnotations'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='ResizePoints'),
    dict(
        type='PackPointDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'flip', 'flip_direction', 'ori_gt_points'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadPointAnnotations'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='ResizePoints'),
    dict(
        type='PackPointDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'flip', 'flip_direction', 'eval_gt_bboxes', 'eval_gt_labels',
            'eval_gt_source', 'instances', 'ori_gt_points'))
]

train_dataloader = dict(
    batch_size=16,
    num_workers=24,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/labelTxt-v2.0/DOTA-v2.0_train',
        data_prefix=dict(img='train/images'),
        point_sampling='random_hbb',
        point_seed=42,
        ignore_difficult=False,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=24,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/labelTxt-v2.0',
        data_prefix=dict(img='val/images'),
        point_sampling='random_hbb',
        point_seed=42,
        ignore_difficult=False,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=test_pipeline,
        backend_args=backend_args,
        test_mode=True))

test_dataloader = val_dataloader

train_evaluator = dict(
    type='MILTrainingMetric',
    prefix='train_mil',
    num_classes=19,
    background_label=0,
    collect_device='cpu')

val_evaluator = dict(
    type='CocoMetric',
    ann_file=None,
    metric='bbox',
    classwise=True,
    pred_label_offset=0,
    prefix='dota_coco',
    collect_device='cpu')

test_evaluator = val_evaluator
