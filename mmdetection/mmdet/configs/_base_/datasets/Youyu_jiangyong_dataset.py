# 数据集类型
dataset_type = 'Wind_turbine_generator_Dataset'

# 数据路径配置 
data_root = '/home/user/Dataset/YouYu-JiangYong/ori/'

# 图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# 训练流程
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadPointAnnotations'),  # 加载点标注
    dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
    dict(type='RandomFlip', prob=0.0),  # 改为 prob 参数
    dict(type='ResizePoints'), # [新增] 必须在这里对点进行对应的几何变换
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PackPointDetInputs')  # 替代 DefaultFormatBundle + Collect
]

# 验证/测试流程
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadPointAnnotations'),  # 加载点标注
    dict(
        type='Resize',
        scale=(2048, 2048),
        keep_ratio=True
    ),
    dict(type='ResizePoints'), # [新增] 测试时也要加，否则推理时的 proposals 也会错
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PackPointDetInputs')
]

# 训练数据加载器配置
train_dataloader = dict(
    batch_size=8,
    num_workers=24,
    persistent_workers=False,  # 保持工作进程以加速数据加载
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='result',  # 相对于 data_root 的路径
        data_prefix=dict(img=''),  # 图像目录前缀
        point_to_bbox_size=20,  # 点转换为边界框的大小 (可调整)
        use_txt_labels=True,  # 是否加载 txt 标签文件
        filter_cfg=dict(filter_empty_gt=True),  # 过滤无标注图像
        pipeline=train_pipeline,
        backend_args=None
    )
)

# 验证数据加载器配置
val_dataloader = dict(
    batch_size=1,
    num_workers=24,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='result',
        data_prefix=dict(img=''),
        point_to_bbox_size=20,
        use_txt_labels=True,
        filter_cfg=dict(filter_empty_gt=False),  # 验证时保留所有图像
        pipeline=test_pipeline,
        backend_args=None,
        test_mode=True  # 明确标记为验证模式
    )
)


# 测试数据加载器配置
test_dataloader = val_dataloader

# 1. 训练时的 MIL 分类准确率监控
train_evaluator = dict(
    type='MILTrainingMetric',
    prefix='train_mil_',
    collect_device='cpu'
)

# 2. 验证/测试时的检测框精度评估
val_evaluator = dict(
    type='PointMilMetric',
    iou_thr=0.5,
    prefix='val_detection_',
    collect_device='cpu'
)

test_evaluator = val_evaluator