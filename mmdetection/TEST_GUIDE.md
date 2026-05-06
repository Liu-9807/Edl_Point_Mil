# EDL PointMIL 模型测试脚本使用指南

## 脚本位置
`tools/test_edl_point_mil.sh`

## 基本用法

### 单GPU测试（推荐）
```bash
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth
```

### 多GPU分布式测试
```bash
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth 2
```

## 使用示例

### 1. 基础测试 - 验证模型加载和推理
```bash
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth
```

### 2. 单GPU测试 - 显式指定GPU数量
```bash
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth 1
```

### 3. 测试并显示可视化结果
```bash
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth 1 -- --show
```

### 4. 测试并保存可视化图片
```bash
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth 1 -- --show-dir output/vis
```

### 5. 测试并输出评估报告
```bash
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth 1 -- --out results.pkl
```

### 6. 多GPU分布式测试 - 2个GPU
```bash
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth 2
```

### 7. 多GPU分布式测试 - 4个GPU + 可视化
```bash
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth 4 -- --show --show-dir output/vis
```

### 8. 使用TTA（测试时增强）
```bash
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth 1 -- --tta
```

### 9. 覆盖配置参数
```bash
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth 1 -- \
  --cfg-options model.test_cfg.rcnn.score_thr=0.5
```

### 10. 组合选项 - 完整测试
```bash
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth 1 -- \
  --show --show-dir output/pred --out results.pkl \
  --cfg-options model.test_cfg.rcnn.score_thr=0.6 model.test_cfg.rcnn.nms.iou_threshold=0.4
```

## 参数说明

| 参数 | 说明 | 必需 | 默认值 |
|------|------|------|--------|
| CHECKPOINT | 模型权重文件路径 | 是 | - |
| GPUS | 使用的GPU数量 | 否 | 1 |
| --show | 显示预测结果 | 否 | - |
| --show-dir | 保存预测结果的目录 | 否 | - |
| --out | 输出pickle文件用于离线评估 | 否 | - |
| --tta | 启用测试时增强 | 否 | - |
| --cfg-options | 覆盖配置文件中的参数 | 否 | - |

## 输出文件

测试生成的文件将保存在：
```
work_dirs/edl_point_mil_r50_fpn_1x_test/
├── *.log              # 测试日志
├── output.pkl         # 模型输出结果（如果指定--out）
└── vis/               # 可视化结果（如果指定--show-dir）
```

## 模型配置说明

该脚本运行的配置文件定义了：

### 骨干网络（Backbone）
- ResNet-50
- 冻结前4个stage
- 使用预训练权重

### 颈部网络（Neck）
- FPN (Feature Pyramid Network)
- 5层特征预测

### 检测头（Head）
- MILRoIHead: 多实例学习RoI头
- EDLHead: 证据深度学习头
- 支持实例掩码增强

### 关键测试参数
```
检测阈值: 0.65
NMS阈值: 0.45
单点Top-K: 1
最大预测数: 50
```

## 故障排除

### 1. Checkpoint 找不到
```
[ERROR] Checkpoint not found: work_dirs/edl_point_mil_r50_fpn_1x/latest.pth
```
解决方案：确保checkpoint文件存在，或指定完整路径

### 2. 配置文件错误
```
[ERROR] Config not found: mmdet/configs/edl_point_mil/edl_point_mil_r50_fpn_1x.py
```
解决方案：确保在项目根目录运行脚本

### 3. GPU不足
多GPU测试时确保系统有足够的显存

### 4. 内存不足
可以在参数中减小batch_size：
```bash
bash tools/test_edl_point_mil.sh checkpoint.pth 1 -- \
  --cfg-options test_dataloader.batch_size=1
```

## 相关文件

- 配置文件: `mmdet/configs/edl_point_mil/edl_point_mil_r50_fpn_1x.py`
- Python测试脚本: `tools/test.py`
- 分布式测试脚本: `tools/dist_test.sh`
- 训练脚本: `tools/train_edl_point_mil.sh`

## 更多信息

运行 `python tools/test.py --help` 查看所有可用参数。
