# EDL PointMIL 测试脚本快速参考

## 📁 文件位置
```
mmdetection/
├── tools/
│   ├── test_edl_point_mil.sh    # 主测试脚本
│   └── test_scenarios.sh        # 测试场景集合
├── TEST_GUIDE.md                # 详细使用指南
└── QUICK_TEST_REFERENCE.md      # 本文件
```

## 🚀 快速开始

### 最简单的方式（假设已有checkpoint）
```bash
cd mmdetection
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth
```

### 或使用预定义场景
```bash
bash tools/test_scenarios.sh basic work_dirs/edl_point_mil_r50_fpn_1x/latest.pth
```

## 📋 测试场景速查表

| 场景 | 命令 | 用途 |
|------|------|------|
| **基础** | `test_scenarios.sh basic` | 验证模型能否加载和推理 |
| **快速** | `test_scenarios.sh quick` | 快速验证（数据量少） |
| **完整** | `test_scenarios.sh full` | 生成详细评估指标 |
| **分布式** | `test_scenarios.sh distributed` | 2GPU并行测试 |
| **可视化** | `test_scenarios.sh vis` | 生成预测结果图片 |
| **严格** | `test_scenarios.sh strict` | 高置信度（阈值0.8） |
| **宽松** | `test_scenarios.sh loose` | 低置信度（阈值0.5） |
| **集成** | `test_scenarios.sh ensemble` | TTA测试 |
| **自定义** | `test_scenarios.sh custom <args>` | 自定义参数 |

## 💻 常用命令示例

### 1️⃣ 基础功能测试
```bash
# 验证模型是否正常工作
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth
```

### 2️⃣ 测试并显示结果
```bash
# 显示前几张的预测结果
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth 1 -- --show
```

### 3️⃣ 保存可视化结果
```bash
# 将预测结果保存为图片
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth 1 -- \
  --show-dir output/predictions
```

### 4️⃣ 完整评估（同时保存结果和可视化）
```bash
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth 1 -- \
  --out results.pkl \
  --show-dir output/vis
```

### 5️⃣ 多GPU测试（2个GPU）
```bash
bash tools/test_edl_point_mil.sh work_dirs/edl_point_mil_r50_fpn_1x/latest.pth 2
```

### 6️⃣ 测试不同的阈值
```bash
# 严格模式：检测阈值 0.8
bash tools/test_edl_point_mil.sh checkpoint.pth 1 -- \
  --cfg-options model.roi_head.test_cfg.rcnn.score_thr=0.8

# 宽松模式：检测阈值 0.5
bash tools/test_edl_point_mil.sh checkpoint.pth 1 -- \
  --cfg-options model.roi_head.test_cfg.rcnn.score_thr=0.5
```

### 7️⃣ TTA测试时增强
```bash
bash tools/test_edl_point_mil.sh checkpoint.pth 1 -- --tta
```

### 8️⃣ 快速测试（小batch size）
```bash
bash tools/test_edl_point_mil.sh checkpoint.pth 1 -- \
  --cfg-options test_dataloader.batch_size=2
```

## 🎯 按使用场景选择测试

### 📊 我想...

**检查模型是否正常工作**
```bash
bash tools/test_scenarios.sh basic [checkpoint]
```

**快速验证（开发调试）**
```bash
bash tools/test_scenarios.sh quick [checkpoint]
```

**获取完整的评估指标**
```bash
bash tools/test_scenarios.sh full [checkpoint]
```

**生成可视化结果**
```bash
bash tools/test_scenarios.sh vis [checkpoint]
```

**对比不同的检测阈值**
```bash
bash tools/test_scenarios.sh strict [checkpoint]
bash tools/test_scenarios.sh loose [checkpoint]
```

**使用所有GPU进行测试**
```bash
bash tools/test_scenarios.sh distributed [checkpoint] 4  # 4GPU
```

**使用集成学习（更高精度）**
```bash
bash tools/test_scenarios.sh ensemble [checkpoint]
```

## 📂 输出文件位置

测试输出通常保存在：
```
work_dirs/
├── edl_point_mil_r50_fpn_1x_test/     # 单个测试的输出
├── results_yyyymmdd_hhmmss/           # 完整评估结果
└── visualization_yyyymmdd_hhmmss/     # 可视化图片
```

## ⚙️ 关键配置参数

当前配置的测试参数：

```yaml
检测阈值 (score_thr):        0.65    # 置信度阈值
NMS IOU阈值:                 0.45    # 非极大值抑制
单点检测Top-K:              1       # 每个点最多检测数
掩码阈值:                   0.65    # 掩码置信度
各分类后处理策略:          weighted_nms
最大输出boxes:              50      # 每张图最多输出个数
```

可以通过 `--cfg-options` 覆盖这些参数。

## 🔍 查找Checkpoint

**列出所有available的checkpoints:**
```bash
bash tools/test_scenarios.sh --list
```

**或手动查找:**
```bash
find work_dirs -name "*.pth"
```

## ⚡ 性能参考

| 场景 | GPU内存 | 预计时间 | 输出大小 |
|------|---------|---------|---------|
| 快速 | ~2GB | 1-2分钟 | ~100MB |
| 基础 | ~2GB | 5-10分钟 | ~500MB |
| 完整 | ~2GB | 10-20分钟 | ~1GB+ |
| 2GPU(分布式) | ~4GB | 5-10分钟 | ~1GB+ |
| TTA | ~4GB | 30-60分钟 | ~2GB+ |

## 🐛 常见问题

**Q: Checkpoint一直找不到？**
```bash
# 先列出可用的checkpoints
find work_dirs -name "*.pth" | head -5
```

**Q: 测试很慢？**
```bash
# 使用较小的batch size但不要太小
bash tools/test_scenarios.sh quick [checkpoint]
```

**Q: 显存不足？**
```bash
# 减少batch size
bash tools/test_edl_point_mil.sh checkpoint.pth 1 -- \
  --cfg-options test_dataloader.batch_size=1
```

**Q: 想要更多细节提示？**
```bash
bash tools/test_scenarios.sh --help
```

## 📖 更多资源

- 完整指南: 查看 `TEST_GUIDE.md`
- 配置文件: `mmdet/configs/edl_point_mil/edl_point_mil_r50_fpn_1x.py`
- 全部参数: 运行 `python tools/test.py --help`

---

**💡 提示:** 大多数情况下，使用 `test_scenarios.sh` 中的预定义场景更简单快捷！
