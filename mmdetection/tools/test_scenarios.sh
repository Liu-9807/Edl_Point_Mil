#!/usr/bin/env bash

# EDL PointMIL 测试场景集合
# 该脚本提供了多个预配置的测试场景

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的信息
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查checkpoint是否存在
check_checkpoint() {
    local checkpoint="$1"
    if [[ ! -f "$checkpoint" ]]; then
        log_error "Checkpoint not found: $checkpoint"
        return 1
    fi
    return 0
}

# 显示用法
show_usage() {
    cat << EOF
EDL PointMIL 测试场景脚本

用法: bash $(basename "$0") <scenario> [checkpoint]

预定义场景:
  1. basic              - 基础功能测试
  2. quick              - 快速验证（单GPU）
  3. full               - 完整评估（单GPU + 评估指标）
  4. distributed        - 分布式测试（2GPU）
  5. vis                - 带可视化的测试
  6. strict             - 严格阈值测试
  7. loose              - 宽松阈值测试
  8. ensemble           - TTA集成测试
  9. custom             - 自定义参数（需要提供checkpoint）

选项:
  -h, --help           显示帮助信息
  -l, --list           列出所有可用的checkpoint

示例:
  bash $(basename "$0") 1 work_dirs/edl_point_mil_r50_fpn_1x/latest.pth
  bash $(basename "$0") basic
  bash $(basename "$0") vis work_dirs/edl_point_mil_r50_fpn_1x/latest.pth
  bash $(basename "$0") full
EOF
}

# 列出available checkpoints
list_checkpoints() {
    log_info "搜索可用的checkpoints..."
    find work_dirs -name "latest.pth" -o -name "epoch_*.pth" 2>/dev/null | sort
}

# 场景1: 基础功能测试 - 验证模型能否正常加载和推理
test_basic() {
    local checkpoint="${1:-work_dirs/edl_point_mil_r50_fpn_1x/latest.pth}"
    
    check_checkpoint "$checkpoint" || return 1
    
    log_info "执行基础功能测试..."
    log_info "- 加载模型和checkpoint"
    log_info "- 执行推理"
    log_info "- 验证输出"
    
    bash mmdetection/tools/test_edl_point_mil.sh "$checkpoint" 1
}

# 场景2: 快速验证 - 快速检查模型是否工作正常
test_quick() {
    local checkpoint="${1:-work_dirs/edl_point_mil_r50_fpn_1x/latest.pth}"
    
    check_checkpoint "$checkpoint" || return 1
    
    log_info "执行快速验证..."
    log_info "- 使用较少的数据进行测试"
    
    bash mmdetection/tools/test_edl_point_mil.sh "$checkpoint" 1 -- \
        --cfg-options test_dataloader.batch_size=2
}

# 场景3: 完整评估 - 生成详细的评估指标
test_full() {
    local checkpoint="${1:-work_dirs/edl_point_mil_r50_fpn_1x/latest.pth}"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    check_checkpoint "$checkpoint" || return 1
    
    log_info "执行完整评估..."
    log_info "- 完整数据集测试"
    log_info "- 生成评估指标"
    log_info "- 保存结果到: work_dirs/results_${timestamp}"
    
    bash mmdetection/tools/test_edl_point_mil.sh "$checkpoint" 1 -- \
        --out "work_dirs/results_${timestamp}/predictions.pkl" \
        --work-dir "work_dirs/results_${timestamp}"
}

# 场景4: 分布式测试 - 多GPU并行测试
test_distributed() {
    local checkpoint="${1:-work_dirs/edl_point_mil_r50_fpn_1x/latest.pth}"
    local gpus="${2:-2}"
    
    check_checkpoint "$checkpoint" || return 1
    
    log_info "执行分布式测试 - 使用${gpus}个GPU..."
    log_info "- 并行处理数据"
    log_info "- 汇总结果"
    
    bash mmdetection/tools/test_edl_point_mil.sh "$checkpoint" "$gpus"
}

# 场景5: 可视化测试 - 生成预测结果可视化
test_vis() {
    local checkpoint="${1:-work_dirs/edl_point_mil_r50_fpn_1x/latest.pth}"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local vis_dir="work_dirs/visualization_${timestamp}"
    
    check_checkpoint "$checkpoint" || return 1
    
    log_info "执行可视化测试..."
    log_info "- 生成预测结果图片"
    log_info "- 保存到: ${vis_dir}"
    
    bash mmdetection/tools/test_edl_point_mil.sh "$checkpoint" 1 -- \
        --show --show-dir "$vis_dir" \
        --work-dir "$vis_dir"
    
    log_info "可视化结果位置: ${vis_dir}"
    if command -v ls &> /dev/null; then
        log_info "生成的图片: $(ls ${vis_dir}/*.jpg 2>/dev/null | wc -l) 张"
    fi
}

# 场景6: 严格阈值测试 - 高置信度检测
test_strict() {
    local checkpoint="${1:-work_dirs/edl_point_mil_r50_fpn_1x/latest.pth}"
    
    check_checkpoint "$checkpoint" || return 1
    
    log_info "执行严格阈值测试..."
    log_info "- 检测阈值: 0.8 (原始: 0.65)"
    log_info "- NMS阈值: 0.3 (原始: 0.45)"
    log_info "- 这会产生更少但更可信的检测"
    
    bash mmdetection/tools/test_edl_point_mil.sh "$checkpoint" 1 -- \
        --cfg-options \
            model.roi_head.test_cfg.rcnn.score_thr=0.8 \
            model.roi_head.test_cfg.rcnn.nms.iou_threshold=0.3
}

# 场景7: 宽松阈值测试 - 低置信度检测
test_loose() {
    local checkpoint="${1:-work_dirs/edl_point_mil_r50_fpn_1x/latest.pth}"
    
    check_checkpoint "$checkpoint" || return 1
    
    log_info "执行宽松阈值测试..."
    log_info "- 检测阈值: 0.5 (原始: 0.65)"
    log_info "- NMS阈值: 0.6 (原始: 0.45)"
    log_info "- 这会产生更多的检测"
    
    bash mmdetection/tools/test_edl_point_mil.sh "$checkpoint" 1 -- \
        --cfg-options \
            model.roi_head.test_cfg.rcnn.score_thr=0.5 \
            model.roi_head.test_cfg.rcnn.nms.iou_threshold=0.6
}

# 场景8: TTA集成测试 - 使用测试时增强
test_ensemble() {
    local checkpoint="${1:-work_dirs/edl_point_mil_r50_fpn_1x/latest.pth}"
    
    check_checkpoint "$checkpoint" || return 1
    
    log_info "执行TTA集成测试..."
    log_info "- 启用测试时增强"
    log_info "- 处理时间更长，准确度通常更高"
    
    bash mmdetection/tools/test_edl_point_mil.sh "$checkpoint" 1 -- --tta
}

# 场景9: 自定义参数测试
test_custom() {
    local checkpoint="${1:-}"
    
    if [[ -z "$checkpoint" ]]; then
        log_error "自定义测试需要指定checkpoint!"
        echo "用法: bash $(basename "$0") custom <checkpoint> [extra_args...]"
        return 1
    fi
    
    check_checkpoint "$checkpoint" || return 1
    
    shift
    local extra_args="$@"
    
    log_info "执行自定义参数测试..."
    log_info "额外参数: ${extra_args:-none}"
    
    bash mmdetection/tools/test_edl_point_mil.sh "$checkpoint" 1 -- ${extra_args}
}

# 主函数
main() {
    local scenario="${1:-}"
    local checkpoint="${2:-}"
    
    case "$scenario" in
        -h|--help)
            show_usage
            exit 0
            ;;
        -l|--list)
            list_checkpoints
            exit 0
            ;;
        1|basic)
            test_basic "$checkpoint"
            ;;
        2|quick)
            test_quick "$checkpoint"
            ;;
        3|full)
            test_full "$checkpoint"
            ;;
        4|distributed)
            test_distributed "$checkpoint" "${3:-2}"
            ;;
        5|vis|visualization)
            test_vis "$checkpoint"
            ;;
        6|strict)
            test_strict "$checkpoint"
            ;;
        7|loose)
            test_loose "$checkpoint"
            ;;
        8|ensemble|tta)
            test_ensemble "$checkpoint"
            ;;
        9|custom)
            shift
            test_custom "$@"
            ;;
        "")
            log_warn "未指定场景，默认执行基础测试"
            test_basic
            ;;
        *)
            log_error "未知的场景: $scenario"
            show_usage
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
