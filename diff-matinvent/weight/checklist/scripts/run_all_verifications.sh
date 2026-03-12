#!/bin/bash
# 完整验证脚本 - 运行所有验证检查
# 用法：
#   ./run_all_verifications.sh --fast           # 仅运行主验证
#   ./run_all_verifications.sh --skip-pytorch  # 跳过 PyTorch 生成
#   ./run_all_verifications.sh --check-only    # 仅运行对比
#   ./run_all_verifications.sh --all           # 运行所有验证

set -e

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DIFF_MATINVENT_ROOT="$PROJECT_ROOT/diff-matinvent"

# Python 解释器
PYTHON="${PYTHON:-$PROJECT_ROOT/../miniconda3/envs/ppmat/bin/python}"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python"
fi

# 默认参数
RUN_FAST=false
SKIP_PYTORCH=false
CHECK_ONLY=false
MODELS="all"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)
            RUN_FAST=true
            shift
            ;;
        --skip-pytorch)
            SKIP_PYTORCH=true
            shift
            ;;
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        --model)
            MODELS="$2"
            shift 2
            ;;
        --all)
            RUN_FAST=false
            SKIP_PYTORCH=false
            CHECK_ONLY=false
            MODELS="all"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 打印配置
echo "=================================="
echo "完整验证脚本"
echo "=================================="
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "PYTHON: $PYTHON"
echo "RUN_FAST: $RUN_FAST"
echo "SKIP_PYTORCH: $SKIP_PYTORCH"
echo "CHECK_ONLY: $CHECK_ONLY"
echo "MODELS: $MODELS"
echo "=================================="
echo ""

# 激活 conda 环境
if [ -f "$PROJECT_ROOT/../miniconda3/etc/profile.d/conda.sh" ]; then
    source "$PROJECT_ROOT/../miniconda3/etc/profile.d/conda.sh"
    conda activate ppmat || echo "Failed to activate ppmat environment"
fi

# 创建输出目录
mkdir -p "$DIFF_MATINVENT_ROOT/tmp/forward_logits"
mkdir -p "$DIFF_MATINVENT_ROOT/tmp/training_alignment"
mkdir -p "$DIFF_MATINVENT_ROOT/tmp/sampling_metrics"

# 记录开始时间
START_TIME=$(date +%s)

echo "开始时间: $(date)"
echo ""

# 验证 1: 前向 logits 验证
echo "=================================="
echo "验证 1: 前向 Logits 验证"
echo "=================================="

if [ "$CHECK_ONLY" = false ]; then
    cd "$SCRIPT_DIR"
    $PYTHON verify_forward_logits.py --model "$MODELS"
    echo ""
else
    echo "跳过（仅检查模式）"
    echo ""
fi

# 验证 2: 训练对齐验证
echo "=================================="
echo "验证 2: 训练对齐验证"
echo "=================================="

if [ "$CHECK_ONLY" = false ]; then
    cd "$SCRIPT_DIR"
    $PYTHON verify_training_alignment.py --model "$MODELS"
    echo ""
else
    echo "跳过（仅检查模式）"
    echo ""
fi

# 验证 3: 采样指标验证
echo "=================================="
echo "验证 3: 采样指标验证"
echo "=================================="

if [ "$CHECK_ONLY" = false ]; then
    cd "$SCRIPT_DIR"
    $PYTHON verify_sampling_metrics.py --model "$MODELS"
    echo ""
else
    echo "跳过（仅检查模式）"
    echo ""
fi

# 生成总体报告
echo "=================================="
echo "生成总体报告"
echo "=================================="

cd "$SCRIPT_DIR"
$PYTHON generate_final_report.py --model "$MODELS"
echo ""

# 计算耗时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "=================================="
echo "验证完成"
echo "=================================="
echo "结束时间: $(date)"
echo "耗时: ${MINUTES}分${SECONDS}秒"
echo ""
echo "报告位置："
echo "  - 前向 logits: $DIFF_MATINVENT_ROOT/tmp/forward_logits/"
echo "  - 训练对齐: $DIFF_MATINVENT_ROOT/tmp/training_alignment/"
echo "  - 采样指标: $DIFF_MATINVENT_ROOT/tmp/sampling_metrics/"
echo "  - 总体报告: $DIFF_MATINVENT_ROOT/tmp/compliance_report/"
echo "=================================="
