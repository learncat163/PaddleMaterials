#!/usr/bin/env python3
"""
DiffCSP 第一层输出对比脚本
============================
对比 PyTorch 和 Paddle 的第一层（node_embedding）输出差异

功能:
1. 加载 PyTorch 导出的第一层输出
2. 加载 Paddle 的逐层验证结果
3. 对比第一层输出的差异
4. 生成详细的对比报告

输出文件:
- tmp/first_layer_comparison.json  # 对比结果
- tmp/first_layer_comparison.md    # 人类可读报告

运行环境: ppmat (PaddlePaddle)

用法:
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/compare_first_layer.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# ── 路径配置 ───────────────────────────────────────────────────────────────────
PYTORCH_FIRST_LAYER = Path(PROJECT_ROOT) / "tmp" / "first_layer_output.json"
PADDLE_LAYERWISE = Path(PROJECT_ROOT) / "tmp" / "diffcsp_layerwise_validation.json"

# 输出路径
OUTPUT_DIR = Path(PROJECT_ROOT) / "tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_JSON = OUTPUT_DIR / "first_layer_comparison.json"
COMPARISON_MD = OUTPUT_DIR / "first_layer_comparison.md"


# ── 数据加载 ─────────────────────────────────────────────────────────────────────

def load_pytorch_first_layer(path: Path) -> Tuple[np.ndarray, Dict]:
    """加载 PyTorch 第一层输出"""
    logger.info(f"Loading PyTorch first layer output from: {path}")

    if not path.exists():
        raise FileNotFoundError(f"PyTorch first layer output not found: {path}")

    with open(path) as f:
        data = json.load(f)

    first_layer_output = np.array(data["first_layer_output"]["output"])
    metadata = data["metadata"]

    logger.info(f"  Shape: {first_layer_output.shape}")
    logger.info(f"  Mean: {first_layer_output.mean():.6f}")
    logger.info(f"  Std: {first_layer_output.std():.6f}")

    return first_layer_output, metadata


def load_paddle_layerwise(path: Path) -> Tuple[np.ndarray, Dict]:
    """加载 Paddle 逐层验证结果"""
    logger.info(f"Loading Paddle layerwise validation from: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Paddle layerwise validation not found: {path}")
        logger.warning("  Please run validate_layerwise.py first!")

    with open(path) as f:
        data = json.load(f)

    # 提取 node_embedding 的输出
    if "node_embedding" not in data["layer_outputs"]:
        raise ValueError("node_embedding not found in Paddle layerwise validation")

    node_embedding_info = data["layer_outputs"]["node_embedding"]

    # 注意：Paddle 的输出只有统计信息，没有完整数据
    # 我们需要重新加载或者从其他地方获取完整数据
    # 这里先提取统计信息
    mean = node_embedding_info["mean"]
    std = node_embedding_info["std"]
    shape = node_embedding_info["shape"]

    logger.info(f"  Shape: {shape}")
    logger.info(f"  Mean: {mean:.6f}")
    logger.info(f"  Std: {std:.6f}")

    return None, data  # 返回 None 表示没有完整数据


# ── 对比分析 ─────────────────────────────────────────────────────────────────────

def compare_first_layer(
    pytorch_output: np.ndarray,
    paddle_layerwise: Dict
) -> Dict[str, Any]:
    """对比第一层输出"""
    logger.info("\nComparing first layer outputs...")

    # 从 Paddle layerwise 中提取统计信息
    paddle_stats = paddle_layerwise["layer_outputs"]["node_embedding"]

    result = {
        "pytorch": {
            "shape": list(pytorch_output.shape),
            "mean": float(pytorch_output.mean()),
            "std": float(pytorch_output.std()),
            "min": float(pytorch_output.min()),
            "max": float(pytorch_output.max()),
            "has_nan": bool(np.isnan(pytorch_output).any()),
            "has_inf": bool(np.isinf(pytorch_output).any()),
        },
        "paddle": {
            "shape": paddle_stats["shape"],
            "mean": paddle_stats["mean"],
            "std": paddle_stats["std"],
        },
        "comparison": {
            "mean_diff": abs(float(pytorch_output.mean()) - paddle_stats["mean"]),
            "std_diff": abs(float(pytorch_output.std()) - paddle_stats["std"]),
        }
    }

    # 检查形状是否一致
    if result["pytorch"]["shape"] == result["paddle"]["shape"]:
        result["comparison"]["shape_match"] = True
    else:
        result["comparison"]["shape_match"] = False
        logger.warning(f"  Shape mismatch: PyTorch {result['pytorch']['shape']} vs Paddle {result['paddle']['shape']}")

    # 检查统计特性是否接近
    mean_diff_pct = (result["comparison"]["mean_diff"] / (abs(result["pytorch"]["mean"]) + 1e-8)) * 100
    std_diff_pct = (result["comparison"]["std_diff"] / (abs(result["pytorch"]["std"]) + 1e-8)) * 100

    result["comparison"]["mean_diff_pct"] = mean_diff_pct
    result["comparison"]["std_diff_pct"] = std_diff_pct

    logger.info(f"  Mean diff: {result['comparison']['mean_diff']:.6e} ({mean_diff_pct:.2f}%)")
    logger.info(f"  Std diff: {result['comparison']['std_diff']:.6e} ({std_diff_pct:.2f}%)")
    logger.info(f"  Shape match: {result['comparison']['shape_match']}")

    return result


# ── 报告生成 ─────────────────────────────────────────────────────────────────────

def generate_markdown_report(
    comparison: Dict[str, Any],
    pytorch_metadata: Dict,
    paddle_metadata: Dict
) -> str:
    """生成 Markdown 格式的报告"""
    lines = [
        "# DiffCSP 第一层输出对比报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 测试配置",
        "",
        f"| 项目 | PyTorch | Paddle |",
        f"|------|---------|--------|",
        f"| Batch Size | {pytorch_metadata['batch_size']} | {paddle_metadata['test_input']['batch_size']} |",
        f"| Total Atoms | {pytorch_metadata['total_atoms']} | {paddle_metadata['test_input']['total_atoms']} |",
        f"| Timestep | {pytorch_metadata.get('t_value', 'N/A')} | {paddle_metadata['test_input'].get('t_value', 'N/A')} |",
        "",
        "## 第一层 (node_embedding) 输出统计",
        "",
        "### PyTorch 输出",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| Shape | {comparison['pytorch']['shape']} |",
        f"| Mean | {comparison['pytorch']['mean']:.6f} |",
        f"| Std | {comparison['pytorch']['std']:.6f} |",
        f"| Min | {comparison['pytorch']['min']:.6f} |",
        f"| Max | {comparison['pytorch']['max']:.6f} |",
        f"| Has NaN | {'⚠️ YES' if comparison['pytorch']['has_nan'] else '✓ NO'} |",
        f"| Has Inf | {'⚠️ YES' if comparison['pytorch']['has_inf'] else '✓ NO'} |",
        "",
        "### Paddle 输出",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| Shape | {comparison['paddle']['shape']} |",
        f"| Mean | {comparison['paddle']['mean']:.6f} |",
        f"| Std | {comparison['paddle']['std']:.6f} |",
        "",
        "## 对比结果",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| Shape Match | {'✓ YES' if comparison['comparison']['shape_match'] else '✗ NO'} |",
        f"| Mean Diff | {comparison['comparison']['mean_diff']:.6e} |",
        f"| Mean Diff % | {comparison['comparison']['mean_diff_pct']:.4f}% |",
        f"| Std Diff | {comparison['comparison']['std_diff']:.6e} |",
        f"| Std Diff % | {comparison['comparison']['std_diff_pct']:.4f}% |",
        "",
    ]

    # 添加结论
    lines.extend([
        "## 结论",
        "",
    ])

    if comparison['comparison']['shape_match']:
        if comparison['comparison']['mean_diff_pct'] < 0.01 and comparison['comparison']['std_diff_pct'] < 0.01:
            lines.extend([
                "✅ **第一层输出一致**",
                "",
                "- 形状匹配 ✓",
                "- 均值差异 < 0.01% ✓",
                "- 标准差差异 < 0.01% ✓",
                "",
                "**PyTorch 和 Paddle 的第一层（node_embedding）输出高度一致！**",
                "",
            ])
        else:
            lines.extend([
                "⚠️ **第一层输出有差异**",
                "",
                "- 形状匹配 ✓",
                f"- 均值差异: {comparison['comparison']['mean_diff_pct']:.2f}%",
                f"- 标准差差异: {comparison['comparison']['std_diff_pct']:.2f}%",
                "",
                "**注意**: 由于 Paddle 逐层验证结果中只保存了统计信息，无法进行元素级对比。",
                "",
            ])
    else:
        lines.extend([
            "❌ **第一层输出不匹配**",
            "",
            "- 形状不匹配 ✗",
            "",
            "**请检查模型配置和数据准备！**",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 说明",
        "",
        "- **第一层 (node_embedding)**: 将原子类型（one-hot 向量）映射到隐藏维度",
        "- **smooth=True 模式**: 使用 `nn.Linear(max_atoms, hidden_dim)` 实现",
        "- **PyTorch**: 直接从 MatInvent checkpoint 运行",
        "- **Paddle**: 使用转换后的权重运行",
        "",
        "---",
        "",
        "*报告生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*"
    ])

    return "\n".join(lines)


def main():
    logger.info("=" * 60)
    logger.info("DiffCSP First Layer Output Comparison")
    logger.info("=" * 60)

    # ── 1. 加载 PyTorch 第一层输出 ───────────────────────────────────────────────
    logger.info("\n[1/3] Loading PyTorch first layer output...")
    try:
        pytorch_output, pytorch_metadata = load_pytorch_first_layer(PYTORCH_FIRST_LAYER)
    except FileNotFoundError as e:
        logger.error(f"  {e}")
        logger.error("  Please run export_first_layer.py in matinvent environment first!")
        sys.exit(1)

    # ── 2. 加载 Paddle 逐层验证结果 ───────────────────────────────────────────────────
    logger.info("\n[2/3] Loading Paddle layerwise validation...")
    try:
        _, paddle_layerwise = load_paddle_layerwise(PADDLE_LAYERWISE)
    except FileNotFoundError as e:
        logger.error(f"  {e}")
        logger.error("  Please run validate_layerwise.py first!")
        sys.exit(1)

    # ── 3. 对比第一层输出 ───────────────────────────────────────────────────────────
    logger.info("\n[3/3] Comparing first layer outputs...")
    comparison = compare_first_layer(pytorch_output, paddle_layerwise)

    # ── 4. 保存报告 ───────────────────────────────────────────────────────────────
    logger.info("\nSaving reports...")

    # JSON 报告
    report_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "pytorch_first_layer": str(PYTORCH_FIRST_LAYER),
            "paddle_layerwise": str(PADDLE_LAYERWISE),
        },
        "pytorch_metadata": pytorch_metadata,
        "paddle_metadata": {
            "test_input": paddle_layerwise["test_input"]
        },
        "comparison": comparison
    }

    with open(COMPARISON_JSON, 'w') as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"  JSON report saved to: {COMPARISON_JSON}")

    # Markdown 报告
    md_report = generate_markdown_report(
        comparison,
        pytorch_metadata,
        paddle_layerwise
    )
    with open(COMPARISON_MD, 'w') as f:
        f.write(md_report)
    logger.info(f"  Markdown report saved to: {COMPARISON_MD}")

    logger.info("\n" + "=" * 60)
    logger.info("Comparison completed!")
    logger.info("=" * 60)

    # 显示摘要
    print("\n" + "=" * 60)
    print("First Layer Comparison Summary:")
    print("=" * 60)
    print(f"  PyTorch shape: {comparison['pytorch']['shape']}")
    print(f"  Paddle shape: {comparison['paddle']['shape']}")
    print(f"  Shape match: {comparison['comparison']['shape_match']}")
    print(f"  Mean diff: {comparison['comparison']['mean_diff']:.6e} ({comparison['comparison']['mean_diff_pct']:.4f}%)")
    print(f"  Std diff: {comparison['comparison']['std_diff']:.6e} ({comparison['comparison']['std_diff_pct']:.4f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
