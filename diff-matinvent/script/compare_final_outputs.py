#!/usr/bin/env python3
"""
DiffCSP 最终输出层元素级对比脚本
================================
对比 PyTorch 和 Paddle 的最终输出层（pred_l, pred_x）差异

功能:
1. 加载 PyTorch 导出的最终输出
2. 运行 Paddle 模型并捕获最终输出
3. 进行元素级对比分析
4. 生成详细的对比报告

输出文件:
- tmp/final_outputs_elementwise_comparison.json  # 对比结果
- tmp/final_outputs_elementwise_comparison.md    # 人类可读报告

运行环境: ppmat (PaddlePaddle)

用法:
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/compare_final_outputs.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import paddle

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# ── 路径配置 ───────────────────────────────────────────────────────────────────
PYTORCH_OUTPUTS = Path(PROJECT_ROOT) / "tmp" / "first_layer_output.json"
PADDLE_CONVERTED_CKPT = Path(PROJECT_ROOT) / "tmp" / "diffcsp_matinvent_converted.pdparams"
EXPORTED_DATA = Path(PROJECT_ROOT) / "diff-matinvent" / "info" / "diffcsp" / "exported_noisy_inputs.json"

# 输出路径
OUTPUT_DIR = Path(PROJECT_ROOT) / "tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_JSON = OUTPUT_DIR / "final_outputs_elementwise_comparison.json"
COMPARISON_MD = OUTPUT_DIR / "final_outputs_elementwise_comparison.md"


# ── 数据加载 ─────────────────────────────────────────────────────────────────────

def load_pytorch_outputs(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """加载 PyTorch 最终输出"""
    logger.info(f"Loading PyTorch outputs from: {path}")

    if not path.exists():
        raise FileNotFoundError(f"PyTorch outputs not found: {path}")

    with open(path) as f:
        data = json.load(f)

    pred_l = np.array(data["final_outputs"]["pred_l"])
    pred_x = np.array(data["final_outputs"]["pred_x"])
    metadata = data["metadata"]

    logger.info(f"  pred_l shape: {pred_l.shape}, mean: {pred_l.mean():.6f}, std: {pred_l.std():.6f}")
    logger.info(f"  pred_x shape: {pred_x.shape}, mean: {pred_x.mean():.6f}, std: {pred_x.std():.6f}")

    return pred_l, pred_x, metadata


def load_exported_data(data_path: Path) -> Dict:
    """加载导出的输入数据"""
    logger.info(f"Loading exported data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Exported data not found: {data_path}")

    with open(data_path) as f:
        data = json.load(f)

    logger.info(f"  Exported data loaded successfully")
    return data


# ── Paddle 模型 ─────────────────────────────────────────────────────────────────────

def load_paddle_model(ckpt_path: Path) -> paddle.nn.Layer:
    """加载 Paddle 模型"""
    logger.info(f"Loading Paddle model from: {ckpt_path}")

    from ppmat.models.diffcsp.diffcsp import CSPNet

    # 构建 CSPNet（使用 smooth=True 匹配 PyTorch）
    decoder = CSPNet(
        hidden_dim=512,
        latent_dim=256,
        num_layers=6,
        act_fn="silu",
        dis_emb="sin",
        num_freqs=128,
        edge_style="fc",
        ln=True,
        ip=True,
        smooth=True,
        pred_type=False,
        prop_dim=512,
        pred_scalar=False,
        num_classes=100,
    )
    decoder.eval()

    # 加载转换后的权重
    state_dict = paddle.load(str(ckpt_path))
    decoder.set_state_dict(state_dict)

    logger.info("  Paddle model loaded successfully")
    return decoder


def run_paddle_forward(
    model: paddle.nn.Layer,
    test_input: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """运行 Paddle 前向传播"""
    logger.info("Running Paddle forward pass...")

    # 转换输入为 tensor
    time_emb = paddle.to_tensor(np.array(test_input["time_emb"], dtype=np.float32))
    atom_types = paddle.to_tensor(np.array(test_input["atom_type_probs"], dtype=np.float32))
    frac_coords = paddle.to_tensor(np.array(test_input["frac_coords"], dtype=np.float32))
    lattices = paddle.to_tensor(np.array(test_input["lattices"], dtype=np.float32))
    num_atoms = paddle.to_tensor(np.array(test_input["num_atoms"], dtype=np.int64))
    batch_idx = paddle.to_tensor(np.array(test_input["batch_idx"], dtype=np.int64))

    # 前向传播
    with paddle.no_grad():
        pred_l, pred_x = model(
            time_emb,
            atom_types,
            frac_coords,
            lattices,
            num_atoms,
            batch_idx
        )

    # 转换为 numpy
    pred_l_np = pred_l.numpy()
    pred_x_np = pred_x.numpy()

    logger.info(f"  pred_l shape: {pred_l_np.shape}, mean: {pred_l_np.mean():.6f}, std: {pred_l_np.std():.6f}")
    logger.info(f"  pred_x shape: {pred_x_np.shape}, mean: {pred_x_np.mean():.6f}, std: {pred_x_np.std():.6f}")

    return pred_l_np, pred_x_np


# ── 元素级对比 ─────────────────────────────────────────────────────────────────────

def compare_outputs_elementwise(
    pytorch_l: np.ndarray,
    paddle_l: np.ndarray,
    pytorch_x: np.ndarray,
    paddle_x: np.ndarray
) -> Dict[str, Any]:
    """元素级对比最终输出"""
    logger.info("\nPerforming element-wise comparison for final outputs...")

    result = {
        "pred_l": compare_single_output(pytorch_l, paddle_l, "pred_l"),
        "pred_x": compare_single_output(pytorch_x, paddle_x, "pred_x"),
    }

    # 计算总体统计
    result["summary"] = {
        "overall_pass_1e_4": result["pred_l"]["pass_1e_4"] and result["pred_x"]["pass_1e_4"],
        "overall_pass_1e_6": result["pred_l"]["pass_1e_6"] and result["pred_x"]["pass_1e_6"],
        "max_diff_all": max(result["pred_l"]["abs_diff"]["max"], result["pred_x"]["abs_diff"]["max"]),
    }

    logger.info(f"\n  pred_l max diff: {result['pred_l']['abs_diff']['max']:.4e}")
    logger.info(f"  pred_x max diff: {result['pred_x']['abs_diff']['max']:.4e}")
    logger.info(f"  Overall pass @ 1e-4: {result['summary']['overall_pass_1e_4']}")
    logger.info(f"  Overall pass @ 1e-6: {result['summary']['overall_pass_1e_6']}")

    return result


def compare_single_output(
    pytorch_output: np.ndarray,
    paddle_output: np.ndarray,
    name: str
) -> Dict[str, Any]:
    """对比单个输出"""
    logger.info(f"\n  Comparing {name}...")

    # 检查形状
    if pytorch_output.shape != paddle_output.shape:
        logger.error(f"    Shape mismatch: PyTorch {pytorch_output.shape} vs Paddle {paddle_output.shape}")
        return {
            "error": "Shape mismatch",
            "pytorch_shape": list(pytorch_output.shape),
            "paddle_shape": list(paddle_output.shape)
        }

    # 计算差异
    abs_diff = np.abs(pytorch_output - paddle_output)
    rel_diff = abs_diff / (np.abs(pytorch_output) + 1e-8)

    result = {
        "name": name,
        "shape": list(pytorch_output.shape),
        "total_elements": int(pytorch_output.size),
        "abs_diff": {
            "max": float(abs_diff.max()),
            "mean": float(abs_diff.mean()),
            "std": float(abs_diff.std()),
            "median": float(np.median(abs_diff)),
        },
        "rel_diff": {
            "max": float(rel_diff.max()),
            "mean": float(rel_diff.mean()),
            "median": float(np.median(rel_diff)),
        },
        "percentiles": {
            "p50": float(np.percentile(abs_diff, 50)),
            "p90": float(np.percentile(abs_diff, 90)),
            "p95": float(np.percentile(abs_diff, 95)),
            "p99": float(np.percentile(abs_diff, 99)),
            "p99.9": float(np.percentile(abs_diff, 99.9)),
        },
        "thresholds": {
            "lt_1e_4": int(np.sum(abs_diff < 1e-4)),
            "lt_1e_5": int(np.sum(abs_diff < 1e-5)),
            "lt_1e_6": int(np.sum(abs_diff < 1e-6)),
            "lt_1e_7": int(np.sum(abs_diff < 1e-7)),
            "lt_1e_8": int(np.sum(abs_diff < 1e-8)),
        }
    }

    # 计算百分比
    total = result["total_elements"]
    result["thresholds_pct"] = {
        "lt_1e_4": 100.0 * result["thresholds"]["lt_1e_4"] / total,
        "lt_1e_5": 100.0 * result["thresholds"]["lt_1e_5"] / total,
        "lt_1e_6": 100.0 * result["thresholds"]["lt_1e_6"] / total,
        "lt_1e_7": 100.0 * result["thresholds"]["lt_1e_7"] / total,
        "lt_1e_8": 100.0 * result["thresholds"]["lt_1e_8"] / total,
    }

    # 判断是否通过
    result["pass_1e_4"] = result["abs_diff"]["max"] < 1e-4
    result["pass_1e_6"] = result["abs_diff"]["max"] < 1e-6

    logger.info(f"    Max abs diff: {result['abs_diff']['max']:.4e}")
    logger.info(f"    Mean abs diff: {result['abs_diff']['mean']:.4e}")
    logger.info(f"    Elements < 1e-6: {result['thresholds_pct']['lt_1e_6']:.2f}%")

    return result


# ── 报告生成 ─────────────────────────────────────────────────────────────────────

def generate_markdown_report(
    pytorch_l: np.ndarray,
    paddle_l: np.ndarray,
    pytorch_x: np.ndarray,
    paddle_x: np.ndarray,
    comparison: Dict,
    metadata: Dict
) -> str:
    """生成 Markdown 格式的报告"""
    lines = [
        "# DiffCSP 最终输出层元素级对比报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 测试配置",
        "",
        f"| 项目 | 值 |",
        f"|------|-----|",
        f"| Batch Size | {metadata['batch_size']} |",
        f"| Total Atoms | {metadata['total_atoms']} |",
        f"| Timestep | {metadata.get('t_value', 'N/A')} |",
        "",
        "## pred_l (晶格预测) 对比",
        "",
    ]

    # 添加 pred_l 统计
    lines.extend([
        "### PyTorch pred_l",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| Shape | {list(pytorch_l.shape)} |",
        f"| Mean | {pytorch_l.mean():.6f} |",
        f"| Std | {pytorch_l.std():.6f} |",
        f"| Min | {pytorch_l.min():.6f} |",
        f"| Max | {pytorch_l.max():.6f} |",
        "",
        "### Paddle pred_l",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| Shape | {list(paddle_l.shape)} |",
        f"| Mean | {paddle_l.mean():.6f} |",
        f"| Std | {paddle_l.std():.6f} |",
        f"| Min | {paddle_l.min():.6f} |",
        f"| Max | {paddle_l.max():.6f} |",
        "",
        "### pred_l 元素级对比",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 总元素数 | {comparison['pred_l']['total_elements']:,} |",
        f"| **最大绝对差异** | **{comparison['pred_l']['abs_diff']['max']:.4e}** |",
        f"| 平均绝对差异 | {comparison['pred_l']['abs_diff']['mean']:.4e} |",
        f"| 中位数绝对差异 | {comparison['pred_l']['abs_diff']['median']:.4e} |",
        f"| < 1e-4 | {comparison['pred_l']['thresholds_pct']['lt_1e_4']:.2f}% |",
        f"| < 1e-6 | {comparison['pred_l']['thresholds_pct']['lt_1e_6']:.2f}% |",
        "",
    ])

    # 添加 pred_x 统计
    lines.extend([
        "## pred_x (坐标预测) 对比",
        "",
        "### PyTorch pred_x",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| Shape | {list(pytorch_x.shape)} |",
        f"| Mean | {pytorch_x.mean():.6f} |",
        f"| Std | {pytorch_x.std():.6f} |",
        f"| Min | {pytorch_x.min():.6f} |",
        f"| Max | {pytorch_x.max():.6f} |",
        "",
        "### Paddle pred_x",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| Shape | {list(paddle_x.shape)} |",
        f"| Mean | {paddle_x.mean():.6f} |",
        f"| Std | {paddle_x.std():.6f} |",
        f"| Min | {paddle_x.min():.6f} |",
        f"| Max | {paddle_x.max():.6f} |",
        "",
        "### pred_x 元素级对比",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 总元素数 | {comparison['pred_x']['total_elements']:,} |",
        f"| **最大绝对差异** | **{comparison['pred_x']['abs_diff']['max']:.4e}** |",
        f"| 平均绝对差异 | {comparison['pred_x']['abs_diff']['mean']:.4e} |",
        f"| 中位数绝对差异 | {comparison['pred_x']['abs_diff']['median']:.4e} |",
        f"| < 1e-4 | {comparison['pred_x']['thresholds_pct']['lt_1e_4']:.2f}% |",
        f"| < 1e-6 | {comparison['pred_x']['thresholds_pct']['lt_1e_6']:.2f}% |",
        "",
    ])

    # 添加总体结论
    lines.extend([
        "## 总体结论",
        "",
    ])

    if comparison['summary']['overall_pass_1e_6']:
        lines.extend([
            "✅ **最终输出高度一致**",
            "",
            f"- pred_l 最大差异: {comparison['pred_l']['abs_diff']['max']:.4e} < 1e-6 ✓",
            f"- pred_x 最大差异: {comparison['pred_x']['abs_diff']['max']:.4e} < 1e-6 ✓",
            f"- 总体最大差异: {comparison['summary']['max_diff_all']:.4e}",
            "",
            "**PyTorch 和 Paddle 的最终输出完全一致！权重转换成功！**",
            "",
        ])
    elif comparison['summary']['overall_pass_1e_4']:
        lines.extend([
            "⚠️ **最终输出基本一致**",
            "",
            f"- pred_l 最大差异: {comparison['pred_l']['abs_diff']['max']:.4e} < 1e-4 ✓",
            f"- pred_x 最大差异: {comparison['pred_x']['abs_diff']['max']:.4e} < 1e-4 ✓",
            f"- 总体最大差异: {comparison['summary']['max_diff_all']:.4e}",
            "",
            "**精度达到 1e-4 级别，基本满足要求。**",
            "",
        ])
    else:
        lines.extend([
            "❌ **最终输出差异过大**",
            "",
            f"- pred_l 最大差异: {comparison['pred_l']['abs_diff']['max']:.4e}",
            f"- pred_x 最大差异: {comparison['pred_x']['abs_diff']['max']:.4e}",
            "",
            "**请检查权重转换和模型实现！**",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 说明",
        "",
        "- **pred_l**: 晶格参数预测输出，shape=[batch_size, 3, 3]",
        "- **pred_x**: 原子坐标预测输出，shape=[total_atoms, 3]",
        "- **对比方法**: 使用完全相同的输入数据，分别运行 PyTorch 和 Paddle 模型",
        "- **精度标准**: max_diff < 1e-4 为合格，< 1e-6 为优秀",
        "",
        "---",
        "",
        "*报告生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*"
    ])

    return "\n".join(lines)


def main():
    logger.info("=" * 60)
    logger.info("DiffCSP Final Outputs Element-wise Comparison")
    logger.info("=" * 60)

    # ── 1. 加载 PyTorch 输出 ─────────────────────────────────────────────────────
    logger.info("\n[1/4] Loading PyTorch outputs...")
    try:
        pytorch_l, pytorch_x, pytorch_metadata = load_pytorch_outputs(PYTORCH_OUTPUTS)
    except FileNotFoundError as e:
        logger.error(f"  {e}")
        logger.error("  Please run export_first_layer.py in matinvent environment first!")
        sys.exit(1)

    # ── 2. 加载导出的输入数据 ─────────────────────────────────────────────────────
    logger.info("\n[2/4] Loading exported input data...")
    try:
        exported_data = load_exported_data(EXPORTED_DATA)
    except FileNotFoundError as e:
        logger.error(f"  {e}")
        sys.exit(1)

    # ── 3. 加载 Paddle 模型并运行前向传播 ─────────────────────────────────────────
    logger.info("\n[3/4] Loading Paddle model and running forward pass...")
    try:
        paddle_model = load_paddle_model(PADDLE_CONVERTED_CKPT)
    except FileNotFoundError as e:
        logger.error(f"  {e}")
        logger.error("  Please run convert_to_paddle.py first!")
        sys.exit(1)

    # 准备测试输入
    noised_input = exported_data["noised_input"]
    metadata = exported_data["metadata"]

    test_input = {
        "time_emb": noised_input["time_emb"],
        "atom_type_probs": noised_input["atom_type_probs"],
        "frac_coords": noised_input["input_frac_coords"],
        "lattices": noised_input["input_lattice"],
        "num_atoms": metadata["num_atoms_list"],
        "batch_idx": noised_input["batch"]
    }

    paddle_l, paddle_x = run_paddle_forward(paddle_model, test_input)

    # ── 4. 元素级对比 ─────────────────────────────────────────────────────────────
    logger.info("\n[4/4] Performing element-wise comparison...")
    comparison = compare_outputs_elementwise(pytorch_l, paddle_l, pytorch_x, paddle_x)

    # ── 5. 保存报告 ───────────────────────────────────────────────────────────────
    logger.info("\nSaving reports...")

    # JSON 报告
    report_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "pytorch_outputs": str(PYTORCH_OUTPUTS),
            "paddle_converted_ckpt": str(PADDLE_CONVERTED_CKPT),
            "exported_data": str(EXPORTED_DATA),
        },
        "pytorch_metadata": pytorch_metadata,
        "pytorch_stats": {
            "pred_l": {
                "shape": list(pytorch_l.shape),
                "mean": float(pytorch_l.mean()),
                "std": float(pytorch_l.std()),
                "min": float(pytorch_l.min()),
                "max": float(pytorch_l.max()),
            },
            "pred_x": {
                "shape": list(pytorch_x.shape),
                "mean": float(pytorch_x.mean()),
                "std": float(pytorch_x.std()),
                "min": float(pytorch_x.min()),
                "max": float(pytorch_x.max()),
            },
        },
        "paddle_stats": {
            "pred_l": {
                "shape": list(paddle_l.shape),
                "mean": float(paddle_l.mean()),
                "std": float(paddle_l.std()),
                "min": float(paddle_l.min()),
                "max": float(paddle_l.max()),
            },
            "pred_x": {
                "shape": list(paddle_x.shape),
                "mean": float(paddle_x.mean()),
                "std": float(paddle_x.std()),
                "min": float(paddle_x.min()),
                "max": float(paddle_x.max()),
            },
        },
        "comparison": comparison
    }

    with open(COMPARISON_JSON, 'w') as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"  JSON report saved to: {COMPARISON_JSON}")

    # Markdown 报告
    md_report = generate_markdown_report(
        pytorch_l, paddle_l, pytorch_x, paddle_x,
        comparison, pytorch_metadata
    )
    with open(COMPARISON_MD, 'w') as f:
        f.write(md_report)
    logger.info(f"  Markdown report saved to: {COMPARISON_MD}")

    logger.info("\n" + "=" * 60)
    logger.info("Element-wise comparison completed!")
    logger.info("=" * 60)

    # 显示摘要
    print("\n" + "=" * 60)
    print("Final Outputs Element-wise Comparison Summary:")
    print("=" * 60)
    print(f"\n  pred_l (晶格预测):")
    print(f"    Shape: {comparison['pred_l']['shape']}")
    print(f"    Elements: {comparison['pred_l']['total_elements']:,}")
    print(f"    Max diff: {comparison['pred_l']['abs_diff']['max']:.4e}")
    print(f"    Mean diff: {comparison['pred_l']['abs_diff']['mean']:.4e}")
    print(f"    < 1e-6: {comparison['pred_l']['thresholds_pct']['lt_1e_6']:.2f}%")
    print(f"\n  pred_x (坐标预测):")
    print(f"    Shape: {comparison['pred_x']['shape']}")
    print(f"    Elements: {comparison['pred_x']['total_elements']:,}")
    print(f"    Max diff: {comparison['pred_x']['abs_diff']['max']:.4e}")
    print(f"    Mean diff: {comparison['pred_x']['abs_diff']['mean']:.4e}")
    print(f"    < 1e-6: {comparison['pred_x']['thresholds_pct']['lt_1e_6']:.2f}%")
    print(f"\n  Overall:")
    print(f"    Max diff: {comparison['summary']['max_diff_all']:.4e}")
    print(f"    Pass @ 1e-4: {comparison['summary']['overall_pass_1e_4']}")
    print(f"    Pass @ 1e-6: {comparison['summary']['overall_pass_1e_6']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
