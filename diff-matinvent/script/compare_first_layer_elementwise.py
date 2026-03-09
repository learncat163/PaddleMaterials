#!/usr/bin/env python3
"""
DiffCSP 第一层元素级对比脚本
============================
直接运行 PyTorch 和 Paddle 模型，捕获完整的第一层输出进行元素级对比

功能:
1. 加载 PyTorch 导出的第一层完整输出
2. 运行 Paddle 模型并捕获第一层完整输出
3. 进行元素级对比分析
4. 生成详细的对比报告

输出文件:
- tmp/first_layer_elementwise_comparison.json  # 对比结果
- tmp/first_layer_elementwise_comparison.md    # 人类可读报告

运行环境: ppmat (PaddlePaddle)

用法:
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/compare_first_layer_elementwise.py
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
PYTORCH_FIRST_LAYER = Path(PROJECT_ROOT) / "tmp" / "first_layer_output.json"
PADDLE_CONVERTED_CKPT = Path(PROJECT_ROOT) / "tmp" / "diffcsp_matinvent_converted.pdparams"
EXPORTED_DATA = Path(PROJECT_ROOT) / "diff-matinvent" / "info" / "diffcsp" / "exported_noisy_inputs.json"

# 输出路径
OUTPUT_DIR = Path(PROJECT_ROOT) / "tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_JSON = OUTPUT_DIR / "first_layer_elementwise_comparison.json"
COMPARISON_MD = OUTPUT_DIR / "first_layer_elementwise_comparison.md"


# ── 数据加载 ─────────────────────────────────────────────────────────────────────

def load_pytorch_first_layer(path: Path) -> Tuple[np.ndarray, Dict]:
    """加载 PyTorch 第一层完整输出"""
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


def load_exported_data(data_path: Path) -> Dict:
    """加载导出的输入数据"""
    logger.info(f"Loading exported data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Exported data not found: {data_path}")

    with open(data_path) as f:
        data = json.load(f)

    logger.info(f"  Exported data loaded successfully")
    return data


# ── Paddle 模型和钩子 ─────────────────────────────────────────────────────────────

class FirstLayerHook:
    """捕获第一层输出的钩子"""

    def __init__(self):
        self.outputs = {}

    def hook(self, layer, input, output):
        """前向钩子，捕获输出"""
        # 保存输出
        if isinstance(output, paddle.Tensor):
            self.outputs['first_layer'] = output.detach().clone()
        elif isinstance(output, tuple):
            # 处理多个输出的情况
            for i, o in enumerate(output):
                if isinstance(o, paddle.Tensor):
                    self.outputs[f'first_layer_{i}'] = o.detach().clone()


def register_first_layer_hook(model):
    """在第一层注册钩子"""
    hook = FirstLayerHook()

    # 在 node_embedding 注册钩子
    model.node_embedding.register_forward_post_hook(hook.hook)

    return hook


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


def run_paddle_forward_with_hook(
    model: paddle.nn.Layer,
    test_input: Dict
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """运行 Paddle 前向传播并捕获第一层输出"""
    logger.info("Running Paddle forward pass with first layer hook...")

    # 注册钩子
    hook = register_first_layer_hook(model)

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
    first_layer_output = hook.outputs['first_layer'].numpy()
    pred_l_np = pred_l.numpy()
    pred_x_np = pred_x.numpy()

    logger.info(f"  First layer output shape: {first_layer_output.shape}")
    logger.info(f"  First layer output mean: {first_layer_output.mean():.6f}")
    logger.info(f"  First layer output std: {first_layer_output.std():.6f}")

    return first_layer_output, (pred_l_np, pred_x_np)


# ── 元素级对比 ─────────────────────────────────────────────────────────────────────

def compare_elementwise(
    pytorch_output: np.ndarray,
    paddle_output: np.ndarray
) -> Dict[str, Any]:
    """元素级对比"""
    logger.info("\nPerforming element-wise comparison...")

    # 检查形状
    if pytorch_output.shape != paddle_output.shape:
        logger.error(f"  Shape mismatch: PyTorch {pytorch_output.shape} vs Paddle {paddle_output.shape}")
        return {
            "error": "Shape mismatch",
            "pytorch_shape": list(pytorch_output.shape),
            "paddle_shape": list(paddle_output.shape)
        }

    # 计算差异
    abs_diff = np.abs(pytorch_output - paddle_output)
    rel_diff = abs_diff / (np.abs(pytorch_output) + 1e-8)

    result = {
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
            "std": float(rel_diff.std()),
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

    logger.info(f"  Max abs diff: {result['abs_diff']['max']:.4e}")
    logger.info(f"  Mean abs diff: {result['abs_diff']['mean']:.4e}")
    logger.info(f"  Median abs diff: {result['abs_diff']['median']:.4e}")
    logger.info(f"  Pass @ 1e-4: {result['pass_1e_4']}")
    logger.info(f"  Pass @ 1e-6: {result['pass_1e_6']}")
    logger.info(f"  Elements < 1e-6: {result['thresholds_pct']['lt_1e_6']:.2f}%")

    return result


# ── 报告生成 ─────────────────────────────────────────────────────────────────────

def generate_markdown_report(
    pytorch_output: np.ndarray,
    paddle_output: np.ndarray,
    elementwise_comparison: Dict,
    pytorch_metadata: Dict
) -> str:
    """生成 Markdown 格式的报告"""
    lines = [
        "# DiffCSP 第一层元素级对比报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 测试配置",
        "",
        f"| 项目 | 值 |",
        f"|------|-----|",
        f"| Batch Size | {pytorch_metadata['batch_size']} |",
        f"| Total Atoms | {pytorch_metadata['total_atoms']} |",
        f"| Timestep | {pytorch_metadata.get('t_value', 'N/A')} |",
        "",
        "## PyTorch 第一层输出统计",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| Shape | {list(pytorch_output.shape)} |",
        f"| Mean | {pytorch_output.mean():.6f} |",
        f"| Std | {pytorch_output.std():.6f} |",
        f"| Min | {pytorch_output.min():.6f} |",
        f"| Max | {pytorch_output.max():.6f} |",
        "",
        "## Paddle 第一层输出统计",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| Shape | {list(paddle_output.shape)} |",
        f"| Mean | {paddle_output.mean():.6f} |",
        f"| Std | {paddle_output.std():.6f} |",
        f"| Min | {paddle_output.min():.6f} |",
        f"| Max | {paddle_output.max():.6f} |",
        "",
        "## 元素级对比结果",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 总元素数 | {elementwise_comparison['total_elements']:,} |",
        f"| **最大绝对差异** | **{elementwise_comparison['abs_diff']['max']:.4e}** |",
        f"| 平均绝对差异 | {elementwise_comparison['abs_diff']['mean']:.4e} |",
        f"| 中位数绝对差异 | {elementwise_comparison['abs_diff']['median']:.4e} |",
        f"| 标准差 | {elementwise_comparison['abs_diff']['std']:.4e} |",
        "",
        "### 百分位数分析",
        "",
        f"| 百分位 | 绝对差异 |",
        f"|--------|----------|",
        f"| P50 | {elementwise_comparison['percentiles']['p50']:.4e} |",
        f"| P90 | {elementwise_comparison['percentiles']['p90']:.4e} |",
        f"| P95 | {elementwise_comparison['percentiles']['p95']:.4e} |",
        f"| P99 | {elementwise_comparison['percentiles']['p99']:.4e} |",
        f"| P99.9 | {elementwise_comparison['percentiles']['p99.9']:.4e} |",
        "",
        "### 阈值分布",
        "",
        f"| 阈值 | 元素数量 | 百分比 |",
        f"|------|----------|--------|",
        f"| < 1e-4 | {elementwise_comparison['thresholds']['lt_1e_4']:,} | {elementwise_comparison['thresholds_pct']['lt_1e_4']:.2f}% |",
        f"| < 1e-5 | {elementwise_comparison['thresholds']['lt_1e_5']:,} | {elementwise_comparison['thresholds_pct']['lt_1e_5']:.2f}% |",
        f"| < 1e-6 | {elementwise_comparison['thresholds']['lt_1e_6']:,} | {elementwise_comparison['thresholds_pct']['lt_1e_6']:.2f}% |",
        f"| < 1e-7 | {elementwise_comparison['thresholds']['lt_1e_7']:,} | {elementwise_comparison['thresholds_pct']['lt_1e_7']:.2f}% |",
        f"| < 1e-8 | {elementwise_comparison['thresholds']['lt_1e_8']:,} | {elementwise_comparison['thresholds_pct']['lt_1e_8']:.2f}% |",
        "",
    ]

    # 添加结论
    lines.extend([
        "## 结论",
        "",
    ])

    if elementwise_comparison.get("error"):
        lines.extend([
            f"❌ **对比失败**: {elementwise_comparison['error']}",
            "",
        ])
    elif elementwise_comparison['pass_1e_6']:
        lines.extend([
            "✅ **第一层输出高度一致**",
            "",
            f"- 最大绝对差异: {elementwise_comparison['abs_diff']['max']:.4e} < 1e-6 ✓",
            f"- 99.9% 的元素差异 < {elementwise_comparison['percentiles']['p99.9']:.4e} ✓",
            f"- {elementwise_comparison['thresholds_pct']['lt_1e_6']:.2f}% 的元素差异 < 1e-6 ✓",
            "",
            "**PyTorch 和 Paddle 的第一层（node_embedding）输出完全一致！**",
            "",
        ])
    elif elementwise_comparison['pass_1e_4']:
        lines.extend([
            "⚠️ **第一层输出基本一致**",
            "",
            f"- 最大绝对差异: {elementwise_comparison['abs_diff']['max']:.4e} < 1e-4 ✓",
            f"- {elementwise_comparison['thresholds_pct']['lt_1e_4']:.2f}% 的元素差异 < 1e-4 ✓",
            "",
            "**注意**: 精度达到 1e-4 级别，基本满足要求。建议检查是否有数值稳定性问题。",
            "",
        ])
    else:
        lines.extend([
            "❌ **第一层输出差异过大**",
            "",
            f"- 最大绝对差异: {elementwise_comparison['abs_diff']['max']:.4e}",
            f"- 仅 {elementwise_comparison['thresholds_pct']['lt_1e_4']:.2f}% 的元素差异 < 1e-4",
            "",
            "**请检查权重转换和模型实现！**",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 说明",
        "",
        "- **第一层 (node_embedding)**: 将原子类型（one-hot 向量）映射到隐藏维度",
        "- **smooth=True 模式**: 使用 `nn.Linear(max_atoms, hidden_dim)` 实现",
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
    logger.info("DiffCSP First Layer Element-wise Comparison")
    logger.info("=" * 60)

    # ── 1. 加载 PyTorch 第一层输出 ───────────────────────────────────────────────
    logger.info("\n[1/4] Loading PyTorch first layer output...")
    try:
        pytorch_output, pytorch_metadata = load_pytorch_first_layer(PYTORCH_FIRST_LAYER)
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

    paddle_output, _ = run_paddle_forward_with_hook(paddle_model, test_input)

    # ── 4. 元素级对比 ─────────────────────────────────────────────────────────────
    logger.info("\n[4/4] Performing element-wise comparison...")
    elementwise_comparison = compare_elementwise(pytorch_output, paddle_output)

    # ── 5. 保存报告 ───────────────────────────────────────────────────────────────
    logger.info("\nSaving reports...")

    # JSON 报告
    report_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "pytorch_first_layer": str(PYTORCH_FIRST_LAYER),
            "paddle_converted_ckpt": str(PADDLE_CONVERTED_CKPT),
            "exported_data": str(EXPORTED_DATA),
        },
        "pytorch_metadata": pytorch_metadata,
        "pytorch_stats": {
            "shape": list(pytorch_output.shape),
            "mean": float(pytorch_output.mean()),
            "std": float(pytorch_output.std()),
            "min": float(pytorch_output.min()),
            "max": float(pytorch_output.max()),
        },
        "paddle_stats": {
            "shape": list(paddle_output.shape),
            "mean": float(paddle_output.mean()),
            "std": float(paddle_output.std()),
            "min": float(paddle_output.min()),
            "max": float(paddle_output.max()),
        },
        "elementwise_comparison": elementwise_comparison
    }

    with open(COMPARISON_JSON, 'w') as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"  JSON report saved to: {COMPARISON_JSON}")

    # Markdown 报告
    md_report = generate_markdown_report(
        pytorch_output,
        paddle_output,
        elementwise_comparison,
        pytorch_metadata
    )
    with open(COMPARISON_MD, 'w') as f:
        f.write(md_report)
    logger.info(f"  Markdown report saved to: {COMPARISON_MD}")

    logger.info("\n" + "=" * 60)
    logger.info("Element-wise comparison completed!")
    logger.info("=" * 60)

    # 显示摘要
    print("\n" + "=" * 60)
    print("First Layer Element-wise Comparison Summary:")
    print("=" * 60)
    print(f"  Shape: {elementwise_comparison['shape']}")
    print(f"  Total elements: {elementwise_comparison['total_elements']:,}")
    print(f"  Max abs diff: {elementwise_comparison['abs_diff']['max']:.4e}")
    print(f"  Mean abs diff: {elementwise_comparison['abs_diff']['mean']:.4e}")
    print(f"  Median abs diff: {elementwise_comparison['abs_diff']['median']:.4e}")
    print(f"  P99.9: {elementwise_comparison['percentiles']['p99.9']:.4e}")
    print(f"  Pass @ 1e-4: {elementwise_comparison['pass_1e_4']}")
    print(f"  Pass @ 1e-6: {elementwise_comparison['pass_1e_6']}")
    print(f"  Elements < 1e-6: {elementwise_comparison['thresholds_pct']['lt_1e_6']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
