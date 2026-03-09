#!/usr/bin/env python3
"""
MatInvent DiffCSP 转换验证脚本
===============================
对比 PyTorch 和 Paddle 的前向传播输出

功能:
1. 准备相同的输入数据
2. 加载 PyTorch 模型并运行前向传播（如果可用）
3. 加载转换后的 Paddle 模型并运行前向传播
4. 对比输出结果（lattice_out, coord_out）
5. 生成详细的精度报告

输出文件:
- tmp/diffcsp_validation_report.json  # 验证结果
- tmp/diffcsp_validation_report.md    # 人类可读报告

运行环境: ppmat (PaddlePaddle)
         如果需要运行 PyTorch，需在 matinvent 环境下

用法:
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/validate_to_paddle.py
"""

import sys
import json
import logging
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any, Optional

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
PT_CKPT = Path("/home/cao/.cache/huggingface/hub/models--jwchen25--MatInvent/snapshots/b192d079a52d16403fcbbb26b73078d23f9e86c2/diffcsp_mp20/last.ckpt")
PD_CONVERTED_CKPT = Path(PROJECT_ROOT) / "tmp" / "diffcsp_matinvent_converted.pdparams"

# 导出的 PyTorch 数据（如果 PyTorch 不可用）
EXPORTED_DATA = Path(PROJECT_ROOT) / "diff-matinvent" / "info" / "diffcsp" / "exported_noisy_inputs.json"

# 输出路径
OUTPUT_DIR = Path(PROJECT_ROOT) / "tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_REPORT = OUTPUT_DIR / "diffcsp_validation_report.json"
VALIDATION_REPORT_MD = OUTPUT_DIR / "diffcsp_validation_report.md"

# ── 模型加载 ─────────────────────────────────────────────────────────────────────

def load_pytorch_model(ckpt_path: Path):
    """
    加载 PyTorch 模型（如果可用）
    """
    try:
        import torch
        from omegaconf import OmegaConf

        logger.info("PyTorch is available, loading model...")

        # 加载配置
        cfg_path = ckpt_path.parent / "hparams.yaml"
        if not cfg_path.exists():
            logger.warning(f"Config file not found: {cfg_path}")
            return None

        cfg = OmegaConf.load(cfg_path)
        cfg.model._target_ = "models.diffcsp.diffusion.DiffCSPModule"

        # 这里需要从 raw-matinvent 导入
        # 由于不能在 ppmat 环境下运行，我们使用已导出的数据
        logger.warning("PyTorch model loading requires matinvent environment")
        logger.info("Will use exported data if available")
        return None

    except ImportError:
        logger.info("PyTorch not available, will use exported data if available")
        return None


def load_paddle_model(ckpt_path: Path) -> paddle.nn.Layer:
    """
    加载转换后的 Paddle 模型
    """
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


# ── 输入数据准备 ───────────────────────────────────────────────────────────────

def load_exported_data(data_path: Path) -> Optional[Dict]:
    """
    加载已导出的 PyTorch 数据
    """
    if not data_path.exists():
        logger.warning(f"Exported data not found: {data_path}")
        return None

    logger.info(f"Loading exported data from: {data_path}")
    with open(data_path) as f:
        data = json.load(f)

    logger.info(f"  Exported data loaded successfully")
    return data


def prepare_test_input() -> Dict[str, Any]:
    """
    准备测试输入数据
    如果有已导出的数据就使用，否则生成简单的测试数据
    """
    # 首先尝试加载已导出的数据
    exported_data = load_exported_data(EXPORTED_DATA)
    if exported_data is not None and "noised_input" in exported_data:
        # 从导出的数据中提取输入
        noised_input = exported_data["noised_input"]
        metadata = exported_data["metadata"]

        # 构建测试输入字典
        test_input = {
            "metadata": {
                "batch_size": metadata["batch_size"],
                "total_atoms": metadata["total_atoms"],
                "num_atoms": metadata["num_atoms_list"],
                "t_value": metadata["t_value"]
            },
            "time_emb": noised_input["time_emb"],
            "atom_type_probs": noised_input["atom_type_probs"],
            "frac_coords": noised_input["input_frac_coords"],
            "lattices": noised_input["input_lattice"],
            "num_atoms": metadata["num_atoms_list"],
            "batch_idx": noised_input["batch"]
        }

        logger.info(f"  Using exported data: batch_size={metadata['batch_size']}, total_atoms={metadata['total_atoms']}")
        return test_input

    # 生成简单的测试数据
    logger.info("Generating simple test data...")

    batch_size = 4
    max_atoms = 20

    # 时间嵌入
    t = np.array([1000], dtype=np.int32)  # 使用 t=1000
    half_dim = 128
    embeddings = np.log(10000) / (half_dim - 1)
    freqs = np.exp(np.arange(half_dim) * -embeddings)
    emb = t[None, :] * freqs[None, :]
    time_emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1).astype(np.float32)
    time_emb = np.repeat(time_emb, batch_size, axis=0)

    # 原子类型 (smooth=True, 使用 one-hot 向量)
    num_atoms_list = [10, 15, 20, 12]
    total_atoms = sum(num_atoms_list)
    atom_types = np.zeros((total_atoms, 100), dtype=np.float32)
    for i, num in enumerate(num_atoms_list):
        start = sum(num_atoms_list[:i])
        end = start + num
        for j in range(start, end):
            atom_type = np.random.randint(0, 100)
            atom_types[j, atom_type] = 1.0

    # 分数坐标
    frac_coords = np.random.rand(total_atoms, 3).astype(np.float32)

    # 晶格
    lattices = np.random.randn(batch_size, 3, 3).astype(np.float32) * 5 + 10

    # 构建输入字典
    test_input = {
        "metadata": {
            "batch_size": batch_size,
            "total_atoms": total_atoms,
            "num_atoms": num_atoms_list,
            "t_value": 1000
        },
        "time_emb": time_emb.tolist(),
        "atom_type_probs": atom_types.tolist(),
        "frac_coords": frac_coords.tolist(),
        "lattices": lattices.tolist(),
        "num_atoms": num_atoms_list,
        "batch_idx": []
    }

    # 生成 batch_idx
    for i, num in enumerate(num_atoms_list):
        test_input["batch_idx"].extend([i] * num)

    logger.info(f"  Test data generated: batch_size={batch_size}, total_atoms={total_atoms}")
    return test_input


# ── 前向传播 ─────────────────────────────────────────────────────────────────────

def run_paddle_forward(
    model: paddle.nn.Layer,
    test_input: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    运行 Paddle 模型前向传播
    """
    logger.info("Running Paddle forward pass...")

    # 准备输入 tensors
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

    pred_l = pred_l.numpy()
    pred_x = pred_x.numpy()

    logger.info(f"  Paddle output shapes: pred_l={pred_l.shape}, pred_x={pred_x.shape}")
    return pred_l, pred_x


# ── 验证对比 ─────────────────────────────────────────────────────────────────────

def compare_outputs(
    paddle_l: np.ndarray,
    paddle_x: np.ndarray,
    pytorch_l: Optional[np.ndarray] = None,
    pytorch_x: Optional[np.ndarray] = None,
    exported_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    对比 PyTorch 和 Paddle 的输出
    """
    logger.info("\nComparing outputs...")

    result = {
        "paddle_outputs": {
            "pred_l_mean": float(paddle_l.mean()),
            "pred_l_std": float(paddle_l.std()),
            "pred_x_mean": float(paddle_x.mean()),
            "pred_x_std": float(paddle_x.std()),
            "pred_l_shape": list(paddle_l.shape),
            "pred_x_shape": list(paddle_x.shape),
        },
        "pytorch_outputs": None,
        "comparison": None
    }

    # 如果有 PyTorch 输出，进行对比
    if pytorch_l is not None and pytorch_x is not None:
        result["pytorch_outputs"] = {
            "pred_l_mean": float(pytorch_l.mean()),
            "pred_l_std": float(pytorch_l.std()),
            "pred_x_mean": float(pytorch_x.mean()),
            "pred_x_std": float(pytorch_x.std()),
        }

        # 计算差异
        l_diff = np.abs(paddle_l - pytorch_l)
        x_diff = np.abs(paddle_x - pytorch_x)

        result["comparison"] = {
            "pred_l_max_diff": float(l_diff.max()),
            "pred_l_mean_diff": float(l_diff.mean()),
            "pred_x_max_diff": float(x_diff.max()),
            "pred_x_mean_diff": float(x_diff.mean()),
            "pred_l_pass": bool(l_diff.max() < 1e-4),
            "pred_x_pass": bool(x_diff.max() < 1e-4),
            "overall_pass": bool(l_diff.max() < 1e-4 and x_diff.max() < 1e-4),
        }

        logger.info(f"  pred_l max_diff = {l_diff.max():.4e} (target < 1e-4)")
        logger.info(f"  pred_x max_diff = {x_diff.max():.4e} (target < 1e-4)")
        logger.info(f"  pred_l PASS: {result['comparison']['pred_l_pass']}")
        logger.info(f"  pred_x PASS: {result['comparison']['pred_x_pass']}")
        logger.info(f"  OVERALL: {'PASS' if result['comparison']['overall_pass'] else 'FAIL'}")
    else:
        logger.info("  No PyTorch output for comparison (this is expected if PyTorch is not available)")
        result["comparison"] = {
            "note": "PyTorch output not available for comparison"
        }

    return result


# ── 报告生成 ─────────────────────────────────────────────────────────────────────

def generate_markdown_report(
    comparison_result: Dict[str, Any],
    test_input: Dict[str, Any]
) -> str:
    """生成 Markdown 格式的验证报告"""
    lines = [
        "# DiffCSP 转换验证报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 测试配置",
        "",
        f"| 项目 | 值 |",
        f"|------|-----|",
        f"| Batch Size | {test_input['metadata']['batch_size']} |",
        f"| Total Atoms | {test_input['metadata']['total_atoms']} |",
        f"| Timestep | {test_input['metadata']['t_value']} |",
        "",
        "## Paddle 输出统计",
        "",
        f"| 指标 | pred_l | pred_x |",
        f"|------|--------|--------|",
        f"| Mean | {comparison_result['paddle_outputs']['pred_l_mean']:.6f} | {comparison_result['paddle_outputs']['pred_x_mean']:.6f} |",
        f"| Std | {comparison_result['paddle_outputs']['pred_l_std']:.6f} | {comparison_result['paddle_outputs']['pred_x_std']:.6f} |",
        f"| Shape | {comparison_result['paddle_outputs']['pred_l_shape']} | {comparison_result['paddle_outputs']['pred_x_shape']} |",
        "",
    ]

    if comparison_result["pytorch_outputs"] is not None:
        lines.extend([
            "## PyTorch 输出统计",
            "",
            f"| 指标 | pred_l | pred_x |",
            f"|------|--------|--------|",
            f"| Mean | {comparison_result['pytorch_outputs']['pred_l_mean']:.6f} | {comparison_result['pytorch_outputs']['pred_x_mean']:.6f} |",
            f"| Std | {comparison_result['pytorch_outputs']['pred_l_std']:.6f} | {comparison_result['pytorch_outputs']['pred_x_std']:.6f} |",
            "",
            "## 对比结果",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| pred_l max_diff | {comparison_result['comparison']['pred_l_max_diff']:.4e} |",
            f"| pred_l mean_diff | {comparison_result['comparison']['pred_l_mean_diff']:.4e} |",
            f"| pred_x max_diff | {comparison_result['comparison']['pred_x_max_diff']:.4e} |",
            f"| pred_x mean_diff | {comparison_result['comparison']['pred_x_mean_diff']:.4e} |",
            f"| **pred_l PASS** | {'✓' if comparison_result['comparison']['pred_l_pass'] else '✗'} |",
            f"| **pred_x PASS** | {'✓' if comparison_result['comparison']['pred_x_pass'] else '✗'} |",
            f"| **OVERALL** | {'✓ PASS' if comparison_result['comparison']['overall_pass'] else '✗ FAIL'} |",
            "",
        ])
    else:
        lines.extend([
            "## 对比结果",
            "",
            "**注意**: PyTorch 输出不可用（需要在 matinvent 环境下运行 PyTorch）",
            "",
            "要获得完整的对比结果，请：",
            "1. 在 matinvent 环境下导出 PyTorch 模型输出",
            "2. 或者在 matinvent 环境下运行此脚本",
            "",
        ])

    lines.extend([
        "## 结论",
        "",
    ])

    if comparison_result["comparison"] is not None and comparison_result["comparison"].get("overall_pass") is not None:
        if comparison_result["comparison"]["overall_pass"]:
            lines.extend([
                "✅ **验证通过**",
                "",
                "Paddle 模型输出与 PyTorch 模型输出一致（max_diff < 1e-4）",
                "权重转换成功！",
                "",
            ])
        else:
            lines.extend([
                "❌ **验证失败**",
                "",
                "Paddle 模型输出与 PyTorch 模型输出差异过大",
                "请检查权重转换过程",
                "",
            ])
    else:
        lines.extend([
            "📝 **验证状态**",
            "",
            "Paddle 模型成功运行前向传播",
            "由于 PyTorch 不可用，无法进行完整对比",
            "",
        ])

    lines.extend([
        "---",
        "",
        "*报告生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*"
    ])

    return "\n".join(lines)


def main():
    logger.info("=" * 60)
    logger.info("MatInvent DiffCSP Conversion Validation")
    logger.info("=" * 60)

    # ── 1. 检查转换后的权重文件 ─────────────────────────────────────────────────
    logger.info("\n[1/5] Checking converted weights...")
    if not PD_CONVERTED_CKPT.exists():
        logger.error(f"Converted weights not found: {PD_CONVERTED_CKPT}")
        logger.error("Please run convert_to_paddle.py first!")
        sys.exit(1)
    logger.info(f"  Found converted weights: {PD_CONVERTED_CKPT}")

    # ── 2. 加载 Paddle 模型 ───────────────────────────────────────────────────────
    logger.info("\n[2/5] Loading Paddle model...")
    pd_model = load_paddle_model(PD_CONVERTED_CKPT)

    # ── 3. 准备测试输入 ───────────────────────────────────────────────────────────
    logger.info("\n[3/5] Preparing test input...")
    test_input = prepare_test_input()

    # ── 4. 运行 Paddle 前向传播 ───────────────────────────────────────────────────
    logger.info("\n[4/5] Running Paddle forward pass...")
    paddle_l, paddle_x = run_paddle_forward(pd_model, test_input)

    # ── 5. 对比结果 ───────────────────────────────────────────────────────────────
    logger.info("\n[5/5] Comparing results...")

    # 尝试从导出的数据中获取 PyTorch 输出
    pytorch_l = None
    pytorch_x = None
    exported_data = load_exported_data(EXPORTED_DATA)

    if exported_data is not None and "model_outputs" in exported_data:
        pytorch_outputs = exported_data["model_outputs"]
        pytorch_l = np.array(pytorch_outputs.get("pred_l"))
        pytorch_x = np.array(pytorch_outputs.get("pred_x"))

        if pytorch_l is not None and pytorch_x is not None and pytorch_l.size > 0 and pytorch_x.size > 0:
            logger.info("  Using PyTorch outputs from exported data")

    comparison_result = compare_outputs(paddle_l, paddle_x, pytorch_l, pytorch_x, exported_data)

    # ── 6. 保存报告 ───────────────────────────────────────────────────────────────
    logger.info("\nSaving reports...")

    # JSON 报告
    report_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "converted_weights": str(PD_CONVERTED_CKPT),
            "exported_data": str(EXPORTED_DATA) if EXPORTED_DATA.exists() else "Not available"
        },
        "test_input": {
            "batch_size": test_input["metadata"]["batch_size"],
            "total_atoms": test_input["metadata"]["total_atoms"],
            "t_value": test_input["metadata"]["t_value"]
        },
        **comparison_result
    }

    with open(VALIDATION_REPORT, 'w') as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"  JSON report saved to: {VALIDATION_REPORT}")

    # Markdown 报告
    md_report = generate_markdown_report(comparison_result, test_input)
    with open(VALIDATION_REPORT_MD, 'w') as f:
        f.write(md_report)
    logger.info(f"  Markdown report saved to: {VALIDATION_REPORT_MD}")

    logger.info("\n" + "=" * 60)
    logger.info("Validation completed!")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("Validation Summary:")
    print("=" * 60)
    print(f"  Paddle output shape: pred_l={paddle_l.shape}, pred_x={paddle_x.shape}")
    if comparison_result["comparison"] is not None and comparison_result["comparison"].get("overall_pass") is not None:
        status = "✓ PASS" if comparison_result["comparison"]["overall_pass"] else "✗ FAIL"
        print(f"  Overall status: {status}")
    else:
        print(f"  Status: Paddle model ran successfully (no PyTorch comparison)")
    print("=" * 60)


if __name__ == "__main__":
    main()
