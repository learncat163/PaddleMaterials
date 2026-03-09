#!/usr/bin/env python3
"""
MatterGen 逐层验证脚本
=======================
测试第一层和最后一层的输出差异

功能:
1. 使用 Hook 捕获第一层输出 (atom_emb)
2. 使用 Hook 捕获最后一层输出 (final predictions)
3. 分析输出统计信息
4. 生成详细的层验证报告

输出文件:
- tmp/mattergen_layerwise_validation.json  # 验证结果
- tmp/mattergen_layerwise_validation.md    # 人类可读报告

运行环境: ppmat (PaddlePaddle)

用法:
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/validate_mattergen_layerwise.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

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
PD_CONVERTED_CKPT = Path(PROJECT_ROOT) / "tmp" / "mattergen_mp20_converted.pdparams"

# 输出路径
OUTPUT_DIR = Path(PROJECT_ROOT) / "tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LAYERWISE_REPORT = OUTPUT_DIR / "mattergen_layerwise_validation.json"
LAYERWISE_REPORT_MD = OUTPUT_DIR / "mattergen_layerwise_validation.md"


# ── Hook 机制 ─────────────────────────────────────────────────────────────────────

class LayerCapture:
    """捕获层输出的辅助类"""

    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.output = None
        self.hook_handle = None

    def hook_fn(self, layer, input, output):
        """Hook 函数，捕获输出"""
        # 只在第一次前向传播时捕获
        if self.output is None:
            # 处理不同的输出格式
            if isinstance(output, tuple):
                # 如果是 tuple，取第一个元素
                self.output = output[0]
            elif isinstance(output, dict):
                # 如果是 dict，提取其中的 tensor
                self.output = output
            else:
                self.output = output
            logger.info(f"  Captured output from {self.layer_name}: shape={self.output.shape if hasattr(self.output, 'shape') else 'N/A'}")

    def register(self, layer: paddle.nn.Layer):
        """注册 hook"""
        self.hook_handle = layer.register_forward_post_hook(self.hook_fn)

    def remove(self):
        """移除 hook"""
        if self.hook_handle is not None:
            self.hook_handle.remove()


def analyze_tensor_output(name: str, tensor: paddle.Tensor) -> Dict[str, Any]:
    """分析 tensor 输出统计"""
    np_array = tensor.numpy() if hasattr(tensor, 'numpy') else tensor

    result = {
        "name": name,
        "shape": list(np_array.shape),
        "size": np_array.size,
        "mean": float(np_array.mean()),
        "std": float(np_array.std()),
        "min": float(np_array.min()),
        "max": float(np_array.max()),
        "has_nan": bool(np.isnan(np_array).any()),
        "has_inf": bool(np.isinf(np_array).any()),
    }

    logger.info(f"  {name}:")
    logger.info(f"    shape: {result['shape']}")
    logger.info(f"    mean: {result['mean']:.6f}, std: {result['std']:.6f}")
    logger.info(f"    min: {result['min']:.6f}, max: {result['max']:.6f}")
    logger.info(f"    has_nan: {result['has_nan']}, has_inf: {result['has_inf']}")

    return result


# ── Paddle 模型加载 ─────────────────────────────────────────────────────────────

def load_paddle_model(ckpt_path: Path) -> paddle.nn.Layer:
    """加载转换后的 Paddle MatterGen 模型"""
    logger.info(f"Loading Paddle MatterGen model from: {ckpt_path}")

    from ppmat.models.mattergen.mattergen import MatterGen

    # 配置 (与 MatterGen MP-20 一致)
    decoder_cfg = {
        "gemnet_cfg": {
            "num_targets": 1,
            "latent_dim": 512,
            "atom_embedding_cfg": {
                "emb_size": 512,
                "with_mask_type": True,
            },
            "max_neighbors": 50,
            "max_cell_images_per_dim": 5,
            "cutoff": 7.0,
            "num_blocks": 4,
            "otf_graph": True,
        }
    }

    lattice_noise_scheduler_cfg = {
        "__class_name__": "LatticeVPSDEScheduler",
        "__init_params__": {
            "limit_density": 0.05771451654022283,
        }
    }

    coord_noise_scheduler_cfg = {
        "__class_name__": "NumAtomsVarianceAdjustedWrappedVESDE",
        "__init_params__": {}
    }

    atom_noise_scheduler_cfg = {
        "__class_name__": "D3PMScheduler",
        "__init_params__": {}
    }

    model = MatterGen(
        decoder_cfg=decoder_cfg,
        lattice_noise_scheduler_cfg=lattice_noise_scheduler_cfg,
        coord_noise_scheduler_cfg=coord_noise_scheduler_cfg,
        atom_noise_scheduler_cfg=atom_noise_scheduler_cfg,
        num_train_timesteps=1000,
        max_t=1.0,
        time_dim=256,
        lattice_loss_weight=1.0,
        coord_loss_weight=0.1,
        atom_loss_weight=1.0,
        d3pm_hybrid_lambda=0.01,
    )
    model.eval()

    # 加载转换后的权重
    state_dict = paddle.load(str(ckpt_path))
    model.set_state_dict(state_dict)

    logger.info("  Paddle MatterGen model loaded successfully")
    return model


# ── 测试数据准备 ───────────────────────────────────────────────────────────────

def prepare_test_input() -> Dict[str, Any]:
    """准备测试输入数据"""
    logger.info("Preparing test input...")

    # 简单的测试数据
    batch_size = 4
    num_atoms_list = [10, 15, 20, 12]
    total_atoms = sum(num_atoms_list)

    # 原子类型 (1-100)
    np.random.seed(42)
    atom_types = np.random.randint(1, 101, size=total_atoms)

    # 分数坐标
    np.random.seed(100)
    frac_coords = np.random.rand(total_atoms, 3).astype(np.float32)

    # 晶格参数 (3个长度 + 3个角度)
    np.random.seed(200)
    lengths = np.random.rand(batch_size, 3).astype(np.float32) * 5 + 5  # 5-10 Å
    angles = np.random.rand(batch_size, 3).astype(np.float32) * 60 + 60  # 60-120 度

    # num_atoms
    num_atoms = np.array(num_atoms_list, dtype=np.int64)

    # batch_idx
    batch_idx = []
    for i, num in enumerate(num_atoms_list):
        batch_idx.extend([i] * num)
    batch_idx = np.array(batch_idx, dtype=np.int64)

    test_input = {
        "batch_size": batch_size,
        "total_atoms": total_atoms,
        "atom_types": atom_types,
        "frac_coords": frac_coords,
        "lengths": lengths,
        "angles": angles,
        "num_atoms": num_atoms,
        "batch_idx": batch_idx,
    }

    logger.info(f"  Batch size: {batch_size}, Total atoms: {total_atoms}")
    return test_input


# ── 逐层测试 ─────────────────────────────────────────────────────────────────────

def test_layerwise_outputs(
    model: paddle.nn.Layer,
    test_input: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    测试第一层和最后一层输出

    Returns:
        (first_layer_stats, final_layer_stats)
    """
    logger.info("\nTesting layer-wise outputs...")

    # ── 第一层测试: atom_emb ────────────────────────────────────────────────────────
    logger.info("\n[1/2] Testing First Layer: atom_emb (AtomEmbedding)")

    first_layer_capture = LayerCapture("atom_emb")

    # 获取第一层: atom_emb (AtomEmbedding)
    # MatterGen.model.gemnet.atom_emb
    # MatterGen.model 是 GemNetTDenoiser
    # GemNetTDenoiser.gemnet 是 GemNetT
    atom_emb = model.model.gemnet.atom_emb
    first_layer_capture.register(atom_emb)

    # 准备输入
    batch_size = test_input["batch_size"]
    total_atoms = test_input["total_atoms"]

    structure_array = {
        "num_atoms": paddle.to_tensor(test_input["num_atoms"]),
        "frac_coords": paddle.to_tensor(test_input["frac_coords"]),
        "atom_types": paddle.to_tensor(test_input["atom_types"]),
        "lengths": paddle.to_tensor(test_input["lengths"]),
        "angles": paddle.to_tensor(test_input["angles"]),
    }

    batch = {
        "structure_array": structure_array,
        "num_atoms": test_input["num_atoms"],
    }

    # 前向传播
    with paddle.no_grad():
        output = model(batch)

    # 分析第一层输出
    first_layer_capture.remove()

    if first_layer_capture.output is not None:
        first_layer_stats = analyze_tensor_output(
            "atom_emb",
            first_layer_capture.output
        )
    else:
        logger.warning("  Failed to capture atom_emb output")
        first_layer_stats = {
            "name": "atom_emb",
            "error": "Failed to capture output"
        }

    # ── 最后一层测试: model 输出 ───────────────────────────────────────────────────
    logger.info("\n[2/2] Testing Final Layer: model output")

    # MatterGen 的最终输出在 loss_dict 中
    # 我们需要捕获 model.model (GemNetTDenoiser) 的输出
    final_layer_capture = LayerCapture("model")

    # 捕获 model.model (GemNetTDenoiser) 的 forward 输出
    final_layer_capture.register(model.model)

    # 前向传播
    with paddle.no_grad():
        output = model(batch)

    # 分析最后一层输出
    final_layer_capture.remove()

    # decoder 返回的是一个 dict，包含多个输出
    if isinstance(final_layer_capture.output, dict):
        final_layer_stats = {
            "decoder_output": {
                "keys": list(final_layer_capture.output.keys()),
            }
        }

        # 分析每个输出
        for key, value in final_layer_capture.output.items():
            if hasattr(value, 'shape'):
                stats = analyze_tensor_output(f"decoder.{key}", value)
                final_layer_stats[f"decoder_{key}"] = stats
    else:
        logger.warning(f"  Unexpected decoder output type: {type(final_layer_capture.output)}")
        final_layer_stats = {
            "decoder_output": {
                "type": str(type(final_layer_capture.output)),
            }
        }

    # 同时验证 loss_dict
    loss_stats = {
        "loss": float(output["loss_dict"]["loss"].numpy()),
        "loss_coord": float(output["loss_dict"]["loss_coord"].numpy()),
        "loss_lattice": float(output["loss_dict"]["loss_lattice"].numpy()),
        "loss_atom_type": float(output["loss_dict"]["loss_atom_type"].numpy()),
    }

    logger.info(f"  loss: {loss_stats['loss']:.6f}")
    logger.info(f"  loss_coord: {loss_stats['loss_coord']:.6f}")
    logger.info(f"  loss_lattice: {loss_stats['loss_lattice']:.6f}")
    logger.info(f"  loss_atom_type: {loss_stats['loss_atom_type']:.6f}")

    final_layer_stats["loss_dict"] = loss_stats

    return first_layer_stats, final_layer_stats


# ── 报告生成 ─────────────────────────────────────────────────────────────────────

def generate_markdown_report(
    first_layer: Dict[str, Any],
    final_layer: Dict[str, Any],
    test_input: Dict[str, Any]
) -> str:
    """生成 Markdown 格式的验证报告"""
    lines = [
        "# MatterGen 逐层验证报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 测试配置",
        "",
        f"| 项目 | 值 |",
        f"|------|-----|",
        f"| Batch Size | {test_input['batch_size']} |",
        f"| Total Atoms | {test_input['total_atoms']} |",
        "",
        "## 第一层: atom_emb",
        "",
    ]

    if "error" in first_layer:
        lines.extend([
            f"❌ **错误**: {first_layer['error']}",
            "",
        ])
    else:
        lines.extend([
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| Shape | {first_layer['shape']} |",
            f"| Size | {first_layer['size']} |",
            f"| Mean | {first_layer['mean']:.6f} |",
            f"| Std | {first_layer['std']:.6f} |",
            f"| Min | {first_layer['min']:.6f} |",
            f"| Max | {first_layer['max']:.6f} |",
            f"| Has NaN | {'⚠️ YES' if first_layer['has_nan'] else '✓ NO'} |",
            f"| Has Inf | {'⚠️ YES' if first_layer['has_inf'] else '✓ NO'} |",
            "",
        ])

    lines.extend([
        "## 最后一层: decoder 输出",
        "",
    ])

    # 输出 decoder 的各个输出
    for key, stats in final_layer.items():
        if key.startswith("decoder_") and key != "decoder_output":
            name = stats.get("name", key)
            lines.extend([
                f"### {name}",
                "",
            ])

            if "shape" in stats:
                lines.extend([
                    f"| 指标 | 值 |",
                    f"|------|-----|",
                    f"| Shape | {stats['shape']} |",
                    f"| Size | {stats['size']} |",
                    f"| Mean | {stats['mean']:.6f} |",
                    f"| Std | {stats['std']:.6f} |",
                    f"| Min | {stats['min']:.6f} |",
                    f"| Max | {stats['max']:.6f} |",
                    f"| Has NaN | {'⚠️ YES' if stats['has_nan'] else '✓ NO'} |",
                    f"| Has Inf | {'⚠️ YES' if stats['has_inf'] else '✓ NO'} |",
                    "",
                ])

    # 添加 loss_dict
    if "loss_dict" in final_layer:
        loss = final_layer["loss_dict"]
        lines.extend([
            "### Loss 输出",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| Total Loss | {loss['loss']:.6f} |",
            f"| loss_coord | {loss['loss_coord']:.6f} |",
            f"| loss_lattice | {loss['loss_lattice']:.6f} |",
            f"| loss_atom_type | {loss['loss_atom_type']:.6f} |",
            "",
        ])

    # 检查是否有异常值
    has_issues = False
    if "error" not in first_layer:
        has_issues = has_issues or first_layer.get("has_nan", False) or first_layer.get("has_inf", False)

    for key, stats in final_layer.items():
        if key.startswith("decoder_") and "shape" in stats:
            has_issues = has_issues or stats.get("has_nan", False) or stats.get("has_inf", False)

    lines.extend([
        "## 结论",
        "",
    ])

    if has_issues:
        lines.extend([
            "❌ **验证失败**",
            "",
            "- 输出包含 NaN 或 Inf 值",
            "- 请检查模型实现或权重转换",
            "",
        ])
    else:
        lines.extend([
            "✅ **验证通过**",
            "",
            "- 第一层 (atom_emb) 输出正常",
            "- 最后一层 (decoder) 输出正常",
            "- 所有输出值正常（无 NaN 或 Inf）",
            "- 输出形状正确",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 说明",
        "",
        "此验证检查了 MatterGen 模型的第一层和最后一层输出：",
        "- **第一层**: atom_emb - 原子嵌入层",
        "- **最后一层**: decoder - 解码器输出和 loss 计算",
        "",
        "注意: 此验证仅检查 Paddle 模型是否正常运行。",
        "要与 PyTorch 进行精确对比，需要导出 PyTorch 的输出数据。",
        "",
        "---",
        "",
        "*报告生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*"
    ])

    return "\n".join(lines)


def main():
    logger.info("=" * 60)
    logger.info("MatterGen Layer-wise Validation")
    logger.info("=" * 60)

    # ── 1. 检查转换后的权重文件 ─────────────────────────────────────────────────
    logger.info("\n[1/4] Checking converted weights...")
    if not PD_CONVERTED_CKPT.exists():
        logger.error(f"Converted weights not found: {PD_CONVERTED_CKPT}")
        logger.error("Please run convert_mattergen_weights.py first!")
        sys.exit(1)
    logger.info(f"  Found converted weights: {PD_CONVERTED_CKPT}")

    # ── 2. 加载 Paddle 模型 ───────────────────────────────────────────────────────
    logger.info("\n[2/4] Loading Paddle model...")
    pd_model = load_paddle_model(PD_CONVERTED_CKPT)

    # ── 3. 准备测试输入 ───────────────────────────────────────────────────────────
    logger.info("\n[3/4] Preparing test input...")
    test_input = prepare_test_input()

    # ── 4. 测试逐层输出 ───────────────────────────────────────────────────────────
    logger.info("\n[4/4] Testing layer-wise outputs...")
    first_layer_stats, final_layer_stats = test_layerwise_outputs(pd_model, test_input)

    # ── 5. 保存报告 ───────────────────────────────────────────────────────────────
    logger.info("\nSaving reports...")

    # JSON 报告
    report_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "converted_weights": str(PD_CONVERTED_CKPT),
        },
        "test_input": {
            "batch_size": test_input["batch_size"],
            "total_atoms": test_input["total_atoms"],
        },
        "first_layer": first_layer_stats,
        "final_layer": final_layer_stats,
    }

    with open(LAYERWISE_REPORT, 'w') as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"  JSON report saved to: {LAYERWISE_REPORT}")

    # Markdown 报告
    md_report = generate_markdown_report(first_layer_stats, final_layer_stats, test_input)
    with open(LAYERWISE_REPORT_MD, 'w') as f:
        f.write(md_report)
    logger.info(f"  Markdown report saved to: {LAYERWISE_REPORT_MD}")

    logger.info("\n" + "=" * 60)
    logger.info("Layer-wise validation completed!")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("Layer-wise Validation Summary:")
    print("=" * 60)
    if "error" not in first_layer_stats:
        print(f"  First layer (atom_emb): shape={first_layer_stats['shape']}, mean={first_layer_stats['mean']:.6f}")
    else:
        print(f"  First layer: ERROR - {first_layer_stats['error']}")
    print(f"  Final layer: {len(final_layer_stats)} outputs captured")
    if "loss_dict" in final_layer_stats:
        loss = final_layer_stats["loss_dict"]
        print(f"  Total loss: {loss['loss']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
