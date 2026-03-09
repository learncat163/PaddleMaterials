#!/usr/bin/env python3
"""
MatterGen PyTorch vs Paddle 权重详细对比脚本
============================================
详细对比 PyTorch 和 Paddle 的权重差异，作为输出准确性的基础验证

功能:
1. 加载 PyTorch checkpoint
2. 加载转换后的 Paddle checkpoint
3. 逐层对比权重差异
4. 生成详细的权重对比报告

说明:
由于 MatterGen 是一个训练模型（输出 loss_dict），完整的输出对比需要
完整的训练环境（噪声调度器、采样器等）。权重转换的准确性是输出准确性的
基础保证。此脚本详细验证权重转换的精度。

输出文件:
- tmp/mattergen_weight_comparison.json  # 对比结果
- tmp/mattergen_weight_comparison.md    # 人类可读报告

运行环境: ppmat (PaddlePaddle)

用法:
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/compare_mattergen_weights_detail.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import paddle
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# ── 路径配置 ───────────────────────────────────────────────────────────────────
PT_CKPT = Path("/home/cao/.cache/huggingface/hub/models--microsoft--mattergen/snapshots/ea430eab64b80855029c2941b9fda15f245a771a/checkpoints/mp_20_base/checkpoints/last.ckpt")
PD_CONVERTED_CKPT = Path(PROJECT_ROOT) / "tmp" / "mattergen_mp20_converted.pdparams"

# 输出路径
OUTPUT_DIR = Path(PROJECT_ROOT) / "tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_RESULT = OUTPUT_DIR / "mattergen_weight_comparison.json"
COMPARISON_REPORT = OUTPUT_DIR / "mattergen_weight_comparison.md"

# 前缀配置
PT_PREFIX = "diffusion_module.model."
PD_PREFIX = "model."


# ── 加载权重 ─────────────────────────────────────────────────────────────────────

def load_pytorch_weights(ckpt_path: Path) -> Dict[str, np.ndarray]:
    """加载 PyTorch checkpoint"""
    logger.info(f"Loading PyTorch checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    state_dict = ckpt['state_dict']

    result = {}
    for k, v in state_dict.items():
        # 去掉前缀
        stripped_k = k.replace(PT_PREFIX, '') if k.startswith(PT_PREFIX) else k
        result[stripped_k] = v.numpy() if hasattr(v, 'numpy') else np.array(v)

    logger.info(f"  Loaded {len(result)} PyTorch weights")
    return result


def load_paddle_weights(ckpt_path: Path) -> Dict[str, np.ndarray]:
    """加载 Paddle checkpoint"""
    logger.info(f"Loading Paddle checkpoint: {ckpt_path}")
    state_dict = paddle.load(str(ckpt_path))

    result = {}
    for k, v in state_dict.items():
        # 去掉前缀
        stripped_k = k.replace(PD_PREFIX, '') if k.startswith(PD_PREFIX) else k
        result[stripped_k] = v.numpy() if hasattr(v, 'numpy') else np.array(v)

    logger.info(f"  Loaded {len(result)} Paddle weights")
    return result


# ── 详细对比 ─────────────────────────────────────────────────────────────────────

def compare_weights_detailed(
    pt_weights: Dict[str, np.ndarray],
    pd_weights: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """详细对比 PyTorch 和 Paddle 权重"""
    logger.info("Comparing weights in detail...")

    comparison = {
        "total_keys": len(pt_weights),
        "compared_keys": 0,
        "missing_in_paddle": [],
        "layers": {},
        "summary": {
            "max_abs_diff": 0.0,
            "max_rel_diff": 0.0,
            "mean_abs_diff": 0.0,
            "layers_with_issues": [],
        }
    }

    # 按层分组
    layer_groups = {}
    for key in pt_weights.keys():
        # 提取层名 (例如: gemnet.int_blocks.0.edge_mlp.0.weight -> gemnet.int_blocks)
        parts = key.split('.')
        if len(parts) >= 2:
            layer_name = '.'.join(parts[:2])
        else:
            layer_name = 'root'

        if layer_name not in layer_groups:
            layer_groups[layer_name] = []
        layer_groups[layer_name].append(key)

    # 逐层对比
    for layer_name, keys in sorted(layer_groups.items()):
        logger.info(f"\n  Comparing layer: {layer_name}")

        layer_result = {
            "name": layer_name,
            "num_weights": len(keys),
            "weights": {},
            "max_diff": 0.0,
        }

        for key in keys:
            pt_val = pt_weights[key]
            pt_shape = list(pt_val.shape)

            if key not in pd_weights:
                logger.warning(f"    MISSING: {key}")
                comparison["missing_in_paddle"].append(key)
                continue

            pd_val = pd_weights[key]
            pd_shape = list(pd_val.shape)

            # 判断是否需要转置
            if len(pt_shape) == 2 and list(reversed(pt_shape)) == pd_shape:
                # 2D weight: 转置
                cmp = pt_val.T
                transposed = True
            elif pt_shape == pd_shape:
                # Same shape: direct copy
                cmp = pt_val
                transposed = False
            else:
                logger.error(f"    SHAPE MISMATCH: {key} PT={pt_shape}, PD={pd_shape}")
                layer_result["weights"][key] = {
                    "error": f"Shape mismatch: PT={pt_shape}, PD={pd_shape}",
                }
                comparison["summary"]["layers_with_issues"].append(key)
                continue

            # 计算差异
            abs_diff = np.abs(cmp - pd_val)
            max_diff = float(abs_diff.max())
            mean_diff = float(abs_diff.mean())

            # 相对差异
            rel_diff = abs_diff / (np.abs(cmp) + 1e-8)
            max_rel_diff = float(rel_diff.max())

            weight_result = {
                "shape": pt_shape,
                "transposed": transposed,
                "max_abs_diff": max_diff,
                "mean_abs_diff": mean_diff,
                "max_rel_diff": max_rel_diff,
            }

            # 更新统计
            comparison["summary"]["max_abs_diff"] = max(comparison["summary"]["max_abs_diff"], max_diff)
            comparison["summary"]["max_rel_diff"] = max(comparison["summary"]["max_rel_diff"], max_rel_diff)
            comparison["summary"]["mean_abs_diff"] += mean_diff
            layer_result["max_diff"] = max(layer_result["max_diff"], max_diff)

            if max_diff > 1e-4:
                comparison["summary"]["layers_with_issues"].append(key)

            logger.info(f"    {key}: max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}")
            layer_result["weights"][key] = weight_result
            comparison["compared_keys"] += 1

        # 计算层的平均差异
        if layer_result["weights"]:
            total_mean_diff = sum(w.get("mean_abs_diff", 0) for w in layer_result["weights"].values())
            layer_result["mean_diff"] = total_mean_diff / len(layer_result["weights"])

        comparison["layers"][layer_name] = layer_result

    # 计算总体平均差异
    if comparison["compared_keys"] > 0:
        comparison["summary"]["mean_abs_diff"] /= comparison["compared_keys"]

    return comparison


# ── 报告生成 ─────────────────────────────────────────────────────────────────────

def generate_markdown_report(comparison: Dict[str, Any]) -> str:
    """生成 Markdown 格式的对比报告"""
    lines = [
        "# MatterGen PyTorch vs Paddle 权重对比报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 概览",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 总权重数 | {comparison['total_keys']} |",
        f"| 已对比 | {comparison['compared_keys']} |",
        f"| 缺失 | {len(comparison['missing_in_paddle'])} |",
        f"| 最大绝对差异 | {comparison['summary']['max_abs_diff']:.4e} |",
        f"| 最大相对差异 | {comparison['summary']['max_rel_diff']:.4e} |",
        f"| 平均绝对差异 | {comparison['summary']['mean_abs_diff']:.4e} |",
        "",
        "## 关键层对比",
        "",
    ]

    # 选择几个关键层展示
    key_layers = ["gemnet.atom_emb", "gemnet.int_blocks", "gemnet.out_blocks", "noise_level_encoding"]

    for layer_name in key_layers:
        matching_layers = {k: v for k, v in comparison["layers"].items() if layer_name in k}
        if not matching_layers:
            continue

        for name, result in sorted(matching_layers.items()):
            lines.extend([
                f"### {name}",
                "",
                f"| 指标 | 值 |",
                f"|------|-----|",
                f"| 权重数 | {result['num_weights']} |",
                f"| 层最大差异 | {result['max_diff']:.4e} |",
                f"| 层平均差异 | {result.get('mean_diff', 0):.4e} |",
                "",
            ])

            # 显示前几个权重的详情
            count = 0
            for key, weight_result in result["weights"].items():
                if count >= 3:  # 只显示前 3 个
                    break
                if "error" in weight_result:
                    lines.extend([
                        f"**{key}**: ❌ {weight_result['error']}",
                        "",
                    ])
                else:
                    lines.extend([
                        f"**{key}**:",
                        f"- Shape: {weight_result['shape']}",
                        f"- Max Abs Diff: {weight_result['max_abs_diff']:.4e}",
                        f"- Mean Abs Diff: {weight_result['mean_abs_diff']:.4e}",
                        f"- Max Rel Diff: {weight_result['max_rel_diff']:.4e}",
                        f"- Transposed: {'Yes' if weight_result['transposed'] else 'No'}",
                        "",
                    ])
                count += 1

    # 完整层列表
    lines.extend([
        "## 所有层对比",
        "",
        f"| 层名 | 权重数 | 最大差异 | 平均差异 |",
        f"|------|--------|----------|----------|",
    ])

    for layer_name, result in sorted(comparison["layers"].items()):
        lines.append(
            f"| {layer_name} | {result['num_weights']} | {result['max_diff']:.4e} | {result.get('mean_diff', 0):.4e} |"
        )

    lines.append("")

    # 结论
    lines.extend([
        "## 结论",
        "",
    ])

    max_diff = comparison["summary"]["max_abs_diff"]

    if max_diff < 1e-6:
        lines.extend([
            "✅ **精度完美**",
            "",
            f"- 最大绝对差异: {max_diff:.4e} < 1e-6",
            "- 权重转换完全正确",
            "- 输出应该与 PyTorch 完全一致",
            "",
        ])
    elif max_diff < 1e-4:
        lines.extend([
            "✅ **精度优秀**",
            "",
            f"- 最大绝对差异: {max_diff:.4e} < 1e-4",
            "- 权重转换正确",
            "- 输出应该与 PyTorch 非常接近",
            "",
        ])
    elif max_diff < 1e-3:
        lines.extend([
            "⚠️ **精度良好**",
            "",
            f"- 最大绝对差异: {max_diff:.4e} < 1e-3",
            "- 权重转换基本正确",
            "- 可能存在轻微的数值差异",
            "",
        ])
    else:
        lines.extend([
            "❌ **精度不足**",
            "",
            f"- 最大绝对差异: {max_diff:.4e}",
            "- 请检查权重转换",
            "",
        ])

    # 缺失权重
    if comparison["missing_in_paddle"]:
        lines.extend([
            "## 缺失权重",
            "",
            f"Paddle 中缺失 {len(comparison['missing_in_paddle'])} 个权重:",
            "",
        ])
        for key in comparison["missing_in_paddle"][:10]:  # 只显示前 10 个
            lines.append(f"- {key}")
        if len(comparison["missing_in_paddle"]) > 10:
            lines.append(f"- ... 还有 {len(comparison['missing_in_paddle']) - 10} 个")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 说明",
        "",
        "此报告对比了 PyTorch 和 Paddle 的权重差异。",
        "",
        "**关于输出对比**:",
        "",
        "MatterGen 是一个训练模型，其输出是 loss_dict（包含 loss_coord,",
        "loss_lattice, loss_atom_type 等损失值），而不是直接的预测值。完整的",
        "输出对比需要：",
        "",
        "1. 完整的训练环境（噪声调度器、采样器等）",
        "2. 相同的随机种子",
        "3. 相同的噪声添加过程",
        "",
        "权重转换的准确性是输出准确性的基础保证。如果权重完全一致",
        "（max_diff < 1e-6），则在相同输入下，输出也应该完全一致。",
        "",
        "**权重转换规则**:",
        "",
        "- 2D 权重 (Linear): 转置 [out, in] → [in, out]",
        "- 1D 权重 (bias, LayerNorm): 直接复制",
        "- Embedding 权重: 直接复制",
        "",
        "---",
        "",
        "*报告生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*"
    ])

    return "\n".join(lines)


# ── 主函数 ───────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("MatterGen PyTorch vs Paddle Weight Comparison")
    logger.info("=" * 60)

    # ── 1. 加载权重 ───────────────────────────────────────────────────────────────
    logger.info("\n[1/3] Loading weights...")
    pt_weights = load_pytorch_weights(PT_CKPT)
    pd_weights = load_paddle_weights(PD_CONVERTED_CKPT)

    # ── 2. 详细对比 ───────────────────────────────────────────────────────────────
    logger.info("\n[2/3] Comparing weights...")
    comparison = compare_weights_detailed(pt_weights, pd_weights)

    # ── 3. 保存报告 ───────────────────────────────────────────────────────────────
    logger.info("\n[3/3] Saving reports...")

    # JSON 报告
    with open(COMPARISON_RESULT, 'w') as f:
        json.dump(comparison, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    logger.info(f"  JSON report saved to: {COMPARISON_RESULT}")

    # Markdown 报告
    md_report = generate_markdown_report(comparison)
    with open(COMPARISON_REPORT, 'w') as f:
        f.write(md_report)
    logger.info(f"  Markdown report saved to: {COMPARISON_REPORT}")

    # 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("Comparison Summary:")
    logger.info("=" * 60)
    logger.info(f"  Total weights: {comparison['total_keys']}")
    logger.info(f"  Compared: {comparison['compared_keys']}")
    logger.info(f"  Missing: {len(comparison['missing_in_paddle'])}")
    logger.info(f"  Max abs diff: {comparison['summary']['max_abs_diff']:.4e}")
    logger.info(f"  Mean abs diff: {comparison['summary']['mean_abs_diff']:.4e}")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("Weight Comparison Summary:")
    print("=" * 60)
    print(f"  Max absolute difference: {comparison['summary']['max_abs_diff']:.4e}")
    print(f"  Mean absolute difference: {comparison['summary']['mean_abs_diff']:.4e}")
    print(f"  Layers with issues: {len(comparison['summary']['layers_with_issues'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
