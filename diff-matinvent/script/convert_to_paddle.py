#!/usr/bin/env python3
"""
MatInvent DiffCSP 权重转换脚本
===============================
将 PyTorch checkpoint 转换为 Paddle pdparams 格式

功能:
1. 加载 PyTorch checkpoint（从 MatInvent）
2. 构建 Paddle 模型（使用 smooth=True 匹配 PyTorch）
3. 根据 mapping 报告进行权重转换
4. 处理特殊情况（smooth 模式、prop_mlp 等）
5. 保存转换后的 Paddle 权重

输出文件:
- tmp/diffcsp_matinvent_converted.pdparams  # 转换后的权重
- tmp/diffcsp_conversion_log.json          # 转换日志

运行环境: ppmat (PaddlePaddle)

用法:
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/convert_to_paddle.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any

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
# PyTorch 源路径
PT_CKPT = Path("/home/cao/.cache/huggingface/hub/models--jwchen25--MatInvent/snapshots/b192d079a52d16403fcbbb26b73078d23f9e86c2/diffcsp_mp20/last.ckpt")

# Paddle 参考路径（用于获取结构）
PD_CFG = "structure_generation/configs/diffcsp/diffcsp_mp20.yaml"

# 输出路径
OUTPUT_DIR = Path(PROJECT_ROOT) / "tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CKPT = OUTPUT_DIR / "diffcsp_matinvent_converted.pdparams"
CONVERSION_LOG = OUTPUT_DIR / "diffcsp_conversion_log.json"

# ── PyTorch checkpoint 加载 ─────────────────────────────────────────────────────

def load_pytorch_weights(ckpt_path: Path) -> Dict[str, np.ndarray]:
    """
    加载 PyTorch checkpoint 权重。
    尝试使用 torch，如果不可用则使用预导出的 NPZ 文件。
    """
    logger.info(f"Loading PyTorch weights from: {ckpt_path}")

    # 首先尝试使用 torch 直接加载
    try:
        import torch
        ckpt = torch.load(str(ckpt_path), map_location='cpu')
        sd = ckpt['state_dict']
        result = {}
        for k, v in sd.items():
            result[k] = v.numpy() if hasattr(v, 'numpy') else np.array(v)
        logger.info(f"  Loaded {len(result)} PyTorch weights using torch")
        return result
    except ImportError:
        logger.warning("torch not available, trying NPZ fallback...")

    # 备用方案：从预导出的 NPZ 文件加载
    npz_path = OUTPUT_DIR / "diffcsp_pytorch_all_weights.npz"
    if npz_path.exists():
        logger.info(f"Loading from pre-exported NPZ: {npz_path}")
        data = np.load(npz_path)
        result = {}
        for k in data.files:
            original_key = k.replace('__DOT__', '.')
            result[original_key] = data[k]
        logger.info(f"  Loaded {len(result)} weights from NPZ")
        return result

    raise RuntimeError(
        "Cannot load PyTorch checkpoint: torch not available and NPZ not found.\n"
        f"Please either:\n"
        f"  1. Run this script in matinvent conda environment (with torch)\n"
        f"  2. Or first export PyTorch weights to NPZ format"
    )


def build_paddle_model_smooth_true() -> paddle.nn.Layer:
    """
    构建 Paddle DiffCSP 模型，使用 smooth=True 匹配 PyTorch 架构
    """
    logger.info("Building Paddle model with smooth=True to match PyTorch...")

    from ppmat.models.diffcsp.diffcsp import CSPNet

    # 手动构建 CSPNet，指定 smooth=True
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
        smooth=True,   # 关键：使用 Linear 而不是 Embedding
        pred_type=False,
        prop_dim=512,
        pred_scalar=False,
        num_classes=100,
    )
    decoder.eval()

    logger.info("  Paddle model built with smooth=True")
    return decoder


# ── 权重转换 ────────────────────────────────────────────────────────────────────

def convert_weights(
    pt_weights: Dict[str, np.ndarray],
    pd_model: paddle.nn.Layer
) -> Tuple[Dict[str, paddle.Tensor], Dict[str, Any]]:
    """
    转换 PyTorch 权重到 Paddle 格式

    转换规则:
    - Linear.weight: [out, in] → [in, out] (转置)
    - Linear.bias: 直接复制
    - LayerNorm.*: 直接复制
    - Embedding.weight: 直接复制
    - 跳过 PyTorch 独有层（scheduler, type_out）
    - 保留 Paddle 独有层（prop_mlp）

    返回:
        converted_weights: 转换后的 Paddle 权重
        conversion_log: 转换日志
    """
    converted_weights = {}
    conversion_log = {
        "timestamp": datetime.now().isoformat(),
        "pytorch_checkpoint": str(PT_CKPT),
        "paddle_config": PD_CFG,
        "converted": [],
        "skipped": [],
        "copied_from_paddle": [],
        "errors": []
    }

    # 获取 Paddle 模型的 state_dict
    pd_state = pd_model.state_dict()
    pd_keys_set = set(pd_state.keys())

    # 由于单独构建的 CSPNet 没有 'decoder.' 前缀，需要添加前缀来匹配 PyTorch
    # 创建带前缀的键集合
    pd_keys_with_prefix = {f'decoder.{k}': k for k in pd_keys_set}

    # 需要跳过的 PyTorch 前缀
    skip_prefixes = [
        'beta_scheduler.',
        'sigma_scheduler.',
        'type_sigma_scheduler.',
    ]

    # 需要跳过的 PyTorch 键（Paddle 没有对应层）
    skip_keys = {
        'decoder.type_out.weight',  # pred_type=False
        'decoder.type_out.bias',
    }

    # Paddle 独有的键（不需要从 PyTorch 转换）
    paddle_only_patterns = ['prop_mlp']

    converted_count = 0
    skipped_count = 0
    paddle_only_count = 0
    error_count = 0

    for pt_key, pt_val in sorted(pt_weights.items()):
        # 检查是否需要跳过
        if any(pt_key.startswith(p) for p in skip_prefixes):
            conversion_log["skipped"].append({
                "key": pt_key,
                "reason": "scheduler parameter (not needed in Paddle)"
            })
            skipped_count += 1
            continue

        if pt_key in skip_keys:
            conversion_log["skipped"].append({
                "key": pt_key,
                "reason": f"Paddle pred_type=False, skipping {pt_key}"
            })
            skipped_count += 1
            continue

        # 直接匹配 PyTorch 键和 Paddle 键
        # PyTorch 键有 'decoder.' 前缀，需要去掉前缀来匹配 Paddle 的键
        pd_key_no_prefix = pt_key.replace('decoder.', '') if pt_key.startswith('decoder.') else pt_key

        if pd_key_no_prefix in pd_keys_set:
            pd_val = pd_state[pd_key_no_prefix]
            pd_shape = tuple(pd_val.shape)
            pt_shape = pt_val.shape

            # 检查是否需要转置（Linear 权重）
            if pt_key.endswith('.weight') and len(pt_shape) == 2:
                # Linear 权重需要转置: PyTorch [out, in] -> Paddle [in, out]
                converted = pt_val.T
                expected_pd_shape = (pt_shape[1], pt_shape[0])

                if pd_shape == expected_pd_shape:
                    converted_weights[pd_key_no_prefix] = paddle.to_tensor(
                        converted.astype(np.float32)
                    )
                    conversion_log["converted"].append({
                        "key": pd_key_no_prefix,
                        "transform": "transpose",
                        "pytorch_shape": list(pt_shape),
                        "paddle_shape": list(pd_shape)
                    })
                    converted_count += 1
                else:
                    conversion_log["errors"].append({
                        "key": pt_key,
                        "error": f"Shape mismatch after transpose: expected {expected_pd_shape}, got {pd_shape}"
                    })
                    error_count += 1
            else:
                # bias, LayerNorm, Embedding 等直接复制
                if pd_shape == pt_shape:
                    converted_weights[pd_key_no_prefix] = paddle.to_tensor(
                        pt_val.astype(np.float32)
                    )
                    conversion_log["converted"].append({
                        "key": pd_key_no_prefix,
                        "transform": "copy",
                        "pytorch_shape": list(pt_shape),
                        "paddle_shape": list(pd_shape)
                    })
                    converted_count += 1
                else:
                    conversion_log["errors"].append({
                        "key": pt_key,
                        "error": f"Shape mismatch: PT={pt_shape}, PD={pd_shape}"
                    })
                    error_count += 1
        else:
            # PyTorch 有但 Paddle 没有的层
            conversion_log["skipped"].append({
                "key": pt_key,
                "reason": "No corresponding key in Paddle model"
            })
            skipped_count += 1

    # 处理 Paddle 独有的层（保留原始值）
    for pd_key in sorted(pd_keys_set):
        if pd_key not in converted_weights and any(pattern in pd_key for pattern in paddle_only_patterns):
            pd_val = pd_state[pd_key]
            converted_weights[pd_key] = pd_val
            conversion_log["copied_from_paddle"].append({
                "key": pd_key,
                "reason": "Paddle-specific layer (prop_mlp)",
                "shape": list(pd_val.shape)
            })
            paddle_only_count += 1

    # 添加汇总信息
    conversion_log["summary"] = {
        "converted": converted_count,
        "skipped": skipped_count,
        "copied_from_paddle": paddle_only_count,
        "errors": error_count,
        "total_output": len(converted_weights)
    }

    return converted_weights, conversion_log


# ── 验证转换结果 ───────────────────────────────────────────────────────────────

def verify_conversion(
    pt_weights: Dict[str, np.ndarray],
    converted_weights: Dict[str, paddle.Tensor]
) -> bool:
    """
    验证转换结果的正确性
    """
    logger.info("\nVerifying conversion...")

    all_pass = True
    checks = [
        # (pt_key, need_transpose)
        ('decoder.node_embedding.weight', True),
        ('decoder.atom_latent_emb.weight', True),
        ('decoder.atom_latent_emb.bias', False),
        ('decoder.csp_layer_0.edge_mlp.0.weight', True),
        ('decoder.csp_layer_0.edge_mlp.0.bias', False),
        ('decoder.coord_out.weight', True),
        ('decoder.lattice_out.weight', True),
    ]

    for pt_key, need_transpose in checks:
        if pt_key not in pt_weights:
            logger.warning(f"  PyTorch key not found: {pt_key}")
            continue

        # 去掉 'decoder.' 前缀来查找转换后的键
        pd_key = pt_key.replace('decoder.', '') if pt_key.startswith('decoder.') else pt_key

        if pd_key not in converted_weights:
            logger.warning(f"  Converted key not found: {pd_key} (from {pt_key})")
            all_pass = False
            continue

        pt_val = pt_weights[pt_key]
        pd_val = converted_weights[pd_key].numpy() if hasattr(converted_weights[pd_key], 'numpy') else converted_weights[pd_key]

        if need_transpose:
            pt_compare = pt_val.T
        else:
            pt_compare = pt_val

        if pt_compare.shape != pd_val.shape:
            logger.error(f"  [{pt_key}] SHAPE MISMATCH: {pt_compare.shape} vs {pd_val.shape}")
            all_pass = False
            continue

        diff = np.abs(pt_compare - pd_val).max()
        status = "PASS" if diff < 1e-5 else "FAIL"
        if status == "FAIL":
            all_pass = False
        logger.info(f"  [{status}] {pt_key}: max_diff={diff:.4e}")

    return all_pass


def main():
    logger.info("=" * 60)
    logger.info("MatInvent DiffCSP Weight Conversion")
    logger.info("=" * 60)

    # ── 1. 加载 PyTorch 权重 ─────────────────────────────────────────────────────
    logger.info("\n[1/4] Loading PyTorch weights...")
    pt_weights = load_pytorch_weights(PT_CKPT)
    logger.info(f"  Successfully loaded {len(pt_weights)} PyTorch weights")

    # ── 2. 构建 Paddle 模型 ───────────────────────────────────────────────────────
    logger.info("\n[2/4] Building Paddle model...")
    pd_model = build_paddle_model_smooth_true()
    pd_state = pd_model.state_dict()
    logger.info(f"  Paddle model has {len(pd_state)} weights")

    # ── 3. 执行转换 ───────────────────────────────────────────────────────────────
    logger.info("\n[3/4] Converting weights...")
    converted_weights, conversion_log = convert_weights(pt_weights, pd_model)

    summary = conversion_log["summary"]
    logger.info(f"  Converted: {summary['converted']} weights")
    logger.info(f"  Skipped: {summary['skipped']} weights")
    logger.info(f"  Copied from Paddle: {summary['copied_from_paddle']} weights")
    logger.info(f"  Errors: {summary['errors']} weights")
    logger.info(f"  Total output: {summary['total_output']} weights")

    if summary['errors'] > 0:
        logger.warning("  ⚠️ Conversion had errors! Check conversion log for details.")

    # ── 4. 验证转换结果 ───────────────────────────────────────────────────────────
    logger.info("\n[4/4] Verifying conversion...")
    all_pass = verify_conversion(pt_weights, converted_weights)

    if all_pass:
        logger.info("  All verification checks PASSED ✓")
    else:
        logger.warning("  Some verification checks FAILED!")

    # ── 5. 保存结果 ───────────────────────────────────────────────────────────────
    logger.info(f"\n[5/5] Saving results...")

    # 保存转换后的权重
    paddle.save(converted_weights, str(OUT_CKPT))
    logger.info(f"  Converted weights saved to: {OUT_CKPT}")

    # 保存转换日志
    with open(CONVERSION_LOG, 'w') as f:
        json.dump(conversion_log, f, indent=2)
    logger.info(f"  Conversion log saved to: {CONVERSION_LOG}")

    logger.info("\n" + "=" * 60)
    logger.info("Conversion completed!")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("Conversion Summary:")
    print("=" * 60)
    print(f"  Output file: {OUT_CKPT}")
    print(f"  Converted: {summary['converted']} weights")
    print(f"  Skipped: {summary['skipped']} weights (scheduler, type_out)")
    print(f"  Copied from Paddle: {summary['copied_from_paddle']} weights (prop_mlp)")
    print(f"  Errors: {summary['errors']} weights")
    print("=" * 60)


if __name__ == "__main__":
    main()
