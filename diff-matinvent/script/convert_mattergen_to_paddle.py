#!/usr/bin/env python3
"""
MatterGen 权重转换脚本 (PyTorch → Paddle)
=============================================
将 PyTorch MatterGen checkpoint 转换为 Paddle pdparams 格式

功能:
1. 加载 PyTorch checkpoint (microsoft/mattergen)
2. 构建 Paddle MatterGen 模型
3. 转换权重 (Linear weight 转置)
4. 保存转换后的权重
5. 验证转换结果

输出文件:
- tmp/mattergen_mp20_converted.pdparams  # 转换后的权重
- tmp/mattergen_conversion_log.json      # 转换日志

运行环境: ppmat (PaddlePaddle)

用法:
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/convert_mattergen_to_paddle.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any

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
# PyTorch 源路径
PT_CKPT = Path("/home/cao/.cache/huggingface/hub/models--microsoft--mattergen/snapshots/ea430eab64b80855029c2941b9fda15f245a771a/checkpoints/mp_20_base/checkpoints/last.ckpt")

# 输出路径
OUTPUT_DIR = Path(PROJECT_ROOT) / "tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CKPT = OUTPUT_DIR / "mattergen_mp20_converted.pdparams"
CONVERSION_LOG = OUTPUT_DIR / "mattergen_conversion_log.json"

# PyTorch 前缀
PT_PREFIX = "diffusion_module.model."
# Paddle 前缀 (用于保存)
PD_PREFIX = "model."


# ── PyTorch checkpoint 加载 ─────────────────────────────────────────────────────

def load_pytorch_weights(ckpt_path: Path) -> Dict[str, np.ndarray]:
    """加载 PyTorch checkpoint 权重"""
    logger.info(f"Loading PyTorch weights from: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    sd = ckpt['state_dict']

    result = {}
    for k, v in sd.items():
        # 去掉前缀
        stripped_k = k.replace(PT_PREFIX, '') if k.startswith(PT_PREFIX) else k
        result[stripped_k] = v.numpy() if hasattr(v, 'numpy') else np.array(v)

    logger.info(f"  Loaded {len(result)} PyTorch weights")
    return result


# ── Paddle 模型构建 ─────────────────────────────────────────────────────────────

def build_paddle_mattergen() -> paddle.nn.Layer:
    """构建 Paddle MatterGen 模型"""
    logger.info("Building Paddle MatterGen model...")

    from ppmat.models.mattergen.mattergen import MatterGen
    from ppmat.schedulers import LatticeVPSDEScheduler, NumAtomsVarianceAdjustedWrappedVESDE

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

    logger.info("  Paddle MatterGen model built successfully")
    return model


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
    - scale_factor 等 buffer: 直接复制
    """
    converted_weights = {}
    conversion_log = {
        "timestamp": datetime.now().isoformat(),
        "pytorch_checkpoint": str(PT_CKPT),
        "converted": [],
        "skipped": [],
        "errors": []
    }

    # 获取 Paddle 模型的 state_dict
    pd_state = pd_model.state_dict()
    pd_keys_set = set(pd_state.keys())

    # 处理每个 PyTorch 权重
    converted_count = 0
    skipped_count = 0
    error_count = 0

    for pt_key, pt_val in sorted(pt_weights.items()):
        pt_shape = list(pt_val.shape)

        # 在 Paddle state_dict 中查找对应的键
        pd_key_found = None
        for pd_key in pd_keys_set:
            # 去掉 PD_PREFIX 后匹配
            pd_key_base = pd_key.replace(PD_PREFIX, '') if pd_key.startswith(PD_PREFIX) else pd_key
            if pd_key_base == pt_key:
                pd_key_found = pd_key
                break

        if pd_key_found is None:
            conversion_log["skipped"].append({
                "key": pt_key,
                "reason": "No corresponding key in Paddle model"
            })
            skipped_count += 1
            continue

        pd_val = pd_state[pd_key_found]
        pd_shape = tuple(pd_val.shape)

        # 检查是否需要转置 (Linear 权重)
        # Embedding 层不转置，只对 Linear 层转置
        if pt_key.endswith('.weight') and len(pt_shape) == 2 and 'embeddings.weight' not in pt_key:
            # Linear 权重需要转置: PyTorch [out, in] -> Paddle [in, out]
            converted = pt_val.T
            expected_pd_shape = (pt_shape[1], pt_shape[0])

            if pd_shape == expected_pd_shape or (len(pd_shape) == 2 and pd_shape[0] == expected_pd_shape[0] and pd_shape[1] == expected_pd_shape[1]):
                converted_weights[pd_key_found] = paddle.to_tensor(
                    converted.astype(np.float32)
                )
                conversion_log["converted"].append({
                    "key": pd_key_found,
                    "transform": "transpose",
                    "pytorch_shape": pt_shape,
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
            # bias, LayerNorm, Embedding, buffer 等直接复制
            # 将 shape 转换为 tuple 进行比较
            pt_shape_tuple = tuple(pt_shape) if isinstance(pt_shape, list) else pt_shape
            pd_shape_tuple = tuple(pd_shape) if isinstance(pd_shape, list) else pd_shape

            # 比较时考虑 1D 数组的情况: [101] vs (101,) 是相同的
            if pt_shape_tuple == pd_shape_tuple or (len(pt_shape_tuple) == 1 and len(pd_shape_tuple) == 1 and pt_shape_tuple[0] == pd_shape_tuple[0]):
                converted_weights[pd_key_found] = paddle.to_tensor(
                    pt_val.astype(np.float32)
                )
                conversion_log["converted"].append({
                    "key": pd_key_found,
                    "transform": "copy",
                    "pytorch_shape": pt_shape,
                    "paddle_shape": list(pd_shape)
                })
                converted_count += 1
            else:
                conversion_log["errors"].append({
                    "key": pt_key,
                    "error": f"Shape mismatch: PT={pt_shape_tuple}, PD={pd_shape_tuple}"
                })
                error_count += 1

    # 添加汇总信息
    conversion_log["summary"] = {
        "converted": converted_count,
        "skipped": skipped_count,
        "errors": error_count,
        "total_output": len(converted_weights)
    }

    return converted_weights, conversion_log


# ── 验证转换结果 ───────────────────────────────────────────────────────────────

def verify_conversion(
    pt_weights: Dict[str, np.ndarray],
    converted_weights: Dict[str, paddle.Tensor]
) -> bool:
    """验证转换结果的正确性"""
    logger.info("\nVerifying conversion...")

    all_pass = True
    checks = [
        # (pt_key, need_transpose)
        ('model.gemnet.atom_emb.embeddings.weight', False),
        ('model.gemnet.atom_latent_emb.weight', True),
        ('model.gemnet.atom_latent_emb.bias', False),
        ('model.gemnet.edge_emb.dense.linear.weight', True),
        ('model.gemnet.int_blocks.0.atom_update.layers.0.linear.weight', True),
        ('model.fc_atom.weight', True),
        ('model.fc_atom.bias', False),
    ]

    for pt_key, need_transpose in checks:
        if pt_key not in pt_weights:
            logger.warning(f"  PyTorch key not found: {pt_key}")
            continue

        pd_key = PD_PREFIX + pt_key.replace('model.', '') if not pt_key.startswith('model.') else pt_key

        if pd_key not in converted_weights:
            logger.warning(f"  Converted key not found: {pd_key}")
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
    logger.info("MatterGen Weight Conversion: PyTorch → Paddle")
    logger.info("=" * 60)

    # ── 1. 加载 PyTorch 权重 ─────────────────────────────────────────────────────
    logger.info("\n[1/4] Loading PyTorch weights...")
    pt_weights = load_pytorch_weights(PT_CKPT)
    logger.info(f"  Successfully loaded {len(pt_weights)} PyTorch weights")

    # ── 2. 构建 Paddle 模型 ───────────────────────────────────────────────────────
    logger.info("\n[2/4] Building Paddle model...")
    pd_model = build_paddle_mattergen()
    pd_state = pd_model.state_dict()
    logger.info(f"  Paddle model has {len(pd_state)} weights")

    # ── 3. 执行转换 ───────────────────────────────────────────────────────────────
    logger.info("\n[3/4] Converting weights...")
    converted_weights, conversion_log = convert_weights(pt_weights, pd_model)

    summary = conversion_log["summary"]
    logger.info(f"  Converted: {summary['converted']} weights")
    logger.info(f"  Skipped: {summary['skipped']} weights")
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
    print(f"  Skipped: {summary['skipped']} weights")
    print(f"  Errors: {summary['errors']} weights")
    print("=" * 60)


if __name__ == "__main__":
    main()
