#!/usr/bin/env python3
"""
MatterGen 权重转换脚本: PyTorch checkpoint → Paddle pdparams
============================================================
目标精度: MatterGen 前向输出 max_diff < 1e-6

说明:
  PyTorch checkpoint (microsoft/mattergen):
    - keys 前缀: diffusion_module.model.
    - Linear weight shape: [out, in]
  
  Paddle checkpoint (mattergen_mp20):
    - keys 前缀: model.
    - Linear weight shape: [in, out]
  
  转换规则:
    - key 名称: 去掉前缀后相同 (281 个 keys 完全匹配)
    - 2D weight (Linear): 转置 [out, in] → [in, out]
    - 1D weight (bias, LayerNorm, etc.): 直接复制
    - noise_level_encoding.div_term: 不可训练参数, 直接复制 (两边应一致)
  
  特殊处理:
    - noise_level_encoding.div_term: 是 buffer 非参数, 直接复制验证
  
  输出:
    tmp/mattergen_mp20_converted.pdparams  (Paddle 格式, model. 前缀)

扩展:
  对于 mattergen_mp20_chemical_system, mattergen_mp20_dft_band_gap 等变体,
  处理方式相同 (更换 checkpoint 路径即可)

运行:
  /home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/convert_mattergen_weights.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import paddle
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── 路径配置 ───────────────────────────────────────────────────────────────────
PT_CKPT = Path("/home/cao/.cache/huggingface/hub/models--microsoft--mattergen/snapshots/ea430eab64b80855029c2941b9fda15f245a771a/checkpoints/mp_20_base/checkpoints/last.ckpt")
PD_CKPT_ORIG = Path("/home/cao/.paddlemat/weights/mattergen_mp20/mattergen_mp20/mattergen_mp20/checkpoints/latest.pdparams")
OUT_CKPT = Path(PROJECT_ROOT) / "tmp" / "mattergen_mp20_converted.pdparams"
OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)

# PyTorch prefix to strip
PT_PREFIX = "diffusion_module.model."
# Paddle prefix (restored after stripping, for saving)
PD_PREFIX = "model."


def load_pt_state_dict(ckpt_path: Path) -> dict:
    """返回 {stripped_key: numpy_array} (去掉 PT_PREFIX)"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt['state_dict']
    result = {}
    for k, v in sd.items():
        stripped_k = k.replace(PT_PREFIX, '') if k.startswith(PT_PREFIX) else k
        result[stripped_k] = v.numpy()
    logger.info(f"Loaded {len(result)} PyTorch weights from {ckpt_path.name}")
    return result


def load_pd_state_dict(ckpt_path: Path) -> dict:
    """返回 {stripped_key: numpy_array} (去掉 PD_PREFIX)"""
    sd = paddle.load(str(ckpt_path))
    result = {}
    for k, v in sd.items():
        stripped_k = k.replace(PD_PREFIX, '') if k.startswith(PD_PREFIX) else k
        result[stripped_k] = v.numpy()
    logger.info(f"Loaded {len(result)} Paddle weights from {ckpt_path.name}")
    return result


def convert_mattergen_weights(pt_sd: dict, pd_sd_orig: dict) -> dict:
    """
    Convert PyTorch MatterGen weights to Paddle format.
    
    规则:
      - Key 匹配: PT 和 PD (去掉前缀后) 完全相同
      - 2D weight (.weight, 且 shape 是转置关系): 转置
      - 其余 (bias, LayerNorm, buffer 等): 直接复制
    
    原始 PyTorch 代码: raw-matinvent/mattergen/
    原始 Paddle 代码: ppmat/models/mattergen/
    
    Returns:
        {PD_PREFIX + key: paddle.Tensor} 格式的 state_dict
    """
    new_pd = {}
    
    # 验证 key 集合完全一致
    pt_keys = set(pt_sd.keys())
    pd_keys = set(pd_sd_orig.keys())
    
    only_pt = pt_keys - pd_keys
    only_pd = pd_keys - pt_keys
    
    if only_pt:
        logger.warning(f"Keys only in PyTorch ({len(only_pt)}): {sorted(only_pt)[:5]}")
    if only_pd:
        logger.warning(f"Keys only in Paddle ({len(only_pd)}): {sorted(only_pd)[:5]}")
    
    logger.info(f"Common keys: {len(pt_keys & pd_keys)}")
    
    converted_count = 0
    skipped_count = 0
    
    for k in sorted(pt_sd.keys()):
        pt_val = pt_sd[k]
        pt_shape = list(pt_val.shape)
        
        if k not in pd_sd_orig:
            logger.warning(f"  NO PADDLE KEY: {k} {pt_shape}")
            skipped_count += 1
            continue
        
        pd_shape = list(pd_sd_orig[k].shape)
        
        # 判断是否需要转置
        if len(pt_shape) == 2 and list(reversed(pt_shape)) == pd_shape:
            # Linear weight: [out, in] → [in, out]
            converted = pt_val.T
            logger.info(f"  T {k}: {pt_shape} → {list(converted.shape)}")
        elif pt_shape == pd_shape:
            # Same shape: direct copy (bias, LayerNorm, buffers, Embedding, etc.)
            converted = pt_val
            logger.info(f"  = {k}: {pt_shape}")
        else:
            logger.error(f"  MISMATCH {k}: PT={pt_shape}, PD={pd_shape}")
            skipped_count += 1
            continue
        
        # 验证最终 shape
        if list(converted.shape) != pd_shape:
            logger.error(f"  SHAPE FAIL {k}: {list(converted.shape)} != {pd_shape}")
            skipped_count += 1
            continue
        
        # 保存时恢复 PD_PREFIX
        final_key = PD_PREFIX + k
        new_pd[final_key] = paddle.to_tensor(converted.astype(np.float32))
        converted_count += 1
    
    # 添加 Paddle 独有 keys (如果有)
    for k in only_pd:
        new_pd[PD_PREFIX + k] = paddle.to_tensor(pd_sd_orig[k].astype(np.float32))
        logger.info(f"  KEEP_PD_ONLY {k}")
    
    logger.info(
        f"\nConversion summary: {converted_count} converted, "
        f"{skipped_count} skipped, {len(only_pd)} paddle-only, "
        f"total={len(new_pd)}"
    )
    return new_pd


def verify_conversion(pt_sd: dict, new_pd: dict, sample_keys: int = 10):
    """
    抽样验证转换后的权重与 PyTorch 一致.
    对 2D weights 验证转置后相同, 对 1D weights 直接验证.
    """
    logger.info("\n=== Verification (sampling) ===")
    
    import random
    keys = sorted(pt_sd.keys())
    
    # 分别抽 2D 和 1D
    keys_2d = [k for k in keys if len(pt_sd[k].shape) == 2]
    keys_1d = [k for k in keys if len(pt_sd[k].shape) == 1]
    
    check_keys = random.sample(keys_2d, min(5, len(keys_2d))) + \
                 random.sample(keys_1d, min(5, len(keys_1d)))
    
    all_pass = True
    for k in check_keys:
        pt_val = pt_sd[k]
        pd_key = PD_PREFIX + k
        
        if pd_key not in new_pd:
            logger.warning(f"  MISSING: {pd_key}")
            continue
        
        pd_val = new_pd[pd_key].numpy() if hasattr(new_pd[pd_key], 'numpy') else new_pd[pd_key]
        
        if len(pt_val.shape) == 2:
            # 2D: 转置后对比
            cmp = pt_val.T
        else:
            # 1D: 直接对比
            cmp = pt_val
        
        if cmp.shape != pd_val.shape:
            logger.error(f"  SHAPE FAIL: {k}: {cmp.shape} vs {pd_val.shape}")
            all_pass = False
            continue
        
        diff = np.abs(cmp - pd_val).max()
        status = "PASS" if diff < 1e-5 else "FAIL"
        if status == "FAIL":
            all_pass = False
        logger.info(f"  [{status}] {k} {list(pt_val.shape)}: max_diff={diff:.4e}")
    
    # 特别验证 noise_level_encoding.div_term (应该两边一致)
    div_term_key = "noise_level_encoding.div_term"
    if div_term_key in pt_sd and (PD_PREFIX + div_term_key) in new_pd:
        pt_div = pt_sd[div_term_key]
        pd_div = new_pd[PD_PREFIX + div_term_key].numpy()
        diff = np.abs(pt_div - pd_div).max()
        status = "PASS" if diff < 1e-5 else "FAIL"
        if status == "FAIL":
            all_pass = False
        logger.info(f"  [{status}] {div_term_key}: max_diff={diff:.4e} (buffer, should be ~0)")
    
    return all_pass


def main():
    logger.info("=" * 60)
    logger.info("MatterGen Weight Conversion: PyTorch → Paddle")
    logger.info("=" * 60)

    # ── 加载 ──────────────────────────────────────────────────────────────────
    logger.info(f"\n1. Loading PyTorch checkpoint: {PT_CKPT}")
    pt_sd = load_pt_state_dict(PT_CKPT)
    
    logger.info(f"\n2. Loading Paddle checkpoint: {PD_CKPT_ORIG}")
    pd_sd_orig = load_pd_state_dict(PD_CKPT_ORIG)
    
    # ── 转换 ──────────────────────────────────────────────────────────────────
    logger.info(f"\n3. Converting weights...")
    new_pd = convert_mattergen_weights(pt_sd, pd_sd_orig)
    
    # ── 验证 ──────────────────────────────────────────────────────────────────
    logger.info(f"\n4. Verifying conversion (sampling)...")
    all_pass = verify_conversion(pt_sd, new_pd)
    
    if all_pass:
        logger.info("   All sampled checks PASSED ✓")
    else:
        logger.warning("   Some checks FAILED, please review")
    
    # ── 保存 ──────────────────────────────────────────────────────────────────
    logger.info(f"\n5. Saving converted checkpoint: {OUT_CKPT}")
    paddle.save(new_pd, str(OUT_CKPT))
    logger.info(f"   Saved {len(new_pd)} keys")
    
    logger.info("\n" + "=" * 60)
    logger.info("Done! Next step: run benchmark_mattergen_converted.py")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
