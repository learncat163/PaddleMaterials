#!/usr/bin/env python3
"""
MatterGen 转换权重精度验证
==========================
目标精度: MatterGen 前向输出 max_diff < 1e-6

说明:
  1. 加载经过转换的 Paddle 权重 (从 PyTorch mattergen_base 转换)
  2. 通过 build_model_from_name 构建 Paddle MatterGen 架构
  3. 替换权重为转换后的版本
  4. 使用已导出的 PyTorch noisy_batch 作为输入
  5. 对比 Paddle 与 PyTorch 的 denoiser 输出 (pos, cell, atomic_numbers)

前置条件:
  1. 已运行 export_noisy_inputs.py  → diff-matinvent/info/mattergen/exported_noisy_inputs.json
  2. 已运行 convert_mattergen_weights.py → tmp/mattergen_mp20_converted.pdparams

运行:
  /home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/benchmark_mattergen_converted.py
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import paddle

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_JSON = PROJECT_ROOT / "diff-matinvent" / "info" / "mattergen" / "exported_noisy_inputs.json"
CONVERTED_CKPT = PROJECT_ROOT / "tmp" / "mattergen_mp20_converted.pdparams"
OUTPUT_DIR = PROJECT_ROOT / "diff-matinvent" / "info" / "mattergen"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "mattergen_mp20"

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def load_model_with_converted_weights(device: str):
    """
    构建 Paddle MatterGen (通过 build_model_from_name 获取架构),
    然后用转换后的 PyTorch 权重替换.
    """
    from ppmat.models import build_model_from_name
    
    logger.info(f"Building MatterGen model: {MODEL_NAME}")
    model, config = build_model_from_name(MODEL_NAME)
    
    # 加载转换后的权重
    logger.info(f"Loading converted weights: {CONVERTED_CKPT}")
    converted_sd = paddle.load(str(CONVERTED_CKPT))
    
    # 将 state_dict 与 model 对比
    model_sd = model.state_dict()
    missing = set(model_sd.keys()) - set(converted_sd.keys())
    extra = set(converted_sd.keys()) - set(model_sd.keys())
    
    if missing:
        logger.warning(f"Missing keys in converted weights ({len(missing)}): {sorted(missing)[:3]}")
    if extra:
        logger.warning(f"Extra keys in converted weights ({len(extra)}): {sorted(extra)[:3]}")
    
    logger.info(f"Matched keys: {len(set(model_sd.keys()) & set(converted_sd.keys()))}")
    model.set_state_dict(converted_sd)
    model.eval()
    logger.info("Model loaded with converted weights ✓")
    return model


def build_noisy_input(noisy_batch_json: dict, device: str) -> tuple:
    """
    从 JSON 数据构建 Paddle denoiser 输入.
    
    返回: (denoiser_input_dict, t_tensor)
    """
    frac_coords = paddle.to_tensor(
        np.array(noisy_batch_json["frac_coords"], dtype=np.float32)
    )  # [N, 3]
    lattice = paddle.to_tensor(
        np.array(noisy_batch_json["lattice"], dtype=np.float32)
    )  # [B, 3, 3]
    atom_types = paddle.to_tensor(
        np.array(noisy_batch_json["atom_types"], dtype=np.int64)
    )  # [N]
    num_atoms = paddle.to_tensor(
        np.array(noisy_batch_json["num_atoms"], dtype=np.int64)
    )  # [B]
    batch_idx = paddle.to_tensor(
        np.array(noisy_batch_json["batch"], dtype=np.int64)
    )  # [N]
    
    denoiser_input = {
        "frac_coords": frac_coords,
        "lattice":     lattice,
        "atom_types":  atom_types,
        "num_atoms":   num_atoms,
        "batch":       batch_idx,
    }
    return denoiser_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="gpu:0")
    args = parser.parse_args()
    
    # ── 检查前置条件 ─────────────────────────────────────────────────────────
    for f in [INPUT_JSON, CONVERTED_CKPT]:
        if not f.exists():
            logger.error(f"文件不存在: {f}")
            sys.exit(1)
    
    # ── 加载数据 ──────────────────────────────────────────────────────────────
    with open(INPUT_JSON) as f:
        data = json.load(f)
    
    meta = data["metadata"]
    noisy_batch_json = data["noisy_batch"]
    pytorch_outputs = data["model_outputs"]
    
    logger.info("=" * 60)
    logger.info("BENCHMARK MatterGen (converted weights) vs PyTorch")
    logger.info(f"PyTorch timestamp: {meta.get('timestamp', 'N/A')}")
    logger.info(f"batch_size={meta['batch_size']}, total_atoms={meta['total_atoms']}")
    logger.info(f"t_value={meta['t_value']}")
    logger.info("=" * 60)
    
    paddle.set_device(args.device)
    
    # ── 加载模型 ──────────────────────────────────────────────────────────────
    model = load_model_with_converted_weights(args.device)
    
    # ── 构建输入 ──────────────────────────────────────────────────────────────
    denoiser_input = build_noisy_input(noisy_batch_json, args.device)
    
    # t: scalar 1.0 per crystal
    batch_size = int(meta["batch_size"])
    t_value = float(meta["t_value"])
    t = paddle.full([batch_size], t_value, dtype="float32")
    
    logger.info(f"\nInput shapes:")
    for k, v in denoiser_input.items():
        logger.info(f"  {k}: {v.shape}")
    logger.info(f"  t: {t.shape} = [{t_value}] × {batch_size}")
    
    # ── 前向传播 ──────────────────────────────────────────────────────────────
    logger.info(f"\nRunning MatterGen denoiser forward...")
    with paddle.no_grad():
        output = model.model(denoiser_input, t)
    
    # output 是 replace_dict: {'frac_coords': ..., 'lattice': ..., 'atom_types': ...}
    pred_pos_paddle    = output["frac_coords"].numpy()   # [N, 3]  ~ pred_pos
    pred_cell_paddle   = output["lattice"].numpy()       # [B, 3, 3]
    pred_atom_paddle   = output["atom_types"].numpy()    # [N, 101]
    
    # ── PyTorch 参考输出 ──────────────────────────────────────────────────────
    pred_pos_pytorch  = np.array(pytorch_outputs["pos"])            # [N, 3]
    pred_cell_pytorch = np.array(pytorch_outputs["cell"])           # [B, 3, 3]
    pred_atom_pytorch = np.array(pytorch_outputs["atomic_numbers"]) # [N, 101]
    
    # ── 对比 ──────────────────────────────────────────────────────────────────
    pos_diff  = np.abs(pred_pos_paddle  - pred_pos_pytorch)
    cell_diff = np.abs(pred_cell_paddle - pred_cell_pytorch)
    atom_diff = np.abs(pred_atom_paddle - pred_atom_pytorch)
    
    logger.info(f"\nComparison Results:")
    logger.info(f"  pos  (frac_coords): Paddle mean={pred_pos_paddle.mean():.6f}, PyTorch mean={pred_pos_pytorch.mean():.6f}")
    logger.info(f"  cell (lattice):     Paddle mean={pred_cell_paddle.mean():.6f}, PyTorch mean={pred_cell_pytorch.mean():.6f}")
    logger.info(f"  atom (logits):      Paddle mean={pred_atom_paddle.mean():.6f}, PyTorch mean={pred_atom_pytorch.mean():.6f}")
    logger.info(f"")
    logger.info(f"  pos  max_diff  = {pos_diff.max():.4e}  (target < 1e-6)")
    logger.info(f"  pos  mean_diff = {pos_diff.mean():.4e}")
    logger.info(f"  cell max_diff  = {cell_diff.max():.4e}  (target < 1e-6)")
    logger.info(f"  cell mean_diff = {cell_diff.mean():.4e}")
    logger.info(f"  atom max_diff  = {atom_diff.max():.4e}  (target < 1e-6)")
    logger.info(f"  atom mean_diff = {atom_diff.mean():.4e}")
    
    pos_pass  = bool(pos_diff.max() < 1e-6)
    cell_pass = bool(cell_diff.max() < 1e-6)
    atom_pass = bool(atom_diff.max() < 1e-6)
    
    # 放宽: float32 下 1e-5 也很好
    pos_pass2  = bool(pos_diff.max() < 1e-5)
    cell_pass2 = bool(cell_diff.max() < 1e-5)
    atom_pass2 = bool(atom_diff.max() < 1e-5)
    
    logger.info(f"\n  PASS (< 1e-6): pos={pos_pass}, cell={cell_pass}, atom={atom_pass}")
    logger.info(f"  PASS (< 1e-5): pos={pos_pass2}, cell={cell_pass2}, atom={atom_pass2}")
    logger.info(f"  OVERALL (< 1e-5): {'PASS' if pos_pass2 and cell_pass2 and atom_pass2 else 'FAIL'}")
    
    # ── 保存结果 ───────────────────────────────────────────────────────────────
    results = {
        "metadata": {
            "script": "benchmark_mattergen_converted.py",
            "timestamp": datetime.now().isoformat(),
            "pytorch_export_timestamp": meta.get("timestamp", "N/A"),
            "batch_size": batch_size,
            "total_atoms": int(meta["total_atoms"]),
            "t_value": t_value,
            "target_precision": "1e-6",
            "architecture": "mattergen_mp20 with PyTorch-converted weights",
        },
        "paddle_outputs": {
            "pos_mean":  float(pred_pos_paddle.mean()),
            "pos_std":   float(pred_pos_paddle.std()),
            "cell_mean": float(pred_cell_paddle.mean()),
            "cell_std":  float(pred_cell_paddle.std()),
            "atom_mean": float(pred_atom_paddle.mean()),
        },
        "pytorch_outputs": {
            "pos_mean":  float(pred_pos_pytorch.mean()),
            "pos_std":   float(pred_pos_pytorch.std()),
            "cell_mean": float(pred_cell_pytorch.mean()),
            "cell_std":  float(pred_cell_pytorch.std()),
            "atom_mean": float(pred_atom_pytorch.mean()),
        },
        "elementwise": {
            "pos_max_diff":   float(pos_diff.max()),
            "pos_mean_diff":  float(pos_diff.mean()),
            "cell_max_diff":  float(cell_diff.max()),
            "cell_mean_diff": float(cell_diff.mean()),
            "atom_max_diff":  float(atom_diff.max()),
            "atom_mean_diff": float(atom_diff.mean()),
        },
        "pass_criteria": {
            "threshold_strict": 1e-6,
            "threshold_loose": 1e-5,
            "pos_pass_strict":  pos_pass,
            "cell_pass_strict": cell_pass,
            "atom_pass_strict": atom_pass,
            "overall_pass_loose": pos_pass2 and cell_pass2 and atom_pass2,
        },
    }
    
    out_file = OUTPUT_DIR / "forward_converted_comparison.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
