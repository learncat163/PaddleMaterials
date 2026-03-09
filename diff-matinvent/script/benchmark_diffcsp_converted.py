#!/usr/bin/env python3
"""
DiffCSP 转换权重精度验证 (smooth=True 架构对齐)
================================================
目标精度: DiffCSP 前向输出 max_diff < 1e-4

说明:
  1. 加载经过转换的 Paddle 权重 (smooth=True)
  2. 构建 smooth=True 的 Paddle CSPNet (与 PyTorch 架构对齐)
  3. 使用已导出的 PyTorch noised_inputs (atom_type_probs: float [N, 100])
  4. 对比 Paddle 与 PyTorch 的 decoder 输出

前置条件:
  1. 已运行 export_diffcsp_noisy_inputs.py  → diff-matinvent/info/diffcsp/exported_noisy_inputs.json
  2. 已运行 convert_diffcsp_weights.py      → tmp/diffcsp_mp20_converted.pdparams

运行:
  /home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/benchmark_diffcsp_converted.py
"""

import sys
import json
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import paddle
import paddle.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_JSON = PROJECT_ROOT / "diff-matinvent" / "info" / "diffcsp" / "exported_noisy_inputs.json"
CONVERTED_CKPT = PROJECT_ROOT / "tmp" / "diffcsp_mp20_converted.pdparams"
OUTPUT_DIR = PROJECT_ROOT / "diff-matinvent" / "info" / "diffcsp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def build_decoder_smooth_true():
    """
    构建 smooth=True 的 Paddle CSPNet (与 PyTorch 架构一致).
    
    直接修改 build_model 接收的 cfg, 设置 smooth=True.
    原始代码: ppmat/models/diffcsp/diffcsp.py CSPNet
    """
    from ppmat.models import build_model
    
    decoder_cfg = {
        "__class_name__": "CSPNet",
        "__init_params__": {
            "hidden_dim": 512,
            "latent_dim": 256,
            "num_layers": 6,
            "act_fn": "silu",
            "dis_emb": "sin",
            "num_freqs": 128,
            "edge_style": "fc",
            "ln": True,
            "ip": True,
            "smooth": True,   # ← 关键: 使用 Linear 代替 Embedding
            "pred_type": False,
            "prop_dim": 512,
            "pred_scalar": False,
            "num_classes": 100,
        },
    }
    # 为了方便, 直接实例化 CSPNet
    from ppmat.models.diffcsp.diffcsp import CSPNet
    import paddle.nn as nn
    cfg = decoder_cfg["__init_params__"]
    model = CSPNet(**cfg)
    return model


def load_converted_weights(model, ckpt_path: Path):
    """加载转换后的权重, 验证 smooth=True 的 node_embedding.bias 是否存在"""
    state_dict = paddle.load(str(ckpt_path))
    
    # 转换的 checkpoint 以 'decoder.' 前缀存储, CSPNet state_dict 无此前缀
    # 需要去掉 'decoder.' 前缀
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith('decoder.'):
            new_sd[k[len('decoder.'):]] = v
        else:
            new_sd[k] = v
    
    # 检查 smooth=True 必须有的 node_embedding.bias
    if 'node_embedding.bias' not in new_sd:
        logger.warning("node_embedding.bias 不在 state_dict 中! smooth=True 需要此 key")
        logger.warning("请重新运行 convert_diffcsp_weights.py")
    else:
        logger.info(f"node_embedding.bias found: shape={list(new_sd['node_embedding.bias'].shape)} ✓")
    
    # 加载到模型
    model_sd = model.state_dict()
    missing = set(model_sd.keys()) - set(new_sd.keys())
    extra = set(new_sd.keys()) - set(model_sd.keys())
    
    if missing:
        logger.warning(f"Missing keys in converted weights: {missing}")
    if extra:
        logger.warning(f"Extra keys in converted weights (not in model): {extra}")
    
    model.set_state_dict(new_sd)
    logger.info(f"Loaded {len(new_sd)} weights from {ckpt_path.name}")
    return model


def rebuild_time_emb(t_value_int: int, batch_size: int, time_dim: int = 256) -> paddle.Tensor:
    """与 DiffCSP PyTorch 相同的 SinusoidalTimeEmbeddings"""
    half_dim = time_dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    freqs = np.exp(np.arange(half_dim) * -embeddings)
    t = np.full(batch_size, t_value_int, dtype=np.float32)
    emb = t[:, None] * freqs[None, :]
    time_emb_np = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
    return paddle.to_tensor(time_emb_np, dtype="float32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="gpu:0")
    args = parser.parse_args()

    # ── 检查前置条件 ─────────────────────────────────────────────────────────
    for f in [INPUT_JSON, CONVERTED_CKPT]:
        if not f.exists():
            logger.error(f"文件不存在: {f}")
            sys.exit(1)

    # ── 加载 PyTorch 导出数据 ─────────────────────────────────────────────────
    with open(INPUT_JSON) as f:
        data = json.load(f)

    meta = data["metadata"]
    noised = data["noised_input"]
    pytorch_out = data["model_outputs"]

    logger.info("=" * 60)
    logger.info("BENCHMARK DiffCSP (converted, smooth=True) vs PyTorch")
    logger.info(f"PyTorch timestamp: {meta['timestamp']}")
    logger.info(f"t_idx={meta['t_idx']}, batch_size={meta['batch_size']}, total_atoms={meta['total_atoms']}")
    logger.info("=" * 60)

    paddle.set_device(args.device)
    
    # ── 构建 smooth=True 的 Paddle CSPNet decoder ─────────────────────────────
    logger.info("\n1. Building Paddle CSPNet with smooth=True...")
    decoder = build_decoder_smooth_true()
    decoder = load_converted_weights(decoder, CONVERTED_CKPT)
    decoder.eval()
    logger.info("   CSPNet decoder ready with smooth=True ✓")

    # ── 构建输入 tensors ──────────────────────────────────────────────────────
    batch_size = int(meta["batch_size"])
    t_value_int = int(meta["t_value"])

    # time_emb 从 JSON 加载 (PyTorch 原始值)
    time_emb = paddle.to_tensor(
        np.array(noised["time_emb"], dtype=np.float32)
    )  # [B, 256]

    # atom_type_probs: float soft distribution [N, 100] (PyTorch smooth=True 使用的输入)
    atom_type_probs = paddle.to_tensor(
        np.array(noised["atom_type_probs"], dtype=np.float32)
    )  # [N, 100]

    input_frac_coords = paddle.to_tensor(
        np.array(noised["input_frac_coords"], dtype=np.float32)
    )  # [N, 3]
    input_lattice = paddle.to_tensor(
        np.array(noised["input_lattice"], dtype=np.float32)
    )  # [B, 3, 3]
    num_atoms = paddle.to_tensor(
        np.array(noised["num_atoms"], dtype=np.int64)
    )  # [B]
    batch_idx = paddle.to_tensor(
        np.array(noised["batch"], dtype=np.int64)
    )  # [N]

    logger.info(f"\n2. Input shapes:")
    logger.info(f"   time_emb: {time_emb.shape}")
    logger.info(f"   atom_type_probs: {atom_type_probs.shape}, mean={float(atom_type_probs.mean()):.6f}")
    logger.info(f"   input_frac_coords: {input_frac_coords.shape}")
    logger.info(f"   input_lattice: {input_lattice.shape}")

    # ── Paddle decoder 前向传播 ────────────────────────────────────────────────
    # CSPNet.forward(t, atom_types, frac_coords, lattices, num_atoms, node2graph)
    # smooth=True → atom_types 是 float [N, 100]
    logger.info("\n3. Running Paddle decoder forward (smooth=True)...")
    with paddle.no_grad():
        pred_l_paddle, pred_x_paddle = decoder(
            time_emb,           # [B, 256]
            atom_type_probs,    # [N, 100] float → Linear node_embedding
            input_frac_coords,  # [N, 3]
            input_lattice,      # [B, 3, 3]
            num_atoms,          # [B]
            batch_idx,          # [N]
        )

    pred_l_paddle = pred_l_paddle.numpy()
    pred_x_paddle = pred_x_paddle.numpy()

    # ── 加载 PyTorch 输出 ──────────────────────────────────────────────────────
    pred_l_pytorch = np.array(pytorch_out["pred_l"])
    pred_x_pytorch = np.array(pytorch_out["pred_x"])

    # ── 对比 ──────────────────────────────────────────────────────────────────
    l_diff = np.abs(pred_l_paddle - pred_l_pytorch)
    x_diff = np.abs(pred_x_paddle - pred_x_pytorch)

    logger.info(f"\n4. Comparison Results:")
    logger.info(f"   pred_l: Paddle mean={pred_l_paddle.mean():.6f}, PyTorch mean={pred_l_pytorch.mean():.6f}")
    logger.info(f"   pred_x: Paddle mean={pred_x_paddle.mean():.6f}, PyTorch mean={pred_x_pytorch.mean():.6f}")
    logger.info(f"   pred_l max_diff = {l_diff.max():.4e}  (target < 1e-4)")
    logger.info(f"   pred_l mean_diff = {l_diff.mean():.4e}")
    logger.info(f"   pred_x max_diff = {x_diff.max():.4e}  (target < 1e-4)")
    logger.info(f"   pred_x mean_diff = {x_diff.mean():.4e}")

    l_pass = bool(l_diff.max() < 1e-4)
    x_pass = bool(x_diff.max() < 1e-4)
    logger.info(f"\n   pred_l PASS: {l_pass}")
    logger.info(f"   pred_x PASS: {x_pass}")
    logger.info(f"   OVERALL: {'PASS' if l_pass and x_pass else 'FAIL'}")

    # ── 保存结果 ───────────────────────────────────────────────────────────────
    results = {
        "metadata": {
            "script": "benchmark_diffcsp_converted.py",
            "timestamp": datetime.now().isoformat(),
            "pytorch_export_timestamp": meta["timestamp"],
            "t_idx": meta["t_idx"],
            "t_value": t_value_int,
            "batch_size": batch_size,
            "total_atoms": int(meta["total_atoms"]),
            "target_precision": "1e-4",
            "architecture": "smooth=True, pred_type=False (converted from PyTorch)",
        },
        "paddle_outputs": {
            "pred_l_mean": float(pred_l_paddle.mean()),
            "pred_l_std":  float(pred_l_paddle.std()),
            "pred_x_mean": float(pred_x_paddle.mean()),
            "pred_x_std":  float(pred_x_paddle.std()),
        },
        "pytorch_outputs": {
            "pred_l_mean": float(pred_l_pytorch.mean()),
            "pred_l_std":  float(pred_l_pytorch.std()),
            "pred_x_mean": float(pred_x_pytorch.mean()),
            "pred_x_std":  float(pred_x_pytorch.std()),
        },
        "elementwise": {
            "pred_l_max_diff":  float(l_diff.max()),
            "pred_l_mean_diff": float(l_diff.mean()),
            "pred_x_max_diff":  float(x_diff.max()),
            "pred_x_mean_diff": float(x_diff.mean()),
        },
        "pass_criteria": {
            "threshold": 1e-4,
            "pred_l_pass": l_pass,
            "pred_x_pass": x_pass,
            "overall_pass": l_pass and x_pass,
        },
    }

    out_file = OUTPUT_DIR / "forward_converted_comparison.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n5. Results saved to: {out_file}")


if __name__ == "__main__":
    main()
