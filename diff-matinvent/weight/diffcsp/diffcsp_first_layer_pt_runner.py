#!/usr/bin/env python3
"""

特别注意：

本脚本，只能 在 matinvent 的独立环境里运行，最好按照官方方式，通过conda环境实现；

避免直接在ppmat里直接耦合执行，会有一些麻烦

PyTorch CSPNet 推理子脚本 —— 在 matinvent conda 环境下执行。

用法（由 diffcsp_first_layer_infer_and_diff.py 通过 subprocess 调用）：
  python diffcsp_first_layer_pt_runner.py <RAW_ROOT> <INPUT_JSON> <PT_CKPT> <OUT_JSON>

参数：
  RAW_ROOT    raw-matinvent 根目录（含 models/diffcsp/cspnet.py）
  INPUT_JSON  固定输入数据 JSON 文件路径
  PT_CKPT     PyTorch 检查点 .ckpt 文件路径
  OUT_JSON    推理结果输出 JSON 文件路径
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

RAW_ROOT   = Path(sys.argv[1])
INPUT_JSON = Path(sys.argv[2])
PT_CKPT    = Path(sys.argv[3])
OUT_JSON   = Path(sys.argv[4])

sys.path.insert(0, str(RAW_ROOT))
from models.diffcsp.cspnet import CSPNet  # noqa: E402

with open(INPUT_JSON) as f:
    inp = json.load(f)

time_emb    = torch.tensor(inp["time_emb"],        dtype=torch.float32)
atom_types  = torch.tensor(inp["atom_type_probs"], dtype=torch.float32)
frac_coords = torch.tensor(inp["frac_coords"],     dtype=torch.float32)
lattices    = torch.tensor(inp["lattices"],        dtype=torch.float32)
num_atoms   = torch.tensor(inp["num_atoms"],       dtype=torch.long)
batch_idx   = torch.tensor(inp["batch_idx"],       dtype=torch.long)

model = CSPNet(
    hidden_dim=512, latent_dim=256, num_layers=6,
    act_fn="silu", dis_emb="sin", num_freqs=128,
    edge_style="fc", ln=True, ip=True, smooth=True,
    pred_type=False, pred_scalar=False,
)
model.eval()

ckpt = torch.load(str(PT_CKPT), map_location="cpu")
sd = ckpt["state_dict"]
decoder_sd = {k.replace("decoder.", "", 1): v for k, v in sd.items() if k.startswith("decoder.")}
decoder_sd = {k: v for k, v in decoder_sd.items() if not k.startswith("type_out")}
missing, unexpected = model.load_state_dict(decoder_sd, strict=False)
print(f"[PT] missing={missing}, unexpected={unexpected}", flush=True)

first_layer_out = {}


def _hook(mod, inp, out):
    first_layer_out["output"] = out.detach().cpu().numpy()


model.node_embedding.register_forward_hook(_hook)

with torch.no_grad():
    pred_l, pred_x = model(time_emb, atom_types, frac_coords, lattices, num_atoms, batch_idx)

fl = first_layer_out["output"]
pl = pred_l.cpu().numpy()
px = pred_x.cpu().numpy()

result = {
    "first_layer_output": fl.tolist(),
    "pred_l": pl.tolist(),
    "pred_x": px.tolist(),
    "stats": {
        "first_layer_mean": float(fl.mean()),
        "first_layer_std":  float(fl.std()),
        "pred_l_mean":      float(pl.mean()),
        "pred_x_mean":      float(px.mean()),
    },
}
with open(OUT_JSON, "w") as f:
    json.dump(result, f)

print(f"[PT] first_layer shape={fl.shape}, mean={fl.mean():.6f}", flush=True)
print(f"[PT] pred_l shape={pl.shape}, mean={pl.mean():.6f}", flush=True)
print(f"[PT] pred_x shape={px.shape}, mean={px.mean():.6f}", flush=True)
print(f"[PT] saved to {OUT_JSON}", flush=True)
