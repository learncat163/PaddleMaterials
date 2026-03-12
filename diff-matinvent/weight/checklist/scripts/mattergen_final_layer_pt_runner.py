#!/usr/bin/env python3
"""

特别注意：

本脚本，只能 在 matinvent 的独立环境里运行，最好按照官方方式，通过conda环境实现；

避免直接在ppmat里直接耦合执行，会有一些麻烦

PyTorch MatterGen 推理子脚本（最终层） —— 在 matinvent conda 环境下执行。

用法（由 mattergen_final_layer_infer_and_diff.py 通过 subprocess 调用）：
  python mattergen_final_layer_pt_runner.py <RAW_ROOT> <INPUT_JSON> <PT_CKPT> <OUT_JSON>

参数：
  RAW_ROOT    raw-matinvent 根目录（含 models/suite/mattergen.py）
  INPUT_JSON  固定输入数据 JSON 文件路径
  PT_CKPT     PyTorch 检查点 .ckpt 文件路径（未使用，自动从 huggingface 下载）
  OUT_JSON    推理结果输出 JSON 文件路径
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

RAW_ROOT   = Path(sys.argv[1])
INPUT_JSON = Path(sys.argv[2])
# PT_CKPT is not used, we load from huggingface directly
OUT_JSON   = Path(sys.argv[4])

sys.path.insert(0, str(RAW_ROOT))

# 使用 mattergen 包直接加载模型
from mattergen.common.utils.eval_utils import MatterGenCheckpointInfo
from mattergen.diffusion.lightning_module import DiffusionLightningModule
from mattergen.common.data.chemgraph import ChemGraph

with open(INPUT_JSON) as f:
    inp = json.load(f)

# 构建输入数据
times        = torch.tensor(inp["times"],           dtype=torch.float32)
atom_types   = torch.tensor(inp["atom_types"],      dtype=torch.long)
frac_coords  = torch.tensor(inp["frac_coords"],     dtype=torch.float32)
lattices     = torch.tensor(inp["lattices"],        dtype=torch.float32)
num_atoms    = torch.tensor(inp["num_atoms"],       dtype=torch.long)

# 创建 batch 索引
batch_idx = []
for i, num in enumerate(inp["num_atoms"]):
    batch_idx.extend([i] * num)
batch_idx = torch.tensor(batch_idx, dtype=torch.long)

# 使用 from_hf_hub 加载模型
print(f"[PT] Loading model from huggingface...", flush=True)
ckpt_info = MatterGenCheckpointInfo.from_hf_hub('mattergen_base')

model = DiffusionLightningModule.load_from_checkpoint_and_config(
    ckpt_info.checkpoint_path,
    config=ckpt_info.config.lightning_module,
    map_location='cpu',
    strict=False,
)[0]
model.eval()

with torch.no_grad():
    # 构建输入 ChemGraph
    # 原始 MatterGen 使用: pos, cell, atomic_numbers, num_atoms
    data_list = []
    start_idx = 0
    for i, num in enumerate(inp["num_atoms"]):
        data_list.append(
            ChemGraph(
                pos=frac_coords[start_idx:start_idx + num],
                cell=lattices[i:i + 1],
                atomic_numbers=atom_types[start_idx:start_idx + num],
                num_atoms=torch.tensor([num]),
            )
        )
        start_idx += num

    from mattergen.common.data.collate import collate
    batch = collate(data_list)

    # 调用 GemNetTDenoiser 的 forward
    output = model.diffusion_module.model(batch, times)

pl = output["cell"].cpu().numpy()
px = output["pos"].cpu().numpy()
pa = output["atomic_numbers"].cpu().numpy()

result = {
    "pred_lattice": pl.tolist(),
    "pred_frac_coords": px.tolist(),
    "pred_atom_types": pa.tolist(),
    "stats": {
        "pred_lattice_mean":   float(pl.mean()),
        "pred_frac_coords_mean": float(px.mean()),
        "pred_atom_types_mean":  float(pa.mean()),
    },
}
with open(OUT_JSON, "w") as f:
    json.dump(result, f)

print(f"[PT] pred_lattice shape={pl.shape}, mean={pl.mean():.6f}", flush=True)
print(f"[PT] pred_frac_coords shape={px.shape}, mean={px.mean():.6f}", flush=True)
print(f"[PT] pred_atom_types shape={pa.shape}, mean={pa.mean():.6f}", flush=True)
print(f"[PT] saved to {OUT_JSON}", flush=True)
