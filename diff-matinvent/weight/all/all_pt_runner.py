#!/usr/bin/env python3
"""
PT (PyTorch / matinvent) 端 全量晶体推理子脚本

本脚本只能在 matinvent conda 环境中运行，由 all_infer_and_diff.py 通过 subprocess 调用。

用法：
  python all_pt_runner.py <RAW_ROOT> <NUM_ATOMS_LIST_JSON> <N_STEPS> <SEED> <OUT_JSON> [USE_SHARED_RNG]

参数：
  RAW_ROOT              raw-matinvent 根目录（含 mattergen 包）
  NUM_ATOMS_LIST_JSON   json 字符串，例如 "[6,8]"
  N_STEPS               推理步数（整数，如 1000）
  SEED                  随机种子（整数）
  OUT_JSON              输出结果 JSON 文件路径
  USE_SHARED_RNG        "1" 表示用 numpy 共享 RNG 替换 torch random（默认 "1"）

注意：原始代码在 raw-matinvent 目录中，不可修改。
"""

import json
import sys
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------
# 参数解析
# -----------------------------------------------------------------
RAW_ROOT          = Path(sys.argv[1])
NUM_ATOMS_LIST    = json.loads(sys.argv[2])   # e.g. [6, 8]
N_STEPS           = int(sys.argv[3])
SEED              = int(sys.argv[4])
OUT_JSON          = Path(sys.argv[5])
USE_SHARED_RNG    = len(sys.argv) <= 6 or sys.argv[6] == "1"

sys.path.insert(0, str(RAW_ROOT / "mattergen"))

import os
os.environ["PYTHONHASHSEED"] = str(SEED)

import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# -----------------------------------------------------------------
# 共享 Numpy RNG —— 与 Paddle 端同步随机数
# -----------------------------------------------------------------

class _NumpyRNGWrapper:
    def __init__(self, seed: int):
        self._seed = seed
        self.rng = np.random.RandomState(seed)
        self.call_count = 0
        self.total_values = 0

    def reset(self, seed: int = None):
        s = seed if seed is not None else self._seed
        self.rng = np.random.RandomState(s)
        self.call_count = 0
        self.total_values = 0

    def randn(self, shape):
        self.call_count += 1
        n = int(np.prod(shape))
        self.total_values += n
        return self.rng.randn(*[int(s) for s in shape]).astype(np.float32)


_numpy_rng = _NumpyRNGWrapper(SEED)


def _install_torch_randn_patch(device):
    """将 torch.randn / torch.randn_like 替换为 numpy-backed 版本。"""
    orig_randn = torch.randn
    orig_randn_like = torch.randn_like

    def _np_torch_randn(*size, **kwargs):
        # torch.randn 可以接受解包的维度，也可以接受 size=(...) kwarg
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = size[0]
        _dtype = kwargs.get('dtype', torch.float32)
        _device = kwargs.get('device', device)
        arr = _numpy_rng.randn(size)
        t = torch.from_numpy(arr)
        if _dtype not in (torch.float32, None):
            t = t.to(dtype=_dtype)
        if _device is not None:
            t = t.to(device=_device)
        return t

    def _np_torch_randn_like(input, **kwargs):
        _dtype = kwargs.get('dtype', input.dtype)
        _device = kwargs.get('device', input.device)
        arr = _numpy_rng.randn(input.shape)
        t = torch.from_numpy(arr).to(dtype=_dtype, device=_device)
        return t

    torch.randn = _np_torch_randn
    torch.randn_like = _np_torch_randn_like
    return orig_randn, orig_randn_like


def _restore_torch_randn(orig_randn, orig_randn_like):
    torch.randn = orig_randn
    torch.randn_like = orig_randn_like


# -----------------------------------------------------------------
# Torch Categorical 补丁 —— 同步 D3PM atom_types 采样
# -----------------------------------------------------------------

_torch_cat_call_count = 0


def _install_torch_categorical_patch(device):
    """将 torch.distributions.Categorical.sample 替换为 numpy-backed 版本。

    每次调用 Categorical.sample() 时，从同一个 _numpy_rng.rng 抽取随机数，
    保证两侧（Paddle/PT）的 D3PM atom_types 采样序列完全相同。
    """
    global _torch_cat_call_count
    _torch_cat_call_count = 0
    orig = torch.distributions.Categorical.sample

    def _np_categorical_sample(self_cat, sample_shape=torch.Size([])):
        global _torch_cat_call_count
        _torch_cat_call_count += 1
        # 从 logits 计算归一化概率
        logits = self_cat.logits.cpu().numpy()    # (N, C) float32
        logits_shifted = logits - logits.max(-1, keepdims=True)
        probs = np.exp(logits_shifted)
        probs /= probs.sum(-1, keepdims=True)
        # 逐原子采样（保持与 Paddle 端相同的调用粒度）
        orig_shape = probs.shape[:-1]
        n_atoms = int(np.prod(orig_shape))
        n_classes = probs.shape[-1]
        flat_probs = probs.reshape(n_atoms, n_classes)
        samples = np.array(
            [_numpy_rng.rng.choice(n_classes, p=flat_probs[i]) for i in range(n_atoms)],
            dtype=np.int64,
        )
        out_device = self_cat.logits.device
        return torch.tensor(samples.reshape(orig_shape), dtype=torch.long, device=out_device)

    torch.distributions.Categorical.sample = _np_categorical_sample
    return orig


def _restore_torch_categorical(orig_fn):
    torch.distributions.Categorical.sample = orig_fn


# -----------------------------------------------------------------
# 加载模型
# -----------------------------------------------------------------
print(f"[PT] 加载模型 from huggingface...", flush=True)

from mattergen.common.utils.eval_utils import (
    MatterGenCheckpointInfo,
    load_model_diffusion,
)
from mattergen.common.utils.globals import get_device
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector

# 原版通过 hydra 管理采样配置
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

ckpt_info = MatterGenCheckpointInfo.from_hf_hub('mattergen_base')
model = load_model_diffusion(ckpt_info)
device = get_device()
model = model.to(device)
model.eval()
print(f"[PT] 模型加载完成，device={device}", flush=True)

# -----------------------------------------------------------------
# 构建 sampler（通过 hydra 加载采样配置，仅用 sampler 相关部分）
# -----------------------------------------------------------------
SAMPLING_CONF_DIR = RAW_ROOT / "mattergen" / "sampling_conf"
batch_size = len(NUM_ATOMS_LIST)

with hydra.initialize_config_dir(os.path.abspath(str(SAMPLING_CONF_DIR)), version_base=None):
    sampling_config = hydra.compose(
        config_name="default",
        overrides=[
            f"sampler_partial.guidance_scale=0.0",
        ]
    )

print(f"[PT] sampler N={sampling_config.sampler_partial.N}, seed={SEED}", flush=True)

sampler_partial = instantiate(sampling_config.sampler_partial)
sampler = sampler_partial(pl_module=model)

# -----------------------------------------------------------------
# 构建 condition_loader（固定 num_atoms，使用官方 NumAtomsCrystalDataset）
# -----------------------------------------------------------------
# 直接使用官方的 NumAtomsCrystalDataset，它会把 pos/cell/atomic_numbers 填 NaN/-1
# _sample_prior 只需要 .shape，所以这完全可行
from mattergen.common.data.dataset import NumAtomsCrystalDataset
from torch.utils.data import DataLoader

dataset = NumAtomsCrystalDataset(num_atoms=np.array(NUM_ATOMS_LIST, dtype=np.int64))

def _collate_fn_wrapper(batch):
    return collate(batch), None

condition_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=_collate_fn_wrapper,
    shuffle=False,
)

# -----------------------------------------------------------------
# 执行采样（安装 numpy patch → 采样 → 恢复）
# -----------------------------------------------------------------
print(f"[PT] 开始采样，num_atoms={NUM_ATOMS_LIST}, N_STEPS={N_STEPS}, seed={SEED}", flush=True)
print(f"[PT] USE_SHARED_RNG={USE_SHARED_RNG}", flush=True)

orig_randn = orig_randn_like = None
orig_cat = None
if USE_SHARED_RNG:
    _numpy_rng.reset(SEED)
    orig_randn, orig_randn_like = _install_torch_randn_patch(device=device)
    orig_cat = _install_torch_categorical_patch(device=device)
    print(f"[PT] 已安装 numpy 共享 RNG patch + Categorical patch", flush=True)

all_mean_samples = []
for conditioning_data, mask in condition_loader:
    conditioning_data = conditioning_data.to(device)
    sample, mean = sampler.sample(conditioning_data, mask)
    all_mean_samples.extend(mean.to_data_list())

if USE_SHARED_RNG and orig_randn is not None:
    _restore_torch_randn(orig_randn, orig_randn_like)
    if orig_cat is not None:
        _restore_torch_categorical(orig_cat)
    print(f"[PT] numpy RNG 统计：randn调用={_numpy_rng.call_count}, "
          f"消耗随机数={_numpy_rng.total_values}, "
          f"categorical调用={_torch_cat_call_count}", flush=True)

print(f"[PT] 采样完成，共 {len(all_mean_samples)} 个结构", flush=True)

# -----------------------------------------------------------------
# 整理结果
# -----------------------------------------------------------------
from mattergen.common.utils.data_utils import lattice_matrix_to_params_torch

results = []
for cg in all_mean_samples:
    # cg.pos: (num_atoms, 3) frac coords
    # cg.cell: (1, 3, 3) or (3, 3) lattice matrix
    # cg.atomic_numbers: (num_atoms,) atom types
    # cg.num_atoms: scalar or (1,)
    na = int(cg.num_atoms.item()) if cg.num_atoms.numel() == 1 else int(cg.num_atoms[0].item())
    pos_np = cg.pos.cpu().numpy().tolist()
    cell_np = cg.cell.cpu().squeeze(0).numpy().tolist() if cg.cell.ndim == 3 else cg.cell.cpu().numpy().tolist()
    atomic_np = cg.atomic_numbers.cpu().numpy().tolist()
    results.append({
        "num_atoms": na,
        "atom_types": atomic_np,
        "frac_coords": pos_np,
        "lattice": cell_np,
    })
    print(f"  [PT] struct na={na} atom_types={atomic_np[:4]}... "
          f"frac_coords[0]={pos_np[0]} cell[0]={cell_np[0]}", flush=True)

output = {"results": results}
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

print(f"[PT] 结果已保存到 {OUT_JSON}", flush=True)
