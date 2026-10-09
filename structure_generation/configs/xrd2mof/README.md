# Xrd2Mof

[Interpreting X-Ray Diffraction Patterns of Metal-Organic Frameworks via Generative Artificial Intelligence](https://doi.org/10.1021/jacs.5c16416)

## Abstract

Xrd2Mof is a crystal structure generation framework that interprets powder X-ray
diffraction (PXRD) patterns and reconstructs coarse-grained (CG) metal-organic
framework (MOF) structures from them. A CLIP-style feature extractor aligns PXRD
patterns, metal nodes and organic linkers with the structural features (lattice
and atom positions) of the corresponding MOFs; a DiffCSP-style diffusion model then
generates the CG building-block sites (metal nodes and linkers) conditioned on the
extracted PXRD embedding. The generated CG structures are the input of a downstream
atomic assembly step, which is not part of this repository.

This directory contains the PaddlePaddle implementation of:

- `ppmat/models/xrd2mof/feature_extractor.py`: PXRD feature extractor (CLIP-style
  contrastive model, `Xrd2MofFeatureExtractor`).
- `ppmat/models/xrd2mof/xrd2mof.py`: CG structure generation model (`Xrd2Mof`),
  a DiffCSP variant with an additional per-sample XRD feature conditioning.

---

## Model Description

### Feature extractor

Five encoders are fused into two 256-dim embeddings:

| Encoder | Input | Output |
|---------|-------|--------|
| `LatticeEncoder` | lattice matrix (3x3) | 32 |
| `AtomEncoder` | atom positions (50x3) | 256 |
| `XrdEncoder` | PXRD pattern (5250) | 1024 |
| `MetalEncoder` | metal prior (65) | 64 |
| `LinkerEncoder` | linker graph (GCN) | 64 |

`cif_features = lattice(32) + atom(256) = 288`, and
`combined_features = xrd(1024) + metal(64) + linker(64) = 1152`. Both branches are
projected to a shared 256-dim space and trained with an in-batch contrastive loss.
For generation only the `feature_embedding` (PXRD + metal + linker branch) is used.

### Generation model

The lattice is diffused with a DDPM Gaussian process (`DDPMScheduler`,
`squaredcos_cap_v2`), while the fractional coordinates are diffused on a torus with
wrapped Gaussian noise (`ScoreSdeVeSchedulerWrapped`). The denoiser is a CSPNet
whose node features fuse the atom type embedding, the per-atom PXRD embedding and
the time embedding before message passing. Sampling uses the predictor-corrector
scheme of the original implementation.

The CG atom types are `0` (metal node) and `1` (organic linker).

---

## Data Preparation

### Bundled test dataset

The packaged dataset (`xrd2mof_cgmof50_data.zip`) contains 16 CG samples derived
from the four bundled PXRD test cases (BIFWIY, DAKZAS, DUDZOR, VAWXAS01), covering
the valence-consistent atom-count configurations of each MOF:

| File | Content | Samples |
|------|---------|---------|
| `train.npz` | CG structures + PXRD embeddings | 12 |
| `val.npz` | CG structures + PXRD embeddings | 4 |
| `test.npz` | PXRD embeddings + CG atom types + ids | 16 |

The archive is downloaded automatically on first use. Each npz file stores the
variable-length fields (`atom_types`, `frac_coords`) concatenated over samples,
with per-sample sizes in `num_atoms`:

| Field | Shape | Notes |
|-------|-------|-------|
| `feature_embedding` | `[N, 256]` | PXRD embedding |
| `num_atoms` | `[N]` | CG atom count per sample |
| `atom_types` | `[sum(num_atoms)]` | `0` metal node, `1` linker |
| `frac_coords` | `[sum(num_atoms), 3]` | training/validation only |
| `lengths` | `[N, 3]` | training/validation only |
| `angles` | `[N, 3]` | training/validation only |
| `mof_id` | `[N]` | test only |

> Note: the bundled training/validation structures are synthesized with a fixed
> seed (the reference CG coordinates are not distributed with the test cases) and
> are intended for smoke training and loss analysis only. Real training requires
> the full CSD-derived dataset.

### Preparing custom data

1. Run the original preprocessing pipeline (`create_data/` and `pretrained_model/`
   in the upstream repository) to obtain `dataset_emb.pt`, which stores one
   `{mof_id, feature_embedding}` record per PXRD pattern.
2. Compute the valence-consistent CG configurations for each MOF (see
   `cal_num_type_to_diffusion_single.py` upstream) to obtain `num_atoms` and
   `atom_types`.
3. Store the arrays as npz files following the table above, place them under
   `./data/xrd2mof_cgmof50/`, and the dataset classes will pick them up.

### Feature extraction (optional)

`Xrd2MofFeatureExtractor` can be used to compute embeddings from raw PXRD data
(`dataset.pt` produced by `create_data/`). The pretrained weights ship inside the
model package at `checkpoints/feature_extractor.pdparams`:

```python
import paddle
from ppmat.models.xrd2mof.feature_extractor import Xrd2MofFeatureExtractor

model = Xrd2MofFeatureExtractor()
model.set_state_dict(paddle.load("checkpoints/feature_extractor.pdparams"))
model.eval()
outputs = model.extract(batch)  # batch: cif_lattice/cif_atom_positions/xrd/metal/linker
feature_embedding = outputs["feature_embedding"]
```

---

## Environment

- PaddlePaddle 3.0+ (tested with 3.3.0)
- pymatgen, numpy

The generation model reuses `ppmat/schedulers` (`DDPMScheduler`,
`ScoreSdeVeSchedulerWrapped`) and `ppmat/utils/crystal.py`.

---

## Training

```bash
python structure_generation/train.py \
    -c structure_generation/configs/xrd2mof/xrd2mof_cgmof50.yaml \
    Trainer.max_epochs=4000
```

Training configuration:

| Setting | Value |
|---------|-------|
| Optimizer | Adam, lr 1e-3, beta (0.9, 0.999) |
| LR scheduler | ReduceLROnPlateau (factor 0.6, patience 30, min lr 1e-4) |
| Lattice loss weight | 0.1 |
| Coordinate loss weight | 1.0 |
| Diffusion steps | 1000 |
| Batch size | 4 |

### Loss analysis

With the bundled dataset the evaluation loss decreases from `6.49` (epoch 1) to
`4.37` (epoch 2). A deterministic 15-step training comparison against the original
PyTorch implementation (identical initial weights, data, optimizer and fixed
timestep/noise schedule) gives:

| Metric | Value |
|--------|-------|
| loss max abs diff | 1.39e-3 |
| loss mean abs diff | 2.49e-4 |
| final loss (PD / PT) | 1.08313 / 1.08308 |

The remaining difference comes from the independently estimated wrapped-normal
normalization table (`sigmas_norm`), which is sampled with the framework RNG at
scheduler construction time.

---

## Inference

The model output is a coarse-grained structure (metal-node / linker sites), so it
is not converted to CIF here; downstream atomic assembly consumes the raw result.

```python
import numpy as np
import paddle
from ppmat.datasets import build_dataloader
from ppmat.models import build_model_from_name

model, config = build_model_from_name("xrd2mof_cgmof50")
model.eval()
loader = build_dataloader(config["Sample"]["data"])
batch = next(iter(loader))
outputs = model.sample(batch, num_inference_steps=1000)
for item in outputs["result"]:
    lattice = np.asarray(item["lattice"])       # [3, 3]
    frac_coords = np.asarray(item["frac_coords"])  # [num_atoms, 3]
    atom_types = item["atom_types"]             # 0 metal node, 1 linker
    mof_id = item["mof_id"]
```

Sampling settings: 1000 inference steps, predictor-corrector step size
(`snr`) 1e-5.

### Sampling

Sampling goes through the model's own `sample` method. The model output is a
coarse-grained structure (metal-node / linker sites) rather than a fully
atomistic crystal, so it is not converted to a CIF here and the config
intentionally does not declare `Sample.build_structure_cfg` (CIF assembly does
not apply to coarse-grained sites). Persistence is caller-side as well; the
fields of each result item (`lattice`, `frac_coords`, `atom_types`,
`num_atoms`, `mof_id`) mirror the dataset npz schema:

```python
import os

import numpy as np
from ppmat.datasets import build_dataloader
from ppmat.models import build_model_from_name

output_dir = "results/xrd2mof"
os.makedirs(output_dir, exist_ok=True)

model, config = build_model_from_name("xrd2mof_cgmof50")
model.eval()
loader = build_dataloader(config["Sample"]["data"])
batch = next(iter(loader))
result = model.sample(batch, num_inference_steps=1000)["result"]
for index, item in enumerate(result):
    np.savez(
        os.path.join(output_dir, f"{item['mof_id']}_{index + 1}.npz"),
        lattice=np.asarray(item["lattice"], dtype="float32"),
        frac_coords=np.asarray(item["frac_coords"], dtype="float32"),
        atom_types=np.asarray(item["atom_types"], dtype="int64"),
        num_atoms=np.asarray(item["num_atoms"], dtype="int64"),
        mof_id=str(item["mof_id"]),
    )
```

> Note: the `StructureSampler` modes other than dataloader-driven sampling
> target fully atomistic models: this model expects coarse-grained atom types
> (`0` metal node, `1` organic linker) plus a per-sample XRD
> `feature_embedding`, so element/count inputs are not applicable and the
> model fails fast on the missing `feature_embedding` key.

There is no CIF-based evaluation command; correctness is checked by
deterministic numerical alignment with the original PyTorch implementation (see
below).

---

## Verification Results

Deterministic forward alignment against the original PyTorch implementation
(reference checkpoint `epoch=2284-step=162234.ckpt`). These numbers document the
migration-time cross-check against the upstream reference; they are not produced
by a script shipped in this repository.

| Component | Metric | Value |
|-----------|--------|-------|
| Generation decoder (t=100/200/500/800) | pred max abs diff | 3.34e-6 |
| Training loss (fixed timestep/noise, `sigmas_norm` aligned) | max abs diff | 2.0e-5 |
| Feature extractor (`cif_embedding`) | max abs diff | 9.18e-6 |
| Feature extractor (`feature_embedding`) | max abs diff | 1.25e-5 |

Scheduler notes:

- `alphas_cumprod` differs by at most `3.84e-4` because the original cosine beta
  schedule clips betas to `[1e-4, 0.9999]` while the shared `DDPMScheduler` uses a
  `0.999` upper bound. The effect on the deterministic loss is below `2e-3`.
- `sigmas_norm` is a Monte-Carlo estimate drawn at scheduler construction time, so
  it differs between frameworks by a few percent. Injecting the PyTorch table
  reproduces the training loss within `2.5e-4` on average.

---

## Citation

```bibtex
@article{feng2026xrd2mof,
  title   = {Interpreting {X}-ray Diffraction Patterns of {M}etal-{O}rganic Frameworks via Generative Artificial Intelligence},
  author  = {Feng, Bin and Wang, Bingxu and Lv, Linpeng and Zhang, Mingzheng and Chen, Zhefeng and Pan, Feng and Li, Shunning},
  journal = {Journal of the American Chemical Society},
  volume  = {148},
  number  = {1},
  pages   = {869--878},
  year    = {2026},
  doi     = {10.1021/jacs.5c16416}
}
```

The upstream implementation is available at
[PKUsam2023/Xrd2Mof](https://github.com/PKUsam2023/Xrd2Mof).
