# MatInvent

[MatInvent: Accelerating inverse materials design using generative diffusion models with reinforcement learning](https://arxiv.org/abs/2511.03112)

## Abstract

MatInvent is a general and efficient reinforcement learning workflow that optimizes diffusion models for goal-directed crystal generation. MatInvent enables robust optimization for inverse material design tasks with single or multiple target properties. Compatible with diverse diffusion model architectures and property constraints, MatInvent could offer broad applicability in materials discovery.

---

## Model Description

### Overview

The framework operates in three alternating phases:

1. **Sample** -- The diffusion backbone generates a batch of crystal structures.
2. **Score** -- Property calculators (density, HHI, band gap, etc.) evaluate each structure; a scalar reward is computed from the multi-property profile.
3. **Fine-tune** -- The generator is updated with a reward-weighted diffusion loss plus a KL regularizer that prevents the agent from drifting too far from the pretrained prior.

After each iteration, high-reward structures are stored in a replay buffer and a long-term memory for diversity filtering.

### Supported Backbones

| Backbone | Description | Config |
|----------|-------------|--------|
| **MatterGen** | E(3)-equivariant GNN denoiser with joint A/X/L diffusion | `matinvent_mattergen.yaml` |
| **DiffCSP** | EGNN-based CSP denoiser with lattice + coordinate diffusion | `matinvent_diffcsp.yaml` |

The RL agent and the KL prior are both initialized from the registered weight package
named by `RL.pretrained_model_name`, so the RL entry point and the released checkpoint
share a single source of weights.

Adding a backbone also requires a sampler entry in `suites.ModelSuite` and, for joint
atom-type diffusion, the atom scheduler exposed by the backbone itself -- the RL loop keys
its atom-diffusion branches off that capability instead of the backbone name. The training
entry reaches the RL procedure through the model's optional `run_training` hook, so no
backbone name is hardcoded in `structure_generation/train.py`.

### Supported Property Calculators

| Calculator | Tasks | Description |
|------------|-------|-------------|
| **PyMatGen** | `density`, `hhi`, `price`, `abundance`, `log_abundance`, `mcia`, `num_atoms`, `num_elements`, `volume` | On-the-fly property computation via pymatgen |
| **DFTCalc** | any task name | Exports CIF files for an external DFT run and reads the values back from `{root_dir}/{label}.txt` |
| **FairChem** | `bulk_modulus`, `heat_capacity` | Reads precomputed values from `{root_dir}/{label}.txt` produced by an external ML-potential run |

Each entry of `RL.reward_cfg.prop_cfg` selects a calculator through `calculator.__class_name__` and is scaled by `target` / `minv` / `maxv`. A property that cannot be computed is reported as NaN, treated as a failed sample and scored with reward 0. `DFTCalc` and `FairChem` raise `FileNotFoundError` until the external results file exists, so they never silently return placeholder rewards. See the original [MatInvent](https://github.com/jwchen25/MatInvent) repository for full reward configurations.

Only `density` is exercised by the two configs shipped here; the other tasks are kept for
parity with the reference implementation and are not covered by the test suite.

Each calculator writes its raw values to `<run_dir>/rewards/<label>.txt`. That directory
belongs to the RL run (`Reward` injects it), so `root_dir` must not be set in a
`calculator` config. The `mcia` task additionally requires `substrate_path`, the path to a
substrate CIF; reference substrates ship under
`structure_generation/configs/matinvent/substrates/`.

---

## Environment

MatInvent depends on the packages already listed in the repository `requirements.txt`:

- **PaddlePaddle** official release (validated with 3.3.0, GPU)
- **pymatgen** (`PyMatGen` calculator and CIF handling)
- **SMACT** (crustal abundance of the `abundance` / `log_abundance` tasks)
- **ase** (writing the sampled structures to `.extxyz`)

GPU is strongly recommended: the two shipped configs run the diffusion sampler and the
reward calculators inside the RL loop.

---

## Dataset Description

### Atom count sampling

Sampled atom counts are drawn from the same empirical distribution the task
`Sample.data` uses (`NumAtomsCrystalDataset` with `dist_name: ALEX_MP_20`, i.e. 1..20
atoms per structure) instead of a hand-rolled uniform range, and every sampled
structure is filtered before scoring:

| Check | Criterion |
|-------|-----------|
| Geometry | minimum interatomic distance 0.5 A, volume > 0.1 A^3 |
| Cell size | largest lattice parameter < 25 A |

The reference `invalid_filter` adds a third check, SMACT composition validity, which is
not ported: its cost grows with the number of distinct elements (about 0.2 s for ten
elements and far worse beyond that), while the DiffCSP sampler starts from random atom
types and routinely yields samples with more than ten elements.

The reference implementation draws the DiffCSP counts from its own `mp_20` table (same
1..20 support, different probabilities) and the MatterGen counts from `ALEX_MP_20`.

DiffCSP additionally starts from random atom types, translated to a uniform draw over
1..100 elements (the reference initialises sampling with random atom-type logits and
takes their argmax).

### Data preparation

For property-conditioned RL, rewards are computed **on-the-fly** by property calculators,
so no pre-labeled training set is required and both shipped configs run directly on a
fresh checkout.

Novelty and stability of generated structures are assessed against a reference convex-hull
dataset (`reference_MP2020correction.gz`) available from
[Hugging Face (jwchen25/MatInvent)](https://huggingface.co/jwchen25/MatInvent). This
dataset is only needed for post-hoc evaluation and is not used by the RL loop.

The RL-side dataset (`RLDataset` / `create_rl_dataloader` / the sample validity filter)
lives in `ppmat/models/matinvent/dataset.py` rather than `ppmat/datasets/`: its samples
carry the RL reward and are fed straight to the diffusion backbones' `num_atoms` /
`atom_types` conventions, so it belongs to the model's training algorithm rather than to
the shared task-dataset layer.

### Pretrained weights

The released packages are registered in `MODEL_REGISTRY` and downloaded automatically:

| Model name | Backbone |
|------------|----------|
| `matinvent_mattergen_mp20` | MatterGen |
| `matinvent_diffcsp_mp20` | DiffCSP |

Each package follows the standard layout, so `build_model_from_name` resolves it without
extra arguments:

```
<model_name>.zip
└── <model_name>/
    ├── <model_name>.yaml
    └── checkpoints/
        └── best.pdparams
```

---

## Configuration

The RL entry point replaces `BaseTrainer`, so only a subset of the usual task sections is
consumed:

| Section | Consumed keys |
|---------|---------------|
| `Global` | `num_train_timesteps`, `num_inference_steps`, `rl_num_inference_steps` |
| `Trainer` | `seed`, `output_dir` |
| `Model` | `MatinventRL` (`model_type`, `num_inference_steps`, `backbone_cfg`) |
| `RL` | the whole RL loop, see below |
| `Sample` | `data`, `build_structure_cfg`, `model_sample_params` (sampling with `sample.py`) |

`Dataset`, `Optimizer` and the training related `Trainer` fields are not used: the sample
-> score -> fine-tune loop is driven by the `RL` section, and the fine-tuning optimizer
lives in `RL.finetune_cfg.optimizer_cfg`.

### RL knobs

| Key | Meaning |
|-----|---------|
| `pretrained_model_name` | `MODEL_REGISTRY` key used for both the agent and the KL prior |
| `rl_epoch` | Number of sample -> score -> fine-tune iterations |
| `topk_ratio` | Fraction of the fine-tuning batch kept by reward ranking |
| `save_freq` | Write `models/loop_NNNN` every N iterations (`models/final` is always written) |
| `sample_cfg.batch_size` / `sample_cfg.num_batches` | Structures per sampler call and number of calls per iteration |
| `sample_cfg.num_inference_steps` | Diffusion steps used while sampling during RL |
| `finetune_cfg.batch_size` / `epochs` / `timesteps` / `accum_steps` | Fine-tuning schedule |
| `finetune_cfg.sigma` | Weight of the KL regularizer |
| `finetune_cfg.optimizer_cfg` | Optimizer of the fine-tuning step (`build_optimizer` format) |
| `reward_cfg.reward_threshold` / `reduce` / `prop_cfg` | Reward aggregation and property targets |
| `replay_cfg` | Replay buffer; set it to `null` to disable experience replay |
| `div_filter_cfg` | Diversity filter (`enabled`, `tol`, `buff`, `method`) |

---

## Command

### Training (RL Fine-tuning)

```bash
# MatterGen backbone
python structure_generation/train.py \
    -c structure_generation/configs/matinvent/matinvent_mattergen.yaml

# DiffCSP backbone
python structure_generation/train.py \
    -c structure_generation/configs/matinvent/matinvent_diffcsp.yaml
```

The task entry runs the MatInvent RL procedure (sample -> score -> fine-tune) instead of
`BaseTrainer`; any `RL` key can be overridden from the command line, which is convenient
for a short smoke run:

```bash
python structure_generation/train.py \
    -c structure_generation/configs/matinvent/matinvent_diffcsp.yaml \
    RL.rl_epoch=2 RL.save_freq=1
```

`Trainer.output_dir` gets a `_t_<timestamp>_s_<seed>` suffix at startup, so repeated runs
never overwrite each other. Checkpoints, sampled structures and the long-term memory are
written below that timestamped directory:

```
<output_dir>_t_<timestamp>_s_<seed>/
├── models/final/model.pdparams      # RL fine-tuned backbone
├── models/loop_NNNN/model.pdparams  # periodic snapshots
├── samples/step_NNNN_valid.extxyz
├── samples/step_NNNN_eval.extxyz
├── samples/long_term_memory.csv
└── rewards/<label>.txt              # per-property values written by the calculators
```

### Monitoring and evaluation

The RL loop reports its own metrics at every iteration, both on stdout and in `run.log`:

- `reward mean` / `reward std` -- aggregated reward of the accepted samples
- `<property> mean` / `<property> std` -- raw property values (e.g. `density mean`)
- `crystal_num` / `unique_comps` -- size of the long-term memory and its unique compositions
- `burden` -- molecules evaluated per high-reward candidate (reported once at least 100 candidates exist)
- `div_ratio` -- unique compositions divided by all evaluated crystals (reported while the memory is within the evaluation budget)
- `cost` -- cumulative number of scored structures

`samples/long_term_memory.csv` stores every scored structure (structure, composition,
reward, RL step and its CIF string), so the run can be re-analysed offline.

Post-hoc structure metrics (e.g. the SUN ratio of the paper) are not wired into the
shipped configs: `Sample.metrics` is not set, so `sample.py --mode=compute_metric` is not
applicable here. The generated `.extxyz` / `.cif` files can be fed to an external
evaluation harness instead.

### Sample

Generate structures using an RL-fine-tuned checkpoint. Replace `<run_dir>` with the
timestamped directory printed at the start of the training run.

```bash
# Mode 1: Use a registered model (downloads automatically)
python structure_generation/sample.py \
    --model_name='matinvent_mattergen_mp20' \
    --output_path='result_matinvent_mattergen/' \
    --mode='by_dataloader'

# Mode 2: Use a local config and the checkpoint produced by the RL run
python structure_generation/sample.py \
    --config_path='structure_generation/configs/matinvent/matinvent_mattergen.yaml' \
    --checkpoint_path='<run_dir>/models/final/model.pdparams' \
    --output_path='result_matinvent_mattergen/' \
    --mode='by_dataloader'

# DiffCSP backbone: the chemical formula provides the atom types
python structure_generation/sample.py \
    --config_path='structure_generation/configs/matinvent/matinvent_diffcsp.yaml' \
    --checkpoint_path='<run_dir>/models/final/model.pdparams' \
    --output_path='result_matinvent_diffcsp/' \
    --mode='by_chemical_formula' \
    --chemical_formula='LiMnO2'
```

`--mode=by_dataloader` samples the atom counts defined by `Sample.data` (the
`NumAtomsCrystalDataset` of the config), while `--mode=by_chemical_formula` uses the
composition given on the command line. The DiffCSP backbone needs explicit atom types,
which is why the DiffCSP example uses a chemical formula.

---

## Results

Both configs were validated end-to-end on the released weights with the RL loop shortened
to three iterations (`RL.rl_epoch=3 RL.save_freq=1`, GPU):

| Backbone | Iteration | `density mean` | `reward mean` | `crystal_num` | `div_ratio` |
|----------|-----------|----------------|---------------|---------------|-------------|
| MatterGen | 0 | 2.160 | 0.216 | 2 | 1.0 |
| MatterGen | 1 | 2.786 | 0.279 | 4 | 1.0 |
| MatterGen | 2 | 1.620 | 0.162 | 6 | 1.0 |
| DiffCSP | 0 | 11.602 | 1.000 | 1 | 1.0 |
| DiffCSP | 1 | 8.790 | 0.879 | 2 | 1.0 |
| DiffCSP | 2 | 8.445 | 0.844 | 3 | 1.0 |

`sample.py` on the resulting checkpoints writes 4 CIF files for MatterGen
(`--mode=by_dataloader`) and 1 CIF file for DiffCSP (`--mode=by_chemical_formula`). These
numbers come from a three-iteration smoke run, not from the full 50-iteration schedule used
for the paper; quantitative comparisons with the reference implementation are reported in
the [paper](https://arxiv.org/abs/2511.03112).

### Hyperparameters

`RL.finetune_cfg.sigma` and `RL.finetune_cfg.optimizer_cfg.lr` follow the reference
implementation (`sigma=0.025`, `lr=1e-5` for MatterGen and `1e-4` for DiffCSP). Keep them
there: with the much larger `sigma=1.0` / `lr=5e-4` the KL term of the MatterGen agent
diverges within one iteration (the logged `loss_kl` grows from ~1 to thousands) and the
resulting checkpoint can no longer be sampled.

The remaining RL knobs were re-tuned for this port and differ from
`jwchen25/MatInvent`:

| Knob | Reference | Here |
|------|-----------|------|
| `finetune_cfg.timesteps` | 1000 | 100 |
| `finetune_cfg.epochs` | 3 | 2 |
| `finetune_cfg.accum_steps` | 50 | 4 |
| `replay_cfg.buffer_size` | 100 | 1000 |
| `replay_cfg.sample_size` | 10 | 32 |
| `replay_cfg.reward_cutoff` | 0.1 | 0.0 |
| `div_filter_cfg.tol` / `buff` | 3 / 6 | 10 / 20 |
| `sample_cfg.batch_size` / `num_batches` | `eval_size * 4` (MatterGen) / `eval_size * 12` (DiffCSP) | 1 / 2 (MatterGen), 1 / 1 (DiffCSP) |
| Atom count distribution (DiffCSP) | its own `mp_20` table | `ALEX_MP_20`, shared with `Sample.data` |
| Fine-tuning batch | one batch holding all fine-tuning samples | mini-batches of `finetune_cfg.batch_size` |

Aligning them with the reference implementation is left to the user through the `RL`
config section.

### Not ported from the reference

The following reference features are intentionally absent; none of them is needed by the
shipped configs:

- MLIP relaxation between sampling and scoring (`sample_cfg.mlip_opt`) and the
  metric-based sample filter / cap (`OptFilter`, `sample_cfg.filter`, `sample_cfg.max_num`):
  both are Hydra-instantiated callables with no PaddleMaterials configuration equivalent.
- The SMACT composition filter (`is_smact_valid`), see "Atom count sampling" above for
  the measured reason.
- Advantage baselines (`LongTimeMem.get_baseline`, the `baseline` argument of the
  reference `ft_step`): the reference computes them but the advantage is the raw reward.
- `LongTimeMem.sample` / `deduplicate`: unused in the reference (the replay buffer owns
  those operations).
- Robustness guards: the reference RL loop crashes when a round yields no valid samples
  or when the long-term memory is still empty; here such a round is logged and skipped.

---

## Citation

```bibtex
@article{matinvent,
  title={Accelerating inverse materials design using generative diffusion models with reinforcement learning},
  author={Chen, Junwu and Guo, Jeff and Fako, Edvin and Schwaller, Philippe},
  journal={arXiv preprint arXiv:2511.03112},
  year={2025}
}
```
