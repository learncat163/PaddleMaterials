# MatInvent

[MatInvent: Accelerating inverse materials design using generative diffusion models with reinforcement learning](https://arxiv.org/abs/2511.03112)

## Abstract

MatInvent is a general and efficient reinforcement learning workflow that optimizes diffusion models for goal-directed crystal generation. MatInvent enables robust optimization for inverse material design tasks with single or multiple target properties. Compatible with diverse diffusion model architectures and property constraints, MatInvent could offer broad applicability in materials discovery.

---

## Model Description

### Overview
MatInvent wraps a pretrained crystal diffusion backbone and optimizes it with reward-weighted fine-tuning. Each iteration alternates three phases:
- **Sample**: the backbone generates a batch of crystal structures.
- **Score**: property calculators evaluate each structure and a scalar reward is computed from the multi-property profile.
- **Fine-tune**: the generator is updated with a reward-weighted diffusion loss plus a KL regularizer that keeps it close to the pretrained prior.

After each iteration, high-reward structures are stored in a replay buffer and in a long-term memory used for diversity filtering. Two backbones are supported, both driven through the `MatinventRL` adapter:

| Backbone | Config |
| --- | --- |
| MatterGen | `matinvent_mattergen.yaml` |
| DiffCSP | `matinvent_diffcsp.yaml` |

The RL agent and the KL prior both start from the registered weight package named by `Model.__init_params__.rl_cfg.pretrained_model_name`, so the RL loop and the released checkpoint share a single source of weights and backbone architecture. Rewards come from the calculators selected in `rl_cfg.reward_cfg.prop_cfg`:
- **PyMatGen**: `density`, `hhi`, `price`, `abundance`, `log_abundance`, `mcia`, `num_atoms`, `num_elements`, `volume`
- **DFTCalc**: any task name, read from `{root_dir}/{label}.txt` written by an external DFT run
- **FairChem**: `bulk_modulus`, `heat_capacity`, read from precomputed external values

### Method

#### 1) Reward-weighted fine-tuning
For a sampled structure with reward $r$, the advantage is the raw reward and the generator minimizes

$$
\mathcal{L} = r\,\mathcal{L}_{\text{diff}} + \sigma\,(1.1 - r)\,\mathrm{KL},
$$

where $\mathcal{L}_{\text{diff}}$ is the weighted sum of the coordinate, lattice and atom-type diffusion losses and $\sigma$ is `rl_cfg.finetune_cfg.sigma`.

#### 2) KL regularization to the prior
The KL term is the mean squared difference between the agent and the frozen prior predictions, which keeps the fine-tuned model from drifting away from the pretrained one.

#### 3) Replay buffer and diversity filter
Fine-tuning batches are ranked by reward and truncated to `rl_cfg.topk_ratio`. High-reward structures are kept in a replay buffer (best structure per composition) and appended to a long-term memory; the diversity filter down-weights structures whose composition or element combination is already over-represented.

---

## Dataset Description

No pre-labeled training set is required: rewards are computed **on-the-fly** by the property calculators, so both shipped configs run directly on a fresh checkout.

Sampled atom counts are drawn from the same empirical distribution the task `Sample.data` uses (`NumAtomsCrystalDataset` with `dist_name: ALEX_MP_20`, 1..20 atoms per structure), and every sample is filtered before scoring:

| Check | Criterion |
| --- | --- |
| Geometry | minimum interatomic distance 0.5 A, volume > 0.1 A^3 |
| Cell size | largest lattice parameter < 25 A |

The RL-side dataset (`RLDataset` / `create_rl_dataloader`) lives in `ppmat/models/matinvent/dataset.py`, because its samples carry the RL reward and feed the backbones' `num_atoms` / `atom_types` conventions.

#### Pretrained weights
The released packages are registered in `MODEL_REGISTRY` and downloaded automatically:

| Model name | Backbone |
| --- | --- |
| `matinvent_mattergen_mp20` | MatterGen |
| `matinvent_diffcsp_mp20` | DiffCSP |

Each package follows the standard layout, so `build_model_from_name` resolves it without extra arguments:

```
<model_name>.zip
└── <model_name>/
    ├── <model_name>.yaml
    └── checkpoints/
        └── best.pdparams
```

Novelty and stability of generated structures are assessed against a reference convex-hull dataset (`reference_MP2020correction.gz`) from [Hugging Face (jwchen25/MatInvent)](https://huggingface.co/jwchen25/MatInvent); it is only needed for post-hoc evaluation and is not used by the RL loop.

---

## Results

Both configs were validated end-to-end on the released weights with the RL loop shortened to three iterations (GPU):

| Backbone | Iteration | `density mean` | `reward mean` | `crystal_num` | `div_ratio` |
| --- | --- | --- | --- | --- | --- |
| MatterGen | 0 | 2.160 | 0.216 | 2 | 1.0 |
| MatterGen | 1 | 2.786 | 0.279 | 4 | 1.0 |
| MatterGen | 2 | 1.620 | 0.162 | 6 | 1.0 |
| DiffCSP | 0 | 11.602 | 1.000 | 1 | 1.0 |
| DiffCSP | 1 | 8.790 | 0.879 | 2 | 1.0 |
| DiffCSP | 2 | 8.445 | 0.844 | 3 | 1.0 |

`sample.py` on the resulting checkpoints writes 4 CIF files for MatterGen (`--mode=by_dataloader`) and 1 CIF file for DiffCSP (`--mode=by_chemical_formula`). These numbers come from a three-iteration smoke run, not from the full 50-iteration schedule used for the paper; quantitative comparisons with the reference implementation are reported in the [paper](https://arxiv.org/abs/2511.03112). `rl_cfg.finetune_cfg.sigma` and `rl_cfg.finetune_cfg.optimizer_cfg.lr` follow the reference implementation (`sigma=0.025`, `lr=1e-5` for MatterGen and `1e-4` for DiffCSP).

---

## Command

### Training
```bash
# MatterGen backbone
python structure_generation/train.py -c structure_generation/configs/matinvent/matinvent_mattergen.yaml

# DiffCSP backbone
python structure_generation/train.py -c structure_generation/configs/matinvent/matinvent_diffcsp.yaml

# The unified entry drives the RL loop through MatinventRL.before_train; any rl_cfg key
# can be overridden on the command line, which is convenient for a short smoke run.
python structure_generation/train.py -c structure_generation/configs/matinvent/matinvent_diffcsp.yaml Model.__init_params__.rl_cfg.rl_epoch=2 Model.__init_params__.rl_cfg.save_freq=1
```

### Validation
```bash
# The RL run reports its own metrics (reward, property means, crystal_num, div_ratio,
# cost) on stdout and in run.log at every iteration.
```

### Testing
```bash
# RL fine-tuning is the task itself; the shipped configs do not define a test split.
```

### Sample
```bash
# Predict crystal structures using a trained model.
# Mode 1: Use a pre-trained model (downloads automatically).
# Mode 2: Use a custom configuration file and checkpoint.
# Results are saved to the folder specified by --output_path (default: results).

# Mode 1: pre-trained model
python structure_generation/sample.py --model_name='matinvent_mattergen_mp20' --output_path='result_matinvent_mattergen/' --mode='by_dataloader'

# Mode 2: custom config + checkpoint produced by the RL run
python structure_generation/sample.py --config_path='structure_generation/configs/matinvent/matinvent_mattergen.yaml' --checkpoint_path='<run_dir>/models/final/model.pdparams' --output_path='result_matinvent_mattergen/' --mode='by_dataloader'

# DiffCSP backbone: the chemical formula provides the atom types
python structure_generation/sample.py --config_path='structure_generation/configs/matinvent/matinvent_diffcsp.yaml' --checkpoint_path='<run_dir>/models/final/model.pdparams' --output_path='result_matinvent_diffcsp/' --mode='by_chemical_formula' --chemical_formula='LiMnO2'
```

---

## Citation
```
@article{matinvent,
  title={Accelerating inverse materials design using generative diffusion models with reinforcement learning},
  author={Chen, Junwu and Guo, Jeff and Fako, Edvin and Schwaller, Philippe},
  journal={arXiv preprint arXiv:2511.03112},
  year={2025}
}
```
