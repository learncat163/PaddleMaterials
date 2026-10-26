# MatInvent

[MatInvent: Accelerating inverse materials design using generative diffusion models with reinforcement learning](https://arxiv.org/abs/2511.03112)

## Abstract

MatInvent is a general and efficient reinforcement learning workflow that optimizes diffusion models for goal-directed crystal generation. MatInvent enables robust optimization for inverse material design tasks with single or multiple target properties. Compatible with diverse diffusion model architectures and property constraints, MatInvent could offer broad applicability in materials discovery.

![MatInvent RL Pipeline](../../docs/matinvent_RL_pipeline.png)

---

## Model Description

### Overview
MatInvent adapts pretrained crystal diffusion models for goal-directed materials generation through reinforcement learning. Each round generates crystal structures, evaluates their properties, and fine-tunes the generator using the resulting rewards. The implementation supports MatterGen and DiffCSP backbones.

| Backbone | Generation task | Config |
| --- | --- | --- |
| MatterGen | Joint generation of atom types, coordinates, and lattice | [matinvent_mattergen.yaml](matinvent_mattergen.yaml) |
| DiffCSP | Coordinate and lattice generation with fixed atom types | [matinvent_diffcsp.yaml](matinvent_diffcsp.yaml) |
| DiffCSP | Joint generation of atom types, coordinates, and lattice | [matinvent_diffcsp_joint.yaml](matinvent_diffcsp_joint.yaml) |

The MatterGen and conditional DiffCSP configurations optimize density. The joint DiffCSP configuration optimizes the Herfindahl-Hirschman Index (HHI) for supply risk, with relaxation, stability, novelty, and uniqueness screening. Reward properties and targets are configured under `Model.__init_params__.rl_cfg.reward_cfg`.

### Method
For a generated structure with reward $r$, fine-tuning minimizes the reward-weighted diffusion objective with regularization toward a frozen pretrained prior:

$$
\mathcal{L} = r\,\mathcal{L}_{\mathrm{diff}} + \sigma\,(1.1-r)\,\mathcal{L}_{\mathrm{prior}}.
$$

Here, $\mathcal{L}_{\mathrm{diff}}$ combines the coordinate, lattice, and, for joint generation, atom-type denoising losses. $\mathcal{L}_{\mathrm{prior}}$ measures the mean squared difference between the agent and prior predictions; $\sigma$ controls its weight.

Structures are ranked by reward before fine-tuning. A replay buffer retains high-reward structures, while a composition-based diversity filter reduces rewards for repeatedly generated compositions. The agent and frozen prior are initialized from the same pretrained backbone.

---

## Dataset Description

RL training uses generated structures and property rewards rather than a fixed labeled training split. The standard configurations draw atom counts from the Alex-MP-20 distribution, with 1 to 20 atoms per structure. The joint DiffCSP configuration uses the MP-20 atom-count distribution and limits sampled structures to 16 atoms.

| Data | Usage | Source |
| --- | --- | --- |
| Generated crystals | Online reward evaluation and fine-tuning | Current diffusion model |
| Pretrained backbones | Agent and frozen-prior initialization | Registered MatInvent packages; original checkpoints are available on [Hugging Face](https://huggingface.co/jwchen25/MatInvent) |
| MP-20 validation structures | Joint DiffCSP validation probe | [MP-20](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp_20/mp_20.zip) |
| MP2020-corrected reference database | Joint DiffCSP stability and novelty screening | [reference_MP2020correction.gz](https://huggingface.co/jwchen25/MatInvent/blob/main/reference_MP2020correction.gz) |

For joint DiffCSP training, place the reference archive at `data/matinvent/reference_MP2020correction.gz` and the MP-20 validation file at `data/mp_20/val.csv`, or update `qualification_cfg.reference_path` and `early_stop_cfg.probe_cfg.data_path` under `Model.__init_params__.rl_cfg`. Relaxation uses the registered `mattersim_5M` model.

---

## Results

| Model | Dataset | Full RL Evaluation | GPUs | Training Time | Config | Checkpoint / Log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| matinvent_mattergen_mp20 | Generated crystals | Not evaluated | - | - | [matinvent_mattergen.yaml](matinvent_mattergen.yaml) | [pretrained checkpoint](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/MatInvert/matinvent_mattergen_mp20.zip) |
| matinvent_diffcsp_mp20 | Generated crystals | Not evaluated | - | - | [matinvent_diffcsp.yaml](matinvent_diffcsp.yaml) | Local registry package |
| matinvent_diffcsp_mp20_joint | Generated crystals | Not evaluated | - | - | [matinvent_diffcsp_joint.yaml](matinvent_diffcsp_joint.yaml) | Local registry package |

The checkpoints initialize the pretrained backbones; they do not represent completed RL runs. Full-run generation quality and training times have not been established. The DiffCSP packages currently use the local model server configured in `MODEL_REGISTRY`.

---

## Command

### Training
```bash
# MatterGen backbone
python structure_generation/train.py -c structure_generation/configs/matinvent/matinvent_mattergen.yaml

# DiffCSP backbone with fixed atom types
python structure_generation/train.py -c structure_generation/configs/matinvent/matinvent_diffcsp.yaml

# Joint DiffCSP generation; prepare the reference database and validation data first
python structure_generation/train.py -c structure_generation/configs/matinvent/matinvent_diffcsp_joint.yaml
```

### Validation
The RL loop reports rewards, property means, and retained structure counts after each round. Joint DiffCSP additionally evaluates the MP-20 probe configured under `Model.__init_params__.rl_cfg.early_stop_cfg`. The configurations do not define a standalone validation split for the unified trainer.

### Testing
The configurations do not define a test split. Use the sampling commands below to generate structures from a pretrained or fine-tuned checkpoint.

### Sample
```bash
# Predict crystal structures using a trained model.
# Mode 1: Use a pre-trained model (downloads automatically).
# Mode 2: Use a custom configuration file and checkpoint.
# Results are saved to the folder specified by --output_path (default: results).

# Mode 1: pre-trained MatterGen backbone
python structure_generation/sample.py --model_name='matinvent_mattergen_mp20' --output_path='result_matinvent_mattergen/' --mode='by_dataloader'

# Mode 2: MatterGen checkpoint produced by an RL run
python structure_generation/sample.py --config_path='structure_generation/configs/matinvent/matinvent_mattergen.yaml' --checkpoint_path='<run_dir>/models/final/model.pdparams' --output_path='result_matinvent_mattergen/' --mode='by_dataloader'

# Mode 2: conditional DiffCSP checkpoint; the formula supplies the atom types
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
