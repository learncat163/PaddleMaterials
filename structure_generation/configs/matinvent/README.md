# MatInvent

Paper: [MatInvent: Combining Diffusion Models with Reinforcement Learning for Crystal Invention](https://arxiv.org/abs/2503.09787)

## Introduction

MatInvent is a crystal generation framework that combines diffusion model with RL fine-tuning loop. It uses a pre-trained diffusion model (MatterGen or DiffCSP) as backbone, generates candidate crystal structures, scores them by property calculator, and fine-tunes the generator with reward-weighted loss. After several rounds, the generator can produce new stable crystals that match user-specified property targets.

## Config Files

| Config | Backbone | Default Reward Target |
|--------|----------|-----------------------|
| `matinvent_mattergen.yaml` | MatterGen | band_gap (ascending) |
| `matinvent_diffcsp.yaml`   | DiffCSP   | formation_energy (descending) |

Both configs follow the same structure as normal training configs, but add an extra `RL:` section.

## Key RL Parameters

| Parameter | Description |
|-----------|-------------|
| `rl_epoch` | how many RL outer-loop rounds to run |
| `topk_ratio` | ratio of top structures to keep for fine-tuning each round |
| `sample_cfg.num_samples` | how many structures to generate per round |
| `finetune_cfg.lr` | learning rate for fine-tuning |
| `reward_cfg.prop_cfg` | target property list (name, direction, value range) |
| `replay_cfg.capacity` | max size of replay buffer |
| `div_filter_cfg.method` | diversity filter method (`composition` or `rmsd`) |

## Usage

### 1. RL Fine-tuning (统一入口)

```bash
# MatterGen backbone
python structure_generation/train.py \
    --config structure_generation/configs/matinvent/matinvent_mattergen.yaml

# DiffCSP backbone
python structure_generation/train.py \
    --config structure_generation/configs/matinvent/matinvent_diffcsp.yaml
```

### 2. Sampling

```bash
# Using pre-trained model
python structure_generation/sample.py \
    --config_path structure_generation/configs/matinvent/matinvent_mattergen.yaml \
    --checkpoint_path diff-matinvent/tmp/matinvent_mattergen_mp20.pdparams \
    --save_path results/matinvent_mattergen \
    --mode by_dataloader

# Using RL-fine-tuned model
python structure_generation/sample.py \
    --config_path structure_generation/configs/matinvent/matinvent_mattergen.yaml \
    --checkpoint_path output/matinvent_mattergen/models/final/model.pdparams \
    --save_path results/matinvent_mattergen_rl \
    --mode by_dataloader
```

For DiffCSP backbone, just replace the config file with `matinvent_diffcsp.yaml`.
