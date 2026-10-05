# FlowMol

[FlowMol3: Flow Matching for 3D De Novo Small-Molecule Generation](https://arxiv.org/abs/2508.12629)

## Abstract

FlowMol is a flow matching model for unconditional 3D de novo small-molecule
generation. The current version is FlowMol3, which generates molecular
conformers jointly with atom types, formal charges and bond orders. FlowMol3
combines continuous flow matching for atomic coordinates with CTMC
(continuous-time Markov chain) discrete flow matching for the categorical
modalities, and achieves state-of-the-art validity and novelty among
non-equivariant baselines while remaining cheap to sample.

## Model Description

### Overview

A molecule is represented as:
- atom positions: $X = (x_1, \ldots, x_N),\; x_i \in \mathbb{R}^3$
- atom types (one-hot): $A \in \{0,1\}^{N \times K}$
- formal charges (one-hot): $C \in \{0,1\}^{N \times 6}$
- bond orders (one-hot, upper triangle of the complete graph): $E \in \{0,1\}^{M \times 4}$

The training objective interpolates between a prior (a centered Gaussian for
positions, the mask token for every categorical feature) and the data
endpoint, and trains a GVP message passing network to predict the endpoint
$x_1$ from the interpolated state $x_t$.

### Method

#### 1) CTMC probability paths for categorical features

Every categorical feature follows the CTMC probability path of discrete flow
matching. With the mask token as the prior state, the feature is revealed
node-by-node with probability $\alpha(t)$:

$$
p(a_t \mid a_1) = (1 - \alpha_t)\,\delta_{\mathrm{mask}} + \alpha_t\,\delta_{a_1}.
$$

The training target is the true endpoint category, restricted to nodes whose
state is still masked (`ignore_index=-100`); already revealed nodes carry no
supervision signal.

#### 2) Linear interpolation paths for positions

Positions follow the linear probability path

$$
x_t = (1 - \alpha_t)\,x_0 + \alpha_t\,x_1,
$$

where $x_0$ is a centered Gaussian prior aligned to $x_1$ with an optimal
assignment (Hungarian) followed by a rigid Kabsch alignment. The training
target is the endpoint $x_1$ (the "endpoint" parameterization of the velocity
field).

#### 3) GVP denoiser

The endpoint predictor is a Geometric Vector Perceptron (GVP) network over
the batched complete graph: scalar node features (atom/charge/time
embeddings), vector features, edge features and radial basis function
distance embeddings are processed by repeated rounds of message passing,
node position updates and edge updates. An optional self-conditioning
residual layer feeds the previous endpoint prediction back into the
network.

#### 4) Loss

$$
\mathcal{L} = \sum_{f \in \{x, a, c, e\}} w_f\, \mathcal{L}_f,
$$

with MSE for positions and cross entropy for the categorical features, both
optionally scaled by the time-dependent weights
$\mathrm{clip}(\alpha_t / (1-\alpha_t), 0.05, 1.5)$.

#### 5) Sampling

Sampling integrates the learned endpoint field with an Euler scheme over
`tspan = [0, 1]` (250 steps by default). At every step, masked nodes are
stochastically unmasked towards sampled endpoint categories (Campbell
discrete flow, with optional purity sampling that favors high-confidence
predictions) and can be re-masked; positions follow the Euler update.

## Dataset Description

The FlowMol released model is trained on GEOM-Drugs. The training entry in
this repository uses the QM9 subset processed by the upstream
`process_qm9.py` pipeline (97,734 training / 20,042 validation / 13,055 test
molecules, elements C/H/N/O/F).

The `flowmol_qm9` data package provides the processed splits consumed by
`flowmol_qm9.yaml` (positions, one-hot atom types, formal charges, upper
triangle bond types plus the atom-count histogram):

```text
flowmol_qm9/
|-- train_data_processed.npz
|-- val_data_processed.npz
|-- test_data_processed.npz
`-- train_data_n_atoms_histogram.npz
```

The package is downloaded automatically by
`download.get_datasets_path_from_url` when the split file is missing.

## Results

| Model | Dataset | frac_valid_mols (RDKit) | Config | Checkpoint |
|---|---|---|---|---|
| FlowMol3 | GEOM-Drugs (upstream training) | 8/8 molecules decoded (50-step smoke) | [flowmol3.yaml](./flowmol3.yaml) | [flowmol3.zip](http://127.0.0.1/models/flowmol3.zip) |

Sampling determinism note: molecule generation is stochastic; the reported
number is a smoke-level sanity check of the registered checkpoint, not a
paper-metric reproduction. The upstream paper metrics (validity, uniqueness,
novelty) require the full GEOM-Drugs evaluation stack.

## Command

### Training

```bash
python structure_generation/train.py -c structure_generation/configs/flowmol/flowmol_qm9.yaml
```

### Validation

```bash
python structure_generation/train.py -c structure_generation/configs/flowmol/flowmol_qm9.yaml \
    Global.do_eval=True Global.do_train=False \
    Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Testing

```bash
python structure_generation/train.py -c structure_generation/configs/flowmol/flowmol_qm9.yaml \
    Global.do_eval=False Global.do_train=False Global.do_test=True \
    Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Sample

Molecular models sample through the framework-level molecular sampling
interface (`MolecularSampler`); molecules, the SDF export and the model-owned
metrics are a few API calls.

Mode 1: sample from the registered FlowMol3 checkpoint (downloaded
automatically from `MODEL_REGISTRY`):

```python
from ppmat.sampler import MolecularSampler

sampler = MolecularSampler(
    model_name="flowmol3",
    config_overrides=[
        "Sampler.data.dataset.__init_params__.total_num=16",
        "Sampler.model_sample_params.num_inference_steps=250",
        "Sampler.output_dir=./output/flowmol3_samples",
    ],
)
results = sampler.sample_by_dataloader()
molecules = results["pred"]
sampler.model.save_samples(molecules, sampler.output_dir)
```

Mode 2: sample from a local training output (config plus checkpoint):

```python
from ppmat.sampler import MolecularSampler

sampler = MolecularSampler(
    config_path="structure_generation/configs/flowmol/flowmol_qm9.yaml",
    checkpoint_path="./output/flowmol_qm9_t_<time>_s_42/checkpoints/best.pdparams",
    config_overrides=[
        "Dataset.train=null",
        "Sampler.data.dataset.__init_params__.total_num=16",
        "Sampler.model_sample_params.num_inference_steps=250",
        "Sampler.output_dir=./output/flowmol_qm9_samples",
    ],
)
results = sampler.sample_by_dataloader()
sampler.model.save_samples(results["pred"], sampler.output_dir)
```

Python API:

```python
from ppmat.models import build_model_from_name

model, config = build_model_from_name("flowmol3")
result = model.sample(
    {"structure_array": {"num_atoms": [16, 22, 18]}},
    num_inference_steps=250,
)
molecules = result["pred"]
model.save_samples(molecules, "./output/flowmol3_samples")
```

## Citation

```bibtex
@article{dunn2025flowmol3,
  title={FlowMol3: Flow Matching for 3D De Novo Small-Molecule Generation},
  author={Dunn, Ian and Koes, David Ryan},
  journal={arXiv preprint arXiv:2508.12629},
  year={2025},
  url={https://arxiv.org/abs/2508.12629}
}
```

Upstream repository: https://github.com/Dunni3/FlowMol
