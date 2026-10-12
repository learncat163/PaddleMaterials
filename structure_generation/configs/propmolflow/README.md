# PropMolFlow

[PropMolFlow: property-guided molecule generation with geometry-complete flow matching](https://www.nature.com/articles/s43588-025-00946-y)

## Abstract

Molecule generation is advancing rapidly in chemical discovery and drug design. Flow-matching methods have recently set the state of the art (SOTA) in unconditional molecule generation, surpassing score-based diffusion models. However, diffusion models still lead in property-guided generation. In this work, we introduce PropMolFlow, an approach for property-guided molecule generation based on geometry-complete SE(3)-equivariant flow matching. Integrating five different property embedding methods with a Gaussian expansion of scalar properties, PropMolFlow achieves competitive performance against previous SOTA diffusion models in conditional molecule generation while maintaining high structural stability and validity. Additionally, it enables higher sampling speed with fewer time steps compared with baseline models. We highlight the importance of validating the properties of generated molecules through density functional theory calculations. Furthermore, we introduce a task to assess the model's ability to propose molecules with under-represented property values, assessing its capacity for out-of-distribution generalization.

![PropMolFlow Overview](../../docs/propmolflow_overview_page.jpg)

---

## Model Description

### Overview
A molecule is represented as a complete atom graph with:
- atom types (composition): $A = (a_1,\ldots,a_N)$
- formal charges: $C = (c_1,\ldots,c_N)$
- 3D coordinates: $X = (x_1,\ldots,x_N),\; x_i \in \mathbb{R}^3$
- bond types: $E$ over every atom pair

PropMolFlow is a geometry-complete SE(3)-equivariant flow matching model for **property-guided molecule generation**. It defines separate flow-matching probability paths for the discrete features (atom types, charges, bond types) and the continuous coordinates, and trains a GVP (Geometric Vector Perceptron) message passing network to predict the trajectory endpoint. The target molecular property is injected into the network through a property embedding module.

### Method

#### 1) Geometry-complete probability paths
The interpolant scheduler assigns a time-dependent weight $\alpha_t \in [0,1]$ to every feature. Coordinates follow linear interpolation between a centered-normal prior $x_0$ and the data coordinate $x_1$:

$$
x_t = (1 - \alpha_t)\, x_0 + \alpha_t\, x_1,
\qquad
v_t = \frac{\alpha_t'}{1 - \alpha_t}\,(x_1 - x_t).
$$

The categorical features (atom types, charges, bond types) follow CTMC (continuous-time Markov chain) probability paths. A masked state $\mathrm{[MASK]}$ is introduced, and each token either stays at its endpoint or becomes masked:

$$
x_t =
\begin{cases}
x_1, & \text{with probability } \alpha_t, \\
\mathrm{[MASK]}, & \text{with probability } 1 - \alpha_t.
\end{cases}
$$

The weight follows a linear ($\alpha_t = t$) or a cosine ($\alpha_t = 1 - \cos(\frac{\pi}{2} t^{\nu})^2$) schedule.

#### 2) SE(3)-equivariant GVP network
The endpoint is predicted by a GVP message passing network. Node scalar features are initialized from the atom types, the formal charges and the time, and edge features from the bond types; the property embedding is fused into the node scalars. A stack of GVP convolutions performs message passing, and every `convs_per_update` convolutions a position updater and an edge updater refresh the coordinates and the edge features. The output heads predict:
- **invariant** logits for atom types and formal charges
- **invariant** logits for bond types (from the summed features of both edge directions)
- **equivariant** updated coordinates

#### 3) Property conditioning
The scalar property is embedded by an optional Gaussian expansion followed by an MLP. Five injection strategies connect the embedding to the node scalar features: `concatenate`, `sum`, `multiply`, `concatenate_sum` and `concatenate_multiply`.

#### 4) Training objective
The training objective is a weighted sum of a coordinate loss and the categorical cross-entropy losses:

$$
\mathcal{L} =
w_X\, \lVert x_1 - \hat{x}_1 \rVert_2^2
+ w_A\, \mathcal{L}_{\mathrm{CE}}(A, \hat{A})
+ w_C\, \mathcal{L}_{\mathrm{CE}}(C, \hat{C})
+ w_E\, \mathcal{L}_{\mathrm{CE}}(E, \hat{E}),
$$

where the coordinate loss is evaluated on every atom and the cross-entropy terms on the masked tokens only.

#### 5) Sampling
Sampling integrates the learned vector field from $t = 0$ to $t = 1$. At every step the network predicts the endpoint, the coordinates are advanced along the linear velocity $v_t$ and the categorical tokens transition according to the CTMC path. An optional purity (high-confidence) threshold and a configurable stochasticity control the trajectory.

---

## Dataset Description

- **rQM9_v0 (corrected QM9)**: 133,174 small molecules with elements C, H, N, O and F. It provides the 15 QM9 property columns (A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv) plus four per-atom energy columns. The released split is 100,000 / 19,857 / 13,317 for train / val / test, and the training set is additionally split into two halves `train_a` and `train_b` of 50,000 molecules each.
- **Bonds and charges**: five bond types over the complete atom graph and formal atom charges. The generation model supervises these fields, which the plain QM9 dataset does not provide.
- **Normalization**: every property is normalized with the train-split statistics, so the shared `train_data_property_normalization.npz` is used by all splits.

Recommended data fields for each sample:
- `atom_types`: length-$N$ atom types (C, H, N, O, F)
- `atom_charges`: length-$N$ formal charges
- `positions`: $N \times 3$ centered Cartesian coordinates
- `bond_types`: bond-type labels over the complete graph
- `property`: the conditioned scalar property (for example `alpha`, `mu`, `gap`)

Optional fields include `num_atoms`.

#### QM9 split (download link)
| Dataset | Train | Val | Test |
| --- | --- | --- | --- |
| [rQM9_v0](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/propmolflow_qm9/propmolflow_qm9.zip) | 100000 | 19857 | 13317 |

---

## Results

| Model Name | Dataset | Property | Property Handling | Distribution | RDKit Valid | Valid & Unique | Config | Checkpoint / Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| propmolflow_qm9_alpha_sum | qm9 | alpha | sum | in-distribution | 0.9844 | 0.9844 | [propmolflow_qm9_alpha_sum.yaml](propmolflow_qm9_alpha_sum.yaml) | [checkpoint / log](https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/propmolflow/propmolflow_qm9_alpha_sum.zip) |
| propmolflow_qm9_Cv_sum | qm9 | Cv | sum | in-distribution | 0.9844 | 0.9844 | [propmolflow_qm9_Cv_sum.yaml](propmolflow_qm9_Cv_sum.yaml) | [checkpoint / log](https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/propmolflow/propmolflow_qm9_Cv_sum.zip) |
| propmolflow_qm9_gap_concatenate_sum | qm9 | gap | concatenate_sum | in-distribution | 1.0000 | 1.0000 | [propmolflow_qm9_gap_concatenate_sum.yaml](propmolflow_qm9_gap_concatenate_sum.yaml) | [checkpoint / log](https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/propmolflow/propmolflow_qm9_gap_concatenate_sum.zip) |
| propmolflow_qm9_homo_concatenate | qm9 | homo | concatenate | in-distribution | 1.0000 | 1.0000 | [propmolflow_qm9_homo_concatenate.yaml](propmolflow_qm9_homo_concatenate.yaml) | [checkpoint / log](https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/propmolflow/propmolflow_qm9_homo_concatenate.zip) |
| propmolflow_qm9_lumo_concatenate_sum | qm9 | lumo | concatenate_sum | in-distribution | 0.9688 | 0.9688 | [propmolflow_qm9_lumo_concatenate_sum.yaml](propmolflow_qm9_lumo_concatenate_sum.yaml) | [checkpoint / log](https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/propmolflow/propmolflow_qm9_lumo_concatenate_sum.zip) |
| propmolflow_qm9_mu_multiply | qm9 | mu | multiply | in-distribution | 0.9844 | 0.9844 | [propmolflow_qm9_mu_multiply.yaml](propmolflow_qm9_mu_multiply.yaml) | [checkpoint / log](https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/propmolflow/propmolflow_qm9_mu_multiply.zip) |
| propmolflow_qm9_gap_concatenate_ood | qm9 | gap | concatenate | out-of-distribution | 0.9688 | 0.9688 | [propmolflow_qm9_gap_concatenate_ood.yaml](propmolflow_qm9_gap_concatenate_ood.yaml) | [checkpoint / log](https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/propmolflow/propmolflow_qm9_gap_concatenate_ood.zip) |
| propmolflow_qm9_homo_concatenate_sum_ood | qm9 | homo | concatenate_sum | out-of-distribution | 1.0000 | 1.0000 | [propmolflow_qm9_homo_concatenate_sum_ood.yaml](propmolflow_qm9_homo_concatenate_sum_ood.yaml) | [checkpoint / log](https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/propmolflow/propmolflow_qm9_homo_concatenate_sum_ood.zip) |
| propmolflow_qm9_lumo_concatenate_multiply_ood | qm9 | lumo | concatenate_multiply | out-of-distribution | 1.0000 | 1.0000 | [propmolflow_qm9_lumo_concatenate_multiply_ood.yaml](propmolflow_qm9_lumo_concatenate_multiply_ood.yaml) | [checkpoint / log](https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/propmolflow/propmolflow_qm9_lumo_concatenate_multiply_ood.zip) |
| propmolflow_qm9_mu_sum_ood | qm9 | mu | sum | out-of-distribution | 0.9844 | 0.9844 | [propmolflow_qm9_mu_sum_ood.yaml](propmolflow_qm9_mu_sum_ood.yaml) | [checkpoint / log](https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/propmolflow/propmolflow_qm9_mu_sum_ood.zip) |

---

## Command

### Training
```bash
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/propmolflow/propmolflow_qm9_alpha_sum.yaml

# multi-gpu training (example with 4 GPUs)
python -m paddle.distributed.launch --gpus="0,1,2,3" structure_generation/train.py -c structure_generation/configs/propmolflow/propmolflow_qm9_alpha_sum.yaml
```

### Validation
```bash
# Adjust program behavior on the fly using command-line parameters without modifying the configuration file directly.
# Example: --Global.do_eval=True
python structure_generation/train.py -c structure_generation/configs/propmolflow/propmolflow_qm9_alpha_sum.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Testing
```bash
# Evaluate the model on the test dataset.
python structure_generation/train.py -c structure_generation/configs/propmolflow/propmolflow_qm9_alpha_sum.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Sample
```python
from ppmat.sampler import MolecularSampler

# Mode 1: Use a pre-trained model (downloads automatically).
# Mode 2: Use a custom configuration file and checkpoint.
sampler = MolecularSampler(
    model_name="propmolflow_qm9_alpha_sum",
    # config_path="structure_generation/configs/propmolflow/propmolflow_qm9_alpha_sum.yaml",
    # checkpoint_path="./output/propmolflow_qm9_alpha_sum/checkpoints/latest.pdparams",
    config_overrides=[
        "Dataset.train=null",
        "Sampler.data.dataset.__init_params__.total_num=64",
        "Sampler.model_sample_params.num_inference_steps=100",
        "Sampler.output_dir=./output/propmolflow_samples",
    ],
)
results = sampler.sample_by_dataloader()
```

Out-of-distribution sampling with a single target value:
```python
    config_overrides=[
        "Sampler.data.dataset.__init_params__.mode=out_of_distribution",
        "Sampler.data.dataset.__init_params__.property_value=93.3",
    ],
```

Sampling with a fixed number of atoms:
```python
    config_overrides=[
        "Sampler.data.dataset.__init_params__.n_atoms=18",
    ],
```

---

## Citation
```
@article{zeng2026propmolflow,
  title={PropMolFlow: property-guided molecule generation with geometry-complete flow matching},
  author={Zeng, Cheng and Jin, Jirui and Ambrose, Connor and Karypis, George and Transtrum, Mark and Tadmor, Ellad B. and Hennig, Richard G. and Roitberg, Adrian and Martiniani, Stefano and Liu, Mingjie},
  journal={Nature Computational Science},
  volume={6},
  pages={233--242},
  year={2026},
  publisher={Nature Publishing Group},
  doi={10.1038/s43588-025-00946-y}
}
```
