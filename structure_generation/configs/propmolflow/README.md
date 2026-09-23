# PropMolFlow

## 1. Introduction

Property-guided molecule generation with geometry-complete flow matching.

- **Task**: generate 3D molecules conditioned on a target molecular property
  (e.g. `alpha`, `mu`, `gap`).
- **Dataset**: corrected QM9 (rQM9_v0), elements C/H/N/O/F, five bond types,
  100000 training molecules.
- **Paper**: Zeng et al., *Nature Computational Science*, 2026.
- **Upstream repository**: https://github.com/Liu-Group-UF/PropMolFlow

The model uses CTMC (continuous-time Markov chain) probability paths for the
discrete features (atom types, charges, bond types) and linear interpolation
paths for coordinates. A GVP (Geometric Vector Perceptron) message passing
network predicts the trajectory endpoints. The conditioned property is injected
through a property embedder; five injection strategies are supported
(`concatenate`, `sum`, `multiply`, `concatenate_sum`, `concatenate_multiply`).

## 2. Data preparation

The `propmolflow_qm9` data package is downloaded and extracted automatically by
`download.get_datasets_path_from_url` into `~/.paddlemat/datasets/`. The package
provides the four splits consumed by the configs plus the shared property
normalization:

```text
propmolflow_qm9/
|-- train_a_data_processed.npz              # default train split (50000 molecules)
|-- train_b_data_processed.npz              # the other half of the training set
|-- val_data_processed.npz                  # validation split
|-- test_data_processed.npz                 # test split
`-- train_data_property_normalization.npz   # per-property mean/std of the train split
```

The splits derive from the corrected rQM9_v0 release of the upstream repository
(https://github.com/Liu-Group-UF/PropMolFlow): the property table
`gdb9.sdf.csv` is committed there under `data/qm9_raw/`, and the geometry file
`all_fixed_gdb9.sdf` comes from the same rQM9_v0 correction release (see the
upstream README for download instructions). Every split is normalized with the
train-split statistics, so `normalization_path` defaults to
`train_data_property_normalization.npz` for all splits.

## 3. Training

```bash
python structure_generation/train.py \
    -c structure_generation/configs/propmolflow/propmolflow_qm9_alpha_sum.yaml
```

Minimal smoke run (5 steps):

```bash
python structure_generation/train.py \
    -c structure_generation/configs/propmolflow/propmolflow_qm9_alpha_sum.yaml \
    Trainer.max_iter=5 Global.do_eval=False Trainer.output_dir=./output/propmolflow_smoke
```

Key configuration entries (`propmolflow_qm9_alpha_sum.yaml`):

| Section | Description |
|---|---|
| `Global.property_name` | Conditioned property name (`alpha`) |
| `Global.num_inference_steps` | Number of integration steps (100) |
| `Model.__init_params__.vector_field_config` | GVP network structure (defaults to the released checkpoint structure) |
| `Model.__init_params__.prior_config` | Prior configuration (x: centered-normal with alignment; a/c/e: ctmc mask) |
| `Model.__init_params__.properties_handle_method` | Property injection strategy (`sum`) |
| `Model.__init_params__.gaussian_expansion` | Gaussian expansion of the property (`false`) |
| `Optimizer` | Adam(lr=2.5e-4, weight_decay=1e-12) with clip_value=1.0 |
| `Dataset.train/val` | Training/validation splits (`data/propmolflow_qm9/*.npz`) |
| `Sampler.data` | Sampling inputs (molecule sampling dataset) |

Sampling and evaluation run on the framework-level molecular sampling
interface; see section 4.

## 4. Sampling and evaluation

Sampling runs on the framework-level molecular sampling interface
(``MolecularSampler``); the molecules, the SDF export and the validity metrics
are three one-line API calls. The following snippet covers all scenarios:

```python
from ppmat.metrics import PropMolFlowMoleculeMetric
from ppmat.sampler import MolecularSampler

sampler = MolecularSampler(
    # Registered package (downloaded automatically):
    model_name="propmolflow_qm9_alpha_sum",
    # or a local config plus checkpoint (the training output dir carries a
    # timestamp suffix, e.g. ./output/propmolflow_qm9_alpha_sum_t_<time>_s_42):
    # config_path="structure_generation/configs/propmolflow/propmolflow_qm9_alpha_sum.yaml",
    # checkpoint_path="./output/propmolflow_qm9_alpha_sum_t_<time>_s_42/checkpoints/best.pdparams",
    config_overrides=[
        # Molecule models have no registered dataset infos class.
        "Dataset.train=null",
        "Sampler.data.dataset.__init_params__.total_num=64",
        "Sampler.model_sample_params.num_inference_steps=100",
        "Sampler.output_dir=./output/propmolflow_samples",
    ],
)
results = sampler.sample_by_dataloader()
molecules = results["pred"]
metrics = PropMolFlowMoleculeMetric()(molecules)
print(metrics)
sampler.model.save_samples(molecules, sampler.output_dir)
```

Out-of-distribution sampling with a single target value adds two overrides:

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

``n_atoms`` and ``property_value`` are orthogonal pins: an explicit value
fixes that side for every sample, while the sampler mode selects the training
distribution of the unpinned side. Setting both pins fixes the atom count and
the property value at the same time.

The molecules are written to ``propmolflow_samples.sdf`` under
``Sampler.output_dir`` and the following RDKit metrics are returned by
``PropMolFlowMoleculeMetric``:

| Metric | Description |
|---|---|
| `rdkit_valid` | Fraction of generated molecules that pass RDKit sanitization |
| `valid_and_unique` | Fraction of unique valid SMILES |
| `close_shell_ratio` | Fraction of molecules with an even number of valence electrons |
| `atom_stable` | Fraction of atoms whose valence matches the expected valence |
| `mol_stable` | Fraction of molecules whose atoms are all valence-consistent and whose total charge is zero |

## 5. Registered checkpoints

All released checkpoints use `gaussian_expansion: false`. The property handling
strategy and the training distribution follow the upstream best-combination
tables (`data/property_mae_results/{in,out}_distribution_best_combo.csv`).
Replace `alpha` with any property in the table to sample another target.

| Registered name | Property | Property handling | Distribution |
|---|---|---|---|
| `propmolflow_qm9_alpha_sum` | alpha | sum | in-distribution |
| `propmolflow_qm9_Cv_sum` | Cv | sum | in-distribution |
| `propmolflow_qm9_gap_concatenate_sum` | gap | concatenate_sum | in-distribution |
| `propmolflow_qm9_homo_concatenate` | homo | concatenate | in-distribution |
| `propmolflow_qm9_lumo_concatenate_sum` | lumo | concatenate_sum | in-distribution |
| `propmolflow_qm9_mu_multiply` | mu | multiply | in-distribution |
| `propmolflow_qm9_gap_concatenate_ood` | gap | concatenate | out-of-distribution |
| `propmolflow_qm9_homo_concatenate_sum_ood` | homo | concatenate_sum | out-of-distribution |
| `propmolflow_qm9_lumo_concatenate_multiply_ood` | lumo | concatenate_multiply | out-of-distribution |
| `propmolflow_qm9_mu_sum_ood` | mu | sum | out-of-distribution |

## 6. Reference results

Released alpha in-distribution checkpoint, 64 molecules with 100 integration
steps:

| Metric | Value |
|---|---|
| rdkit_valid | 0.9844 |
| valid_and_unique | 0.9844 |
| close_shell_ratio | 0.9531 |
| atom_stable | 0.9964 |
| mol_stable | 0.9375 |

Training for 3 epochs on `train_a` (50000 molecules): total loss decreases from
1.099 to 0.783 with a validation loss of 0.765.

## 7. CINN compiled execution

The model supports the optional CINN compiled backend. The runtime boundary is
`denoise_step` on `CTMCVectorField` (SOT capture); the top-level model only
forwards the backend to it. Compilation requires a CUDA Paddle build with CINN
support, a single GPU and `use_amp=False`.

```python
sampler = MolecularSampler(
    model_name="propmolflow_qm9_alpha_sum",
    config_overrides=[
        "Dataset.train=null",
        "Execution.backend=cinn",
    ],
)
results = sampler.sample_by_dataloader()
```

## 8. Unit tests

The model tests are organized along the training / sampling / evaluation
chains and the CINN runtime orchestration:

```bash
python -m pytest test/propmolflow/test_propmolflow.py -v
```

## 9. Differences from the upstream implementation

- Only the `ctmc` parameterization is supported (all 61 upstream configs use `ctmc`).
- Priors are sampled inside the model instead of the dataset. The semantics are
  unchanged and deterministic prior injection is supported for alignment runs.
- Property normalization is handled by the dataset instead of the model reading
  the normalization file.
- Random number-of-atoms sampling is provided by `PropMolFlowSamplingDataset`.
- The dataset is built from the corrected rQM9_v0 release because the model
  supervises covalent bond orders and formal charges, which the framework
  `QM9Dataset` (original QM9) does not provide; the shared download and caching
  infrastructure is reused unchanged.
- The total-charge check of `mol_stable` is fixed (the upstream check was a dead
  condition that was never evaluated).
- The dead `scalar_embedding_cond` call of the upstream `concatenate_multiply`
  conditioning is removed (its output was discarded); the remaining semantics
  are numerically identical.
- The unused `InterpolantScheduler.loss_weights` schedule and the unused
  prior `align` fields of the categorical features are dropped.
- Sampling runs on the framework-level `MolecularSampler` API untouched by
  this PR; the entry-side overrides skip the training-split statistics pass
  (`Dataset.train=null`, unused by molecular models without a registered
  infos class). The model returns `{"pred": ...}` and the one-line
  `save_samples` call serializes the molecules to SDF.
- No top-level `Metric` section is shipped in the configs: the
  `structure_generation/train.py` per-key metric interface does not apply to
  a generation model (evaluation reports losses only, like the other models
  of this task), so `PropMolFlowMoleculeMetric` is consumed at the sampling
  API level.

## 10. Citation

```bibtex
@article{zeng_propmolflow_2026,
  title = {{PropMolFlow}: property-guided molecule generation with geometry-complete flow matching},
  journal = {Nature Computational Science},
  author = {Zeng, Cheng and Jin, Jirui and Ambrose, Connor and Karypis, George and Transtrum, Mark and Tadmor, Ellad B. and Hennig, Richard G. and Roitberg, Adrian and Martiniani, Stefano and Liu, Mingjie},
  year = {2026},
  doi = {10.1038/s43588-025-00946-y},
}
```
