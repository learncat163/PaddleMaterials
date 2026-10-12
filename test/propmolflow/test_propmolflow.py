# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PropMolFlow integration tests over the training/sampling/evaluation chains."""

from __future__ import annotations

import os
import os.path as osp

import numpy as np
import paddle
import pytest
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets.collate_fn import PropMolFlowCollator
from ppmat.datasets.propmolflow_dataset import PROPERTY_MAP
from ppmat.datasets.propmolflow_dataset import PropMolFlowSamplingDataset
from ppmat.metrics import PropMolFlowMoleculeMetric
from ppmat.models import CTMCVectorField
from ppmat.models import build_model
from ppmat.optimizer import build_optimizer
from ppmat.sampler import MolecularSampler
from ppmat.trainer.base_trainer import BaseTrainer

CONFIG_PATH = osp.join(
    osp.dirname(__file__),
    "..",
    "..",
    "structure_generation",
    "configs",
    "propmolflow",
    "propmolflow_qm9_alpha_sum.yaml",
)

PRIOR_CONFIG = {
    "positions": {"type": "centered-normal", "align": True, "kwargs": {"std": 1.0}},
    "atom_types": {"type": "ctmc"},
    "atom_charges": {"type": "ctmc"},
    "bond_types": {"type": "ctmc"},
}

SMALL_VECTOR_FIELD = {
    "n_charges": 6,
    "n_bond_types": 5,
    "charge_offset": 2,
    "n_hidden_scalars": 32,
    "n_hidden_edge_feats": 16,
    "n_vec_channels": 8,
    "n_cp_feats": 2,
    "n_message_gvps": 2,
    "n_update_gvps": 2,
    "n_position_update_gvps": 3,
    "n_molecule_updates": 2,
    "convs_per_update": 1,
    "n_recycles": 1,
    "rbf_dim": 8,
    "rbf_dmax": 14,
    "message_norm": 100,
    "separate_mol_updaters": True,
    "update_edge_w_distance": True,
    "stochasticity": 10.0,
    "high_confidence_threshold": 0.0,
    "cat_temperature": 0.05,
}


def _build_model(vector_field_config=None, **overrides):
    init_params = {
        "atom_type_map": ["C", "H", "N", "O", "F"],
        "total_loss_weights": {
            "positions": 3.0,
            "atom_types": 0.4,
            "atom_charges": 1.0,
            "bond_types": 2.0,
        },
        "interpolant_scheduler_config": {
            "__class_name__": "InterpolantScheduler",
            "__init_params__": {
                "schedule_type": {
                    "positions": "linear",
                    "atom_types": "linear",
                    "atom_charges": "linear",
                    "bond_types": "linear",
                },
                "cosine_params": {
                    "positions": 1,
                    "atom_types": 2,
                    "atom_charges": 2,
                    "bond_types": 1.5,
                },
            },
        },
        "vector_field_config": (
            SMALL_VECTOR_FIELD if vector_field_config is None else vector_field_config
        ),
        "prior_config": PRIOR_CONFIG,
        "property_embedding_dim": 32,
        "properties_handle_method": "sum",
        "gaussian_expansion": False,
        "n_gaussians": 5,
        "gaussian_start": None,
        "gaussian_stop": None,
    }
    init_params.update(overrides)
    model = build_model(
        {"__class_name__": "PropMolFlow", "__init_params__": init_params}
    )
    model.eval()
    return model


def _write_synthetic_split(tmp_path, num_molecules=6, seed=0):
    rng = np.random.RandomState(seed)
    positions, atom_types, atom_charges = [], [], []
    bond_types, bond_idxs = [], []
    node_idx_array = []
    edge_idx_array = []
    node_offset = 0
    edge_offset = 0
    for _ in range(num_molecules):
        n_atoms = int(rng.randint(3, 6))
        positions.append(rng.standard_normal((n_atoms, 3)).astype("float32"))
        one_hot = np.zeros((n_atoms, 5), dtype=bool)
        one_hot[np.arange(n_atoms), rng.randint(0, 5, n_atoms)] = True
        atom_types.append(one_hot)
        atom_charges.append(rng.randint(-2, 4, n_atoms).astype("int32"))

        upper_i, upper_j = np.triu_indices(n_atoms, k=1)
        keep = rng.rand(upper_i.shape[0]) < 0.4
        mol_bond_idxs = np.stack((upper_i[keep], upper_j[keep]), axis=1).astype("int32")
        bond_idxs.append(mol_bond_idxs)
        bond_types.append(rng.randint(0, 5, mol_bond_idxs.shape[0]).astype("int32"))

        node_idx_array.append([node_offset, node_offset + n_atoms])
        edge_idx_array.append([edge_offset, edge_offset + mol_bond_idxs.shape[0]])
        node_offset += n_atoms
        edge_offset += mol_bond_idxs.shape[0]

    properties = rng.standard_normal((num_molecules, 19)).astype("float32")
    path = tmp_path / "train_a_data_processed.npz"
    np.savez(
        path,
        positions=np.concatenate(positions, axis=0),
        atom_types=np.concatenate(atom_types, axis=0),
        atom_charges=np.concatenate(atom_charges, axis=0),
        bond_types=np.concatenate(bond_types, axis=0),
        bond_idxs=np.concatenate(bond_idxs, axis=0),
        node_idx_array=np.array(node_idx_array, dtype="int32"),
        edge_idx_array=np.array(edge_idx_array, dtype="int32"),
        properties=properties,
    )
    np.savez(
        tmp_path / "train_data_property_normalization.npz",
        mean=properties.mean(axis=0),
        std=properties.std(axis=0, ddof=1) + 1e-6,
    )
    return str(path), properties


def _build_batch(num_atoms_list):
    rng = np.random.RandomState(0)
    samples = []
    for n_atoms in num_atoms_list:
        one_hot_a = np.zeros((n_atoms, 5), dtype="float32")
        one_hot_a[np.arange(n_atoms), rng.randint(0, 5, n_atoms)] = 1.0
        one_hot_c = np.zeros((n_atoms, 6), dtype="float32")
        one_hot_c[np.arange(n_atoms), rng.randint(0, 6, n_atoms)] = 1.0
        upper_i, upper_j = np.triu_indices(n_atoms, k=1)
        bond_types = rng.randint(0, 5, upper_i.shape[0])
        one_hot_e = np.zeros((upper_i.shape[0], 5), dtype="float32")
        one_hot_e[np.arange(upper_i.shape[0]), bond_types] = 1.0
        samples.append(
            {
                "structure_array": {
                    "num_atoms": n_atoms,
                    "positions": rng.standard_normal((n_atoms, 3)).astype("float32"),
                    "atom_types": one_hot_a,
                    "atom_charges": one_hot_c,
                    "bond_types": one_hot_e,
                },
                "prop": float(rng.standard_normal()),
            }
        )
    return PropMolFlowCollator()(samples)


def _sampling_dataset(tmp_path, **kwargs):
    path, _ = _write_synthetic_split(tmp_path)
    dataset_kwargs = {
        "path": path,
        "property_name": "alpha",
        "mode": "in_distribution",
        "total_num": 2,
        "seed": 0,
    }
    dataset_kwargs.update(kwargs)
    return PropMolFlowSamplingDataset(**dataset_kwargs)


def _sampling_inputs(dataset):
    num_atoms = paddle.to_tensor(
        np.concatenate(
            [
                dataset[i]["structure_array"]["num_atoms"].data
                for i in range(len(dataset))
            ]
        ).astype("int64")
    )
    properties = paddle.to_tensor(
        np.concatenate(
            [dataset[i]["properties"].data for i in range(len(dataset))]
        ).astype("float32")
    )
    return {
        "structure_array": {"num_atoms": num_atoms},
        "properties": properties,
    }


def _assert_molecule_dicts(result, num_atoms):
    assert len(result) == num_atoms.shape[0]
    for molecule, expected_atoms in zip(result, num_atoms.numpy().tolist()):
        assert molecule["num_atoms"] == expected_atoms
        assert len(molecule["atom_types"]) == expected_atoms
        assert np.asarray(molecule["positions"]).shape == (expected_atoms, 3)
        assert all(
            symbol in ["C", "H", "N", "O", "F", "Se"]
            for symbol in molecule["atom_types"]
        )
        for src, dst, bond_type in molecule["bonds"]:
            assert 0 <= src < expected_atoms
            assert 0 <= dst < expected_atoms
            assert 1 <= bond_type <= 4


# ----------------------------- training chain -----------------------------


def test_train_forward_from_config():
    config = OmegaConf.load(CONFIG_PATH)
    config = OmegaConf.to_container(config, resolve=True)
    model = build_model(config["Model"])
    model.eval()

    batch = _build_batch([4, 5])
    output = model(batch)
    assert set(output["loss_dict"]) == {
        "loss",
        "loss_positions",
        "loss_atom_types",
        "loss_atom_charges",
        "loss_bond_types",
    }
    for value in output["loss_dict"].values():
        assert np.isfinite(float(value))
    output["loss_dict"]["loss"].backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if not parameter.stop_gradient
    )

    # Training-forward variants over the released configuration space.
    for overrides in (
        {"gaussian_expansion": True, "gaussian_start": 6.31, "gaussian_stop": 196.62},
        {
            "vector_field_config": dict(
                SMALL_VECTOR_FIELD, n_molecule_updates=1, convs_per_update=2
            )
        },
        {"vector_field_config": dict(SMALL_VECTOR_FIELD, separate_mol_updaters=False)},
        *(
            {"properties_handle_method": method}
            for method in [
                "concatenate",
                "sum",
                "multiply",
                "concatenate_sum",
                "concatenate_multiply",
            ]
        ),
    ):
        variant = _build_model(**overrides)
        loss = variant(batch)["loss_dict"]["loss"]
        assert np.isfinite(float(loss))
        loss.backward()
        assert all(
            parameter.grad is not None
            for parameter in variant.parameters()
            if not parameter.stop_gradient
        )

    with pytest.raises(ValueError):
        _build_model(gaussian_expansion=True, gaussian_start=None, gaussian_stop=None)
    with pytest.raises(ValueError):
        _build_model(properties_handle_method="unsupported")
    with pytest.raises(ValueError, match="at least two convolutions"):
        _build_model(vector_field_config=dict(SMALL_VECTOR_FIELD, n_molecule_updates=1))
    for name in ("n_recycles", "n_molecule_updates", "convs_per_update"):
        with pytest.raises(ValueError, match="must be positive"):
            _build_model(vector_field_config=dict(SMALL_VECTOR_FIELD, **{name: 0}))


def test_train_data_pipeline_feeds_forward(tmp_path):
    path, properties = _write_synthetic_split(tmp_path)

    config = OmegaConf.to_container(OmegaConf.load(CONFIG_PATH), resolve=True)
    config["Dataset"]["train"]["dataset"]["__init_params__"]["path"] = path
    config["Dataset"]["train"]["sampler"]["__init_params__"]["shuffle"] = False
    data_loader = build_dataloader(config["Dataset"]["train"])
    batch = next(iter(data_loader))

    assert batch["batch_size"] == 6
    assert batch["upper_edge_mask"].sum() == batch["upper_edge_mask"].shape[0] // 2

    normalization = np.load(osp.join(tmp_path, "train_data_property_normalization.npz"))
    property_idx = PROPERTY_MAP["alpha"]
    restored = float(batch["prop"][0]) * float(
        normalization["std"][property_idx]
    ) + float(normalization["mean"][property_idx])
    assert np.isclose(restored, float(properties[0, property_idx]), atol=1e-5)

    model = _build_model()
    assert np.isfinite(float(model(batch)["loss_dict"]["loss"]))


# ----------------------------- sampling chain -----------------------------


def test_sample_produces_valid_molecules(tmp_path):
    model = _build_model()
    sampling_datasets = (
        _sampling_dataset(tmp_path),
        _sampling_dataset(
            tmp_path,
            mode="out_of_distribution",
            property_value=3.0,
        ),
        _sampling_dataset(tmp_path, n_atoms=4, property_value=2.0),
    )

    for dataset in sampling_datasets:
        batch_data = _sampling_inputs(dataset)
        for sample_kwargs in (
            {},
            {"stochasticity": 8.0, "high_confidence_threshold": 0.9},
        ):
            result = model.sample(batch_data, num_inference_steps=2, **sample_kwargs)[
                "pred"
            ]
            _assert_molecule_dicts(result, batch_data["structure_array"]["num_atoms"])


def test_sampling_framework_end_to_end(tmp_path):
    path, _ = _write_synthetic_split(tmp_path)
    checkpoint_path = str(tmp_path / "best.pdparams")
    paddle.save(_build_model().state_dict(), checkpoint_path)

    config = OmegaConf.load(CONFIG_PATH)
    config.Model.__init_params__.vector_field_config = OmegaConf.create(
        SMALL_VECTOR_FIELD
    )
    config.Model.__init_params__.property_embedding_dim = 32
    sampler_config_path = tmp_path / "sampler_config.yaml"
    OmegaConf.save(config, str(sampler_config_path))

    for condition_overrides in (
        [],
        [
            "Sampler.data.dataset.__init_params__.mode=out_of_distribution",
            "Sampler.data.dataset.__init_params__.property_value=3.0",
        ],
    ):
        sampler = MolecularSampler(
            config_path=str(sampler_config_path),
            checkpoint_path=checkpoint_path,
            config_overrides=[
                "Dataset.train=null",
                f"Sampler.data.dataset.__init_params__.path={path}",
                "Sampler.data.dataset.__init_params__.total_num=4",
                f"Sampler.output_dir={tmp_path / 'out'}",
                "Sampler.model_sample_params.num_inference_steps=2",
            ]
            + condition_overrides,
        )
        results = sampler.sample_by_dataloader(save_path=str(tmp_path / "out"))
        assert results is sampler.last_samples
        assert len(results["pred"]) == 4

        # API-level evaluation and serialization (no top-level Metric section
        # is shipped: the SG task train.py has no generation-model metrics).
        metrics = PropMolFlowMoleculeMetric()(results["pred"])
        assert set(metrics) == {
            "rdkit_valid",
            "valid_and_unique",
            "close_shell_ratio",
            "atom_stable",
            "mol_stable",
        }

    # The one-line model-owned serialization writes the SDF file.
    save_samples = getattr(sampler.model, "save_samples", None)
    assert callable(save_samples)
    save_samples(sampler.last_samples["pred"], sampler.output_dir)
    assert osp.exists(str(tmp_path / "out" / "propmolflow_samples.sdf"))


# ---------------------------- evaluation chain ----------------------------


def test_metric_one_shot_matches_streaming():
    hydrogen = {
        "atom_types": ["H", "H"],
        "charges": [0, 0],
        "positions": [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]],
        "bonds": [[0, 1, 1]],
    }
    invalid_carbon = {
        "atom_types": ["C"] + ["H"] * 6,
        "charges": [0] * 7,
        "positions": [[0.0, 0.0, 0.0]] * 7,
        "bonds": [[0, index, 1] for index in range(1, 7)],
    }
    molecules = [hydrogen, invalid_carbon, hydrogen]
    one_shot = PropMolFlowMoleculeMetric()(molecules)
    assert one_shot == pytest.approx(
        {
            "rdkit_valid": 2 / 3,
            "valid_and_unique": 1 / 3,
            "close_shell_ratio": 1.0,
            "atom_stable": 10 / 11,
            "mol_stable": 2 / 3,
        }
    )

    streaming = PropMolFlowMoleculeMetric()
    for batch in (molecules[:2], molecules[2:]):
        streaming.update_step(
            result={"samples": {"pred": batch}}, batch=None, stage="sample"
        )
    streamed = streaming.compute_epoch(stage="sample")

    assert set(one_shot) == set(streamed)
    for name, value in one_shot.items():
        assert np.isclose(value, streamed[name])

    streaming.reset()
    # Non-sample stages carry no molecules; the metric must stay untouched.
    streaming.update_step(result={"loss_dict": {}}, batch=None, stage="eval")
    with pytest.raises(ValueError):
        streaming.compute_epoch(stage="sample")


# ------------------------ runtime orchestration (CINN) ------------------------


def test_trainer_checkpoint_resume_restores_training_state(tmp_path, monkeypatch):
    monkeypatch.setattr(
        CTMCVectorField, "validate_execution_backend", lambda self, **kwargs: None
    )
    monkeypatch.setattr(
        CTMCVectorField,
        "_run_runtime",
        lambda self, name, function, *args, **kwargs: function(*args, **kwargs),
    )
    path, _ = _write_synthetic_split(tmp_path)
    config = OmegaConf.to_container(OmegaConf.load(CONFIG_PATH), resolve=True)
    data_config = config["Dataset"]["train"]
    data_config["dataset"]["__init_params__"]["path"] = path
    data_config["sampler"]["__init_params__"].update(batch_size=3, shuffle=False)
    loader = build_dataloader(data_config)
    optimizer_config = config["Optimizer"]
    optimizer_config["__init_params__"]["lr"] = {
        "__class_name__": "ExponentialDecay",
        "__init_params__": {
            "learning_rate": 0.00025,
            "gamma": 0.9,
            "decay_steps": 1,
        },
    }

    def build_trainer(name, max_epochs, backend):
        with paddle.utils.unique_name.guard():
            model = _build_model()
            optimizer, scheduler = build_optimizer(
                optimizer_config,
                (model,),
                epochs=max_epochs,
                iters_per_epoch=len(loader),
            )
        trainer_config = dict(
            config["Trainer"],
            output_dir=str(tmp_path / name),
            max_epochs=max_epochs,
            save_freq=1,
        )
        return BaseTrainer(
            trainer_config,
            model,
            train_dataloader=loader,
            val_dataloader=loader,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            execution_config={"backend": backend},
        )

    first = build_trainer("first", 1, "cinn")
    with paddle.utils.unique_name.guard():
        first.train()
    checkpoint_dir = tmp_path / "first" / "checkpoints"
    for prefix in ("epoch_1", "latest", "best"):
        for suffix in ("pdparams", "pdopt", "pdstates"):
            assert (checkpoint_dir / f"{prefix}.{suffix}").is_file()
    weights = paddle.load(str(checkpoint_dir / "latest.pdparams"))
    optimizer_state = paddle.load(str(checkpoint_dir / "latest.pdopt"))
    assert weights.keys() == first.model.state_dict().keys()

    resumed = build_trainer("resumed", 2, "cinn")
    restored_steps = []

    def check_restored_state(model, inputs):
        if restored_steps:
            return
        assert resumed.state.global_step == first.state.global_step == len(loader)
        assert resumed.state.epoch == first.state.epoch + 1
        assert resumed.optimizer.get_lr() == first.optimizer.get_lr()
        actual = resumed.optimizer.state_dict()
        assert actual.keys() == optimizer_state.keys()
        for name, value in optimizer_state.items():
            if isinstance(value, paddle.Tensor):
                np.testing.assert_array_equal(actual[name].numpy(), value.numpy())
            else:
                assert actual[name] == value
        for name, value in weights.items():
            np.testing.assert_array_equal(
                model.state_dict()[name].numpy(), value.numpy()
            )
        restored_steps.append(resumed.state.global_step)

    hook = resumed.model.register_forward_pre_hook(check_restored_state)
    with paddle.utils.unique_name.guard():
        resumed.train(resume_from_checkpoint=str(checkpoint_dir / "latest"))
    hook.remove()
    assert restored_steps == [len(loader)]

    straight = build_trainer("straight", 2, "eager")
    with paddle.utils.unique_name.guard():
        straight.train()
    assert resumed.model.execution_backend == "cinn"
    assert resumed.state.global_step == straight.state.global_step == 2 * len(loader)
    assert resumed.state.epoch == straight.state.epoch == 2
    assert resumed.optimizer.get_lr() == straight.optimizer.get_lr()
    assert resumed.lr_scheduler.state_dict() == straight.lr_scheduler.state_dict()
    assert (
        resumed.optimizer.state_dict().keys() == straight.optimizer.state_dict().keys()
    )
    assert resumed.model.state_dict().keys() == straight.model.state_dict().keys()


def test_cinn_dispatch_boundary_and_eager_equivalence(monkeypatch):
    """CINN orchestration: boundary dispatch and eager equivalence of the
    full training forward (fixed seed keeps the stochastic path deterministic)."""
    monkeypatch.setattr(
        CTMCVectorField, "validate_execution_backend", lambda self, **kwargs: None
    )
    boundary_names = []

    def spy(self, name, function, *args, **kwargs):
        boundary_names.append(name)
        return function(*args, **kwargs)

    monkeypatch.setattr(CTMCVectorField, "_run_runtime", spy)

    batch = _build_batch([4, 5])

    eager_model = _build_model()
    paddle.seed(42)
    eager_losses = {
        name: float(value) for name, value in eager_model(batch)["loss_dict"].items()
    }
    assert boundary_names == ["denoise_step"]

    cinn_model = _build_model(execution_backend="cinn")
    cinn_model.set_state_dict(eager_model.state_dict())
    assert sorted(eager_model.state_dict()) == sorted(cinn_model.state_dict())

    boundary_names.clear()
    paddle.seed(42)
    cinn_losses = {
        name: float(value) for name, value in cinn_model(batch)["loss_dict"].items()
    }
    assert boundary_names == ["denoise_step"]

    for name, value in eager_losses.items():
        np.testing.assert_allclose(cinn_losses[name], value, rtol=0.0, atol=1e-5)

    # The sampling path dispatches the same boundary once per integration step.
    boundary_names.clear()
    cinn_model.sample(
        {
            "structure_array": {"num_atoms": paddle.to_tensor([3, 4], dtype="int64")},
            "properties": paddle.to_tensor([[0.1], [0.2]], dtype="float32"),
        },
        num_inference_steps=3,
    )
    assert boundary_names == ["denoise_step", "denoise_step"]


def _build_deterministic_state(model, num_atoms_list):
    batch = _build_batch(num_atoms_list)
    rng = np.random.RandomState(0)
    num_nodes = batch["node_batch_idx"].shape[0]
    num_edges = batch["src"].shape[0]

    state = {
        key: batch[key]
        for key in (
            "src",
            "dst",
            "upper_edge_mask",
            "node_batch_idx",
            "edge_batch_idx",
        )
    }
    state["batch_size"] = batch["batch_size"]
    state["positions"] = paddle.to_tensor(
        rng.standard_normal((num_nodes, 3)).astype("float32")
    )
    for key, num_classes in (("atom_types", 6), ("atom_charges", 7)):
        one_hot = np.zeros((num_nodes, num_classes), dtype="float32")
        one_hot[np.arange(num_nodes), rng.randint(0, num_classes, num_nodes)] = 1.0
        state[key] = paddle.to_tensor(one_hot)
    edge_one_hot = np.zeros((num_edges, 6), dtype="float32")
    edge_one_hot[np.arange(num_edges), rng.randint(0, 6, num_edges)] = 1.0
    state["bond_types"] = paddle.to_tensor(edge_one_hot)

    properties = paddle.to_tensor([[0.5], [0.5]], dtype="float32")
    t = paddle.to_tensor([0.3, 0.7])
    prop_emb = model.property_embedder(properties)
    state["t_combined"] = paddle.concat([t.unsqueeze(-1), prop_emb], axis=-1)
    return state


def _run_vector_field(model, state):
    t_combined = state["t_combined"]
    with paddle.no_grad():
        return model.vector_field(
            state,
            t_combined,
            node_batch_idx=state["node_batch_idx"],
            upper_edge_mask=state["upper_edge_mask"],
        )


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") != "1",
    reason="Set PPMAT_RUN_CINN_WORKFLOW_TESTS=1 to run the GPU CINN comparison.",
)
def test_gpu_cinn_forward_matches_eager():
    if not paddle.is_compiled_with_cuda():
        pytest.skip("Paddle was not compiled with CUDA.")
    if not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle was not compiled with CINN.")
    paddle.set_device("gpu:0")

    # The vector field is called directly because the GPU comparison needs a
    # fully deterministic input; the top-level forward/sample paths are
    # stochastic (time sampling, CTMC transitions).
    eager_model = _build_model()
    cinn_model = _build_model(execution_backend="cinn")
    cinn_model.set_state_dict(eager_model.state_dict())

    state = _build_deterministic_state(eager_model, [4, 5])
    eager_output = _run_vector_field(eager_model, state)
    cinn_output = _run_vector_field(cinn_model, state)
    # The eight GVP convolutions accumulate a per-layer compilation error of
    # about 1e-6, which the logits heads scale to a few 1e-4.
    for key in ("positions", "atom_types", "atom_charges", "bond_types"):
        np.testing.assert_allclose(
            eager_output[key].numpy(), cinn_output[key].numpy(), atol=5e-4
        )
