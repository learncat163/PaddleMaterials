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

"""Integration tests for the MatInvent RL workflow.

Covers the workflow end to end: config to backbone, registered package
build/load/sample, dataset to diffusion loss, reward scoring, the RL loop, and
the CINN execution backend (nested runtime ownership plus boundary dispatch).
"""

import copy
import os
import pathlib
import tempfile
import unittest.mock as mock

import numpy as np
import paddle
import pytest
from omegaconf import OmegaConf
from pymatgen.core.structure import Lattice
from pymatgen.core.structure import Structure

from ppmat.models import MODEL_REGISTRY
from ppmat.models import build_model
from ppmat.models.diffcsp.diffcsp import DiffCSP
from ppmat.models.matinvent.dataset import create_rl_dataloader
from ppmat.models.matinvent.mattergen import MatinventMatterGen
from ppmat.models.matinvent.rewards.reward import CALCULATOR_REGISTRY
from ppmat.models.matinvent.rewards.reward import Reward
from ppmat.models.matinvent.rl_finetune import MatinventRL
from ppmat.models.matinvent.rl_finetune import run_rl_training
from ppmat.utils import download
from ppmat.utils import save_load
from ppmat.utils.model_package import resolve_model_package_dir

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "structure_generation" / "configs" / "matinvent"
_CONFIG_PATH = {
    "mattergen": str(_CONFIG_DIR / "matinvent_mattergen.yaml"),
    "diffcsp": str(_CONFIG_DIR / "matinvent_diffcsp.yaml"),
}
_BACKBONE_CLASS = {"mattergen": MatinventMatterGen, "diffcsp": DiffCSP}
_BACKBONE_METHOD = {"mattergen": "_denoise", "diffcsp": "_decode"}
_WEIGHTS_NAME = "best.pdparams"
_DISPATCH_SEED = 2026


def _load_config(model_type):
    config = OmegaConf.load(_CONFIG_PATH[model_type])
    return OmegaConf.to_container(config, resolve=True)


def _make_structure(species, coords, a=5.43):
    return Structure(lattice=Lattice.cubic(a), species=species, coords=coords)


def _tiny_backbone_cfg(model_type):
    """Shrunk copy of the released backbone: same code path, fast numerics."""
    config = _load_config(model_type)
    params = config["Model"]["__init_params__"]["backbone_cfg"]["__init_params__"]
    settings = copy.deepcopy(params)
    if model_type == "mattergen":
        # Hidden/latent widths must stay coupled for the denoiser concat.
        settings["decoder_cfg"]["hidden_dim"] = 32
        gemnet = settings["decoder_cfg"]["gemnet_cfg"]
        gemnet["latent_dim"] = 32
        gemnet["atom_embedding_cfg"]["emb_size"] = 32
        gemnet["emb_size_atom"] = 32
        gemnet["emb_size_edge"] = 32
        gemnet["num_blocks"] = 1
        gemnet["max_neighbors"] = 12
        gemnet["cutoff"] = 5.0
    else:
        settings["time_dim"] = 32
        decoder = settings["decoder_cfg"]
        decoder["hidden_dim"] = 32
        decoder["latent_dim"] = 32
        decoder["num_layers"] = 1
        decoder["num_freqs"] = 6
    backbone_cfg = config["Model"]["__init_params__"]["backbone_cfg"]
    backbone_cfg["__init_params__"] = settings
    return backbone_cfg


def _write_model_package(directory, model_type, backbone_cfg=None, package_name=None):
    """Write a MatInvent package laid out like the MODEL_REGISTRY archives."""
    config = _load_config(model_type)
    if backbone_cfg is not None:
        config["Model"]["__init_params__"]["backbone_cfg"] = backbone_cfg
    package_name = package_name or f"test_matinvent_{model_type}"
    package_dir = pathlib.Path(directory) / package_name
    (package_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with paddle.utils.unique_name.guard():
        paddle.seed(_DISPATCH_SEED)
        model = build_model(config["Model"])
    paddle.save(
        model.backbone().state_dict(),
        str(package_dir / "checkpoints" / _WEIGHTS_NAME),
    )
    OmegaConf.save(OmegaConf.create(config), package_dir / f"{package_name}.yaml")
    return package_name, package_dir


def _make_backbone(model_type, execution_backend="eager"):
    with paddle.utils.unique_name.guard():
        paddle.seed(_DISPATCH_SEED)
        backbone = build_model(_tiny_backbone_cfg(model_type))
    backbone.set_execution_backend(execution_backend)
    return backbone


def _clone_state_dict(model):
    return {key: value.clone() for key, value in model.state_dict().items()}


def _make_dispatch_pair(model_type):
    initial_state = _clone_state_dict(_make_backbone(model_type))
    paddle.seed(_DISPATCH_SEED)
    eager_backbone = _make_backbone(model_type)
    paddle.seed(_DISPATCH_SEED)
    cinn_backbone = _make_backbone(model_type, execution_backend="cinn")
    eager_backbone.set_state_dict(initial_state)
    cinn_backbone.set_state_dict(initial_state)
    eager_backbone.eval()
    cinn_backbone.eval()
    return eager_backbone, cinn_backbone


def _record_boundaries(names):
    def run_boundary(self, name, function, *args, **kwargs):
        names.append(name)
        return function(*args, **kwargs)

    return run_boundary


def _make_denoise_inputs(model_type, backbone, batch_size=1):
    """Deterministic boundary inputs, no dependence on sampler randomness."""
    num_atoms = paddle.to_tensor([2] * batch_size, dtype="int64")
    total_atoms = int(num_atoms.sum())
    batch_idx = paddle.repeat_interleave(paddle.arange(batch_size), repeats=num_atoms)
    lattice = paddle.eye(3).unsqueeze(0).tile([batch_size, 1, 1]) * 5.43
    frac_coords = paddle.to_tensor(
        [0.0, 0.0, 0.0, 0.25, 0.25, 0.25] * batch_size,
        dtype="float32",
    ).reshape([total_atoms, 3])
    atom_types = paddle.to_tensor([14] * total_atoms, dtype="int64")
    structure_array = {
        "frac_coords": frac_coords,
        "lattice": lattice,
        "atom_types": atom_types,
        "num_atoms": num_atoms,
        "batch": batch_idx,
    }
    if model_type == "mattergen":
        # Fractional diffusion time in (0, 1).
        return (structure_array, paddle.full([batch_size], 0.5, dtype="float32"))
    # DiffCSP._decode consumes the embedded time, matching DiffCSP.forward.
    time_emb = backbone.time_embedding(paddle.full([batch_size], 500, dtype="int64"))
    return (time_emb, atom_types - 1, frac_coords, lattice, num_atoms, batch_idx)


def _flatten_result(output):
    if isinstance(output, dict):
        for value in output.values():
            yield from _flatten_result(value)
    elif isinstance(output, (list, tuple)):
        for value in output:
            yield from _flatten_result(value)
    else:
        yield output


def _make_adapter(model_type, execution_backend="eager"):
    return MatinventRL(
        model_type=model_type,
        num_inference_steps=3,
        backbone_cfg=_tiny_backbone_cfg(model_type),
        execution_backend=execution_backend,
    )


def _patched_rl_environment(monkeypatch, tmp_path, model_type, execution):
    """Registered package, no download, isolated output, runtime hooks."""
    backbone_cfg = _tiny_backbone_cfg(model_type)
    package_name, _ = _write_model_package(tmp_path, model_type, backbone_cfg)
    config = _load_config(model_type)
    config["RL"]["pretrained_model_name"] = package_name
    config["RL"]["replay_cfg"] = None
    config["Model"]["__init_params__"]["backbone_cfg"] = backbone_cfg
    config["Execution"] = {"backend": execution, "__init_params__": {}}
    config["Trainer"]["output_dir"] = str(tmp_path / "output")

    monkeypatch.setitem(
        MODEL_REGISTRY, package_name, f"http://localhost/{package_name}.zip"
    )
    monkeypatch.setattr(
        download, "get_weights_path_from_url", lambda url: str(tmp_path)
    )
    return config


class TestModelPackage:
    """Cover config -> model, package yaml -> build -> load -> sample."""

    @pytest.mark.parametrize("model_type", ["mattergen", "diffcsp"])
    def test_package_yaml_builds_and_loads(self, model_type, tmp_path):
        package_name, package_dir = _write_model_package(tmp_path, model_type)
        resolved = resolve_model_package_dir(package_name, str(tmp_path))
        assert resolved == str(package_dir)

        package_config = OmegaConf.load(os.path.join(resolved, f"{package_name}.yaml"))
        package_config = OmegaConf.to_container(package_config, resolve=True)
        model = build_model(package_config["Model"])
        assert isinstance(model, MatinventRL)
        assert type(model.backbone()).__name__ == _BACKBONE_CLASS[model_type].__name__
        expected_steps = package_config["Global"]["num_inference_steps"]
        assert model.num_inference_steps == expected_steps

        save_load.load_pretrain(model, resolved, weights_name=_WEIGHTS_NAME)
        saved = paddle.load(str(package_dir / "checkpoints" / _WEIGHTS_NAME))
        loaded = model.backbone().state_dict()
        key = next(iter(saved))
        assert np.allclose(saved[key].numpy(), loaded[key].numpy())

    def test_sampled_structures_from_package_model(self, tmp_path):
        package_name, package_dir = _write_model_package(tmp_path, "diffcsp")
        config = OmegaConf.to_container(
            OmegaConf.load(str(package_dir / f"{package_name}.yaml")), resolve=True
        )
        model = build_model(config["Model"])
        save_load.load_pretrain(model, str(package_dir), weights_name=_WEIGHTS_NAME)
        model.eval()

        num_atoms = paddle.to_tensor([2], dtype="int64")
        atom_types = paddle.to_tensor([14, 14], dtype="int64")
        result = model.sample(
            {"structure_array": {"num_atoms": num_atoms, "atom_types": atom_types}},
            num_inference_steps=2,
        )

        assert len(result["result"]) == 1


class TestMatterGenBackbone:
    """The MatInvent MatterGen variant must keep the mainline graph numerics."""

    def test_paddle_graph_keeps_mainline_numerics(self):
        mainline_cfg = copy.deepcopy(_tiny_backbone_cfg("mattergen"))
        mainline_cfg["__class_name__"] = "MatterGen"
        with paddle.utils.unique_name.guard():
            paddle.seed(_DISPATCH_SEED)
            mainline = build_model(mainline_cfg)
        with paddle.utils.unique_name.guard():
            paddle.seed(_DISPATCH_SEED)
            adapter = build_model(_tiny_backbone_cfg("mattergen"))
        assert type(adapter).__name__ == "MatinventMatterGen"

        adapter.set_state_dict(_clone_state_dict(mainline))
        mainline.eval()
        adapter.eval()

        inputs = _make_denoise_inputs("mattergen", adapter)
        expected = mainline._denoise(*inputs)
        actual = adapter._denoise(*inputs)
        for expected_value, actual_value in zip(
            _flatten_result(expected), _flatten_result(actual), strict=True
        ):
            np.testing.assert_allclose(
                np.asarray(actual_value), np.asarray(expected_value), atol=0, rtol=0
            )


class TestDataFlow:
    """Dataset -> backbone forward/loss, the RL training data path."""

    def test_dataset_to_forward_loss(self):
        backbone = build_model(_load_config("diffcsp")["Model"]).backbone()
        structures = [
            _make_structure(["Si", "Si"], [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
            _make_structure(
                ["C", "O", "O"],
                [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
                a=4.0,
            ),
        ]
        loader = create_rl_dataloader(
            structures,
            np.array([0.8, 0.3], dtype="float32"),
            batch_size=2,
            shuffle=False,
        )
        batch = next(iter(loader))
        structure_array = batch["structure_array"]

        assert structure_array["num_atoms"].numpy().tolist() == [2, 3]
        assert tuple(structure_array["frac_coords"].shape) == (5, 3)
        assert tuple(structure_array["reward"].shape) == (2,)

        result = backbone.forward(batch)
        loss = result["loss_dict"]["loss"]
        assert np.isfinite(float(loss))
        loss.backward()


class TestRewardScoring:
    """Reward scoring over sampled structures, including failed properties."""

    def test_density_reward_and_failed_property(self, monkeypatch):
        structures = [
            _make_structure(["Si", "Si"], [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
        ]

        class FailingCalculator:
            def __init__(self, root_dir, task=None):
                self.root_dir = root_dir

            def calc(self, samples, label="tmp"):
                return np.full(len(samples[0]), np.nan)

        with tempfile.TemporaryDirectory() as tmp:
            prop_cfg = [
                {
                    "name": "density",
                    "target": "ascending",
                    "minv": 0.0,
                    "maxv": 10.0,
                    "calculator": {
                        "__class_name__": "PyMatGen",
                        "task": "density",
                    },
                }
            ]
            reward = Reward(
                root_dir=tmp, prop_cfg=prop_cfg, reward_threshold=0.5, reduce="mean"
            )
            rewards, prop_dict, failed = reward.scoring((structures, "valid"))

            assert not failed.any()
            assert 0.0 <= float(rewards[0]) <= 1.0
            assert np.isfinite(prop_dict["density"][0])

            monkeypatch.setitem(CALCULATOR_REGISTRY, "Failing", FailingCalculator)
            prop_cfg[0]["calculator"]["__class_name__"] = "Failing"
            failing_reward = Reward(
                root_dir=tmp, prop_cfg=prop_cfg, reward_threshold=0.5, reduce="mean"
            )
            rewards, _, failed = failing_reward.scoring((structures, "failed"))

            assert failed.tolist() == [True]
            assert float(rewards[0]) == 0.0


class TestRLTraining:
    """The RL entry point: checkpointing and execution-backend orchestration."""

    def test_run_rl_training_writes_checkpoints(self, monkeypatch, tmp_path):
        package_name, _ = _write_model_package(tmp_path, "diffcsp")

        config = _load_config("diffcsp")
        config["Trainer"]["output_dir"] = str(tmp_path / "output")
        config["RL"]["pretrained_model_name"] = package_name
        config["RL"]["rl_epoch"] = 1
        config["RL"]["save_freq"] = 1
        config["RL"]["replay_cfg"] = None
        config["RL"]["div_filter_cfg"]["enabled"] = False
        config["RL"]["sample_cfg"].update(
            {"batch_size": 1, "num_batches": 1, "num_inference_steps": 2}
        )
        config["RL"]["finetune_cfg"].update(
            {"batch_size": 2, "epochs": 1, "timesteps": 1, "accum_steps": 1}
        )

        monkeypatch.setitem(
            MODEL_REGISTRY, package_name, "http://localhost/package.zip"
        )
        monkeypatch.setattr(
            download, "get_weights_path_from_url", lambda url: str(tmp_path)
        )

        model = build_model(config["Model"])
        run_rl_training(config, model)

        output_dir = config["Trainer"]["output_dir"]
        model_dir = os.path.join(output_dir, "models")
        assert os.path.isfile(os.path.join(model_dir, "final", "model.pdparams"))
        assert os.path.isfile(os.path.join(model_dir, "loop_0000", "model.pdparams"))
        assert os.path.isfile(
            os.path.join(output_dir, "samples", "long_term_memory.csv")
        )

    def test_run_rl_training_applies_execution_backend(self, monkeypatch, tmp_path):
        config = _patched_rl_environment(monkeypatch, tmp_path, "diffcsp", "cinn")

        # Orchestration test only: no GPU compiler is involved.
        monkeypatch.setattr(
            DiffCSP, "validate_execution_backend", lambda self, **kwargs: None
        )
        names = []
        monkeypatch.setattr(DiffCSP, "_run_runtime", _record_boundaries(names))

        model = build_model(config["Model"])
        run_rl_training(config, model)

        model_dir = os.path.join(config["Trainer"]["output_dir"], "models")
        final_state = paddle.load(os.path.join(model_dir, "final", "model.pdparams"))
        # Structural only: the RL loop trains a separate agent and is stochastic.
        assert final_state.keys() == model.state_dict().keys()
        assert names, "sampling must dispatch through the denoise boundary"
        assert set(names) == {"denoise_step"}
        assert os.path.isfile(
            os.path.join(
                config["Trainer"]["output_dir"], "samples", "long_term_memory.csv"
            )
        )


class TestCinnDispatch:
    """CINN boundary dispatch: compiled numbers must match eager exactly."""

    @pytest.mark.parametrize("model_type", ["mattergen", "diffcsp"])
    def test_boundary_dispatch_matches_eager(self, model_type):
        backbone_cls = _BACKBONE_CLASS[model_type]
        method_name = _BACKBONE_METHOD[model_type]
        eager_backbone, cinn_backbone = _make_dispatch_pair(model_type)
        assert cinn_backbone.execution_backend == "cinn"

        inputs = _make_denoise_inputs(model_type, eager_backbone)
        expected = getattr(eager_backbone, method_name)(*inputs)

        names = []
        with mock.patch.object(backbone_cls, "_run_runtime", _record_boundaries(names)):
            actual = getattr(cinn_backbone, method_name)(*inputs)

        assert set(names) == {"denoise_step"}
        for expected_value, actual_value in zip(
            _flatten_result(expected), _flatten_result(actual), strict=True
        ):
            np.testing.assert_allclose(
                np.asarray(actual_value), np.asarray(expected_value), atol=0, rtol=0
            )
        assert cinn_backbone.state_dict().keys() == eager_backbone.state_dict().keys()

    def test_nested_adapter_samples_with_cinn_backend(self, monkeypatch):
        """The adapter forwards the backend; sampling hits the denoise boundary."""
        with paddle.utils.unique_name.guard():
            paddle.seed(_DISPATCH_SEED)
            adapter = _make_adapter("diffcsp", execution_backend="cinn")
        assert adapter.execution_backend == "cinn"
        keys_before = list(adapter.state_dict().keys())

        names = []
        monkeypatch.setattr(DiffCSP, "_run_runtime", _record_boundaries(names))
        result = adapter.sample(
            {
                "structure_array": {
                    "num_atoms": paddle.to_tensor([2], dtype="int64"),
                    "atom_types": paddle.to_tensor([14, 14], dtype="int64"),
                }
            },
            num_inference_steps=2,
        )

        assert len(result["result"]) == 1
        assert names and set(names) == {"denoise_step"}

        adapter.set_execution_backend("eager")
        assert list(adapter.state_dict().keys()) == keys_before


def _gate_gpu_cinn_rl_loop():
    if os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") != "1":
        pytest.skip("Set PPMAT_RUN_CINN_WORKFLOW_TESTS=1 to run the GPU CINN test")
    if not paddle.is_compiled_with_cuda() or not paddle.base.is_compiled_with_cinn():
        pytest.skip("Paddle must be compiled with CUDA and CINN")
    paddle.set_device("gpu:0")


def test_gpu_cinn_rl_loop_matches_eager(monkeypatch):
    """Gated real-compile check: eager and cinn loops stay numerically close."""
    _gate_gpu_cinn_rl_loop()

    with tempfile.TemporaryDirectory() as tmp_dir:
        final_states = {}
        for execution in ("eager", "cinn"):
            config = _patched_rl_environment(monkeypatch, tmp_dir, "diffcsp", execution)
            config["Trainer"]["output_dir"] = os.path.join(tmp_dir, "output", execution)
            paddle.seed(_DISPATCH_SEED)
            np.random.seed(_DISPATCH_SEED)
            model = build_model(config["Model"])
            run_rl_training(config, model)
            checkpoint = os.path.join(
                config["Trainer"]["output_dir"], "models", "final", "model.pdparams"
            )
            final_states[execution] = paddle.load(checkpoint)

        for key, value in final_states["cinn"].items():
            np.testing.assert_allclose(
                value.numpy(),
                final_states["eager"][key].numpy(),
                atol=2e-5,
                rtol=2e-5,
            )
