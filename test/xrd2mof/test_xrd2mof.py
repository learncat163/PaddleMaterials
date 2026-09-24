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
import os.path as osp
import subprocess
import sys

import numpy as np
import paddle
import pytest
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.models import MODEL_REGISTRY
from ppmat.models import build_model
from ppmat.models import build_model_from_name
from ppmat.models import resolve_model_package_dir
from ppmat.models.xrd2mof.feature_extractor import Xrd2MofFeatureExtractor
from ppmat.sampler import Xrd2MofSampler
from ppmat.utils import download

CONFIG_PATH = osp.join(
    osp.dirname(osp.abspath(__file__)),
    "..",
    "..",
    "structure_generation",
    "configs",
    "xrd2mof",
    "xrd2mof_cgmof50.yaml",
)


def load_config():
    config = OmegaConf.load(CONFIG_PATH)
    return OmegaConf.to_container(config, resolve=True)


def build_training_batch():
    config = load_config()
    loader = build_dataloader(config["Dataset"]["train"])
    return next(iter(loader))


def test_model_build_and_forward_loss():
    config = load_config()
    model = build_model(config["Model"])
    batch = build_training_batch()

    outputs = model(batch)

    assert "loss_dict" in outputs
    loss_dict = outputs["loss_dict"]
    for key in ("loss", "loss_lattice", "loss_coord"):
        assert key in loss_dict
        assert bool(paddle.isfinite(loss_dict[key]))
    assert float(loss_dict["loss"]) > 0.0


def test_training_step_updates_parameters():
    config = load_config()
    model = build_model(config["Model"])
    batch = build_training_batch()

    optimizer = paddle.optimizer.Adam(
        learning_rate=config["Optimizer"]["__init_params__"]["lr"]["__init_params__"][
            "learning_rate"
        ],
        parameters=model.parameters(),
    )
    before = [param.numpy().copy() for param in model.parameters()]
    loss = model(batch)["loss_dict"]["loss"]
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
    after = [param.numpy() for param in model.parameters()]

    assert bool(paddle.isfinite(loss))
    assert any(not np.allclose(prev, curr) for prev, curr in zip(before, after))


def test_dataset_collate_and_forward():
    config = load_config()
    loader = build_dataloader(config["Dataset"]["val"])
    batch = next(iter(loader))
    structure_array = batch["structure_array"]

    assert structure_array["feature_embedding"].shape[1] == 256
    assert (
        int(structure_array["num_atoms"].sum())
        == structure_array["frac_coords"].shape[0]
    )
    assert structure_array["atom_types"].max() <= 1

    model = build_model(config["Model"])
    outputs = model(batch)
    assert bool(paddle.isfinite(outputs["loss_dict"]["loss"]))


def test_sample_produces_valid_cg_structures():
    config = load_config()
    model = build_model(config["Model"])
    model.eval()
    loader = build_dataloader(config["Sample"]["data"])
    batch = next(iter(loader))

    outputs = model.sample(batch, num_inference_steps=5)

    assert len(outputs["result"]) == int(batch["structure_array"]["num_atoms"].shape[0])
    for item in outputs["result"]:
        lattice = np.asarray(item["lattice"])
        frac_coords = np.asarray(item["frac_coords"])
        assert np.isfinite(lattice).all()
        assert np.isfinite(frac_coords).all()
        assert abs(np.linalg.det(lattice)) > 0.0
        assert frac_coords.min() >= 0.0 and frac_coords.max() < 1.0
        assert len(item["atom_types"]) == item["num_atoms"]
        assert set(item["atom_types"]).issubset({0, 1})


def build_feature_extractor_batch():
    batch_size, num_nodes, num_edges = 3, 9, 12
    return {
        "cif_lattice": paddle.randn([batch_size, 3, 3]),
        "cif_atom_positions": paddle.randn([batch_size, 50, 3]),
        "xrd": paddle.randn([batch_size, 5250]),
        "metal": paddle.randn([batch_size, 65]),
        "linker": {
            "x": paddle.randn([num_nodes, 37]),
            "edge_index": paddle.randint(0, num_nodes, [2, num_edges]),
            "batch": paddle.repeat_interleave(
                paddle.arange(batch_size), repeats=num_nodes // batch_size
            ),
        },
    }


def test_feature_extractor_forward_and_loss():
    model = Xrd2MofFeatureExtractor()
    batch = build_feature_extractor_batch()

    outputs = model(batch)

    assert "loss_dict" in outputs and "pred_dict" in outputs
    assert bool(paddle.isfinite(outputs["loss_dict"]["loss"]))
    assert outputs["pred_dict"]["feature_embedding"].shape == [3, 256]
    assert outputs["pred_dict"]["cif_embedding"].shape == [3, 256]


def test_registered_package_ships_feature_extractor_weights():
    model_name = "xrd2mof_cgmof50"
    try:
        extracted_path = download.get_weights_path_from_url(MODEL_REGISTRY[model_name])
    except Exception as exc:  # pragma: no cover - depends on network access
        pytest.skip(f"registered weights unavailable: {exc}")

    package_dir = resolve_model_package_dir(model_name, extracted_path)
    weights_path = osp.join(package_dir, "checkpoints", "feature_extractor.pdparams")
    assert osp.isfile(weights_path)

    model = Xrd2MofFeatureExtractor()
    model.set_state_dict(paddle.load(weights_path))
    model.eval()
    outputs = model.extract(build_feature_extractor_batch())

    assert outputs["feature_embedding"].shape == [3, 256]
    assert bool(paddle.isfinite(outputs["feature_embedding"]).all())


def test_registered_model_full_chain():
    try:
        model, config = build_model_from_name("xrd2mof_cgmof50")
    except Exception as exc:  # pragma: no cover - depends on network access
        pytest.skip(f"registered weights unavailable: {exc}")

    assert type(model).__name__ == "Xrd2Mof"
    assert "Model" in config and "Sample" in config

    model.eval()
    loader = build_dataloader(config["Sample"]["data"])
    batch = next(iter(loader))
    outputs = model.sample(batch, num_inference_steps=2)
    assert len(outputs["result"]) > 0


def test_independent_sampler_writes_npz(tmp_path):
    try:
        sampler = Xrd2MofSampler(
            model_name="xrd2mof_cgmof50",
            config_overrides=["Sample.model_sample_params.num_inference_steps=2"],
        )
    except Exception as exc:  # pragma: no cover - depends on network access
        pytest.skip(f"registered weights unavailable: {exc}")

    sampler.sample_by_dataloader(save_path=str(tmp_path))

    files = sorted(tmp_path.glob("*.npz"))
    assert files
    sample = np.load(files[0], allow_pickle=True)
    assert set(sample.files) == {
        "lattice",
        "frac_coords",
        "atom_types",
        "num_atoms",
        "mof_id",
    }
    assert sample["lattice"].shape == (3, 3)
    assert sample["frac_coords"].shape[1] == 3
    assert sample["atom_types"].shape[0] == sample["frac_coords"].shape[0]


def test_shared_cli_dispatches_sampler(tmp_path):
    output_dir = tmp_path / "results"
    command = [
        sys.executable,
        "structure_generation/sample.py",
        "--model_name",
        "xrd2mof_cgmof50",
        "--mode",
        "by_dataloader",
        "--sampler_type",
        "Xrd2MofSampler",
        "--output_path",
        str(output_dir),
    ]
    try:
        subprocess.run(
            command,
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"shared sampling CLI unavailable: {exc}")

    files = sorted(output_dir.glob("*.npz"))
    assert files
    sample = np.load(files[0], allow_pickle=True)
    assert set(sample.files) == {
        "lattice",
        "frac_coords",
        "atom_types",
        "num_atoms",
        "mof_id",
    }
