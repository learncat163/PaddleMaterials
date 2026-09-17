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

"""Integration tests for MatInvent, run from the repository root."""

import os
import shutil
import tempfile

import numpy as np
import paddle
import pytest
from omegaconf import OmegaConf
from pymatgen.core.structure import Lattice
from pymatgen.core.structure import Structure

from ppmat.models import MODEL_REGISTRY
from ppmat.models import build_model
from ppmat.models.matinvent import rl_finetune
from ppmat.models.matinvent.dataset import create_rl_dataloader
from ppmat.models.matinvent.rewards.reward import CALCULATOR_REGISTRY
from ppmat.models.matinvent.rewards.reward import Reward
from ppmat.models.matinvent.rl_finetune import MatinventRL
from ppmat.models.matinvent.rl_finetune import run_rl_training
from ppmat.utils import download

_CONFIG_PATH = {
    "mattergen": "structure_generation/configs/matinvent/matinvent_mattergen.yaml",
    "diffcsp": "structure_generation/configs/matinvent/matinvent_diffcsp.yaml",
}
_BACKBONE_CLASS = {"mattergen": "MatterGen", "diffcsp": "DiffCSP"}


def _load_config(model_type):
    config = OmegaConf.load(_CONFIG_PATH[model_type])
    return OmegaConf.to_container(config, resolve=True)


def _make_structure(species, coords, a=5.43):
    return Structure(lattice=Lattice.cubic(a), species=species, coords=coords)


def _tiny_sampling_config(config, output_dir, package_name):
    config["Trainer"]["output_dir"] = output_dir
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
    return config


class TestConfigToModel:
    @pytest.mark.parametrize("model_type", ["mattergen", "diffcsp"])
    def test_config_builds_matinvent_backbone(self, model_type):
        config = _load_config(model_type)
        model = build_model(config["Model"])

        assert isinstance(model, MatinventRL)
        assert type(model.backbone()).__name__ == _BACKBONE_CLASS[model_type]
        expected_steps = config["Global"]["num_inference_steps"]
        assert model.num_inference_steps == expected_steps
        assert (
            config["Sample"]["model_sample_params"]["num_inference_steps"]
            == expected_steps
        )


class TestSampling:
    def test_sample_delegates_to_backbone(self, monkeypatch):
        calls = {}

        class StubBackbone:
            def sample(self, data, **sample_params):
                calls.update(sample_params)
                return {"result": ["sampled"]}

        monkeypatch.setattr(
            rl_finetune, "build_backbone", lambda backbone_cfg: StubBackbone()
        )
        model = MatinventRL(
            model_type="mattergen", num_inference_steps=7, backbone_cfg={"dummy": True}
        )
        result = model.sample({"structure_array": {}})

        assert result == {"result": ["sampled"]}
        assert calls == {"num_inference_steps": 7}


class TestDataFlow:
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
            structures, np.array([0.8, 0.3], dtype="float32"), batch_size=2, shuffle=False
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
    def test_reward_scoring_marks_failed_structures(self, monkeypatch):
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
                        "root_dir": tmp,
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


class TestUnifiedEntry:
    def test_run_rl_training_writes_checkpoints(self, monkeypatch, tmp_path):
        package_name = "test_matinvent_diffcsp"
        package_dir = tmp_path / package_name
        (package_dir / "checkpoints").mkdir(parents=True)
        shutil.copyfile(_CONFIG_PATH["diffcsp"], package_dir / f"{package_name}.yaml")

        config = _load_config("diffcsp")
        model = build_model(config["Model"])
        paddle.save(
            model.backbone().state_dict(),
            str(package_dir / "checkpoints" / "best.pdparams"),
        )
        monkeypatch.setitem(
            MODEL_REGISTRY, package_name, "http://localhost/package.zip"
        )
        monkeypatch.setattr(
            download, "get_weights_path_from_url", lambda url: str(tmp_path)
        )

        output_dir = str(tmp_path / "output")
        config = _tiny_sampling_config(config, output_dir, package_name)
        run_rl_training(config, model)

        model_dir = os.path.join(output_dir, "models")
        assert os.path.isfile(os.path.join(model_dir, "final", "model.pdparams"))
        assert os.path.isfile(os.path.join(model_dir, "loop_0000", "model.pdparams"))
        assert os.path.isfile(
            os.path.join(output_dir, "samples", "long_term_memory.csv")
        )
