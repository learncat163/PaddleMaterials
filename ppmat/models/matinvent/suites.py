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

import os
from typing import List

import numpy as np
import paddle
from pymatgen.core.structure import Structure

from ppmat.models.matinvent.dataset import create_rl_dataloader
from ppmat.models.matinvent.sampler import DiffCSPSampler
from ppmat.models.matinvent.sampler import MatterGenSampler
from ppmat.utils import save_load

_SAMPLER_MAP = {"diffcsp": DiffCSPSampler, "mattergen": MatterGenSampler}


def build_backbone(backbone_cfg: dict):
    """Build one backbone instance for MatInvent sampling and fine-tuning."""
    from ppmat.models import build_model

    return build_model(backbone_cfg)


class ModelSuite:
    """Backbone factory, weights and RL hyperparameters for MatInvent."""

    def __init__(self, model_name, backbone_cfg, sample_cfg, finetune_cfg,
                 pretrained_model_path=None, device=None, build_structure_cfg=None):
        self.model_name = model_name
        self.backbone_cfg = backbone_cfg
        self.sample_cfg = sample_cfg
        self.finetune_cfg = finetune_cfg
        self.pretrained_model_path = pretrained_model_path
        self.build_structure_cfg = build_structure_cfg or {"format": "array"}
        if device is None:
            device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(device)

    def load_model(self):
        if self.pretrained_model_path is None:
            raise ValueError(
                f"pretrained_model_path must be specified for {self.model_name}")
        model = build_backbone(self.backbone_cfg)
        save_load.load_pretrain(model, self.pretrained_model_path)
        model.eval()
        return model

    def get_sampler(self):
        cls = _SAMPLER_MAP[self.model_name]
        return cls(
            batch_size=self.sample_cfg.get("batch_size", 16),
            num_batches=self.sample_cfg.get("num_batches", 4),
            num_inference_steps=self.sample_cfg.get("num_inference_steps", 1000),
            build_structure_cfg=self.build_structure_cfg)

    @staticmethod
    def get_dataloader(
        samples: List[Structure], rewards: np.ndarray, batch_size: int = 8
    ):
        return create_rl_dataloader(
            structures=samples, rewards=rewards, batch_size=batch_size)

    @staticmethod
    def save_model(model, ckpt_dir: str):
        os.makedirs(ckpt_dir, exist_ok=True)
        paddle.save(model.state_dict(), os.path.join(ckpt_dir, "model.pdparams"))
