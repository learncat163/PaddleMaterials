# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DiffCSP model suite for reinforcement learning.

This code is adapted from:
convert-matinvent/models/suite/diffcsp.py
"""

import os
from typing import List

import numpy as np
import paddle
from pymatgen.core.structure import Structure

from ppmat.rl.datasets import create_rl_dataloader
from ppmat.rl.models.base import ModelSuite, get_device
from ppmat.rl.samplers import DiffCSPSampler as Sampler


class DiffCSPSuite(ModelSuite):
    """DiffCSP model suite for RL training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None

    def load_model(self):
        """Load DiffCSP model from checkpoint."""
        # #ISPL-TODO: Load model from MODEL_REGISTRY or checkpoint
        # from ppmat.models import MODEL_REGISTRY
        # self.model = MODEL_REGISTRY[self.model_name].load(
        #     path=self.model_path,
        #     config_overrides=self.config_overrides
        # )
        # self.model.to(self.device)
        # return self.model

        # Placeholder for now
        raise NotImplementedError("#ISPL-TODO: Implement model loading")

    def get_sampler(self):
        """Get sampler for DiffCSP generation."""
        return Sampler(
            batch_size=self.sample_cfg.get("batch_size", 16),
            num_batches=self.sample_cfg.get("num_batches", 4),
            num_inference_steps=self.sample_cfg.get("num_inference_steps", 1000),
        )

    def get_dataloader(
        self,
        samples: List[Structure],
        rewards: np.ndarray,
        batch_size: int = 8,
    ):
        """Create dataloader from samples and rewards."""
        return create_rl_dataloader(
            structures=samples,
            rewards=rewards,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

    def save_model(self, model, ckpt_dir: str):
        """Save model checkpoint."""
        os.makedirs(ckpt_dir, exist_ok=True)
        paddle.save(model.state_dict(), os.path.join(ckpt_dir, "model.pdparams"))

