# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Base reinforcement learning pipeline for crystal generation.

This module is migrated from OMG (Open Materials Generation).
Original code: pipeline.base
"""

import os
import logging
from typing import List, Dict

import numpy as np
import paddle

from pymatgen.core.structure import Structure


def get_device(device: str | None = None):
    """Get the device to run computations on."""
    if device is None:
        if paddle.device.is_compulsory_for_backend("gpu"):
            device = 'gpu:0'
        else:
            device = 'cpu'
    return paddle.device.set_device(device)


class ReinL:
    """
    Base reinforcement learning pipeline for crystal generation.

    This class provides the base structure for RL-based crystal generation pipelines.
    """

    def __init__(
        self,
        rl_epoch: int,
        model_suite,  # ModelSuite type
        reward,  # Reward type
        sample_cfg,  # DictConfig
        finetune_cfg,  # DictConfig
        save_dir: str,
        save_freq: int,
        device: str = None,
        logger=None,
        replay: bool = False,
        replay_args: Dict = None,
        **kwargs,
    ) -> None:
        """Initialize the ReinL pipeline."""
        self.rl_epoch = rl_epoch
        self.model_suite = model_suite
        self.reward = reward
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.logger = logger
        self.device = get_device(device)
        self.cfg = kwargs  # OmegaConf.create(kwargs)
        self.step = 0
        self.cost = 0

        self.sample_cfg = sample_cfg  # OmegaConf.merge(model_suite.sample_cfg, sample_cfg)
        self.finetune_cfg = finetune_cfg  # OmegaConf.merge(model_suite.finetune_cfg, finetune_cfg)

        self.sampler = model_suite.get_sampler()

        # long-term memory
        self.ltm = None  # LongTimeMem()

        self.models_dir = os.path.join(save_dir, 'models')
        self.sample_dir = os.path.join(save_dir, 'samples')

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        if replay:
            self.replay = None  # ReplayBuffer(**replay_args)
        else:
            self.replay = None

    def init_optimizer(self, lr=5e-4, optimizer=paddle.optimizer.Adam):
        """Initialize the optimizer."""
        self.optimizer = optimizer(learning_rate=lr, parameters=self.agent.parameters())

    def init_scheduler(self, start_factor=0.1, total_iters=10):
        """Initialize the learning rate scheduler."""
        self.scheduler = paddle.optim.lr.LinearWarmUp(
            self.optimizer,
            warmup_steps=int(total_iters),
        )

    def freeze_model(self, freeze):
        """Freeze model parameters."""
        n_freeze = freeze * 4 + 1
        for i, param in enumerate(self.agent.parameters()):
            if i < n_freeze:  # Freeze parameter
                param.stop_gradient = True

    def reward_step(
        self,
        sample_data: list,
        sample_struc: List[Structure],
        xyz_path: str,
        label: str = 'tmp'
    ):
        """Process samples and calculate rewards."""
        rewards, prop_dict, failed_mask = self.reward.scoring(
            (sample_struc, xyz_path), label,
        )
        self.cost += len(sample_struc)

        # remove failed samples/jobs
        success_rewards = rewards[~failed_mask].astype(float)
        success_prop_dict = {
            k: v[~failed_mask] for k, v in prop_dict.items()
        }
        success_data, success_struc = [], []
        for i, failed in enumerate(failed_mask):
            if not failed:
                success_data.append(sample_data[i])
                success_struc.append(sample_struc[i])

        logging.info(f'Evaluation costs to date: {self.cost}')
        logging.info(f'Number of samples that successfully obtained rewards: {len(success_struc)}')
        logging.info(f'reward mean={success_rewards.mean():.4f} std={success_rewards.std():.4f}')
        prop_str = [f'{k} mean={v.mean():.4f} std={v.std():.4f}' for k, v in success_prop_dict.items()]
        logging.info(' | '.join(prop_str))

        return success_data, success_struc, success_rewards, success_prop_dict

    def load_model(self):
        """Load the model. Must be implemented by subclass."""
        raise NotImplementedError

    def sample_step(self):
        """Sample new structures. Must be implemented by subclass."""
        raise NotImplementedError

    def ft_step(self, data_list):
        """Finetune the model. Must be implemented by subclass."""
        raise NotImplementedError

    def rl_step(self):
        """Single RL step. Must be implemented by subclass."""
        raise NotImplementedError

    def run_rl(self):
        """Run the full RL loop. Must be implemented by subclass."""
        raise NotImplementedError
