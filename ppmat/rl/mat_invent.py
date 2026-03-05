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
MatInvent main class for reinforcement learning.

This code is adapted from:
convert-matinvent/pipeline/mat_invent.py
raw-matinvent/pipeline/mat_invent.py
"""

import os
import time
import logging
from typing import Dict, List, Tuple
import numpy as np
from omegaconf import DictConfig
from pymatgen.core.structure import Structure
from ase.io import write

from ppmat.rl.base import ReinL
from ppmat.models.matinvent.rewards.reward import Reward
from ppmat.rl.models.base import ModelSuite


class MatInvent(ReinL):
    """MatInvent reinforcement learning pipeline for material generation."""

    def __init__(
        self,
        rl_epoch: int,
        model_suite: ModelSuite,
        reward: Reward,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        topk_ratio: float,
        save_dir: str,
        save_freq: int = 50,
        device: str = None,
        logger=None,
        replay: bool = False,
        replay_args: Dict = None,
        div_filter: bool = False,
        df_args: Dict = None,
        **kwargs,
    ) -> None:
        super().__init__(
            rl_epoch=rl_epoch,
            model_suite=model_suite,
            reward=reward,
            sample_cfg=sample_cfg,
            finetune_cfg=finetune_cfg,
            save_dir=save_dir,
            save_freq=save_freq,
            device=device,
            logger=logger,
            replay=replay,
            replay_args=replay_args,
            **kwargs,
        )
        assert topk_ratio > 0.0 and topk_ratio <= 1.0
        self.topk_ratio = topk_ratio

        # diversity filter
        self.div_filter = div_filter
        self.df_args = df_args

        self.load_model()

    def load_model(self):
        """Load agent and prior models."""
        self.agent = self.model_suite.load_model()
        self.prior = self.model_suite.load_model()

        for param in self.agent.parameters():
            param.trainable = True
        # Freeze the parameter of prior (pretrained) model
        for param in self.prior.parameters():
            param.stop_gradient = True

    def sample_step(self) -> Tuple[List, List[Structure], str, Dict]:
        """Generate samples using the agent model.

        Returns:
            Tuple of (sample_data, sample_struc, eval_xyz_path, metrics)
        """
        # Generate samples using the sampler
        sample_data, sample_struc = self.sampler.generate(
            model=self.agent,
        )

        # Filter invalid samples (basic validity check)
        valid_data = []
        valid_struc = []
        for data, struc in zip(sample_data, sample_struc):
            if self._is_valid_structure(struc):
                valid_data.append(data)
                valid_struc.append(struc)

        if len(valid_struc) == 0:
            logging.warning("No valid structures generated!")
            return [], [], "", {}

        # save all generated valid structures
        valid_xyz_path = self._save_structures(
            structures=valid_struc,
            save_dir=self.sample_dir,
            filename=f'step_{self.step:0>4d}_valid.extxyz',
        )

        # Filter bad samples by selected metrics (if configured)
        if self.sample_cfg.get('filter'):
            filter_fn = self.sample_cfg.filter
            valid_data, valid_struc, metrics = filter_fn(
                valid_data, valid_struc, None
            )
            logging.info(f'Number of filtered samples: {len(valid_struc)}')
        else:
            metrics = {}

        # max sample size to score/reward
        if self.sample_cfg.get('max_num'):
            max_num = self.sample_cfg.max_num
            if len(valid_struc) > max_num:
                valid_data = valid_data[:max_num]
                valid_struc = valid_struc[:max_num]

        # save structures for evaluation
        eval_xyz_path = self._save_structures(
            structures=valid_struc,
            save_dir=self.sample_dir,
            filename=f'step_{self.step:0>4d}_eval.extxyz',
        )

        return valid_data, valid_struc, eval_xyz_path, metrics

    def _is_valid_structure(self, struc: Structure) -> bool:
        """Check if a structure is valid.

        Args:
            struc: pymatgen Structure

        Returns:
            True if structure is valid
        """
        try:
            # Basic validity checks
            if struc.num_sites == 0:
                return False
            if struc.volume <= 0:
                return False
            # Check for valid coordinates
            for site in struc.sites:
                if not (0 <= site.frac_coords[0] < 1 and
                        0 <= site.frac_coords[1] < 1 and
                        0 <= site.frac_coords[2] < 1):
                    return False
            return True
        except:
            return False

    def _save_structures(
        self,
        structures: List[Structure],
        save_dir: str,
        filename: str,
    ) -> str:
        """Save structures to extxyz file.

        Args:
            structures: List of pymatgen Structure objects
            save_dir: Directory to save to
            filename: Output filename

        Returns:
            Path to saved file
        """
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, filename)

        # Convert pymatgen structures to ASE atoms and write
        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()

        with open(out_path, 'w') as f:
            for struc in structures:
                atoms = adaptor.get_atoms(struc)
                write(out_path, atoms, append=True)

        return out_path

    def ft_step(self, data_list: List, rewards: np.ndarray, baseline: float):
        """Fine-tune the agent model on high-reward samples.

        Args:
            data_list: List of data samples (structures)
            rewards: Array of reward values
            baseline: Baseline reward for advantage calculation
        """
        cfg = self.finetune_cfg
        loader = self.model_suite.get_dataloader(
            samples=data_list,
            rewards=rewards,
            batch_size=cfg.batch_size,
        )

        optimizer = paddle.optimizer.Adam(
            parameters=self.agent.parameters(),
            learning_rate=cfg.lr
        )
        accum_steps = cfg.accum_steps

        for epoch in range(cfg.epochs):
            self.agent.train()

            loss_all, loss_diff_all, loss_kl_all = 0., 0., 0.
            for batch in loader:
                # Calculate advantage (reward - baseline)
                batch_rewards = batch["structure_array"]["reward"]
                adv = batch_rewards - baseline

                loss, loss_diff, loss_kl = 0., 0., 0.

                for t in range(cfg.timesteps):
                    # Noised input from agent
                    noised_input_agent = self.agent.add_noise(batch, t)
                    sample_loss, agent_pred = self.agent.calc_sample_loss(noised_input_agent)

                    # Noised input from prior (frozen)
                    noised_input_prior = self.prior.add_noise(batch, t)
                    _, prior_pred = self.prior.calc_sample_loss(noised_input_prior)

                    # Diffusion loss with advantage weighting
                    _loss_diff = adv * sample_loss

                    # KL regularization
                    kl_term = self.agent.calc_kl_reg(agent_pred, prior_pred, batch)
                    _loss_kl = kl_term * cfg.sigma

                    # Combined loss
                    _loss = (_loss_diff + _loss_kl).mean() / accum_steps

                    # Backward pass
                    _loss.backward()
                    if (t + 1) % accum_steps == 0:
                        optimizer.step()
                        optimizer.clear_grad()

                    # Track losses
                    loss += _loss.item() * accum_steps
                    loss_diff += _loss_diff.sum().item()
                    loss_kl += _loss_kl.sum().item()

                # Average losses over timesteps
                loss_diff = loss_diff / cfg.timesteps
                loss_kl = loss_kl / cfg.timesteps
                loss = loss / cfg.timesteps

                # Handle any remaining gradients
                if (cfg.timesteps) % accum_steps != 0:
                    optimizer.step()
                    optimizer.clear_grad()

                loss_all += loss * len(data_list)
                loss_diff_all += loss_diff
                loss_kl_all += loss_kl

            # Log epoch losses
            loss_dict = {
                'loss': loss_all / len(data_list),
                'loss_diff': loss_diff_all / len(data_list),
                'loss_kl': loss_kl_all / len(data_list),
            }
            log_str = [f'{k}: {v:.4f}' for k, v in loss_dict.items()]
            logging.info(f'Epoch {epoch}: ' + ', '.join(log_str))

    def rl_step(self):
        """Execute one reinforcement learning step."""
        logging.info(f'*****   LOOP {self.step} START   *****')
        start_time = time.time()

        logging.info('SAMPLE:')
        sample_list, sample_struc, xyz_path, sample_metrics = self.sample_step()

        # sample scoring, remove failed samples, ranking and get top k samples
        logging.info('SCORE:')
        sample_list, sample_struc, rewards, prop_dict = self.reward_step(
            sample_list, sample_struc, xyz_path, f'step_{self.step:0>4d}',
        )

        log_dict = {f'{k} mean': v.mean() for k, v in prop_dict.items()}
        log_dict.update({f'{k} std': v.std() for k, v in prop_dict.items()})
        log_dict.update({'reward mean': rewards.mean(), 'reward std': rewards.std()})
        log_dict.update(sample_metrics)

        # long-term memory
        self.ltm.extend(sample_struc, rewards, self.step)
        metrics = self.ltm.calc_metrics(self.reward.threshold)
        self.ltm.save(os.path.join(self.sample_dir, 'long_term_memory.csv'))
        logging.info(
            f'{len(self.ltm)} crystals generated so far, ' +
            f'{len(self.ltm.unique_comps)} unique components.' +
            f'  Burden: {metrics[0]}, Div. Ratio: {metrics[1]}.'
        )
        log_dict.update(
            {
                'crystal_num': len(self.ltm),
                'unique_comps': len(self.ltm.unique_comps),
                'burden': metrics[0],
                'div_ratio': metrics[1],
                'cost': self.cost,
            }
        )
        if self.logger is not None:
            self.logger.log(log_dict, step=self.step)

        # diversity filter
        if self.div_filter:
            rewards, penalty_idx, tol_n, buff_n = self.ltm.div_filter(
                sample_struc, rewards, **self.df_args
            )
            penalty_sample = [sample_list[p] for p in penalty_idx]
            penalty_strucs = [sample_struc[p] for p in penalty_idx]
            logging.info(f'Diversity filter: tol_n={tol_n}, buff_n={buff_n}')

        # topk data points
        sort_idx = np.argsort(rewards)[::-1]
        topk_idx = sort_idx[: int(self.finetune_cfg.batch_size * self.topk_ratio)]
        sample_topk = [sample_list[_i] for _i in topk_idx]
        strucs_topk = [sample_struc[_i] for _i in topk_idx]
        reward_topk = rewards[topk_idx]

        # experience replay
        if self.replay is not None:
            if self.div_filter and len(penalty_strucs) > 0:
                self.replay.memory_purge(penalty_strucs)
            data_replay, reward_replay = self.replay.sample()
            ft_data = sample_topk + data_replay
            ft_reward = np.concatenate((reward_topk, reward_replay))
            self.replay.extend(sample_topk, strucs_topk, reward_topk)
            logging.info(f'replay buffer size={len(self.replay)}')
            logging.info(f'buffer reward mean={self.replay.buffer["reward"].values.mean()}')
        else:
            ft_data = sample_topk
            ft_reward = reward_topk

        # finetuning
        logging.info('FINETUNE:')
        baseline = self.ltm.get_baseline(self.step)
        baseline = min(baseline, ft_reward.min())
        self.ft_step(ft_data, ft_reward, baseline)

        end_time = time.time()
        total_time = (end_time - start_time) / 60
        logging.info(f'*****   LOOP {self.step} FINISH   *****')
        logging.info(f'Total time taken: {total_time:.2f} min.\n\n')

    def run_rl(self):
        """Run the full reinforcement learning loop."""
        logging.info('*****   RL START   *****')
        start_time = time.time()

        for step in range(self.rl_epoch):
            self.step = step
            self.rl_step()
            # Save the agent weights every few iterations
            if (step + 1) % self.save_freq == 0:
                ckpt_dir = os.path.join(self.models_dir, f'loop_{step:0>4d}')
                self.model_suite.save_model(self.agent, ckpt_dir)
        # If the entire training finishes, clean up
        ckpt_dir = os.path.join(self.models_dir, 'final')
        self.model_suite.save_model(self.agent, ckpt_dir)

        logging.info('*****   RL END   *****')
        end_time = time.time()
        logging.info(f'Total time taken: {int(end_time - start_time)} s.')
