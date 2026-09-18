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
import time
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import paddle
from omegaconf import OmegaConf
from pymatgen.core.structure import Structure

from ppmat.models.matinvent.dataset import create_rl_dataloader
from ppmat.models.matinvent.dataset import is_valid_structure
from ppmat.models.matinvent.dataset import save_structures
from ppmat.models.matinvent.memory import LongTimeMem
from ppmat.models.matinvent.memory import ReplayBuffer
from ppmat.models.matinvent.rewards.reward import Reward
from ppmat.optimizer import build_optimizer
from ppmat.utils import logger
from ppmat.utils.crystal import compute_lattice_polar_decomposition
from ppmat.utils.misc import make_noise_symmetric_preserve_variance
from ppmat.utils.scatter import scatter

# Fixed offset of the reference advantage term: ``1.1 - reward``.
_ADVANTAGE_REWARD_OFFSET = 1.1


class MatInvent:
    """Sample, score and fine-tune a diffusion backbone with reward-weighted loss."""

    def __init__(
        self,
        rl_epoch: int,
        model_suite,
        reward: Reward,
        topk_ratio: float,
        save_dir: str,
        save_freq: int = 50,
        replay: bool = False,
        replay_args: Dict = None,
        div_filter: bool = False,
        df_args: Dict = None,
    ):
        if not 0.0 < topk_ratio <= 1.0:
            raise ValueError(f"topk_ratio must be in (0.0, 1.0], but got {topk_ratio}")
        batch_size = model_suite.finetune_cfg["batch_size"]
        topk_num = int(batch_size * topk_ratio)
        if topk_num < 1:
            raise ValueError(
                f"finetune_cfg.batch_size * topk_ratio must select at least one "
                f"structure, but got batch_size={batch_size} and "
                f"topk_ratio={topk_ratio}"
            )
        self.topk_num = topk_num
        self.rl_epoch = rl_epoch
        self.model_suite = model_suite
        self.reward = reward
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.step = 0
        self.cost = 0
        self.finetune_cfg = OmegaConf.create(model_suite.finetune_cfg)
        self.sampler = model_suite.get_sampler()
        self.ltm = LongTimeMem()
        self.models_dir = os.path.join(save_dir, "models")
        self.sample_dir = os.path.join(save_dir, "samples")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        self.replay = ReplayBuffer(**replay_args) if replay else None
        self.div_filter = div_filter
        self.df_args = df_args
        self.load_model()
        # Branch on the atom-scheduler capability, not on the backbone name.
        self.has_atom_diffusion = hasattr(self.agent, "atom_scheduler")

    def load_model(self):
        self.agent = self.model_suite.load_model()
        self.prior = self.model_suite.load_model()
        for param in self.agent.parameters():
            param.stop_gradient = False
        for param in self.prior.parameters():
            param.stop_gradient = True
        self.agent.train()
        self.prior.eval()

    def _save_model(self, model, ckpt_dir: str):
        os.makedirs(ckpt_dir, exist_ok=True)
        paddle.save(model.state_dict(), os.path.join(ckpt_dir, "model.pdparams"))

    def reward_step(
        self, sample_struc: List[Structure], xyz_path: str, label: str = "tmp"
    ) -> Tuple[List[Structure], np.ndarray, Dict]:
        rewards, prop_dict, failed_mask = self.reward.scoring(
            (sample_struc, xyz_path), label
        )
        self.cost += len(sample_struc)
        success = ~failed_mask
        scaled_rewards = rewards[success].astype(float)
        scaled_props = {k: v[success] for k, v in prop_dict.items()}
        strucs = [struc for struc, ok in zip(sample_struc, success) if ok]
        logger.info(f"Evaluation costs to date: {self.cost}")
        logger.info(
            f"Number of samples that successfully obtained rewards: {len(strucs)}"
        )
        if len(scaled_rewards) > 0:
            logger.info(
                f"reward mean={scaled_rewards.mean():.4f} "
                f"std={scaled_rewards.std():.4f}"
            )
            logger.info(
                " | ".join(
                    f"{k} mean={v.mean():.4f} std={v.std():.4f}"
                    for k, v in scaled_props.items()
                )
            )
        else:
            logger.info("reward mean=nan std=nan")
        return strucs, scaled_rewards, scaled_props

    def sample_step(self) -> Tuple[List[Structure], str]:
        self.agent.eval()
        structures = self.sampler.generate(self.agent)
        valid_struc = [struc for struc in structures if is_valid_structure(struc)]
        if len(valid_struc) == 0:
            logger.warning("No valid structures generated!")
            return [], ""
        save_structures(
            structures=valid_struc,
            save_dir=self.sample_dir,
            filename=f"step_{self.step:0>4d}_valid.extxyz",
        )
        eval_xyz_path = save_structures(
            structures=valid_struc,
            save_dir=self.sample_dir,
            filename=f"step_{self.step:0>4d}_eval.extxyz",
        )
        return valid_struc, eval_xyz_path

    def ft_step(self, structures: List[Structure], rewards: np.ndarray):
        cfg = self.finetune_cfg
        loader = create_rl_dataloader(
            structures=structures, rewards=rewards, batch_size=cfg["batch_size"]
        )
        optimizer, _ = build_optimizer(
            cfg["optimizer_cfg"], (self.agent,), cfg["epochs"], len(loader)
        )
        accum_steps = cfg["accum_steps"]
        for epoch in range(cfg["epochs"]):
            self.agent.train()
            loss_all = loss_diff_all = loss_kl_all = 0.0
            for batch in loader:
                adv = batch["structure_array"]["reward"]
                n_graphs = int(batch["structure_array"]["num_atoms"].shape[0])
                loss = loss_diff = loss_kl = 0.0
                for t in range(cfg["timesteps"]):
                    noised_input = self._add_noise_to_model(self.agent, batch, t)
                    sample_loss, agent_pred = self._calc_sample_loss_from_model(
                        self.agent, noised_input
                    )
                    with paddle.no_grad():
                        _, prior_pred = self._calc_sample_loss_from_model(
                            self.prior, noised_input
                        )
                    _loss_diff = adv * sample_loss
                    kl_term = self._calc_kl_reg_from_models(
                        agent_pred, prior_pred, batch
                    )
                    _loss_kl = kl_term * (_ADVANTAGE_REWARD_OFFSET - adv)
                    _loss = (_loss_diff + _loss_kl * cfg["sigma"]).mean() / accum_steps
                    _loss.backward()
                    if (t + 1) % accum_steps == 0:
                        optimizer.step()
                        optimizer.clear_grad()
                    loss += _loss.item() * accum_steps
                    loss_diff += _loss_diff.sum().item()
                    loss_kl += _loss_kl.sum().item()
                if cfg["timesteps"] % accum_steps != 0:
                    optimizer.step()
                    optimizer.clear_grad()
                loss_diff /= cfg["timesteps"]
                loss_kl /= cfg["timesteps"]
                loss /= cfg["timesteps"]
                loss_all += loss * n_graphs
                loss_diff_all += loss_diff
                loss_kl_all += loss_kl
            logger.info(
                f"Epoch {epoch}: loss={loss_all / len(structures):.4f}, "
                f"loss_diff={loss_diff_all / len(structures):.4f}, "
                f"loss_kl={loss_kl_all / len(structures):.4f}"
            )

    def rl_step(self):
        logger.info(f"*****   LOOP {self.step} START   *****")
        start_time = time.time()
        logger.info("SAMPLE:")
        sample_struc, xyz_path = self.sample_step()
        logger.info("SCORE:")
        sample_struc, rewards, prop_dict = self.reward_step(
            sample_struc, xyz_path, f"step_{self.step:0>4d}"
        )
        has_rewards = len(rewards) > 0
        if has_rewards:
            log_dict = {f"{k} mean": v.mean() for k, v in prop_dict.items()}
            log_dict.update({f"{k} std": v.std() for k, v in prop_dict.items()})
            log_dict.update(
                {"reward mean": rewards.mean(), "reward std": rewards.std()}
            )
            self.ltm.extend(sample_struc, rewards, self.step)
        else:
            log_dict = {f"{k} mean": float("nan") for k in prop_dict}
            log_dict.update({f"{k} std": float("nan") for k in prop_dict})
            log_dict.update({"reward mean": float("nan"), "reward std": float("nan")})
            logger.warning(
                "No successful rewards, skip memory/replay/finetune updates."
            )
        metrics = self.ltm.calc_metrics(self.reward.threshold)
        self.ltm.save(os.path.join(self.sample_dir, "long_term_memory.csv"))
        logger.info(
            f"{len(self.ltm)} crystals, {len(self.ltm.unique_comps)} unique. "
            f"Burden: {metrics[0]}, Div.Ratio: {metrics[1]}."
        )
        log_dict.update(
            {
                "crystal_num": len(self.ltm),
                "unique_comps": len(self.ltm.unique_comps),
                "burden": metrics[0],
                "div_ratio": metrics[1],
                "cost": self.cost,
            }
        )
        logger.info(f"step={self.step} metrics={log_dict}")
        if not has_rewards:
            if self.replay is not None and len(self.replay) > 0:
                logger.info("FINETUNE (replay fallback):")
                replay_data, replay_rewards = self.replay.sample()
                if len(replay_data) > 0:
                    self.ft_step(replay_data, np.asarray(replay_rewards, dtype=float))
            logger.info(f"*****   LOOP {self.step} FINISH   *****")
            logger.info(f"Total time: {(time.time() - start_time) / 60:.2f} min.\n")
            return
        penalty_strucs = []
        if self.div_filter:
            rewards, penalty_idx, tol_n, buff_n = self.ltm.div_filter(
                sample_struc, rewards, **self.df_args
            )
            penalty_strucs = [sample_struc[i] for i in penalty_idx]
            logger.info(f"Diversity filter: tol_n={tol_n}, buff_n={buff_n}")
        sort_idx = np.argsort(rewards)[::-1]
        topk_idx = sort_idx[: self.topk_num]
        strucs_topk = [sample_struc[i] for i in topk_idx]
        reward_topk = rewards[topk_idx]
        if self.replay is not None:
            if len(penalty_strucs) > 0:
                self.replay.memory_purge(penalty_strucs)
            replay_data, replay_rewards = self.replay.sample()
            ft_data = strucs_topk + replay_data
            ft_reward = np.concatenate((reward_topk, replay_rewards))
            self.replay.extend(strucs_topk, reward_topk)
            logger.info(f"replay buffer size={len(self.replay)}")
        else:
            ft_data, ft_reward = strucs_topk, reward_topk
        logger.info("FINETUNE:")
        self.ft_step(ft_data, ft_reward)
        logger.info(f"*****   LOOP {self.step} FINISH   *****")
        logger.info(f"Total time: {(time.time() - start_time) / 60:.2f} min.\n")

    def run_rl(self):
        logger.info("*****   RL START   *****")
        start_time = time.time()
        for step in range(self.rl_epoch):
            self.step = step
            self.rl_step()
            if (step + 1) % self.save_freq == 0:
                self._save_model(
                    self.agent, os.path.join(self.models_dir, f"loop_{step:0>4d}")
                )
        self._save_model(self.agent, os.path.join(self.models_dir, "final"))
        logger.info(
            f"*****   RL END   ***** Total time: {int(time.time() - start_time)} s."
        )

    def _add_noise_to_model(self, model, batch, timestep: int):
        sa = batch["structure_array"]
        num_atoms = sa["num_atoms"]
        batch_size = num_atoms.shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=num_atoms
        )
        num_train_timesteps = model.num_train_timesteps
        if self.has_atom_diffusion:
            t = paddle.full(
                [batch_size],
                paddle.linspace(
                    model.max_t, 1.0 / num_train_timesteps, num_train_timesteps
                )[timestep],
            )
        else:
            t = paddle.full([batch_size], timestep, dtype="int64")
        frac_coords = sa["frac_coords"] % 1.0
        rand_x = paddle.randn(shape=frac_coords.shape, dtype=frac_coords.dtype)
        if self.has_atom_diffusion:
            input_frac_coords = model.coord_scheduler.add_noise(
                frac_coords,
                rand_x,
                timesteps=t,
                batch_idx=batch_idx,
                num_atoms=num_atoms,
            )
        else:
            input_frac_coords = model.coord_scheduler.add_noise(
                frac_coords, rand_x, timesteps=t.repeat_interleave(repeats=num_atoms)
            )
            input_frac_coords = input_frac_coords % 1.0
        lattices = sa["lattice"]
        rand_l = paddle.randn(shape=lattices.shape, dtype=lattices.dtype)
        if self.has_atom_diffusion:
            # Reference applies ``symmetrize_lattice`` (polar decomposition) first.
            lattices = compute_lattice_polar_decomposition(lattices)
            rand_l = make_noise_symmetric_preserve_variance(rand_l)
        noise_kwargs = {"timesteps": t}
        if self.has_atom_diffusion:
            noise_kwargs["num_atoms"] = num_atoms
        input_lattice = model.lattice_scheduler.add_noise(
            lattices, rand_l, **noise_kwargs
        )
        noisy = {
            "structure_array": {
                "frac_coords": input_frac_coords,
                "lattice": input_lattice,
                "num_atoms": num_atoms,
            },
            "batch_idx": batch_idx,
            "rand_l": rand_l,
            "rand_x": rand_x,
            "clean_frac_coords": frac_coords,
        }
        if self.has_atom_diffusion:
            atom_types_zero_based = sa["atom_types"] - 1
            input_atom_types_zero_based = model.atom_scheduler.add_noise(
                atom_types_zero_based, timesteps=t, batch_idx=batch_idx
            )
            noisy["structure_array"]["atom_types"] = input_atom_types_zero_based + 1
            noisy["atom_type_zero_based"] = atom_types_zero_based
            noisy["input_atom_type_zero_based"] = input_atom_types_zero_based
        else:
            noisy["structure_array"]["atom_types"] = sa["atom_types"]
        return noisy, batch, t

    def _calc_sample_loss_from_model(self, model, noised_input):
        noisy_batch, clean_batch, t = noised_input
        batch_idx = noisy_batch["batch_idx"]
        sn = noisy_batch["structure_array"]
        num_atoms = sn["num_atoms"]
        if self.has_atom_diffusion:
            noise_batch = {
                "frac_coords": sn["frac_coords"],
                "lattice": sn["lattice"],
                "atom_types": sn["atom_types"],
                "num_atoms": num_atoms,
                "batch": batch_idx,
            }
            output = model.model(noise_batch, t)
            eps_pos, lattice_update = output["frac_coords"], output["lattice"]
        else:
            pred_l, eps_pos = model.decoder(
                model.time_embedding(t),
                sn["atom_types"] - 1,
                sn["frac_coords"],
                sn["lattice"],
                num_atoms,
                batch_idx,
            )
            lattice_update = pred_l
        rand_l = noisy_batch["rand_l"]
        clean_frac_coords = noisy_batch["clean_frac_coords"]
        if self.has_atom_diffusion:
            from ppmat.models.mattergen.mattergen import wrapped_normal_loss

            loss_lattice = (lattice_update + rand_l).square().mean(axis=[1, 2])
            loss_coord = wrapped_normal_loss(
                corruption=model.coord_scheduler,
                score_model_output=eps_pos,
                t=t,
                batch_idx=batch_idx,
                batch_size=num_atoms.shape[0],
                x=clean_frac_coords,
                noisy_x=sn["frac_coords"],
                reduce="sum",
                batch=clean_batch["structure_array"],
            )
            loss_atom_type, _, _ = model.atom_scheduler.compute_loss(
                score_model_output=output["atom_types"],
                t=t,
                batch_idx=batch_idx,
                batch_size=num_atoms.shape[0],
                x=noisy_batch["atom_type_zero_based"],
                noisy_x=noisy_batch["input_atom_type_zero_based"],
                reduce="sum",
                d3pm_hybrid_lambda=model.d3pm_hybrid_lambda,
            )
            total_loss = (
                model.coord_loss_weight * loss_coord
                + model.lattice_loss_weight * loss_lattice
                + model.atom_loss_weight * loss_atom_type
            )
            prediction_dict = {
                "pos": eps_pos,
                "cell": lattice_update,
                "atomic_numbers": output["atom_types"],
            }
        else:
            loss_lattice = (lattice_update - rand_l).square().mean(axis=[1, 2])
            loss_coord = paddle.pow(eps_pos - clean_frac_coords, 2).mean(axis=1)
            loss_coord = scatter(loss_coord, batch_idx, dim=0, reduce="mean")
            total_loss = (
                model.coord_loss_weight * loss_coord
                + model.lattice_loss_weight * loss_lattice
            )
            prediction_dict = {"coords": eps_pos, "lattice": lattice_update}
        return total_loss, prediction_dict

    def _calc_kl_reg_from_models(self, agent_pred, prior_pred, batch):
        num_atoms = batch["structure_array"]["num_atoms"]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(num_atoms.shape[0]), repeats=num_atoms
        )
        if self.has_atom_diffusion:
            pred_x, pred_l = agent_pred["pos"], agent_pred["cell"]
            prior_x = prior_pred["pos"].detach()
            prior_l = prior_pred["cell"].detach()
        else:
            pred_x, pred_l = agent_pred["coords"], agent_pred["lattice"]
            prior_x = prior_pred["coords"].detach()
            prior_l = prior_pred["lattice"].detach()
        kl = paddle.pow(pred_l - prior_l, 2).mean(axis=(1, 2))
        kl += scatter(
            paddle.pow(pred_x - prior_x, 2).mean(axis=1),
            batch_idx,
            dim=0,
            reduce="mean",
        )
        if self.has_atom_diffusion:
            pred_t = agent_pred["atomic_numbers"]
            prior_t = prior_pred["atomic_numbers"].detach()
            kl += scatter(
                paddle.pow(pred_t - prior_t, 2).mean(axis=1),
                batch_idx,
                dim=0,
                reduce="mean",
            )
        return kl
