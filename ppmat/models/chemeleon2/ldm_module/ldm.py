# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from functools import partial

import numpy as np
import paddle
import paddle.nn as nn

from ppmat.models.chemeleon2.common import apply_lora_to_linear
from ppmat.models.chemeleon2.common import merge_lora_weights
from ppmat.models.chemeleon2.common import print_trainable_parameters
from ppmat.models.chemeleon2.common import to_dense_batch
from ppmat.models.chemeleon2.common.schema import CrystalBatch
from ppmat.models.chemeleon2.common.schema import build_structure_array
from ppmat.models.chemeleon2.common.schema import create_empty_batch
from ppmat.models.chemeleon2.ldm_module.diffusion import create_diffusion
from ppmat.utils import logger
from ppmat.utils.crystal import lattice_params_to_matrix_paddle

# MP-20 caps the structures at 20 atoms per unit cell.
DEFAULT_NUM_ATOMS = 20


class LDMModule(nn.Layer):
    def __init__(
        self,
        denoiser=None,
        augmentation=None,
        diffusion_configs=None,
        condition_module=None,
        vae=None,
        vae_ckpt_path=None,
        ldm_ckpt_path=None,
        lora_configs=None,
    ):
        super().__init__()
        from ppmat.models import build_model

        if isinstance(denoiser, dict):
            self.denoiser = build_model(denoiser)
        else:
            self.denoiser = denoiser

        if isinstance(condition_module, dict):
            self.condition_module = build_model(condition_module)
            self.use_cfg = True
        elif condition_module is not None:
            self.condition_module = condition_module
            self.use_cfg = True
        else:
            self.use_cfg = False

        if isinstance(vae, dict):
            vae = build_model(vae)

        # Registered as a buffer so checkpoints restore it via load_pretrain
        self.register_buffer("latent_std", paddle.to_tensor(1.0))

        self.diffusion_configs = diffusion_configs
        self.augmentation = augmentation
        self.lora_configs = lora_configs

        if diffusion_configs is not None:
            self.diffusion = create_diffusion(**diffusion_configs)
        else:
            self.diffusion = None

        if vae is not None:
            self.vae = vae

        if vae_ckpt_path is not None:
            if not hasattr(self, "vae") or self.vae is None:
                raise ValueError(
                    "vae_ckpt_path requires a built VAE model. "
                    "Pass a VAE config dict via the `vae` parameter."
                )
            vae_state = paddle.load(vae_ckpt_path)
            if "model_state_dict" in vae_state:
                vae_state = vae_state["model_state_dict"]
            self.vae.set_state_dict(vae_state)

        if hasattr(self, "vae"):
            for param in self.vae.parameters():
                param.stop_gradient = True
            self.vae.eval()

        if ldm_ckpt_path is not None:
            checkpoint = paddle.load(ldm_ckpt_path)
            if "model_state_dict" in checkpoint:
                self.set_state_dict(checkpoint["model_state_dict"])
                if "latent_std" in checkpoint:
                    self.latent_std = checkpoint["latent_std"]
            else:
                # Bare state_dict payload without latent_std metadata.
                self.set_state_dict(checkpoint)

        if lora_configs is not None:
            rank = lora_configs.get("r", 8)
            alpha = lora_configs.get("lora_alpha", 16)
            dropout = lora_configs.get("lora_dropout", 0.0)
            target_modules = lora_configs.get("target_modules", None)

            self.denoiser = apply_lora_to_linear(
                self.denoiser,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules,
            )
            print_trainable_parameters(self.denoiser)

    @property
    def condition_names(self):
        """Names of the supported sampling conditions (empty if unconditional).

        ``StructureSampler.sample_by_condition`` reads this attribute to build
        the condition payload of the input dict.
        """
        if not self.use_cfg:
            return None
        return list(self.condition_module.target_condition)

    def forward(self, batch):
        crystal_batch = self._convert_sample_batch(batch)
        loss_dict = self.calculate_loss(crystal_batch, training=True)
        return {"loss_dict": loss_dict}

    def _convert_sample_batch(self, batch):
        structure_array = batch["structure_array"]
        num_atoms = structure_array["num_atoms"]
        batch_size = num_atoms.shape[0]

        crystal_batch = build_structure_array(CrystalBatch(), structure_array)

        # Neutral placeholder lattice; the real one is predicted by the VAE decoder
        if not hasattr(crystal_batch, "lattices") or crystal_batch.lattices is None:
            crystal_batch.lengths = paddle.rand([batch_size, 3]) * 10 + 5
            crystal_batch.angles = paddle.rand([batch_size, 3]) * 60 + 60
            crystal_batch.lattices = lattice_params_to_matrix_paddle(
                crystal_batch.lengths, crystal_batch.angles
            )

        return crystal_batch

    def calculate_loss(self, batch, training=True):
        if not hasattr(self, "vae") or self.vae is None:
            raise ValueError(
                "VAE must be loaded before training. Set vae_ckpt_path in __init__."
            )

        with paddle.no_grad():
            encoded = self.vae.encode(batch)
            x = encoded["posterior"].sample() / self.latent_std
            x, mask = to_dense_batch(x, encoded["batch"])

        t = paddle.randint(
            0, self.diffusion.num_timesteps, shape=[x.shape[0]], dtype="int64"
        )

        y = None
        if self.use_cfg:
            y = batch.y
            assert y is not None, "Batch must contain 'y' field when use_cfg=True"
            y = self.condition_module(y, training=training)
            if y.shape[0] != x.shape[0]:
                # CFG doubling from the condition module; align the latent
                # stream with the duplicated batch.
                x = paddle.concat([x, x], axis=0)
                mask = paddle.concat([mask, mask], axis=0)
                t = paddle.concat([t, t], axis=0)

        model_kwargs = {"mask": mask, "y": y}
        loss_dict = self.diffusion.training_losses(
            model=self.denoiser,
            x_start=x,
            t=t,
            model_kwargs=model_kwargs,
        )
        for key in list(loss_dict.keys()):
            if paddle.is_tensor(loss_dict[key]):
                loss_dict[key] = loss_dict[key].mean()

        loss_dict["total_loss"] = loss_dict.get("loss", paddle.to_tensor([0.0]))
        return loss_dict

    def sample(
        self,
        batch,
        sampler="ddim",
        num_inference_steps=50,
        eta=1.0,
        cfg_scale=2.0,
        return_atoms=False,
        return_structure=False,
        progress=True,
    ):
        if isinstance(batch, dict):
            # Conditions arrive as top-level keys (see sample_by_condition)
            names = self.condition_names or []
            entries = {
                name: ([batch[name]] if isinstance(batch[name], str) else batch[name])
                for name in names
                if name in batch
            }
            if not self.use_cfg:
                # Fail loudly instead of silently ignoring condition payloads
                passed = [
                    name for name in batch if name not in ("structure_array", "id")
                ]
                if passed:
                    raise ValueError(
                        "This LDM was built without a condition_module, but the "
                        f"input dict carries condition fields {passed}. Remove "
                        "them or rebuild the model with a condition_module."
                    )
            batch = self._convert_sample_batch(batch)
            if entries:
                batch.y = entries

        if sampler == "ddim":
            timestep_respacing = "ddim" + str(num_inference_steps)
        else:
            timestep_respacing = str(num_inference_steps)

        if self.diffusion_configs is None:
            raise ValueError("diffusion_configs must be set before sampling")
        sampling_configs = self.diffusion_configs.copy()
        sampling_configs.update(timestep_respacing=timestep_respacing)
        sampling_diffusion = create_diffusion(**sampling_configs)

        sampler_fn = (
            partial(sampling_diffusion.ddim_sample_loop, eta=eta)
            if sampler == "ddim"
            else sampling_diffusion.p_sample_loop
        )

        if progress:
            logger.info(
                f"Using {sampler} sampler with "
                f"{sampling_diffusion.num_timesteps} timesteps."
            )

        if not hasattr(self, "vae") or self.vae is None:
            raise ValueError(
                "VAE must be loaded before sampling. Set vae_ckpt_path in __init__."
            )

        if isinstance(batch.num_nodes, (int, np.integer)):
            num_nodes = int(batch.num_nodes)
        else:
            num_nodes = int(batch.num_nodes.item())
        z = paddle.randn([num_nodes, self.vae.latent_dim])
        z, mask = to_dense_batch(z, batch.batch)

        y = None
        if self.use_cfg:
            y = batch.y
            assert y is not None, "Batch must contain 'y' field when use_cfg=True"
            z = paddle.concat([z, z], axis=0)
            mask = paddle.concat([mask, mask], axis=0)
            y = self.condition_module(y, training=False)

        model_kwargs = {
            "mask": mask,
            "y": y,
        }
        if self.use_cfg:
            model_kwargs["cfg_scale"] = cfg_scale

        model_fn = (
            self.denoiser.forward_with_cfg if self.use_cfg else self.denoiser.forward
        )

        diffusion_out = sampler_fn(
            model=model_fn,
            shape=z.shape,
            noise=z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=progress,
        )

        if self.use_cfg:
            diffusion_out, _ = paddle.chunk(diffusion_out, 2, axis=0)
            mask, _ = paddle.chunk(mask, 2, axis=0)

        diffusion_out = diffusion_out * self.latent_std

        encoded_batch = {
            "x": diffusion_out[mask],
            "num_atoms": batch.num_atoms,
            "batch": batch.batch,
            "token_idx": batch.token_idx,
        }
        decoder_out = self.vae.decode(encoded_batch)
        batch_rec = self.vae.reconstruct(decoder_out, batch)
        batch_rec.mask = mask

        if return_atoms:
            return batch_rec.to_atoms()
        elif return_structure:
            return batch_rec.to_structures()

        # Per-structure array dicts for BuildStructure.build_one(format="array")
        structures_list = batch_rec._split_by_batch_index()
        result_list = []
        for struct_data in structures_list:
            result_dict = {}
            if "atom_types" in struct_data:
                result_dict["atom_types"] = (
                    struct_data["atom_types"].cpu().numpy().tolist()
                )
            if "frac_coords" in struct_data:
                result_dict["frac_coords"] = (
                    struct_data["frac_coords"].cpu().numpy().tolist()
                )
            if "lattices" in struct_data:
                result_dict["lattice"] = struct_data["lattices"].cpu().numpy().tolist()
            result_list.append(result_dict)

        return {"result": result_list}

    def merge_lora(self):
        if self.lora_configs is not None:
            self.denoiser = merge_lora_weights(self.denoiser)
            self.lora_configs = None
            logger.info("LoRA weights merged into base model")
        else:
            logger.info("No LoRA weights to merge")

    def get_config(self):
        config = {
            "diffusion_configs": self.diffusion_configs,
            "augmentation": self.augmentation,
            "lora_configs": self.lora_configs,
            "use_cfg": self.use_cfg,
        }
        if hasattr(self, "vae") and self.vae is not None:
            config["vae"] = self.vae.get_config()
        return config

    def predict(self, data, num_inference_steps=50, sampler="ddim"):
        num_samples = data.get("num_samples", 1) if isinstance(data, dict) else 1
        batch_size = (
            data.get("batch_size", num_samples)
            if isinstance(data, dict)
            else num_samples
        )
        if "num_atoms" in data:
            num_atoms_list = [data["num_atoms"]] * num_samples
        else:
            num_atoms_list = [DEFAULT_NUM_ATOMS] * num_samples

        condition = data.get("condition", None) if isinstance(data, dict) else None
        if condition is not None and not self.use_cfg:
            raise ValueError(
                "This LDM was built without a condition_module (unconditional). "
                "Passing 'condition' has no effect; rebuild the model with "
                "condition_module to use conditional sampling."
            )

        all_results = []
        for i in range(0, num_samples, batch_size):
            cb = min(batch_size, num_samples - i)
            cur = num_atoms_list[i : i + cb]
            batch = create_empty_batch(cur)
            batch.y = condition
            with paddle.no_grad():
                result = self.sample(
                    batch,
                    sampler=sampler,
                    num_inference_steps=num_inference_steps,
                    progress=False,
                )
            all_results.extend(result["result"])
        # Downstream contract of StructureSampler: {"result": [...]}
        return {"result": all_results}
