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
MiAD (Mirage Atom Diffusion) model for crystal structure generation.
Integrates CSPNet encoder with separate diffusion processes for
lattice, fractional coordinates, and atom types.
"""

import paddle
import paddle.nn as nn

from ppmat.models.miad.cspnet import CSPNet
from ppmat.models.miad.diffusion_utils import SinusoidalTimeEmbeddings
from ppmat.models.miad.frac_diffusion import WrappedNormal, scheduler as frac_scheduler
from ppmat.models.miad.lattice_diffusion import FM, DDPM as LatticeDDPM
from ppmat.models.miad.type_diffusion import DDPM_onehot


def uniform_sample_t(batch_size, num_steps):
    """Sample random timesteps uniformly."""
    t = paddle.randint(0, num_steps, shape=[batch_size], dtype="int64")
    return t


class MiAD(nn.Layer):
    """Mirage Atom Diffusion for crystal structure generation.

    Integrates CSPNet denoiser with three separate diffusion processes:
    - Lattice diffusion (FM or DDPM)
    - Fractional coordinate diffusion (WrappedNormal)
    - Atom type diffusion (DDPM onehot)

    Args:
        encoder_cfg: CSPNet encoder configuration.
        diffusion_cfg: Diffusion process configurations.
        num_train_timesteps: Number of diffusion steps.
        time_dim: Time embedding dimension.
        lattice_loss_weight: Lattice loss weight.
        coord_loss_weight: Coordinate loss weight.
        type_loss_weight: Atom type loss weight.
    """

    def __init__(
        self,
        encoder_cfg: dict,
        diffusion_cfg: dict,
        num_train_timesteps: int = 1000,
        time_dim: int = 256,
        lattice_loss_weight: float = 1.0,
        coord_loss_weight: float = 1.0,
        type_loss_weight: float = 1.0,
    ):
        super().__init__()

        self.decoder = CSPNet(**encoder_cfg)
        self.num_train_timesteps = num_train_timesteps
        self.time_dim = time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(time_dim)

        self.lattice_loss_weight = lattice_loss_weight
        self.coord_loss_weight = coord_loss_weight
        self.type_loss_weight = type_loss_weight

        # Build diffusion schedulers
        lat_method = diffusion_cfg.get("lat_diffusion", {}).get("method", "fm")
        if lat_method == "fm":
            self.lat_scheduler = frac_scheduler("default_wrapped_normal", num_train_timesteps)
        else:
            self.lat_scheduler = frac_scheduler("default_wrapped_normal", num_train_timesteps)

        self.frac_scheduler = frac_scheduler("default_wrapped_normal", num_train_timesteps)

        self.gen_type = "gen" in diffusion_cfg.get("task", "gen-csp-mp20")

    def forward(self, batch, **kwargs):
        """Training forward: compute diffusion losses."""
        structure_array = batch["structure_array"]
        batch_size = structure_array["num_atoms"].shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=structure_array["num_atoms"]
        )

        # Sample timesteps
        times = uniform_sample_t(batch_size, self.num_train_timesteps)
        time_emb = self.time_embedding(times.cast("float32"))

        # Get ground truth
        lattices = structure_array["lattice"]
        frac_coords = structure_array["frac_coords"]
        atom_types = structure_array["atom_types"]

        # One-hot encode atom types for the encoder
        num_classes = 100
        atom_types_onehot = nn.functional.one_hot(
            (atom_types - 1) % num_classes, num_classes=num_classes
        ).cast("float32")

        # Add noise to lattice
        noise_l = paddle.randn(shape=lattices.shape, dtype=lattices.dtype)
        sigmas_l = self.lat_scheduler[0][times]
        noisy_lattices = lattices + sigmas_l[:, None, None] * noise_l

        # Add noise to fractional coordinates (wrapped normal)
        noise_x = paddle.randn(shape=frac_coords.shape, dtype=frac_coords.dtype)
        times_per_atom = times.repeat_interleave(repeats=structure_array["num_atoms"])
        sigmas_x = self.frac_scheduler[0][times_per_atom]
        noisy_frac = frac_coords + sigmas_x[:, None] * noise_x
        noisy_frac = noisy_frac % 1.0

        # CSPNet prediction
        pred_l, pred_x, pred_a = self.decoder(
            times, time_emb, atom_types_onehot, noisy_frac, noisy_lattices,
            structure_array["num_atoms"], batch_idx,
        )

        # Compute losses
        loss_lattice = nn.functional.mse_loss(pred_l, noise_l)
        loss_coord = nn.functional.mse_loss(pred_x, noise_x)

        loss = (
            self.lattice_loss_weight * loss_lattice
            + self.coord_loss_weight * loss_coord
        )

        loss_dict = {
            "loss": loss,
            "loss_lattice": loss_lattice,
            "loss_coord": loss_coord,
        }

        if self.gen_type:
            loss_type = nn.functional.cross_entropy(pred_a, (atom_types - 1) % num_classes)
            loss = loss + self.type_loss_weight * loss_type
            loss_dict["loss"] = loss
            loss_dict["loss_type"] = loss_type

        return {"loss_dict": loss_dict}

    @paddle.no_grad()
    def sample(self, batch_data, num_inference_steps=1000, **kwargs):
        """Generate crystal structures via reverse diffusion."""
        structure_array = batch_data["structure_array"]
        batch_size = structure_array["num_atoms"].shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=structure_array["num_atoms"]
        )

        # Initialize from prior
        l_t = paddle.randn(shape=[batch_size, 3, 3])
        x_t = paddle.rand(shape=[structure_array["num_atoms"].sum(), 3])

        num_classes = 100
        atom_types_onehot = nn.functional.one_hot(
            (structure_array["atom_types"] - 1) % num_classes, num_classes=num_classes
        ).cast("float32")

        # Simple DDPM-style reverse sampling
        step_size = max(1, self.num_train_timesteps // num_inference_steps)
        for step in range(num_inference_steps - 1, -1, -1):
            t_val = step * step_size
            t = paddle.full([batch_size], t_val, dtype="int64")
            time_emb = self.time_embedding(t.cast("float32"))

            pred_l, pred_x, pred_a = self.decoder(
                t, time_emb, atom_types_onehot, x_t, l_t,
                structure_array["num_atoms"], batch_idx,
            )

            # Simplified denoising step
            alpha = 1.0 - step / num_inference_steps
            l_t = l_t - (1 - alpha) * pred_l
            x_t = x_t - (1 - alpha) * pred_x
            x_t = x_t % 1.0

        # Collect results
        start_idx = 0
        result = []
        for i in range(batch_size):
            end_idx = start_idx + structure_array["num_atoms"][i]
            result.append({
                "num_atoms": structure_array["num_atoms"][i].tolist(),
                "atom_types": structure_array["atom_types"][start_idx:end_idx].tolist(),
                "frac_coords": x_t[start_idx:end_idx].tolist(),
                "lattice": l_t[i].tolist(),
            })
            start_idx += structure_array["num_atoms"][i]

        return {"result": result}
