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
Fractional coordinate diffusion models for MiAD with periodic boundary conditions.
Converted from PyTorch to PaddlePaddle.
"""

import os
import paddle
import numpy as np


def scheduler(scheduler_name, num_steps):
    """
    Create diffusion scheduler.

    Args:
        scheduler_name: Name of the scheduler
        num_steps: Number of diffusion steps

    Returns:
        scheduler_params: Tuple of scheduler parameters
    """
    if scheduler_name == 'default_wrapped_normal':
        # params
        sigma_begin, sigma_end = 0.005, 0.5
        sigmas = paddle.to_tensor(
            np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), num_steps)),
            dtype='float32'
        )

        # approx of WrappedNormal(x|sigma)
        def p_wrapped_normal(x, sigma, N=10, T=1.0):
            p_ = 0
            for i in range(-N, N + 1):
                p_ += paddle.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)
            return p_

        # approx of grad_x(log(WrappedNormal(x|sigma)))
        def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
            p_ = 0
            for i in range(-N, N + 1):
                p_ += (x + T * i) / sigma ** 2 * paddle.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)
            return p_ / p_wrapped_normal(x, sigma, N, T)

        # approx of expectation_x(grad_x(log(WrappedNormal(x|sigma)))^2)
        def sigma_norm(sigma, T=1.0, sn=100000):
            sigmas = paddle.tile(sigma[None, :], [sn, 1])
            x_sample = sigma * paddle.randn(sigmas.shape)
            x_sample = x_sample % T
            normal_ = d_log_p_wrapped_normal(x_sample, sigmas, T=T)
            return (normal_ ** 2).mean(axis=0)

        # computation
        sigmas_norm_ = sigma_norm(sigmas)
        sigmas_t = sigmas
        sigmas_norm_t = sigmas_norm_
        d_log_p = d_log_p_wrapped_normal
        return sigmas_t, sigmas_norm_t, sigma_begin, sigma_end, d_log_p

    raise NotImplementedError(f"Scheduler {scheduler_name} not implemented")


class WrappedNormal:
    """
    Wrapped Normal diffusion for fractional coordinates with periodic boundary conditions.
    """

    def __init__(self, diffusion_config):
        """
        Args:
            diffusion_config: Diffusion configuration dictionary
        """
        self.config = diffusion_config.frac_diffusion
        self.num_steps = diffusion_config.num_steps

        sigmas_t, sigmas_norm_t, sb, se, d_log_p = scheduler(self.config.scheduler, self.num_steps)

        self.sigmas_t = sigmas_t[:, None]
        self.sigmas_norm_t = sigmas_norm_t[:, None]
        self.sb = sb
        self.se = se
        self.d_log_p = d_log_p

        # Optimal gamma for different tasks
        switch_optimal_gamma = {
            'csp_perov5': 5e-7,
            'gen_perov5': 5e-7,
            'csp_mp20': 1e-5,
            'gen_mp20': 1e-5,
            'csp_alex_mp20': 1e-5,
            'gen_alex_mp20': 1e-5,
            'csp_mpts52': 1e-5,
            'gen_mpts52': 1e-5,
            'csp_carbon24': 5e-7,
            'gen_carbon24': 1e-5
        }
        self.step_lr = switch_optimal_gamma[diffusion_config.task]

        self.drift_step_coef = paddle.ones([self.num_steps], dtype='float32')[:, None]
        self.diff_step_coef = paddle.ones([self.num_steps], dtype='float32')[:, None]
        self.to_domain = lambda x: x

    def output_transform(self, x0, batch):
        """Transform output before returning."""
        return x0

    def forward_step_sample(self, x0, t, batch):
        """
        Forward diffusion step: add noise to fractional coordinates.

        Args:
            x0: Original fractional coordinates of shape (total_atoms, 3)
            t: Timestep of shape (batch_size,)
            batch: Batch dictionary

        Returns:
            xt: Noisy fractional coordinates with periodic boundary
        """
        st = self.sigmas_t[t.cast('int64')]
        self.randn_x = paddle.randn_like(x0)
        xt = (x0 + st * self.randn_x) % 1.
        return xt

    def reverse_step_sample_part_1(self, normed_score_pred, xt, t, batch):
        """
        First part of reverse diffusion step.

        Args:
            normed_score_pred: Normalized score prediction
            xt: Noisy fractional coordinates
            t: Timestep
            batch: Batch dictionary

        Returns:
            xt_05: Intermediate denoised coordinates
        """
        st, snt = self.sigmas_t[t.cast('int64')], self.sigmas_norm_t[t.cast('int64')]
        step_size = self.step_lr * (st / self.sb) ** 2
        std_x = paddle.sqrt(2 * step_size)
        drift = -step_size * normed_score_pred * paddle.sqrt(snt) * self.drift_step_coef[t.cast('int64')]
        diffusion = std_x * paddle.randn_like(xt) * self.diff_step_coef[t.cast('int64')]
        xt_05 = xt + drift + diffusion
        return xt_05

    def reverse_step_sample_part_2(self, normed_score_pred, xt_05, t, batch):
        """
        Second part of reverse diffusion step.

        Args:
            normed_score_pred: Normalized score prediction
            xt_05: Intermediate coordinates from part 1
            t: Timestep
            batch: Batch dictionary

        Returns:
            xt_1: Denoised fractional coordinates
        """
        t_idx = t.cast('int64')
        st, st_1 = self.sigmas_t[t_idx], self.sigmas_t[paddle.maximum(t_idx - 1, 0)]
        snt = self.sigmas_norm_t[t_idx]
        step_size = st ** 2 - st_1 ** 2
        std_x = paddle.sqrt((st_1 ** 2 * (st ** 2 - st_1 ** 2)) / (st ** 2))
        drift = -step_size * normed_score_pred * paddle.sqrt(snt) * self.drift_step_coef[t_idx]
        diffusion = std_x * paddle.randn_like(xt_05) * self.diff_step_coef[t_idx]
        xt_1 = (xt_05 + drift + diffusion) % 1.
        return xt_1

    def reverse_step_sample(self, normed_score_pred, xt, t, batch):
        """
        Combined reverse diffusion step.

        Args:
            normed_score_pred: Normalized score prediction
            xt: Noisy fractional coordinates
            t: Timestep
            batch: Batch dictionary

        Returns:
            xt_1: Denoised fractional coordinates
        """
        xt_05 = self.reverse_step_sample_part_1(normed_score_pred, xt, t, batch)
        xt_1 = self.reverse_step_sample_part_2(normed_score_pred, xt_05, t, batch)
        return xt_1

    def prior_sample(self, batch):
        """
        Sample from prior distribution (uniform on [0,1)^3).

        Args:
            batch: Batch dictionary

        Returns:
            sample: Random fractional coordinates
        """
        return paddle.rand([batch['num_atoms'], 3], dtype='float32')

    def loss(self, batch):
        """
        Compute Wrapped Normal loss.

        Args:
            batch: Batch dictionary

        Returns:
            loss: Per-sample loss
        """
        st, snt = self.sigmas_t[batch['t'][1].cast('int64')], self.sigmas_norm_t[batch['t'][1].cast('int64')]
        normed_score_pred = batch['prediction'][1]
        normed_score = self.d_log_p(st * self.randn_x, st) / paddle.sqrt(snt)
        l2 = ((normed_score_pred - normed_score) ** 2).reshape([normed_score_pred.shape[0], -1]).mean(axis=1)

        # mirage infusion code
        if "miad:add_mirage_atoms_upto" in os.environ['MODIFICATIONS_FIELD']:
            mirage_type = 0
            atom_types = batch['x0'][2]
            mask = (atom_types != mirage_type).cast(l2.dtype)
            # coef to keep the same loss scale after average over increased number of atoms
            coef = mask.shape[0] / mask.sum()
            l2 = l2 * mask * coef

        return l2

    def get_x0_prediction(self, normed_score_pred, xt, t, batch):
        """
        Get x0 prediction from score prediction.

        Args:
            normed_score_pred: Normalized score prediction
            xt: Noisy fractional coordinates
            t: Timestep
            batch: Batch dictionary

        Returns:
            x0_pred: Predicted original fractional coordinates
        """
        x0_pred = (xt - self.sigmas_t[t.cast('int64')] * normed_score_pred) % 1.
        return x0_pred
