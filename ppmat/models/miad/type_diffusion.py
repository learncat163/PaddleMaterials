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
Atom type diffusion models for MiAD.
Converted from PyTorch to PaddlePaddle.
"""

import paddle
import paddle.nn.functional as F


class DDPM_onehot:
    """
    DDPM for atom types with one-hot encoding.

    Note: This class inherits logic from lattice DDPM but applies it to discrete atom types.
    """

    def __init__(self, diffusion_config):
        """
        Args:
            diffusion_config: Diffusion configuration dictionary
        """
        # Reuse lattice diffusion config structure
        self.num_steps = diffusion_config.num_steps
        self.beta = paddle.linspace(diffusion_config.beta_start, diffusion_config.beta_end, self.num_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = paddle.cumprod(self.alpha, axis=0)

        # Reshape for atom types
        for attr, value in self.__dict__.items():
            if isinstance(value, paddle.Tensor) and len(value.shape) == 3:
                self.__dict__[attr] = value.reshape([-1, 1])

        self.num_types = 100
        self.to_domain = lambda types: F.one_hot(types - 1, num_classes=self.num_types).cast('float32')
        self.from_domain = lambda onehot: onehot.argmax(axis=-1) + 1

    def forward_step_sample(self, x0, t, batch):
        """
        Forward diffusion step: add noise to atom types (one-hot).

        Args:
            x0: Original atom types of shape (total_atoms,)
            t: Timestep of shape (batch_size,)
            batch: Batch dictionary

        Returns:
            onehot_xt: Noisy one-hot atom types
        """
        onehot_x0 = self.to_domain(x0)
        onehot_xt = self._forward_step_sample(onehot_x0, t, batch)
        return onehot_xt

    def _forward_step_sample(self, onehot_x0, t, batch):
        """Internal forward step for one-hot encoded types."""
        alpha_bar_t = self.alpha_bar[t.cast('int64')]
        noise = paddle.randn(onehot_x0.shape)
        onehot_xt = paddle.sqrt(alpha_bar_t[:, None]) * onehot_x0 + paddle.sqrt(1 - alpha_bar_t[:, None]) * noise
        return onehot_xt

    def reverse_step_sample(self, onehot_eps_pred, onehot_xt, t, batch):
        """
        Reverse diffusion step: denoise atom types.

        Args:
            onehot_eps_pred: Predicted noise (one-hot)
            onehot_xt: Noisy one-hot atom types
            t: Timestep
            batch: Batch dictionary

        Returns:
            onehot_xt_1: Denoised atom types (one-hot or discrete)
        """
        onehot_xt_1 = self._reverse_step_sample(onehot_eps_pred, onehot_xt, t, batch)
        if t[0] == 0:
            xt_1 = self.from_domain(onehot_xt_1)
            return xt_1
        return onehot_xt_1

    def _reverse_step_sample(self, onehot_eps_pred, onehot_xt, t, batch):
        """Internal reverse step for one-hot encoded types."""
        t_idx = t.cast('int64')
        alpha_t = self.alpha[t_idx]
        alpha_bar_t = self.alpha_bar[t_idx]
        beta_t = self.beta[t_idx]
        alpha_bar_t_prev = self.alpha_bar[paddle.maximum(t_idx - 1, 0)]

        # Compute predicted x0
        pred_x0 = (onehot_xt - paddle.sqrt(1 - alpha_bar_t[:, None]) * onehot_eps_pred) / paddle.sqrt(alpha_bar_t[:, None])

        # Compute mean for reverse step
        mean = (paddle.sqrt(alpha_bar_t_prev[:, None]) * beta_t[:, None] * pred_x0 +
                paddle.sqrt(1 - beta_t[:, None]) * (1 - alpha_bar_t_prev[:, None]) * onehot_xt) / (
                    1 - alpha_bar_t[:, None])

        # Add noise if not last step
        if t_idx[0] > 0:
            noise = paddle.randn(onehot_xt.shape)
            onehot_xt_1 = mean + paddle.sqrt(beta_t[:, None]) * noise
        else:
            onehot_xt_1 = mean

        return onehot_xt_1

    def prior_sample(self, batch):
        """
        Sample from prior distribution (random noise).

        Args:
            batch: Batch dictionary

        Returns:
            sample: Random noise of shape (num_atoms, num_types)
        """
        return paddle.randn([batch['num_atoms'], self.num_types], dtype='float32')

    def loss(self, batch):
        """
        Compute DDPM loss for atom types.

        Args:
            batch: Batch dictionary

        Returns:
            loss: Per-sample loss
        """
        eps_pred = batch['prediction'][2]
        l2 = ((eps_pred - self.randn_x) ** 2).reshape([eps_pred.shape[0], -1]).mean(axis=1)
        return l2

    def get_x0_prediction(self, onehot_eps_pred, onehot_xt, t, batch, x0_format='disc'):
        """
        Get x0 prediction from noise prediction.

        Args:
            onehot_eps_pred: Predicted noise
            onehot_xt: Noisy one-hot atom types
            t: Timestep
            batch: Batch dictionary
            x0_format: Output format ('onehot' or 'disc')

        Returns:
            x0_pred: Predicted original atom types
        """
        alpha_bar_t = self.alpha_bar[t.cast('int64')]
        onehot_x0_pred = (onehot_xt - paddle.sqrt(1 - alpha_bar_t[:, None]) * onehot_eps_pred) / paddle.sqrt(alpha_bar_t[:, None])

        if x0_format == 'onehot':
            return onehot_x0_pred
        elif x0_format == 'disc':
            return self.from_domain(onehot_x0_pred)

    def output_transform(self, x0, batch):
        """Transform output before returning."""
        return x0
