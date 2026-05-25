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
Lattice diffusion models for MiAD.
Converted from PyTorch to PaddlePaddle.
"""

import paddle
import paddle.nn as nn
import numpy as np


class DDPM(nn.Module):
    """
    DDPM (Denoising Diffusion Probabilistic Model) for lattice diffusion.
    """

    def __init__(self, config):
        super().__init__()
        self.num_steps = config.num_steps
        self.beta = paddle.linspace(config.beta_start, config.beta_end, self.num_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = paddle.cumprod(self.alpha, axis=0)

    def forward_step_sample(self, l0, t, batch):
        """
        Forward diffusion step: add noise to lattice.

        Args:
            l0: Original lattice of shape (batch_size, 3, 3)
            t: Timestep of shape (batch_size,)
            batch: Batch dictionary

        Returns:
            lt: Noisy lattice at timestep t
        """
        device = l0.place
        batch_size = l0.shape[0]

        # Get alpha_bar for each timestep
        alpha_bar_t = self.alpha_bar[t.cast('int64')]

        # Add noise
        noise = paddle.randn(l0.shape)
        lt = paddle.sqrt(alpha_bar_t[:, None, None]) * l0 + paddle.sqrt(1 - alpha_bar_t[:, None, None]) * noise

        return lt

    def reverse_step_sample(self, l_pred, lt, t, batch):
        """
        Reverse diffusion step: denoise lattice.

        Args:
            l_pred: Predicted lattice (x0)
            lt: Noisy lattice at timestep t
            t: Timestep of shape (batch_size,)
            batch: Batch dictionary

        Returns:
            lt_1: Denoised lattice at timestep t-1
        """
        device = lt.place
        batch_size = lt.shape[0]
        t_idx = t.cast('int64')

        # Get diffusion parameters
        alpha_t = self.alpha[t_idx]
        alpha_bar_t = self.alpha_bar[t_idx]
        beta_t = self.beta[t_idx]
        alpha_bar_t_prev = self.alpha_bar[paddle.maximum(t_idx - 1, 0)]

        # Compute predicted x0
        pred_x0 = (lt - paddle.sqrt(1 - alpha_bar_t[:, None, None]) * l_pred) / paddle.sqrt(alpha_bar_t[:, None, None])

        # Compute mean for reverse step
        mean = (paddle.sqrt(alpha_bar_t_prev[:, None, None]) * beta_t[:, None, None] * pred_x0 +
                paddle.sqrt(1 - beta_t[:, None, None]) * (1 - alpha_bar_t_prev[:, None, None]) * lt) / (
                    1 - alpha_bar_t[:, None, None])

        # Add noise if not last step
        if t_idx[0] > 0:
            noise = paddle.randn(lt.shape)
            lt_1 = mean + paddle.sqrt(beta_t[:, None, None]) * noise
        else:
            lt_1 = mean

        return lt_1

    def get_x0_prediction(self, l_pred, lt, t, batch):
        """
        Get x0 prediction from model prediction.

        Args:
            l_pred: Model prediction (usually noise)
            lt: Noisy lattice at timestep t
            t: Timestep
            batch: Batch dictionary

        Returns:
            x0_pred: Predicted original lattice
        """
        alpha_bar_t = self.alpha_bar[t.cast('int64')]
        x0_pred = (lt - paddle.sqrt(1 - alpha_bar_t[:, None, None]) * l_pred) / paddle.sqrt(alpha_bar_t[:, None, None])
        return x0_pred

    def prior_sample(self, batch):
        """
        Sample from prior distribution (pure noise).

        Args:
            batch: Batch dictionary

        Returns:
            sample: Random noise of shape (batch_size, 3, 3)
        """
        batch_size = batch['batch_size']
        return paddle.randn([batch_size, 3, 3])

    def loss(self, batch):
        """
        Compute DDPM loss.

        Args:
            batch: Batch dictionary with 'xt', 'prediction', 't', 'x0'

        Returns:
            loss: Per-sample loss
        """
        lt = batch['xt'][0]  # Noisy lattice
        l_pred = batch['prediction'][0]  # Model prediction (noise)
        l0 = batch['x0'][0]  # Original lattice
        t = batch['t'][0]  # Timestep

        # Target is the noise that was added
        # For simplicity, using MSE loss between predicted and actual noise
        loss = paddle.mean((l_pred - (lt - paddle.sqrt(self.alpha_bar[t.cast('int64')])[:, None, None] * l0) /
                           paddle.sqrt(1 - self.alpha_bar[t.cast('int64')])[:, None, None]) ** 2,
                          axis=[1, 2])

        return loss

    def output_transform(self, x0, batch):
        """
        Transform output before returning.

        Args:
            x0: Output lattice
            batch: Batch dictionary

        Returns:
            x0: Transformed output
        """
        return x0


class FM(nn.Module):
    """
    Flow Matching for lattice diffusion.
    """

    def __init__(self, config):
        super().__init__()
        self.num_steps = config.num_steps

    def forward_step_sample(self, l0, t, batch):
        """
        Forward diffusion step: add noise to lattice.

        Args:
            l0: Original lattice of shape (batch_size, 3, 3)
            t: Timestep of shape (batch_size,)
            batch: Batch dictionary

        Returns:
            lt: Noisy lattice at timestep t
        """
        noise = paddle.randn(l0.shape)
        lt = (1 - t[:, None, None]) * l0 + t[:, None, None] * noise
        return lt

    def reverse_step_sample(self, l_pred, lt, t, batch):
        """
        Reverse diffusion step: denoise lattice.

        Args:
            l_pred: Model prediction (velocity)
            lt: Noisy lattice at timestep t
            t: Timestep of shape (batch_size,)
            batch: Batch dictionary

        Returns:
            lt_1: Denoised lattice at timestep t-1
        """
        # Simple Euler step
        dt = 1.0 / self.num_steps
        lt_1 = lt - dt * l_pred
        return lt_1

    def get_x0_prediction(self, l_pred, lt, t, batch):
        """
        Get x0 prediction from velocity prediction.

        Args:
            l_pred: Velocity prediction
            lt: Noisy lattice at timestep t
            t: Timestep
            batch: Batch dictionary

        Returns:
            x0_pred: Predicted original lattice
        """
        x0_pred = lt - t[:, None, None] * l_pred
        return x0_pred

    def prior_sample(self, batch):
        """
        Sample from prior distribution (pure noise).

        Args:
            batch: Batch dictionary

        Returns:
            sample: Random noise of shape (batch_size, 3, 3)
        """
        batch_size = batch['batch_size']
        return paddle.randn([batch_size, 3, 3])

    def loss(self, batch):
        """
        Compute Flow Matching loss.

        Args:
            batch: Batch dictionary

        Returns:
            loss: Per-sample loss
        """
        l0 = batch['x0'][0]
        lt = batch['xt'][0]
        l_pred = batch['prediction'][0]
        t = batch['t'][0]

        # Target velocity
        target = paddle.randn(l0.shape)
        noise = lt - (1 - t[:, None, None]) * l0

        loss = paddle.mean((l_pred - target) ** 2, axis=[1, 2])
        return loss

    def output_transform(self, x0, batch):
        """Transform output before returning."""
        return x0
