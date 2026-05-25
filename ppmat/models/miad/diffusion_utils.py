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
Diffusion utilities for MiAD model.
Converted from PyTorch to PaddlePaddle.
"""

import paddle
import paddle.nn as nn
import math


class SinusoidalTimeEmbeddings(nn.Module):
    """
    Sinusoidal time embeddings for diffusion models.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: Tensor of shape (batch_size,) containing time values

        Returns:
            embeddings: Tensor of shape (batch_size, dim)
        """
        device = time.place
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = paddle.exp(paddle.arange(half_dim, dtype='float32') * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = paddle.concat([paddle.sin(embeddings), paddle.cos(embeddings)], axis=-1)
        return embeddings


class TimeDistribution:
    """
    Time distribution for diffusion sampling.
    """

    def __init__(self, num_steps, cont_time):
        """
        Args:
            num_steps: Total number of diffusion steps
            cont_time: Whether to use continuous time
        """
        self.num_steps = num_steps
        self.cont_time = cont_time

    def sample(self, batch, mode):
        """
        Sample random timesteps.

        Args:
            batch: Batch dictionary containing batch information
            mode: 'uniform' or other sampling mode

        Returns:
            t: Sampled timesteps
        """
        if self.cont_time:
            t = paddle.rand([batch['batch_size']])
        else:
            t = paddle.randint(0, self.num_steps, [batch['batch_size']], dtype='int64')
            t = t.cast('float32')
        return [t, t, t]

    def reverse_time_iterator(self, batch, start_from=None):
        """
        Generate reverse time iterator for sampling.

        Args:
            batch: Batch dictionary
            start_from: Starting timestep (default: num_steps - 1)

        Yields:
            t_vector: Time vectors for each step
        """
        if start_from is None:
            start_from = self.num_steps - 1

        for t_value in range(start_from, -1, -1):
            if self.cont_time:
                t = paddle.full([batch['batch_size']], t_value / self.num_steps, dtype='float32')
            else:
                t = paddle.full([batch['batch_size']], t_value, dtype='float32')
            yield [t, t, t]

    def to_cuda(self, t_vector, batch):
        """
        Move time vectors to device.

        Args:
            t_vector: Time vectors
            batch: Batch dictionary

        Returns:
            t_vector: Time vectors on device
        """
        return t_vector


def mean_interleave(tensor, num_atoms):
    """
    Interleave mean values across atoms.

    Args:
        tensor: Tensor of shape (total_atoms,)
        num_atoms: Number of atoms per sample

    Returns:
        result: Interleaved tensor of shape (num_steps,)
    """
    # TODO: Implement proper interleave logic
    return tensor.mean()
