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

"""
Diffusion utilities for MiAD model.
"""

import os
import math
import paddle
import paddle.nn as nn


class SinusoidalTimeEmbeddings(nn.Layer):
    """Sinusoidal time embeddings."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = paddle.exp(
            paddle.arange(half_dim, dtype='float32') * -embeddings
        )
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = paddle.concat(
            [paddle.sin(embeddings), paddle.cos(embeddings)], axis=-1
        )
        return embeddings


class TimeDistribution:
    """Time sampling and iteration utilities."""

    def __init__(self, num_steps, cont_time):
        self.num_steps = num_steps
        self.cont_time = cont_time
        self.eps = 1e-3
        modifications = os.environ.get('MODIFICATIONS_FIELD', '')
        if 'time_log_normal_0_1_clip' in modifications:
            print("MODIF: time_log_normal_0_1_clip", flush=True)

    def _atom_expand(self, batch, t):
        if 'batch' in batch:
            num_atoms = batch['batch'].num_atoms
        elif 'batch_idx' in batch:
            num_atoms = batch['num_atoms']
        else:
            raise ValueError("Cannot determine num_atoms from batch")

        t_per_atom = t.repeat_interleave(num_atoms)
        return [t, t_per_atom]

    def sample(self, batch, mode=None):
        modifications = os.environ.get('MODIFICATIONS_FIELD', '')
        if 'time_log_normal_0_1_clip' in modifications:
            s_eps = 2e-2
            t = paddle.clip(
                (1 + 2 * s_eps)
                * paddle.nn.functional.sigmoid(paddle.randn(batch['batch_size']))
                - s_eps,
                0,
                1,
            )
        else:
            t = paddle.rand([batch['batch_size']])

        if 'max_time_' in modifications:
            max_time = float(
                modifications.split("max_time_")[1].split("+")[0]
            )
            t = t * max_time

        t = self.eps + (self.num_steps - 1 - self.eps) * t
        if not self.cont_time:
            t = t.round().cast('int64')

        return self._atom_expand(batch, t)

    def get_time_points_tensor(self, batch, t):
        if t == -1:
            t = self.num_steps - 1
        t = paddle.full(
            [batch['batch_size']], t, dtype='float32'
        )
        return self._atom_expand(batch, t)

    def reverse_time_iterator(self, batch, start_from=-1):
        if start_from == -1:
            start_from = self.num_steps - 1
        for t_val in range(start_from, -1, -1):
            t = paddle.full(
                [batch['batch_size']], t_val, dtype='float32'
            )
            yield self._atom_expand(batch, t)

    def forward_time_iterator(self, batch, start_from):
        for t_val in range(start_from, self.num_steps):
            t = paddle.full(
                [batch['batch_size']], t_val, dtype='float32'
            )
            yield self._atom_expand(batch, t)

    def to_cuda(self, t_vector, batch):
        return t_vector


def mean_interleave(t, num_repeats):
    mean_t = paddle.zeros([num_repeats.shape[0]], dtype=t.dtype)
    shift = 0
    for i, n in enumerate(num_repeats):
        n = int(n)
        mean_t[i] = t[shift : shift + n].mean()
        shift += n
    return mean_t
