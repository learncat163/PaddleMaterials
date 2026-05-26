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

import paddle
import numpy as np


def lengths_and_angles_to_lattice(lengths, angles):
    angles_r = paddle.deg2rad(angles)
    coses = paddle.cos(angles_r)
    sins = paddle.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    val = paddle.clip(val, -1., 1.)
    gamma_star = paddle.acos(val)

    vector_a = paddle.stack([
        lengths[:, 0] * sins[:, 1],
        paddle.zeros_like(lengths[:, 0]),
        lengths[:, 0] * coses[:, 1]], axis=1)
    vector_b = paddle.stack([
        -lengths[:, 1] * sins[:, 0] * paddle.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * paddle.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], axis=1)
    vector_c = paddle.stack([
        paddle.zeros_like(lengths[:, 0]),
        paddle.zeros_like(lengths[:, 0]),
        lengths[:, 2]], axis=1)

    return paddle.stack([vector_a, vector_b, vector_c], axis=1)


def lattice_to_lengths_and_angles(lattices):
    lengths = paddle.sqrt(paddle.sum(lattices ** 2, axis=-1))
    angles = paddle.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[..., i] = paddle.clip(paddle.sum(lattices[..., j, :] * lattices[..., k, :], axis=-1) /
                                     (lengths[..., j] * lengths[..., k]), -1., 1.)
    angles = paddle.acos(angles) * 180.0 / np.pi
    return lengths, angles
