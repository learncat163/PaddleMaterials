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
Crystal utility functions for MiAD model.
Converted from PyTorch to PaddlePaddle.
"""

import paddle
import numpy as np


def lengths_and_angles_to_lattice(lengths, angles):
    """
    Convert lattice lengths and angles to lattice matrix.

    Args:
        lengths: Tensor of shape (batch_size, 3) containing lattice lengths a, b, c
        angles: Tensor of shape (batch_size, 3) containing lattice angles alpha, beta, gamma in degrees

    Returns:
        lattice: Tensor of shape (batch_size, 3, 3) containing lattice matrix
    """
    angles_r = paddle.deg2rad(angles)
    coses = paddle.cos(angles_r)
    sins = paddle.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
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
    """
    Convert lattice matrix to lattice lengths and angles.

    Args:
        lattices: Tensor of shape (batch_size, 3, 3) containing lattice matrix

    Returns:
        lengths: Tensor of shape (batch_size, 3) containing lattice lengths a, b, c
        angles: Tensor of shape (batch_size, 3) containing lattice angles alpha, beta, gamma in degrees
    """
    lengths = paddle.sqrt(paddle.sum(lattices ** 2, axis=-1))
    angles = paddle.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[..., i] = paddle.clip(paddle.sum(lattices[..., j, :] * lattices[..., k, :], axis=-1) /
                                     (lengths[..., j] * lengths[..., k]), -1., 1.)
    angles = paddle.acos(angles) * 180.0 / np.pi
    return lengths, angles


def frac_and_lattice_to_cart_coords(frac_coords, lattice, num_atoms):
    """
    Convert fractional coordinates to Cartesian coordinates.

    Args:
        frac_coords: Tensor of shape (total_atoms, 3) containing fractional coordinates
        lattice: Tensor of shape (batch_size, 3, 3) containing lattice matrix
        num_atoms: Tensor or list of shape (batch_size,) containing number of atoms per crystal

    Returns:
        cart_coords: Tensor of shape (total_atoms, 3) containing Cartesian coordinates
    """
    if len(lattice.shape) == 2:
        lattice = lattice.unsqueeze(0)
        num_atoms = paddle.tensor(num_atoms)
    proj = paddle.repeat_interleave(lattice, num_atoms, axis=0)
    cart_coords = paddle.einsum('bi,bij->bj', frac_coords, proj)
    return cart_coords
