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
Model module for OMG (Open Materials Generation).

This module provides the CSPNet model for crystal structure prediction.

Migrated from OMG (Open Materials Generation).
Original code: models.diffcsp
"""

from .cspnet import CSPNet, CSPLayer, SinusoidsEmbedding
from .scheduler import (
    BetaScheduler,
    SigmaScheduler,
    cosine_beta_schedule,
    linear_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule,
    p_wrapped_normal,
    d_log_p_wrapped_normal,
    sigma_norm,
)
from .utils import (
    lattice_params_to_matrix_paddle,
    frac_to_cart_coords,
    cart_to_frac_coords,
    radius_graph_pbc,
    repeat_blocks,
    chemical_symbols,
)

__all__ = [
    "CSPNet",
    "CSPLayer",
    "SinusoidsEmbedding",
    "BetaScheduler",
    "SigmaScheduler",
    "cosine_beta_schedule",
    "linear_beta_schedule",
    "quadratic_beta_schedule",
    "sigmoid_beta_schedule",
    "p_wrapped_normal",
    "d_log_p_wrapped_normal",
    "sigma_norm",
    "lattice_params_to_matrix_paddle",
    "frac_to_cart_coords",
    "cart_to_frac_coords",
    "radius_graph_pbc",
    "repeat_blocks",
    "chemical_symbols",
]
