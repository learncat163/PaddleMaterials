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
Stochastic Interpolants (SI) module for OMG.

This module is migrated from OMG (Open Materials Generation).
Original code: omg.si
"""

from ppmat.models.omg.si.abstracts import (
    Corrector,
    Epsilon,
    Interpolant,
    LatentGamma,
    Sigma,
    StochasticInterpolant,
    StochasticInterpolantSpecies,
    TimeChecker,
)
from ppmat.models.omg.si.corrector import (
    IdentityCorrector,
    PeriodicBoundaryConditionsCorrector,
)
from ppmat.models.omg.si.interpolants import (
    ExponentialInterpolant,
    LinearInterpolant,
    PeriodicLinearInterpolant,
    TrigonometricInterpolant,
)
from ppmat.models.omg.si.single_stochastic_interpolant import (
    DifferentialEquationType,
    SingleStochasticInterpolant,
)
from ppmat.models.omg.si.single_stochastic_interpolant_identity import (
    SingleStochasticInterpolantIdentity,
)
from ppmat.models.omg.si.stochastic_interpolants import (
    BIG_TIME,
    DataField,
    SMALL_TIME,
    reshape_t,
    StochasticInterpolants,
)

__all__ = [
    # Abstracts
    "Corrector",
    "Epsilon",
    "Interpolant",
    "LatentGamma",
    "Sigma",
    "StochasticInterpolant",
    "StochasticInterpolantSpecies",
    "TimeChecker",
    # Corrector
    "IdentityCorrector",
    "PeriodicBoundaryConditionsCorrector",
    # Interpolants
    "ExponentialInterpolant",
    "LinearInterpolant",
    "PeriodicLinearInterpolant",
    "TrigonometricInterpolant",
    # Single Stochastic Interpolants
    "DifferentialEquationType",
    "SingleStochasticInterpolant",
    "SingleStochasticInterpolantIdentity",
    # Main class
    "BIG_TIME",
    "DataField",
    "SMALL_TIME",
    "reshape_t",
    "StochasticInterpolants",
]
