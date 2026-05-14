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
OMG: Open Materials Generation

A state-of-the-art generative model for crystal structure prediction and 
de novo generation of inorganic crystals.

Based on Stochastic Interpolants framework (ICML 2025, NeurIPS 2025).
"""

from ppmat.models.omg.datamodule.structure import Structure
from ppmat.models.omg.datamodule.omg_data import OMGData
from ppmat.models.omg.si import (
    StochasticInterpolants,
    SingleStochasticInterpolant,
    Interpolant,
    Corrector,
    TimeChecker,
)
from ppmat.models.omg.sampler import (
    IndependentSampler,
    PositionDistribution,
    CellDistribution,
    SpeciesDistribution,
    MirrorPosition,
    NormalPositionDistribution,
    UniformPositionDistribution,
    MirrorCell,
    NormalCellDistribution,
    InformedLatticeDistribution,
    MirrorSpecies,
    UniformSpeciesDistribution,
    WeightedSpeciesDistribution,
)
from ppmat.models.omg.model import (
    CSPNet,
    CSPLayer,
    SinusoidsEmbedding,
    BetaScheduler,
    SigmaScheduler,
    lattice_params_to_matrix_paddle,
    frac_to_cart_coords,
    cart_to_frac_coords,
    radius_graph_pbc,
)
from ppmat.models.omg.pipeline import (
    ReinL,
    MatInvent,
)

__all__ = [
    # datamodule
    "Structure",
    "OMGData",
    # si
    "StochasticInterpolants",
    "SingleStochasticInterpolant",
    "Interpolant",
    "Corrector",
    "TimeChecker",
    # sampler
    "IndependentSampler",
    "PositionDistribution",
    "CellDistribution",
    "SpeciesDistribution",
    "MirrorPosition",
    "NormalPositionDistribution",
    "UniformPositionDistribution",
    "MirrorCell",
    "NormalCellDistribution",
    "InformedLatticeDistribution",
    "MirrorSpecies",
    "UniformSpeciesDistribution",
    "WeightedSpeciesDistribution",
    # model
    "CSPNet",
    "CSPLayer",
    "SinusoidsEmbedding",
    "BetaScheduler",
    "SigmaScheduler",
    "lattice_params_to_matrix_paddle",
    "frac_to_cart_coords",
    "cart_to_frac_coords",
    "radius_graph_pbc",
    # pipeline
    "ReinL",
    "MatInvent",
]
