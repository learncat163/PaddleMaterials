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
Abstract classes for Sampler module.

This module is migrated from OMG (Open Materials Generation).
Original code: omg.sampler.abstracts
"""

from abc import ABC, abstractmethod

import numpy as np
import paddle


class SpeciesDistribution(ABC):
    """
    Abstract base class for all species base distributions.

    The base distribution can be conditioned on species data.
    """

    def __init__(self) -> None:
        """Constructor for the SpeciesDistribution class."""
        super().__init__()

    @abstractmethod
    def __call__(self, species: paddle.Tensor) -> np.ndarray:
        """
        Sample species from the base distribution given the species of a single structure.

        We use numpy arrays for the returned array because numpy offers a broader collection of distributions.

        :param species:
            The atomic numbers of all atoms in the structure in a tensor of shape (number_atoms, ).
        :type species: paddle.Tensor

        :return:
            A sample of species from the base distribution in a tensor of shape (number_atoms, ).
        :rtype: np.ndarray
        """
        raise NotImplementedError


class CellDistribution(ABC):
    """
    Abstract base class for cell base distributions.

    The base distribution can be conditioned on cell data.
    """

    def __init__(self) -> None:
        """Constructor for the CellDistribution class."""
        super().__init__()

    @abstractmethod
    def __call__(self, cell: paddle.Tensor) -> np.ndarray:
        """
        Sample a cell from the base distribution given the cell of a single structure.

        We use numpy arrays for the returned array because numpy offers a broader collection of distributions.

        :param cell:
            The cell of a single structure in a tensor of shape (3, 3).
        :type cell: paddle.Tensor

        :return:
            A sampled cell from the base distribution in a tensor of shape (3, 3).
        :rtype: np.ndarray
        """
        raise NotImplementedError


class PositionDistribution(ABC):
    """
    Abstract base class for position base distributions.

    Note that the sampled positions can be fractional or Cartesian coordinates.

    The base distribution can be conditioned on position data.
    """

    def __init__(self) -> None:
        """Constructor for the PositionDistribution class."""
        super().__init__()

    @abstractmethod
    def __call__(self, pos: paddle.Tensor, pos_is_fractional: bool) -> tuple[np.ndarray, bool]:
        """
        Sample positions from the base distribution given the atomic positions of a single structure.

        We use numpy arrays for the returned array because numpy offers a broader collection of distributions.

        :param pos:
            A tensor of shape (number_atoms, 3) containing the positions of the atoms in the structure.
        :type pos: paddle.Tensor
        :param pos_is_fractional:
            Whether the input positions are in fractional coordinates.
        :type pos_is_fractional: bool

        :return:
            (A sample of positions from the base distribution in a tensor of shape (number_atoms, 3),
             Whether the sampled positions are in fractional coordinates.)
        :rtype: tuple[np.ndarray, bool]
        """
        raise NotImplementedError
