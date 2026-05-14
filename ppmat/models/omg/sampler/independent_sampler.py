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
Independent Sampler implementation.

This module is migrated from OMG (Open Materials Generation).
Original code: omg.sampler.independent_sampler
"""

import numpy as np
import paddle

from .abstracts import PositionDistribution, CellDistribution, SpeciesDistribution


class IndependentSampler:
    """
    Samples from a product of independent distributions for the cell, positions and species.

    :param position_distribution:
        The base distribution for the atomic positions.
    :type position_distribution: PositionDistribution
    :param cell_distribution:
        The base distribution for the cell.
    :type cell_distribution: CellDistribution
    :param species_distribution:
        The base distribution for the atomic species.
    :type species_distribution: SpeciesDistribution

    :raises ValueError:
        If no distribution is provided.
    """

    def __init__(
        self,
        position_distribution: PositionDistribution | None = None,
        cell_distribution: CellDistribution | None = None,
        species_distribution: SpeciesDistribution | None = None,
    ) -> None:
        """Constructor of the IndependentSampler class."""
        super().__init__()

        self._position_distribution = position_distribution
        self._cell_distribution = cell_distribution
        self._species_distribution = species_distribution

        if position_distribution is None and cell_distribution is None and species_distribution is None:
            raise ValueError("At least one distribution must be provided.")

    def sample(
        self,
        pos: paddle.Tensor | None = None,
        pos_is_fractional: bool | None = None,
        cell: paddle.Tensor | None = None,
        species: paddle.Tensor | None = None,
    ) -> tuple[
        np.ndarray | None,
        bool | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        """
        Sample from each base distribution if it is provided.

        :param pos:
            The positions of the atoms in a single structure.
            Only required if position_distribution is provided.
        :type pos: paddle.Tensor | None
        :param pos_is_fractional:
            Whether the positions are in fractional coordinates.
            Only required if position_distribution is provided.
        :type pos_is_fractional: bool | None
        :param cell:
            The cell of a single structure.
            Only required if cell_distribution is provided.
        :type cell: paddle.Tensor | None
        :param species:
            The species (atomic numbers) of all atoms in the structure.
            Only required if species_distribution is provided.
        :type species: paddle.Tensor | None

        :return:
            (Sampled positions, Whether the sampled positions are in fractional coordinates,
             Sampled cell, Sampled species)
        :rtype: tuple[np.ndarray | None, bool | None, np.ndarray | None, np.ndarray | None]
        """
        sampled_pos = None
        sampled_pos_is_fractional = None
        sampled_cell = None
        sampled_species = None

        if self._position_distribution is not None:
            if pos is None:
                raise ValueError("Position distribution provided but no positions given.")
            if pos_is_fractional is None:
                raise ValueError("Position distribution provided but no coordinate system specified.")
            sampled_pos, sampled_pos_is_fractional = self._position_distribution(
                pos, pos_is_fractional
            )

        if self._cell_distribution is not None:
            if cell is None:
                raise ValueError("Cell distribution provided but no cell given.")
            sampled_cell = self._cell_distribution(cell)

        if self._species_distribution is not None:
            if species is None:
                raise ValueError("Species distribution provided but no species given.")
            sampled_species = self._species_distribution(species)

        return (
            sampled_pos,
            sampled_pos_is_fractional,
            sampled_cell,
            sampled_species,
        )
