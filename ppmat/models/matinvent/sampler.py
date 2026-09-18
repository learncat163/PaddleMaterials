# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import numpy as np
import paddle
from pymatgen.core.structure import Structure

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.num_atom_dists import NUM_ATOMS_DISTRIBUTIONS
from ppmat.utils import logger

# Reference DiffCSP argmaxes random atom-type logits over 1..MAX_ATOMIC_NUM.
_MAX_ATOMIC_NUM = 100
# Match the atom-count distribution used by the task ``Sample.data``.
_ATOM_COUNT_DISTRIBUTION = "ALEX_MP_20"
# Random atom types make some model calls fail; anything else must surface.
_TOLERATED_SAMPLE_ERRORS = (IndexError, RuntimeError)
# Random atom types also make pymatgen reject some sampled structures.
_TOLERATED_PARSE_ERRORS = (ValueError,)


def _build_structure(result: dict, build_structure_cfg: dict):
    """Parse one sampled structure into a pymatgen ``Structure``."""
    frac_coords = np.asarray(result["frac_coords"], dtype=np.float32)
    atom_types = np.asarray(result["atom_types"], dtype=np.int64)
    lattice = np.asarray(result["lattice"], dtype=np.float32).reshape(3, 3)
    cfg = dict(build_structure_cfg)
    try:
        return BuildStructure.build_one(
            {
                "frac_coords": frac_coords,
                "atom_types": atom_types,
                # The array format expects ``lattice`` as a list.
                "lattice": lattice.tolist(),
            },
            cfg.pop("format"),
            **cfg,
        )
    except _TOLERATED_PARSE_ERRORS as error:
        logger.warning(f"Failed to parse a sampled structure: {error}")
        return None


class BaseSampler:
    """Generate one round of structures from a MatInvent diffusion backbone."""

    def __init__(
        self,
        batch_size: int,
        num_batches: int,
        num_inference_steps: int,
        build_structure_cfg: dict,
    ):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_inference_steps = num_inference_steps
        self.build_structure_cfg = build_structure_cfg

    def generate(self, model) -> List[Structure]:
        raise NotImplementedError

    def _num_atoms(self, batch_size: int):
        distribution = NUM_ATOMS_DISTRIBUTIONS[_ATOM_COUNT_DISTRIBUTION]
        counts = np.random.choice(
            list(distribution.keys()),
            size=batch_size,
            p=list(distribution.values()),
        )
        return paddle.to_tensor(counts, dtype="int64")

    def _sample(self, model, structure_array: dict):
        try:
            with paddle.no_grad():
                return model.sample(
                    {"structure_array": structure_array},
                    num_inference_steps=self.num_inference_steps,
                )
        except _TOLERATED_SAMPLE_ERRORS as error:
            logger.warning(f"Skipped one sampling call: {error}")
            return None

    def _collect(self, results: List[dict]) -> List[Structure]:
        structures = []
        for result in results:
            structure = _build_structure(result, self.build_structure_cfg)
            if structure is not None:
                structures.append(structure)
        return structures


class MatterGenSampler(BaseSampler):
    def generate(self, model) -> List[Structure]:
        structures = []
        for _ in range(self.num_batches):
            num_atoms = self._num_atoms(batch_size=self.batch_size)
            sampled = self._sample(model, {"num_atoms": num_atoms})
            if sampled is None:
                continue
            structures.extend(self._collect(sampled["result"]))
        return structures


class DiffCSPSampler(BaseSampler):
    def generate(self, model) -> List[Structure]:
        structures = []
        for _ in range(self.num_batches):
            num_atoms = self._num_atoms(batch_size=self.batch_size)
            atom_types = paddle.randint(
                low=1,
                high=_MAX_ATOMIC_NUM + 1,
                shape=[int(num_atoms.sum().item())],
                dtype="int64",
            )
            sampled = self._sample(
                model, {"num_atoms": num_atoms, "atom_types": atom_types}
            )
            if sampled is None:
                continue
            structures.extend(self._collect(sampled["result"]))
        return structures
