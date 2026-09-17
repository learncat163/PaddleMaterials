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
from typing import Tuple

import numpy as np
import paddle
from pymatgen.core.structure import Structure

from ppmat.utils import logger

_ATOM_TYPE_MIN = 1
_ATOM_TYPE_MAX = 100
# One failed sampling call is tolerated and skipped; anything outside this set is a
# bug and must surface instead of being silently swallowed.
_TOLERATED_SAMPLING_ERRORS = (IndexError, RuntimeError, ValueError)


def _process_result(result: dict, build_structure_cfg: dict):
    """Convert one sampled structure into RL batch data and a Structure."""
    num_atoms = int(result["num_atoms"])
    frac_coords = np.asarray(result["frac_coords"], dtype=np.float32)
    atom_types = np.asarray(result["atom_types"], dtype=np.int64)
    lattice = np.asarray(result["lattice"], dtype=np.float32).reshape(3, 3)

    data = {
        "structure_array": {
            "num_atoms": paddle.to_tensor([num_atoms], dtype="int64"),
            "frac_coords": paddle.to_tensor(frac_coords),
            "atom_types": paddle.to_tensor(atom_types),
            "lattice": paddle.to_tensor(lattice).unsqueeze(0),
        }
    }

    from ppmat.datasets.build_structure import BuildStructure

    cfg = dict(build_structure_cfg)
    structure = None
    try:
        structure = BuildStructure.build_one(
            {"frac_coords": frac_coords, "atom_types": atom_types, "lattice": lattice},
            cfg.pop("format"),
            **cfg,
        )
    except Exception as e:
        logger.warning(f"Failed to parse a sampled structure: {e}")
    return data, structure


class BaseSampler:
    """Generate one round of structures from a MatInvent diffusion backbone.

    `min_atoms` / `max_atoms` are the atom-count ranges documented in the model
    README; `build_structure_cfg` is the task config `Sample.build_structure_cfg`
    so sampled structures are parsed exactly like the task sample flow.
    """

    min_atoms = 1
    max_atoms = 50

    def __init__(self, batch_size: int = 16, num_batches: int = 4,
                 num_inference_steps: int = 1000, build_structure_cfg: dict = None):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_inference_steps = num_inference_steps
        self.build_structure_cfg = build_structure_cfg or {"format": "array"}

    def generate(self, model, **kwargs):
        raise NotImplementedError

    def _num_atoms(self, kwargs: dict, batch_size: int, index: int = 0):
        counts = kwargs.get("num_atoms")
        if counts is None:
            return paddle.randint(low=self.min_atoms, high=self.max_atoms + 1,
                                  shape=[batch_size], dtype="int64")
        counts = np.atleast_1d(np.asarray(counts, dtype="int64"))
        selected = counts[(index + np.arange(batch_size)) % counts.size]
        return paddle.to_tensor(
            np.clip(selected, self.min_atoms, self.max_atoms), dtype="int64")

    def _sample(self, model, structure_array: dict):
        try:
            with paddle.no_grad():
                return model.sample(
                    {"structure_array": structure_array},
                    num_inference_steps=self.num_inference_steps)
        except _TOLERATED_SAMPLING_ERRORS as e:
            logger.warning(f"Skipped one sampling call: {e}")
            return None

    def _collect(self, results):
        data_list, structures = [], []
        for result in results:
            data, structure = _process_result(result, self.build_structure_cfg)
            data_list.append(data)
            structures.append(structure)
        return data_list, structures


class MatterGenSampler(BaseSampler):
    min_atoms = 2
    max_atoms = 50

    def generate(self, model, **kwargs) -> Tuple[List, List[Structure]]:
        data_list, structures = [], []
        for idx in range(self.num_batches * self.batch_size):
            num_atoms = self._num_atoms(kwargs, batch_size=1, index=idx)
            num_atoms = paddle.concat([num_atoms, num_atoms], axis=0)
            out = self._sample(model, {"num_atoms": num_atoms})
            if out is None:
                continue
            batch_data, batch_structures = self._collect(out["result"])
            data_list.extend(batch_data)
            structures.extend(batch_structures)
        return data_list, structures


class DiffCSPSampler(BaseSampler):
    min_atoms = 1
    max_atoms = 50

    def generate(self, model, **kwargs) -> Tuple[List, List[Structure]]:
        data_list, structures = [], []
        for _ in range(self.num_batches):
            num_atoms = self._num_atoms(kwargs, batch_size=self.batch_size)
            atom_types = paddle.randint(
                low=_ATOM_TYPE_MIN,
                high=_ATOM_TYPE_MAX + 1,
                shape=[int(num_atoms.sum().item())],
                dtype="int64",
            )
            out = self._sample(
                model, {"num_atoms": num_atoms, "atom_types": atom_types})
            if out is None:
                continue
            batch_data, batch_structures = self._collect(out["result"])
            data_list.extend(batch_data)
            structures.extend(batch_structures)
        return data_list, structures
