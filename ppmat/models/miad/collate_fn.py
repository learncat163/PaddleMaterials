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
MiAD batch collate functions.
Converts from PaddleMaterials DefaultCollator output to MiAD CrystalGen format.
"""

import numpy as np
import paddle
from collections import namedtuple

from ppmat.datasets.custom_data_type import ConcatData


def _extract_concat_data(data):
    """Extract numpy array from ConcatData or passthrough."""
    if data is None:
        return None
    if isinstance(data, ConcatData):
        return np.array(data.data)
    return data


class MiADCollator:
    """Collate function that converts MP20Dataset batch to MiAD CrystalGen format.

    Converts from:
        batch["structure_array"] = {
            "frac_coords": ConcatData([N, 3]),
            "atom_types": ConcatData([N]),
            "lattice": ConcatData([B, 3, 3]),
            "num_atoms": ConcatData([B]),
        }
    To:
        batch["x0"] = [lattice, frac_coords, atom_types]
        batch["batch_size"] = B
        batch["batch"] = BatchInfo(num_atoms, batch_idx, atom_types)
    """

    def __call__(self, batch_list):
        """Collate a list of samples into MiAD format.

        Args:
            batch_list: List of samples from MP20Dataset.__getitem__
                OR list of structure_array dicts (when called recursively by DefaultCollator)

        Returns:
            Dict compatible with MiAD CrystalGen.train_step()
        """
        if not batch_list:
            return {}

        sample = batch_list[0]

        # Check if this is being called recursively by DefaultCollator
        # (where batch_list is a list of structure_array dicts)
        if isinstance(sample, dict):
            # Check if this is a structure_array dict (directly has frac_coords, atom_types, etc.)
            # vs a full dataset sample dict (has structure_array key)
            if 'frac_coords' in sample and 'atom_types' in sample and 'lattice' in sample:
                # This is a structure_array dict from recursive call
                # We need to treat it as a single sample's structure_array
                sa = sample
            elif 'structure_array' in sample:
                # This is a full dataset sample dict
                sa = sample["structure_array"]
            else:
                # Unknown format, return as-is
                return batch_list[0] if len(batch_list) == 1 else batch_list
        else:
            # Not a dict, return as-is
            return batch_list[0] if len(batch_list) == 1 else batch_list

        batch_size = len(batch_list)

        # Concatenate variable-length data
        all_frac_coords = []
        all_atom_types = []
        all_num_atoms = []
        all_lattices = []
        batch_idx_list = []

        for i, sample_i in enumerate(batch_list):
            # Extract structure data from sample
            if isinstance(sample_i, dict):
                if "structure_array" in sample_i:
                    sa_i = sample_i["structure_array"]
                elif "frac_coords" in sample_i:
                    sa_i = sample_i
                else:
                    continue
            else:
                continue

            frac_coords = _extract_concat_data(sa_i.get("frac_coords"))
            atom_types = _extract_concat_data(sa_i.get("atom_types"))
            lattice = _extract_concat_data(sa_i.get("lattice"))
            num_atoms = _extract_concat_data(sa_i.get("num_atoms"))

            # Handle single sample vs batch
            if frac_coords is None or atom_types is None or lattice is None:
                continue

            # Determine number of atoms
            if frac_coords.ndim >= 2:
                n_atoms = frac_coords.shape[0]
            elif num_atoms is not None:
                if isinstance(num_atoms, np.ndarray) and num_atoms.ndim > 0:
                    n_atoms = int(num_atoms.flatten()[0])
                else:
                    n_atoms = int(num_atoms)
            else:
                continue

            # Reshape to standard form
            if frac_coords.ndim == 1:
                frac_coords = frac_coords.reshape(-1, 3)
            elif frac_coords.ndim > 2:
                frac_coords = frac_coords.reshape(-1, 3)

            if atom_types.ndim > 1:
                atom_types = atom_types.flatten()

            # Lattice shape: [3, 3] per sample, need [B, 3, 3]
            if lattice.ndim == 2:
                lattice = lattice.reshape(1, 3, 3)
            elif lattice.ndim == 1:
                lattice = lattice.reshape(1, 3, 3)

            all_frac_coords.append(frac_coords)
            all_atom_types.append(atom_types)
            all_lattices.append(lattice[0] if lattice.shape[0] == 1 else lattice)
            all_num_atoms.append(n_atoms)
            batch_idx_list.extend([i] * n_atoms)

        # Concatenate
        frac_coords_cat = np.concatenate(all_frac_coords, axis=0).astype("float32")
        atom_types_cat = np.concatenate(all_atom_types, axis=0).astype("int64")
        lattices_cat = np.stack(all_lattices, axis=0).astype("float32")
        num_atoms_arr = np.array(all_num_atoms, dtype="int64")
        batch_idx = np.array(batch_idx_list, dtype="int64")

        # Convert to paddle tensors
        frac_coords_t = paddle.to_tensor(frac_coords_cat)
        atom_types_t = paddle.to_tensor(atom_types_cat)
        lattices_t = paddle.to_tensor(lattices_cat)
        num_atoms_t = paddle.to_tensor(num_atoms_arr)
        batch_idx_t = paddle.to_tensor(batch_idx)

        # Return dict with individual keys that DefaultCollator can handle
        # NOTE: We use individual tensor keys instead of a BatchInfo namedtuple
        # because Paddle DataLoader's collate_fn will try to process the dict
        # and fails to handle namedtuple properly
        return {
            "x0": [lattices_t, frac_coords_t, atom_types_t],
            "batch_size": batch_size,
            "num_atoms": num_atoms_t,
            "batch_idx": batch_idx_t,
            "atom_types": atom_types_t,
        }