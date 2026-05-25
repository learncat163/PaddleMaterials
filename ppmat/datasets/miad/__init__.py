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
MiAD-specific dataset and collate functions.
Integrates with PaddleMaterials MP20Dataset for MiAD training and generation.
"""

import os
import numpy as np
import paddle
from typing import List, Dict, Any, Optional

from ppmat.datasets.mp20_dataset import MP20Dataset
from ppmat.datasets.custom_data_type import ConcatData


def _extract_concat_data(data):
    """Extract numpy array from ConcatData or passthrough."""
    if data is None:
        return None
    if isinstance(data, ConcatData):
        return np.array(data.data)
    return data


class MiADCollator:
    """Collate function for MiAD model.

    Converts MP20Dataset batch to CrystalGen format:
        Input:  List of samples with structure_array dict
        Output: Dict with x0=[lattice, frac_coords, atom_types], batch info
    """

    def __init__(
        self,
        mirage_max_atoms: Optional[int] = None,
        use_mirage: bool = False,
    ):
        """
        Args:
            mirage_max_atoms: If set, pad all crystals to this atom count
                with mirage atoms (type 0). Used for MiAD dynamic generation.
            use_mirage: Whether to enable mirage atom padding.
        """
        self.mirage_max_atoms = mirage_max_atoms
        self.use_mirage = use_mirage

        if mirage_max_atoms is not None:
            self.N_m = mirage_max_atoms
        else:
            self.N_m = self._parse_mirage_from_env()

    def _parse_mirage_from_env(self) -> Optional[int]:
        """Parse mirage max atoms from MODIFICATIONS_FIELD env var."""
        modifications = os.environ.get('MODIFICATIONS_FIELD', '')
        if 'miad:add_mirage_atoms_upto' in modifications:
            return int(
                modifications.split('miad:add_mirage_atoms_upto')[1].split('+')[0]
            )
        return None

    def __call__(self, batch_list: List[Dict]) -> Dict[str, Any]:
        """Collate a list of samples into MiAD/CrystalGen format.

        Args:
            batch_list: List of samples from MP20Dataset.__getitem__

        Returns:
            Dict with keys:
                - x0: [lattice, frac_coords, atom_types] as paddle tensors
                - batch_size: Number of crystals in batch
                - num_atoms: Total number of atoms (or N_m * batch_size if mirage)
                - batch_idx: Node-to-graph mapping
                - atom_types: Atom type tensor
                - batch: CrystalBatch object for CrystalGen compatibility
        """
        if not batch_list:
            return {}

        batch_size = len(batch_list)

        all_frac_coords = []
        all_atom_types = []
        all_num_atoms = []
        all_lattices = []
        batch_idx_list = []

        for i, sample_i in enumerate(batch_list):
            if not isinstance(sample_i, dict):
                continue

            # Extract structure_array
            if "structure_array" in sample_i:
                sa = sample_i["structure_array"]
            elif "frac_coords" in sample_i:
                sa = sample_i
            else:
                continue

            frac_coords = _extract_concat_data(sa.get("frac_coords"))
            atom_types = _extract_concat_data(sa.get("atom_types"))
            lattice = _extract_concat_data(sa.get("lattice"))
            num_atoms = _extract_concat_data(sa.get("num_atoms"))

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
            if atom_types.ndim > 1:
                atom_types = atom_types.flatten()

            # Lattice shape: [3, 3] per sample, need [B, 3, 3]
            if lattice.ndim == 2:
                lattice = lattice.reshape(1, 3, 3)
            elif lattice.ndim == 1:
                lattice = lattice.reshape(1, 3, 3)

            # Apply mirage atom padding if enabled
            if self.N_m is not None and self.use_mirage and n_atoms < self.N_m:
                frac_coords, atom_types = self._pad_mirage(
                    frac_coords, atom_types, n_atoms, self.N_m
                )
                n_atoms = self.N_m

            all_frac_coords.append(frac_coords)
            all_atom_types.append(atom_types)
            all_lattices.append(lattice[0] if lattice.shape[0] == 1 else lattice)
            all_num_atoms.append(n_atoms)
            batch_idx_list.extend([i] * n_atoms)

        if not all_frac_coords:
            return {}

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

        # Create CrystalBatch for compatibility
        crystal_batch = CrystalBatch(
            num_atoms=num_atoms_t,
            atom_types=atom_types_t,
            batch=batch_idx_t,
            frac_coords=frac_coords_t,
            lattice=lattices_t,
        )

        return {
            "x0": [lattices_t, frac_coords_t, atom_types_t],
            "batch_size": batch_size,
            "num_atoms": paddle.to_tensor(int(sum(all_num_atoms)), dtype='int64'),
            "batch_idx": batch_idx_t,
            "atom_types": atom_types_t,
            "batch": crystal_batch,
        }

    @staticmethod
    def _pad_mirage(frac_coords, atom_types, n_atoms, N_m):
        """Pad crystal with mirage atoms (type 0) to N_m atoms."""
        pad_n = N_m - n_atoms
        if pad_n > 0:
            pad_fc = np.random.rand(pad_n, 3).astype("float32")
            pad_at = np.zeros(pad_n, dtype="int64")
            frac_coords = np.concatenate([frac_coords, pad_fc], axis=0)
            atom_types = np.concatenate([atom_types, pad_at], axis=0)
        return frac_coords, atom_types


class CrystalBatch:
    """Minimal batch object mimicking torch_geometric Batch interface.

    CrystalGen and TimeDistribution access batch.num_atoms and batch.batch.
    """

    def __init__(
        self,
        num_atoms: paddle.Tensor,
        atom_types: paddle.Tensor,
        batch: paddle.Tensor,
        frac_coords: Optional[paddle.Tensor] = None,
        lattice: Optional[paddle.Tensor] = None,
    ):
        self.num_atoms = num_atoms
        self.atom_types = atom_types
        self.batch = batch
        self.frac_coords = frac_coords
        self.lattice = lattice

    def to(self, device: str):
        """Move all tensors to device."""
        self.num_atoms = self.num_atoms.to(device)
        self.atom_types = self.atom_types.to(device)
        self.batch = self.batch.to(device)
        if self.frac_coords is not None:
            self.frac_coords = self.frac_coords.to(device)
        if self.lattice is not None:
            self.lattice = self.lattice.to(device)
        return self


def create_miad_dataloader(
    dataset: MP20Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    mirage_max_atoms: Optional[int] = None,
    use_mirage: bool = False,
    **kwargs,
) -> paddle.io.DataLoader:
    """Create a DataLoader with MiAD collate function.

    Args:
        dataset: MP20Dataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of worker processes.
        mirage_max_atoms: Max atoms for mirage padding (optional).
        use_mirage: Enable mirage padding.
        **kwargs: Additional arguments for DataLoader.

    Returns:
        DataLoader with MiADCollator.
    """
    collate_fn = MiADCollator(
        mirage_max_atoms=mirage_max_atoms,
        use_mirage=use_mirage,
    )

    return paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs,
    )


def create_sampling_batch(
    batch_size: int,
    num_atoms: Optional[paddle.Tensor] = None,
    mirage_max_atoms: Optional[int] = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Create a batch for generation/sampling.

    Args:
        batch_size: Number of crystals to generate.
        num_atoms: Tensor of atom counts per crystal. If None, uses distribution.
        mirage_max_atoms: Fixed atom count for all crystals (enables mirage).
        device: Target device.

    Returns:
        Batch dict for CrystalGen.sampling_procedure.
    """
    if mirage_max_atoms is not None:
        # Mirage mode: fixed atom count
        N_m = mirage_max_atoms
        num_atoms = paddle.full([batch_size], N_m, dtype='int64')
        total_atoms = batch_size * N_m
    elif num_atoms is not None:
        total_atoms = int(num_atoms.sum())
    else:
        # Default: use simple distribution
        num_atoms = paddle.randint(5, 30, [batch_size], dtype='int64')
        total_atoms = int(num_atoms.sum())

    node2graph = paddle.repeat_interleave(
        paddle.arange(batch_size, dtype='int64'), num_atoms
    )

    crystal_batch = CrystalBatch(
        num_atoms=num_atoms,
        atom_types=paddle.zeros([total_atoms], dtype='int64'),
        batch=node2graph,
        frac_coords=paddle.zeros([total_atoms, 3], dtype='float32'),
        lattice=paddle.zeros([batch_size, 3, 3], dtype='float32'),
    )

    return {
        'x0': [None, None, None],
        'batch_size': batch_size,
        'num_atoms': paddle.to_tensor(total_atoms, dtype='int64'),
        'device': device,
        'batch': crystal_batch,
    }