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
Batch utilities for MiAD training and generation.
Converts collated dataset outputs into the format expected by CrystalGen.
"""

import os
import numpy as np
import paddle


class MiADBatchConverter:
    """Converts collated MP20Dataset batches to CrystalGen format.

    Handles:
    - Converting structure_array to x0 = [lattice, frac_coords, atom_types]
    - Building node2graph mapping
    - Mirage Infusion: padding atoms to fixed count for generation training
    """

    def __init__(self, mirage_max_atoms=None):
        """
        Args:
            mirage_max_atoms: If set, pad all crystals to this atom count
                with mirage atoms (type 0). Enables MiAD's dynamic atom
                count generation. Read from MODIFICATIONS_FIELD env var
                if not provided.
        """
        if mirage_max_atoms is not None:
            self.mirage_max_atoms = mirage_max_atoms
        else:
            self.mirage_max_atoms = self._get_mirage_max_atoms()

    @staticmethod
    def _get_mirage_max_atoms():
        """Parse mirage max atoms from MODIFICATIONS_FIELD env var."""
        modifications = os.environ.get('MODIFICATIONS_FIELD', '')
        if 'miad:add_mirage_atoms_upto' in modifications:
            return int(
                modifications.split('miad:add_mirage_atoms_upto')[1].split('+')[0]
            )
        return None

    def training_batch(self, collated_batch, device='cpu'):
        """Convert collated dataset output to CrystalGen training format.

        Args:
            collated_batch: Dict from DefaultCollator containing
                structure_array with keys: lattice, frac_coords,
                atom_types, num_atoms.
            device: Target device string.

        Returns:
            batch: Dict with 'x0', 'batch_size', 'num_atoms', 'device',
                'batch' (crystal_batch with num_atoms and batch index).
        """
        sa = collated_batch['structure_array']
        num_atoms = paddle.to_tensor(sa['num_atoms'], dtype='int64')
        lattice = paddle.to_tensor(sa['lattice'], dtype='float32')
        frac_coords = paddle.to_tensor(sa['frac_coords'], dtype='float32')
        atom_types = paddle.to_tensor(sa['atom_types'], dtype='int64')

        batch_size = num_atoms.shape[0]
        total_atoms = frac_coords.shape[0]

        # Build node2graph: maps each atom to its crystal index
        node2graph = paddle.repeat_interleave(
            paddle.arange(batch_size, dtype='int64'), num_atoms
        )

        # Mirage Infusion: pad to fixed atom count
        if self.mirage_max_atoms is not None:
            N_m = self.mirage_max_atoms
            frac_coords, atom_types, num_atoms, node2graph = (
                self._apply_mirage(
                    frac_coords, atom_types, num_atoms, node2graph, batch_size, N_m
                )
            )
            total_atoms = frac_coords.shape[0]

        # Build crystal_batch object expected by CrystalGen
        crystal_batch = _CrystalBatch(
            num_atoms=num_atoms,
            atom_types=atom_types,
            batch=node2graph,
            frac_coords=frac_coords,
            lattice=lattice,
        )

        x0 = [lattice, frac_coords, atom_types]

        return {
            'x0': x0,
            'batch_size': batch_size,
            'num_atoms': total_atoms,
            'device': device,
            'batch': crystal_batch,
        }

    def sampling_batch(self, batch_size, num_atoms, device='cpu'):
        """Create a batch for generation/sampling.

        Args:
            batch_size: Number of crystals to generate.
            num_atoms: Tensor of atom counts per crystal.
            device: Target device.

        Returns:
            batch: Dict for CrystalGen.sampling_procedure.
        """
        total_atoms = int(num_atoms.sum())

        # Mirage: override num_atoms to fixed count
        if self.mirage_max_atoms is not None:
            N_m = self.mirage_max_atoms
            num_atoms = paddle.full([batch_size], N_m, dtype='int64')
            total_atoms = batch_size * N_m

        node2graph = paddle.repeat_interleave(
            paddle.arange(batch_size, dtype='int64'), num_atoms
        )

        crystal_batch = _CrystalBatch(
            num_atoms=num_atoms,
            atom_types=paddle.zeros([total_atoms], dtype='int64'),
            batch=node2graph,
            frac_coords=paddle.zeros([total_atoms, 3], dtype='float32'),
            lattice=paddle.zeros([batch_size, 3, 3], dtype='float32'),
        )

        return {
            'x0': [None, None, None],
            'batch_size': batch_size,
            'num_atoms': total_atoms,
            'device': device,
            'batch': crystal_batch,
        }

    @staticmethod
    def _apply_mirage(frac_coords, atom_types, num_atoms, node2graph,
                      batch_size, N_m):
        """Pad each crystal with mirage atoms (type 0) to N_m atoms.

        Args:
            frac_coords: (total_atoms, 3)
            atom_types: (total_atoms,)
            num_atoms: (batch_size,)
            node2graph: (total_atoms,)
            batch_size: int
            N_m: target atom count per crystal

        Returns:
            Padded frac_coords, atom_types, num_atoms, node2graph.
        """
        all_frac = []
        all_types = []
        offset = 0
        for i in range(batch_size):
            n = int(num_atoms[i])
            fc = frac_coords[offset:offset + n]
            at = atom_types[offset:offset + n]

            # Pad with random fractional coords and mirage type 0
            pad_n = N_m - n
            if pad_n > 0:
                pad_fc = paddle.rand([pad_n, 3], dtype='float32')
                pad_at = paddle.zeros([pad_n], dtype='int64')
                fc = paddle.concat([fc, pad_fc], axis=0)
                at = paddle.concat([at, pad_at], axis=0)

            all_frac.append(fc)
            all_types.append(at)
            offset += n

        frac_coords = paddle.concat(all_frac, axis=0)
        atom_types = paddle.concat(all_types, axis=0)
        num_atoms = paddle.full([batch_size], N_m, dtype='int64')
        node2graph = paddle.repeat_interleave(
            paddle.arange(batch_size, dtype='int64'), num_atoms
        )
        return frac_coords, atom_types, num_atoms, node2graph


class _CrystalBatch:
    """Minimal batch object mimicking torch_geometric Batch interface.

    CrystalGen and TimeDistribution access batch.num_atoms and batch.batch.
    """

    def __init__(self, num_atoms, atom_types, batch, frac_coords=None,
                 lattice=None):
        self.num_atoms = num_atoms
        self.atom_types = atom_types
        self.batch = batch
        self.frac_coords = frac_coords
        self.lattice = lattice
