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
MiAD (Mirage Atom Diffusion) top-level model.
Thin nn.Layer wrapper that delegates training/sampling to CrystalGen or DiffCSP.
"""

import os
import types
import numpy as np

import paddle
import paddle.nn as nn

from ppmat.models.miad.cspnet_complete import CSPNet
from ppmat.models.miad.crystal_diffusion import init_diffusion


class DictToAttr:
    """Convert a dict to an object with attribute access."""

    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, DictToAttr(v))
            else:
                setattr(self, k, v)


class _DefaultLogger:
    """Minimal logger for standalone usage."""

    def add(self, *args, **kwargs):
        pass

    def fprint(self, *args, **kwargs):
        pass

    def root_fprint(self, *args, **kwargs):
        pass


class MiAD(nn.Layer):
    """Mirage Atom Diffusion for crystal structure generation.

    Thin integration layer that:
    - Holds the CSPNet denoiser (nn.Layer with parameters)
    - Holds the diffusion controller (CrystalGen or DiffCSP, plain Python)
    - Delegates forward (training) and sample (generation) to the controller

    Args:
        model_cfg: CSPNet encoder configuration dict.
        diffusion_cfg: Diffusion process configuration dict.
    """

    def __init__(self, model_cfg=None, diffusion_cfg=None, **kwargs):
        super().__init__()

        model_cfg = model_cfg or {}
        diffusion_cfg = diffusion_cfg or {}

        # Build CSPNet denoiser
        self.decoder = CSPNet(**model_cfg)

        # Build diffusion controller (convert dict to object with attribute access)
        if isinstance(diffusion_cfg, dict):
            diffusion_cfg = DictToAttr(diffusion_cfg)
        logger = _DefaultLogger()
        self.diffusion = init_diffusion(diffusion_cfg, logger)

    def forward(self, batch, **kwargs):
        """Training forward: delegates to diffusion.train_step().

        Args:
            batch: Dict with 'x0', 'batch_size', 'batch' keys.

        Returns:
            Dict with 'loss_dict' containing loss values.
        """
        batch = self.diffusion.train_step(
            batch=batch,
            model=self.decoder,
            mode='train',
        )
        loss = batch['loss']
        loss_dict = {
            'loss': loss,
        }
        return {'loss_dict': loss_dict}

    @paddle.no_grad()
    def sample(self, batch_data, num_inference_steps=None, **kwargs):
        """Generate crystal structures via reverse diffusion.

        Args:
            batch_data: Dict with batch information.
            num_inference_steps: Number of reverse steps (unused, kept for API).

        Returns:
            Dict with 'result' containing generated crystal data.
        """
        # Ensure batch has all required keys for sampling
        if 'structure_array' in batch_data:
            num_atoms_data = batch_data['structure_array'].get('num_atoms', None)
        else:
            num_atoms_data = batch_data.get('num_atoms', None)

        if num_atoms_data is not None:
            # Construct batch with all required keys for sampling
            if hasattr(num_atoms_data, 'numpy'):
                num_atoms_np = num_atoms_data.numpy().flatten()
            else:
                num_atoms_np = np.array(num_atoms_data).flatten()

            batch_size = len(num_atoms_np)
            num_atoms = paddle.to_tensor(num_atoms_np.astype('int64'))

            # Build batch_idx
            batch_idx_np = np.concatenate([np.full(int(n), i) for i, n in enumerate(num_atoms_np)])
            batch_idx = paddle.to_tensor(batch_idx_np.astype('int64'))

            # For sampling, atom_types will be generated
            atom_types = paddle.zeros([int(num_atoms_np.sum())], dtype='int64')

            batch_data = {
                'num_atoms': num_atoms,
                'batch_idx': batch_idx,
                'atom_types': atom_types,
                'batch_size': batch_size,
            }

        progress_printer = lambda t: None
        batch = self.diffusion.sampling_procedure(
            model=self.decoder,
            batch=batch_data,
            progress_printer=progress_printer,
        )

        # Extract results
        x0_pred = batch['x0_prediction']
        lattices = x0_pred[0]
        frac_coords = x0_pred[1]
        atom_types = x0_pred[2]

        result = []
        num_atoms = batch_data.get('num_atoms', None)
        batch_size = batch_data.get('batch_size', lattices.shape[0])

        start_idx = 0
        for i in range(batch_size):
            if num_atoms is not None:
                n = int(num_atoms[i])
            else:
                n = frac_coords.shape[0]
                if i > 0:
                    break
            result.append({
                'num_atoms': n,
                'atom_types': atom_types[start_idx:start_idx + n],
                'frac_coords': frac_coords[start_idx:start_idx + n],
                'lattice': lattices[i],
            })
            start_idx += n

        return {'result': result}
