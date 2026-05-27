# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
import paddle.nn as nn

from ppmat.models.miad.cspnet_complete import CSPNet
from ppmat.models.miad.crystal_diffusion import init_diffusion, _parse_num_atoms_to_per_crystal


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
    """Mirage Atom Diffusion model."""

    def __init__(self, model_cfg=None, diffusion_cfg=None, **kwargs):
        super().__init__()

        model_cfg = model_cfg or {}
        diffusion_cfg = diffusion_cfg or {}

        self.decoder = CSPNet(**model_cfg)
        if isinstance(diffusion_cfg, dict):
            diffusion_cfg = DictToAttr(diffusion_cfg)
        logger = _DefaultLogger()
        self.diffusion = init_diffusion(diffusion_cfg, logger)

    def set_state_dict(self, state_dict, use_structured_name=True):
        """
        Load checkpoint with automatic legacy format detection.

        MiAD wraps CSPNet as self.decoder, so all parameter keys in a native
        Paddle checkpoint are prefixed with "decoder." (e.g. decoder.csp_layer_0.edge_mlp.0.weight).

        Legacy PyTorch checkpoints have bare keys (e.g. csp_layer_0.edge_mlp.0.weight)
        and Linear weights stored in (out_features, in_features) format.
        This method auto-detects and converts them: adds decoder. prefix and transposes
        all 2D weight tensors to Paddle's (in_features, out_features) format.
        """
        model_keys = set(self.state_dict().keys())
        state_keys = set(state_dict.keys())

        if len(state_keys) == 0:
            return super().set_state_dict(state_dict, use_structured_name)

        has_decoder_prefix = any(k.startswith('decoder.') for k in state_keys)
        keys_native = state_keys == model_keys or state_keys.issubset(model_keys)

        if keys_native and has_decoder_prefix:
            return super().set_state_dict(state_dict, use_structured_name)

        is_legacy = not has_decoder_prefix
        processed = {}
        for k, v in state_dict.items():
            new_key = f"decoder.{k}" if is_legacy else k
            if is_legacy and new_key.endswith('.weight') and len(v.shape) == 2:
                processed[new_key] = v.T
            else:
                processed[new_key] = v

        loaded = super().set_state_dict(processed, use_structured_name)
        model_in_keys = set(model_keys) - set(processed.keys())
        extra_in_ckpt = set(processed.keys()) - set(model_keys)
        if model_in_keys:
            print(f"  [MiAD] {len(model_in_keys)} keys in model not in checkpoint (likely buffers): {sorted(model_in_keys)[:3]}...")
        if extra_in_ckpt:
            print(f"  [MiAD] {len(extra_in_ckpt)} keys in checkpoint not in model (skipped): {sorted(extra_in_ckpt)[:3]}...")
        if is_legacy:
            print(f"  [MiAD] Converted legacy checkpoint: added decoder. prefix + transposed 2D weights")
        return loaded

    def forward(self, batch, **kwargs):
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
        if 'structure_array' in batch_data:
            num_atoms_data = batch_data['structure_array'].get('num_atoms', None)
        else:
            num_atoms_data = batch_data.get('num_atoms', None)

        if 'batch_idx' in batch_data:
            pass
        elif num_atoms_data is not None:
            parsed = _parse_num_atoms_to_per_crystal(num_atoms_data)
            if parsed is not None:
                num_atoms, num_atoms_np = parsed
                batch_size = len(num_atoms_np)
                batch_idx_np = np.concatenate([np.full(int(n), i) for i, n in enumerate(num_atoms_np)])
                batch_idx = paddle.to_tensor(batch_idx_np.astype('int64'))
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
