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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MiAD Generation Pipeline.
Crystal structure generation via reverse diffusion with optional Mirage Infusion.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable

import numpy as np
import paddle

from ppmat.models.miad.miad import MiAD
from ppmat.datasets.miad import create_sampling_batch, CrystalBatch


@dataclass
class GenerationConfig:
    """Configuration for MiAD generation.

    Attributes:
        model: Model configuration dict (passed to CSPNet)
        diffusion: Diffusion configuration dict
        sampling: Sampling configuration
        output: Output configuration
        mirage: Mirage infusion configuration
    """
    model: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_dim': 128,
        'latent_dim': 256,
        'num_layers': 4,
        'max_atoms': 100,
        'act_fn': 'silu',
        'dis_emb': 'sin',
        'num_freqs': 10,
        'edge_style': 'fc',
        'cutoff': 6.0,
        'max_neighbors': 20,
        'ln': False,
        'ip': True,
        'smooth': False,
        'pred_type': False,
    })
    diffusion: Dict[str, Any] = field(default_factory=dict)
    sampling: Dict[str, Any] = field(default_factory=lambda: {
        'num_steps': 1000,
        'batch_size': 16,
    })
    output: Dict[str, Any] = field(default_factory=lambda: {
        'save_dir': './generated',
        'save_format': 'cif',  # 'cif' or 'numpy'
        'save_crystal': True,
    })
    mirage: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'max_atoms': 25,
        'remove_threshold': 0.5,
    })


class MiADGenerator:
    """MiAD Crystal Generator.

    Features:
        - Single-GPU generation
        - Mirage Infusion for variable atom counts
        - CIF output (optional, requires pymatgen)
        - EMA model support
    """

    def __init__(
        self,
        config: GenerationConfig,
        checkpoint_path: Optional[str] = None,
        device: str = 'gpu',
        use_ema: bool = True,
    ):
        """
        Args:
            config: GenerationConfig instance.
            checkpoint_path: Path to model checkpoint (.pdparams).
            device: Device to use ('gpu' or 'cpu').
            use_ema: Whether to use EMA version of model.
        """
        self.config = config
        self.device = device
        self.use_ema = use_ema

        # Build model
        self.model = MiAD(
            model_cfg=config.model,
            diffusion_cfg=config.diffusion,
        )

        # Load checkpoint
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, use_ema=use_ema)

        # Move to device
        if 'gpu' in device and paddle.device.is_compiled_with_cuda():
            self.model = self.model.cuda()

        self.model.eval()

    def load_checkpoint(self, path: str, use_ema: bool = True):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.
            use_ema: Whether to load EMA parameters.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if use_ema and 'ema' not in path:
            # Try to load EMA checkpoint
            ema_path = path.replace('.pdparams', '_ema.pdparams')
            if os.path.exists(ema_path):
                print(f"Loading EMA checkpoint: {ema_path}")
                state_dict = paddle.load(ema_path)
            else:
                print(f"EMA checkpoint not found, using base model: {path}")
                state_dict = paddle.load(path)
        else:
            state_dict = paddle.load(path)

        self.model.set_state_dict(state_dict)
        print(f"Checkpoint loaded: {path}")

    def generate(
        self,
        batch_size: int,
        num_atoms: Optional[paddle.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate crystal structures.

        Args:
            batch_size: Number of crystals to generate.
            num_atoms: Tensor of atom counts per crystal (optional).
            num_inference_steps: Number of diffusion steps (uses config default).
            progress_callback: Callback(t_current, t_total) for progress.

        Returns:
            List of generated crystal dicts with keys:
                - num_atoms: Number of atoms
                - atom_types: Atom type tensor
                - frac_coords: Fractional coordinates
                - lattice: Lattice matrix
        """
        if num_inference_steps is None:
            num_inference_steps = self.config.sampling.get('num_steps', 1000)

        # Create sampling batch
        mirage_max_atoms = None
        if self.config.mirage.get('enabled', False):
            mirage_max_atoms = self.config.mirage.get('max_atoms', 25)

        batch = create_sampling_batch(
            batch_size=batch_size,
            num_atoms=num_atoms,
            mirage_max_atoms=mirage_max_atoms,
            device=self.device,
        )

        # Move batch to device
        if 'gpu' in self.device and paddle.device.is_compiled_with_cuda():
            batch = self._move_batch_to_device(batch)

        # Progress printer
        def progress_printer(t_value):
            if progress_callback is not None:
                progress_callback(int(t_value), num_inference_steps)

        # Sampling
        @paddle.no_grad()
        def sampling():
            return self.model.sample(
                batch_data=batch,
                num_inference_steps=num_inference_steps,
                progress_printer=progress_printer,
            )

        result = sampling()

        # Post-process: remove mirage atoms
        if self.config.mirage.get('enabled', False):
            result = self._remove_mirage_atoms(result)

        return result.get('result', [result])

    def generate_with_distribution(
        self,
        num_crystals: int,
        num_atoms_distribution: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate crystals with atom count distribution.

        Args:
            num_crystals: Total number of crystals to generate.
            num_atoms_distribution: Distribution of atom counts (optional).
            **kwargs: Additional arguments for generate().

        Returns:
            List of generated crystal dicts.
        """
        batch_size = self.config.sampling.get('batch_size', 16)
        all_results = []

        num_batches = (num_crystals + batch_size - 1) // batch_size

        for i in range(num_batches):
            current_batch_size = min(batch_size, num_crystals - i * batch_size)

            # Sample from distribution
            if num_atoms_distribution is not None:
                num_atoms = num_atoms_distribution[
                    i * batch_size:(i + 1) * batch_size
                ][:current_batch_size]
            else:
                num_atoms = None

            results = self.generate(
                batch_size=current_batch_size,
                num_atoms=num_atoms,
                **kwargs,
            )
            all_results.extend(results)

            print(f"Generated {len(all_results)}/{num_crystals} crystals", flush=True)

        return all_results

    def _remove_mirage_atoms(self, results: List[Dict]) -> List[Dict]:
        """Remove mirage atoms (type 0) from generated crystals.

        Args:
            results: List of crystal dicts.

        Returns:
            Filtered list without mirage atoms.
        """
        threshold = self.config.mirage.get('remove_threshold', 0.5)
        filtered = []

        for result in results:
            atom_types = result['atom_types']
            frac_coords = result['frac_coords']
            lattice = result['lattice']

            # Create mask for non-mirage atoms
            if hasattr(atom_types, 'numpy'):
                atom_types_np = atom_types.numpy()
            else:
                atom_types_np = np.array(atom_types)

            # For type prediction models, types are probabilities
            if atom_types_np.ndim > 1:
                # Get probabilities
                probs = atom_types_np
                mirage_prob = probs[:, 0]
                mask = mirage_prob < threshold
            else:
                # Discrete types: 0 = mirage
                mask = atom_types_np != 0

            # Apply mask
            num_real_atoms = int(mask.sum())

            if num_real_atoms == 0:
                # Skip empty crystals
                continue

            # Handle different tensor formats
            if hasattr(frac_coords, 'numpy'):
                frac_coords_np = frac_coords.numpy()
            else:
                frac_coords_np = np.array(frac_coords)

            if hasattr(lattice, 'numpy'):
                lattice_np = lattice.numpy()
            else:
                lattice_np = np.array(lattice)

            filtered_result = {
                'num_atoms': num_real_atoms,
                'atom_types': atom_types[mask] if hasattr(atom_types, '__getitem__') else paddle.to_tensor(atom_types_np[mask]),
                'frac_coords': frac_coords[mask] if hasattr(frac_coords, '__getitem__') else paddle.to_tensor(frac_coords_np[mask]),
                'lattice': lattice,
            }
            filtered.append(filtered_result)

        return filtered

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        if 'x0' in batch and batch['x0'] is not None:
            batch['x0'] = [
                x.cuda() if hasattr(x, 'cuda') else x
                for x in batch['x0']
            ]
        if 'batch' in batch and hasattr(batch['batch'], 'to'):
            batch['batch'] = batch['batch'].to(self.device)
        return batch

    def save_results(
        self,
        results: List[Dict[str, Any]],
        save_dir: str,
        prefix: str = 'crystal',
        save_format: str = 'numpy',
    ):
        """Save generated results.

        Args:
            results: List of crystal dicts.
            save_dir: Directory to save results.
            prefix: Filename prefix.
            save_format: 'numpy' or 'cif'.
        """
        os.makedirs(save_dir, exist_ok=True)

        if save_format == 'numpy':
            self._save_numpy(results, save_dir, prefix)
        elif save_format == 'cif':
            self._save_cif(results, save_dir, prefix)
        else:
            raise ValueError(f"Unknown save_format: {save_format}")

    def _save_numpy(
        self,
        results: List[Dict[str, Any]],
        save_dir: str,
        prefix: str,
    ):
        """Save results as numpy files."""
        for i, result in enumerate(results):
            path = os.path.join(save_dir, f'{prefix}_{i:06d}.npz')

            # Convert tensors to numpy
            data = {}
            for key, value in result.items():
                if hasattr(value, 'numpy'):
                    data[key] = value.numpy()
                elif hasattr(value, 'cpu'):
                    data[key] = value.cpu().numpy()
                else:
                    data[key] = np.array(value)

            np.savez_compressed(path, **data)

        print(f"Saved {len(results)} results to {save_dir}")

    def _save_cif(
        self,
        results: List[Dict[str, Any]],
        save_dir: str,
        prefix: str,
    ):
        """Save results as CIF files (requires pymatgen)."""
        try:
            from pymatgen.core.structure import Structure
            from pymatgen.core.lattice import Lattice
        except ImportError:
            print("Warning: pymatgen not installed. Cannot save CIF files.")
            print("Falling back to numpy format.")
            self._save_numpy(results, save_dir, prefix)
            return

        for i, result in enumerate(results):
            path = os.path.join(save_dir, f'{prefix}_{i:06d}.cif')

            # Get data
            lattice = result['lattice']
            frac_coords = result['frac_coords']
            atom_types = result['atom_types']

            # Convert to numpy
            if hasattr(lattice, 'numpy'):
                lattice = lattice.numpy()
            if hasattr(frac_coords, 'numpy'):
                frac_coords = frac_coords.numpy()
            if hasattr(atom_types, 'numpy'):
                atom_types = atom_types.numpy()

            # Handle one-hot encoding
            if atom_types.ndim > 1:
                atom_types = atom_types.argmax(axis=-1) + 1  # 1-indexed

            # Convert to atomic numbers (Z)
            # Element numbers: 1=H, 6=C, 8=O, etc.
            # Map atom type index to atomic number
            atom_z = self._type_to_atomic_number(atom_types)

            # Create Structure
            try:
                structure = Structure(
                    lattice=Lattice(lattice),
                    species=atom_z,
                    coords=frac_coords,
                    coords_are_cartesian=False,
                )
                structure.to(filename=path)
            except Exception as e:
                print(f"Error saving CIF {path}: {e}")
                # Fallback to numpy
                fallback_path = path.replace('.cif', '.npz')
                np.savez_compressed(
                    fallback_path,
                    lattice=lattice,
                    frac_coords=frac_coords,
                    atom_types=atom_types,
                )

        print(f"Saved {len(results)} CIF files to {save_dir}")

    @staticmethod
    def _type_to_atomic_number(atom_types):
        """Convert atom type indices to atomic numbers.

        MiAD uses 1-indexed atom types. We need to map to atomic numbers.
        """
        # Common mapping: atom_type -> atomic_number
        # This is a simplified mapping; actual mapping depends on training data
        type_to_z = {
            1: 1,   # H
            2: 6,   # C
            3: 7,   # N
            4: 8,   # O
            5: 9,   # F
            6: 11,  # Na
            7: 12,  # Mg
            8: 13,  # Al
            9: 14,  # Si
            10: 15, # P
            11: 16, # S
            12: 17, # Cl
            13: 19, # K
            14: 20, # Ca
            15: 21, # Sc
            16: 22, # Ti
            17: 23, # V
            18: 24, # Cr
            19: 25, # Mn
            20: 26, # Fe
            21: 27, # Co
            22: 28, # Ni
            23: 29, # Cu
            24: 30, # Zn
            25: 31, # Ga
            26: 32, # Ge
            27: 33, # As
            28: 34, # Se
            29: 35, # Br
            30: 39, # Y
            31: 40, # Zr
            32: 41, # Nb
            33: 42, # Mo
            34: 45, # Rh
            35: 46, # Pd
            36: 47, # Ag
            37: 48, # Cd
            38: 49, # In
            39: 50, # Sn
            40: 51, # Sb
            41: 52, # Te
            42: 53, # I
            43: 56, # Ba
            44: 57, # La
            45: 72, # Hf
            46: 73, # Ta
            47: 74, # W
            48: 75, # Re
            49: 76, # Os
            50: 77, # Ir
            51: 78, # Pt
            52: 79, # Au
            53: 80, # Hg
            54: 81, # Tl
            55: 82, # Pb
            56: 83, # Bi
            57: 84, # Po
            58: 88, # Ra
            59: 89, # Ac
            60: 104, # Rf
            61: 105, # Db
            62: 106, # Sg
            63: 107, # Bh
            64: 108, # Hs
            65: 109, # Mt
            66: 110, # Ds
            67: 111, # Rg
            68: 112, # Cn
        }

        atomic_numbers = np.array([
            type_to_z.get(int(t), int(t)) for t in atom_types.flatten()
        ])
        return atomic_numbers


def create_generator_from_checkpoint(
    checkpoint_path: str,
    config: Optional[GenerationConfig] = None,
    device: str = 'gpu',
) -> MiADGenerator:
    """Create generator from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        config: GenerationConfig (uses defaults if None).
        device: Device to use.

    Returns:
        MiADGenerator instance.
    """
    if config is None:
        config = GenerationConfig()

    generator = MiADGenerator(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    return generator


# Utility function for command-line usage
def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate crystals with MiAD')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num_crystals', type=int, default=100,
                        help='Number of crystals to generate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for generation')
    parser.add_argument('--num_steps', type=int, default=1000,
                        help='Number of diffusion steps')
    parser.add_argument('--output_dir', type=str, default='./generated',
                        help='Output directory')
    parser.add_argument('--save_format', type=str, default='cif',
                        choices=['cif', 'numpy'],
                        help='Output format')
    parser.add_argument('--enable_mirage', action='store_true',
                        help='Enable Mirage Infusion')
    parser.add_argument('--mirage_max_atoms', type=int, default=25,
                        help='Max atoms for Mirage')
    parser.add_argument('--device', type=str, default='gpu',
                        help='Device')
    args = parser.parse_args()

    # Build config
    config = GenerationConfig(
        sampling={
            'batch_size': args.batch_size,
            'num_steps': args.num_steps,
        },
        output={
            'save_dir': args.output_dir,
            'save_format': args.save_format,
        },
        mirage={
            'enabled': args.enable_mirage,
            'max_atoms': args.mirage_max_atoms,
        },
    )

    # Create generator
    generator = create_generator_from_checkpoint(
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device,
    )

    # Generate
    print(f"Generating {args.num_crystals} crystals...", flush=True)
    start_time = time.time()

    results = generator.generate_with_distribution(
        num_crystals=args.num_crystals,
    )

    elapsed = time.time() - start_time
    print(f"Generation completed in {elapsed:.2f}s", flush=True)
    print(f"Average time per crystal: {elapsed / len(results):.3f}s", flush=True)

    # Save results
    generator.save_results(
        results=results,
        save_dir=args.output_dir,
        save_format=args.save_format,
    )

    print(f"Results saved to {args.output_dir}", flush=True)


if __name__ == '__main__':
    main()