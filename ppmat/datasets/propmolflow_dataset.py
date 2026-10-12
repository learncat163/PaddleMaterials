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

"""PropMolFlow molecule datasets.

Compared with ``QM9Dataset`` (15 property columns), the released rQM9_v0 data
used here provides the same 15 columns (A, B, C, mu, alpha, homo, lumo, gap,
r2, zpve, U0, U, H, G, Cv) plus four per-atom columns (u0_atom, u298_atom,
h298_atom, g298_atom). It also adds the covalent bond and formal-charge fields
the generation model supervises, which ``QM9Dataset`` does not provide. The
shared download and caching infrastructure is reused unchanged.
"""

from __future__ import annotations

import os.path as osp
from typing import Dict

import numpy as np
from paddle.io import Dataset

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.utils import download
from ppmat.utils import logger

PROPERTY_MAP = {
    "A": 0,
    "B": 1,
    "C": 2,
    "mu": 3,
    "alpha": 4,
    "homo": 5,
    "lumo": 6,
    "gap": 7,
    "r2": 8,
    "zpve": 9,
    "U0": 10,
    "U": 11,
    "H": 12,
    "G": 13,
    "Cv": 14,
    "u0_atom": 15,
    "u298_atom": 16,
    "h298_atom": 17,
    "g298_atom": 18,
}

N_CHARGES = 6
N_BOND_TYPES = 5
CHARGE_OFFSET = 2


def _default_normalization_path(path: str) -> str:
    """Return the shared train-split property normalization file path."""
    return osp.join(osp.dirname(path), "train_data_property_normalization.npz")


class _PropMolFlowSplitDataset(Dataset):
    """Shared download and split resolution of the PropMolFlow npz package."""

    name = "propmolflow_qm9"
    # Local test URL, replace with the online address before submitting.
    url = "http://127.0.0.1/models/propmolflow_qm9.zip"
    md5 = "51a812b6992dc6996ec7d012a54a7857"

    def _resolve_split_path(self, path: str) -> str:
        """Resolve a split path, downloading the package when it is missing."""
        if osp.exists(path):
            return path
        logger.message("The dataset is not found. Will download it now.")
        root_path = download.get_datasets_path_from_url(self.url, self.md5)
        return osp.join(root_path, self.name, osp.basename(path))


class PropMolFlowDataset(_PropMolFlowSplitDataset):
    """PropMolFlow molecule dataset backed by preprocessed npz splits.

    Every sample provides the endpoint features (positions, one-hot atom types,
    one-hot charges and upper-triangle one-hot bond types) and, when
    conditioning is enabled, the normalized property value.
    """

    def __init__(
        self,
        path: str = "./data/propmolflow_qm9/train_a_data_processed.npz",
        atom_map: list = None,
        conditioning_cfg: Dict = None,
        normalization_path: str = None,
    ):
        super().__init__()

        if atom_map is None:
            raise ValueError("atom_map must be provided")
        if conditioning_cfg is None:
            raise ValueError("conditioning_cfg must be provided")

        self.atom_map = list(atom_map)
        self.conditioning = conditioning_cfg

        path = self._resolve_split_path(path)
        self.path = path

        if normalization_path is None:
            normalization_path = _default_normalization_path(path)
        self.normalization_path = normalization_path

        with np.load(path) as data:
            self.positions = data["positions"]
            self.atom_types = data["atom_types"]
            self.atom_charges = data["atom_charges"]
            self.bond_types = data["bond_types"]
            self.bond_idxs = data["bond_idxs"]
            self.node_idx_array = data["node_idx_array"]
            self.edge_idx_array = data["edge_idx_array"]
            self.properties = data["properties"] if "properties" in data else None

        if self.conditioning.get("enabled", False):
            if self.properties is None:
                raise ValueError("Properties not found in the processed data")
            property_name = self.conditioning.get("property")
            if property_name not in PROPERTY_MAP:
                raise ValueError(
                    f"Target property {property_name} not found in PROPERTY_MAP"
                )
            self.property_idx = PROPERTY_MAP[property_name]
            if self.conditioning.get("normalize", False):
                with np.load(self.normalization_path) as norm:
                    self.property_mean = float(norm["mean"][self.property_idx])
                    self.property_std = float(norm["std"][self.property_idx])

    def __len__(self):
        return self.node_idx_array.shape[0]

    def __getitem__(self, idx):
        node_start, node_end = self.node_idx_array[idx]
        edge_start, edge_end = self.edge_idx_array[idx]

        positions = self.positions[node_start:node_end].astype("float32")
        positions = positions - positions.mean(axis=0, keepdims=True)
        atom_types_idx = self.atom_types[node_start:node_end].argmax(-1)
        atom_types = _one_hot(atom_types_idx, len(self.atom_map))
        atom_charges = _one_hot(
            self.atom_charges[node_start:node_end] + CHARGE_OFFSET, N_CHARGES
        )

        n_atoms = int(node_end - node_start)

        adjacency = np.zeros((n_atoms, n_atoms), dtype=np.int64)
        bond_idxs = self.bond_idxs[edge_start:edge_end]
        adjacency[bond_idxs[:, 0], bond_idxs[:, 1]] = self.bond_types[
            edge_start:edge_end
        ]
        upper_i, upper_j = np.triu_indices(n_atoms, k=1)
        bond_types = _one_hot(adjacency[upper_i, upper_j], N_BOND_TYPES)

        structure_array = {
            "num_atoms": n_atoms,
            "positions": positions,
            "atom_types": atom_types,
            "atom_charges": atom_charges,
            "bond_types": bond_types,
        }

        sample = {"structure_array": structure_array}
        if self.conditioning.get("enabled", False):
            property_value = float(self.properties[idx, self.property_idx])
            if self.conditioning.get("normalize", False):
                property_value = (property_value - self.property_mean) / (
                    self.property_std
                )
            sample["prop"] = property_value
        return sample


class PropMolFlowSamplingDataset(_PropMolFlowSplitDataset):
    """Sampling inputs for PropMolFlow: number of atoms and property values.

    ``n_atoms`` and ``property_value`` are orthogonal pins: an explicit value
    fixes that side for every sample, while ``mode`` selects the training
    distribution of the unpinned side. ``in_distribution`` draws atom counts
    from the split atom-count distribution and property values from the
    per-atom-count property histograms. ``out_of_distribution`` requires an
    explicit ``property_value`` and draws atom counts from the molecules whose
    property lies in the top quantile range.
    """

    def __init__(
        self,
        path: str = "./data/propmolflow_qm9/train_a_data_processed.npz",
        property_name: str = None,
        mode: str = None,
        total_num: int = None,
        seed: int = None,
        normalization_path: str = None,
        property_value: float = None,
        n_atoms: int = None,
        num_bins: int = 1000,
    ):
        super().__init__()

        if property_name is None:
            raise ValueError("property_name must be provided")
        if mode is None:
            raise ValueError("mode must be provided")
        if total_num is None:
            raise ValueError("total_num must be provided")
        if seed is None:
            raise ValueError("seed must be provided")
        if mode not in ["in_distribution", "out_of_distribution"]:
            raise ValueError(
                "mode must be either 'in_distribution' or 'out_of_distribution', "
                f"got {mode}"
            )
        if property_name not in PROPERTY_MAP:
            raise ValueError(
                f"Target property {property_name} not found in PROPERTY_MAP"
            )
        if mode == "out_of_distribution" and property_value is None:
            raise ValueError(
                "property_value must be provided for out_of_distribution sampling"
            )

        path = self._resolve_split_path(path)
        if normalization_path is None:
            normalization_path = _default_normalization_path(path)

        self.property_name = property_name
        self.mode = mode
        self.total_num = total_num
        self.property_idx = PROPERTY_MAP[property_name]

        with np.load(normalization_path) as norm:
            self.property_mean = float(norm["mean"][self.property_idx])
            self.property_std = float(norm["std"][self.property_idx])

        with np.load(path) as data:
            node_idx_array = data["node_idx_array"]
            num_atoms_all = (node_idx_array[:, 1] - node_idx_array[:, 0]).astype(
                "int64"
            )
            properties_all = data["properties"][:, self.property_idx].astype("float64")

        # The two sides are independent: pin one with its explicit argument,
        # sample the other from the mode-selected training distribution.
        if n_atoms is not None:
            self.num_atoms = np.full(total_num, int(n_atoms), dtype="int64")
        elif mode == "in_distribution":
            self.num_atoms = self._sample_num_atoms(num_atoms_all, seed)
        else:
            self.num_atoms = self._sample_ood_num_atoms(
                properties_all, num_atoms_all, seed
            )

        if property_value is not None:
            self.property_values = np.full(total_num, float(property_value))
        else:
            self.property_values = self._sample_properties_given_atoms(
                self.num_atoms, properties_all, num_atoms_all, num_bins, seed
            )

    def _sample_num_atoms(self, num_atoms, seed):
        """Draw atom counts from the training atom-count distribution."""
        rng = np.random.RandomState(seed)
        unique_num_atoms, counts = np.unique(num_atoms, return_counts=True)
        num_atoms_probs = counts / counts.sum()
        return rng.choice(unique_num_atoms, size=self.total_num, p=num_atoms_probs)

    def _sample_ood_num_atoms(self, properties, num_atoms, seed):
        """Draw atom counts from molecules in the top property quantile."""
        rng = np.random.RandomState(seed)
        _, low, high = np.quantile(properties, [0.5, 0.97, 1.0])
        mask = (properties >= low) & (properties <= high)
        candidates = num_atoms[mask]
        if candidates.size == 0:
            raise ValueError(
                "No training molecules found in the requested property range"
            )
        return rng.choice(candidates, size=self.total_num, replace=True)

    def _sample_properties_given_atoms(
        self, num_atoms_seq, properties_all, num_atoms_all, num_bins, seed
    ):
        """Sample property values from the per-atom-count training histograms."""
        rng = np.random.RandomState(seed)
        histograms = {}
        for n in np.unique(num_atoms_all):
            values = properties_all[num_atoms_all == n]
            prop_min, prop_max = values.min(), values.max()
            prop_range = prop_max - prop_min + 1e-12
            hist, _ = np.histogram(values, bins=num_bins, range=(prop_min, prop_max))
            hist = hist.astype("float64")
            hist = hist / hist.sum()
            histograms[int(n)] = (hist, prop_min, prop_range)

        sampled = np.empty(self.total_num, dtype="float64")
        for i, n in enumerate(num_atoms_seq):
            if int(n) not in histograms:
                raise ValueError(f"No training molecules with {int(n)} atoms")
            hist, prop_min, prop_range = histograms[int(n)]
            bin_idx = rng.choice(num_bins, p=hist)
            left = bin_idx / num_bins * prop_range + prop_min
            right = (bin_idx + 1) / num_bins * prop_range + prop_min
            sampled[i] = rng.uniform(left, right)
        return sampled

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        normalized = (
            self.property_values[idx] - self.property_mean
        ) / self.property_std
        return {
            "structure_array": {
                "num_atoms": ConcatData(np.array([self.num_atoms[idx]], dtype="int64")),
            },
            "properties": ConcatData(np.array([normalized], dtype="float32")),
        }


def _one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    indices = np.asarray(indices, dtype="int64")
    out = np.zeros(indices.shape + (num_classes,), dtype="float32")
    np.put_along_axis(out, indices[..., None], 1.0, axis=-1)
    return out
