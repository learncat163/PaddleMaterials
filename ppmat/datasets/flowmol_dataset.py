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

"""FlowMol molecule datasets backed by preprocessed npz splits.

Every sample provides the endpoint features (COM-free positions, one-hot atom
types, one-hot charges and upper-triangle one-hot bond types) plus the aligned
position prior consumed by the flow matching training path.
"""

from __future__ import annotations

import os.path as osp

import numpy as np
from paddle.io import Dataset
from scipy.optimize import linear_sum_assignment

from ppmat.utils import download
from ppmat.utils import logger

DATASET_NAME = "flowmol_qm9"
# Local test URL, replace with the online address before submitting.
DATASET_URL = "http://127.0.0.1/models/flowmol_qm9.zip"  # noqa: E501
DATASET_MD5 = "4d7174578ed4ad8b7efbd0f470980800"

N_CHARGES = 6
N_BOND_TYPES = 4
CHARGE_OFFSET = 2


def _one_hot(idx: np.ndarray, num_classes: int) -> np.ndarray:
    one_hot = np.zeros((idx.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(idx.shape[0]), idx] = 1.0
    return one_hot


def _align_prior(prior: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Align a position prior to the endpoint (Kabsch, after permutation)."""
    cost = np.linalg.norm(dst[:, None, :] - prior[None, :, :], axis=-1)
    _, prior_idx = linear_sum_assignment(cost)
    prior = prior[prior_idx]

    prior_mean = prior.mean(axis=0, keepdims=True)
    dst_mean = dst.mean(axis=0, keepdims=True)
    prior_c = prior - prior_mean
    dst_c = dst - dst_mean

    h = prior_c.T @ dst_c
    u, _, vh = np.linalg.svd(h)
    r = vh.T @ u.T
    aligned = prior_c @ r.T
    return (aligned + dst_mean).astype("float32")


def _resolve_split_path(path: str, name: str, url: str, md5: str) -> str:
    """Resolve a split path, downloading the package when it is missing.

    The download source comes from the caller class attributes so that every
    dataset subclass resolves against its own package (``name/url/md5``).
    """
    if osp.exists(path):
        return path
    logger.message("The dataset is not found. Will download it now.")
    root_path = download.get_datasets_path_from_url(url, md5)
    return osp.join(root_path, name, osp.basename(path))


class FlowMolDataset(Dataset):
    """FlowMol training dataset over a preprocessed npz split."""

    name = DATASET_NAME
    url = DATASET_URL
    md5 = DATASET_MD5

    def __init__(
        self,
        path: str = "./data/flowmol_qm9/train_data_processed.npz",
        atom_map: list = None,
        fake_atom_p: float = 0.0,
        fake_atom_std: float = 1.0,
        prior_std: float = 1.0,
        prior_align: bool = True,
        seed: int = None,
    ):
        super().__init__()

        if atom_map is None:
            raise ValueError("atom_map must be provided")

        self.atom_map = list(atom_map)
        self.n_atom_types = len(self.atom_map)
        self.fake_atom_p = fake_atom_p
        self.fake_atom_std = fake_atom_std
        self.use_fake_atoms = fake_atom_p > 0
        self.prior_std = prior_std
        self.prior_align = prior_align
        self.rng = np.random.RandomState(seed)

        path = _resolve_split_path(path, self.name, self.url, self.md5)
        self.path = path

        with np.load(path) as data:
            self.positions = data["positions"]
            self.atom_types = data["atom_types"]
            self.atom_charges = data["atom_charges"]
            self.bond_types = data["bond_types"]
            self.bond_idxs = data["bond_idxs"]
            self.node_idx_array = data["node_idx_array"]
            self.edge_idx_array = data["edge_idx_array"]

    def __len__(self):
        return self.node_idx_array.shape[0]

    def __getitem__(self, idx):
        node_start, node_end = self.node_idx_array[idx]
        edge_start, edge_end = self.edge_idx_array[idx]

        positions = self.positions[node_start:node_end].astype("float32")
        atom_types_idx = self.atom_types[node_start:node_end].argmax(-1)
        atom_charges_idx = self.atom_charges[node_start:node_end].astype("int64")

        if self.use_fake_atoms:
            n_real_atoms = positions.shape[0]
            max_num_fake_atoms = int(np.ceil(n_real_atoms * self.fake_atom_p))
            num_fake_atoms = int(self.rng.randint(0, max_num_fake_atoms + 1))
            anchor_atom_idxs = self.rng.randint(0, n_real_atoms, num_fake_atoms)
            fake_positions = positions[anchor_atom_idxs]
            fake_positions = fake_positions + (
                self.rng.randn(*fake_positions.shape) * self.fake_atom_std
            ).astype("float32")

            positions = np.concatenate((positions, fake_positions), axis=0)
            atom_types_idx = np.concatenate(
                (
                    atom_types_idx,
                    np.full(num_fake_atoms, self.n_atom_types, dtype="int64"),
                ),
                axis=0,
            )
            atom_charges_idx = np.concatenate(
                (
                    atom_charges_idx,
                    np.zeros(num_fake_atoms, dtype="int64"),
                ),
                axis=0,
            )

        positions = positions - positions.mean(axis=0, keepdims=True)

        n_atoms = positions.shape[0]
        atom_types = _one_hot(
            atom_types_idx, self.n_atom_types + int(self.use_fake_atoms)
        )
        atom_charges = _one_hot(atom_charges_idx + CHARGE_OFFSET, N_CHARGES)

        adjacency = np.zeros((n_atoms, n_atoms), dtype=np.int64)
        bond_idxs = self.bond_idxs[edge_start:edge_end]
        adjacency[bond_idxs[:, 0], bond_idxs[:, 1]] = self.bond_types[
            edge_start:edge_end
        ]
        upper_i, upper_j = np.triu_indices(n_atoms, k=1)
        bond_types = _one_hot(adjacency[upper_i, upper_j], N_BOND_TYPES)

        prior_positions = self.rng.randn(n_atoms, 3).astype("float32") * self.prior_std
        prior_positions = prior_positions - prior_positions.mean(axis=0, keepdims=True)
        if self.prior_align:
            prior_positions = _align_prior(prior_positions, positions)

        structure_array = {
            "num_atoms": n_atoms,
            "positions": positions,
            "atom_types": atom_types,
            "atom_charges": atom_charges,
            "bond_types": bond_types,
        }
        return {
            "structure_array": structure_array,
            "prior_positions": prior_positions,
        }


class FlowMolSamplingDataset(Dataset):
    """Sampling inputs for FlowMol: atom counts drawn from the train histogram.

    ``path`` points at the atom-count histogram npz of the training split;
    the histogram package is downloaded through the shared pipeline when the
    file is missing.
    """

    name = DATASET_NAME
    url = DATASET_URL
    md5 = DATASET_MD5

    def __init__(
        self,
        path: str = "./data/flowmol_qm9/train_data_n_atoms_histogram.npz",
        total_num: int = None,
        seed: int = None,
    ):
        super().__init__()

        if total_num is None:
            raise ValueError("total_num must be provided")
        if seed is None:
            raise ValueError("seed must be provided")

        self.total_num = total_num

        path = _resolve_split_path(path, self.name, self.url, self.md5)
        self.path = path

        with np.load(path) as hist:
            n_atoms = hist["n_atoms"]
            n_atom_counts = hist["n_atom_counts"]

        probs = n_atom_counts / n_atom_counts.sum()
        self.rng = np.random.RandomState(seed)
        self.sampled_num_atoms = self.rng.choice(
            n_atoms, size=total_num, p=probs
        ).astype("int64")

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        return {
            "structure_array": {
                "num_atoms": int(self.sampled_num_atoms[idx]),
            }
        }


class FlowMol3GeomSamplingDataset(FlowMolSamplingDataset):
    """Sampling inputs of the released FlowMol3 model (GEOM atom histogram)."""

    name = "flowmol3_geom_hist"
    url = "http://127.0.0.1/models/flowmol3_geom_hist.zip"  # noqa: E501
    md5 = "6f5bfe153b9ecf83a86d4088ec290839"

    def __init__(
        self,
        path: str = "./data/flowmol3_geom_hist/train_data_n_atoms_histogram.npz",
        total_num: int = None,
        seed: int = None,
    ):
        super().__init__(path=path, total_num=total_num, seed=seed)
