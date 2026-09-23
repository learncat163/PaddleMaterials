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

from __future__ import annotations

from typing import Dict

from rdkit import Chem
from rdkit.Geometry import Point3D

BOND_TYPE_MAP = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def build_rdkit_mol(molecule: Dict) -> Chem.Mol:
    """Build an RDKit molecule from a sampled molecule dictionary.

    The dictionary carries ``atom_types`` (symbols, including the ``Se`` mask
    placeholder), ``positions``, ``charges`` and ``bonds`` (source, destination,
    bond type index).
    """
    mol = Chem.RWMol()
    for symbol, charge in zip(molecule["atom_types"], molecule["charges"]):
        atom = Chem.Atom(symbol)
        if charge != 0:
            atom.SetFormalCharge(int(charge))
        mol.AddAtom(atom)

    for src, dst, bond_type in molecule["bonds"]:
        bond_type = int(bond_type)
        if not 0 < bond_type < len(BOND_TYPE_MAP):
            raise ValueError(f"unsupported bond type index: {bond_type}")
        mol.AddBond(int(src), int(dst), BOND_TYPE_MAP[bond_type])

    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, (x, y, z) in enumerate(molecule["positions"]):
        conformer.SetAtomPosition(index, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conformer)
    return mol.GetMol()
