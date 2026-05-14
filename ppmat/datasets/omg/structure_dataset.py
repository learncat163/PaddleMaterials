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
Structure Dataset for OMG.

This module is migrated from OMG (Open Materials Generation).
Original code: omg.datamodule.structure_dataset

Supports reading crystalline structures from LMDB, CSV, and Parquet files.
"""

import hashlib
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import lmdb
import numpy as np
import paddle

from ppmat.models.omg.datamodule.structure import Structure


class StructureDataset(paddle.io.Dataset):
    """
    Dataset for reading crystalline structures from several file formats.

    This dataset optionally allows for lazy reading of the structures from LMDB files.
    
    Original: omg-raw/omg/datamodule/structure_dataset.py:StructureDataset

    :param file_path:
        Path to the file containing the structures.
        Supported formats are .lmdb, .csv, and .parquet.
    :type file_path: str
    :param property_keys:
        An optional sequence of property keys that should be read from the file.
        Defaults to None.
    :type property_keys: Optional[Sequence[str]]
    :param lazy_storage:
        Whether to read the structures lazily from a LMDB file when they are requested.
        Defaults to True.
    :type lazy_storage: bool
    :param convert_to_fractional:
        Whether to convert the atomic positions to fractional coordinates.
        Defaults to True.
    :type convert_to_fractional: bool
    :param niggli_reduce:
        Whether to apply a Niggli reduction to the returned structures.
        Defaults to False.
    :type niggli_reduce: bool
    """

    def __init__(
        self,
        file_path: str,
        property_keys: Optional[Sequence[str]] = None,
        lazy_storage: bool = True,
        convert_to_fractional: bool = True,
        niggli_reduce: bool = False,
    ) -> None:
        """Constructor for the StructureDataset class."""
        self.file_path = file_path
        self.property_keys = property_keys if property_keys is not None else []
        self.lazy_storage = lazy_storage
        self.convert_to_fractional = convert_to_fractional
        self.niggli_reduce = niggli_reduce
        
        # Determine file format from extension
        self.file_format = Path(file_path).suffix.lower()
        
        # Initialize dataset based on format
        if self.file_format == ".lmdb":
            self._init_from_lmdb()
        elif self.file_format == ".csv":
            self._init_from_csv()
        elif self.file_format == ".parquet":
            self._init_from_parquet()
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
    
    def _init_from_lmdb(self) -> None:
        """
        Initialize dataset from LMDB file.
        
        Original: omg-raw/omg/datamodule/structure_dataset.py:StructureDataset._from_lmdb
        """
        self.env = lmdb.open(
            self.file_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        
        with self.env.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
        
        if not self.lazy_storage:
            # Load all structures into memory
            self.structures = []
            with self.env.begin() as txn:
                for key in self.keys:
                    data = pickle.loads(txn.get(key))
                    structure = self._create_structure(data)
                    self.structures.append(structure)
    
    def _init_from_csv(self) -> None:
        """
        Initialize dataset from CSV file.
        
        Original: omg-raw/omg/datamodule/structure_dataset.py:StructureDataset._from_csv
        
        Note: CSV files are converted to LMDB format for caching.
        """
        # For now, raise NotImplementedError
        # TODO: Implement CSV to LMDB conversion
        raise NotImplementedError("CSV format is not yet supported. Please use LMDB format.")
    
    def _init_from_parquet(self) -> None:
        """
        Initialize dataset from Parquet file.
        
        Original: omg-raw/omg/datamodule/structure_dataset.py:StructureDataset._from_parquet
        
        Note: Parquet files are converted to LMDB format for caching.
        """
        # For now, raise NotImplementedError
        # TODO: Implement Parquet to LMDB conversion
        raise NotImplementedError("Parquet format is not yet supported. Please use LMDB format.")
    
    def _create_structure(self, data: dict[str, Any]) -> Structure:
        """
        Create a Structure object from data dictionary.
        
        Original: omg-raw/omg/datamodule/structure.py:Structure.from_dictionary
        
        :param data:
            Dictionary containing structure data.
        :return:
            Structure object.
        """
        # Extract required fields
        cell = paddle.to_tensor(data["cell"], dtype="float32")
        atomic_numbers = paddle.to_tensor(data["atomic_numbers"], dtype="int64")
        pos = paddle.to_tensor(data["pos"], dtype="float32")
        
        # Extract properties
        property_dict = {}
        for key in self.property_keys:
            if key in data:
                property_dict[key] = paddle.to_tensor(data[key], dtype="float32")
        
        # Create structure
        structure = Structure(
            cell=cell,
            atomic_numbers=atomic_numbers,
            pos=pos,
            property_dict=property_dict if property_dict else None,
            pos_is_fractional=False,  # LMDB stores Cartesian coordinates
        )
        
        # Apply transformations
        if self.convert_to_fractional:
            structure.to_fractional()
        
        if self.niggli_reduce:
            structure.niggli_reduce()
        
        return structure
    
    def __len__(self) -> int:
        """Get the number of structures in the dataset."""
        if self.lazy_storage:
            return len(self.keys)
        else:
            return len(self.structures)
    
    def __getitem__(self, idx: int) -> Structure:
        """
        Get a structure from the dataset.
        
        :param idx:
            Index of the structure to retrieve.
        :return:
            Structure object.
        """
        if self.lazy_storage:
            # Lazy loading from LMDB
            with self.env.begin() as txn:
                key = self.keys[idx]
                data = pickle.loads(txn.get(key))
            return self._create_structure(data)
        else:
            # Loading from memory
            return self.structures[idx]
    
    def __del__(self) -> None:
        """Clean up LMDB environment."""
        if hasattr(self, 'env'):
            self.env.close()
