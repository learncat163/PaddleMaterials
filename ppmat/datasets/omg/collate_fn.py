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
Collate function for OMG dataset.

This module provides collate functions for batching OMG structures.
Original code: omg-raw/omg/datamodule/omg_data.py
"""

from typing import List, Optional

import paddle

from ppmat.models.omg.datamodule.omg_data import OMGData
from ppmat.models.omg.datamodule.structure import Structure


def collate_omg_structures(structures: List[Structure]) -> OMGData:
    """
    Collate a list of Structure objects into a single OMGData batch.
    
    Original: omg-raw/omg/datamodule/omg_data.py:OMGData.from_batch
    
    :param structures:
        List of Structure objects to collate.
    :return:
        OMGData object containing the batched structures.
    """
    return OMGData.from_batch(structures, concatenate=True)


def collate_omg_structures_with_properties(
    structures: List[Structure],
    property_keys: Optional[List[str]] = None,
) -> OMGData:
    """
    Collate a list of Structure objects with properties into a single OMGData batch.
    
    :param structures:
        List of Structure objects to collate.
    :param property_keys:
        List of property keys to include in the batch.
    :return:
        OMGData object containing the batched structures.
    """
    batch = OMGData.from_batch(structures, concatenate=True)
    
    # Filter properties if needed
    if property_keys is not None and batch.property_dict is not None:
        filtered_properties = {}
        for key in property_keys:
            if key in batch.property_dict:
                filtered_properties[key] = batch.property_dict[key]
        batch.property_dict = filtered_properties if filtered_properties else None
    
    return batch
