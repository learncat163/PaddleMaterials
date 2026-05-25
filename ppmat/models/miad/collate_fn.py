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
MiAD batch collate functions.
DEPRECATED: Use ppmat.datasets.miad instead.
This module is kept for backwards compatibility.
"""

# Re-export from the canonical location
from ppmat.datasets.miad import (
    MiADCollator,
    CrystalBatch,
    create_miad_dataloader,
    create_sampling_batch,
)

__all__ = [
    'MiADCollator',
    'CrystalBatch',
    'create_miad_dataloader',
    'create_sampling_batch',
]