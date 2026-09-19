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

from ppmat.models.chemeleon2.common.distributions import DiagonalGaussianDistribution
from ppmat.models.chemeleon2.common.lora import LoRALayer
from ppmat.models.chemeleon2.common.lora import LoRALinear
from ppmat.models.chemeleon2.common.lora import apply_lora_to_linear
from ppmat.models.chemeleon2.common.lora import get_lora_parameters
from ppmat.models.chemeleon2.common.lora import merge_lora_weights
from ppmat.models.chemeleon2.common.lora import print_trainable_parameters
from ppmat.models.chemeleon2.common.schema import CrystalBatch
from ppmat.models.chemeleon2.common.schema import build_structure_array
from ppmat.models.chemeleon2.common.schema import create_empty_batch
from ppmat.models.chemeleon2.common.utils import apply_augmentation
from ppmat.models.chemeleon2.common.utils import apply_noise
from ppmat.models.chemeleon2.common.utils import get_index_embedding
from ppmat.models.chemeleon2.common.utils import lattice_vector_to_volume
from ppmat.models.chemeleon2.common.utils import make_attn_mask
from ppmat.models.chemeleon2.common.utils import set_gelu_approx
from ppmat.models.chemeleon2.common.utils import to_dense_batch

__all__ = [
    "DiagonalGaussianDistribution",
    "CrystalBatch",
    "build_structure_array",
    "create_empty_batch",
    "lattice_vector_to_volume",
    "to_dense_batch",
    "apply_augmentation",
    "apply_noise",
    "get_index_embedding",
    "make_attn_mask",
    "set_gelu_approx",
    "LoRALayer",
    "LoRALinear",
    "apply_lora_to_linear",
    "get_lora_parameters",
    "merge_lora_weights",
    "print_trainable_parameters",
]
