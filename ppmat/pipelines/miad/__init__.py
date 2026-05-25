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
MiAD Pipelines.

Modules:
    - train: Training pipeline
    - generate: Generation pipeline
"""

from ppmat.pipelines.miad.train import (
    MiADConfig,
    MiADTrainer,
    create_miad_trainer_from_config,
)
from ppmat.pipelines.miad.generate import (
    GenerationConfig,
    MiADGenerator,
    create_generator_from_checkpoint,
)

__all__ = [
    # Training
    'MiADConfig',
    'MiADTrainer',
    'create_miad_trainer_from_config',
    # Generation
    'GenerationConfig',
    'MiADGenerator',
    'create_generator_from_checkpoint',
]