# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MatInvent model module.

This module contains the MatInvent material generation model and its associated
components for reinforcement learning based material discovery.

Components:
- memory: Long-term memory and replay buffer for RL
- rewards: Reward system and property calculators
- common: Shared utilities that reuse ppmat built-in functionality
"""

from ppmat.models.matinvent.common import (
    setup_matinvent_logging,
    load_config,
    build_model_from_config,
    prepare_output_dir,
)
from ppmat.models.matinvent.memory import ReplayBuffer, LongTimeMem
from ppmat.models.matinvent.rewards import Reward, Calculator

__all__ = [
    # Common utilities
    "setup_matinvent_logging",
    "load_config",
    "build_model_from_config",
    "prepare_output_dir",
    # Memory
    "ReplayBuffer",
    "LongTimeMem",
    # Rewards
    "Reward",
    "Calculator",
]
