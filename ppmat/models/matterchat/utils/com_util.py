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

"""Common utilities for MatterChat."""

import paddle


def custom_send_to_device(batch, device):
    """Send batch data to specified device."""
    if isinstance(batch, paddle.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: custom_send_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [custom_send_to_device(item, device) for item in batch]
    elif isinstance(batch, tuple):
        return tuple(custom_send_to_device(item, device) for item in batch)
    else:
        return batch
