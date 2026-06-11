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

"""Utility functions for MatterChat."""

import os

import yaml


def now():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d%H%M")[:-1]


def get_abs_path(path):
    """Get absolute path relative to the calling file's directory."""
    return os.path.abspath(path)


def load_yaml(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data, filename):
    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
