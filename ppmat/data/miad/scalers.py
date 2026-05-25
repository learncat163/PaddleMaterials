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
Data scaler classes for MiAD model.
Converted from PyTorch to PaddlePaddle.
"""

import paddle


def init_scaler(scaler_name, data_sample):
    """
    Initialize a data scaler.

    Args:
        scaler_name: Name of the scaler class (e.g., 'StandardContiniousScaler')
        data_sample: Sample data for fitting the scaler

    Returns:
        scaler: Initialized scaler object
    """
    switch = {
        'StandardContiniousScaler': StandardContiniousScaler
    }
    if scaler_name not in switch:
        raise NotImplementedError(f"Scaler {scaler_name} not implemented")
    return switch[scaler_name](data_sample)


class StandardContiniousScaler:
    """
    Standard continuous data scaler using mean and std normalization.
    """

    def __init__(self, data_sample):
        """
        Initialize the scaler with data sample.

        Args:
            data_sample: List of tensors for computing mean and std
        """
        data_sample = paddle.vstack(data_sample)
        self.mean = data_sample.mean(axis=0)
        self.std = data_sample.std(axis=0)
        self.eps = 1e-3

    def align_devices(self, data):
        """Align devices between data and scaler parameters."""
        self.mean = self.mean.to(data.place)
        self.std = self.std.to(data.place)
        return data
        return data

    def rescale(self, data):
        """
        Rescale data to normalized space (data - mean) / (std + eps).

        Args:
            data: Input tensor to rescale

        Returns:
            rescaled_data: Normalized tensor
        """
        # Note: Original code had a bug calling align_dtypes_and_devices,
        # but the actual method is align_devices. Keeping original behavior.
        data = self.align_devices(data)
        return (data - self.mean) / (self.std + self.eps)

    def scaleup(self, data):
        """
        Scale data back to original space data * (std + eps) + mean.

        Args:
            data: Normalized tensor

        Returns:
            scaled_data: Tensor in original scale
        """
        # Note: Same bug as above in original code
        data = self.align_devices(data)
        return data * (self.std + self.eps) + self.mean
