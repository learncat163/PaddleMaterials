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
Corrector classes for Stochastic Interpolants.

This module is migrated from OMG (Open Materials Generation).
Original code: omg.si.corrector
"""

import paddle
from .abstracts import Corrector


class IdentityCorrector(Corrector):
    """
    Corrector that does nothing.
    """

    def __init__(self):
        """Construct identity corrector."""
        super().__init__()

    def correct(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Correct the input x.

        :param x:
            Input to correct.
        :type x: paddle.Tensor

        :return:
            Corrected input.
        :rtype: paddle.Tensor
        """
        return x

    def unwrap(self, x_0: paddle.Tensor, x_1: paddle.Tensor) -> paddle.Tensor:
        """
        Correct the input x_1 based on the reference input x_0.

        This method just returns x_1.

        :param x_0:
            Reference input.
        :type x_0: paddle.Tensor
        :param x_1:
            Input to correct.
        :type x_1: paddle.Tensor

        :return:
            Unwrapped x_1 value.
        :rtype: paddle.Tensor
        """
        return x_1.clone()


class PeriodicBoundaryConditionsCorrector(Corrector):
    """
    Corrector function that wraps back coordinates to the interval [min, max]
    with periodic boundary conditions.

    :param min_value:
        Minimum value of the interval.
    :type min_value: float
    :param max_value:
        Maximum value of the interval.
    :type max_value: float

    :raises ValueError:
        If the minimum value is greater than the maximum value.
    """

    def __init__(self, min_value: float, max_value: float) -> None:
        """
        Construct corrector function.
        """
        super().__init__()
        self._min_value = min_value
        self._max_value = max_value
        if self._min_value >= self._max_value:
            raise ValueError("Minimum value must be less than maximum value.")

    def correct(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Correct the input x.

        :param x:
            Input to correct.
        :type x: paddle.Tensor

        :return:
            Corrected input.
        :rtype: paddle.Tensor
        """
        range_val = self._max_value - self._min_value
        return paddle.remainder(x - self._min_value, range_val) + self._min_value

    def unwrap(self, x_0: paddle.Tensor, x_1: paddle.Tensor) -> paddle.Tensor:
        """
        Correct the input x_1 based on the reference input x_0.

        This method returns the image of x_1 closest to x_0 in periodic boundary conditions.

        :param x_0:
            Reference input.
        :type x_0: paddle.Tensor
        :param x_1:
            Input to correct.
        :type x_1: paddle.Tensor

        :return:
            Unwrapped x_1 value.
        :rtype: paddle.Tensor
        """
        separation_vector = x_1 - x_0
        length_over_two = (self._max_value - self._min_value) / 2.0
        range_val = self._max_value - self._min_value
        # Shortest separation lies in interval [-L/2, L/2].
        shortest_separation_vector = paddle.remainder(
            separation_vector + length_over_two, range_val
        ) - length_over_two
        return x_0 + shortest_separation_vector
