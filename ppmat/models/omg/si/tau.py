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
Tau schedule implementations for Stochastic Interpolants.

This module is migrated from OMG (Open Materials Generation).
Original code: omg.si.tau
"""

from math import exp, pi, sin

import paddle
from .abstracts import Tau


class TauConstantSchedule(Tau):
    """
    Tau function tau(t) = t corresponding to a constant noise schedule beta(s) = 2 in a variance-preserving interpolant.
    """

    def __init__(self) -> None:
        """
        Construct constant tau function.
        """
        super().__init__()

    def tau(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Evaluate the tau function at times t.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Tau function tau(t).
        :rtype: paddle.Tensor
        """
        self._check_t(t)
        return t.clone()

    def tau_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Compute the derivative of the tau function with respect to time.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivative of the tau function at the given times.
        :rtype: paddle.Tensor
        """
        self._check_t(t)
        return paddle.ones_like(t)


class TauLinearSchedule(Tau):
    """
    Tau function tau(t) = exp(1/2 * beta_min * log(t) - 1/4 * (beta_max - beta_min) * log^2(t)) corresponding to a
    linear noise schedule beta(s) = beta_min + (beta_max - beta_min) * s in a variance-preserving interpolant.

    :param beta_min:
        The minimum noise level.
    :type beta_min: float
    :param beta_max:
        The maximum noise level.
    :type beta_max: float

    :raises ValueError:
        If beta_min is not positive or if beta_max is not positive.
    :raises ValueError:
        If beta_max is not greater than beta_min.
    """

    def __init__(self, beta_min: float, beta_max: float) -> None:
        """
        Construct linear tau function.
        """
        super().__init__()
        if beta_min <= 0.0:
            raise ValueError("beta_min must be positive.")
        if beta_max <= 0.0:
            raise ValueError("beta_max must be positive.")
        if beta_max <= beta_min:
            raise ValueError("beta_max must be greater than beta_min.")
        self._beta_min = beta_min
        self._beta_max = beta_max

    def tau(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Evaluate the tau function at times t.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Tau function tau(t).
        :rtype: paddle.Tensor
        """
        self._check_t(t)
        log_t = paddle.log(t)
        return paddle.exp(0.5 * self._beta_min * log_t - 0.25 * (self._beta_max - self._beta_min) * log_t ** 2)

    def tau_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Compute the derivative of the tau function with respect to time.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivative of the tau function at the given times.
        :rtype: paddle.Tensor
        """
        self._check_t(t)
        log_t = paddle.log(t)
        exp_factor = paddle.exp(0.5 * self._beta_min * log_t - 0.25 * (self._beta_max - self._beta_min) * log_t ** 2)
        return exp_factor * (0.5 * self._beta_min / t - 0.5 * (self._beta_max - self._beta_min) * log_t / t)


class TauCosineSchedule(Tau):
    """
    Tau function tau(t) = csc(pi / (2 + 2 * d)) * sin((pi + pi * log(t))/ (2 + 2 * d)) corresponding to a
    cosine noise schedule beta(s) = pi / (1 + d) tan(pi / 2 * (s + d) / (1 + d)) in a variance-preserving interpolant.

    :param offset:
        The offset d in the cosine noise schedule.
    :type offset: float

    :raises ValueError:
        If offset is not zero or positive.
    """

    def __init__(self, offset: float) -> None:
        """
        Construct linear tau function.
        """
        super().__init__()
        if offset < 0.0:
            raise ValueError("offset must be non-negative.")
        self._offset = offset
        self._offset_factor = pi / (2.0 + 2.0 * self._offset)
        self._csc_prefactor = 1.0 / sin(self._offset_factor)
        self._one_over_e = 1.0 / exp(1.0)

    def tau(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Evaluate the tau function at times t.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Tau function tau(t).
        :rtype: paddle.Tensor
        """
        self._check_t(t)
        log_t = paddle.log(t)
        return self._csc_prefactor * paddle.sin((pi + pi * log_t) / (2.0 + 2.0 * self._offset))

    def tau_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Compute the derivative of the tau function with respect to time.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivative of the tau function at the given times.
        :rtype: paddle.Tensor
        """
        self._check_t(t)
        log_t = paddle.log(t)
        offset_term = pi / (2.0 + 2.0 * self._offset)
        cos_term = paddle.cos(offset_term + log_t * offset_term)
        return self._csc_prefactor * offset_term * cos_term / t
