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
Interpolant classes for Stochastic Interpolants.

This module is migrated from OMG (Open Materials Generation).
Original code: omg.si.interpolants
"""

import math
import paddle
from .abstracts import Interpolant, Sigma, Tau
from .corrector import Corrector, IdentityCorrector, PeriodicBoundaryConditionsCorrector


class LinearInterpolant(Interpolant):
    """
    Linear interpolant I(t, x_0, x_1) = (1 - t) * x_0 + t * x_1 between points x_0 and x_1
    from two distributions p_0 and p_1 at times t.
    """

    def __init__(self) -> None:
        """
        Construct linear interpolant.
        """
        super().__init__()

    def alpha(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Alpha function alpha(t) in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Values of the alpha function at the given times.
        :rtype: paddle.Tensor
        """
        return 1.0 - t

    def alpha_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Time derivative of the alpha function in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivatives of the alpha function at the given times.
        :rtype: paddle.Tensor
        """
        return -paddle.ones_like(t)

    def beta(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Beta function beta(t) in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Values of the beta function at the given times.
        :rtype: paddle.Tensor
        """
        return t.clone()

    def beta_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Time derivative of the beta function in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivatives of the beta function at the given times.
        :rtype: paddle.Tensor
        """
        return paddle.ones_like(t)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()


class PeriodicLinearInterpolant(LinearInterpolant):
    """
    Linear interpolant I(t, x_0, x_1) = (1 - t) * x_0 + t * x_1 between points x_0 and x_1
    from two distributions p_0 and p_1 at times t with periodic boundary conditions.
    The coordinates are assumed to be in [0,1].
    """

    def __init__(self) -> None:
        """
        Construct PeriodicLinearInterpolant.
        """
        super().__init__()
        self._corrector = PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Corrector that corrects for periodic boundary conditions.
        :rtype: Corrector
        """
        return PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)


class TrigonometricInterpolant(Interpolant):
    """
    Trigonometric interpolant I(t, x_0, x_1) = cos(pi / 2 * t) * x_0 + sin(pi / 2 * t) * x_1
    between points x_0 and x_1 from two distributions p_0 and p_1 at times t.
    """

    def __init__(self) -> None:
        """
        Construct trigonometric interpolant.
        """
        super().__init__()

    def alpha(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Alpha function alpha(t) in the trigonometric interpolant.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Values of the alpha function at the given times.
        :rtype: paddle.Tensor
        """
        return paddle.cos(math.pi * t / 2.0)

    def alpha_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Time derivative of the alpha function in the trigonometric interpolant.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivatives of the alpha function at the given times.
        :rtype: paddle.Tensor
        """
        return -(math.pi / 2.0) * paddle.sin(math.pi * t / 2.0)

    def beta(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Beta function beta(t) in the trigonometric interpolant.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Values of the beta function at the given times.
        :rtype: paddle.Tensor
        """
        return paddle.sin(math.pi * t / 2.0)

    def beta_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Time derivative of the beta function in the trigonometric interpolant.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivatives of the beta function at the given times.
        :rtype: paddle.Tensor
        """
        return (math.pi / 2.0) * paddle.cos(math.pi * t / 2.0)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()


class ExponentialInterpolant(Interpolant):
    """
    Exponential interpolant I(t, x_0, x_1) = exp(-t) * x_0 + (1 - exp(-t)) * x_1
    between points x_0 and x_1 from two distributions p_0 and p_1 at times t.
    """

    def __init__(self) -> None:
        """
        Construct exponential interpolant.
        """
        super().__init__()

    def alpha(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Alpha function alpha(t) = exp(-t).

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Values of the alpha function at the given times.
        :rtype: paddle.Tensor
        """
        return paddle.exp(-t)

    def alpha_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Time derivative of the alpha function.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivatives of the alpha function at the given times.
        :rtype: paddle.Tensor
        """
        return -paddle.exp(-t)

    def beta(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Beta function beta(t) = 1 - exp(-t).

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Values of the beta function at the given times.
        :rtype: paddle.Tensor
        """
        return 1.0 - paddle.exp(-t)

    def beta_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Time derivative of the beta function.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivatives of the beta function at the given times.
        :rtype: paddle.Tensor
        """
        return paddle.exp(-t)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()
