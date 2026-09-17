# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np

from ppmat.models.matinvent.rewards.base import Calculator


class FairChem(Calculator):
    """FairChem property calculator for bulk modulus and heat capacity.

    This calculator expects external computation to produce a results file at:
        {root_dir}/{label}.txt
    with one float value per structure (NaN for failed computations).

    Supported tasks: 'bulk_modulus', 'heat_capacity'
    """

    def calc(self, samples, label="tmp"):
        if self.task not in ("bulk_modulus", "heat_capacity"):
            raise ValueError(f"{self.task} is unknown task for FairChem calculator, "
                             f"expected 'bulk_modulus' or 'heat_capacity'")

        out_path = os.path.abspath(os.path.join(self.root_dir, f"{label}.txt"))
        if not os.path.isfile(out_path):
            raise FileNotFoundError(
                f"FairChem results are missing: {out_path}. Run the external ML "
                f"potential computation and write one value per structure to "
                f"{out_path} before rerunning."
            )
        results = np.loadtxt(out_path, dtype=float, ndmin=1)
        if results.shape[0] != len(samples[0]):
            raise ValueError(
                f"{out_path} holds {results.shape[0]} values, expected {len(samples[0])}"
            )
        return results


__all__ = ["FairChem"]
