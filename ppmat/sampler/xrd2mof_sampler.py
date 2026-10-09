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

from ppmat.datasets import build_dataloader
from ppmat.sampler.structure_sampler import StructureSampler


class Xrd2MofSampler(StructureSampler):
    """Independent sampler for the coarse-grained Xrd2Mof generation model.

    Loading, execution backend and the ``model.sample`` contract are inherited
    from ``StructureSampler``. Only result persistence is overridden: the model
    emits coarse-grained metal-node/linker sites rather than a fully atomistic
    crystal, so each sample is written as one npz file instead of a CIF.

    Enable it from the shared sampling CLI with
    ``--sampler_type Xrd2MofSampler``.
    """

    supported_modes = frozenset({"by_dataloader"})

    def sample_by_dataloader(self, save_path=None):
        dataset_cfg = self.sample_config["data"]
        data_loader = build_dataloader(dataset_cfg)

        total_results = []
        for iter_id, batch_data in enumerate(data_loader):
            pred_data = self.sample(batch_data)
            if save_path is not None:
                self._save_results(pred_data["result"], save_path, iter_id)
            total_results.extend(pred_data["result"])
        return total_results

    @staticmethod
    def _save_results(results, save_path, iter_id):
        os.makedirs(save_path, exist_ok=True)
        for index, result in enumerate(results):
            mof_id = result.get("mof_id", "sample")
            np.savez(
                os.path.join(save_path, f"{mof_id}_{iter_id + 1}_{index + 1}.npz"),
                lattice=np.asarray(result["lattice"], dtype="float32"),
                frac_coords=np.asarray(result["frac_coords"], dtype="float32"),
                atom_types=np.asarray(result["atom_types"], dtype="int64"),
                num_atoms=np.asarray(result["num_atoms"], dtype="int64"),
                mof_id=str(mof_id),
            )
