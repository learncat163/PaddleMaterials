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

import pickle

from p_tqdm import p_map
from pymatgen.analysis.structure_matcher import StructureMatcher

from ppmat.utils import logger

__all__ = [
    "StructGenMetric",
]


class StructGenMetric:
    """Metrics for unconditional crystal structure generation.

    Evaluates a list of generated structures (the ``array`` dicts produced by
    structure generation models) with three standard indicators:

    - ``validity``: fraction of generated structures that pass the shared
      ``Crystal`` validity checks (smact composition validity + distances).
    - ``uniqueness``: fraction of unique structures among valid ones, judged
      by ``StructureMatcher.group_structures``.
    - ``novelty``: fraction of unique structures that do not match any
      structure in an optional reference set (e.g. the MP-20 training data
      pickle shipped with the dataset).

    The metric is consumed offline and in one shot by
    ``StructureSampler.compute_metric`` (``metric(total_results)``). It
    intentionally does not implement ``StreamingMetricBase``: there is no
    incremental train/eval consumer for structure-generation metrics.
    """

    def __init__(self, reference_file_path=None, stol=0.5, angle_tol=10, ltol=0.3):
        if reference_file_path is not None and not reference_file_path.endswith(".pkl"):
            raise ValueError(
                "reference_file_path should be a pickle file with "
                "pymatgen structures, got: " + reference_file_path
            )
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.reference_structures = None
        if reference_file_path is not None:
            with open(reference_file_path, "rb") as f:
                self.reference_structures = pickle.load(f)
            if not isinstance(self.reference_structures, (list, tuple)):
                raise ValueError(
                    "reference pickle must contain a list of pymatgen structures"
                )

    def _novelty_one(self, structure):
        try:
            # Novel: matches none of the reference structures
            is_new = all(
                not self.matcher.fit(structure, ref_structure)
                for ref_structure in self.reference_structures
            )
        except Exception as e:
            # StructureMatcher can fail on degenerate lattices; the structure
            # is treated as not novel but the failure is logged, not silenced.
            logger.warning(f"novelty check failed for one structure: {e}")
            return False
        return bool(is_new)

    def __call__(self, pred_data):
        from ppmat.metrics.utils import Crystal

        crystals = [Crystal(crys_array_dict) for crys_array_dict in pred_data]
        valid_crystals = [c for c in crystals if c.valid and c.constructed]
        total = len(crystals)
        n_valid = len(valid_crystals)

        results = {
            "validity": n_valid / total if total > 0 else 0.0,
            "uniqueness": 0.0,
            "novelty": 0.0,
        }
        if n_valid == 0:
            return results

        groups = self.matcher.group_structures([c.structure for c in valid_crystals])
        n_unique = len(groups)
        results["uniqueness"] = n_unique / n_valid

        if self.reference_structures is not None:
            unique_structures = [group[0] for group in groups]
            # O(n_unique x n_reference) StructureMatcher.fit calls; p_map
            # parallelizes the per-structure scan to keep MP-20-scale
            # reference sets tractable.
            flags = p_map(
                self._novelty_one,
                unique_structures,
                desc="Computing novelty",
            )
            results["novelty"] = sum(flags) / max(n_unique, 1)
        return results
