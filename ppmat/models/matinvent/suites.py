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

from ppmat.models.matinvent.sampler import DiffCSPSampler
from ppmat.models.matinvent.sampler import MatterGenSampler
from ppmat.utils import save_load
from ppmat.utils.execution import configure_execution_backend

_SAMPLER_MAP = {"diffcsp": DiffCSPSampler, "mattergen": MatterGenSampler}
_WEIGHTS_NAME = "best.pdparams"


def build_backbone(backbone_cfg: dict):
    """Build one backbone instance for MatInvent sampling and fine-tuning."""
    from ppmat.models import build_model

    return build_model(backbone_cfg)


class ModelSuite:
    """Backbone factory, weights and RL hyperparameters for MatInvent."""

    def __init__(
        self,
        model_name,
        backbone_cfg,
        sample_cfg,
        finetune_cfg,
        build_structure_cfg,
        pretrained_model_path=None,
        execution_config=None,
    ):
        self.model_name = model_name
        self.backbone_cfg = backbone_cfg
        self.sample_cfg = sample_cfg
        self.finetune_cfg = finetune_cfg
        self.build_structure_cfg = build_structure_cfg
        self.pretrained_model_path = pretrained_model_path
        self.execution_backend = (execution_config or {}).get("backend")
        self.runtime_init_params = (execution_config or {}).get("__init_params__")

    def load_model(self):
        if self.pretrained_model_path is None:
            raise ValueError(
                f"pretrained_model_path must be specified for {self.model_name}"
            )
        model = build_backbone(self.backbone_cfg)
        save_load.load_pretrain(
            model, self.pretrained_model_path, weights_name=_WEIGHTS_NAME
        )
        # The backbone owns the runtime; switching backend never changes state dict.
        configure_execution_backend(
            model,
            self.execution_backend,
            init_params=self.runtime_init_params,
            owner="ModelSuite",
        )
        model.eval()
        return model

    def get_sampler(self):
        cls = _SAMPLER_MAP[self.model_name]
        return cls(
            batch_size=self.sample_cfg["batch_size"],
            num_batches=self.sample_cfg["num_batches"],
            num_inference_steps=self.sample_cfg["num_inference_steps"],
            build_structure_cfg=self.build_structure_cfg,
        )
