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

import paddle.nn as nn

from ppmat.models.matinvent.rl_loop import MatInvent
from ppmat.models.matinvent.rewards.reward import Reward
from ppmat.models.matinvent.suites import ModelSuite
from ppmat.models.matinvent.suites import build_backbone
from ppmat.utils import logger


class MatinventRL(nn.Layer):
    """Expose the MatInvent backbone to the framework Sample flow.

    RL fine-tuning itself is driven by the task entry through
    `run_rl_training`; this layer only builds the backbone from
    `backbone_cfg` and delegates sampling and weight loading to it.
    """

    def __init__(self, model_type, num_inference_steps, backbone_cfg):
        super().__init__()
        self.model_type = model_type
        self.num_inference_steps = num_inference_steps
        self.backbone_cfg = backbone_cfg
        self._backbone = None

    def backbone(self):
        if self._backbone is None:
            self._backbone = build_backbone(self.backbone_cfg)
        return self._backbone

    def sample(self, data, **sample_params):
        sample_params.setdefault("num_inference_steps", self.num_inference_steps)
        return self.backbone().sample(data, **sample_params)

    def set_state_dict(self, state_dict, **kwargs):
        return self.backbone().set_state_dict(state_dict, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.backbone().state_dict(*args, **kwargs)

    def eval(self):
        super().eval()
        if self._backbone is not None:
            self._backbone.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        if self._backbone is not None:
            self._backbone.train(mode)
        return self


def run_rl_training(config, model):
    """Run MatInvent RL fine-tuning, driven by the task training entry.

    Args:
        config (dict): Resolved task config containing the `RL` section. The
            RL agent and prior start from the registered weight package named
            by `RL.pretrained_model_name`.
        model (MatinventRL): Model built by the entry; supplies the backbone
            config shared with the RL agent and prior.
    """
    rl_cfg = config["RL"]
    output_dir = config["Trainer"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    from ppmat.models import MODEL_REGISTRY
    from ppmat.utils import download
    from ppmat.utils.model_package import resolve_model_package_dir

    model_name = rl_cfg["pretrained_model_name"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"pretrained_model_name {model_name} is not registered in MODEL_REGISTRY, "
            f"available: {sorted(MODEL_REGISTRY)}")
    extracted_path = download.get_weights_path_from_url(MODEL_REGISTRY[model_name])
    package_dir = resolve_model_package_dir(model_name, extracted_path)

    model_suite = ModelSuite(
        model_name=model.model_type,
        backbone_cfg=model.backbone_cfg,
        sample_cfg=rl_cfg["sample_cfg"],
        finetune_cfg=rl_cfg["finetune_cfg"],
        pretrained_model_path=package_dir,
        build_structure_cfg=config["Sample"]["build_structure_cfg"],
    )
    reward = Reward(
        root_dir=os.path.join(output_dir, "rewards"),
        prop_cfg=rl_cfg["reward_cfg"]["prop_cfg"],
        reward_threshold=rl_cfg["reward_cfg"]["reward_threshold"],
        reduce=rl_cfg["reward_cfg"]["reduce"],
    )
    mat_invent = MatInvent(
        rl_epoch=rl_cfg["rl_epoch"],
        model_suite=model_suite,
        reward=reward,
        topk_ratio=rl_cfg["topk_ratio"],
        save_dir=output_dir,
        save_freq=rl_cfg["save_freq"],
        replay=rl_cfg.get("replay_cfg") is not None,
        replay_args=rl_cfg.get("replay_cfg", {}),
        div_filter=rl_cfg["div_filter_cfg"]["enabled"],
        df_args=rl_cfg["div_filter_cfg"],
    )
    logger.info(f"Start MatInvent RL fine-tuning on {model.model_type}")
    mat_invent.run_rl()
    logger.info(f"MatInvent RL fine-tuning finished, results saved to {output_dir}")
