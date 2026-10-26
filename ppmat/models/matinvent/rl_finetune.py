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
from omegaconf import OmegaConf

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import check_backend
from ppmat.models.matinvent.rewards.reward import Reward
from ppmat.models.matinvent.rl_loop import MatInvent
from ppmat.models.matinvent.suites import ModelSuite
from ppmat.models.matinvent.suites import build_backbone
from ppmat.utils import logger
from ppmat.utils.execution import DEFAULT_EXECUTION_BACKEND


class MatinventRL(RuntimeMixin, nn.Layer):
    """Framework-facing adapter around one MatInvent diffusion backbone.

    The backbone owns the runtime; this parameter-free layer lazily builds it and
    forwards sampling, protocol and state-dict calls to it, so
    ``build_model_from_name`` and the framework ``Sample`` flow drive one object.

    The backbone architecture is not repeated in the task config: it is read from
    the registered package named by ``rl_cfg.pretrained_model_name`` unless
    ``backbone_cfg`` is given. RL fine-tuning runs through the unified training
    entry, which calls ``before_train(trainer)`` before its own epoch loop.
    """

    def __init__(
        self,
        model_type,
        num_inference_steps,
        rl_cfg,
        build_structure_cfg,
        backbone_cfg=None,
        execution_backend="eager",
        runtime_options=None,
    ):
        super().__init__()
        self.model_type = model_type
        self.num_inference_steps = num_inference_steps
        self.rl_cfg = rl_cfg
        self.build_structure_cfg = build_structure_cfg
        self.backbone_cfg = backbone_cfg
        self._backbone = None
        self.requested_execution_backend = check_backend(execution_backend)
        self.requested_runtime_options = runtime_options or {}

    @property
    def execution_backend(self):
        return self.backbone().execution_backend

    def backbone(self):
        if self._backbone is None:
            cfg = self.backbone_cfg
            if cfg is None:
                cfg = resolve_backbone_cfg(self.rl_cfg["pretrained_model_name"])
            self._backbone = build_backbone(cfg)
            self._backbone.set_execution_backend(self.requested_execution_backend)
            self._backbone.set_runtime_options(self.requested_runtime_options)
        return self._backbone

    def set_execution_backend(self, backend):
        self.requested_execution_backend = check_backend(backend)
        if self._backbone is not None:
            self._backbone.set_execution_backend(backend)

    def set_runtime_options(self, runtime_options):
        self.requested_runtime_options = runtime_options or {}
        if self._backbone is not None:
            self._backbone.set_runtime_options(runtime_options)

    def validate_execution_backend(self, *, use_amp=False, world_size=1):
        self.backbone().validate_execution_backend(
            use_amp=use_amp, world_size=world_size
        )

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

    def before_train(self, trainer):
        """Run the MatInvent RL loop from the unified training entry."""
        run_rl_training(
            model_type=self.model_type,
            rl_cfg=self.rl_cfg,
            build_structure_cfg=self.build_structure_cfg,
            output_dir=trainer.output_dir,
            backbone_cfg=self.backbone_cfg,
            execution_config=self._execution_config(),
        )

    def _execution_config(self):
        backend = self.requested_execution_backend
        if backend == DEFAULT_EXECUTION_BACKEND:
            return None
        return {
            "backend": backend,
            "__init_params__": self.requested_runtime_options.get(backend, {}),
        }


def _download_package(model_name):
    from ppmat.models import MODEL_REGISTRY
    from ppmat.utils import download
    from ppmat.utils.model_package import resolve_model_package_dir

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"{model_name} is not registered in MODEL_REGISTRY, "
            f"available: {sorted(MODEL_REGISTRY)}"
        )
    extracted_path = download.get_weights_path_from_url(MODEL_REGISTRY[model_name])
    return resolve_model_package_dir(model_name, extracted_path)


def _read_backbone_cfg(model_name, package_dir):
    from ppmat.utils.model_package import get_model_config_path

    config = OmegaConf.to_container(
        OmegaConf.load(get_model_config_path(model_name, package_dir)), resolve=True
    )
    return config["Model"]["__init_params__"]["backbone_cfg"]


def resolve_backbone_cfg(model_name):
    """Read the backbone architecture from the registered model package."""
    return _read_backbone_cfg(model_name, _download_package(model_name))


def run_rl_training(
    model_type,
    rl_cfg,
    build_structure_cfg,
    output_dir,
    backbone_cfg=None,
    execution_config=None,
):
    """Run MatInvent RL fine-tuning.

    The RL agent and KL prior both start from the registered weight package named
    by ``rl_cfg.pretrained_model_name``, which also provides the backbone
    architecture when ``backbone_cfg`` is not given.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_name = rl_cfg["pretrained_model_name"]
    pretrained_model_path = rl_cfg.get("pretrained_model_path")
    if pretrained_model_path is None:
        package_dir = _download_package(model_name)
        if backbone_cfg is None:
            backbone_cfg = _read_backbone_cfg(model_name, package_dir)
    else:
        package_dir = os.path.expanduser(pretrained_model_path)
        if backbone_cfg is None:
            raise ValueError(
                "backbone_cfg is required when rl_cfg.pretrained_model_path is set"
            )

    model_suite = ModelSuite(
        model_name=model_type,
        backbone_cfg=backbone_cfg,
        sample_cfg=rl_cfg["sample_cfg"],
        finetune_cfg=rl_cfg["finetune_cfg"],
        build_structure_cfg=build_structure_cfg,
        pretrained_model_path=package_dir,
        execution_config=execution_config,
    )
    reward_cfg = rl_cfg["reward_cfg"]
    reward = Reward(
        root_dir=os.path.join(output_dir, "rewards"),
        prop_cfg=reward_cfg["prop_cfg"],
        reward_threshold=reward_cfg["reward_threshold"],
        reduce=reward_cfg["reduce"],
    )
    replay_cfg = rl_cfg.get("replay_cfg")
    div_cfg = rl_cfg["div_filter_cfg"]
    mat_invent = MatInvent(
        rl_epoch=rl_cfg["rl_epoch"],
        model_suite=model_suite,
        reward=reward,
        topk_ratio=rl_cfg["topk_ratio"],
        save_dir=output_dir,
        save_freq=rl_cfg["save_freq"],
        replay=replay_cfg is not None,
        replay_args=replay_cfg,
        div_filter=div_cfg["enabled"],
        df_args={
            "tol": div_cfg["tol"],
            "buff": div_cfg["buff"],
            "method": div_cfg["method"],
        },
        anchor_cfg=rl_cfg.get("anchor_cfg"),
        early_stop_cfg=rl_cfg.get("early_stop_cfg"),
        qualification_cfg=rl_cfg.get("qualification_cfg"),
    )
    logger.info(f"Start MatInvent RL fine-tuning on {model_type}")
    mat_invent.run_rl()
    logger.info(f"MatInvent RL fine-tuning finished, results saved to {output_dir}")
