# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reinforcement learning training script for material generation.

This code is adapted from:
https://github.com/your-repo/raw-matinvent/blob/main/main.py

Now uses ppmat.models.matinvent.common for shared utilities.
"""

import os
import sys
import argparse
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from ppmat.models.matinvent.common import (
    setup_matinvent_logging,
    load_config,
    build_model_from_config,
    prepare_output_dir,
)
from ppmat.models.matinvent.rl.mat_invent import MatInvent
from ppmat.models.matinvent.rewards.reward import Reward


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reinforcement learning training for material generation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="structure_generation/configs/rl/mattergen_rl.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mattergen",
        choices=["mattergen", "diffcsp"],
        help="Model to use for training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (gpu/cpu)",
    )
    return parser.parse_args()


def build_reward(cfg, output_dir: str):
    """Build reward function based on configuration."""
    reward_cfg = cfg.RL.reward_cfg
    return Reward(
        root_dir=os.path.join(output_dir, "rewards"),
        prop_cfg=reward_cfg.prop_cfg,
        reward_threshold=reward_cfg.reward_threshold,
        reduce=reward_cfg.reduce,
    )


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Override output directory if specified
    if args.output_dir is not None:
        cfg.Global.output_dir = args.output_dir

    # Prepare output directory
    prepare_output_dir(cfg.Global.output_dir)

    # Setup logging using ppmat logger
    logger = setup_matinvent_logging(cfg.Global.output_dir)
    logger.info(f"Output directory: {cfg.Global.output_dir}")

    # Build model suite using common utility
    model_suite = build_model_from_config(
        cfg=cfg,
        model_type=args.model,
    )

    # Build reward function
    reward = build_reward(cfg, cfg.Global.output_dir)

    # Create MatInvent instance
    mat_invent = MatInvent(
        rl_epoch=cfg.RL.rl_epoch,
        model_suite=model_suite,
        reward=reward,
        sample_cfg=cfg.RL.sample_cfg,
        finetune_cfg=cfg.RL.finetune_cfg,
        topk_ratio=cfg.RL.topk_ratio,
        save_dir=cfg.Global.output_dir,
        save_freq=cfg.RL.save_freq,
        device=args.device,
        logger=logger,
        replay=cfg.RL.get("replay_cfg", None) is not None,
        replay_args=cfg.RL.get("replay_cfg", {}),
        div_filter=cfg.RL.div_filter_cfg.enabled,
        df_args=cfg.RL.div_filter_cfg,
    )

    # Run reinforcement learning
    logger.info("Starting reinforcement learning training...")
    mat_invent.run_rl()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
