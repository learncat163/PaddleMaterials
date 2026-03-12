#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Materials Authors. All Rights Reserved.

import paddle
import paddle.nn as nn
import sys
import os
from omegaconf import OmegaConf

class RLWrapperModel(nn.Layer):
    """Wrapper model that executes RL training when called."""
    
    def __init__(self, **kwargs):
        super().__init__()
        # Store kwargs for potential use
        self.kwargs = kwargs
        
    def forward(self, batch_data):
        """Execute RL training when forward is called."""
        # Get config path from environment variable or current working directory
        config_path = os.environ.get('CONFIG_PATH', None)
        
        # If config_path not set, try to find it from common locations
        if not config_path:
            # Check current directory
            if os.path.exists('structure_generation/configs/matinvent/matinvent_mattergen.yaml'):
                config_path = 'structure_generation/configs/matinvent/matinvent_mattergen.yaml'
            elif os.path.exists('configs/matinvent/matinvent_mattergen.yaml'):
                config_path = 'configs/matinvent/matinvent_mattergen.yaml'
            else:
                # Try to find any yaml file in configs directory
                import glob
                yaml_files = glob.glob('**/*.yaml', recursive=True)
                if yaml_files:
                    config_path = yaml_files[0]
        
        if not config_path:
            raise ValueError("Could not find config file. Set CONFIG_PATH environment variable.")
        
        # Load config to determine model type
        config = OmegaConf.load(config_path)
        model_type = "mattergen"
        if "diffcsp" in config_path.lower() or "diffcsp" in str(config.get("Model", {})).lower():
            model_type = "diffcsp"
        
        # Get output directory from config
        output_dir = config.get("Global", {}).get("output_dir", "./output/matinvent")
        
        # Build RL training arguments
        rl_args = [
            "--config", config_path,
            "--model", model_type,
            "--output_dir", output_dir
        ]
        
        # Temporarily modify sys.argv and execute RL training
        original_argv = sys.argv
        sys.argv = ["rl_wrapper.py"] + rl_args
        
        try:
            from ppmat.models.matinvent.rl_train import main as rl_main
            rl_main()
        finally:
            sys.argv = original_argv
        
        # Return a dummy loss dict to satisfy the trainer
        return {"loss_dict": {"loss": paddle.to_tensor(0.0)}}
