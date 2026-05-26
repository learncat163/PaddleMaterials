# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MiAD Training Pipeline.
Integrated with PaddlePaddle training framework.
"""

import os
import time
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.io import DataLoader, Dataset

from ppmat.models.miad.miad import MiAD
from ppmat.models.miad.crystal_diffusion import init_diffusion
from ppmat.datasets.miad import MiADCollator, create_sampling_batch


@dataclass
class MiADConfig:
    """Configuration for MiAD training.

    Attributes:
        model: Model configuration dict (passed to CSPNet)
        diffusion: Diffusion configuration dict
        data: Data configuration
        optimization: Optimization configuration
        training: Training loop configuration
        checkpoints: Checkpoint configuration
        mirage_max_atoms: Max atoms for mirage infusion (None to disable)
    """
    model: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_dim': 128,
        'latent_dim': 256,
        'num_layers': 4,
        'max_atoms': 100,
        'act_fn': 'silu',
        'dis_emb': 'sin',
        'num_freqs': 10,
        'edge_style': 'fc',
        'cutoff': 6.0,
        'max_neighbors': 20,
        'ln': False,
        'ip': True,
        'smooth': False,
        'pred_type': False,
    })
    diffusion: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=lambda: {
        'train_path': './data/mp_20/train.csv',
        'val_path': './data/mp_20/val.csv',
        'test_path': './data/mp_20/test.csv',
        'batch_size': 32,
        'num_workers': 0,
    })
    optimization: Dict[str, Any] = field(default_factory=lambda: {
        'lr': 1e-4,
        'weight_decay': 1e-6,
        'grad_clip': 1.0,
        'ema_decay': 0.999,
        'beta1': 0.9,
        'beta2': 0.999,
    })
    training: Dict[str, Any] = field(default_factory=lambda: {
        'n_epochs': 100,
        'eval_freq': 10,
        'save_freq': 10,
        'log_freq': 10,
        'distributed': False,
    })
    checkpoints: Dict[str, Any] = field(default_factory=lambda: {
        'save_dir': './checkpoints',
        'keep_last_n': 5,
    })
    mirage_max_atoms: Optional[int] = None


class MiADTrainer:
    """MiAD Trainer with PaddlePaddle integration.

    Features:
        - Single-GPU and Distributed training
        - EMA (Exponential Moving Average)
        - Gradient clipping
        - Checkpoint saving/loading
        - Logging (console + tensorboard)
    """

    def __init__(
        self,
        config: MiADConfig,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        device: str = 'gpu',
    ):
        """
        Args:
            config: MiADConfig instance with all settings.
            train_dataset: Training dataset (MP20Dataset).
            val_dataset: Validation dataset.
            device: Device to use ('gpu' or 'cpu').
        """
        self.config = config
        self.device = device
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.is_root = self.rank == 0

        # Build model
        self.model = MiAD(
            model_cfg=config.model,
            diffusion_cfg=config.diffusion,
        )

        # Optimizer
        self.optimizer = self._build_optimizer()

        # EMA
        self.ema = EMAHelper(
            model=self.model,
            decay=config.optimization.get('ema_decay', 0.999),
        )

        # Dataloaders
        self.train_loader = None
        self.val_loader = None
        if train_dataset is not None:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=config.data['batch_size'],
                shuffle=True,
                num_workers=config.data.get('num_workers', 0),
                collate_fn=MiADCollator(
                    mirage_max_atoms=config.mirage_max_atoms,
                    use_mirage=config.mirage_max_atoms is not None,
                ),
            )
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.data['batch_size'],
                shuffle=False,
                num_workers=config.data.get('num_workers', 0),
                collate_fn=MiADCollator(
                    mirage_max_atoms=config.mirage_max_atoms,
                    use_mirage=config.mirage_max_atoms is not None,
                ),
            )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Logger
        self.logger = TrainingLogger(
            log_dir=config.checkpoints.get('save_dir', './checkpoints'),
            log_freq=config.training.get('log_freq', 10),
        )

        # Move to device
        if 'gpu' in device and paddle.device.is_compiled_with_cuda():
            self.model = self.model.cuda()
            self.optimizer = self.optimizer.cuda()

    def _build_optimizer(self):
        """Build optimizer with AMSGrad."""
        opt_cfg = self.config.optimization
        return paddle.optimizer.AdamW(
            learning_rate=opt_cfg.get('lr', 1e-4),
            parameters=self.model.parameters(),
            weight_decay=opt_cfg.get('weight_decay', 1e-6),
            beta1=opt_cfg.get('beta1', 0.9),
            beta2=opt_cfg.get('beta2', 0.999),
            grad_clip=paddle.nn.ClipGradByNorm(
                opt_cfg.get('grad_clip', 1.0)
            ),
        )

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step.

        Args:
            batch: Batch dict from DataLoader.

        Returns:
            Dict with loss values.
        """
        self.model.train()

        # Forward pass
        output = self.model(batch)
        loss_dict = output['loss_dict']
        loss = loss_dict.get('loss', sum(loss_dict.values()))

        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()

        # Update EMA
        self.ema.update()

        return {
            'loss': float(loss),
            'loss_lat': float(loss_dict.get('loss_lat', 0)),
            'loss_frac': float(loss_dict.get('loss_frac', 0)),
            'loss_type': float(loss_dict.get('loss_type', 0)),
        }

    @paddle.no_grad()
    def eval_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single evaluation step.

        Args:
            batch: Batch dict from DataLoader.

        Returns:
            Dict with loss values.
        """
        self.model.eval()
        output = self.model(batch)
        loss_dict = output['loss_dict']
        loss = loss_dict.get('loss', sum(loss_dict.values()))

        return {
            'loss': float(loss),
            'loss_lat': float(loss_dict.get('loss_lat', 0)),
            'loss_frac': float(loss_dict.get('loss_frac', 0)),
            'loss_type': float(loss_dict.get('loss_type', 0)),
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_losses = []
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            if 'gpu' in self.device and paddle.device.is_compiled_with_cuda():
                batch = self._move_batch_to_device(batch)

            # Training step
            losses = self.train_step(batch)

            # Log
            if self.is_root and batch_idx % self.config.training.get('log_freq', 10) == 0:
                self.logger.log_step(
                    self.global_step,
                    losses,
                    mode='train',
                )

            epoch_losses.append(losses['loss'])
            self.global_step += 1

        return {'train_loss': np.mean(epoch_losses)}

    @paddle.no_grad()
    def eval_epoch(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return {'val_loss': 0.0}

        epoch_losses = []
        for batch in self.val_loader:
            if 'gpu' in self.device and paddle.device.is_compiled_with_cuda():
                batch = self._move_batch_to_device(batch)

            losses = self.eval_step(batch)
            epoch_losses.append(losses['loss'])

        return {'val_loss': np.mean(epoch_losses)}

    def train(self):
        """Main training loop."""
        n_epochs = self.config.training.get('n_epochs', 100)
        eval_freq = self.config.training.get('eval_freq', 10)
        save_freq = self.config.training.get('save_freq', 10)

        for epoch in range(self.current_epoch, n_epochs):
            self.current_epoch = epoch

            if self.is_root:
                print(f"\nEpoch {epoch}/{n_epochs}", flush=True)

            # Train
            train_metrics = self.train_epoch()

            # Eval
            if self.val_loader is not None and epoch % eval_freq == 0:
                val_metrics = self.eval_epoch()
                if self.is_root:
                    print(f"Val Loss: {val_metrics['val_loss']:.6f}", flush=True)

                    # Save best model
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint('best_model')

            # Save checkpoint
            if self.is_root and epoch % save_freq == 0:
                self.save_checkpoint(f'epoch_{epoch}')

        if self.is_root:
            print("Training completed!", flush=True)

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        if 'x0' in batch and batch['x0'] is not None:
            batch['x0'] = [x.cuda() if hasattr(x, 'cuda') else x for x in batch['x0']]
        if 'batch' in batch and hasattr(batch['batch'], 'cuda'):
            batch['batch'] = batch['batch'].cuda()
        return batch

    def save_checkpoint(self, tag: str):
        """Save model checkpoint."""
        save_dir = self.config.checkpoints.get('save_dir', './checkpoints')
        os.makedirs(save_dir, exist_ok=True)

        checkpoint_path = os.path.join(save_dir, f'miad_{tag}.pdparams')
        optimizer_path = os.path.join(save_dir, f'miad_{tag}.pdopt')

        # Save model
        paddle.save(self.model.state_dict(), checkpoint_path)
        paddle.save(self.optimizer.state_dict(), optimizer_path)

        # Save EMA
        ema_path = os.path.join(save_dir, f'miad_{tag}_ema.pdparams')
        paddle.save(self.ema.state_dict(), ema_path)

        # Save training state
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }
        paddle.save(state, os.path.join(save_dir, f'miad_{tag}_state.pdparams'))

        if self.is_root:
            print(f"Checkpoint saved: {checkpoint_path}", flush=True)

    def load_checkpoint(self, tag: str):
        """Load model checkpoint."""
        save_dir = self.config.checkpoints.get('save_dir', './checkpoints')

        checkpoint_path = os.path.join(save_dir, f'miad_{tag}.pdparams')
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}", flush=True)
            return

        self.model.set_state_dict(paddle.load(checkpoint_path))

        optimizer_path = os.path.join(save_dir, f'miad_{tag}.pdopt')
        if os.path.exists(optimizer_path):
            self.optimizer.set_state_dict(paddle.load(optimizer_path))

        ema_path = os.path.join(save_dir, f'miad_{tag}_ema.pdparams')
        if os.path.exists(ema_path):
            self.ema.set_state_dict(paddle.load(ema_path))

        state_path = os.path.join(save_dir, f'miad_{tag}_state.pdparams')
        if os.path.exists(state_path):
            state = paddle.load(state_path)
            self.current_epoch = state['epoch']
            self.global_step = state['global_step']
            self.best_val_loss = state['best_val_loss']

        if self.is_root:
            print(f"Checkpoint loaded: {checkpoint_path}", flush=True)


class EMAHelper:
    """Exponential Moving Average helper for model parameters."""

    def __init__(self, model: paddle.nn.Layer, decay: float = 0.999):
        """
        Args:
            model: Paddle model.
            decay: EMA decay rate.
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.stop_gradient is False:
                self.shadow[name] = param.clone()

    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.stop_gradient is False:
                new_average = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param
                )
                self.shadow[name] = new_average

    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if param.stop_gradient is False:
                self.backup[name] = param.clone()
                param.set_value(self.shadow[name])

    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.stop_gradient is False:
                param.set_value(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, paddle.Tensor]:
        """Get state dict for saving."""
        return self.shadow.copy()

    def set_state_dict(self, state_dict: Dict[str, paddle.Tensor]):
        """Load state dict."""
        self.shadow = state_dict


class TrainingLogger:
    """Simple training logger."""

    def __init__(self, log_dir: str, log_freq: int = 10):
        """
        Args:
            log_dir: Directory to save logs.
            log_freq: Logging frequency.
        """
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.metrics_history = {
            'train': [],
            'val': [],
        }

    def log_step(self, step: int, metrics: Dict[str, float], mode: str = 'train'):
        """Log metrics for a single step."""
        metrics['step'] = step
        self.metrics_history[mode].append(metrics)

        if step % self.log_freq == 0:
            loss_str = ' | '.join([f'{k}: {v:.6f}' for k, v in metrics.items() if k != 'step'])
            print(f"Step {step} | {loss_str}", flush=True)

    def save_metrics(self, path: Optional[str] = None):
        """Save metrics history."""
        if path is None:
            path = os.path.join(self.log_dir, 'metrics.pkl')

        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.metrics_history, f)


def create_miad_trainer_from_config(config: MiADConfig) -> MiADTrainer:
    """Create MiADTrainer from config with automatic dataset setup.

    Args:
        config: MiADConfig instance.

    Returns:
        MiADTrainer instance ready for training.
    """
    from ppmat.datasets.mp20_dataset import MP20Dataset

    # Create datasets
    train_dataset = MP20Dataset(
        path=config.data.get('train_path', './data/mp_20/train.csv'),
        build_graph_cfg=None,
    )
    val_dataset = MP20Dataset(
        path=config.data.get('val_path', './data/mp_20/val.csv'),
        build_graph_cfg=None,
    )

    # Create trainer
    trainer = MiADTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    return trainer


def load_yaml_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_config_from_yaml(yaml_cfg):
    model_cfg = yaml_cfg.get('Model', {}).get('__init_params__', {}).get('model_cfg', {})
    diffusion_cfg = yaml_cfg.get('Model', {}).get('__init_params__', {}).get('diffusion_cfg', {})

    dataset_cfg = yaml_cfg.get('Dataset', {})
    train_cfg = dataset_cfg.get('train', {})
    val_cfg = dataset_cfg.get('val', {})

    trainer_cfg = yaml_cfg.get('Trainer', {})
    opt_cfg = yaml_cfg.get('Optimizer', {}).get('__init_params__', {})

    data_cfg = {
        'train_path': train_cfg.get('dataset', {}).get('__init_params__', {}).get('path', './data/mp_20/train.csv'),
        'val_path': val_cfg.get('dataset', {}).get('__init_params__', {}).get('path', './data/mp_20/val.csv'),
        'batch_size': train_cfg.get('sampler', {}).get('__init_params__', {}).get('batch_size', 4),
        'num_workers': train_cfg.get('loader', {}).get('num_workers', 0),
    }

    optim_cfg = {
        'lr': opt_cfg.get('lr', {}).get('__init_params__', {}).get('learning_rate', 0.001),
        'weight_decay': 1e-6,
        'beta1': opt_cfg.get('beta1', 0.9),
        'beta2': opt_cfg.get('beta2', 0.999),
    }

    training_cfg = {
        'n_epochs': trainer_cfg.get('max_epochs', 100),
        'eval_freq': trainer_cfg.get('eval_freq', 10),
        'save_freq': trainer_cfg.get('save_freq', 100),
        'log_freq': trainer_cfg.get('log_freq', 10),
    }

    checkpoint_cfg = {
        'save_dir': trainer_cfg.get('output_dir', './output/miad_mp20'),
    }

    config = MiADConfig(
        model=model_cfg,
        diffusion=diffusion_cfg,
        data=data_cfg,
        optimization=optim_cfg,
        training=training_cfg,
        checkpoints=checkpoint_cfg,
    )
    return config


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Train MiAD model')
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='YAML config file path')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--device', type=str, default='gpu',
                        help='Device: gpu or cpu')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory (overrides config)')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU ids, e.g. 0 or 0_1')
    args = parser.parse_args()

    paddle.seed(args.seed)
    np.random.seed(args.seed)

    if args.config:
        yaml_cfg = load_yaml_config(args.config)
        config = build_config_from_yaml(yaml_cfg)
    else:
        config = MiADConfig()

    if args.epochs is not None:
        config.training['n_epochs'] = args.epochs
    if args.batch_size is not None:
        config.data['batch_size'] = args.batch_size
    if args.lr is not None:
        config.optimization['lr'] = args.lr
    if args.checkpoint_dir is not None:
        config.checkpoints['save_dir'] = args.checkpoint_dir

    print("MiAD Training Configuration:")
    print(f"  Epochs: {config.training.get('n_epochs')}")
    print(f"  Batch size: {config.data.get('batch_size')}")
    print(f"  Learning rate: {config.optimization.get('lr')}")
    print(f"  Device: {args.device}")

    trainer = create_miad_trainer_from_config(config)
    trainer.train()


if __name__ == '__main__':
    main()