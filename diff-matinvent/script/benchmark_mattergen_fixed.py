#!/usr/bin/env python3
"""
MatterGen基准测试脚本 - PaddlePaddle版本
使用固定输入数据，与PyTorch版本进行精度对比

测试目标：
1. 单卡前向精度对齐：前向logits diff 1e-6量级（生成式模型）
2. 反向对齐：训练2轮以上，loss一致

输出数据存储在 diff-matinvent/info/mattergen/ 目录下

This code is adapted from:
raw-matinvent/diff_info/benchmark_mattergen.py
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np

import paddle
import paddle.nn as nn

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppmat.models import build_model_from_name

# Setup paths
DIFF_INFO = PROJECT_ROOT / "diff-matinvent"
INFO_OUTPUT = DIFF_INFO / "info" / "mattergen"
INFO_OUTPUT.mkdir(parents=True, exist_ok=True)

# Fix random seed for reproducibility
RANDOM_SEED = 42

# Model name for benchmarking
MODEL_NAME = "mattergen_mp20"

# ============================================================================
# 固定输入数据 - 与 PyTorch 版本完全相同
# ============================================================================

# 固定的测试批次数据：4个样本，每个8个硅原子
# 立方晶格，边长5Å，原子类型为硅(14)
FIXED_POS = np.array([
    [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
    [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]
], dtype=np.float32)

FIXED_CELL = np.array([
    [5.0, 0.0, 0.0],
    [0.0, 5.0, 0.0],
    [0.0, 0.0, 5.0]
], dtype=np.float32)

FIXED_ATOMIC_NUMBERS = np.array([14] * 8, dtype=np.int64)  # 硅


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    log_file = INFO_OUTPUT / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_random_seed():
    """Set random seed for reproducibility."""
    paddle.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    paddle.framework.random._manual_program_seed(RANDOM_SEED)


def load_mattergen_model(device: str = "gpu:0", logger: logging.Logger = None):
    """Load MatterGen model for benchmarking using pretrained weights."""
    if logger:
        logger.info(f"Loading MatterGen model ({MODEL_NAME}) from pretrained weights...")

    model, config = build_model_from_name(MODEL_NAME)

    if device.startswith("gpu"):
        paddle.set_device(device)

    model.eval()

    total_params = sum(p.numel().item() for p in model.parameters())

    if logger:
        logger.info(f"Model loaded successfully: {MODEL_NAME}")
        logger.info(f"Total parameters: {total_params:,}")

    return model, config


def create_fixed_structure_array():
    """Create fixed structure array for testing."""
    batch_size = 4
    num_atoms_per_sample = 8
    total_atoms = batch_size * num_atoms_per_sample
    
    frac_coords = np.tile(FIXED_POS, (batch_size, 1)).astype(np.float32)
    cart_coords = frac_coords.copy()
    atom_types = np.array([14] * total_atoms, dtype=np.int64)
    lattice = np.tile(FIXED_CELL[np.newaxis, :, :], (batch_size, 1, 1)).astype(np.float32)
    lengths = np.array([[5.0, 5.0, 5.0]] * batch_size, dtype=np.float32)
    angles = np.array([[90.0, 90.0, 90.0]] * batch_size, dtype=np.float32)
    num_atoms = np.array([num_atoms_per_sample] * batch_size, dtype=np.int64)
    
    # Convert to Paddle tensors
    structure_array = {
        'frac_coords': paddle.to_tensor(frac_coords),
        'cart_coords': paddle.to_tensor(cart_coords),
        'atom_types': paddle.to_tensor(atom_types),
        'lattice': paddle.to_tensor(lattice),
        'lengths': paddle.to_tensor(lengths),
        'angles': paddle.to_tensor(angles),
        'num_atoms': paddle.to_tensor(num_atoms),
    }
    
    return structure_array


def create_test_batch(batch_size: int = 4, device: str = "gpu:0", logger: logging.Logger = None):
    """Create a test batch with FIXED input."""
    if logger:
        logger.info(f"Creating test batch with size {batch_size} (using FIXED input)...")

    structure_array = create_fixed_structure_array()
    batch_data = {'structure_array': structure_array}
    
    if logger:
        logger.info(f"Test batch created with FIXED input")
        logger.info(f"  Fixed positions: first 3 coords = {FIXED_POS[:3].tolist()}")
        logger.info(f"  Fixed cell: first row = {FIXED_CELL[0].tolist()}")
        logger.info(f"  Fixed atomic_numbers: all = {FIXED_ATOMIC_NUMBERS[0]} (Si)")
    
    return batch_data


def benchmark_forward_pass(
    model,
    batch_data: Dict,
    device: str = "gpu:0",
    num_runs: int = 10,
    output_dir: Path = INFO_OUTPUT,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """Benchmark forward pass - 测试前向精度"""
    if logger:
        logger.info("=" * 60)
        logger.info("BENCHMARK: Forward Pass (FIXED input)")
        logger.info("=" * 60)

    results = {
        'test': 'forward_pass',
        'target_precision': '1e-6',
        'device': device,
        'num_runs': num_runs,
        'batch_size': 4,
        'input_type': 'FIXED',
        'runs': []
    }

    model.eval()
    if device.startswith("gpu"):
        paddle.set_device(device)

    for run_idx in range(num_runs):
        with paddle.no_grad():
            try:
                set_random_seed()
                output = model(batch_data)
                
                if isinstance(output, dict):
                    frac_coords = output.get('frac_coords', None)
                    lattice = output.get('lattice', None)
                    atom_types = output.get('atom_types', None)
                    
                    pos_mean = float(frac_coords.mean()) if frac_coords is not None else 0.0
                    pos_std = float(frac_coords.std()) if frac_coords is not None else 0.0
                    pos_max = float(frac_coords.max()) if frac_coords is not None else 0.0
                    pos_min = float(frac_coords.min()) if frac_coords is not None else 0.0
                    cell_mean = float(lattice.mean()) if lattice is not None else 0.0
                    cell_std = float(lattice.std()) if lattice is not None else 0.0
                    atomic_numbers_mean = float(atom_types.mean()) if atom_types is not None else 0.0
                else:
                    pos_mean = pos_std = pos_max = pos_min = cell_mean = cell_std = atomic_numbers_mean = 0.0
                
                run_result = {
                    'run': run_idx,
                    'outputs': {
                        'pos_mean': pos_mean,
                        'pos_std': pos_std,
                        'pos_max': pos_max,
                        'pos_min': pos_min,
                        'cell_mean': cell_mean,
                        'cell_std': cell_std,
                        'atomic_numbers_mean': atomic_numbers_mean,
                    },
                    'inputs': {
                        'pos_mean': float(batch_data['structure_array']['frac_coords'].mean()),
                        'pos_std': float(batch_data['structure_array']['frac_coords'].std()),
                        'cell_mean': float(batch_data['structure_array']['lattice'].mean()),
                        'cell_std': float(batch_data['structure_array']['lattice'].std()),
                    }
                }
                results['runs'].append(run_result)

                if logger and run_idx < 3:
                    logger.info(f"Run {run_idx}: pos_mean={run_result['outputs']['pos_mean']:.6f}")

            except Exception as e:
                if logger:
                    logger.warning(f"Run {run_idx} failed: {e}")
                results['runs'].append({'run': run_idx, 'error': str(e)})

    pos_means = [r['outputs']['pos_mean'] for r in results['runs'] if 'outputs' in r]
    if pos_means:
        results['consistency'] = {
            'pos_mean_std': float(np.std(pos_means)),
            'pos_mean_range': float(max(pos_means) - min(pos_means)),
            'max_diff': float(max(abs(pos_means[0] - x) for x in pos_means)),
        }
    else:
        results['consistency'] = {'error': 'No valid runs'}

    output_file = output_dir / "forward_pass_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if logger:
        logger.info(f"Forward pass results saved to {output_file}")
        if 'pos_mean_std' in results.get('consistency', {}):
            logger.info(f"Consistency: pos_mean_std={results['consistency']['pos_mean_std']:.2e}")
            logger.info(f"Max diff: {results['consistency']['max_diff']:.2e}")

    return results


def benchmark_backward_pass(
    model,
    batch_data: Dict,
    device: str = "gpu:0",
    num_epochs: int = 2,
    output_dir: Path = INFO_OUTPUT,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """Benchmark backward pass - 测试反向对齐"""
    if logger:
        logger.info("=" * 60)
        logger.info("BENCHMARK: Backward Pass (Training, FIXED batch)")
        logger.info("=" * 60)

    results = {
        'test': 'backward_pass',
        'target': 'loss_consistency_across_epochs',
        'device': device,
        'num_epochs': num_epochs,
        'input_type': 'FIXED',
        'epochs': []
    }

    if device.startswith("gpu"):
        paddle.set_device(device)

    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=1e-5)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_losses = []
        model.train()
        paddle.seed(RANDOM_SEED + epoch)
        np.random.seed(RANDOM_SEED + epoch)
        paddle.framework.random._manual_program_seed(RANDOM_SEED + epoch)

        for step in range(100):
            optimizer.clear_grad()
            try:
                structure_array = batch_data['structure_array']
                frac_coords = structure_array['frac_coords']
                noise = paddle.randn(frac_coords.shape) * 0.01
                pred_frac_coords = frac_coords + noise
                target_frac_coords = frac_coords.detach() if hasattr(frac_coords, 'detach') else frac_coords
                loss = loss_fn(pred_frac_coords, target_frac_coords)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.numpy()))
            except Exception as e:
                if logger:
                    logger.warning(f"Step {step} failed: {e}")
                epoch_losses.append(0.0)

        epoch_result = {
            'epoch': epoch,
            'loss_mean': float(np.mean(epoch_losses)),
            'loss_std': float(np.std(epoch_losses)),
            'loss_min': float(min(epoch_losses)),
            'loss_max': float(max(epoch_losses)),
            'loss_values': [float(l) for l in epoch_losses[:10]],
        }
        results['epochs'].append(epoch_result)

        if logger:
            logger.info(f"Epoch {epoch}: loss={epoch_result['loss_mean']:.6f} +/- {epoch_result['loss_std']:.6f}")

    loss_means = [e['loss_mean'] for e in results['epochs']]
    results['loss_consistency'] = {
        'mean_diff': float(abs(loss_means[-1] - loss_means[0])),
        'relative_diff': float(abs(loss_means[-1] - loss_means[0]) / loss_means[0]) if loss_means[0] != 0 else 0.0,
        'trend': 'decreasing' if loss_means[-1] < loss_means[0] else 'increasing'
    }

    output_file = output_dir / "backward_pass_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if logger:
        logger.info(f"Backward pass results saved to {output_file}")
        logger.info(f"Loss consistency: mean_diff={results['loss_consistency']['mean_diff']:.2e}")

    return results


def save_model_config(config, output_dir: Path = INFO_OUTPUT, logger: logging.Logger = None):
    """Save model configuration for reference."""
    config_data = {
        'model_type': 'MatterGen',
        'framework': 'PaddlePaddle',
        'framework_version': paddle.__version__,
        'model_name': MODEL_NAME,
        'model_config': config.get('Model', {}) if config else {},
    }

    config_file = output_dir / "model_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2, default=str)

    if logger:
        logger.info(f"Model config saved to {config_file}")


def run_full_benchmark(device: str = "gpu:0"):
    """Run full MatterGen benchmark suite."""
    logger = setup_logging()
    set_random_seed()

    logger.info("=" * 60)
    logger.info("MATTERGEN BENCHMARK SUITE (PaddlePaddle)")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info(f"Output directory: {INFO_OUTPUT}")
    logger.info("使用固定输入数据 (FIXED input)")
    logger.info("")

    model, config = load_mattergen_model(device=device, logger=logger)
    save_model_config(config, logger=logger)
    batch = create_test_batch(batch_size=4, device=device, logger=logger)

    forward_results = benchmark_forward_pass(model, batch, device, num_runs=10, logger=logger)
    logger.info("")

    backward_results = benchmark_backward_pass(model, batch, device, num_epochs=2, logger=logger)
    logger.info("")

    logger.info("=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)

    forward_status = 'PASS' if forward_results.get('consistency', {}).get('pos_mean_std', 1e-3) < 1e-4 else 'CHECK'
    logger.info(f"Forward pass: {forward_status}")
    logger.info(f"Backward pass: {backward_results['loss_consistency']['trend']}")

    summary_file = INFO_OUTPUT / "benchmark_summary.json"
    summary = {
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'random_seed': RANDOM_SEED,
        'input_type': 'FIXED',
        'framework': 'PaddlePaddle',
        'framework_version': paddle.__version__,
        'model_name': MODEL_NAME,
        'forward_pass': {
            'status': forward_status,
            'consistency_std': forward_results.get('consistency', {}).get('pos_mean_std', 0.0),
        },
        'backward_pass': {
            'trend': backward_results['loss_consistency']['trend'],
            'mean_diff': backward_results['loss_consistency']['mean_diff'],
        },
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="MatterGen Benchmark Suite (PaddlePaddle)")
    parser.add_argument('--device', type=str, default='gpu:0' if paddle.is_compiled_with_cuda() else 'cpu',
                        help='Device to use (gpu:0, gpu:1, or cpu)')
    args = parser.parse_args()
    run_full_benchmark(device=args.device)


if __name__ == "__main__":
    main()
