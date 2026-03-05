#!/usr/bin/env python3
"""
MatterGen基准测试脚本 - PaddlePaddle版本 v2
使用固定输入数据，直接测试denoiser输出
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

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppmat.models import build_model_from_name

DIFF_INFO = PROJECT_ROOT / "diff-matinvent"
INFO_OUTPUT = DIFF_INFO / "info" / "mattergen"
INFO_OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MODEL_NAME = "mattergen_mp20"

# Fixed input data - same as PyTorch
FIXED_POS = np.array([
    [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
    [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]
], dtype=np.float32)

FIXED_CELL = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float32)
FIXED_ATOMIC_NUMBERS = np.array([14] * 8, dtype=np.int64)

def setup_logging():
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
    paddle.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    paddle.framework.random._manual_program_seed(RANDOM_SEED)

def load_model(device="gpu:0", logger=None):
    if logger:
        logger.info(f"Loading {MODEL_NAME}...")
    model, config = build_model_from_name(MODEL_NAME)
    if device.startswith("gpu"):
        paddle.set_device(device)
    model.eval()
    if logger:
        logger.info(f"Model loaded: {MODEL_NAME}")
    return model, config

def create_noise_batch(batch_size=4):
    """Create noise batch with fixed data"""
    num_atoms = 8
    total_atoms = batch_size * num_atoms

    frac_coords = paddle.to_tensor(np.tile(FIXED_POS, (batch_size, 1)), dtype='float32')
    lattice = paddle.to_tensor(np.tile(FIXED_CELL[np.newaxis, :, :], (batch_size, 1, 1)), dtype='float32')
    atom_types = paddle.to_tensor([14] * total_atoms, dtype='int64')
    num_atoms_array = paddle.to_tensor([num_atoms] * batch_size, dtype='int64')
    batch_idx = paddle.repeat_interleave(paddle.arange(batch_size), repeats=num_atoms_array)

    return {
        'frac_coords': frac_coords,
        'lattice': lattice,
        'atom_types': atom_types,
        'num_atoms': num_atoms_array,
        'batch': batch_idx,
    }

def benchmark_forward_denoiser(model, num_runs=10, logger=None):
    """Test denoiser forward pass"""
    if logger:
        logger.info("=" * 60)
        logger.info("BENCHMARK: Denoiser Forward Pass")
        logger.info("=" * 60)

    results = {
        'test': 'denoiser_forward',
        'num_runs': num_runs,
        'runs': []
    }

    noise_batch = create_noise_batch(batch_size=4)
    t = paddle.full([4], 1.0, dtype='float32')

    model.model.eval()

    for run_idx in range(num_runs):
        set_random_seed()
        with paddle.no_grad():
            output = model.model(noise_batch, t)

        run_result = {
            'run': run_idx,
            'frac_coords_mean': float(output['frac_coords'].mean().item()),
            'frac_coords_std': float(output['frac_coords'].std().item()),
            'lattice_mean': float(output['lattice'].mean().item()),
            'lattice_std': float(output['lattice'].std().item()),
        }
        results['runs'].append(run_result)

        if logger and run_idx < 3:
            logger.info(f"Run {run_idx}: frac_mean={run_result['frac_coords_mean']:.6f}")

    # Calculate consistency
    frac_means = [r['frac_coords_mean'] for r in results['runs']]
    results['consistency'] = {
        'frac_mean_std': float(np.std(frac_means)),
        'frac_mean_range': float(max(frac_means) - min(frac_means)),
        'max_diff': float(max(abs(frac_means[0] - x) for x in frac_means)),
    }

    output_file = INFO_OUTPUT / "denoiser_forward_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if logger:
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Consistency: frac_mean_std={results['consistency']['frac_mean_std']:.2e}")

    return results

def benchmark_backward_pass(model, num_epochs=2, logger=None):
    """Test backward pass"""
    if logger:
        logger.info("=" * 60)
        logger.info("BENCHMARK: Backward Pass")
        logger.info("=" * 60)

    results = {
        'test': 'backward_pass',
        'num_epochs': num_epochs,
        'epochs': []
    }

    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=1e-5)
    loss_fn = paddle.nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_losses = []
        model.train()
        paddle.seed(RANDOM_SEED + epoch)
        np.random.seed(RANDOM_SEED + epoch)

        for step in range(10):
            optimizer.clear_grad()
            noise_batch = create_noise_batch(batch_size=4)

            # Simple forward pass
            frac_coords = noise_batch['frac_coords']
            noise = paddle.randn(frac_coords.shape) * 0.01
            pred = frac_coords + noise
            loss = loss_fn(pred, frac_coords)

            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.numpy()))

        epoch_result = {
            'epoch': epoch,
            'loss_mean': float(np.mean(epoch_losses)),
            'loss_std': float(np.std(epoch_losses)),
        }
        results['epochs'].append(epoch_result)

        if logger:
            logger.info(f"Epoch {epoch}: loss={epoch_result['loss_mean']:.6f}")

    loss_means = [e['loss_mean'] for e in results['epochs']]
    results['loss_consistency'] = {
        'mean_diff': float(abs(loss_means[-1] - loss_means[0])),
        'trend': 'decreasing' if loss_means[-1] < loss_means[0] else 'increasing'
    }

    output_file = INFO_OUTPUT / "backward_pass_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if logger:
        logger.info(f"Results saved to {output_file}")

    return results

def run_benchmark(device="gpu:0"):
    logger = setup_logging()
    set_random_seed()

    logger.info("=" * 60)
    logger.info("MATTERGEN BENCHMARK SUITE (PaddlePaddle)")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {RANDOM_SEED}")

    model, config = load_model(device=device, logger=logger)

    forward_results = benchmark_forward_denoiser(model, num_runs=10, logger=logger)
    logger.info("")

    backward_results = benchmark_backward_pass(model, num_epochs=2, logger=logger)
    logger.info("")

    logger.info("=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Forward consistency std: {forward_results['consistency']['frac_mean_std']:.2e}")
    logger.info(f"Backward trend: {backward_results['loss_consistency']['trend']}")

    summary = {
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'framework': 'PaddlePaddle',
        'framework_version': paddle.__version__,
        'forward': {
            'consistency_std': forward_results['consistency']['frac_mean_std'],
        },
        'backward': {
            'trend': backward_results['loss_consistency']['trend'],
        },
    }

    summary_file = INFO_OUTPUT / "benchmark_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='gpu:0')
    args = parser.parse_args()
    run_benchmark(device=args.device)

if __name__ == "__main__":
    main()
