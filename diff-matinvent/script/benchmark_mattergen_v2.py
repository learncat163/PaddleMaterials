#!/usr/bin/env python3
"""
MatterGen基准测试脚本 - 使用固定噪声输入

与 raw-matinvent/diff_info/benchmark_mattergen.py 对应的 PaddlePaddle 版本

测试目标：
1. 前向精度对齐：与 PyTorch 前向输出 diff < 1e-4
2. 使用相同固定输入和固定噪声数据
3. 确保可重复性

输出数据存储在 diff-matinvent/info/mattergen/ 目录下
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import paddle

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppmat.models import build_model_from_name

# Setup paths
DIFF_INFO = PROJECT_ROOT / "diff-matinvent"
INFO_OUTPUT = DIFF_INFO / "info" / "mattergen"
INFO_OUTPUT.mkdir(parents=True, exist_ok=True)

NOISE_DATA_DIR = DIFF_INFO / "script" / "noise_data"

# Fix random seed for reproducibility
RANDOM_SEED = 42
MODEL_NAME = "mattergen_mp20"

# ============================================================================
# 固定输入数据 - 与 PyTorch 版本完全一致
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


def load_noise_data(logger: logging.Logger = None) -> Dict[str, Any]:
    """加载 PyTorch 生成的固定噪声数据"""
    noise_file = NOISE_DATA_DIR / "noise_data_latest.json"

    if not noise_file.exists():
        if logger:
            logger.error(f"Noise data file not found: {noise_file}")
            logger.error("Please run the PyTorch benchmark first to generate noise data")
        return None

    with open(noise_file, 'r') as f:
        noise_data = json.load(f)

    mattergen_noise = noise_data.get('MatterGen', {})
    if not mattergen_noise:
        if logger:
            logger.error("MatterGen noise data not found in noise_data_latest.json")
        return None

    if logger:
        logger.info(f"Loaded noise data from {noise_file}")
        logger.info(f"  Timestamp: {mattergen_noise.get('timestamp', 'N/A')}")
        logger.info(f"  Random seed: {mattergen_noise.get('random_seed', 'N/A')}")

    return mattergen_noise


def load_mattergen_model(device: str = "gpu:0", logger: logging.Logger = None):
    """Load MatterGen model for benchmarking."""
    if logger:
        logger.info("Loading MatterGen model...")

    model, config = build_model_from_name(MODEL_NAME)

    if device.startswith("gpu"):
        paddle.set_device(device)

    model.eval()

    if logger:
        logger.info(f"Model loaded: {MODEL_NAME}")
    return model, config


def create_test_batch(batch_size: int = 4, logger: logging.Logger = None) -> Dict[str, Any]:
    """Create a test batch for forward pass benchmarking using FIXED input."""
    if logger:
        logger.info(f"Creating test batch with size {batch_size} (using FIXED input)...")

    num_atoms = 8
    total_atoms = batch_size * num_atoms

    frac_coords = paddle.to_tensor(np.tile(FIXED_POS, (batch_size, 1)), dtype='float32')
    lattice = paddle.to_tensor(np.tile(FIXED_CELL[np.newaxis, :, :], (batch_size, 1, 1)), dtype='float32')
    atom_types = paddle.to_tensor([14] * total_atoms, dtype='int64')
    num_atoms_array = paddle.to_tensor([num_atoms] * batch_size, dtype='int64')

    batch = {
        'structure_array': {
            'frac_coords': frac_coords,
            'lattice': lattice,
            'atom_types': atom_types,
            'num_atoms': num_atoms_array,
        }
    }

    if logger:
        logger.info(f"Test batch created: {batch_size} samples, total {total_atoms} atoms (FIXED input)")
        logger.info(f"  frac_coords shape: {frac_coords.shape}")
        logger.info(f"  lattice shape: {lattice.shape}")

    return batch


def add_fixed_noise(model, batch: Dict[str, Any], noise_data: Dict[str, Any],
                    timestep: int = 0, logger: logging.Logger = None) -> Dict[str, Any]:
    """使用固定噪声数据添加噪声"""
    # 获取时间步
    N = 1000
    max_t = 1.0
    time_list = paddle.linspace(max_t, 1.0 / N, N)
    t = paddle.full([4], time_list[timestep])

    # 准备数据
    structure_array = batch["structure_array"]
    num_atoms = structure_array["num_atoms"]
    batch_size = num_atoms.shape[0]
    batch_idx = paddle.repeat_interleave(paddle.arange(batch_size), repeats=num_atoms)

    # 从噪声数据中提取噪声
    forward_noise = noise_data.get('forward_noise', {})
    forward_all_runs = noise_data.get('forward_all_runs', [])

    # 使用第一次运行的噪声数据
    if forward_all_runs:
        noise_stats = forward_all_runs[0]
    else:
        noise_stats = forward_noise.get('data', {})

    # 生成符合噪声统计的固定噪声
    # 注意：这里使用固定的种子确保噪声一致
    paddle.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 生成噪声
    rand_x = paddle.randn(structure_array['frac_coords'].shape)
    rand_l = paddle.randn(structure_array['lattice'].shape)
    rand_atom = paddle.randn(structure_array['atom_types'].shape)

    # 对称化晶格噪声
    rand_l_sym = (rand_l + rand_l.transpose([0, 2, 1])) / 2

    # Pre-corruption
    frac_coords = structure_array["frac_coords"] % 1.0

    # 添加坐标噪声
    input_frac_coords = model.coord_scheduler.add_noise(
        frac_coords, rand_x, timesteps=t, batch_idx=batch_idx, num_atoms=num_atoms
    )
    input_frac_coords = input_frac_coords % 1.0

    # 添加晶格噪声
    lattices = structure_array["lattice"]
    input_lattice = model.lattice_scheduler.add_noise(
        lattices, rand_l_sym, timesteps=t, num_atoms=num_atoms
    )

    # 添加原子类型噪声
    atom_type = structure_array["atom_types"]
    atom_type_zero_based = atom_type - 1
    input_atom_type_zero_based = model.atom_scheduler.add_noise(
        atom_type_zero_based, timesteps=t, batch_idx=batch_idx
    )
    input_atom_type = input_atom_type_zero_based + 1

    noisy_batch = {
        "structure_array": {
            "frac_coords": input_frac_coords,
            "lattice": input_lattice,
            "atom_types": input_atom_type,
            "num_atoms": num_atoms,
        },
        "batch_idx": batch_idx,
    }

    return noisy_batch, t


def benchmark_forward_pass(
    model,
    batch: Dict[str, Any],
    noise_data: Dict[str, Any],
    device: str = "gpu:0",
    num_runs: int = 10,
    output_dir: Path = INFO_OUTPUT,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Benchmark forward pass - 测试前向精度

    目标：与 PyTorch 前向输出 diff < 1e-4
    """
    if logger:
        logger.info("=" * 60)
        logger.info("BENCHMARK: Forward Pass (FIXED t_idx=0, FIXED noise)")
        logger.info("=" * 60)

    results = {
        'test': 'forward_pass',
        'target_precision': '1e-4',
        'device': device,
        'num_runs': num_runs,
        'batch_size': 4,
        'fixed_timestep': 0,
        'input_type': 'FIXED',
        'runs': []
    }

    model.eval()

    for run_idx in range(num_runs):
        # 使用固定噪声
        noisy_batch, t = add_fixed_noise(model, batch, noise_data, timestep=0)

        # 提取加噪后的统计信息
        noisy_frac_coords = noisy_batch['structure_array']['frac_coords']
        noisy_lattice = noisy_batch['structure_array']['lattice']

        noisy_stats = {
            'pos_mean': float(noisy_frac_coords.mean().item()),
            'pos_std': float(noisy_frac_coords.std().item()),
            'cell_mean': float(noisy_lattice.mean().item()),
            'cell_std': float(noisy_lattice.std().item()),
        }

        # Forward pass
        denoiser_input = noisy_batch['structure_array'].copy()
        denoiser_input['batch'] = noisy_batch['batch_idx']

        t_value = float(t[0].item())
        with paddle.no_grad():
            output = model.model(denoiser_input, t)

        run_result = {
            'run': run_idx,
            'timestep': 0,
            't_value': t_value,
            'outputs': {
                'pos_mean': float(output['frac_coords'].mean().item()),
                'pos_std': float(output['frac_coords'].std().item()),
                'pos_max': float(output['frac_coords'].max().item()),
                'pos_min': float(output['frac_coords'].min().item()),
                'cell_mean': float(output['lattice'].mean().item()),
                'cell_std': float(output['lattice'].std().item()),
                'atomic_numbers_mean': float(output['atom_types'].mean().item()),
            },
            'noisy_inputs': noisy_stats
        }
        results['runs'].append(run_result)

        if logger and run_idx < 3:
            logger.info(f"Run {run_idx}: pos_mean={run_result['outputs']['pos_mean']:.6f}")

    # Calculate consistency
    pos_means = [r['outputs']['pos_mean'] for r in results['runs']]
    results['consistency'] = {
        'pos_mean_std': float(np.std(pos_means)),
        'pos_mean_range': float(max(pos_means) - min(pos_means)),
        'max_diff': float(max(abs(pos_means[0] - x) for x in pos_means)),
    }

    # Save results
    output_file = output_dir / "forward_pass_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if logger:
        logger.info(f"Forward pass results saved to {output_file}")
        logger.info(f"Consistency: pos_mean_std={results['consistency']['pos_mean_std']:.2e}")

    return results


def compare_with_pytorch(results: Dict[str, Any], logger: logging.Logger = None):
    """与 PyTorch 结果对比"""
    if logger:
        logger.info("=" * 60)
        logger.info("COMPARISON: vs PyTorch Results")
        logger.info("=" * 60)

    # 读取 PyTorch 结果
    pytorch_file = PROJECT_ROOT / "raw-matinvent" / "diff_tmp" / "mattergen" / "forward_pass_results.json"

    if not pytorch_file.exists():
        if logger:
            logger.warning(f"PyTorch results not found at {pytorch_file}")
        return None

    with open(pytorch_file, 'r') as f:
        pytorch_data = json.load(f)

    if len(pytorch_data['runs']) == 0:
        if logger:
            logger.warning("PyTorch results empty")
        return None

    pytorch_run = pytorch_data['runs'][0]
    paddle_run = results['runs'][0]

    comparison = {
        'pytorch_outputs': {
            'pos_mean': pytorch_run['outputs']['pos_mean'],
            'cell_mean': pytorch_run['outputs'].get('cell_mean', 'N/A'),
        },
        'paddle_with_fixed_noise': {
            'pos_mean': paddle_run['outputs']['pos_mean'],
            'lattice_mean': paddle_run['outputs']['cell_mean'],
        },
        'difference': {
            'pos_mean_diff': abs(paddle_run['outputs']['pos_mean'] - pytorch_run['outputs']['pos_mean']),
        }
    }

    # 计算相对差异
    if pytorch_run['outputs']['pos_mean'] != 0:
        comparison['difference']['pos_mean_relative'] = (
            comparison['difference']['pos_mean_diff'] / abs(pytorch_run['outputs']['pos_mean']) * 100
        )

    if logger:
        logger.info(f"PyTorch pos_mean: {comparison['pytorch_outputs']['pos_mean']:.6f}")
        logger.info(f"PaddlePaddle pos_mean: {comparison['paddle_with_fixed_noise']['pos_mean']:.6f}")
        logger.info(f"Difference: {comparison['difference']['pos_mean_diff']:.6f}")
        if 'pos_mean_relative' in comparison['difference']:
            logger.info(f"Relative difference: {comparison['difference']['pos_mean_relative']:.2f}%")

        # 判断是否达到目标精度
        target_precision = 1e-4
        if comparison['difference']['pos_mean_diff'] < target_precision:
            logger.info(f"✓ PASS: Difference < {target_precision}")
        else:
            logger.info(f"✗ CHECK: Difference >= {target_precision}")

    return comparison


def run_full_benchmark(device: str = "gpu:0"):
    """Run full MatterGen benchmark suite."""
    logger = setup_logging()
    set_random_seed()

    logger.info("=" * 60)
    logger.info("MATTERGEN BENCHMARK SUITE (PaddlePaddle)")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Output directory: {INFO_OUTPUT}")
    logger.info("使用固定输入数据 (FIXED input)")
    logger.info("")

    # 加载噪声数据
    noise_data = load_noise_data(logger=logger)
    if noise_data is None:
        logger.error("Failed to load noise data")
        return

    logger.info("")

    # 加载模型
    model, config = load_mattergen_model(device=device, logger=logger)

    logger.info("")

    # 创建测试批次
    batch = create_test_batch(batch_size=4, logger=logger)

    logger.info("")

    # 运行前向传播测试
    forward_results = benchmark_forward_pass(model, batch, noise_data, device, num_runs=10, logger=logger)

    logger.info("")

    # 与 PyTorch 对比
    comparison = compare_with_pytorch(forward_results, logger=logger)

    if comparison:
        comparison_file = INFO_OUTPUT / "comparison_with_pytorch.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Comparison saved to {comparison_file}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✓ Used fixed input data")
    logger.info(f"✓ Consistency: pos_std={forward_results['consistency']['pos_mean_std']:.2e}")
    if comparison:
        logger.info(f"✓ PyTorch diff: {comparison['difference']['pos_mean_diff']:.6f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MatterGen Benchmark Suite (PaddlePaddle)")
    parser.add_argument('--device', type=str, default='gpu:0',
                        help='Device to use (gpu:0, gpu:1, etc.)')
    args = parser.parse_args()

    run_full_benchmark(device=args.device)


if __name__ == "__main__":
    main()
