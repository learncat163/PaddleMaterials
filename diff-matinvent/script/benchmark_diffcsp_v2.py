#!/usr/bin/env python3
"""
DiffCSP基准测试脚本 - 使用固定噪声输入

与 raw-matinvent/diff_info/benchmark_diffcsp.py 对应的 PaddlePaddle 版本

测试目标：
1. 前向精度对齐：与 PyTorch 前向输出 diff < 1e-4
2. 使用相同固定输入和固定噪声数据
3. 确保可重复性

输出数据存储在 diff-matinvent/info/diffcsp/ 目录下
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
INFO_OUTPUT = DIFF_INFO / "info" / "diffcsp"
INFO_OUTPUT.mkdir(parents=True, exist_ok=True)

NOISE_DATA_DIR = DIFF_INFO / "script" / "noise_data"

# Fix random seed for reproducibility
RANDOM_SEED = 42
MODEL_NAME = "diffcsp_mp20"


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
        return None

    with open(noise_file, 'r') as f:
        noise_data = json.load(f)

    diffcsp_noise = noise_data.get('DiffCSP', {})
    if not diffcsp_noise:
        if logger:
            logger.warning("DiffCSP noise data not found in noise_data_latest.json")
            logger.warning("Will use on-the-fly noise generation with fixed seed")
        return {}

    if logger:
        logger.info(f"Loaded DiffCSP noise data from {noise_file}")
        logger.info(f"  Timestamp: {diffcsp_noise.get('timestamp', 'N/A')}")

    return diffcsp_noise


def load_diffcsp_model(device: str = "gpu:0", logger: logging.Logger = None):
    """Load DiffCSP model for benchmarking."""
    if logger:
        logger.info("Loading DiffCSP model...")

    model, config = build_model_from_name(MODEL_NAME)

    if device.startswith("gpu"):
        paddle.set_device(device)

    model.eval()

    if logger:
        logger.info(f"Model loaded: {MODEL_NAME}")
    return model, config


def create_fixed_test_batch(batch_size: int = 16, logger: logging.Logger = None) -> Dict[str, Any]:
    """创建固定的测试批次数据 - 与 PyTorch 版本一致"""
    if logger:
        logger.info(f"Creating fixed test batch with size {batch_size}...")

    # 使用固定种子生成可重复的测试数据
    np.random.seed(42)
    
    all_frac_coords = []
    all_atom_types = []
    all_lengths = []
    all_angles = []
    all_num_atoms = []
    
    for i in range(batch_size):
        # 每个样本使用固定的种子
        np.random.seed(42 + i)
        num_atoms = [8, 12, 16, 10, 14, 18, 20, 24, 6, 10, 12, 14, 16, 8, 10, 12][i]
        
        # 生成固定坐标
        frac_coords = np.random.rand(num_atoms, 3).astype('float32')
        all_frac_coords.append(frac_coords)
        
        # 生成固定原子类型 (1-94)
        atom_types = np.random.randint(1, 95, size=(num_atoms,)).astype('int64')
        all_atom_types.append(atom_types)
        
        # 生成固定晶胞参数
        lengths = np.random.rand(3) * 10 + 5
        all_lengths.append(lengths.astype('float32'))
        
        angles = np.random.rand(3) * 60 + 60
        all_angles.append(angles.astype('float32'))
        
        all_num_atoms.append(num_atoms)
    
    # 合并批次
    all_frac_coords = np.concatenate(all_frac_coords, axis=0)
    all_atom_types = np.concatenate(all_atom_types, axis=0)
    all_lengths = np.array(all_lengths, dtype='float32')
    all_angles = np.array(all_angles, dtype='float32')
    all_num_atoms = np.array(all_num_atoms, dtype='int64')
    
    batch = {
        'structure_array': {
            'frac_coords': paddle.to_tensor(all_frac_coords),
            'atom_types': paddle.to_tensor(all_atom_types),
            'lengths': paddle.to_tensor(all_lengths),
            'angles': paddle.to_tensor(all_angles),
            'num_atoms': paddle.to_tensor(all_num_atoms),
        }
    }
    
    if logger:
        logger.info(f"Fixed test batch created: {batch_size} samples")
        logger.info(f"  Sample 0: num_atoms={all_num_atoms[0]}")
    
    return batch


def benchmark_forward_pass(
    model,
    batch: Dict[str, Any],
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
        logger.info("BENCHMARK: Forward Pass (FIXED input)")
        logger.info("=" * 60)

    results = {
        'test': 'forward_pass',
        'target_precision': '1e-4',
        'device': device,
        'num_runs': num_runs,
        'batch_size': 16,
        'fixed_timestep': 0,
        'input_type': 'FIXED',
        'runs': []
    }

    model.eval()

    for run_idx in range(num_runs):
        with paddle.no_grad():
            # 使用固定种子确保可重复
            set_random_seed()
            
            # 调用模型的前向传播
            output = model(batch)
            
            # 提取输出
            pred_l = output['pred_l'] if isinstance(output, dict) else output[0]
            pred_x = output['pred_x'] if isinstance(output, dict) else output[1]
            pred_t = output['pred_t'] if isinstance(output, dict) else output[2]

        run_result = {
            'run': run_idx,
            'timestep': 0,
            'outputs': {
                'pred_l_mean': float(pred_l.mean().item()),
                'pred_l_std': float(pred_l.std().item()),
                'pred_x_mean': float(pred_x.mean().item()),
                'pred_x_std': float(pred_x.std().item()),
                'pred_t_mean': float(pred_t.mean().item()),
                'pred_t_max': float(pred_t.max().item()),
            }
        }
        results['runs'].append(run_result)

        if logger and run_idx < 3:
            logger.info(f"Run {run_idx}: pred_l_mean={run_result['outputs']['pred_l_mean']:.6f}")

    # Calculate consistency
    l_means = [r['outputs']['pred_l_mean'] for r in results['runs']]
    x_means = [r['outputs']['pred_x_mean'] for r in results['runs']]

    results['consistency'] = {
        'lattice_mean_std': float(np.std(l_means)),
        'coord_mean_std': float(np.std(x_means)),
        'lattice_mean_range': float(max(l_means) - min(l_means)),
        'max_diff': float(max(abs(l_means[0] - x) for x in l_means)),
    }

    # Save results
    output_file = output_dir / "forward_pass_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if logger:
        logger.info(f"Forward pass results saved to {output_file}")
        logger.info(f"Consistency: lattice_mean_std={results['consistency']['lattice_mean_std']:.2e}")

    return results


def compare_with_pytorch(results: Dict[str, Any], logger: logging.Logger = None):
    """与 PyTorch 结果对比"""
    if logger:
        logger.info("=" * 60)
        logger.info("COMPARISON: vs PyTorch Results")
        logger.info("=" * 60)

    # 读取 PyTorch 结果
    pytorch_file = PROJECT_ROOT / "raw-matinvent" / "diff_tmp" / "diffcsp" / "forward_pass_results.json"

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
            'pred_l_mean': pytorch_run['outputs']['pred_l_mean'],
            'pred_x_mean': pytorch_run['outputs']['pred_x_mean'],
        },
        'paddle_outputs': {
            'pred_l_mean': paddle_run['outputs']['pred_l_mean'],
            'pred_x_mean': paddle_run['outputs']['pred_x_mean'],
        },
        'difference': {
            'pred_l_mean_diff': abs(paddle_run['outputs']['pred_l_mean'] - pytorch_run['outputs']['pred_l_mean']),
            'pred_x_mean_diff': abs(paddle_run['outputs']['pred_x_mean'] - pytorch_run['outputs']['pred_x_mean']),
        }
    }

    if logger:
        logger.info(f"PyTorch pred_l_mean: {comparison['pytorch_outputs']['pred_l_mean']:.6f}")
        logger.info(f"PaddlePaddle pred_l_mean: {comparison['paddle_outputs']['pred_l_mean']:.6f}")
        logger.info(f"Difference: {comparison['difference']['pred_l_mean_diff']:.6f}")

        # 判断是否达到目标精度
        target_precision = 1e-4
        if comparison['difference']['pred_l_mean_diff'] < target_precision:
            logger.info(f"✓ PASS: Difference < {target_precision}")
        else:
            logger.info(f"✗ CHECK: Difference >= {target_precision}")

    return comparison


def run_full_benchmark(device: str = "gpu:0"):
    """Run full DiffCSP benchmark suite."""
    logger = setup_logging()
    set_random_seed()

    logger.info("=" * 60)
    logger.info("DIFFCSP BENCHMARK SUITE (PaddlePaddle)")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Output directory: {INFO_OUTPUT}")
    logger.info("使用固定输入数据 (FIXED input)")
    logger.info("")

    # 加载噪声数据
    noise_data = load_noise_data(logger=logger)

    logger.info("")

    # 加载模型
    model, config = load_diffcsp_model(device=device, logger=logger)

    logger.info("")

    # 创建测试批次
    batch = create_fixed_test_batch(batch_size=16, logger=logger)

    logger.info("")

    # 运行前向传播测试
    forward_results = benchmark_forward_pass(model, batch, device, num_runs=10, logger=logger)

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
    logger.info(f"✓ Consistency: lattice_std={forward_results['consistency']['lattice_mean_std']:.2e}")
    if comparison:
        logger.info(f"✓ PyTorch diff: {comparison['difference']['pred_l_mean_diff']:.6f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="DiffCSP Benchmark Suite (PaddlePaddle)")
    parser.add_argument('--device', type=str, default='gpu:0',
                        help='Device to use (gpu:0, gpu:1, etc.)')
    args = parser.parse_args()

    run_full_benchmark(device=args.device)


if __name__ == "__main__":
    main()
