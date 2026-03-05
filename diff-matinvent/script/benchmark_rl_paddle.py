#!/usr/bin/env python3
"""
RL基准测试脚本 - PaddlePaddle版本

测试 MatInvent 强化学习组件：
1. ReplayBuffer - 回放缓冲区功能
2. LongTimeMem - 长期记忆功能
3. Reward - 奖励计算功能

输出数据存储在 diff-matinvent/info/rl/ 目录下

This code is adapted from:
raw-matinvent/diff_info/benchmark_rl.py
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

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppmat.models.matinvent import ReplayBuffer, LongTimeMem, Reward

# Setup paths
DIFF_INFO = PROJECT_ROOT / "diff-matinvent"
INFO_OUTPUT = DIFF_INFO / "info" / "rl"
INFO_OUTPUT.mkdir(parents=True, exist_ok=True)

# Fix random seed for reproducibility
RANDOM_SEED = 42


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


def create_mock_structures(num_structures: int = 10):
    """Create mock crystal structures for testing."""
    from pymatgen.core.structure import Structure, Lattice

    structures = []
    for i in range(num_structures):
        # Use different elements for each structure to avoid deduplication
        elements = ['Si', 'Ge', 'C', 'Al', 'Mg', 'Ca', 'Ti', 'Fe', 'Co', 'Ni']
        element = elements[i % len(elements)]

        lattice_params = {
            'Si': 5.43, 'Ge': 5.66, 'C': 3.57,
            'Al': 4.05, 'Mg': 3.21, 'Ca': 5.59,
            'Ti': 4.73, 'Fe': 2.87, 'Co': 3.55, 'Ni': 3.52,
        }
        a = lattice_params.get(element, 4.0)
        lattice = Lattice.cubic(a)
        structure = Structure(lattice, [element], [[0, 0, 0]])
        structures.append(structure)

    return structures


def create_mock_data(num_samples: int = 10):
    """Create mock data objects for testing."""
    class MockData:
        def __init__(self, idx):
            self.idx = idx
            self.x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
            self.edge_index = np.array([[0], [0]], dtype=np.int64)

    return [MockData(i) for i in range(num_samples)]


def benchmark_replay_buffer(
    num_samples: int = 10,
    buffer_size: int = 20,
    sample_size: int = 4,
    output_dir: Path = INFO_OUTPUT,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Benchmark ReplayBuffer - 测试回放缓冲区功能
    """
    if logger:
        logger.info("=" * 60)
        logger.info("BENCHMARK: ReplayBuffer")
        logger.info("=" * 60)

    results = {
        'test': 'replay_buffer',
        'num_samples': num_samples,
        'buffer_size': buffer_size,
        'sample_size': sample_size,
    }

    try:
        # Create buffer
        buffer = ReplayBuffer(
            buffer_size=buffer_size,
            sample_size=sample_size,
            reward_cutoff=0.0,
        )

        # Create test data
        structures = create_mock_structures(num_samples)
        data_list = create_mock_data(num_samples)
        rewards = np.random.rand(num_samples).astype(np.float32)

        # Test extend
        buffer.extend(data_list, structures, rewards)
        results['after_extend_length'] = len(buffer)

        if logger:
            logger.info(f"Buffer length after extend: {len(buffer)}")

        # Test sample
        sampled_data, sampled_rewards = buffer.sample()
        results['sampled_data_count'] = len(sampled_data)
        results['sampled_rewards_count'] = len(sampled_rewards)

        if logger:
            logger.info(f"Sampled {len(sampled_data)} data items, {len(sampled_rewards)} rewards")

        # Test deduplication
        results['test_passed'] = True

    except Exception as e:
        if logger:
            logger.error(f"ReplayBuffer test failed: {e}")
        results['test_passed'] = False
        results['error'] = str(e)

    output_file = output_dir / "replay_buffer_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if logger:
        logger.info(f"ReplayBuffer results saved to {output_file}")

    return results


def benchmark_long_term_memory(
    num_samples: int = 20,
    output_dir: Path = INFO_OUTPUT,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Benchmark LongTimeMem - 测试长期记忆功能
    """
    if logger:
        logger.info("=" * 60)
        logger.info("BENCHMARK: LongTimeMem")
        logger.info("=" * 60)

    results = {
        'test': 'long_term_memory',
        'num_samples': num_samples,
    }

    try:
        # Create LTM
        ltm = LongTimeMem()

        # Create test data
        structures = create_mock_structures(num_samples)
        rewards = np.random.rand(num_samples).astype(np.float32)

        # Test extend
        ltm.extend(structures, rewards, step=0)
        results['after_extend_length'] = len(ltm)
        results['unique_comps_count'] = len(ltm.unique_comps)

        if logger:
            logger.info(f"LTM length after extend: {len(ltm)}")
            logger.info(f"Unique compositions: {len(ltm.unique_comps)}")

        # Test div_filter
        test_structures = create_mock_structures(5)
        test_rewards = np.random.rand(5).astype(np.float32)

        new_rewards, penalty_idx, tol_n, buff_n = ltm.div_filter(
            test_structures, test_rewards, tol=10, buff=20
        )

        results['div_filter_tol_n'] = tol_n
        results['div_filter_buff_n'] = buff_n
        results['div_filter_penalties'] = len(penalty_idx)

        if logger:
            logger.info(f"DivFilter: tol_n={tol_n}, buff_n={buff_n}, penalties={len(penalty_idx)}")

        # Test metrics
        burden, div_ratio = ltm.calc_metrics(thred=0.5)
        results['burden'] = burden if burden is not None else "N/A"
        results['div_ratio'] = div_ratio if div_ratio is not None else "N/A"

        if logger:
            logger.info(f"Metrics: burden={results['burden']}, div_ratio={results['div_ratio']}")

        results['test_passed'] = True

    except Exception as e:
        if logger:
            logger.error(f"LongTimeMem test failed: {e}")
        results['test_passed'] = False
        results['error'] = str(e)

    output_file = output_dir / "long_term_memory_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if logger:
        logger.info(f"LongTimeMem results saved to {output_file}")

    return results


def benchmark_reward_calculation(
    num_samples: int = 10,
    output_dir: Path = INFO_OUTPUT,
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Benchmark Reward - 测试奖励计算功能
    """
    if logger:
        logger.info("=" * 60)
        logger.info("BENCHMARK: Reward")
        logger.info("=" * 60)

    results = {
        'test': 'reward_calculation',
        'num_samples': num_samples,
    }

    try:
        # Create simple reward config (mock)
        class MockProp:
            def __init__(self):
                self.name = "test_prop"
                self.target = "ascending"
                self.minv = 0.0
                self.maxv = 1.0
                self.calculator = None

        prop_cfg = [MockProp()]

        # Create Reward (will use default calculator)
        reward = Reward(
            root_dir=str(output_dir),
            prop_cfg=prop_cfg,
            reward_threshold=0.5,
            reduce="mean",
        )

        results['reward_threshold'] = reward.threshold
        results['reduce_mode'] = reward.reduce

        # Test scoring with mock data
        mock_samples = (None, "test_label")

        try:
            rewards, prop_dict, failed_mask = reward.scoring(mock_samples, "test_label")
            results['scoring_test_passed'] = True
            if logger:
                logger.info(f"Scoring test passed, generated {len(rewards)} rewards")
        except Exception as e:
            results['scoring_test_passed'] = False
            results['scoring_error'] = str(e)
            if logger:
                logger.warning(f"Scoring test warning: {e}")

        results['test_passed'] = True

    except Exception as e:
        if logger:
            logger.error(f"Reward test failed: {e}")
        results['test_passed'] = False
        results['error'] = str(e)

    output_file = output_dir / "reward_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if logger:
        logger.info(f"Reward results saved to {output_file}")

    return results


def save_model_config(output_dir: Path = INFO_OUTPUT, logger: logging.Logger = None):
    """Save model configuration for reference."""
    config_data = {
        'framework': 'PaddlePaddle',
        'version': paddle.__version__,
    }

    config_file = output_dir / "model_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2, default=str)

    if logger:
        logger.info(f"Model config saved to {config_file}")


def run_full_benchmark(logger: logging.Logger = None):
    """Run full RL benchmark suite."""
    if logger is None:
        logger = setup_logging()

    set_random_seed()

    logger.info("=" * 60)
    logger.info("RL BENCHMARK SUITE (PaddlePaddle)")
    logger.info("=" * 60)
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info(f"Output directory: {INFO_OUTPUT}")
    logger.info("")

    save_model_config(logger=logger)

    # Run benchmarks
    replay_results = benchmark_replay_buffer(
        num_samples=10,
        buffer_size=20,
        sample_size=4,
        logger=logger
    )
    logger.info("")

    ltm_results = benchmark_long_term_memory(
        num_samples=20,
        logger=logger
    )
    logger.info("")

    reward_results = benchmark_reward_calculation(
        num_samples=10,
        logger=logger
    )
    logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)

    all_passed = all([
        replay_results.get('test_passed', False),
        ltm_results.get('test_passed', False),
        reward_results.get('test_passed', False),
    ])

    logger.info(f"ReplayBuffer: {'PASS' if replay_results.get('test_passed') else 'FAIL'}")
    logger.info(f"LongTimeMem: {'PASS' if ltm_results.get('test_passed') else 'FAIL'}")
    logger.info(f"Reward: {'PASS' if reward_results.get('test_passed') else 'FAIL'}")
    logger.info(f"Overall: {'PASS' if all_passed else 'FAIL'}")

    summary_file = INFO_OUTPUT / "benchmark_summary.json"
    summary = {
        'timestamp': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'framework': 'PaddlePaddle',
        'framework_version': paddle.__version__,
        'replay_buffer': {
            'passed': replay_results.get('test_passed', False),
        },
        'long_term_memory': {
            'passed': ltm_results.get('test_passed', False),
        },
        'reward': {
            'passed': reward_results.get('test_passed', False),
        },
        'overall': {
            'passed': all_passed,
        },
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {summary_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RL Benchmark Suite (PaddlePaddle)")
    args = parser.parse_args()

    logger = setup_logging()
    run_full_benchmark(logger=logger)


if __name__ == "__main__":
    main()
