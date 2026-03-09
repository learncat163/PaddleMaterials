#!/usr/bin/env python3
"""
MatterGen采样功能测试
验证生成晶体结构的质量
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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppmat.models import build_model_from_name

DIFF_INFO = PROJECT_ROOT / "diff-matinvent"
INFO_OUTPUT = DIFF_INFO / "info" / "mattergen"
INFO_OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MODEL_NAME = "mattergen_mp20"


def setup_logging():
    log_file = INFO_OUTPUT / f"sampling_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


def create_sampling_batches(batch_size=4, num_batches=2):
    """创建采样批次"""
    batches = []
    
    for batch_idx in range(num_batches):
        # 生成随机原子数量
        np.random.seed(RANDOM_SEED + batch_idx)
        num_atoms_list = np.random.randint(4, 20, size=batch_size).tolist()
        
        # 创建batch数据
        batch_data = {
            'structure_array': {
                'num_atoms': paddle.to_tensor(num_atoms_list, dtype='int64'),
            }
        }
        batches.append(batch_data)
    
    return batches


def analyze_sample(sample_data, logger=None):
    """分析生成的样本"""
    frac_coords = sample_data['frac_coords']
    lattice = sample_data['lattice']
    atom_types = sample_data['atom_types']
    num_atoms = sample_data['num_atoms']
    
    # 计算晶胞体积
    lattice_np = lattice.numpy()
    volume = abs(np.linalg.det(lattice_np))
    
    # 计算统计信息
    stats = {
        'num_atoms': int(num_atoms.item()),
        'cell_volume': float(volume),
        'properties': {
            'pos_mean': float(frac_coords.mean().item()),
            'pos_std': float(frac_coords.std().item()),
            'cell_mean': float(lattice.mean().item()),
            'cell_std': float(lattice.std().item()),
        }
    }
    
    # 获取原子类型
    atom_types_np = atom_types.numpy()
    unique_elements = np.unique(atom_types_np)
    element_counts = {}
    for elem in unique_elements:
        element_counts[int(elem)] = int(np.sum(atom_types_np == elem))
    
    # 生成化学式
    composition = get_composition_string(element_counts)
    stats['composition'] = composition
    
    if logger:
        logger.info(f"  Sample: {composition}, n_atoms={stats['num_atoms']}, volume={volume:.2f}")
    
    return stats


def get_composition_string(element_counts):
    """生成化学式字符串"""
    # 元素周期表前94个元素
    elements = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
        11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
        21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
        31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr',
        41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
        51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
        61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
        71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
        81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
        91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu'
    }
    
    # 按元素序号排序
    sorted_elements = sorted(element_counts.items())
    composition_parts = []
    for elem_num, count in sorted_elements:
        elem_symbol = elements.get(elem_num, f'X{elem_num}')
        composition_parts.append(f"{elem_symbol}{count}")
    
    return ' '.join(composition_parts)


def test_sampling(model, batch_size=4, num_batches=2, num_inference_steps=10, logger=None):
    """测试采样功能"""
    if logger:
        logger.info("=" * 60)
        logger.info("TEST: MatterGen Sampling")
        logger.info("=" * 60)
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Num batches: {num_batches}")
        logger.info(f"Num inference steps: {num_inference_steps}")
    
    results = {
        'test': 'sampling',
        'batch_size': batch_size,
        'num_batches': num_batches,
        'num_inference_steps': num_inference_steps,
        'samples': []
    }
    
    batches = create_sampling_batches(batch_size, num_batches)
    sample_id = 0
    
    for batch_idx, batch_data in enumerate(batches):
        if logger:
            logger.info(f"\nBatch {batch_idx + 1}/{num_batches}")
        
        set_random_seed()
        
        # 采样
        try:
            with paddle.no_grad():
                samples = model.sample(
                    batch_data,
                    num_inference_steps=num_inference_steps,
                    _eps_t=0.001,
                )
            
            # 处理每个样本
            sample_results = samples['result']
            
            for i, sample_result in enumerate(sample_results):
                sample_data = {
                    'frac_coords': paddle.to_tensor(sample_result['frac_coords'], dtype='float32'),
                    'lattice': paddle.to_tensor(sample_result['lattice'], dtype='float32'),
                    'atom_types': paddle.to_tensor(sample_result['atom_types'], dtype='int64'),
                    'num_atoms': paddle.to_tensor([sample_result['num_atoms']], dtype='int64'),
                }
                
                sample_stats = analyze_sample(sample_data, logger=logger)
                sample_stats['sample_id'] = sample_id
                sample_stats['batch_id'] = batch_idx
                results['samples'].append(sample_stats)
                sample_id += 1
        
        except Exception as e:
            if logger:
                logger.error(f"Sampling failed for batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # 计算统计信息
    if results['samples']:
        num_atoms_list = [s['num_atoms'] for s in results['samples']]
        volume_list = [s['cell_volume'] for s in results['samples']]
        
        results['statistics'] = {
            'num_samples': len(results['samples']),
            'num_atoms_mean': float(np.mean(num_atoms_list)),
            'num_atoms_std': float(np.std(num_atoms_list)),
            'volume_mean': float(np.mean(volume_list)),
            'volume_std': float(np.std(volume_list)),
        }
    
    return results


def compare_with_pytorch(results, logger=None):
    """与PyTorch采样结果对比"""
    if logger:
        logger.info("=" * 60)
        logger.info("COMPARISON: vs PyTorch Sampling")
        logger.info("=" * 60)
    
    # 读取PyTorch结果
    pytorch_file = PROJECT_ROOT / "raw-matinvent" / "diff_tmp" / "mattergen" / "sampling_results.json"
    
    if not pytorch_file.exists():
        if logger:
            logger.warning(f"PyTorch sampling results not found at {pytorch_file}")
        return None
    
    with open(pytorch_file, 'r') as f:
        pytorch_data = json.load(f)
    
    pytorch_stats = pytorch_data['statistics']
    paddle_stats = results['statistics']
    
    comparison = {
        'pytorch': pytorch_stats,
        'paddle': paddle_stats,
        'difference': {
            'num_atoms_mean_diff': abs(paddle_stats['num_atoms_mean'] - pytorch_stats['num_atoms_mean']),
            'volume_mean_diff': abs(paddle_stats['volume_mean'] - pytorch_stats['volume_mean']),
        }
    }
    
    if logger:
        logger.info(f"PyTorch num_atoms_mean: {pytorch_stats['num_atoms_mean']:.2f}")
        logger.info(f"PaddlePaddle num_atoms_mean: {paddle_stats['num_atoms_mean']:.2f}")
        logger.info(f"Difference: {comparison['difference']['num_atoms_mean_diff']:.2f}")
        logger.info(f"\nPyTorch volume_mean: {pytorch_stats['volume_mean']:.2f}")
        logger.info(f"PaddlePaddle volume_mean: {paddle_stats['volume_mean']:.2f}")
        logger.info(f"Difference: {comparison['difference']['volume_mean_diff']:.2f}")
    
    return comparison


def run_test(device="gpu:0", num_inference_steps=10):
    logger = setup_logging()
    set_random_seed()
    
    logger.info("=" * 60)
    logger.info("MATTERGEN SAMPLING TEST")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info(f"Num inference steps: {num_inference_steps}")
    
    # 加载模型
    model, config = load_model(device=device, logger=logger)
    
    logger.info("")
    
    # 运行采样测试
    results = test_sampling(
        model,
        batch_size=4,
        num_batches=2,
        num_inference_steps=num_inference_steps,
        logger=logger
    )
    
    logger.info("")
    
    # 保存结果
    output_file = INFO_OUTPUT / "sampling_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")
    
    logger.info("")
    
    # 与PyTorch对比
    comparison = compare_with_pytorch(results, logger=logger)
    if comparison:
        comparison_file = INFO_OUTPUT / "sampling_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Comparison saved to {comparison_file}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("SAMPLING TEST COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✓ Generated {results['statistics']['num_samples']} samples")
    logger.info(f"✓ Average num_atoms: {results['statistics']['num_atoms_mean']:.2f}")
    logger.info(f"✓ Average volume: {results['statistics']['volume_mean']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='gpu:0')
    parser.add_argument('--num_inference_steps', type=int, default=10,
                       help='Number of inference steps (default: 10 for quick test)')
    args = parser.parse_args()
    run_test(device=args.device, num_inference_steps=args.num_inference_steps)


if __name__ == "__main__":
    main()
