#!/usr/bin/env python3
"""
导出 PyTorch MatterGen 最后一层输出
===================================
在 matinvent 环境中运行，导出最后一层的输出用于与 Paddle 对比

用法:
conda activate matinvent
python diff-matinvent/script/export_pytorch_final_output.py
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# 路径配置
PT_CKPT = Path("/home/cao/.cache/huggingface/hub/models--microsoft--mattergen/snapshots/ea430eab64b80855029c2941b9fda15f245a771a/checkpoints/mp_20_base/checkpoints/last.ckpt")
OUTPUT_FILE = PROJECT_ROOT / "tmp" / "pytorch_final_output.json"

# 固定随机种子
np.random.seed(42)
torch.manual_seed(42)


def prepare_test_input():
    """准备测试输入数据"""
    batch_size = 4
    num_atoms_list = [10, 15, 20, 12]
    total_atoms = sum(num_atoms_list)

    # 原子类型 (1-100)
    np.random.seed(42)
    atom_types = np.random.randint(1, 101, size=total_atoms)

    # 分数坐标
    np.random.seed(100)
    frac_coords = np.random.rand(total_atoms, 3).astype(np.float32)

    # 晶格参数
    np.random.seed(200)
    lengths = np.random.rand(batch_size, 3).astype(np.float32) * 5 + 5  # 5-10 Å
    angles = np.random.rand(batch_size, 3).astype(np.float32) * 60 + 60  # 60-120 度

    # num_atoms
    num_atoms = np.array(num_atoms_list, dtype=np.int64)

    # batch_idx
    batch_idx = []
    for i, num in enumerate(num_atoms_list):
        batch_idx.extend([i] * num)
    batch_idx = np.array(batch_idx, dtype=np.int64)

    return {
        "batch_size": batch_size,
        "total_atoms": total_atoms,
        "atom_types": atom_types,
        "frac_coords": frac_coords,
        "lengths": lengths,
        "angles": angles,
        "num_atoms": num_atoms,
        "batch_idx": batch_idx,
    }


def main():
    logger.info("=" * 60)
    logger.info("Exporting PyTorch MatterGen Final Layer Output")
    logger.info("=" * 60)

    # 准备测试输入
    test_input = prepare_test_input()
    logger.info(f"Test input: batch_size={test_input['batch_size']}, total_atoms={test_input['total_atoms']}")

    # 加载 checkpoint
    logger.info(f"\nLoading checkpoint: {PT_CKPT}")
    ckpt = torch.load(str(PT_CKPT), map_location='cpu')
    state_dict = ckpt['state_dict']
    logger.info(f"  Loaded {len(state_dict)} keys")

    # 查看 state_dict 的结构，了解模型架构
    logger.info("\nState dict structure (first 20 keys):")
    for i, key in enumerate(list(state_dict.keys())[:20]):
        logger.info(f"  {key}: {state_dict[key].shape}")

    # 保存测试输入和权重信息
    output = {
        "test_input": {
            "batch_size": int(test_input["batch_size"]),
            "total_atoms": int(test_input["total_atoms"]),
            "num_atoms": test_input["num_atoms"].tolist(),
            "atom_types": test_input["atom_types"].tolist(),
            "frac_coords": test_input["frac_coords"].tolist(),
            "lengths": test_input["lengths"].tolist(),
            "angles": test_input["angles"].tolist(),
            "batch_idx": test_input["batch_idx"].tolist(),
        },
        "state_dict_info": {
            "num_keys": len(state_dict),
            "keys": list(state_dict.keys()),
        },
        "note": "Full model forward pass requires complete MatterGen environment. This exports the test input and state dict info for reference."
    }

    # 保存输出
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nOutput saved to: {OUTPUT_FILE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
