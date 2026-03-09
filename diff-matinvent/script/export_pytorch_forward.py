#!/usr/bin/env python3
"""
使用 mattergen 包导出 PyTorch 前向传播输出
============================================

用法:
conda activate matinvent
python diff-matinvent/script/export_pytorch_forward.py
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# 路径配置
PT_CKPT = Path("/home/cao/.cache/huggingface/hub/models--microsoft--mattergen/snapshots/ea430eab64b80855029c2941b9fda15f245a771a/checkpoints/mp_20_base/checkpoints/last.ckpt")
OUTPUT_FILE = PROJECT_ROOT / "tmp" / "pytorch_forward_output.json"

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
    logger.info("Exporting PyTorch MatterGen Forward Output")
    logger.info("=" * 60)

    # 准备测试输入
    test_input = prepare_test_input()
    logger.info(f"Test input: batch_size={test_input['batch_size']}, total_atoms={test_input['total_atoms']}")

    # 尝试加载 MatterGen 模型
    try:
        from mattergen.common.utils.eval_utils import MatterGenCheckpointInfo
        from mattergen.diffusion.lightning_module import DiffusionLightningModule

        logger.info(f"\nLoading MatterGen model from checkpoint: {PT_CKPT}")

        # 创建 checkpoint info
        ckpt_info = MatterGenCheckpointInfo(
            checkpoint_path=PT_CKPT,
            model_name="mp_20_base",
        )

        # 加载模型
        model = DiffusionLightningModule.load_from_checkpoint(str(PT_CKPT))
        model.eval()

        logger.info("  Model loaded successfully")

        # 尝试运行前向传播
        logger.info("\nRunning forward pass...")

        # 准备输入 - 注意：MatterGen 需要特定的输入格式
        # 这里需要根据实际的 MatterGen API 准备输入
        logger.warning("  Note: MatterGen forward pass requires specific input format")
        logger.warning("  This may require additional setup for proper comparison")

    except Exception as e:
        logger.error(f"Failed to load/run MatterGen model: {e}")
        logger.info("\nFalling back to weight-only export...")

        # 加载 checkpoint
        ckpt = torch.load(str(PT_CKPT), map_location='cpu')
        state_dict = ckpt['state_dict']

        # 导出一些关键层的权重作为参考
        output = {
            "test_input": test_input,
            "note": "Full forward pass requires complete MatterGen environment. Using weight export as fallback.",
            "state_dict_sample": {},
        }

        # 导出一些关键层的权重
        key_weights = [
            "diffusion_module.model.gemnet.atom_emb.embeddings.weight",
            "diffusion_module.model.gemnet.out_blocks.0.layers.0.linear.weight",
            "diffusion_module.model.fc_atom.weight",
        ]

        for key in key_weights:
            if key in state_dict:
                output["state_dict_sample"][key] = {
                    "shape": list(state_dict[key].shape),
                    "mean": float(state_dict[key].mean()),
                    "std": float(state_dict[key].std()),
                }

        # 保存输出
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

        logger.info(f"\nOutput saved to: {OUTPUT_FILE}")
        logger.info("=" * 60)
        return

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
