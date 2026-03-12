#!/usr/bin/env python3
"""
MatterGen 晶体采样脚本
=====================

使用 Paddle 版本的 MatterGen 模型生成晶体结构。

功能：
  1. 加载 MatterGen 模型权重
  2. 生成指定数量的晶体结构
  3. 保存为 JSON 和 CIF 格式
  4. 验证生成的结构有效性

使用方法：
    python sample_crystals.py --num_samples 10 --num_atoms 6 --output_dir ./output

注意：
  - 需要在 ppmat conda 环境中运行
  - 模型权重路径需正确配置
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import paddle
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppmat.models.mattergen.mattergen import MatterGen

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 配置
# ==============================================================================

# 默认模型配置 (与训练时保持一致)
_DEFAULT_MODEL_CFG = {
    'decoder_cfg': {
        'gemnet_cfg': {
            'num_targets': 1,
            'latent_dim': 512,
            'atom_embedding_cfg': {
                'emb_size': 512,
                'with_mask_type': True,
            },
            'max_neighbors': 50,
            'max_cell_images_per_dim': 5,
            'cutoff': 7.0,
            'num_blocks': 4,
            'otf_graph': True,
        }
    },
    'lattice_noise_scheduler_cfg': {
        '__class_name__': 'LatticeVPSDEScheduler',
        'limit_density': 0.05771451654022283,
        '__init_params__': {}
    },
    'coord_noise_scheduler_cfg': {
        '__class_name__': 'NumAtomsVarianceAdjustedWrappedVESDE',
        '__init_params__': {}
    },
    'atom_noise_scheduler_cfg': {
        '__class_name__': 'D3PMScheduler',
        '__init_params__': {}
    },
    'num_train_timesteps': 1000,
    'time_dim': 256,
    'lattice_loss_weight': 1,
    'coord_loss_weight': 0.1,
    'atom_loss_weight': 1,
}

# 默认采样参数
_DEFAULT_SAMPLING_CFG = {
    'num_inference_steps': 1000,
    'batch_size': 4,
    'seed': 42,
}


# ==============================================================================
# 模型加载
# ==============================================================================

def load_model(ckpt_path: str | Path) -> MatterGen:
    """加载 MatterGen 模型

    Args:
        ckpt_path: 模型权重文件路径 (.pdparams)

    Returns:
        加载好的 MatterGen 模型
    """
    ckpt_path = Path(ckpt_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"模型权重文件不存在: {ckpt_path}")

    logger.info(f"正在加载模型: {ckpt_path}")

    model = MatterGen(**_DEFAULT_MODEL_CFG)
    model.eval()

    state_dict = paddle.load(str(ckpt_path))
    model.set_state_dict(state_dict)

    logger.info("模型加载完成")
    return model


# ==============================================================================
# 晶体采样
# ==============================================================================

def sample_crystals(
    model: MatterGen,
    num_atoms_list: List[int],
    num_inference_steps: int = 1000,
    seed: int = 42,
) -> List[Dict]:
    """生成晶体结构

    Args:
        model: MatterGen 模型
        num_atoms_list: 每个样本的原子数列表
        num_inference_steps: 推理步数
        seed: 随机种子

    Returns:
        生成的晶体结构列表
    """
    # 设置随机种子
    paddle.seed(seed)
    np.random.seed(seed)

    logger.info(f"开始采样: num_atoms={num_atoms_list}, steps={num_inference_steps}, seed={seed}")

    batch_size = len(num_atoms_list)
    batch_data = {
        "structure_array": {
            "num_atoms": paddle.to_tensor(
                np.array(num_atoms_list, dtype=np.int64)
            ),
        }
    }

    with paddle.no_grad():
        output = model.sample(
            batch_data,
            num_inference_steps=num_inference_steps,
        )

    results = output["result"]
    logger.info(f"采样完成，共生成 {len(results)} 个结构")

    for i, r in enumerate(results):
        logger.info(
            f"  样本 {i}: 原子数={r['num_atoms']}, "
            f"原子类型={r['atom_types'][:4]}..., "
            f"晶格参数={r['lattice'][0][:2]}..."
        )

    return results


# ==============================================================================
# 结构验证
# ==============================================================================

def validate_structure(struct_dict: Dict) -> tuple[bool, str]:
    """验证晶体结构的有效性

    检查项：
      1. 体积 > 0.1 A^3
      2. 最小原子间距 > 0.5 A
      3. 最大晶格参数 < 25 A

    Args:
        struct_dict: 结构字典

    Returns:
        (is_valid, message): 是否有效及原因
    """
    try:
        num_atoms = struct_dict['num_atoms']
        frac_coords = np.array(struct_dict['frac_coords'])
        atom_types = np.array(struct_dict['atom_types'])
        lattice = np.array(struct_dict['lattice'])

        # 检查体积
        volume = np.abs(np.linalg.det(lattice))
        if volume <= 0.1:
            return False, f"体积过小: {volume:.4f} A^3"

        # 转换为 pymatgen Structure 进行更详细检查
        from pymatgen.core.structure import Structure
        from pymatgen.core.lattice import Lattice

        structure = Structure(
            lattice=Lattice(lattice),
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False,
        )

        # 检查晶格参数
        max_lattice_param = max(structure.lattice.abc)
        if max_lattice_param > 25.0:
            return False, f"晶格参数过大: {max_lattice_param:.2f} A"

        # 检查最小原子间距
        dmat = structure.distance_matrix.copy()
        np.fill_diagonal(dmat, np.inf)
        min_distance = dmat.min()
        if min_distance < 0.5:
            return False, f"最小原子间距过小: {min_distance:.4f} A"

        return True, "结构有效"

    except Exception as e:
        return False, f"验证失败: {str(e)}"


# ==============================================================================
# 结果保存
# ==============================================================================

def save_results(
    results: List[Dict],
    output_dir: str | Path,
    prefix: str = "sample",
):
    """保存采样结果

    保存格式：
      1. JSON: 原始数据
      2. CIF: 晶体结构文件

    Args:
        results: 采样结果列表
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 JSON
    json_path = output_dir / f"{prefix}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"JSON 文件已保存: {json_path}")

    # 保存 CIF 文件
    valid_count = 0
    for i, result in enumerate(results):
        is_valid, msg = validate_structure(result)

        if not is_valid:
            logger.warning(f"  样本 {i} 无效: {msg}")
            continue

        valid_count += 1

        # 创建 pymatgen Structure
        from pymatgen.core.structure import Structure
        from pymatgen.core.lattice import Lattice

        structure = Structure(
            lattice=Lattice(result['lattice']),
            species=result['atom_types'],
            coords=result['frac_coords'],
            coords_are_cartesian=False,
        )

        # 保存 CIF
        cif_path = output_dir / f"{prefix}_{i:04d}.cif"
        CifWriter(structure).write_file(cif_path)

    logger.info(f"CIF 文件已保存: {valid_count}/{len(results)} 个有效结构")


# ==============================================================================
# 主函数
# ==============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="MatterGen 晶体采样脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 模型参数
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='模型权重文件路径 (.pdparams)'
    )

    # 采样参数
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='生成的样本数量'
    )
    parser.add_argument(
        '--num_atoms',
        type=int,
        nargs='+',
        default=[6, 8, 10],
        help='每个样本的原子数（可指定多个）'
    )
    parser.add_argument(
        '--num_inference_steps',
        type=int,
        default=1000,
        help='推理步数'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='批处理大小'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )

    # 输出参数
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='输出目录'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default=f'sample_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        help='输出文件名前缀'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("MatterGen 晶体采样")
    logger.info("=" * 60)

    # 打印配置
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"样本数量: {args.num_samples}")
    logger.info(f"原子数列表: {args.num_atoms}")
    logger.info(f"推理步数: {args.num_inference_steps}")
    logger.info(f"批处理大小: {args.batch_size}")
    logger.info(f"随机种子: {args.seed}")
    logger.info(f"输出目录: {args.output_dir}")

    # 加载模型
    model = load_model(args.model_path)

    # 生成原子数列表
    num_atoms_list = []
    for _ in range(args.num_samples):
        num_atoms_list.extend(args.num_atoms)
        if len(num_atoms_list) >= args.num_samples:
            break
    num_atoms_list = num_atoms_list[:args.num_samples]

    # 分批采样
    all_results = []
    num_batches = (len(num_atoms_list) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(num_atoms_list))
        batch_num_atoms = num_atoms_list[start_idx:end_idx]

        logger.info(f"\n批次 {batch_idx + 1}/{num_batches}")

        results = sample_crystals(
            model,
            batch_num_atoms,
            args.num_inference_steps,
            args.seed + batch_idx,
        )

        all_results.extend(results)

    # 保存结果
    logger.info("\n保存结果...")
    save_results(all_results, args.output_dir, args.prefix)

    # 统计有效结构
    valid_count = sum(1 for r in all_results if validate_structure(r)[0])
    logger.info(f"\n完成! 共生成 {len(all_results)} 个结构，其中 {valid_count} 个有效")
    logger.info(f"结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
