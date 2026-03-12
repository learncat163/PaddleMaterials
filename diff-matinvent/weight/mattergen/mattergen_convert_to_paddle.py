#!/usr/bin/env python3
"""
MatInvent 内部的 MatterGen 权重转换脚本
================================
将 PyTorch checkpoint 转换为 Paddle pdparams 格式

特别注意：

1. MatInvent 使用了 huggingface 上的 mattergen 模型，但做了一定的定制
2. 当前转换脚本，必须放在 ppmat项目的py环境里执行，因为导出模型需要用到 ppmat的 MatterGen
3. 权重映射规则：
   - PyTorch 前缀: diffusion_module.model.xxx
   - Paddle 前缀: model.xxx
   - Linear 层需要转置: PyTorch [out, in] -> Paddle [in, out]
   - Embedding 层不转置
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import paddle

# 根据当前脚本的实际位置选择
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# huggingface 缓存的 mattergen 权重
PT_CKPT = Path("~/.cache/huggingface/hub/models--microsoft--mattergen"
               "/snapshots/ea430eab64b80855029c2941b9fda15f245a771a"
               "/checkpoints/mattergen_base/checkpoints/last.ckpt").expanduser()

# 输出路径
OUTPUT_DIR = Path(PROJECT_ROOT) / "tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CKPT = OUTPUT_DIR / "matinvent_mattergen_mp20.pdparams"


def load_pytorch_weights(ckpt_path: Path) -> Dict[str, np.ndarray]:
    """加载 PyTorch 权重"""
    try:
        import torch
        ckpt = torch.load(str(ckpt_path), map_location='cpu')
        sd = ckpt['state_dict']
        result = {}
        for k, v in sd.items():
            result[k] = v.numpy() if hasattr(v, 'numpy') else np.array(v)
        return result
    except ImportError:
        logger.warning("PyTorch not available!")
        raise RuntimeError(
            "PyTorch is required to load raw weights.\n"
            "Please install PyTorch or run this script in an environment with PyTorch."
        )


def build_paddle_model() -> paddle.nn.Layer:
    """构建 Paddle MatterGen 模型"""
    from ppmat.models.mattergen.mattergen import MatterGen
    from ppmat.schedulers import LatticeVPSDEScheduler, D3PMScheduler
    from ppmat.schedulers.scheduling_wrapped_sde_ve import NumAtomsVarianceAdjustedWrappedVESDE

    # 创建 MatterGen 模型
    model = MatterGen(
        decoder_cfg={
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
                'otf_graph': False,
            }
        },
        lattice_noise_scheduler_cfg={
            '__class_name__': 'LatticeVPSDEScheduler',
            'limit_density': 0.05771451654022283,
            '__init_params__': {}
        },
        coord_noise_scheduler_cfg={
            '__class_name__': 'NumAtomsVarianceAdjustedWrappedVESDE',
            '__init_params__': {}
        },
        atom_noise_scheduler_cfg={
            '__class_name__': 'D3PMScheduler',
            '__init_params__': {}
        },
        num_train_timesteps=1000,
        time_dim=256,
        lattice_loss_weight=1,
        coord_loss_weight=0.1,
        atom_loss_weight=1,
    )

    model.eval()
    return model


def convert_weights(
    pt_weights: Dict[str, np.ndarray],
    pd_model: paddle.nn.Layer
) -> Tuple[Dict[str, paddle.Tensor], Dict[str, int]]:
    """
    转换 PyTorch 权重到 Paddle 格式

    特别注意：
    - 前缀映射: diffusion_module.model.xxx -> model.xxx
    - Linear 层的输入输出转置
    - Embedding 层不转置
    """
    converted_weights = {}

    pd_state = pd_model.state_dict()
    pd_keys_set = set(pd_state.keys())

    converted_count = 0
    skipped_count = 0
    error_count = 0

    for pt_key, pt_val in sorted(pt_weights.items()):
        # 移除 PyTorch 前缀
        if not pt_key.startswith('diffusion_module.model.'):
            skipped_count += 1
            continue

        pd_key = pt_key.replace('diffusion_module.model.', 'model.')

        if pd_key in pd_keys_set:
            pd_val = pd_state[pd_key]
            pd_shape = tuple(pd_val.shape)
            pt_shape = pt_val.shape

            # Embedding 层不转置
            if '.embeddings.weight' in pt_key:
                # Embedding 层直接复制
                if pd_shape == pt_shape:
                    converted_weights[pd_key] = paddle.to_tensor(
                        pt_val.astype(np.float32)
                    )
                    converted_count += 1
                else:
                    error_count += 1
                    logger.warning("Shape mismatch for Embedding %s", pt_key)
                continue

            # 检查是否需要转置（Linear 层的 weight）
            if pt_key.endswith('.weight') and len(pt_shape) == 2:
                # Linear 权重需要转置: PyTorch [out, in] -> Paddle [in, out]
                converted = pt_val.T
                expected_pd_shape = (pt_shape[1], pt_shape[0])

                if pd_shape == expected_pd_shape:
                    converted_weights[pd_key] = paddle.to_tensor(
                        converted.astype(np.float32)
                    )
                    converted_count += 1
                else:
                    logger.warning(
                        "Shape mismatch after transpose for %s: expected %s, got %s",
                        pt_key, expected_pd_shape, pd_shape,
                    )
                    error_count += 1
            else:
                # 其他层直接复制
                if pd_shape == pt_shape:
                    converted_weights[pd_key] = paddle.to_tensor(
                        pt_val.astype(np.float32)
                    )
                    converted_count += 1
                else:
                    logger.warning("Shape mismatch for %s: PT=%s, PD=%s", pt_key, pt_shape, pd_shape)
                    error_count += 1
        else:
            skipped_count += 1

    summary = {
        "converted": converted_count,
        "skipped": skipped_count,
        "errors": error_count,
        "total_output": len(converted_weights),
        "total_paddle": len(pd_state),
    }

    return converted_weights, summary


def check_conversion(
    pt_weights: Dict[str, np.ndarray],
    converted_weights: Dict[str, paddle.Tensor]
) -> bool:
    """检查转换结果"""
    all_pass = True
    checks = [
        # (pt_key, need_transpose)
        ('diffusion_module.model.fc_atom.weight', True),
        ('diffusion_module.model.fc_atom.bias', False),
        ('diffusion_module.model.gemnet.atom_emb.embeddings.weight', False),
        ('diffusion_module.model.gemnet.atom_latent_emb.weight', True),
        ('diffusion_module.model.gemnet.atom_latent_emb.bias', False),
        ('diffusion_module.model.gemnet.angle_edge_emb.0.weight', True),
        ('diffusion_module.model.gemnet.angle_edge_emb.0.bias', False),
    ]

    for pt_key, need_transpose in checks:
        if pt_key not in pt_weights:
            logger.warning(f"  PyTorch key not found: {pt_key}")
            continue

        pd_key = pt_key.replace('diffusion_module.model.', 'model.')

        if pd_key not in converted_weights:
            logger.warning(f"  Converted key not found: {pd_key} (from {pt_key})")
            all_pass = False
            continue

        pt_val = pt_weights[pt_key]
        pd_val = converted_weights[pd_key].numpy() if hasattr(converted_weights[pd_key], 'numpy') else converted_weights[pd_key]

        if need_transpose:
            pt_compare = pt_val.T
        else:
            pt_compare = pt_val

        if pt_compare.shape != pd_val.shape:
            logger.error(f"  [{pt_key}] SHAPE MISMATCH: {pt_compare.shape} vs {pd_val.shape}")
            all_pass = False
            continue

        diff = np.abs(pt_compare - pd_val).max()
        status = "PASS" if diff < 1e-5 else "FAIL"
        if status == "FAIL":
            all_pass = False
        logger.info(f"  [{status}] {pt_key}: max_diff={diff:.4e}")

    return all_pass


def main():
    logger.info("=" * 60)
    logger.info("MatterGen weight conversion: PyTorch -> Paddle")
    logger.info("=" * 60)

    # [1/5] 加载 PyTorch 权重
    logger.info("\n[1/5] Loading PyTorch weights...")
    if not PT_CKPT.exists():
        logger.error(f"PyTorch checkpoint not found: {PT_CKPT}")
        logger.error("Please run the MatterGen model in MatInvent first to download the checkpoint.")
        sys.exit(1)

    pt_weights = load_pytorch_weights(PT_CKPT)
    logger.info(f"  Loaded {len(pt_weights)} PyTorch weights")

    # [2/5] 构建 Paddle 模型
    logger.info("\n[2/5] Building Paddle model...")
    pd_model = build_paddle_model()
    pd_state = pd_model.state_dict()
    logger.info(f"  Paddle model has {len(pd_state)} weights")

    # [3/5] 转换权重
    logger.info("\n[3/5] Converting weights...")
    converted_weights, summary = convert_weights(pt_weights, pd_model)

    if summary['errors'] > 0:
        logger.warning(f"  Conversion had {summary['errors']} errors")
    logger.info(f"  Converted: {summary['converted']}")
    logger.info(f"  Skipped: {summary['skipped']}")

    # [4/5] 检查转换结果
    logger.info("\n[4/5] Checking conversion...")
    all_pass = check_conversion(pt_weights, converted_weights)

    if all_pass:
        logger.info("All checks PASSED ✓")
    else:
        logger.warning("Some checks FAILED!")

    # [5/5] 保存结果
    logger.info(f"\n[5/5] Saving to: {OUT_CKPT}")
    paddle.save(converted_weights, str(OUT_CKPT))
    logger.info(f"  Saved {len(converted_weights)} weights")

    # 打印摘要
    print("\n" + "=" * 60)
    print("Conversion Summary:")
    print("=" * 60)
    print(f"  Output file: {OUT_CKPT}")
    print(f"  Converted: {summary['converted']}")
    print(f"  Skipped: {summary['skipped']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Total output: {summary['total_output']}")
    print(f"  Total Paddle: {summary['total_paddle']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
