#!/usr/bin/env python3
"""
MatInvent 内部的 DiffCSP 权重转换脚本
===============================
将 PyTorch checkpoint 转换为 Paddle pdparams 格式

特别注意：

1.MatInvent 是自己内置手搓了一份 DiffCSP，所以不能用网上公开的权重信息，必须用 github上自带的hg上的仓库地址

本代码主要参考了 paddle自带的 diffcsp 的 mp20的模型的映射信息，下载地址：

ppmat/models/__init__.py 代码中的 diffcsp_mp20
https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/diffcsp/diffcsp_mp20.zip

2. 当前转换脚本，必须放在 ppmat项目的py环境里执行，因为导出模型需要用到 ppmat的 CSPNet

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

# 如果运行过原生的 MatInvent，权重会在这里，如果不想运行原生代码，参考：
# https://huggingface.co/jwchen25/MatInvent/tree/main/diffcsp_mp20
PT_CKPT = Path("~/.cache/huggingface/hub/models--jwchen25--MatInvent/snapshots/b192d079a52d16403fcbbb26b73078d23f9e86c2/diffcsp_mp20/last.ckpt").expanduser()


# 输出路径
OUTPUT_DIR = Path(PROJECT_ROOT) / "tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CKPT = OUTPUT_DIR / "matinvent_diffcsp_mp20.pdparams"


def load_pytorch_weights(ckpt_path: Path) -> Dict[str, np.ndarray]:
    try:
        import torch
        ckpt = torch.load(str(ckpt_path), map_location='cpu')
        sd = ckpt['state_dict']
        result = {}
        for k, v in sd.items():
            result[k] = v.numpy() if hasattr(v, 'numpy') else np.array(v)
        return result
    except ImportError:
        logger.warning("load raw paddle error!!!")

    raise RuntimeError(
        "check you diffcsp raw torch weight file.\n"
    )


def build_paddle_model_smooth_true() -> paddle.nn.Layer:
    from ppmat.models.diffcsp.diffcsp import CSPNet
    decoder = CSPNet(
        hidden_dim=512,
        latent_dim=256,
        num_layers=6,
        act_fn="silu",
        dis_emb="sin",
        num_freqs=128,
        edge_style="fc",
        ln=True,
        ip=True,
        smooth=True,
        pred_type=False,
        prop_dim=512,
        pred_scalar=False,
        num_classes=100,
    )
    decoder.eval()
    return decoder




def convert_weights(
    pt_weights: Dict[str, np.ndarray],
    pd_model: paddle.nn.Layer
) -> Tuple[Dict[str, paddle.Tensor], Dict[str, int]]:
    """
    转换 PyTorch 权重到 Paddle 格式, 特别注意如下的细节：
    - 线性层的输入输出转置
    """
    converted_weights = {}

    pd_state = pd_model.state_dict()
    pd_keys_set = set(pd_state.keys())

    skip_prefixes = [
        'beta_scheduler.',
        'sigma_scheduler.',
        'type_sigma_scheduler.',
    ]

    skip_keys = {
        'decoder.type_out.weight',
        'decoder.type_out.bias',
    }

    paddle_only_patterns = ['prop_mlp']

    converted_count = 0
    skipped_count = 0
    paddle_only_count = 0
    error_count = 0

    for pt_key, pt_val in sorted(pt_weights.items()):
        if any(pt_key.startswith(p) for p in skip_prefixes):
            skipped_count += 1
            continue

        if pt_key in skip_keys:
            skipped_count += 1
            continue

        pd_key_no_prefix = pt_key.replace('decoder.', '') if pt_key.startswith('decoder.') else pt_key

        if pd_key_no_prefix in pd_keys_set:
            pd_val = pd_state[pd_key_no_prefix]
            pd_shape = tuple(pd_val.shape)
            pt_shape = pt_val.shape

            if pt_key.endswith('.weight') and len(pt_shape) == 2:
                # Linear 权重需要转置: PyTorch [out, in] -> Paddle [in, out]
                converted = pt_val.T
                expected_pd_shape = (pt_shape[1], pt_shape[0])

                if pd_shape == expected_pd_shape:
                    converted_weights[pd_key_no_prefix] = paddle.to_tensor(
                        converted.astype(np.float32)
                    )
                    converted_count += 1
                else:
                    logger.debug(
                        "Shape mismatch after transpose for %s: expected %s, got %s",
                        pt_key, expected_pd_shape, pd_shape,
                    )
                    error_count += 1
            else:
                if pd_shape == pt_shape:
                    converted_weights[pd_key_no_prefix] = paddle.to_tensor(
                        pt_val.astype(np.float32)
                    )
                    converted_count += 1
                else:
                    logger.debug("Shape mismatch for %s: PT=%s, PD=%s", pt_key, pt_shape, pd_shape)
                    error_count += 1
        else:
            skipped_count += 1

    for pd_key in sorted(pd_keys_set):
        if pd_key not in converted_weights and any(pattern in pd_key for pattern in paddle_only_patterns):
            pd_val = pd_state[pd_key]
            converted_weights[pd_key] = pd_val
            paddle_only_count += 1

    summary = {
        "converted": converted_count,
        "skipped": skipped_count,
        "copied_from_paddle": paddle_only_count,
        "errors": error_count,
        "total_output": len(converted_weights)
    }

    return converted_weights, summary



def check_conversion(
    pt_weights: Dict[str, np.ndarray],
    converted_weights: Dict[str, paddle.Tensor]
) -> bool:
    all_pass = True
    checks = [
        # (pt_key, need_transpose)
        ('decoder.node_embedding.weight', True),
        ('decoder.atom_latent_emb.weight', True),
        ('decoder.atom_latent_emb.bias', False),
        ('decoder.csp_layer_0.edge_mlp.0.weight', True),
        ('decoder.csp_layer_0.edge_mlp.0.bias', False),
        ('decoder.coord_out.weight', True),
        ('decoder.lattice_out.weight', True),
    ]

    for pt_key, need_transpose in checks:
        if pt_key not in pt_weights:
            logger.warning(f"  PyTorch key not found: {pt_key}")
            continue
        pd_key = pt_key.replace('decoder.', '') if pt_key.startswith('decoder.') else pt_key

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
    logger.info("start diffcsp mp20 model to paddle format ... ")
    logger.info("=" * 60)

    logger.info("\n[1/4] Loading PyTorch raw weights...")
    pt_weights = load_pytorch_weights(PT_CKPT)
    logger.info(f"  Successfully loaded {len(pt_weights)} PyTorch weights")

    logger.info("\n[2/4] Building Paddle model...")
    pd_model = build_paddle_model_smooth_true()
    pd_state = pd_model.state_dict()
    logger.info(f"  Paddle model has {len(pd_state)} weights")

    logger.info("\n[3/4] Converting weights from pytorch  to paddle ...")
    converted_weights, summary = convert_weights(pt_weights, pd_model)

    if summary['errors'] > 0:
        logger.warning("convert has error. should debug this file")
        logger.warning("conversion summary: %s", summary)

    logger.info("\n[4/4] check conversion...")
    all_pass = check_conversion(pt_weights, converted_weights)

    if all_pass:
        logger.info("All checks PASSED ✓")
    else:
        logger.warning("checks FAILED!")

    logger.info(f"\n[5/5] Saving results...")
    paddle.save(converted_weights, str(OUT_CKPT))
    logger.info(f"paddle model saved to: {OUT_CKPT}")

    print("\n" + "=" * 60)
    print("Conversion Summary:")
    print("=" * 60)
    print(f"  Output file: {OUT_CKPT}")
    print(f"  Converted: {summary['converted']} weights")
    print(f"  Skipped: {summary['skipped']} weights (scheduler, type_out)")
    print(f"  Copied from Paddle: {summary['copied_from_paddle']} weights (prop_mlp)")
    print(f"  Errors: {summary['errors']} weights")
    print("=" * 60)


if __name__ == "__main__":
    main()
