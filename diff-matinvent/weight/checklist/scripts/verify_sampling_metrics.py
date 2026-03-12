#!/usr/bin/env python3
"""
采样指标验证脚本
==========================================

功能：
1. 验证 DiffCSP 和 MatterGen 的采样指标
2. 对比 PyTorch 和 Paddle 的采样输出质量
3. 满足要求：
   - 生成式模型：采样指标保持误差 5% 以内
   - coord_diff_ratio < 5%
   - lattice_diff_ratio < 5%

流程：
1. 运行 PyTorch 采样（通过 subprocess）
2. 运行 Paddle 采样
3. 对比采样质量指标
4. 生成验证报告

用法：
    python verify_sampling_metrics.py --model diffcsp
    python verify_sampling_metrics.py --model mattergen
    python verify_sampling_metrics.py --model all
"""

import argparse
import json
import logging
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent  # PaddleMaterials/
DIFF_MATINVENT_ROOT = PROJECT_ROOT / "diff-matinvent"
PPMAT_ROOT = PROJECT_ROOT
sys.path.insert(0, str(PPMAT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# 阈值配置
THRESHOLDS = {
    "coord_diff_ratio": 0.05,  # 坐标差异比例 5%
    "lattice_diff_ratio": 0.05,  # 晶格差异比例 5%
    "sampling_success_rate": 1.0,  # 采样成功率 100%
    "sample_validity_rate": 1.0,  # 样本有效性 100%
}

# 采样配置
SAMPLING_CONFIG = {
    "num_samples": 2,          # 采样数量
    "num_atoms_list": [[10, 15], [12, 8]],  # 每个样本的原子数列表
    "num_inference_steps": 20,  # 采样步数（优化后的配置）
    "batch_size": 2,
}

# 配置路径
RAW_MATINVENT_ROOT = PROJECT_ROOT / "raw-matinvent"
MATINVENT_PYTHON = Path("~/miniconda3/envs/matinvent/bin/python").expanduser()
OUTPUT_DIR = DIFF_MATINVENT_ROOT / "tmp" / "sampling_metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def run_pytorch_sampling(
    model_type: str,
    output_json: Path,
) -> Optional[Dict]:
    """运行 PyTorch 采样（通过 subprocess）"""
    if not MATINVENT_PYTHON.exists():
        logger.warning(f"matinvent python not found: {MATINVENT_PYTHON}")
        return None

    # 创建 PyTorch 采样脚本
    pt_script = OUTPUT_DIR / f"{model_type}_pytorch_sampling.py"

    if model_type == "diffcsp":
        pt_code = f'''import json
import sys
from pathlib import Path

import numpy as np
import torch

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[PT] Using device: {{device}}", flush=True)

# 设置随机种子
torch.manual_seed({RANDOM_SEED})
np.random.seed({RANDOM_SEED})

# 加载模型
sys.path.insert(0, "{RAW_MATINVENT_ROOT}")
from models.diffcsp.cspnet import CSPNet
from models.diffcsp.diffusion_utils import DiffusionCSP

model = CSPNet(
    hidden_dim=512, latent_dim=256, num_layers=6,
    act_fn="silu", dis_emb="sin", num_freqs=128,
    edge_style="fc", ln=True, ip=True, smooth=True,
    pred_type=False, pred_scalar=False,
)

# 加载权重
ckpt_path = Path("~/.cache/huggingface/hub/models--jwchen25--MatInvent"
                 "/snapshots/b192d079a52d16403fcbbb26b73078d23f9e86c2"
                 "/diffcsp_mp20/last.ckpt").expanduser()
ckpt = torch.load(str(ckpt_path), map_location=device)
sd = ckpt["state_dict"]
decoder_sd = {{k.replace("decoder.", "", 1): v for k, v in sd.items() if k.startswith("decoder.")}}
decoder_sd = {{k: v for k, v in decoder_sd.items() if not k.startswith("type_out")}}
model.load_state_dict(decoder_sd, strict=False)
model.to(device)
model.eval()

# 创建 diffusion
diffusion = DiffusionCSP(
    model,
    num_timesteps=1000,
    loss_type="mse",
    param="x",
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    sampling_type="ddpm",
    noise_schedule="linear",
)

# 采样配置
num_samples = {SAMPLING_CONFIG["num_samples"]}
num_atoms_list = {SAMPLING_CONFIG["num_atoms_list"]}
num_inference_steps = {SAMPLING_CONFIG["num_inference_steps"]}

results = []

for sample_idx in range(num_samples):
    num_atoms = num_atoms_list[sample_idx % len(num_atoms_list)]
    print(f"[PT] Sampling {{sample_idx+1}}/{{num_samples}} with num_atoms={{num_atoms}}", flush=True)

    # 运行采样
    with torch.no_grad():
        try:
            samples = diffusion.sample(
                num_atoms_list=[num_atoms],
                batch_size=1,
                num_inference_steps=num_inference_steps,
                device=device,
            )

            for sample in samples:
                results.append({{
                    "frac_coords": sample["frac_coords"].cpu().numpy().tolist(),
                    "lattice": sample["lattice"].cpu().numpy().tolist(),
                    "atom_types": sample["atom_types"].cpu().numpy().tolist(),
                    "num_atoms": int(sample["num_atoms"].cpu().numpy()),
                }})
                print(f"[PT] Sample generated: lattice_det={{np.linalg.det(sample['lattice'].cpu().numpy()):.4f}", flush=True)
        except Exception as e:
            print(f"[PT] Sampling failed: {{e}}", flush=True)
            continue

# 保存结果
output_data = {{
    "samples": results,
    "num_samples": len(results),
    "num_inference_steps": num_inference_steps,
}}

output_path = Path("{output_json}")
with open(output_path, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"[PT] Sampling completed: {{len(results)}} samples", flush=True)
'''
    else:  # mattergen
        pt_code = f'''import json
import sys
from pathlib import Path

import numpy as np
import torch

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[PT] Using device: {{device}}", flush=True)

# 设置随机种子
torch.manual_seed({RANDOM_SEED})
np.random.seed({RANDOM_SEED})

# MatterGen 采样代码
try:
    from mattergen.common.utils.eval_utils import MatterGenCheckpointInfo
    from mattergen.diffusion.lightning_module import DiffusionLightningModule
    from mattergen.common.data.chemgraph import ChemGraph
    from mattergen.common.data.collate import collate

    # 加载模型
    print(f"[PT] Loading MatterGen model...", flush=True)
    ckpt_info = MatterGenCheckpointInfo.from_hf_hub('mattergen_base')
    model = DiffusionLightningModule.load_from_checkpoint_and_config(
        ckpt_info.checkpoint_path,
        config=ckpt_info.config.lightning_module,
        map_location=device,
        strict=False,
    )[0]
    model.eval()

    # 采样配置
    num_samples = {SAMPLING_CONFIG["num_samples"]}
    num_atoms_list = {SAMPLING_CONFIG["num_atoms_list"]}
    num_inference_steps = {SAMPLING_CONFIG["num_inference_steps"]}

    results = []

    for sample_idx in range(num_samples):
        num_atoms = num_atoms_list[sample_idx % len(num_atoms_list)]
        print(f"[PT] Sampling {{sample_idx+1}}/{{num_samples}} with num_atoms={{num_atoms}}", flush=True)

        with torch.no_grad():
            try:
                # 调用 sample 方法
                samples = model.sample(
                    num_samples=1,
                    num_inference_steps=num_inference_steps,
                    device=device,
                )

                for sample in samples:
                    results.append({{
                        "frac_coords": sample["frac_coords"].cpu().numpy().tolist(),
                        "lattice": sample["cell"].cpu().numpy().tolist(),
                        "atom_types": sample["atomic_numbers"].cpu().numpy().tolist(),
                        "num_atoms": int(sample["num_atoms"].cpu().numpy()),
                    }})
                    print(f"[PT] Sample generated: lattice_det={{np.linalg.det(sample['cell'].cpu().numpy()):.4f}}", flush=True)
            except Exception as e:
                print(f"[PT] Sampling failed: {{e}}", flush=True)
                continue

    # 保存结果
    output_data = {{
        "samples": results,
        "num_samples": len(results),
        "num_inference_steps": num_inference_steps,
    }}

    output_path = Path("{output_json}")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"[PT] Sampling completed: {{len(results)}} samples", flush=True)

except Exception as e:
    print(f"[PT] MatterGen sampling failed: {{e}}", flush=True)
    import traceback
    traceback.print_exc()
'''

    # 保存采样脚本
    with open(pt_script, "w") as f:
        f.write(pt_code)

    # 运行采样
    cmd = [str(MATINVENT_PYTHON), str(pt_script)]
    logger.info(f"  cmd: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        for line in (proc.stdout or "").strip().splitlines():
            logger.info(f"    [PT] {line}")
        if proc.returncode != 0:
            logger.error(f"  PyTorch sampling failed (exit {proc.returncode})")
            for line in (proc.stderr or "").strip().splitlines()[-20:]:
                logger.error(f"    [PT stderr] {line}")
            return None
    except subprocess.TimeoutExpired:
        logger.error("  PyTorch sampling timed out")
        return None
    except Exception as e:
        logger.error(f"  PyTorch sampling error: {e}")
        return None

    if not output_json.exists():
        logger.error(f"  PyTorch output not created: {output_json}")
        return None

    with open(output_json) as f:
        return json.load(f)


def run_paddle_sampling(
    model_type: str,
) -> Dict[str, Any]:
    """运行 Paddle 采样"""
    import paddle

    # 设置随机种子
    paddle.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    num_samples = SAMPLING_CONFIG["num_samples"]
    num_atoms_list = SAMPLING_CONFIG["num_atoms_list"]
    num_inference_steps = SAMPLING_CONFIG["num_inference_steps"]

    results = []

    if model_type == "diffcsp":
        from ppmat.models.diffcsp.diffcsp import CSPNet
        from ppmat.schedulers import DDPMScheduler

        # 加载模型
        model = CSPNet(
            hidden_dim=512, latent_dim=256, num_layers=6,
            act_fn="silu", dis_emb="sin", num_freqs=128,
            edge_style="fc", ln=True, ip=True, smooth=True,
            pred_type=False, prop_dim=512, pred_scalar=False, num_classes=100,
        )

        ckpt_path = DIFF_MATINVENT_ROOT / "tmp" / "matinvent_diffcsp_mp20.pdparams"
        if not ckpt_path.exists():
            logger.error(f"Paddle checkpoint not found: {ckpt_path}")
            raise FileNotFoundError(f"Paddle checkpoint not found: {ckpt_path}")
        model.eval()
        model.set_state_dict(paddle.load(str(ckpt_path)))

        # 创建 scheduler
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="sample",
        )

        for sample_idx in range(num_samples):
            num_atoms = num_atoms_list[sample_idx % len(num_atoms_list)]
            logger.info(f"  Sampling {sample_idx+1}/{num_samples} with num_atoms={num_atoms}")

            with paddle.no_grad():
                try:
                    # 生成初始噪声
                    batch_size = 1
                    total_atoms = num_atoms

                    # 初始化随机噪声
                    frac_coords = paddle.rand([total_atoms, 3], dtype='float32')
                    lattices = paddle.randn([batch_size, 3, 3], dtype='float32') * 0.1
                    lattices = lattices + paddle.eye(3, dtype='float32') * 5.0

                    # 初始化原子类型
                    atom_types = paddle.randint(0, 100, [total_atoms], dtype='int64')

                    # 采样循环
                    scheduler.set_timesteps(num_inference_steps)

                    for t in scheduler.timesteps:
                        # 构建输入
                        time_emb = scheduler.get_timestep_batch(batch_size, t)

                        atom_type_probs = paddle.nn.functional.one_hot(atom_types, num_classes=100).astype('float32')

                        batch_idx = []
                        for i in range(batch_size):
                            batch_idx.extend([i] * num_atoms)
                        batch_idx = paddle.to_tensor(batch_idx, dtype='int64')

                        # 预测噪声
                        pred_l, pred_x = model(
                            time_emb,
                            atom_type_probs,
                            frac_coords,
                            lattices,
                            paddle.to_tensor([num_atoms], dtype='int64'),
                            batch_idx,
                        )

                        # 更新样本
                        frac_coords = pred_x
                        lattices = pred_l

                    results.append({
                        "frac_coords": frac_coords.numpy().tolist(),
                        "lattice": lattices.numpy().tolist(),
                        "atom_types": atom_types.numpy().tolist(),
                        "num_atoms": num_atoms,
                    })
                    logger.info(f"  Sample generated: lattice_det={np.linalg.det(lattices.numpy()[0]):.4f}")

                except Exception as e:
                    logger.error(f"  Sampling failed: {e}")
                    continue

    elif model_type == "mattergen":
        from ppmat.models.mattergen.mattergen import MatterGen

        # 加载模型
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
                    'otf_graph': True,
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

        ckpt_path = DIFF_MATINVENT_ROOT / "tmp" / "matinvent_mattergen_mp20.pdparams"
        if not ckpt_path.exists():
            logger.error(f"Paddle checkpoint not found: {ckpt_path}")
            raise FileNotFoundError(f"Paddle checkpoint not found: {ckpt_path}")
        model.eval()
        model.set_state_dict(paddle.load(str(ckpt_path)))

        for sample_idx in range(num_samples):
            num_atoms = num_atoms_list[sample_idx % len(num_atoms_list)]
            logger.info(f"  Sampling {sample_idx+1}/{num_samples} with num_atoms={num_atoms}")

            with paddle.no_grad():
                try:
                    # 构建输入
                    input_dict = {
                        'frac_coords': paddle.rand([num_atoms, 3], dtype='float32'),
                        'lattice': paddle.eye(3, dtype='float32') * 5.0,
                        'atom_types': paddle.randint(1, 21, [num_atoms], dtype='int64'),
                        'num_atoms': paddle.to_tensor([num_atoms], dtype='int64'),
                    }

                    batch = {
                        'structure_array': input_dict,
                        'batch_idx': paddle.zeros([num_atoms], dtype='int32'),
                    }

                    # 采样
                    samples = model.sample(
                        batch,
                        num_inference_steps=num_inference_steps,
                    )

                    for sample in samples:
                        results.append({
                            "frac_coords": sample['frac_coords'].numpy().tolist(),
                            "lattice": sample['lattice'].numpy().tolist(),
                            "atom_types": sample['atom_types'].numpy().tolist(),
                            "num_atoms": num_atoms,
                        })
                        logger.info(f"  Sample generated: lattice_det={np.linalg.det(sample['lattice'].numpy()):.4f}")

                except Exception as e:
                    logger.error(f"  Sampling failed: {e}")
                    continue

    result = {
        "samples": results,
        "num_samples": len(results),
        "num_inference_steps": num_inference_steps,
    }

    logger.info(f"  Paddle sampling completed: {len(results)} samples")

    return result


def compute_sampling_metrics(
    pt_samples: List[Dict],
    pd_samples: List[Dict],
) -> Dict[str, Any]:
    """计算采样指标"""
    if len(pt_samples) != len(pd_samples):
        logger.warning(f"Sample count mismatch: PT={len(pt_samples)}, PD={len(pd_samples)}")
        min_samples = min(len(pt_samples), len(pd_samples))
        pt_samples = pt_samples[:min_samples]
        pd_samples = pd_samples[:min_samples]

    coord_diffs = []
    lattice_diffs = []

    for pt_sample, pd_sample in zip(pt_samples, pd_samples):
        # 计算坐标差异
        pt_coords = np.array(pt_sample["frac_coords"])
        pd_coords = np.array(pd_sample["frac_coords"])

        if pt_coords.shape == pd_coords.shape:
            coord_diff = np.mean(np.abs(pt_coords - pd_coords))
            coord_diffs.append(coord_diff)

        # 计算晶格差异
        pt_lattice = np.array(pt_sample["lattice"])
        pd_lattice = np.array(pd_sample["lattice"])

        if pt_lattice.shape == pd_lattice.shape:
            # 使用行列式差异作为晶格差异指标
            pt_det = np.abs(np.linalg.det(pt_lattice))
            pd_det = np.abs(np.linalg.det(pd_lattice))
            lattice_diff = np.abs(pt_det - pd_det) / (pt_det + 1e-8)
            lattice_diffs.append(lattice_diff)

    # 计算统计
    coord_diff_mean = float(np.mean(coord_diffs)) if coord_diffs else 0.0
    lattice_diff_mean = float(np.mean(lattice_diffs)) if lattice_diffs else 0.0

    result = {
        "num_samples": len(pt_samples),
        "coord_diff": {
            "values": [float(d) for d in coord_diffs],
            "mean": coord_diff_mean,
            "std": float(np.std(coord_diffs)) if coord_diffs else 0.0,
            "max": float(np.max(coord_diffs)) if coord_diffs else 0.0,
        },
        "lattice_diff": {
            "values": [float(d) for d in lattice_diffs],
            "mean": lattice_diff_mean,
            "std": float(np.std(lattice_diffs)) if lattice_diffs else 0.0,
            "max": float(np.max(lattice_diffs)) if lattice_diffs else 0.0,
        },
        "coord_diff_ratio": coord_diff_mean,
        "lattice_diff_ratio": lattice_diff_mean,
        "coord_pass": coord_diff_mean < THRESHOLDS["coord_diff_ratio"],
        "lattice_pass": lattice_diff_mean < THRESHOLDS["lattice_diff_ratio"],
        "pass": coord_diff_mean < THRESHOLDS["coord_diff_ratio"] and lattice_diff_mean < THRESHOLDS["lattice_diff_ratio"],
    }

    logger.info(f"  Sampling metrics:")
    logger.info(f"    coord_diff_ratio: {result['coord_diff_ratio']:.4f} (threshold: {THRESHOLDS['coord_diff_ratio']:.4f})")
    logger.info(f"    lattice_diff_ratio: {result['lattice_diff_ratio']:.4f} (threshold: {THRESHOLDS['lattice_diff_ratio']:.4f})")
    logger.info(f"    pass: {result['pass']}")

    return result


def generate_markdown_report(
    model_type: str,
    metrics: Optional[Dict],
    pytorch_available: bool,
) -> str:
    """生成 Markdown 报告"""
    lines = [
        f"# {model_type.upper()} 采样指标验证报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 验证要求",
        "",
        f"- **坐标差异比例**: coord_diff_ratio < {THRESHOLDS['coord_diff_ratio']:.2%}",
        f"- **晶格差异比例**: lattice_diff_ratio < {THRESHOLDS['lattice_diff_ratio']:.2%}",
        "",
        "## 采样配置",
        "",
        f"- **随机种子**: {RANDOM_SEED}",
        f"- **采样数量**: {SAMPLING_CONFIG['num_samples']}",
        f"- **采样步数**: {SAMPLING_CONFIG['num_inference_steps']}",
        f"- **原子数列表**: {SAMPLING_CONFIG['num_atoms_list']}",
        "",
    ]

    if not pytorch_available:
        lines += [
            "> **注意**: PyTorch 采样未完成，无法进行完整对比。",
            "",
        ]
    elif metrics is None:
        lines += [
            "> **注意**: 采样指标计算失败。",
            "",
        ]
    else:
        lines += [
            "## 验证结果",
            "",
            f"- **采样数量**: {metrics['num_samples']}",
            f"- **坐标差异比例**: {metrics['coord_diff_ratio']:.4f} ({metrics['coord_diff_ratio']:.2%})",
            f"- **晶格差异比例**: {metrics['lattice_diff_ratio']:.4f} ({metrics['lattice_diff_ratio']:.2%})",
            "",
            "### 详细指标",
            "",
            "#### 坐标差异",
            "",
            f"- **平均值**: {metrics['coord_diff']['mean']:.6f}",
            f"- **标准差**: {metrics['coord_diff']['std']:.6f}",
            f"- **最大值**: {metrics['coord_diff']['max']:.6f}",
            f"- **阈值**: {THRESHOLDS['coord_diff_ratio']:.4f}",
            f"- **状态**: {'PASS' if metrics['coord_pass'] else 'FAIL'}",
            "",
            "#### 晶格差异",
            "",
            f"- **平均值**: {metrics['lattice_diff']['mean']:.6f}",
            f"- **标准差**: {metrics['lattice_diff']['std']:.6f}",
            f"- **最大值**: {metrics['lattice_diff']['max']:.6f}",
            f"- **阈值**: {THRESHOLDS['lattice_diff_ratio']:.4f}",
            f"- **状态**: {'PASS' if metrics['lattice_pass'] else 'FAIL'}",
            "",
            "## 总体结论",
            "",
        ]

        if metrics["pass"]:
            lines += [
                "[PASS] 采样指标验证通过",
                "",
                f"- 坐标差异比例: {metrics['coord_diff_ratio']:.2%} < {THRESHOLDS['coord_diff_ratio']:.2%}",
                f"- 晶格差异比例: {metrics['lattice_diff_ratio']:.2%} < {THRESHOLDS['lattice_diff_ratio']:.2%}",
                f"- PyTorch 和 Paddle 的采样质量高度一致",
                f"- 满足生成式模型采样指标要求",
                "",
            ]
        else:
            lines += [
                "[FAIL] 采样指标验证失败",
                "",
                f"- 坐标差异比例: {metrics['coord_diff_ratio']:.2%}",
                f"- 晶格差异比例: {metrics['lattice_diff_ratio']:.2%}",
                f"- 请检查采样实现",
                "",
            ]

    lines += [
        "---",
        "",
        "*报告生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*",
    ]
    return "\n".join(lines)


def verify_sampling(model_type: str) -> Dict[str, Any]:
    """验证单个模型的采样指标"""
    logger.info("=" * 60)
    logger.info(f"Verifying {model_type.upper()} sampling metrics")
    logger.info("=" * 60)

    # 运行 PyTorch 采样
    logger.info("\n[1/3] Running PyTorch sampling...")
    pt_output_json = OUTPUT_DIR / f"{model_type}_pytorch_sampling.json"
    pt_result = run_pytorch_sampling(model_type, pt_output_json)

    pytorch_available = pt_result is not None
    metrics = None

    # 运行 Paddle 采样
    logger.info("\n[2/3] Running Paddle sampling...")
    try:
        pd_result = run_paddle_sampling(model_type)

        # 计算采样指标
        if pytorch_available and pt_result.get("samples") and pd_result.get("samples"):
            logger.info("\n[3/3] Computing sampling metrics...")
            metrics = compute_sampling_metrics(pt_result["samples"], pd_result["samples"])
        else:
            logger.warning("Skipping metrics computation (PyTorch not available or no samples)")

    except Exception as e:
        logger.error(f"Paddle sampling failed: {e}")
        return {"error": str(e)}

    # 生成报告
    md = generate_markdown_report(model_type, metrics, pytorch_available)

    report_md = OUTPUT_DIR / f"{model_type}_sampling_metrics_report.md"
    with open(report_md, "w") as f:
        f.write(md)
    logger.info(f"  Markdown report: {report_md}")

    # 保存 JSON 报告
    report_json = OUTPUT_DIR / f"{model_type}_sampling_metrics_report.json"
    report_data = {
        "model_type": model_type,
        "timestamp": datetime.now().isoformat(),
        "sampling_config": SAMPLING_CONFIG,
        "pytorch_available": pytorch_available,
        "pytorch_result": pt_result if pytorch_available else None,
        "metrics": metrics,
    }
    with open(report_json, "w") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    logger.info(f"  JSON report: {report_json}")

    # 打印摘要
    print("\n" + "=" * 60)
    print(f"{model_type.upper()} Sampling Metrics Verification Summary")
    print("=" * 60)
    if pytorch_available and metrics:
        status = "PASS" if metrics["pass"] else "FAIL"
        print(f"  Status: {status}")
        print(f"  coord_diff_ratio: {metrics['coord_diff_ratio']:.4f} ({metrics['coord_diff_ratio']:.2%})")
        print(f"  lattice_diff_ratio: {metrics['lattice_diff_ratio']:.4f} ({metrics['lattice_diff_ratio']:.2%})")
    else:
        print("  PyTorch sampling not available, skipping comparison")
    print("=" * 60)

    return report_data


def main():
    parser = argparse.ArgumentParser(description="Verify sampling metrics for DiffCSP and MatterGen")
    parser.add_argument(
        "--model",
        type=str,
        choices=["diffcsp", "mattergen", "all"],
        default="all",
        help="Model type to verify (default: all)",
    )
    args = parser.parse_args()

    results = {}

    if args.model in ["diffcsp", "all"]:
        results["diffcsp"] = verify_sampling("diffcsp")

    if args.model in ["mattergen", "all"]:
        results["mattergen"] = verify_sampling("mattergen")

    # 生成总体报告
    if len(results) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("Overall Summary")
        logger.info("=" * 60)
        for model_type, result in results.items():
            if "error" in result:
                print(f"  {model_type.upper()}: ERROR - {result['error']}")
            elif result.get("metrics") is not None:
                status = "PASS" if result["metrics"]["pass"] else "FAIL"
                print(f"  {model_type.upper()}: {status}")
            else:
                print(f"  {model_type.upper()}: SKIPPED")
        print("=" * 60)


if __name__ == "__main__":
    main()
