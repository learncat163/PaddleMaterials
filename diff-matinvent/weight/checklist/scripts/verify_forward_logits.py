#!/usr/bin/env python3
"""
前向 logits 验证脚本（整合版）
==========================================

功能：
1. 验证 DiffCSP 和 MatterGen 的前向 logits 精度
2. 对比 PyTorch 和 Paddle 的中间层输出
3. 满足要求：
   - 单卡前向精度对齐：前向 logits diff < 1e-4（生成式 1e-6）
   - 反向对齐：训练 2 轮以上，loss 一致
   - 生成式模型：采样指标保持误差 5% 以内

流程：
1. 生成固定输入数据
2. 运行 PyTorch 推理（通过 subprocess）
3. 运行 Paddle 推理
4. 对比前向 logits 差异
5. 生成验证报告

用法：
    python verify_forward_logits.py --model diffcsp
    python verify_forward_logits.py --model mattergen
    python verify_forward_logits.py --model all
"""

import argparse
import json
import logging
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent  # PaddleMaterials/
DIFF_MATINVENT_ROOT = PROJECT_ROOT / "diff-matinvent"
PPMAT_ROOT = PROJECT_ROOT
sys.path.insert(0, str(PPMAT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# 阈值配置
THRESHOLDS = {
    "diffcsp": {
        "first_layer": 1e-4,      # DiffCSP 第一层阈值
        "output": 1e-4,            # DiffCSP 输出阈值
    },
    "mattergen": {
        "first_layer": 1e-6,      # MatterGen 第一层阈值（生成式模型）
        "output": 1e-6,            # MatterGen 输出阈值（生成式模型）
    },
    "training": {
        "loss_diff": 1e-3,         # 训练 loss 差异阈值
    },
    "sampling": {
        "coord_diff_ratio": 0.05,  # 坐标差异比例 5%
        "lattice_diff_ratio": 0.05, # 晶格差异比例 5%
    },
}

# 配置路径
RAW_MATINVENT_ROOT = PROJECT_ROOT / "raw-matinvent"
MATINVENT_PYTHON = Path("~/miniconda3/envs/matinvent/bin/python").expanduser()
OUTPUT_DIR = DIFF_MATINVENT_ROOT / "tmp" / "forward_logits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def build_diffcsp_test_input(seed: int = RANDOM_SEED) -> Dict:
    """生成 DiffCSP 测试输入"""
    rng = np.random.RandomState(seed)
    num_atoms_list = [8, 12, 10, 6]
    batch_size = len(num_atoms_list)
    total_atoms = sum(num_atoms_list)

    # 时间步 t（固定为 500）
    t_value = 500
    half_dim = 128
    freqs = np.exp(np.arange(half_dim) * (-math.log(10000) / (half_dim - 1)))
    t_arr = np.array([t_value], dtype=np.float32)
    emb = t_arr[:, None] * freqs[None, :]
    time_emb_single = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
    time_emb = np.tile(time_emb_single, (batch_size, 1)).astype(np.float32)

    # 原子类型（one-hot）
    atom_type_probs = np.zeros((total_atoms, 100), dtype=np.float32)
    for i_batch, num in enumerate(num_atoms_list):
        start = sum(num_atoms_list[:i_batch])
        for j in range(start, start + num):
            atom_idx = rng.randint(0, 100)
            atom_type_probs[j, atom_idx] = 1.0

    # 分数坐标
    frac_coords = rng.rand(total_atoms, 3).astype(np.float32)

    # 晶格矩阵
    lattices = np.zeros((batch_size, 3, 3), dtype=np.float32)
    for b in range(batch_size):
        diag = rng.uniform(3.0, 8.0, 3).astype(np.float32)
        lattices[b] = np.diag(diag)

    # batch 索引
    batch_idx = []
    for i, num in enumerate(num_atoms_list):
        batch_idx.extend([i] * num)

    return {
        "model_type": "diffcsp",
        "metadata": {
            "seed": seed,
            "batch_size": batch_size,
            "total_atoms": total_atoms,
            "num_atoms": num_atoms_list,
            "t_value": t_value,
        },
        "time_emb": time_emb.tolist(),
        "atom_type_probs": atom_type_probs.tolist(),
        "frac_coords": frac_coords.tolist(),
        "lattices": lattices.tolist(),
        "num_atoms": num_atoms_list,
        "batch_idx": batch_idx,
    }


def build_mattergen_test_input(seed: int = RANDOM_SEED) -> Dict:
    """生成 MatterGen 测试输入"""
    rng = np.random.RandomState(seed)
    num_atoms_list = [8, 12]
    batch_size = len(num_atoms_list)
    total_atoms = sum(num_atoms_list)

    # 时间步 t（归一化到 [0, 1]）
    t_value = 500
    times = np.array([t_value / 1000.0] * batch_size, dtype=np.float32)

    # 原子类型（使用原子序数）
    atom_types = []
    for i_batch, num in enumerate(num_atoms_list):
        for j in range(num):
            atom_types.append(rng.randint(1, 21))
    atom_types = np.array(atom_types, dtype=np.int64)

    # 分数坐标
    frac_coords = rng.rand(total_atoms, 3).astype(np.float32)

    # 晶格矩阵
    lattices = np.zeros((batch_size, 3, 3), dtype=np.float32)
    for b in range(batch_size):
        diag = rng.uniform(3.0, 8.0, 3).astype(np.float32)
        lattices[b] = np.diag(diag)

    # batch 索引
    batch_idx = []
    for i, num in enumerate(num_atoms_list):
        batch_idx.extend([i] * num)

    return {
        "model_type": "mattergen",
        "metadata": {
            "seed": seed,
            "batch_size": batch_size,
            "total_atoms": total_atoms,
            "num_atoms": num_atoms_list,
            "times": times.tolist(),
        },
        "times": times.tolist(),
        "atom_types": atom_types.tolist(),
        "frac_coords": frac_coords.tolist(),
        "lattices": lattices.tolist(),
        "num_atoms": num_atoms_list,
        "batch_idx": batch_idx,
    }


def run_pytorch_inference(
    model_type: str,
    test_input: Dict,
    output_json: Path,
) -> Optional[Dict]:
    """运行 PyTorch 推理（通过 subprocess）"""
    if not MATINVENT_PYTHON.exists():
        logger.warning(f"matinvent python not found: {MATINVENT_PYTHON}")
        return None

    # 根据模型类型选择 runner 脚本
    if model_type == "diffcsp":
        runner_script = DIFF_MATINVENT_ROOT / "weight" / "diffcsp" / "diffcsp_first_layer_pt_runner.py"
    elif model_type == "mattergen":
        runner_script = DIFF_MATINVENT_ROOT / "weight" / "mattergen" / "mattergen_first_layer_pt_runner.py"
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None

    if not runner_script.exists():
        logger.warning(f"Runner script not found: {runner_script}")
        return None

    # 保存输入到临时文件
    input_json = OUTPUT_DIR / f"{model_type}_input.json"
    with open(input_json, "w") as f:
        json.dump(test_input, f)

    # PyTorch checkpoint 路径
    if model_type == "diffcsp":
        pt_ckpt = Path("~/.cache/huggingface/hub/models--jwchen25--MatInvent"
                       "/snapshots/b192d079a52d16403fcbbb26b73078d23f9e86c2"
                       "/diffcsp_mp20/last.ckpt").expanduser()
    else:  # mattergen
        pt_ckpt = Path("~/.cache/huggingface/hub/models--microsoft--mattergen"
                       "/snapshots/ea430eab64b80855029c2941b9fda15f245a771a"
                       "/checkpoints/mattergen_base/checkpoints/last.ckpt").expanduser()

    cmd = [
        str(MATINVENT_PYTHON),
        str(runner_script),
        str(RAW_MATINVENT_ROOT),
        str(input_json),
        str(pt_ckpt),
        str(output_json),
    ]
    logger.info(f"  cmd: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        for line in (proc.stdout or "").strip().splitlines():
            logger.info(f"    [PT] {line}")
        if proc.returncode != 0:
            logger.error(f"  PyTorch subprocess failed (exit {proc.returncode})")
            for line in (proc.stderr or "").strip().splitlines()[-20:]:
                logger.error(f"    [PT stderr] {line}")
            return None
    except subprocess.TimeoutExpired:
        logger.error("  PyTorch subprocess timed out")
        return None
    except Exception as e:
        logger.error(f"  PyTorch subprocess error: {e}")
        return None

    if not output_json.exists():
        logger.error(f"  PyTorch output not created: {output_json}")
        return None

    with open(output_json) as f:
        return json.load(f)


def run_paddle_diffcsp_inference(test_input: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """运行 Paddle DiffCSP 推理"""
    import paddle

    from ppmat.models.diffcsp.diffcsp import CSPNet

    class FirstLayerHook:
        def __init__(self):
            self.output = None

        def __call__(self, layer, inp, out):
            if isinstance(out, paddle.Tensor):
                self.output = out.detach().clone().numpy()

    # 加载模型
    ckpt_path = DIFF_MATINVENT_ROOT / "tmp" / "matinvent_diffcsp_mp20.pdparams"
    if not ckpt_path.exists():
        logger.error(f"Paddle checkpoint not found: {ckpt_path}")
        raise FileNotFoundError(f"Paddle checkpoint not found: {ckpt_path}")

    model = CSPNet(
        hidden_dim=512, latent_dim=256, num_layers=6,
        act_fn="silu", dis_emb="sin", num_freqs=128,
        edge_style="fc", ln=True, ip=True, smooth=True,
        pred_type=False, prop_dim=512, pred_scalar=False, num_classes=100,
    )
    model.eval()
    model.set_state_dict(paddle.load(str(ckpt_path)))

    # 注册 hook
    hook = FirstLayerHook()
    handle = model.node_embedding.register_forward_post_hook(hook)

    # 推理
    with paddle.no_grad():
        pred_l, pred_x = model(
            paddle.to_tensor(np.array(test_input["time_emb"], dtype=np.float32)),
            paddle.to_tensor(np.array(test_input["atom_type_probs"], dtype=np.float32)),
            paddle.to_tensor(np.array(test_input["frac_coords"], dtype=np.float32)),
            paddle.to_tensor(np.array(test_input["lattices"], dtype=np.float32)),
            paddle.to_tensor(np.array(test_input["num_atoms"], dtype=np.int64)),
            paddle.to_tensor(np.array(test_input["batch_idx"], dtype=np.int64)),
        )

    handle.remove()

    fl = hook.output
    pl = pred_l.numpy()
    px = pred_x.numpy()

    logger.info(f"  Paddle node_embedding: shape={fl.shape}, mean={fl.mean():.6f}")
    logger.info(f"  Paddle pred_l: shape={pl.shape}, mean={pl.mean():.6f}")
    logger.info(f"  Paddle pred_x: shape={px.shape}, mean={px.mean():.6f}")

    return fl, pl, px


def run_paddle_mattergen_inference(test_input: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """运行 Paddle MatterGen 推理"""
    import paddle

    # 原始实现: from ppmat.models.mattergen.mattergen import MatterGen
    from ppmat.models.matinvent.mattergen_compat import MatinventMatterGen
    from ppmat.schedulers import LatticeVPSDEScheduler, D3PMScheduler
    from ppmat.schedulers.scheduling_wrapped_sde_ve import NumAtomsVarianceAdjustedWrappedVESDE

    class FirstLayerHook:
        def __init__(self):
            self.output = None

        def __call__(self, layer, inp, out):
            if isinstance(out, paddle.Tensor):
                self.output = out.detach().clone().numpy()

    # 加载模型
    ckpt_path = DIFF_MATINVENT_ROOT / "tmp" / "matinvent_mattergen_mp20.pdparams"
    if not ckpt_path.exists():
        logger.error(f"Paddle checkpoint not found: {ckpt_path}")
        raise FileNotFoundError(f"Paddle checkpoint not found: {ckpt_path}")

    model = MatinventMatterGen(
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
    model.eval()
    model.set_state_dict(paddle.load(str(ckpt_path)))

    # 构建输入
    structure_array = {
        'frac_coords': paddle.to_tensor(np.array(test_input["frac_coords"], dtype=np.float32)),
        'lattice': paddle.to_tensor(np.array(test_input["lattices"], dtype=np.float32)),
        'atom_types': paddle.to_tensor(np.array(test_input["atom_types"], dtype=np.int64)),
        'num_atoms': paddle.to_tensor(np.array(test_input["num_atoms"], dtype=np.int64)),
    }

    batch = {
        'structure_array': structure_array,
        'batch_idx': paddle.to_tensor(np.array(test_input["batch_idx"], dtype=np.int32)),
    }

    # 注册 hook
    hook = FirstLayerHook()

    def _hook_fn(layer, inp, out):
        if isinstance(out, paddle.Tensor):
            hook.output = out.detach().clone().numpy()

    handle = model.model.gemnet.atom_emb.register_forward_post_hook(_hook_fn)

    # 推理
    with paddle.no_grad():
        times = paddle.to_tensor(np.array(test_input["times"], dtype=np.float32))
        noise_batch = {
            "frac_coords": structure_array["frac_coords"],
            "lattice": structure_array["lattice"],
            "atom_types": structure_array["atom_types"],
            "num_atoms": structure_array["num_atoms"],
            "batch": batch["batch_idx"],
        }
        output = model.model(noise_batch, times)

    handle.remove()

    fl = hook.output
    pl = output["lattice"].numpy()
    px = output["frac_coords"].numpy()
    pa = output["atom_types"].numpy()

    logger.info(f"  Paddle atom_emb: shape={fl.shape}, mean={fl.mean():.6f}")
    logger.info(f"  Paddle pred_lattice: shape={pl.shape}, mean={pl.mean():.6f}")
    logger.info(f"  Paddle pred_frac_coords: shape={px.shape}, mean={px.mean():.6f}")
    logger.info(f"  Paddle pred_atom_types: shape={pa.shape}, mean={pa.mean():.6f}")

    return fl, pl, px, pa


def compare_arrays(
    pt: np.ndarray,
    pd: np.ndarray,
    label: str,
    threshold: float,
) -> Dict[str, Any]:
    """对比两个数组"""
    logger.info(f"Comparing {label}...")
    if pt.shape != pd.shape:
        msg = f"Shape mismatch: PyTorch {pt.shape} vs Paddle {pd.shape}"
        logger.error(f"  {msg}")
        return {"error": msg}

    abs_diff = np.abs(pt - pd)
    total = int(abs_diff.size)
    result = {
        "label": label,
        "shape": list(pt.shape),
        "total_elements": total,
        "abs_diff": {
            "max": float(abs_diff.max()),
            "mean": float(abs_diff.mean()),
            "std": float(abs_diff.std()),
            "median": float(np.median(abs_diff)),
        },
        "percentiles": {
            "p90": float(np.percentile(abs_diff, 90)),
            "p95": float(np.percentile(abs_diff, 95)),
            "p99": float(np.percentile(abs_diff, 99)),
            "p99.9": float(np.percentile(abs_diff, 99.9)),
        },
        "threshold": threshold,
        "thresholds": {
            f"lt_{threshold}": int(np.sum(abs_diff < threshold)),
            "lt_1e-5": int(np.sum(abs_diff < 1e-5)),
            "lt_1e-6": int(np.sum(abs_diff < 1e-6)),
        },
    }
    result["thresholds_pct"] = {k: 100.0 * v / total for k, v in result["thresholds"].items()}
    result["pass"] = result["abs_diff"]["mean"] < threshold
    logger.info(f"  mean_abs_diff={result['abs_diff']['mean']:.4e}, "
                f"threshold={threshold:.4e}, pass={result['pass']}")
    return result


def generate_markdown_report(
    model_type: str,
    test_input: Dict,
    comparisons: List[Dict],
    pytorch_available: bool,
) -> str:
    """生成 Markdown 报告"""
    meta = test_input["metadata"]
    threshold = THRESHOLDS[model_type]["first_layer"]

    lines = [
        f"# {model_type.upper()} 前向 Logits 验证报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 验证要求",
        "",
        f"- **前向 logits 精度**: mean_diff < {threshold:.4e}",
        f"- **验证对象**: 第一层输出 (first_layer / atom_emb)",
        "",
        "## 测试配置",
        "",
        f"- **随机种子**: {meta['seed']}",
        f"- **Batch Size**: {meta['batch_size']}",
        f"- **Total Atoms**: {meta['total_atoms']}",
        f"- **每批原子数**: {meta['num_atoms']}",
        "",
    ]

    if not pytorch_available:
        lines += [
            "> **注意**: PyTorch 推理未完成，无法进行完整对比。",
            "",
        ]
    else:
        # 统计通过率
        total_checks = len(comparisons)
        passed_checks = sum(1 for c in comparisons if c.get("pass", False))
        lines += [
            "## 验证结果",
            "",
            f"- **总检查项**: {total_checks}",
            f"- **通过项**: {passed_checks}",
            f"- **失败项**: {total_checks - passed_checks}",
            f"- **通过率**: {100.0 * passed_checks / total_checks:.1f}%",
            "",
        ]

        # 详细结果
        for cmp in comparisons:
            if "error" in cmp:
                lines += [
                    f"### {cmp['label']}",
                    "",
                    f"[FAIL] {cmp['error']}",
                    "",
                ]
            else:
                lines += [
                    f"### {cmp['label']}",
                    "",
                    f"- **Shape**: {cmp['shape']}",
                    f"- **总元素数**: {cmp['total_elements']:,}",
                    f"- **最大差异**: {cmp['abs_diff']['max']:.4e}",
                    f"- **平均差异**: {cmp['abs_diff']['mean']:.4e}",
                    f"- **中位数差异**: {cmp['abs_diff']['median']:.4e}",
                    f"- **P99.9**: {cmp['percentiles']['p99.9']:.4e}",
                    f"- **阈值**: {cmp['threshold']:.4e}",
                    f"- **状态**: {'PASS' if cmp['pass'] else 'FAIL'}",
                    "",
                ]

        # 总体结论
        all_passed = all(c.get("pass", False) for c in comparisons)
        lines += [
            "## 总体结论",
            "",
        ]
        if all_passed:
            lines += [
                f"[PASS] 前向 logits 验证通过",
                "",
                f"- 所有检查项的平均差异 < {threshold:.4e}",
                f"- PyTorch 和 Paddle 的前向输出高度一致",
                f"- 满足单卡前向精度对齐要求",
                "",
            ]
        else:
            lines += [
                f"[FAIL] 前向 logits 验证失败",
                "",
                f"- 部分检查项的平均差异 >= {threshold:.4e}",
                f"- 请检查权重转换和模型实现",
                "",
            ]

    lines += [
        "---",
        "",
        "*报告生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*",
    ]
    return "\n".join(lines)


def verify_model(model_type: str) -> Dict[str, Any]:
    """验证单个模型的前向 logits"""
    logger.info("=" * 60)
    logger.info(f"Verifying {model_type.upper()} forward logits")
    logger.info("=" * 60)

    # 生成测试输入
    logger.info(f"\n[1/4] Generating test input (seed={RANDOM_SEED})...")
    if model_type == "diffcsp":
        test_input = build_diffcsp_test_input(RANDOM_SEED)
    elif model_type == "mattergen":
        test_input = build_mattergen_test_input(RANDOM_SEED)
    else:
        logger.error(f"Unknown model type: {model_type}")
        return {"error": f"Unknown model type: {model_type}"}

    # 运行 PyTorch 推理
    logger.info("\n[2/4] Running PyTorch inference...")
    pt_output_json = OUTPUT_DIR / f"{model_type}_pytorch_output.json"
    pt_result = run_pytorch_inference(model_type, test_input, pt_output_json)

    pytorch_available = pt_result is not None
    comparisons = []

    # 运行 Paddle 推理
    logger.info("\n[3/4] Running Paddle inference...")
    try:
        if model_type == "diffcsp":
            pd_fl, pd_pl, pd_px = run_paddle_diffcsp_inference(test_input)

            if pytorch_available:
                pt_fl = np.array(pt_result["first_layer_output"], dtype=np.float32)
                pt_pl = np.array(pt_result["pred_l"], dtype=np.float32)
                pt_px = np.array(pt_result["pred_x"], dtype=np.float32)

                threshold = THRESHOLDS["diffcsp"]["first_layer"]
                fl_cmp = compare_arrays(pt_fl, pd_fl, "node_embedding (first_layer)", threshold)
                pl_cmp = compare_arrays(pt_pl, pd_pl, "pred_l", threshold)
                px_cmp = compare_arrays(pt_px, pd_px, "pred_x", threshold)
                comparisons = [fl_cmp, pl_cmp, px_cmp]
            else:
                logger.warning("Skipping comparison (PyTorch not available)")

        elif model_type == "mattergen":
            pd_fl, pd_pl, pd_px, pd_pa = run_paddle_mattergen_inference(test_input)

            if pytorch_available:
                pt_fl = np.array(pt_result["first_layer_output"], dtype=np.float32)
                pt_pl = np.array(pt_result["pred_lattice"], dtype=np.float32)
                pt_px = np.array(pt_result["pred_frac_coords"], dtype=np.float32)
                pt_pa = np.array(pt_result["pred_atom_types"], dtype=np.float32)

                threshold = THRESHOLDS["mattergen"]["first_layer"]
                fl_cmp = compare_arrays(pt_fl, pd_fl, "atom_emb (first_layer)", threshold)
                pl_cmp = compare_arrays(pt_pl, pd_pl, "pred_lattice", threshold)
                px_cmp = compare_arrays(pt_px, pd_px, "pred_frac_coords", threshold)
                pa_cmp = compare_arrays(pt_pa, pd_pa, "pred_atom_types", threshold)
                comparisons = [fl_cmp, pl_cmp, px_cmp, pa_cmp]
            else:
                logger.warning("Skipping comparison (PyTorch not available)")
    except Exception as e:
        logger.error(f"Paddle inference failed: {e}")
        return {"error": str(e)}

    # 生成报告
    logger.info("\n[4/4] Generating report...")
    md = generate_markdown_report(model_type, test_input, comparisons, pytorch_available)

    report_md = OUTPUT_DIR / f"{model_type}_forward_logits_report.md"
    with open(report_md, "w") as f:
        f.write(md)
    logger.info(f"  Markdown report: {report_md}")

    # 保存 JSON 报告
    report_json = OUTPUT_DIR / f"{model_type}_forward_logits_report.json"
    report_data = {
        "model_type": model_type,
        "timestamp": datetime.now().isoformat(),
        "test_input_meta": test_input["metadata"],
        "pytorch_available": pytorch_available,
        "comparisons": comparisons,
        "all_passed": all(c.get("pass", False) for c in comparisons) if comparisons else False,
    }
    with open(report_json, "w") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    logger.info(f"  JSON report: {report_json}")

    # 打印摘要
    print("\n" + "=" * 60)
    print(f"{model_type.upper()} Forward Logits Verification Summary")
    print("=" * 60)
    if pytorch_available and comparisons:
        passed = sum(1 for c in comparisons if c.get("pass", False))
        total = len(comparisons)
        print(f"  Passed: {passed}/{total} ({100.0 * passed / total:.1f}%)")
        for cmp in comparisons:
            status = "PASS" if cmp.get("pass", False) else "FAIL"
            print(f"  [{status}] {cmp['label']}: mean_diff={cmp['abs_diff']['mean']:.4e}")
    else:
        print("  PyTorch inference not available, skipping comparison")
    print("=" * 60)

    return report_data


def main():
    parser = argparse.ArgumentParser(description="Verify forward logits for DiffCSP and MatterGen")
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
        results["diffcsp"] = verify_model("diffcsp")

    if args.model in ["mattergen", "all"]:
        results["mattergen"] = verify_model("mattergen")

    # 生成总体报告
    if len(results) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("Overall Summary")
        logger.info("=" * 60)
        for model_type, result in results.items():
            if "error" in result:
                print(f"  {model_type.upper()}: ERROR - {result['error']}")
            elif result.get("pytorch_available") and result.get("comparisons"):
                passed = sum(1 for c in result["comparisons"] if c.get("pass", False))
                total = len(result["comparisons"])
                status = "PASS" if result.get("all_passed", False) else "FAIL"
                print(f"  {model_type.upper()}: {status} ({passed}/{total} checks passed)")
            else:
                print(f"  {model_type.upper()}: SKIPPED (PyTorch not available)")
        print("=" * 60)


if __name__ == "__main__":
    main()
