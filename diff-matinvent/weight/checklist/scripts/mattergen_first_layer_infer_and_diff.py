#!/usr/bin/env python3
"""
MatterGen 第一层推理 + 输出diff对比
==========================================

流程：
  1. 用固定随机种子生成确定性测试输入，保存到 tmp/ 供两侧共享
  2. 用 subprocess 调用 matinvent 环境运行 PyTorch 推理，结果写到 tmp/
  3. 在 ppmat 环境中用 Paddle 推理
  4. 对比 first-layer 输出 (atom_emb) 和 pred_lattice / pred_frac_coords / pred_atom_types

特别注意：
1. 我们把 matinvent 通过软连接的方式，软链接到 raw-matinvent；
2. 使用miniconda创建了 matinvent 的py环境，并且pip install -e 。 处理好了 matinvent 环境，并可以正常运行。
3. 当前脚本，假设自身运行在 ppmat的环境里
"""

import json
import logging
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import paddle

PROJECT_ROOT = Path(__file__).parent.parent.parent      # diff-matinvent/
PPMAT_ROOT   = PROJECT_ROOT.parent                      # PaddleMaterials/
sys.path.insert(0, str(PPMAT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# 依赖于mattergen_convert_to_paddle.py 脚本提前导出一份
PD_CONVERTED_CKPT  = PROJECT_ROOT / "tmp" / "matinvent_mattergen_mp20.pdparams"

# https://huggingface.co/microsoft/mattergen
# 如果运行原版代码，会自动下载，或者手动到hg下下载放好
PT_CKPT            = Path("~/.cache/huggingface/hub/models--microsoft--mattergen"
                          "/snapshots/ea430eab64b80855029c2941b9fda15f245a771a"
                          "/checkpoints/mattergen_base/checkpoints/last.ckpt").expanduser()

# 配置conda环境的 matinvent 准备对比
RAW_MATINVENT_ROOT = PROJECT_ROOT.parent / "raw-matinvent"
MATINVENT_PYTHON   = Path("~/miniconda3/envs/matinvent/bin/python").expanduser()

OUTPUT_DIR = PROJECT_ROOT / "tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHARED_INPUT_JSON  = OUTPUT_DIR / "mattergen_infer_and_diff_fixed_input.json"
PT_OUTPUT_JSON     = OUTPUT_DIR / "mattergen_pytorch_output.json"
DIFF_REPORT_OUT_JSON           = OUTPUT_DIR / "mattergen_infer_and_diff_report.json"
DIFF_REPORT_OUT_MD             = OUTPUT_DIR / "mattergen_infer_and_diff_report.md"

# 固定随机种子，保证确定性
RANDOM_SEED = 42


def build_fixed_test_input(seed: int = RANDOM_SEED) -> Dict:
    """
    生成确定性的测试输入，固定随机种子，确保每次运行完全一致。
    batch_size=2, 每个晶体的原子数固定为 [8, 12]
    """
    rng = np.random.RandomState(seed)

    num_atoms_list = [8, 12]
    batch_size = len(num_atoms_list)
    total_atoms = sum(num_atoms_list)

    # 时间步 t（固定为 500，对应 diffusion timestep）
    t_value = 500
    # 归一化到 [0, 1] 范围（MatterGen 使用 max_t=1.0）
    times = np.array([t_value / 1000.0] * batch_size, dtype=np.float32)

    # 原子类型（使用原子序数，1-118）
    atom_types = []
    for i_batch, num in enumerate(num_atoms_list):
        for j in range(num):
            # 随机选择前20个元素（H到Ca）
            atom_types.append(rng.randint(1, 21))
    atom_types = np.array(atom_types, dtype=np.int64)

    # 分数坐标
    frac_coords = rng.rand(total_atoms, 3).astype(np.float32)

    # 晶格矩阵（对角线为 3~8，其余项为 0）
    lattices = np.zeros((batch_size, 3, 3), dtype=np.float32)
    for b in range(batch_size):
        diag = rng.uniform(3.0, 8.0, 3).astype(np.float32)
        lattices[b] = np.diag(diag)

    # batch 索引
    batch_idx: list = []
    for i, num in enumerate(num_atoms_list):
        batch_idx.extend([i] * num)

    return {
        "metadata": {
            "seed":        seed,
            "batch_size":  batch_size,
            "total_atoms": total_atoms,
            "num_atoms":   num_atoms_list,
            "times":       times.tolist(),
            "source":      "fixed_seed",
        },
        "times":           times.tolist(),
        "atom_types":      atom_types.tolist(),
        "frac_coords":     frac_coords.tolist(),
        "lattices":        lattices.tolist(),
        "num_atoms":       num_atoms_list,
        "batch_idx":       batch_idx,
    }


# PyTorch 推理子脚本路径（在 matinvent 环境下执行）
_PT_RUNNER_SCRIPT = Path(__file__).parent / "mattergen_first_layer_pt_runner.py"


def run_pytorch_inference(
    shared_input_json: Path,
    pt_output_json: Path,
) -> Optional[Dict]:
    """
    通过 subprocess 在 matinvent 环境下运行 PyTorch MatterGen 推理。
    返回推理结果字典，失败时返回 None。
    """
    if not MATINVENT_PYTHON.exists():
        logger.warning(f"matinvent python not found: {MATINVENT_PYTHON}")
        return None
    if not PT_CKPT.exists():
        logger.warning(f"PyTorch checkpoint not found: {PT_CKPT}")
        return None
    if not _PT_RUNNER_SCRIPT.exists():
        logger.error(f"PT runner script not found: {_PT_RUNNER_SCRIPT}")
        return None

    cmd = [
        str(MATINVENT_PYTHON),
        str(_PT_RUNNER_SCRIPT),
        str(RAW_MATINVENT_ROOT),
        str(shared_input_json),
        str(PT_CKPT),
        str(pt_output_json),
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

    if not pt_output_json.exists():
        logger.error(f"  PyTorch output not created: {pt_output_json}")
        return None

    with open(pt_output_json) as f:
        return json.load(f)


class _FirstLayerHook:
    """捕获 atom_emb 输出的 forward hook"""
    def __init__(self):
        self.output: Optional[np.ndarray] = None

    def __call__(self, layer, inp, out):
        if isinstance(out, paddle.Tensor):
            self.output = out.detach().clone().numpy()


def load_paddle_model(ckpt_path: Path) -> paddle.nn.Layer:
    # 原始实现: from ppmat.models.mattergen.mattergen import MatterGen
    from ppmat.models.matinvent.mattergen_compat import MatinventMatterGen
    from ppmat.schedulers import LatticeVPSDEScheduler, D3PMScheduler
    from ppmat.schedulers.scheduling_wrapped_sde_ve import NumAtomsVarianceAdjustedWrappedVESDE

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
    logger.info("  Paddle model loaded")
    return model


def run_paddle_forward(
    model: paddle.nn.Layer,
    test_input: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """运行 Paddle 推理，返回 atom_emb, pred_lattice, pred_frac_coords, pred_atom_types"""
    # 构建 structure_array
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

    # 注册 hook 捕获 atom_emb 输出
    hook = _FirstLayerHook()

    def _hook_fn(layer, inp, out):
        if isinstance(out, paddle.Tensor):
            hook.output = out.detach().clone().numpy()

    handle = model.model.gemnet.atom_emb.register_forward_post_hook(_hook_fn)

    with paddle.no_grad():
        # 调用 model 的 forward，传入 times
        times = paddle.to_tensor(np.array(test_input["times"], dtype=np.float32))

        # 手动调用 GemNetTDenoiser 的 forward
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

    logger.info(f"  atom_emb: shape={fl.shape}, mean={fl.mean():.6f}, std={fl.std():.6f}")
    logger.info(f"  pred_lattice: shape={pl.shape}, mean={pl.mean():.6f}")
    logger.info(f"  pred_frac_coords: shape={px.shape}, mean={px.mean():.6f}")
    logger.info(f"  pred_atom_types: shape={pa.shape}, mean={pa.mean():.6f}")

    return fl, pl, px, pa


def _stats(a: np.ndarray) -> Dict:
    return {"shape": list(a.shape), "mean": float(a.mean()),
            "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}


def compare_arrays(pt: np.ndarray, pd: np.ndarray, label: str) -> Dict[str, Any]:
    logger.info(f"Comparing {label}...")
    if pt.shape != pd.shape:
        msg = f"Shape mismatch: PyTorch {pt.shape} vs Paddle {pd.shape}"
        logger.error(f"  {msg}")
        return {"error": msg}

    abs_diff = np.abs(pt - pd)
    total = int(abs_diff.size)
    result: Dict[str, Any] = {
        "shape": list(pt.shape),
        "total_elements": total,
        "abs_diff": {
            "max":    float(abs_diff.max()),
            "mean":   float(abs_diff.mean()),
            "std":    float(abs_diff.std()),
            "median": float(np.median(abs_diff)),
        },
        "percentiles": {
            "p90":   float(np.percentile(abs_diff, 90)),
            "p99":   float(np.percentile(abs_diff, 99)),
            "p99.9": float(np.percentile(abs_diff, 99.9)),
        },
        "thresholds": {
            "lt_1e-4": int(np.sum(abs_diff < 1e-4)),
            "lt_1e-5": int(np.sum(abs_diff < 1e-5)),
            "lt_1e-6": int(np.sum(abs_diff < 1e-6)),
        },
    }
    result["thresholds_pct"] = {k: 100.0 * v / total for k, v in result["thresholds"].items()}
    result["pass_1e-4"] = result["abs_diff"]["median"] < 1e-4
    result["pass_1e-6"] = result["abs_diff"]["median"] < 1e-6
    logger.info(f"  median_abs_diff={result['abs_diff']['median']:.4e}, "
                f"pass@1e-4={result['pass_1e-4']}, pass@1e-6={result['pass_1e-6']}")
    return result


def _md_table(rows: list, headers=("项目", "值")) -> str:
    h0, h1 = headers
    lines = [f"| {h0} | {h1} |", "|------|------|"]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def _cmp_section(title: str, cmp: Optional[Dict]) -> list:
    lines = [f"## {title}", ""]
    if cmp is None:
        lines += ["> PyTorch 推理未完成，跳过此对比。", ""]
        return lines
    if "error" in cmp:
        lines += [f"[FAIL] 对比失败: {cmp['error']}", ""]
        return lines
    lines += [
        _md_table([
            ("总元素数",         f"{cmp['total_elements']:,}"),
            ("最大绝对差异",     f"{cmp['abs_diff']['max']:.4e}"),
            ("平均绝对差异",       f"{cmp['abs_diff']['mean']:.4e}"),
            ("**中位数绝对差异**", f"**{cmp['abs_diff']['median']:.4e}**"),
            ("P99.9",              f"{cmp['percentiles']['p99.9']:.4e}"),
            ("< 1e-4",             f"{cmp['thresholds']['lt_1e-4']:,} ({cmp['thresholds_pct']['lt_1e-4']:.2f}%)"),
            ("< 1e-6",             f"{cmp['thresholds']['lt_1e-6']:,} ({cmp['thresholds_pct']['lt_1e-6']:.2f}%)"),
            ("Pass @ 1e-4 (median)", "PASS" if cmp["pass_1e-4"] else "FAIL"),
            ("Pass @ 1e-6 (median)", "PASS" if cmp["pass_1e-6"] else "FAIL"),
        ]),
        "",
    ]
    if cmp["pass_1e-6"]:
        lines += ["[PASS] 高度一致（median_diff < 1e-6）", ""]
    elif cmp["pass_1e-4"]:
        lines += ["[WARN] 基本一致（median_diff < 1e-4）", ""]
    else:
        lines += [f"[FAIL] 差异过大（median_diff = {cmp['abs_diff']['median']:.4e}）", ""]
    return lines


def generate_markdown_report(
    test_input: Dict,
    pd_fl: np.ndarray, pt_fl: Optional[np.ndarray],
    pd_pl: np.ndarray, pd_px: np.ndarray, pd_pa: np.ndarray,
    pt_pl: Optional[np.ndarray], pt_px: Optional[np.ndarray], pt_pa: Optional[np.ndarray],
    fl_cmp: Optional[Dict],
    pl_cmp: Optional[Dict],
    px_cmp: Optional[Dict],
    pa_cmp: Optional[Dict],
) -> str:
    meta = test_input["metadata"]
    lines = [
        "# MatterGen 第一层推理 + 元素级对比报告（固定输入）",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 测试配置",
        "",
        _md_table([
            ("输入模式",    "固定随机种子（确定性）"),
            ("随机种子",    str(meta["seed"])),
            ("Batch Size",  str(meta["batch_size"])),
            ("Total Atoms", str(meta["total_atoms"])),
            ("每批原子数",  str(meta["num_atoms"])),
            ("Times",       str(meta["times"])),
        ]),
        "",
        "## Paddle 推理输出统计",
        "",
        _md_table([
            ("atom_emb shape",      str(list(pd_fl.shape))),
            ("atom_emb mean",       f"{pd_fl.mean():.6f}"),
            ("pred_lattice shape",  str(list(pd_pl.shape))),
            ("pred_lattice mean",   f"{pd_pl.mean():.6f}"),
            ("pred_frac_coords shape", str(list(pd_px.shape))),
            ("pred_frac_coords mean",  f"{pd_px.mean():.6f}"),
            ("pred_atom_types shape",  str(list(pd_pa.shape))),
            ("pred_atom_types mean",   f"{pd_pa.mean():.6f}"),
        ]),
        "",
    ]

    if pt_fl is not None:
        lines += [
            "## PyTorch 推理输出统计",
            "",
            _md_table([
                ("atom_emb shape",      str(list(pt_fl.shape))),
                ("atom_emb mean",       f"{pt_fl.mean():.6f}"),
                ("pred_lattice shape",  str(list(pt_pl.shape))),
                ("pred_lattice mean",   f"{pt_pl.mean():.6f}"),
                ("pred_frac_coords shape", str(list(pt_px.shape))),
                ("pred_frac_coords mean",  f"{pt_px.mean():.6f}"),
                ("pred_atom_types shape",  str(list(pt_pa.shape))),
                ("pred_atom_types mean",   f"{pt_pa.mean():.6f}"),
            ]),
            "",
        ]

    lines += _cmp_section("对比 1：atom_emb（第一层输出）", fl_cmp)
    lines += _cmp_section("对比 2：pred_lattice（晶格预测）", pl_cmp)
    lines += _cmp_section("对比 3：pred_frac_coords（坐标预测）", px_cmp)
    lines += _cmp_section("对比 4：pred_atom_types（原子类型预测）", pa_cmp)
    lines += [
        "---",
        "",
        "*报告生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*",
    ]
    return "\n".join(lines)


def main():
    logger.info("=" * 60)
    logger.info("MatterGen 第一层 infer+diff（固定输入,paddle+原版推理 并diff）")
    logger.info("=" * 60)

    # [1] 检查 Paddle 权重
    logger.info("\n[1/5] Checking Paddle weights...")
    if not PD_CONVERTED_CKPT.exists():
        logger.error(f"Paddle 权重未发现: {PD_CONVERTED_CKPT}")
        logger.error("请先运行 mattergen_convert_to_paddle.py")
        sys.exit(1)

    # [2] 生成固定输入
    logger.info("\n[2/5] Generating fixed test inputs (seed=%d)...", RANDOM_SEED)
    test_input = build_fixed_test_input(RANDOM_SEED)
    with open(SHARED_INPUT_JSON, "w") as f:
        json.dump(test_input, f)
    logger.info(f"  Shared input saved: {SHARED_INPUT_JSON}")

    # [3] PyTorch 推理（subprocess）,避免2个环境ppmat和原版直接耦合环境在，造成干扰
    logger.info("\n[3/5] Running PyTorch inference (via matinvent env subprocess)...")
    pt_result = run_pytorch_inference(SHARED_INPUT_JSON, PT_OUTPUT_JSON)

    pt_fl: Optional[np.ndarray] = None
    pt_pl: Optional[np.ndarray] = None
    pt_px: Optional[np.ndarray] = None
    pt_pa: Optional[np.ndarray] = None
    if pt_result is not None:
        pt_fl = np.array(pt_result["first_layer_output"], dtype=np.float32)
        pt_pl = np.array(pt_result["pred_lattice"],        dtype=np.float32)
        pt_px = np.array(pt_result["pred_frac_coords"],    dtype=np.float32)
        pt_pa = np.array(pt_result["pred_atom_types"],     dtype=np.float32)
        logger.info(f"  PT atom_emb: shape={pt_fl.shape}, mean={pt_fl.mean():.6f}")
        logger.info(f"  PT pred_lattice: shape={pt_pl.shape}, mean={pt_pl.mean():.6f}")
        logger.info(f"  PT pred_frac_coords: shape={pt_px.shape}, mean={pt_px.mean():.6f}")
        logger.info(f"  PT pred_atom_types: shape={pt_pa.shape}, mean={pt_pa.mean():.6f}")
    else:
        logger.error("  PyTorch inference skipped, comparison will be partial")

    # [4] Paddle自己的模块 推理
    logger.info("\n[4/5] Running Paddle inference...")
    pd_model = load_paddle_model(PD_CONVERTED_CKPT)
    pd_fl, pd_pl, pd_px, pd_pa = run_paddle_forward(pd_model, test_input)

    # [5] 对比
    logger.info("\n[5/5] Comparing outputs...")
    fl_cmp = compare_arrays(pt_fl, pd_fl, "atom_emb")         if pt_fl is not None else None
    pl_cmp = compare_arrays(pt_pl, pd_pl, "pred_lattice")     if pt_pl is not None else None
    px_cmp = compare_arrays(pt_px, pd_px, "pred_frac_coords") if pt_px is not None else None
    pa_cmp = compare_arrays(pt_pa, pd_pa, "pred_atom_types")  if pt_pa is not None else None

    logger.info("\nSaving reports...")
    report_data = {
        "metadata": {
            "timestamp":         datetime.now().isoformat(),
            "input_seed":        RANDOM_SEED,
            "pd_converted_ckpt": str(PD_CONVERTED_CKPT),
            "pt_ckpt":           str(PT_CKPT),
            "shared_input":      str(SHARED_INPUT_JSON),
        },
        "test_input_meta":            test_input["metadata"],
        "paddle_first_layer_stats":   _stats(pd_fl),
        "paddle_pred_lattice_stats":  _stats(pd_pl),
        "paddle_pred_frac_coords_stats": _stats(pd_px),
        "paddle_pred_atom_types_stats":  _stats(pd_pa),
        "pytorch_first_layer_stats":  _stats(pt_fl) if pt_fl is not None else None,
        "pytorch_pred_lattice_stats": _stats(pt_pl) if pt_pl is not None else None,
        "pytorch_pred_frac_coords_stats": _stats(pt_px) if pt_px is not None else None,
        "pytorch_pred_atom_types_stats":  _stats(pt_pa) if pt_pa is not None else None,
        "first_layer_comparison":     fl_cmp,
        "pred_lattice_comparison":    pl_cmp,
        "pred_frac_coords_comparison": px_cmp,
        "pred_atom_types_comparison":  pa_cmp,
    }
    with open(DIFF_REPORT_OUT_JSON, "w") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    logger.info(f"  JSON report: {DIFF_REPORT_OUT_JSON}")

    md = generate_markdown_report(
        test_input, pd_fl, pt_fl, pd_pl, pd_px, pd_pa, pt_pl, pt_px, pt_pa,
        fl_cmp, pl_cmp, px_cmp, pa_cmp,
    )
    with open(DIFF_REPORT_OUT_MD, "w") as f:
        f.write(md)
    logger.info(f"  MD   report: {DIFF_REPORT_OUT_MD}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Input        : fixed seed={RANDOM_SEED}, "
          f"batch={test_input['metadata']['batch_size']}, "
          f"atoms={test_input['metadata']['total_atoms']}")
    print(f"  Paddle pred_lattice: {pd_pl.shape}  pred_frac_coords: {pd_px.shape}  pred_atom_types: {pd_pa.shape}")
    if pt_fl is not None:
        print(f"  PyTorch pred_lattice: {pt_pl.shape}  pred_frac_coords: {pt_px.shape}  pred_atom_types: {pt_pa.shape}")

    def _summary_line(label: str, cmp: Optional[Dict]) -> str:
        if cmp is None:
            return f"  [{label}] skipped (PyTorch not available)"
        if "error" in cmp:
            return f"  [{label}] ERROR: {cmp['error']}"
        status = "PASS" if cmp["pass_1e-4"] else "FAIL"
        return (f"  [{label}] median_diff={cmp['abs_diff']['median']:.4e}  "
                f"pass@1e-4={cmp['pass_1e-4']}  pass@1e-6={cmp['pass_1e-6']}  [{status}]")

    print(_summary_line("atom_emb         ", fl_cmp))
    print(_summary_line("pred_lattice     ", pl_cmp))
    print(_summary_line("pred_frac_coords ", px_cmp))
    print(_summary_line("pred_atom_types  ", pa_cmp))
    print(f"  JSON : {DIFF_REPORT_OUT_JSON}")
    print(f"  MD   : {DIFF_REPORT_OUT_MD}")
    print("=" * 60)


if __name__ == "__main__":
    main()
