#!/usr/bin/env python3
"""
DiffCSP 最终层推理 + 输出diff对比
==========================================


流程：
  1. 用固定随机种子生成确定性测试输入，保存到 tmp/ 供两侧共享
  2. 用 subprocess 调用 matinvent 环境运行 PyTorch 推理，结果写到 tmp/
  3. 在 ppmat 环境中用 Paddle 推理
  4. 对比 pred_l / pred_x 最终输出

特别注意：
1. 我们把 matinvent 通过软连接的方式，软链接到 raw-matinvent；
2. 使用miniconda创建了 matinvent 的py环境，并且pip install -e 。 处理好了 matinvent 环境，并可以正常运行。
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

# 依赖于diffcsp_convert_to_paddle.py 脚本提前导出一份
PD_CONVERTED_CKPT  = PROJECT_ROOT / "tmp" / "matinvent_diffcsp_mp20.pdparams"

# https://huggingface.co/jwchen25/MatInvent/tree/main/diffcsp_mp20
# 如果运行原版代码，会自动下载，或者手动到hg下下载放好
PT_CKPT            = Path("~/.cache/huggingface/hub/models--jwchen25--MatInvent"
                          "/snapshots/b192d079a52d16403fcbbb26b73078d23f9e86c2"
                          "/diffcsp_mp20/last.ckpt").expanduser()

# 配置conda环境的 matinvent 准备对比
RAW_MATINVENT_ROOT = PROJECT_ROOT.parent / "raw-matinvent"
MATINVENT_PYTHON   = Path("~/miniconda3/envs/matinvent/bin/python").expanduser()

OUTPUT_DIR = PROJECT_ROOT / "tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHARED_INPUT_JSON  = OUTPUT_DIR / "diffcsp_final_layer_fixed_input.json"
PT_OUTPUT_JSON     = OUTPUT_DIR / "diffcsp_final_pytorch_output.json"
OUT_JSON           = OUTPUT_DIR / "diffcsp_final_layer_infer_and_diff_report.json"
OUT_MD             = OUTPUT_DIR / "diffcsp_final_layer_infer_and_diff_report.md"

# 固定随机种子，保证确定性
RANDOM_SEED = 42


# 生成固定的输入，用来给 paddle和原版的同时输入，方便对比
def build_fixed_test_input(seed: int = RANDOM_SEED) -> Dict:
    """
    生成确定性的测试输入，固定随机种子，确保每次运行完全一致。
    batch_size=4, 每个晶体的原子数固定为 [8, 12, 10, 6]
    """
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
    time_emb_single = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)  # (1, 256)
    time_emb = np.tile(time_emb_single, (batch_size, 1)).astype(np.float32)  # (4, 256)

    # 原子类型（one-hot, smooth=True 模式）
    atom_type_probs = np.zeros((total_atoms, 100), dtype=np.float32)
    for i_batch, num in enumerate(num_atoms_list):
        start = sum(num_atoms_list[:i_batch])
        for j in range(start, start + num):
            atom_idx = rng.randint(0, 100)
            atom_type_probs[j, atom_idx] = 1.0

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
            "t_value":     t_value,
            "source":      "fixed_seed",
        },
        "time_emb":        time_emb.tolist(),
        "atom_type_probs": atom_type_probs.tolist(),
        "frac_coords":     frac_coords.tolist(),
        "lattices":        lattices.tolist(),
        "num_atoms":       num_atoms_list,
        "batch_idx":       batch_idx,
    }


# PyTorch 推理子脚本路径（在 matinvent 环境下执行）
_PT_RUNNER_SCRIPT = Path(__file__).parent / "diffcsp_final_layer_pt_runner.py"


def run_pytorch_inference(
    shared_input_json: Path,
    pt_output_json: Path,
) -> Optional[Dict]:
    """
    通过 subprocess 在 matinvent 环境下运行 PyTorch CSPNet 推理。
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


# ── Paddle 模型加载 + 前向 ─────────────────────────────────────────

def load_paddle_model(ckpt_path: Path) -> paddle.nn.Layer:
    from ppmat.models.diffcsp.diffcsp import CSPNet
    decoder = CSPNet(
        hidden_dim=512, latent_dim=256, num_layers=6,
        act_fn="silu", dis_emb="sin", num_freqs=128,
        edge_style="fc", ln=True, ip=True, smooth=True,
        pred_type=False, prop_dim=512, pred_scalar=False, num_classes=100,
    )
    decoder.eval()
    decoder.set_state_dict(paddle.load(str(ckpt_path)))
    logger.info("  Paddle model loaded")
    return decoder


def run_paddle_forward(
    model: paddle.nn.Layer,
    test_input: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    with paddle.no_grad():
        pred_l, pred_x = model(
            paddle.to_tensor(np.array(test_input["time_emb"],        dtype=np.float32)),
            paddle.to_tensor(np.array(test_input["atom_type_probs"], dtype=np.float32)),
            paddle.to_tensor(np.array(test_input["frac_coords"],     dtype=np.float32)),
            paddle.to_tensor(np.array(test_input["lattices"],        dtype=np.float32)),
            paddle.to_tensor(np.array(test_input["num_atoms"],       dtype=np.int64)),
            paddle.to_tensor(np.array(test_input["batch_idx"],       dtype=np.int64)),
        )

    pl = pred_l.numpy()
    px = pred_x.numpy()
    logger.info(f"  pred_l: shape={pl.shape}, mean={pl.mean():.6f}")
    logger.info(f"  pred_x: shape={px.shape}, mean={px.mean():.6f}")
    return pl, px


# ── 对比工具 ──────────────────────────────────────────────────────────────────

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
    rel_diff = abs_diff / (np.abs(pt) + 1e-8)
    total = int(abs_diff.size)
    result: Dict[str, Any] = {
        "label": label,
        "shape": list(pt.shape),
        "total_elements": total,
        "abs_diff": {
            "max":    float(abs_diff.max()),
            "mean":   float(abs_diff.mean()),
            "std":    float(abs_diff.std()),
            "median": float(np.median(abs_diff)),
        },
        "rel_diff": {
            "max":    float(rel_diff.max()),
            "mean":   float(rel_diff.mean()),
            "median": float(np.median(rel_diff)),
        },
        "percentiles": {
            "p50":    float(np.percentile(abs_diff, 50)),
            "p90":    float(np.percentile(abs_diff, 90)),
            "p95":    float(np.percentile(abs_diff, 95)),
            "p99":    float(np.percentile(abs_diff, 99)),
            "p99.9":  float(np.percentile(abs_diff, 99.9)),
        },
        "thresholds": {
            "lt_1e_4": int(np.sum(abs_diff < 1e-4)),
            "lt_1e_5": int(np.sum(abs_diff < 1e-5)),
            "lt_1e_6": int(np.sum(abs_diff < 1e-6)),
            "lt_1e_7": int(np.sum(abs_diff < 1e-7)),
            "lt_1e_8": int(np.sum(abs_diff < 1e-8)),
        },
    }
    result["thresholds_pct"] = {k: 100.0 * v / total for k, v in result["thresholds"].items()}
    result["pass_1e-4"] = result["abs_diff"]["mean"] < 1e-4
    result["pass_1e-6"] = result["abs_diff"]["mean"] < 1e-6
    logger.info(f"  mean_abs_diff={result['abs_diff']['mean']:.4e}, "
                f"pass@1e-4={result['pass_1e-4']}, pass@1e-6={result['pass_1e-6']}")
    return result


# ── 报告生成 ──────────────────────────────────────────────────────────────────

def _md_table(rows: list, headers=("项目", "值")) -> str:
    h0, h1 = headers
    lines = [f"| {h0} | {h1} |", "|------|-----|"]
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
            ("总元素数",           f"{cmp['total_elements']:,}"),
            ("最大绝对差异",       f"{cmp['abs_diff']['max']:.4e}"),
            ("**平均绝对差异**",   f"**{cmp['abs_diff']['mean']:.4e}**"),
            ("中位数绝对差异",     f"{cmp['abs_diff']['median']:.4e}"),
            ("P90",                f"{cmp['percentiles']['p90']:.4e}"),
            ("P95",                f"{cmp['percentiles']['p95']:.4e}"),
            ("P99",                f"{cmp['percentiles']['p99']:.4e}"),
            ("P99.9",              f"{cmp['percentiles']['p99.9']:.4e}"),
            ("< 1e-4",             f"{cmp['thresholds']['lt_1e_4']:,} ({cmp['thresholds_pct']['lt_1e_4']:.2f}%)"),
            ("< 1e-6",             f"{cmp['thresholds']['lt_1e_6']:,} ({cmp['thresholds_pct']['lt_1e_6']:.2f}%)"),
            ("Pass @ 1e-4 (mean)", "PASS" if cmp["pass_1e-4"] else "FAIL"),
            ("Pass @ 1e-6 (mean)", "PASS" if cmp["pass_1e-6"] else "FAIL"),
        ]),
        "",
    ]
    if cmp["pass_1e-6"]:
        lines += ["[PASS] 高度一致（mean_diff < 1e-6）", ""]
    elif cmp["pass_1e-4"]:
        lines += ["[WARN] 基本一致（mean_diff < 1e-4）", ""]
    else:
        lines += [f"[FAIL] 差异过大（mean_diff = {cmp['abs_diff']['mean']:.4e}）", ""]
    return lines


def generate_markdown_report(
    test_input: Dict,
    pd_pl: np.ndarray, pd_px: np.ndarray,
    pt_pl: Optional[np.ndarray], pt_px: Optional[np.ndarray],
    pl_cmp: Optional[Dict],
    px_cmp: Optional[Dict],
) -> str:
    meta = test_input["metadata"]
    lines = [
        "# DiffCSP 最终层推理 + 元素级对比报告（固定输入）",
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
            ("Timestep",    str(meta["t_value"])),
        ]),
        "",
        "## Paddle 推理输出统计",
        "",
        _md_table([
            ("pred_l shape", str(list(pd_pl.shape))),
            ("pred_l mean",  f"{pd_pl.mean():.6f}"),
            ("pred_l std",   f"{pd_pl.std():.6f}"),
            ("pred_x shape", str(list(pd_px.shape))),
            ("pred_x mean",  f"{pd_px.mean():.6f}"),
            ("pred_x std",   f"{pd_px.std():.6f}"),
        ]),
        "",
    ]

    if pt_pl is not None:
        lines += [
            "## PyTorch 推理输出统计",
            "",
            _md_table([
                ("pred_l shape", str(list(pt_pl.shape))),
                ("pred_l mean",  f"{pt_pl.mean():.6f}"),
                ("pred_l std",   f"{pt_pl.std():.6f}"),
                ("pred_x shape", str(list(pt_px.shape))),
                ("pred_x mean",  f"{pt_px.mean():.6f}"),
                ("pred_x std",   f"{pt_px.std():.6f}"),
            ]),
            "",
        ]

    lines += _cmp_section("对比 1：pred_l（晶格预测）", pl_cmp)
    lines += _cmp_section("对比 2：pred_x（坐标预测）", px_cmp)

    # 总体结论
    lines += [
        "## 总体结论",
        "",
    ]

    if pl_cmp and px_cmp:
        if pl_cmp.get("pass_1e-6") and px_cmp.get("pass_1e-6"):
            lines += [
                "[PASS] 最终输出高度一致",
                "",
                f"- pred_l 平均差异: {pl_cmp['abs_diff']['mean']:.4e} < 1e-6",
                f"- pred_x 平均差异: {px_cmp['abs_diff']['mean']:.4e} < 1e-6",
                "",
                "PyTorch 和 Paddle 的最终输出完全一致，权重转换成功。",
                "",
            ]
        elif pl_cmp.get("pass_1e-4") and px_cmp.get("pass_1e-4"):
            lines += [
                "[WARN] 最终输出基本一致",
                "",
                f"- pred_l 平均差异: {pl_cmp['abs_diff']['mean']:.4e} < 1e-4",
                f"- pred_x 平均差异: {px_cmp['abs_diff']['mean']:.4e} < 1e-4",
                "",
                "精度达到 1e-4 级别，基本满足要求。",
                "",
            ]
        else:
            lines += [
                "[FAIL] 最终输出差异过大",
                "",
                f"- pred_l 平均差异: {pl_cmp.get('abs_diff', {}).get('mean', 'N/A')}",
                f"- pred_x 平均差异: {px_cmp.get('abs_diff', {}).get('mean', 'N/A')}",
                "",
                "请检查权重转换和模型实现。",
                "",
            ]

    lines += [
        "---",
        "",
        "## 说明",
        "",
        "- pred_l: 晶格参数预测输出，shape=[batch_size, 3, 3]",
        "- pred_x: 原子坐标预测输出，shape=[total_atoms, 3]",
        "- 输入数据: 使用固定种子运行时生成，确保可重复性",
        "- 对比方法: 使用相同的输入数据，分别运行 PyTorch 和 Paddle 模型",
        "- 精度标准: mean_diff < 1e-4 为合格，< 1e-6 为优秀",
        "",
        "---",
        "",
        "*报告生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*",
    ]
    return "\n".join(lines)


def main():
    logger.info("=" * 60)
    logger.info("DiffCSP 最终层 infer+diff（固定输入,paddle+原版推理 并diff）")
    logger.info("=" * 60)

    # [1] 检查 Paddle 权重
    logger.info("\n[1/5] Checking Paddle weights...")
    if not PD_CONVERTED_CKPT.exists():
        logger.error(f"Paddle 权重未发现: {PD_CONVERTED_CKPT}")
        logger.error("请先运行 diffcsp_convert_to_paddle.py")
        sys.exit(1)

    # [2] 生成固定输入
    logger.info("\n[2/5] Generating fixed test inputs (seed=%d)...", RANDOM_SEED)
    test_input = build_fixed_test_input(RANDOM_SEED)
    with open(SHARED_INPUT_JSON, "w") as f:
        json.dump(test_input, f)
    logger.info(f"  Shared input saved: {SHARED_INPUT_JSON}")

    # [3] PyTorch 推理（subprocess）
    logger.info("\n[3/5] Running PyTorch inference (via matinvent env subprocess)...")
    pt_result = run_pytorch_inference(SHARED_INPUT_JSON, PT_OUTPUT_JSON)

    pt_pl: Optional[np.ndarray] = None
    pt_px: Optional[np.ndarray] = None
    if pt_result is not None:
        pt_pl = np.array(pt_result["pred_l"], dtype=np.float32)
        pt_px = np.array(pt_result["pred_x"], dtype=np.float32)
        logger.info(f"  PT pred_l: shape={pt_pl.shape}, mean={pt_pl.mean():.6f}")
        logger.info(f"  PT pred_x: shape={pt_px.shape}, mean={pt_px.mean():.6f}")
    else:
        logger.error("  PyTorch inference skipped, comparison will be partial")

    # [4] Paddle 推理
    logger.info("\n[4/5] Running Paddle inference...")
    pd_model = load_paddle_model(PD_CONVERTED_CKPT)
    pd_pl, pd_px = run_paddle_forward(pd_model, test_input)

    # [5] 对比
    logger.info("\n[5/5] Comparing outputs...")
    pl_cmp = compare_arrays(pt_pl, pd_pl, "pred_l") if pt_pl is not None else None
    px_cmp = compare_arrays(pt_px, pd_px, "pred_x") if pt_px is not None else None

    # ── 保存报告 ──────────────────────────────────────────────────────────────
    logger.info("\nSaving reports...")
    report_data = {
        "metadata": {
            "timestamp":         datetime.now().isoformat(),
            "input_seed":        RANDOM_SEED,
            "pd_converted_ckpt": str(PD_CONVERTED_CKPT),
            "pt_ckpt":           str(PT_CKPT),
            "shared_input":      str(SHARED_INPUT_JSON),
        },
        "test_input_meta":      test_input["metadata"],
        "paddle_pred_l_stats":  _stats(pd_pl),
        "paddle_pred_x_stats":  _stats(pd_px),
        "pytorch_pred_l_stats": _stats(pt_pl) if pt_pl is not None else None,
        "pytorch_pred_x_stats": _stats(pt_px) if pt_px is not None else None,
        "pred_l_comparison":    pl_cmp,
        "pred_x_comparison":    px_cmp,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    logger.info(f"  JSON report: {OUT_JSON}")

    md = generate_markdown_report(
        test_input, pd_pl, pd_px, pt_pl, pt_px,
        pl_cmp, px_cmp,
    )
    with open(OUT_MD, "w") as f:
        f.write(md)
    logger.info(f"  MD   report: {OUT_MD}")

    # ── 终端摘要 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Input        : fixed seed={RANDOM_SEED}, "
          f"batch={test_input['metadata']['batch_size']}, "
          f"atoms={test_input['metadata']['total_atoms']}")
    print(f"  Paddle pred_l: {pd_pl.shape}  pred_x: {pd_px.shape}")
    if pt_pl is not None:
        print(f"  PyTorch pred_l: {pt_pl.shape}  pred_x: {pt_px.shape}")

    def _summary_line(label: str, cmp: Optional[Dict]) -> str:
        if cmp is None:
            return f"  [{label}] skipped (PyTorch not available)"
        if "error" in cmp:
            return f"  [{label}] ERROR: {cmp['error']}"
        status = "PASS" if cmp["pass_1e-4"] else "FAIL"
        return (f"  [{label}] mean_diff={cmp['abs_diff']['mean']:.4e}  "
                f"pass@1e-4={cmp['pass_1e-4']}  pass@1e-6={cmp['pass_1e-6']}  [{status}]")

    print(_summary_line("pred_l", pl_cmp))
    print(_summary_line("pred_x", px_cmp))
    print(f"  JSON : {OUT_JSON}")
    print(f"  MD   : {OUT_MD}")
    print("=" * 60)


if __name__ == "__main__":
    main()
