#!/usr/bin/env python3
"""
MatterGen 全量晶体推理 + diff 对比
=====================================

完整的端到端 denoising 过程对比：
  1. 用 Paddle (ppmat 环境) 进行晶体生成（完整 denoising 推理）
  2. 通过 subprocess 调用 PT (matinvent 环境) 进行晶体生成
  3. 对比生成的晶体结构 (frac_coords / lattice / atom_types)

注意事项：
  - 固定随机种子 (SEED=42) 以保证可复现性
  - 使用较少推理步数 (N_STEPS=20) 减少运行时间和浮点累积误差
  - 两侧 1000 步推理误差会累积但趋势应一致
  - 本脚本假设运行在 ppmat conda 环境中

特别说明：
  - raw-matinvent 目录内容不可修改
  - 当前项目的核心框架代码不可修改
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import paddle

# -----------------------------------------------------------------
# 路径配置
# -----------------------------------------------------------------
PROJECT_ROOT       = Path(__file__).parent.parent.parent    # diff-matinvent/
PPMAT_ROOT         = PROJECT_ROOT.parent                    # PaddleMaterials/
sys.path.insert(0, str(PPMAT_ROOT))

RAW_MATINVENT_ROOT = PPMAT_ROOT / "raw-matinvent"
MATINVENT_PYTHON   = Path("~/miniconda3/envs/matinvent/bin/python").expanduser()

PD_CKPT            = PROJECT_ROOT / "tmp" / "matinvent_mattergen_mp20.pdparams"

OUTPUT_DIR         = PROJECT_ROOT / "tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PT_OUTPUT_JSON     = OUTPUT_DIR / "all_pytorch_result.json"
DIFF_REPORT_JSON   = OUTPUT_DIR / "all_infer_and_diff_report.json"
DIFF_REPORT_MD     = OUTPUT_DIR / "all_infer_and_diff_report.md"

_PT_RUNNER_SCRIPT  = Path(__file__).parent / "all_pt_runner.py"

# -----------------------------------------------------------------
# 推理参数（两侧保持一致）
# -----------------------------------------------------------------
SEED        = 42
N_STEPS     = 1000         # 完整 denoising 步数（与训练保持一致）
NUM_ATOMS   = [6, 8]       # 每个样本的原子数列表
USE_SHARED_RNG = True      # 是否使用共享 numpy RNG 同步两侧随机数

# -----------------------------------------------------------------
# 日志
# -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


# =================================================================
# 共享 Numpy RNG — 用于同步两侧随机数
# =================================================================

class _NumpyRNGWrapper:
    """
    用 NumPy 替代框架原生随机函数，使 Paddle 和 PyTorch 两侧
    在完全相同的随机数序列下运行，从而让两侧生成极为相近的晶体结构。

    工作原理：
    - 调用 reset(seed) 后，后续所有 randn / rand 均来自该 numpy RNG
    - 由于 PT 和 Paddle 的 prior_sampling / step_correct / step_pred
      以相同顺序调用 randn（已通过代码追踪验证），故两侧会消耗
      完全相同的随机数序列

    随机数调用顺序（已验证）：
    Prior sampling：
      1. lattice/cell (18 值): paddle.randn(shape=(2,3,3)) / torch.randn(2,1,3,3)
      2. frac_coords/pos (42 值): paddle.randn(shape=(14,3)) / torch.randn(14,3)
      3. atom_types: D3PM Categorical (不用 randn，跳过)
    每推理步 (x1000)：
      4. coord corrector  (42 值)
      5. lattice corrector (18 值)
      6. coord predictor  (42 值)
      7. lattice predictor (18 值)
      8. atom predictor: D3PM Categorical (不用 randn，跳过)
    """

    def __init__(self, seed: int):
        self._seed = seed
        self.rng = np.random.RandomState(seed)
        self.call_count = 0
        self.total_values = 0

    def reset(self, seed: int | None = None):
        s = seed if seed is not None else self._seed
        self.rng = np.random.RandomState(s)
        self.call_count = 0
        self.total_values = 0

    def randn(self, shape) -> np.ndarray:
        self.call_count += 1
        n = int(np.prod(shape))
        self.total_values += n
        return self.rng.randn(*[int(s) for s in shape]).astype(np.float32)


_numpy_rng = _NumpyRNGWrapper(SEED)


def _install_paddle_randn_patch():
    """将 paddle.randn 替换为 numpy-backed 版本，返回 (is_patched, orig_fn) 。"""
    orig = paddle.randn

    def _np_paddle_randn(shape, dtype=None, name=None):
        arr = _numpy_rng.randn(shape)
        t = paddle.to_tensor(arr)
        if dtype is not None and dtype not in ('float32', paddle.float32):
            t = paddle.cast(t, dtype)
        return t

    paddle.randn = _np_paddle_randn
    return orig


def _restore_paddle_randn(orig_fn):
    paddle.randn = orig_fn


# -----------------------------------------------------------------
# Paddle Categorical 补丁 —— 同步 D3PM atom_types 采样
# -----------------------------------------------------------------

# 用于统计 Categorical 调用次数
_paddle_cat_call_count = 0


def _install_paddle_categorical_patch():
    """将 paddle.distribution.Categorical.sample 替换为 numpy-backed 版本。

    每次调用 Categorical.sample() 时，从 _numpy_rng.rng 抽取等量随机数，
    保证两侧（Paddle/PT）的 D3PM atom_types 采样序列完全一致。
    """
    global _paddle_cat_call_count
    _paddle_cat_call_count = 0
    orig = paddle.distribution.Categorical.sample

    def _np_categorical_sample(self_cat, shape=()):
        global _paddle_cat_call_count
        _paddle_cat_call_count += 1
        # 从 logits 计算归一化概率
        logits = self_cat.logits.numpy()          # (N, C) float32
        logits_shifted = logits - logits.max(-1, keepdims=True)
        probs = np.exp(logits_shifted)
        probs /= probs.sum(-1, keepdims=True)
        # 逐原子采样（保持与 PT 端相同的调用粒度）
        orig_shape = probs.shape[:-1]
        n_atoms = int(np.prod(orig_shape))
        n_classes = probs.shape[-1]
        flat_probs = probs.reshape(n_atoms, n_classes)
        samples = np.array(
            [_numpy_rng.rng.choice(n_classes, p=flat_probs[i]) for i in range(n_atoms)],
            dtype=np.int64,
        )
        return paddle.to_tensor(samples.reshape(orig_shape))

    paddle.distribution.Categorical.sample = _np_categorical_sample
    return orig


def _restore_paddle_categorical(orig_fn):
    paddle.distribution.Categorical.sample = orig_fn


# =================================================================
# Paddle 推理
# =================================================================

def load_paddle_model(ckpt_path: Path) -> paddle.nn.Layer:
    """加载 Paddle MatterGen 模型"""
    from ppmat.models.mattergen.mattergen import MatterGen

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
    model.eval()
    model.set_state_dict(paddle.load(str(ckpt_path)))
    logger.info("Paddle 模型加载完成")
    return model


def run_paddle_inference(
    model: paddle.nn.Layer,
    num_atoms_list: List[int],
    n_steps: int,
    seed: int,
) -> List[Dict]:
    """
    使用 Paddle MatterGen.sample() 进行晶体生成。
    若 USE_SHARED_RNG=True，则用 numpy 共享 RNG 替换 paddle.randn，
    使两侧随机数序列完全一致。
    """
    # 安装 numpy patch（若启用）
    orig_paddle_randn = None
    orig_paddle_categorical = None
    if USE_SHARED_RNG:
        orig_paddle_randn = _install_paddle_randn_patch()
        orig_paddle_categorical = _install_paddle_categorical_patch()
        _numpy_rng.reset(seed)
        logger.info(f"Paddle 推理：已启用 numpy 共享 RNG + Categorical patch (seed={seed})")
    else:
        paddle.seed(seed)
        np.random.seed(seed)

    logger.info(f"Paddle 推理：num_atoms={num_atoms_list}, n_steps={n_steps}, seed={seed}")

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
            num_inference_steps=n_steps,
            _eps_t=0.001,
            n_step_corrector=1,
        )

    # 恢复原始 paddle.randn 和 Categorical.sample
    if USE_SHARED_RNG and orig_paddle_randn is not None:
        _restore_paddle_randn(orig_paddle_randn)
        if orig_paddle_categorical is not None:
            _restore_paddle_categorical(orig_paddle_categorical)
        logger.info(f"  Paddle numpy RNG 统计：randn调用={_numpy_rng.call_count}, "
                    f"消耗随机数={_numpy_rng.total_values}, "
                    f"categorical调用={_paddle_cat_call_count}")

    results = output["result"]
    logger.info(f"Paddle 推理完成，共 {len(results)} 个结构")
    for i, r in enumerate(results):
        logger.info(
            f"  [Paddle] struct {i}: na={r['num_atoms']} "
            f"atom_types={r['atom_types'][:4]}... "
            f"frac_coords[0]={r['frac_coords'][0]} "
            f"lattice[0]={r['lattice'][0]}"
        )
    return results


# =================================================================
# PyTorch (matinvent) 推理
# =================================================================

def run_pytorch_inference(
    num_atoms_list: List[int],
    n_steps: int,
    seed: int,
    pt_output_json: Path,
) -> Optional[List[Dict]]:
    """
    通过 subprocess 在 matinvent 环境下运行 PyTorch MatterGen 推理。
    返回推理结果列表，失败时返回 None。
    """
    if not MATINVENT_PYTHON.exists():
        logger.warning(f"matinvent python 未找到：{MATINVENT_PYTHON}")
        return None
    if not _PT_RUNNER_SCRIPT.exists():
        logger.error(f"PT runner 脚本未找到：{_PT_RUNNER_SCRIPT}")
        return None

    cmd = [
        str(MATINVENT_PYTHON),
        str(_PT_RUNNER_SCRIPT),
        str(RAW_MATINVENT_ROOT),
        json.dumps(num_atoms_list),
        str(n_steps),
        str(seed),
        str(pt_output_json),
        "1" if USE_SHARED_RNG else "0",   # use_shared_rng flag
    ]
    logger.info(f"执行 PT runner：{' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200,   # 全量 denoising 较慢，给 20 分钟
        )
        for line in (proc.stdout or "").strip().splitlines():
            logger.info(f"  [PT stdout] {line}")
        if proc.returncode != 0:
            logger.error(f"PT subprocess 失败 (exit={proc.returncode})")
            for line in (proc.stderr or "").strip().splitlines()[-30:]:
                logger.error(f"  [PT stderr] {line}")
            return None
    except subprocess.TimeoutExpired:
        logger.error("PT subprocess 超时")
        return None

    if not pt_output_json.exists():
        logger.error(f"PT 输出文件未生成：{pt_output_json}")
        return None

    with open(pt_output_json) as f:
        data = json.load(f)
    return data.get("results", [])


# =================================================================
# 对比逻辑
# =================================================================

def _align_structures(
    pd_structs: List[Dict],
    pt_structs: List[Dict],
) -> List[tuple]:
    """
    按照 num_atoms 对齐两侧的结构列表。
    如果数量相同，按顺序配对；否则尽量按 num_atoms 匹配。
    """
    if len(pd_structs) != len(pt_structs):
        logger.warning(
            f"结构数量不匹配：Paddle={len(pd_structs)}, PT={len(pt_structs)}"
        )
    pairs = []
    for i in range(min(len(pd_structs), len(pt_structs))):
        pairs.append((pd_structs[i], pt_structs[i]))
    return pairs


def compare_arrays(
    pd_arr: np.ndarray,
    pt_arr: np.ndarray,
    label: str,
) -> Dict[str, Any]:
    """对比两个 numpy 数组，返回统计信息"""
    logger.info(f"对比 {label}...")
    if pd_arr.shape != pt_arr.shape:
        msg = f"形状不匹配：Paddle {pd_arr.shape} vs PT {pt_arr.shape}"
        logger.error(f"  {msg}")
        return {"error": msg}

    abs_diff = np.abs(pd_arr.astype(np.float64) - pt_arr.astype(np.float64))
    total = int(abs_diff.size)
    result: Dict[str, Any] = {
        "shape": list(pd_arr.shape),
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
            "lt_1e-1": int(np.sum(abs_diff < 1e-1)),
            "lt_1e-2": int(np.sum(abs_diff < 1e-2)),
            "lt_1e-3": int(np.sum(abs_diff < 1e-3)),
        },
    }
    result["thresholds_pct"] = {
        k: 100.0 * v / total for k, v in result["thresholds"].items()
    }
    # 全量推理误差较大，标准放宽到 0.5
    result["pass_05"]  = result["abs_diff"]["median"] < 0.5
    result["pass_01"]  = result["abs_diff"]["median"] < 0.1
    result["pass_001"] = result["abs_diff"]["median"] < 0.01
    logger.info(
        f"  {label}: median={result['abs_diff']['median']:.4e}, "
        f"pass@0.5={result['pass_05']}, pass@0.1={result['pass_01']}"
    )
    return result


def compare_frac_coords_periodic(
    pd_fc: np.ndarray,
    pt_fc: np.ndarray,
    label: str = "frac_coords_periodic",
) -> Dict[str, Any]:
    """
    周期性感知的分数坐标对比。
    分数坐标在 [0,1) 周期性等价，因此实际差异应为
        d_wrap = min(|a - b|, 1 - |a - b|)
    这样 0.02 和 0.98 的差变为 0.04（而非 0.96）。
    """
    logger.info(f"对比 {label} (周期性感知)...")
    if pd_fc.shape != pt_fc.shape:
        msg = f"形状不匹配：Paddle {pd_fc.shape} vs PT {pt_fc.shape}"
        logger.error(f"  {msg}")
        return {"error": msg}

    raw_diff = np.abs(pd_fc.astype(np.float64) - pt_fc.astype(np.float64))
    wrap_diff = np.minimum(raw_diff, 1.0 - raw_diff)  # 周期性折叠
    total = int(wrap_diff.size)
    result: Dict[str, Any] = {
        "shape": list(pd_fc.shape),
        "total_elements": total,
        "raw_abs_diff": {
            "median": float(np.median(raw_diff)),
            "mean":   float(raw_diff.mean()),
        },
        "periodic_abs_diff": {
            "max":    float(wrap_diff.max()),
            "mean":   float(wrap_diff.mean()),
            "std":    float(wrap_diff.std()),
            "median": float(np.median(wrap_diff)),
        },
        "thresholds": {
            "lt_1e-1": int(np.sum(wrap_diff < 1e-1)),
            "lt_1e-2": int(np.sum(wrap_diff < 1e-2)),
        },
    }
    result["thresholds_pct"] = {
        k: 100.0 * v / total for k, v in result["thresholds"].items()
    }
    result["pass_05"]  = result["periodic_abs_diff"]["median"] < 0.5
    result["pass_01"]  = result["periodic_abs_diff"]["median"] < 0.1
    logger.info(
        f"  {label}: raw_median={result['raw_abs_diff']['median']:.4e}, "
        f"periodic_median={result['periodic_abs_diff']['median']:.4e}, "
        f"pass@0.1={result['pass_01']}"
    )
    return result


def compare_lattice_physics(
    pd_lt: np.ndarray,
    pt_lt: np.ndarray,
    num_atoms: int,
) -> Dict[str, Any]:
    """
    物理量级的晶格对比：体积、归一化体积。
    原始矩阵元素对比对旋转敏感，体积对比更有意义。
    """
    raw = compare_arrays(pd_lt, pt_lt, "lattice_raw")

    pd_vol = float(np.abs(np.linalg.det(pd_lt.astype(np.float64))))
    pt_vol = float(np.abs(np.linalg.det(pt_lt.astype(np.float64))))
    vol_diff = abs(pd_vol - pt_vol)
    vol_rel  = vol_diff / max(pd_vol, pt_vol, 1e-8)

    # 归一化体积（体积 / 原子数，代表平均每个原子占的空间）
    pd_vol_per_atom = pd_vol / num_atoms
    pt_vol_per_atom = pt_vol / num_atoms

    raw["volume"] = {
        "pd_vol":          pd_vol,
        "pt_vol":          pt_vol,
        "vol_diff":        vol_diff,
        "vol_rel_diff":    vol_rel,
        "pd_vol_per_atom": pd_vol_per_atom,
        "pt_vol_per_atom": pt_vol_per_atom,
        "vol_pass_20pct":  vol_rel < 0.20,
        "vol_pass_50pct":  vol_rel < 0.50,
    }
    logger.info(
        f"  lattice volume: pd={pd_vol:.2f} pt={pt_vol:.2f} "
        f"rel_diff={vol_rel:.2%} pass@20%={vol_rel < 0.20}"
    )
    return raw


def compare_crystal_pair(
    pd_s: Dict,
    pt_s: Dict,
) -> Dict[str, Any]:
    """对比一对晶体结构（含周期性感知 frac_coords 和物理晶格对比）"""
    na_pd = pd_s["num_atoms"]
    na_pt = pt_s["num_atoms"]
    comparisons: Dict[str, Any] = {
        "num_atoms_pd": na_pd,
        "num_atoms_pt": na_pt,
        "num_atoms_match": na_pd == na_pt,
    }

    if na_pd != na_pt:
        logger.warning(f"  num_atoms 不匹配 ({na_pd} vs {na_pt})，跳过此对")
        comparisons["error"] = "num_atoms mismatch"
        return comparisons

    pd_fc = np.array(pd_s["frac_coords"], dtype=np.float32)
    pt_fc = np.array(pt_s["frac_coords"], dtype=np.float32)
    pd_lt = np.array(pd_s["lattice"], dtype=np.float32)
    pt_lt = np.array(pt_s["lattice"], dtype=np.float32)

    # 周期性感知分数坐标对比
    comparisons["frac_coords"] = compare_frac_coords_periodic(pd_fc, pt_fc)

    # 物理量级晶格对比（含体积）
    comparisons["lattice"] = compare_lattice_physics(pd_lt, pt_lt, na_pd)

    # atom_types（整数，计算完全匹配率；另外对比 sorted 后的元素组成）
    pd_at = np.array(pd_s["atom_types"], dtype=np.int64)
    pt_at = np.array(pt_s["atom_types"], dtype=np.int64)
    at_match = int(np.sum(pd_at == pt_at))
    at_total = len(pd_at)
    # 元素组成匹配（排序后比较，不依赖顺序）
    composition_match = sorted(pd_at.tolist()) == sorted(pt_at.tolist())
    comparisons["atom_types"] = {
        "total": at_total,
        "exact_match": at_match,
        "exact_match_pct": 100.0 * at_match / at_total if at_total > 0 else 0.0,
        "composition_match": composition_match,
        "pd_values": pd_at.tolist(),
        "pt_values": pt_at.tolist(),
        "pd_sorted": sorted(pd_at.tolist()),
        "pt_sorted": sorted(pt_at.tolist()),
    }
    logger.info(
        f"  atom_types: exact_match={at_match}/{at_total} "
        f"({100.0*at_match/at_total:.1f}%), "
        f"composition_match={composition_match}"
    )

    # KL 散度（使用与 calc_kl_reg 相同的 MSE 公式）
    comparisons["kl"] = compute_pseudo_kl(pd_s, pt_s)
    return comparisons


def compute_pseudo_kl(pd_s: Dict, pt_s: Dict) -> Dict[str, float]:
    """使用与 calc_kl_reg 相同的 MSE 公式计算伪 KL 距离。

    kl0: 晶格矩阵 MSE
    kl1: 分数坐标 per-atom MSE 均值
    kl2: 原子类型 (归一化到 [0,1]) MSE 均值
    """
    pd_lt = np.array(pd_s["lattice"], dtype=np.float32)   # (3,3)
    pt_lt = np.array(pt_s["lattice"], dtype=np.float32)
    kl0 = float(np.mean((pd_lt - pt_lt) ** 2))

    pd_fc = np.array(pd_s["frac_coords"], dtype=np.float32)  # (N,3)
    pt_fc = np.array(pt_s["frac_coords"], dtype=np.float32)
    kl1 = float(np.mean(np.mean((pd_fc - pt_fc) ** 2, axis=-1)))

    pd_at = np.array(pd_s["atom_types"], dtype=np.float32) / 118.0  # 归一化
    pt_at = np.array(pt_s["atom_types"], dtype=np.float32) / 118.0
    kl2 = float(np.mean((pd_at - pt_at) ** 2))

    kl_total = kl0 + kl1 + kl2
    level = "低" if kl_total < 0.5 else ("中" if kl_total < 2.0 else "高")
    return {"kl0": kl0, "kl1": kl1, "kl2": kl2, "kl_total": kl_total, "level": level}


# =================================================================
# 报告生成
# =================================================================

def _md_table(rows: list, headers=("项目", "值")) -> str:
    h0, h1 = headers
    lines = [f"| {h0} | {h1} |", "|------|------|"]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def _cmp_section(title: str, cmp: Optional[Dict], is_int=False) -> list:
    lines = [f"### {title}", ""]
    if cmp is None:
        lines += ["> PyTorch 推理未完成，跳过此对比。", ""]
        return lines
    if "error" in cmp:
        lines += [f"[FAIL] 对比失败: {cmp['error']}", ""]
        return lines

    if is_int:
        # atom_types
        comp_match = cmp.get("composition_match", False)
        lines += [
            _md_table([
                ("总原子数",            f"{cmp['total']}"),
                ("完全匹配数 (顺序)",   f"{cmp['exact_match']}"),
                ("完全匹配率",          f"{cmp['exact_match_pct']:.2f}%"),
                ("元素组成相同 (无序)", "YES" if comp_match else "NO"),
                ("Paddle 原子类型",     str(cmp['pd_values'][:8]) + ("..." if len(cmp['pd_values']) > 8 else "")),
                ("PT 原子类型",         str(cmp['pt_values'][:8]) + ("..." if len(cmp['pt_values']) > 8 else "")),
                ("Paddle 排序后",       str(cmp.get('pd_sorted', [])[:8])),
                ("PT 排序后",           str(cmp.get('pt_sorted', [])[:8])),
            ]),
            "",
        ]
        status = "PASS" if comp_match else ("OK" if cmp["exact_match_pct"] >= 30.0 else "WARN")
        lines += [f"[{status}] 原子类型顺序匹配率={cmp['exact_match_pct']:.1f}%, 组成相同={comp_match}", ""]
    elif "periodic_abs_diff" in cmp:
        # 周期性感知的 frac_coords
        pd = cmp["periodic_abs_diff"]
        rd = cmp["raw_abs_diff"]
        lines += [
            _md_table([
                ("形状",                  str(cmp["shape"])),
                ("总元素数",              f"{cmp['total_elements']:,}"),
                ("原始中位数差异",        f"{rd['median']:.4e}"),
                ("**周期性中位数差异**",  f"**{pd['median']:.4e}**"),
                ("周期性最大差异",        f"{pd['max']:.4e}"),
                ("< 0.1 (周期性)",        f"{cmp['thresholds']['lt_1e-1']:,} ({cmp['thresholds_pct']['lt_1e-1']:.2f}%)"),
                ("< 0.01 (周期性)",       f"{cmp['thresholds']['lt_1e-2']:,} ({cmp['thresholds_pct']['lt_1e-2']:.2f}%)"),
                ("Pass @ periodic<0.5",   "PASS" if cmp["pass_05"] else "FAIL"),
                ("Pass @ periodic<0.1",   "PASS" if cmp["pass_01"] else "FAIL"),
            ]),
            "",
        ]
        if cmp["pass_01"]:
            lines += ["[PASS] 高度一致（周期性 median < 0.1）", ""]
        elif cmp["pass_05"]:
            lines += ["[WARN] 有差异（周期性 median < 0.5）", ""]
        else:
            lines += [f"[FAIL] 差异较大（周期性 median = {pd['median']:.4e}）", ""]
    else:
        # 晶格矩阵
        ab = cmp["abs_diff"]
        vol = cmp.get("volume", {})
        rows = [
            ("形状",               str(cmp["shape"])),
            ("矩阵中位数差异",     f"{ab['median']:.4e}"),
            ("矩阵最大差异",       f"{ab['max']:.4e}"),
        ]
        if vol:
            rows += [
                ("Paddle 体积 (A^3)",   f"{vol['pd_vol']:.2f}"),
                ("PT 体积 (A^3)",       f"{vol['pt_vol']:.2f}"),
                ("**体积相对差异**",    f"**{vol['vol_rel_diff']:.2%}**"),
                ("Paddle 原子体积",     f"{vol['pd_vol_per_atom']:.2f} A^3/atom"),
                ("PT 原子体积",         f"{vol['pt_vol_per_atom']:.2f} A^3/atom"),
                ("Pass @ vol<20%",      "PASS" if vol.get("vol_pass_20pct") else "FAIL"),
                ("Pass @ vol<50%",      "PASS" if vol.get("vol_pass_50pct") else "FAIL"),
            ]
        lines += [_md_table(rows), ""]
        if vol:
            if vol.get("vol_pass_20pct"):
                lines += ["[PASS] 晶格体积一致（相对差 < 20%）", ""]
            elif vol.get("vol_pass_50pct"):
                lines += ["[WARN] 晶格体积近似（相对差 < 50%）", ""]
            else:
                lines += [f"[FAIL] 晶格体积差异大（相对差 = {vol['vol_rel_diff']:.2%}）", ""]
        else:
            if ab["median"] < 0.1:
                lines += ["[PASS] 矩阵一致（median < 0.1）", ""]
            else:
                lines += [f"[FAIL] 矩阵差异较大（median = {ab['median']:.4e}）", ""]
    return lines


def _kl_section(kl: Optional[Dict]) -> list:
    """渲染 KL 散度 Markdown 片段。"""
    lines = ["### KL 散度（伪距离）", ""]
    if kl is None:
        lines += ["> KL 数据缺失。", ""]
        return lines
    lines += [
        _md_table([
            ("晶格 kl0 (MSE)",    f"{kl['kl0']:.6f}"),
            ("坐标 kl1 (MSE)",    f"{kl['kl1']:.6f}"),
            ("原子类型 kl2 (MSE)",f"{kl['kl2']:.6f}"),
            ("**kl_total**",       f"**{kl['kl_total']:.4f}**"),
            ("差异程度",           kl["level"]),
        ]),
        "",
    ]
    badge = {"低": "PASS", "中": "WARN", "高": "FAIL"}.get(kl["level"], "INFO")
    lines += [f"[{badge}] kl_total={kl['kl_total']:.4f} [{kl['level']}]", ""]
    return lines


def generate_markdown_report(
    num_atoms_list: List[int],
    n_steps: int,
    seed: int,
    pd_results: List[Dict],
    pt_results: Optional[List[Dict]],
    pair_comparisons: List[Dict],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# MatterGen 全量晶体推理对比报告",
        "",
        f"> 生成时间：{now}",
        "",
        "## 参数配置",
        "",
        _md_table([
            ("随机种子",    str(seed)),
            ("推理步数 N",  str(n_steps)),
            ("num_atoms",   str(num_atoms_list)),
            ("对比结构数",  str(len(pair_comparisons))),
        ]),
        "",
        "## 基本信息",
        "",
        f"- Paddle 生成结构数: {len(pd_results)}",
        f"- PyTorch 生成结构数: {len(pt_results) if pt_results else '未完成'}",
        "",
    ]

    if pt_results is None:
        lines += [
            "> 注意：PyTorch (matinvent) 推理未能完成，无法进行对比。",
            "> 请确认 matinvent conda 环境及 HuggingFace 模型已正确配置。",
            "",
        ]
        return "\n".join(lines)

    for i, pair_cmp in enumerate(pair_comparisons):
        na = pair_cmp.get("num_atoms_pd", "?")
        lines += [f"## 结构 {i + 1}（num_atoms={na}）", ""]

        if "error" in pair_cmp:
            lines += [f"> [FAIL] {pair_cmp['error']}", ""]
            continue

        if "frac_coords" in pair_cmp:
            lines += _cmp_section("分数坐标 (frac_coords)", pair_cmp["frac_coords"])
        if "lattice" in pair_cmp:
            lines += _cmp_section("晶格矩阵 (lattice)", pair_cmp["lattice"])
        if "atom_types" in pair_cmp:
            lines += _cmp_section("原子类型 (atom_types)", pair_cmp["atom_types"], is_int=True)
        if "kl" in pair_cmp:
            lines += _kl_section(pair_cmp["kl"])

    # KL 汇总表
    kl_rows_exists = [c for c in pair_comparisons if "kl" in c]
    if kl_rows_exists:
        lines += ["## KL 散度汇总", ""]
        kl_rows = []
        for i, c in enumerate(pair_comparisons):
            kl = c.get("kl", {})
            if kl:
                kl_rows.append(
                    f"| 结构 {i+1} (na={c.get('num_atoms_pd','?')}) "
                    f"| {kl['kl0']:.4f} | {kl['kl1']:.4f} | {kl['kl2']:.4f} "
                    f"| **{kl['kl_total']:.4f}** | {kl['level']} |"
                )
        lines += [
            "| 结构 | kl0(晶格) | kl1(坐标) | kl2(原子) | kl_total | 程度 |",
            "|------|-----------|-----------|-----------|----------|------|",
        ] + kl_rows + [""]

    lines += [
        "## 总结",
        "",
        ("> 此对比是全量 denoising 对比。由于随机初始化方式、浮点运算顺序、"
         "框架级差异（PyTorch vs Paddle），两侧的数值会有累积差异。"),
        "> 主要验证：",
        "> 1. 两侧均能正常完成 denoising 生成过程",
        "> 2. 生成的晶格参数数量级一致",
        "> 3. 生成的分数坐标在合理范围内 [0, 1]",
        "",
    ]

    return "\n".join(lines)


# =================================================================
# 主流程
# =================================================================

def main():
    logger.info("=" * 60)
    logger.info("MatterGen 全量晶体推理对比")
    logger.info(f"  seed={SEED}, n_steps={N_STEPS}, num_atoms={NUM_ATOMS}")
    logger.info("=" * 60)

    # ------ 1. Paddle 推理 ------
    logger.info("\n[Step 1] Paddle (ppmat) 推理...")
    if not PD_CKPT.exists():
        logger.error(f"Paddle 检查点未找到：{PD_CKPT}")
        sys.exit(1)

    model = load_paddle_model(PD_CKPT)
    pd_results = run_paddle_inference(model, NUM_ATOMS, N_STEPS, SEED)

    # ------ 2. PyTorch 推理 ------
    logger.info("\n[Step 2] PyTorch (matinvent) 推理...")
    pt_results = run_pytorch_inference(NUM_ATOMS, N_STEPS, SEED, PT_OUTPUT_JSON)

    # ------ 3. 对比 ------
    logger.info("\n[Step 3] 对比两侧结果...")
    pair_comparisons = []
    if pt_results is not None:
        pairs = _align_structures(pd_results, pt_results)
        for pd_s, pt_s in pairs:
            cmp = compare_crystal_pair(pd_s, pt_s)
            pair_comparisons.append(cmp)

    # ------ 4. 保存报告 ------
    report_data = {
        "metadata": {
            "seed": SEED,
            "n_steps": N_STEPS,
            "num_atoms": NUM_ATOMS,
            "timestamp": datetime.now().isoformat(),
        },
        "pd_results": pd_results,
        "pt_results": pt_results,
        "pair_comparisons": pair_comparisons,
    }
    with open(DIFF_REPORT_JSON, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"JSON 报告已保存：{DIFF_REPORT_JSON}")

    md_content = generate_markdown_report(
        NUM_ATOMS, N_STEPS, SEED,
        pd_results, pt_results, pair_comparisons,
    )
    with open(DIFF_REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Markdown 报告已保存：{DIFF_REPORT_MD}")

    # ------ 5. 打印摘要 ------
    logger.info("\n" + "=" * 60)
    logger.info("对比摘要")
    logger.info("=" * 60)
    if pt_results is None:
        logger.info("PyTorch 推理未成功，对比已跳过。")
        logger.info("Paddle 推理结果：")
        for i, r in enumerate(pd_results):
            logger.info(f"  struct {i}: na={r['num_atoms']}, "
                        f"lattice_diag=[{r['lattice'][0][0]:.3f}, "
                        f"{r['lattice'][1][1]:.3f}, {r['lattice'][2][2]:.3f}]")
    else:
        for i, cmp in enumerate(pair_comparisons):
            na = cmp.get("num_atoms_pd", "?")
            logger.info(f"\n结构 {i + 1} (na={na}):")
            if "error" in cmp:
                logger.info(f"  [FAIL] {cmp['error']}")
                continue
            # frac_coords (周期性感知)
            fc = cmp.get("frac_coords", {})
            if "periodic_abs_diff" in fc:
                pm = fc["periodic_abs_diff"]["median"]
                rm = fc["raw_abs_diff"]["median"]
                p = "PASS" if fc["pass_01"] else ("WARN" if fc["pass_05"] else "FAIL")
                logger.info(f"  frac_coords: raw_median={rm:.4e}, periodic_median={pm:.4e} [{p}]")
            # lattice (体积感知)
            lt = cmp.get("lattice", {})
            vol = lt.get("volume", {})
            if vol:
                logger.info(
                    f"  lattice: pd_vol={vol['pd_vol']:.1f} pt_vol={vol['pt_vol']:.1f} "
                    f"rel_diff={vol['vol_rel_diff']:.1%} "
                    f"[{'PASS' if vol['vol_pass_20pct'] else ('WARN' if vol['vol_pass_50pct'] else 'FAIL')}]"
                )
            # atom_types
            at = cmp.get("atom_types", {})
            if at:
                comp = at.get("composition_match", False)
                logger.info(
                    f"  atom_types: 顺序匹配率={at['exact_match_pct']:.1f}%, "
                    f"组成相同={comp}"
                )
            # KL
            kl = cmp.get("kl", {})
            if kl:
                logger.info(
                    f"  kl: kl0={kl['kl0']:.4f} kl1={kl['kl1']:.4f} "
                    f"kl2={kl['kl2']:.4f} total={kl['kl_total']:.4f} [{kl['level']}]"
                )


if __name__ == "__main__":
    main()
