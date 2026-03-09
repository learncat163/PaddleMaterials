#!/usr/bin/env python3
"""
DiffCSP 模型映射分析脚本
============================
分析 PyTorch (MatInvent) 和 Paddle (PaddleMaterials) 之间的层映射关系

功能:
1. 加载 PyTorch 模型权重（从 MatInvent checkpoint）
2. 加载 Paddle 模型结构（从 PaddleMaterials）
3. 对比两边的层名称、形状、类型
4. 生成详细的映射关系报告

输出文件:
- tmp/diffcsp_mapping_report.json  # 机器可读的映射数据
- tmp/diffcsp_mapping_report.md    # 人类可读的映射报告

运行环境: ppmat (PaddlePaddle)

用法:
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/diff_mapping.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import paddle

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from ppmat.models import build_model

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# ── 路径配置 ───────────────────────────────────────────────────────────────────
PT_CKPT = Path("/home/cao/.cache/huggingface/hub/models--jwchen25--MatInvent/snapshots/b192d079a52d16403fcbbb26b73078d23f9e86c2/diffcsp_mp20/last.ckpt")
PD_CFG = "structure_generation/configs/diffcsp/diffcsp_mp20.yaml"
OUTPUT_DIR = Path(PROJECT_ROOT) / "tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── PyTorch checkpoint 加载（使用 torch 环境）───────────────────────────────────

def load_pytorch_weights(ckpt_path: Path) -> Dict[str, np.ndarray]:
    """
    加载 PyTorch checkpoint 权重。
    尝试使用 torch，如果不可用则使用预导出的 NPZ 文件。
    """
    logger.info(f"Loading PyTorch weights from: {ckpt_path}")

    # 首先尝试使用 torch 直接加载
    try:
        import torch
        ckpt = torch.load(str(ckpt_path), map_location='cpu')
        sd = ckpt['state_dict']
        result = {}
        for k, v in sd.items():
            result[k] = v.numpy() if hasattr(v, 'numpy') else np.array(v)
        logger.info(f"Loaded {len(result)} PyTorch weights using torch")
        return result
    except ImportError:
        logger.warning("torch not available, trying NPZ fallback...")

    # 备用方案：从预导出的 NPZ 文件加载
    npz_path = OUTPUT_DIR / "diffcsp_pytorch_all_weights.npz"
    if npz_path.exists():
        logger.info(f"Loading from pre-exported NPZ: {npz_path}")
        data = np.load(npz_path)
        result = {}
        for k in data.files:
            original_key = k.replace('__DOT__', '.')
            result[original_key] = data[k]
        logger.info(f"Loaded {len(result)} weights from NPZ")
        return result

    raise RuntimeError(
        "Cannot load PyTorch checkpoint: torch not available and NPZ not found.\n"
        f"Please either:\n"
        f"  1. Run this script in matinvent conda environment (with torch)\n"
        f"  2. Or first export PyTorch weights to NPZ format"
    )


def build_paddle_model(cfg_path: str) -> paddle.nn.Layer:
    """
    构建 Paddle DiffCSP 模型（使用配置文件）
    """
    logger.info(f"Building Paddle model from config: {cfg_path}")

    # 使用 OmegaConf 加载配置
    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # 构建 DiffCSP 模型
    model_config = cfg.get("Model", None)
    if model_config is None:
        raise ValueError("Model config not found in config file")

    model = build_model(model_config)
    model.eval()

    logger.info(f"Paddle model built successfully")
    return model


# ── 映射分析 ────────────────────────────────────────────────────────────────────

class LayerMapping:
    """表示一个层的映射关系"""

    def __init__(
        self,
        pt_key: str,
        pd_key: str,
        pt_shape: Tuple[int, ...],
        pd_shape: Tuple[int, ...],
        transform: str,
        status: str
    ):
        self.pt_key = pt_key
        self.pd_key = pd_key
        self.pt_shape = pt_shape
        self.pd_shape = pd_shape
        self.transform = transform  # "transpose", "copy", "skip", "missing"
        self.status = status  # "matched", "partial", "mismatch", "pytorch_only", "paddle_only"

    def to_dict(self) -> Dict:
        return {
            "pytorch_key": self.pt_key,
            "paddle_key": self.pd_key,
            "pytorch_shape": list(self.pt_shape),
            "paddle_shape": list(self.pd_shape),
            "transform": self.transform,
            "status": self.status
        }


def analyze_mapping(
    pt_weights: Dict[str, np.ndarray],
    pd_model: paddle.nn.Layer
) -> Tuple[List[LayerMapping], Dict[str, Any]]:
    """
    分析 PyTorch 和 Paddle 模型的映射关系

    返回:
        mappings: 层映射列表
        summary: 汇总信息
    """
    mappings = []
    summary = {
        "total_pytorch_layers": len(pt_weights),
        "matched": 0,
        "transposed": 0,
        "copied": 0,
        "skipped": 0,
        "pytorch_only": 0,
        "paddle_only": 0,
        "mismatch": 0
    }

    # 获取 Paddle 模型的 state_dict
    pd_state = pd_model.state_dict()
    pd_keys_set = set(pd_state.keys())

    # 需要跳过的 PyTorch 前缀
    skip_prefixes = [
        'beta_scheduler.',
        'sigma_scheduler.',
        'type_sigma_scheduler.',
    ]

    # 需要跳过的 PyTorch 键（Paddle 没有对应层）
    skip_keys = {
        'decoder.type_out.weight',  # pred_type=False
        'decoder.type_out.bias',
    }

    # Paddle 独有的键（不需要从 PyTorch 转换）
    paddle_only_patterns = ['prop_mlp', 'lattice_scheduler', 'coord_scheduler', 'time_embedding']

    # 分析每个 PyTorch 层
    for pt_key, pt_val in sorted(pt_weights.items()):
        pt_shape = pt_val.shape

        # 检查是否需要跳过
        if any(pt_key.startswith(p) for p in skip_prefixes):
            mappings.append(LayerMapping(
                pt_key, "", pt_shape, (), "skip", "pytorch_only"
            ))
            summary["skipped"] += 1
            summary["pytorch_only"] += 1
            continue

        if pt_key in skip_keys:
            mappings.append(LayerMapping(
                pt_key, "", pt_shape, (), "skip", "pytorch_only"
            ))
            summary["skipped"] += 1
            summary["pytorch_only"] += 1
            continue

        # 直接使用 PyTorch 键查找 Paddle 对应的键
        # Paddle 的键也有 'decoder.' 前缀，所以直接匹配
        pd_key = pt_key

        if pd_key in pd_keys_set:
            pd_val = pd_state[pd_key]
            pd_shape = tuple(pd_val.shape)

            # 检查是否需要转置（Linear 权重）
            if pt_key.endswith('.weight') and len(pt_shape) == 2:
                # Linear 权重需要转置: PyTorch [out, in] -> Paddle [in, out]
                expected_pd_shape = (pt_shape[1], pt_shape[0])
                if pd_shape == expected_pd_shape:
                    mappings.append(LayerMapping(
                        pt_key, pd_key, pt_shape, pd_shape, "transpose", "matched"
                    ))
                    summary["matched"] += 1
                    summary["transposed"] += 1
                else:
                    mappings.append(LayerMapping(
                        pt_key, pd_key, pt_shape, pd_shape, "transpose", "mismatch"
                    ))
                    summary["mismatch"] += 1
            else:
                # bias, LayerNorm, Embedding 等直接复制
                if pd_shape == pt_shape:
                    mappings.append(LayerMapping(
                        pt_key, pd_key, pt_shape, pd_shape, "copy", "matched"
                    ))
                    summary["matched"] += 1
                    summary["copied"] += 1
                else:
                    mappings.append(LayerMapping(
                        pt_key, pd_key, pt_shape, pd_shape, "copy", "mismatch"
                    ))
                    summary["mismatch"] += 1
        else:
            # PyTorch 有但 Paddle 没有的层
            mappings.append(LayerMapping(
                pt_key, "", pt_shape, (), "skip", "pytorch_only"
            ))
            summary["pytorch_only"] += 1

    # 找出 Paddle 独有的层
    for pd_key in sorted(pd_keys_set):
        # 检查是否在 PyTorch 中有对应
        if pd_key not in pt_weights:
            # 检查是否是 Paddle 独有的模式
            if any(pattern in pd_key for pattern in paddle_only_patterns):
                pd_val = pd_state[pd_key]
                pd_shape = tuple(pd_val.shape)
                mappings.append(LayerMapping(
                    "", pd_key, (), pd_shape, "skip", "paddle_only"
                ))
                summary["paddle_only"] += 1

    return mappings, summary


# ── 报告生成 ────────────────────────────────────────────────────────────────────

def generate_json_report(
    mappings: List[LayerMapping],
    summary: Dict[str, Any],
    pt_weights: Dict[str, np.ndarray],
    pd_model: paddle.nn.Layer
) -> Dict:
    """生成 JSON 格式的报告"""
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "pytorch_checkpoint": str(PT_CKPT),
            "paddle_config": PD_CFG,
            "pytorch_total_layers": len(pt_weights),
            "paddle_total_layers": len(pd_model.state_dict())
        },
        "summary": summary,
        "mappings": [m.to_dict() for m in mappings],
        "layer_groups": group_layers_by_type(mappings)
    }
    return report


def group_layers_by_type(mappings: List[LayerMapping]) -> Dict[str, List[Dict]]:
    """按层类型分组"""
    groups = defaultdict(list)

    for m in mappings:
        if m.pt_key == "":
            layer_type = "paddle_only"
        elif m.pd_key == "":
            layer_type = "pytorch_only"
        elif "csp_layer_" in m.pt_key:
            layer_type = "csp_layer"
        elif "node_embedding" in m.pt_key:
            layer_type = "node_embedding"
        elif "atom_latent_emb" in m.pt_key:
            layer_type = "atom_latent_emb"
        elif "coord_out" in m.pt_key:
            layer_type = "coord_out"
        elif "lattice_out" in m.pt_key:
            layer_type = "lattice_out"
        elif "type_out" in m.pt_key:
            layer_type = "type_out"
        elif "layer_norm" in m.pt_key.lower():
            layer_type = "layer_norm"
        elif "scheduler" in m.pt_key:
            layer_type = "scheduler"
        else:
            layer_type = "other"

        groups[layer_type].append(m.to_dict())

    return dict(groups)


def generate_markdown_report(
    mappings: List[LayerMapping],
    summary: Dict[str, Any],
    json_report: Dict
) -> str:
    """生成 Markdown 格式的报告"""
    lines = [
        "# DiffCSP 模型映射分析报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 汇总信息",
        "",
        f"| 指标 | 数量 |",
        f"|------|------|",
        f"| PyTorch 总层数 | {summary['total_pytorch_layers']} |",
        f"| 匹配的层数 | {summary['matched']} |",
        f"| 需要转置的层数 | {summary['transposed']} |",
        f"| 直接复制的层数 | {summary['copied']} |",
        f"| 跳过的层数 | {summary['skipped']} |",
        f"| PyTorch 独有层数 | {summary['pytorch_only']} |",
        f"| Paddle 独有层数 | {summary['paddle_only']} |",
        f"| 形状不匹配的层数 | {summary['mismatch']} |",
        "",
        "## 按层类型分组",
        ""
    ]

    groups = json_report["layer_groups"]

    for group_name, group_layers in sorted(groups.items()):
        lines.append(f"### {group_name} ({len(group_layers)} 层)")
        lines.append("")
        lines.append("| PyTorch Key | Paddle Key | Transform | Shape (PT) | Shape (PD) | Status |")
        lines.append("|-------------|------------|-----------|-------------|-------------|--------|")

        for layer in group_layers:
            pt_key = layer.get("pytorch_key", "-")
            pd_key = layer.get("paddle_key", "-")
            transform = layer.get("transform", "-")
            pt_shape = str(layer.get("pytorch_shape", []))
            pd_shape = str(layer.get("paddle_shape", []))
            status = layer.get("status", "-")

            lines.append(f"| `{pt_key}` | `{pd_key}` | {transform} | {pt_shape} | {pd_shape} | {status} |")

        lines.append("")

    # 添加转换规则说明
    lines.extend([
        "## 转换规则说明",
        "",
        "### 1. Linear 权重转换",
        "```python",
        "# PyTorch: weight.shape = [out_features, in_features]",
        "# Paddle:  weight.shape = [in_features, out_features]",
        "converted_weight = pytorch_weight.T  # 转置",
        "```",
        "",
        "### 2. Bias 转换",
        "```python",
        "# 直接复制，形状不变",
        "converted_bias = pytorch_bias",
        "```",
        "",
        "### 3. LayerNorm 转换",
        "```python",
        "# 直接复制，形状不变",
        "converted_weight = pytorch_weight",
        "converted_bias = pytorch_bias",
        "```",
        "",
        "### 4. Embedding 转换",
        "```python",
        "# 直接复制，形状不变 [num_embeddings, embedding_dim]",
        "converted_weight = pytorch_weight",
        "```",
        "",
        "## 特殊说明",
        "",
        "### smooth 模式",
        "- `smooth=True`: `node_embedding` 使用 `nn.Linear(max_atoms, hidden_dim)`",
        "- `smooth=False`: `node_embedding` 使用 `nn.Embedding(max_atoms, hidden_dim)`",
        "",
        "### Paddle 独有层",
        "- `prop_mlp.*`: Paddle 扩展的属性引导层，推理时 `property_emb=None` 不会被调用",
        "- `*_scheduler.*`: 噪声调度器参数，Paddle 不存储在权重中",
        "",
        "### PyTorch 独有层",
        "- `type_out.*`: PyTorch `pred_type=True` 时的原子类型预测层，Paddle `pred_type=False` 跳过",
        "",
        "---",
        "",
        "*报告生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*"
    ])

    return "\n".join(lines)


def main():
    logger.info("=" * 60)
    logger.info("DiffCSP 模型映射分析")
    logger.info("=" * 60)

    # ── 1. 加载 PyTorch 权重 ─────────────────────────────────────────────────────
    logger.info("\n[1/4] 加载 PyTorch 权重...")
    pt_weights = load_pytorch_weights(PT_CKPT)
    logger.info(f"  成功加载 {len(pt_weights)} 个 PyTorch 权重")

    # ── 2. 构建 Paddle 模型 ───────────────────────────────────────────────────────
    logger.info("\n[2/4] 构建 Paddle 模型...")
    pd_model = build_paddle_model(PD_CFG)
    pd_state = pd_model.state_dict()
    logger.info(f"  Paddle 模型有 {len(pd_state)} 个权重")

    # ── 3. 分析映射关系 ───────────────────────────────────────────────────────────
    logger.info("\n[3/4] 分析映射关系...")
    mappings, summary = analyze_mapping(pt_weights, pd_model)

    logger.info(f"  匹配的层数: {summary['matched']}")
    logger.info(f"  需要转置的层数: {summary['transposed']}")
    logger.info(f"  直接复制的层数: {summary['copied']}")
    logger.info(f"  跳过的层数: {summary['skipped']}")
    logger.info(f"  PyTorch 独有层数: {summary['pytorch_only']}")
    logger.info(f"  Paddle 独有层数: {summary['paddle_only']}")
    logger.info(f"  形状不匹配的层数: {summary['mismatch']}")

    if summary['mismatch'] > 0:
        logger.warning(f"  ⚠️ 发现 {summary['mismatch']} 个形状不匹配的层！")

    # ── 4. 生成报告 ───────────────────────────────────────────────────────────────
    logger.info("\n[4/4] 生成报告...")

    json_report = generate_json_report(mappings, summary, pt_weights, pd_model)
    json_path = OUTPUT_DIR / "diffcsp_mapping_report.json"
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    logger.info(f"  JSON 报告已保存: {json_path}")

    md_report = generate_markdown_report(mappings, summary, json_report)
    md_path = OUTPUT_DIR / "diffcsp_mapping_report.md"
    with open(md_path, 'w') as f:
        f.write(md_report)
    logger.info(f"  Markdown 报告已保存: {md_path}")

    logger.info("\n" + "=" * 60)
    logger.info("分析完成！")
    logger.info("=" * 60)

    # 显示关键信息
    print("\n" + "=" * 60)
    print("关键映射信息:")
    print("=" * 60)
    print("\n【重要层映射】")
    for m in mappings:
        if "node_embedding" in m.pt_key or "atom_latent_emb" in m.pt_key or \
           "coord_out" in m.pt_key or "lattice_out" in m.pt_key:
            print(f"  {m.pt_key:40s} -> {m.pd_key:40s} [{m.transform}]")
            if m.status == "mismatch":
                print(f"    ⚠️ 形状: PT={m.pt_shape} vs PD={m.pd_shape}")

    if summary['mismatch'] > 0:
        print("\n【形状不匹配的层】")
        for m in mappings:
            if m.status == "mismatch":
                print(f"  ⚠️ {m.pt_key:40s} -> {m.pd_key:40s}")
                print(f"     PT: {m.pt_shape}")
                print(f"     PD: {m.pd_shape}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
