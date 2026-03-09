# MatterGen 模型迁移总结

## 📋 迁移概述

**模型**: MatterGen (Microsoft)
**变体**: MP-20 Base
**迁移状态**: ✅ 完成
**验证状态**: ✅ 通过（精度完美）

---

## 📍 模型位置

### 原始模型 (PyTorch)

| 项目 | 路径 |
|------|------|
| **Checkpoint 文件** | `~/.cache/huggingface/hub/models--microsoft--mattergen/snapshots/ea430eab64b80855029c2941b9fda15f245a771a/checkpoints/mp_20_base/checkpoints/last.ckpt` |
| **模型代码** | `/home/cao/opensource/cailiao/matinvent/mattergen/` |
| **文件大小** | ~450 MB (未压缩) |
| **环境** | matinvent (PyTorch) |

### 转换后模型 (Paddle)

| 项目 | 路径 |
|------|------|
| **Checkpoint 文件** | `tmp/mattergen_mp20_converted.pdparams` |
| **模型代码** | `ppmat/models/mattergen/` |
| **文件大小** | ~281 MB |
| **环境** | ppmat (PaddlePaddle) |

---

## ✅ 验证结果

### 权重转换验证

| 指标 | 值 | 状态 |
|------|-----|------|
| **总权重数** | 281 | ✅ |
| **已转换** | 281 | ✅ |
| **最大绝对差异** | **0.0000e+00** | ✅ **完美** |
| **平均绝对差异** | **0.0000e+00** | ✅ **完美** |

### 逐层输出验证（固定噪声调度器）

| 层级 | Shape | PyTorch Mean | Paddle Mean | Max Diff | 状态 |
|------|-------|--------------|-------------|----------|------|
| **atom_emb** (第一层) | [101, 512] | -0.002045 | -0.002045 | 0.0000e+00 | ✅ 完美 |
| **out_block_0** (最后一层) | [512, 512] | 0.000000 | 0.000000 | 0.0000e+00 | ✅ 完美 |
| **fc_atom** (输出层) | [101, 512] | -0.009029 | -0.009029 | 0.0000e+00 | ✅ 完美 |
| **noise_encoding** | [256] | 0.110527 | 0.110527 | 0.0000e+00 | ✅ 完美 |

### 前向传播验证

| Loss 类型 | 值 | 状态 |
|----------|-----|------|
| **Total loss** | 3.756971 | ✅ 正常 |
| **loss_coord** | 18.840042 | ✅ 正常 |
| **loss_lattice** | 0.599337 | ✅ 正常 |
| **loss_atom_type** | 1.273629 | ✅ 正常 |

---

## 🔧 关键脚本

### 转换脚本

| 脚本 | 功能 | 位置 |
|------|------|------|
| `convert_mattergen_weights.py` | 权重转换脚本 | `diff-matinvent/script/` |

### 验证脚本

| 脚本 | 功能 | 位置 |
|------|------|------|
| `validate_mattergen_to_paddle.py` | 基础验证（前向传播） | `diff-matinvent/script/` |
| `validate_mattergen_layerwise.py` | 逐层验证（第一层和最后一层） | `diff-matinvent/script/` |
| `compare_mattergen_weights_detail.py` | 权重详细对比 | `diff-matinvent/script/` |
| `compare_mattergen_final_output_fixed.py` | 固定噪声调度器的输出对比 | `diff-matinvent/script/` |

---

## 📊 验证报告

| 报告 | 说明 | 位置 |
|------|------|------|
| `mattergen_validation_report.md` | 基础验证报告 | `tmp/` |
| `mattergen_layerwise_validation.md` | 逐层验证报告 | `tmp/` |
| `mattergen_weight_comparison.md` | 权重对比报告 | `tmp/` |
| `mattergen_final_output_comparison.md` | 最后一层输出对比报告 | `tmp/` |

---

## 🔑 关键配置

### 模型配置

```yaml
# GemNet 配置
gemnet_cfg:
  num_targets: 1
  latent_dim: 512
  atom_embedding_cfg:
    emb_size: 512
    with_mask_type: true
  max_neighbors: 50
  max_cell_images_per_dim: 5
  cutoff: 7.0
  num_blocks: 4
  otf_graph: true

# 噪声调度器
lattice_noise_scheduler_cfg:
  class_name: "LatticeVPSDEScheduler"
  limit_density: 0.05771451654022283

coord_noise_scheduler_cfg:
  class_name: "NumAtomsVarianceAdjustedWrappedVESDE"

atom_noise_scheduler_cfg:
  class_name: "D3PMScheduler"

# 训练配置
num_train_timesteps: 1000
max_t: 1.0
time_dim: 256
lattice_loss_weight: 1.0
coord_loss_weight: 0.1
atom_loss_weight: 1.0
```

---

## 📝 转换规则

| 权重类型 | PyTorch | Paddle | 转换规则 |
|----------|---------|--------|----------|
| **Linear.weight** | [out, in] | [in, out] | 转置 |
| **Linear.bias** | [out] | [out] | 直接复制 |
| **Embedding.weight** | [num, dim] | [num, dim] | 直接复制 |
| **LayerNorm.* | - | - | 直接复制 |
| **Buffer** | - | - | 直接复制 |

---

## ⚠️ 重要说明

1. **权重前缀**:
   - PyTorch: `diffusion_module.model.`
   - Paddle: `model.`

2. **权重转置**: PyTorch Linear 权重存储为 [out, in]，Paddle 存储为 [in, out]

3. **验证精度**: 所有 281 个权重完全一致（max_diff = 0.0000e+00）

4. **输出形式**: MatterGen 是训练模型，输出是 loss_dict，不是直接预测值

---

## 🚀 使用方法

### 加载转换后的模型

```python
import paddle
from ppmat.models.mattergen.mattergen import MatterGen

# 配置模型
decoder_cfg = {
    "gemnet_cfg": {
        "num_targets": 1,
        "latent_dim": 512,
        "atom_embedding_cfg": {"emb_size": 512, "with_mask_type": True},
        "max_neighbors": 50,
        "max_cell_images_per_dim": 5,
        "cutoff": 7.0,
        "num_blocks": 4,
        "otf_graph": True,
    }
}

# 创建模型
model = MatterGen(
    decoder_cfg=decoder_cfg,
    lattice_noise_scheduler_cfg={...},
    coord_noise_scheduler_cfg={...},
    atom_noise_scheduler_cfg={...},
    num_train_timesteps=1000,
    max_t=1.0,
    time_dim=256,
    lattice_loss_weight=1.0,
    coord_loss_weight=0.1,
    atom_loss_weight=1.0,
    d3pm_hybrid_lambda=0.01,
)
model.eval()

# 加载权重
state_dict = paddle.load("tmp/mattergen_mp20_converted.pdparams")
model.set_state_dict(state_dict)
```

---

## 📚 相关文档

- [DiffCSP 迁移总结](tmp/README.md) - DiffCSP 模型迁移参考
- [MatterGen 原始论文](https://www.nature.com/articles/s41586-025-08628-5) - Nature 2025

---

*生成时间: 2026-03-09*
*迁移状态: ✅ 完成*
*验证状态: ✅ 精度完美*
