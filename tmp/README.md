# MatInvent DiffCSP → Paddle 权重转换

## 📋 目录结构

```
tmp/
├── diffcsp_matinvent_converted.pdparams          # 🎯 转换后的模型（59MB）
├── MODEL_PATHS_SUMMARY.md                        # 📍 模型位置说明
├── CLEANUP_SUMMARY.md                            # 🧹 清理总结
├── diffcsp_conversion_log.json                   # 📝 转换日志
│
├── diffcsp_mapping_report.json                   # 📊 映射分析
├── diffcsp_mapping_report.md
│
├── diffcsp_validation_report.json                # ✅ 验证报告
├── diffcsp_validation_report.md
├── diffcsp_layerwise_validation.json
├── diffcsp_layerwise_validation.md
│
├── first_layer_output.json                       # 🔬 测试数据
├── first_layer_comparison.json
├── first_layer_comparison.md
├── first_layer_elementwise_comparison.json
├── first_layer_elementwise_comparison.md
├── final_outputs_elementwise_comparison.json
└── final_outputs_elementwise_comparison.md
```

## 🚀 快速开始

### 1. 转换权重（如果还没有转换）
```bash
cd /home/cao/code/github/PaddleMaterials
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/convert_to_paddle.py
```

### 2. 验证转换结果
```bash
# 基础验证
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/validate_to_paddle.py

# 逐层验证
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/validate_layerwise.py
```

### 3. 使用转换后的模型
```python
from ppmat.models.diffcsp.diffcsp import CSPNet
import paddle

# 构建模型（注意 smooth=True）
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
    smooth=True,  # 关键：匹配 MatInvent 配置
    pred_type=False,
    prop_dim=512,
    pred_scalar=False,
    num_classes=100,
)

# 加载权重
state_dict = paddle.load("tmp/diffcsp_matinvent_converted.pdparams")
decoder.set_state_dict(state_dict)
decoder.eval()

# 前向传播
pred_l, pred_x = decoder(time_emb, atom_types, frac_coords, lattices, num_atoms, batch_idx)
```

## ✅ 验证结果

| 层级 | 元素数 | 最大差异 | 状态 |
|------|--------|----------|------|
| **node_embedding** | 107,520 | 0.00e+00 | ✅ 完美 |
| **pred_l** (晶格) | 144 | 5.96e-07 | ✅ 优秀 |
| **pred_x** (坐标) | 630 | 1.34e-07 | ✅ 优秀 |

**结论**: 权重转换完全成功，所有输出差异 < 1e-6

## 📁 模型位置

| 项目 | 路径 |
|------|------|
| **原始模型 (PyTorch)** | `~/.cache/huggingface/hub/models--jwchen25--MatInvent/.../diffcsp_mp20/last.ckpt` |
| **转换后模型 (Paddle)** | `tmp/diffcsp_matinvent_converted.pdparams` |

详细说明见 [MODEL_PATHS_SUMMARY.md](MODEL_PATHS_SUMMARY.md)

## 🔧 关键脚本

所有脚本位于 `diff-matinvent/script/` 目录：

| 脚本 | 功能 |
|------|------|
| `diff_mapping.py` | 分析 PyTorch 和 Paddle 层映射关系 |
| `convert_to_paddle.py` | 权重转换脚本 |
| `validate_to_paddle.py` | 前向传播验证 |
| `validate_layerwise.py` | 逐层中间输出验证 |
| `compare_first_layer_elementwise.py` | 第一层元素级对比 |
| `compare_final_outputs.py` | 最终输出元素级对比 |

## ⚠️ 重要说明

1. **smooth=True**: MatInvent 使用 `smooth=True`，`node_embedding` 使用 `nn.Linear` 而非 `nn.Embedding`
2. **权重转置**: PyTorch Linear 权重 [out, in] → Paddle [in, out]
3. **prop_mlp 层**: Paddle 独有层（24个权重），转换时保留原始值

## 📚 相关文档

- [MODEL_PATHS_SUMMARY.md](MODEL_PATHS_SUMMARY.md) - 模型位置详细说明
- [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) - 清理总结和文件清单
- [diffcsp_mapping_report.md](diffcsp_mapping_report.md) - 层映射分析报告
- [diffcsp_validation_report.md](diffcsp_validation_report.md) - 前向传播验证报告
- [diffcsp_layerwise_validation.md](diffcsp_layerwise_validation.md) - 逐层验证报告
- [first_layer_elementwise_comparison.md](first_layer_elementwise_comparison.md) - 第一层对比报告
- [final_outputs_elementwise_comparison.md](final_outputs_elementwise_comparison.md) - 最终输出对比报告

---
*生成时间: 2026-03-09*
*转换状态: ✅ 完成*
*验证状态: ✅ 通过*
