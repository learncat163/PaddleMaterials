# MatInvent DiffCSP 模型位置总结

## 原始模型（PyTorch）

| 项目 | 路径 |
|------|------|
| **Checkpoint 文件** | `~/.cache/huggingface/hub/models--jwchen25--MatInvent/snapshots/b192d079a52d16403fcbbb26b73078d23f9e86c2/diffcsp_mp20/last.ckpt` |
| **模型定义** | `raw-matinvent/models/diffcsp/cspnet.py` |
| **文件大小** | ~47 MB (未压缩) |
| **环境** | matinvent (PyTorch) |

## 转换后的模型（Paddle）

| 项目 | 路径 |
|------|------|
| **Checkpoint 文件** | `tmp/diffcsp_matinvent_converted.pdparams` |
| **模型定义** | `ppmat/models/diffcsp/diffcsp.py` |
| **文件大小** | ~59 MB |
| **环境** | ppmat (PaddlePaddle) |

## 转换验证结果

| 层级 | 元素数 | 最大差异 | 状态 |
|------|--------|----------|------|
| **node_embedding** (第一层) | 107,520 | 0.00e+00 | ✅ 完美 |
| **pred_l** (晶格输出) | 144 | 5.96e-07 | ✅ 优秀 |
| **pred_x** (坐标输出) | 630 | 1.34e-07 | ✅ 优秀 |

## 关键脚本

| 脚本 | 功能 | 位置 |
|------|------|------|
| `diff_mapping.py` | 分析 PyTorch 和 Paddle 层映射关系 | `diff-matinvent/script/` |
| `convert_to_paddle.py` | 权重转换脚本 | `diff-matinvent/script/` |
| `validate_to_paddle.py` | 前向传播验证 | `diff-matinvent/script/` |
| `validate_layerwise.py` | 逐层中间输出验证 | `diff-matinvent/script/` |
| `compare_first_layer_elementwise.py` | 第一层元素级对比 | `diff-matinvent/script/` |
| `compare_final_outputs.py` | 最终输出元素级对比 | `diff-matinvent/script/` |

## 使用方法

```bash
# 1. 转换权重
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/convert_to_paddle.py

# 2. 验证转换结果
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/validate_to_paddle.py

# 3. 逐层验证
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/validate_layerwise.py

# 4. 第一层元素级对比
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/compare_first_layer_elementwise.py

# 5. 最终输出元素级对比
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/compare_final_outputs.py
```

## 配置参数

**CSPNet 模型配置（与 MatInvent 一致）：**
- `hidden_dim`: 512
- `latent_dim`: 256
- `num_layers`: 6
- `act_fn`: "silu"
- `dis_emb`: "sin"
- `num_freqs`: 128
- `edge_style`: "fc"
- `ln`: True
- `ip`: True
- `smooth`: True  ← 关键：使用 Linear 而非 Embedding
- `pred_type`: False
- `prop_dim`: 512
- `pred_scalar`: False
- `num_classes`: 100

---
*生成时间: 2026-03-09*
