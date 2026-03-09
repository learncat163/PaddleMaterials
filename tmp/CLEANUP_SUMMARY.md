# MatInvent DiffCSP 转换与验证 - 清理总结

## 1. 模型位置

### 原始模型（PyTorch）
- **路径**: `~/.cache/huggingface/hub/models--jwchen25--MatInvent/snapshots/b192d079a52d16403fcbbb26b73078d23f9e86c2/diffcsp_mp20/last.ckpt`
- **大小**: ~47 MB
- **环境**: matinvent (PyTorch)

### 转换后模型（Paddle）
- **路径**: `tmp/diffcsp_matinvent_converted.pdparams`
- **大小**: ~59 MB
- **环境**: ppmat (PaddlePaddle)

## 2. 转换验证结果

| 层级 | 元素数 | 最大差异 | 状态 |
|------|--------|----------|------|
| **node_embedding** (第一层) | 107,520 | 0.00e+00 | ✅ 完美 |
| **pred_l** (晶格输出) | 144 | 5.96e-07 | ✅ 优秀 |
| **pred_x** (坐标输出) | 630 | 1.34e-07 | ✅ 优秀 |

**结论**: 权重转换完全成功，精度远超预期目标（1e-4）

## 3. 保留的文件

### 3.1 模型文件
```
tmp/
└── diffcsp_matinvent_converted.pdparams    # 59 MB - 转换后的最终模型
```

### 3.2 转换相关
```
diff-matinvent/script/
├── diff_mapping.py                         # 映射分析脚本
├── convert_to_paddle.py                    # 权重转换脚本
└── validate_layerwise.py                   # 逐层验证脚本

tmp/
├── diffcsp_mapping_report.json             # 映射报告（JSON）
├── diffcsp_mapping_report.md               # 映射报告（Markdown）
└── diffcsp_conversion_log.json             # 转换日志
```

### 3.3 验证报告
```
tmp/
├── diffcsp_validation_report.json          # 前向传播验证（JSON）
├── diffcsp_validation_report.md            # 前向传播验证（Markdown）
├── diffcsp_layerwise_validation.json       # 逐层验证（JSON）
├── diffcsp_layerwise_validation.md         # 逐层验证（Markdown）
├── first_layer_comparison.json             # 第一层统计对比
├── first_layer_comparison.md               # 第一层统计对比
├── first_layer_elementwise_comparison.json # 第一层元素级对比
├── first_layer_elementwise_comparison.md   # 第一层元素级对比
├── final_outputs_elementwise_comparison.json # 最终输出元素级对比
└── final_outputs_elementwise_comparison.md   # 最终输出元素级对比
```

### 3.4 测试数据
```
tmp/
└── first_layer_output.json                 # 3 MB - PyTorch 第一层输出（用于对比）
```

### 3.5 总结文档
```
tmp/
└── MODEL_PATHS_SUMMARY.md                  # 模型位置总结
```

## 4. 已删除的文件

### 4.1 旧的转换文件（已被替代）
- `diffcsp_mp20_converted.pdparams`
- `diffcsp_mp20_from_pytorch.pdparams`
- `mattergen_mp20_converted.pdparams`

### 4.2 中间分析文件（已完成验证，无需保留）
- `analyze_model_diff.py`
- `compare_diffcsp_weights.py`
- `deep_analysis_29_percent.py`
- `verify_diffcsp_test.py`
- `prove_pytorch_vs_paddle_rng.py`
- `prove_rng_difference.py`
- 各种 RNG 相关分析文件

### 4.3 中间数据文件
- `diffcsp_pytorch_weights.npz`
- `mattergen_pytorch_weights.npz`
- `pytorch_weights.json`
- `paddle_model_analysis.json`
- `pytorch_model_analysis.json`
- `test_input_data.json`
- `diffcsp_paddle_weights_info.json`
- `diffcsp_pytorch_weights_info.json`

### 4.4 临时脚本（已整合到最终版本）
- `step1_load_pytorch.py`
- `step2_convert_to_paddle.py`
- `step2_convert_to_paddle_fixed.py`
- `step2_convert_to_paddle_final.py`
- `cleanup_diff_matinvert_scripts.py`
- `cleanup_info_files.py`

### 4.5 临时文档（已完成任务）
- `cleanup_plan.md`
- `model_comparison_analysis.md`
- `model_files_location.md`
- `DIFFCSP_WEIGHT_COMPARISON_FINAL_REPORT.md`
- `diffcsp_weight_difference_solution.md`
- `diffcsp_core_logic_comparison.md`
- `conversion_log.json`（旧版本）
- `conversion_summary.md`
- `fixed_noise_29_percent_analysis.md`
- `diffcsp_model_comparison_report.txt`
- 各种 RNG 相关分析文档

## 5. 快速开始

### 转换权重
```bash
cd /home/cao/code/github/PaddleMaterials
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/convert_to_paddle.py
```

### 验证转换
```bash
# 前向传播验证
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/validate_to_paddle.py

# 逐层验证
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/validate_layerwise.py

# 元素级对比
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/compare_first_layer_elementwise.py
/home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/compare_final_outputs.py
```

### 使用转换后的模型
```python
from ppmat.models.diffcsp.diffcsp import CSPNet
import paddle

# 构建模型（注意：使用 smooth=True 匹配 MatInvent）
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
    smooth=True,  # 关键：使用 Linear 而非 Embedding
    pred_type=False,
    prop_dim=512,
    pred_scalar=False,
    num_classes=100,
)

# 加载转换后的权重
state_dict = paddle.load("tmp/diffcsp_matinvent_converted.pdparams")
decoder.set_state_dict(state_dict)
decoder.eval()
```

## 6. 重要说明

1. **smooth=True**: MatInvent 使用 `smooth=True`，这会使 `node_embedding` 使用 `nn.Linear` 而非 `nn.Embedding`
2. **权重转置**: PyTorch Linear 权重存储为 [out, in]，Paddle 存储为 [in, out]，转换时需要转置
3. **prop_mlp 层**: Paddle 独有的层（24个权重，12MB），转换时保留原始初始化值
4. **验证精度**: 所有层输出差异 < 1e-6，远超预期目标

---
*生成时间: 2026-03-09*
*清理完成时间: 2026-03-09*
