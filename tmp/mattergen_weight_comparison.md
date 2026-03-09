# MatterGen PyTorch vs Paddle 权重对比报告

**生成时间**: 2026-03-09 11:33:15

## 概览

| 指标 | 值 |
|------|-----|
| 总权重数 | 281 |
| 已对比 | 281 |
| 缺失 | 0 |
| 最大绝对差异 | 0.0000e+00 |
| 最大相对差异 | 0.0000e+00 |
| 平均绝对差异 | 0.0000e+00 |

## 关键层对比

### gemnet.atom_emb

| 指标 | 值 |
|------|-----|
| 权重数 | 1 |
| 层最大差异 | 0.0000e+00 |
| 层平均差异 | 0.0000e+00 |

**gemnet.atom_emb.embeddings.weight**:
- Shape: [101, 512]
- Max Abs Diff: 0.0000e+00
- Mean Abs Diff: 0.0000e+00
- Max Rel Diff: 0.0000e+00
- Transposed: No

### gemnet.int_blocks

| 指标 | 值 |
|------|-----|
| 权重数 | 108 |
| 层最大差异 | 0.0000e+00 |
| 层平均差异 | 0.0000e+00 |

**gemnet.int_blocks.0.dense_ca.linear.weight**:
- Shape: [512, 512]
- Max Abs Diff: 0.0000e+00
- Mean Abs Diff: 0.0000e+00
- Max Rel Diff: 0.0000e+00
- Transposed: Yes

**gemnet.int_blocks.0.trip_interaction.dense_ba.linear.weight**:
- Shape: [512, 512]
- Max Abs Diff: 0.0000e+00
- Mean Abs Diff: 0.0000e+00
- Max Rel Diff: 0.0000e+00
- Transposed: Yes

**gemnet.int_blocks.0.trip_interaction.mlp_rbf.linear.weight**:
- Shape: [512, 16]
- Max Abs Diff: 0.0000e+00
- Mean Abs Diff: 0.0000e+00
- Max Rel Diff: 0.0000e+00
- Transposed: Yes

### gemnet.out_blocks

| 指标 | 值 |
|------|-----|
| 权重数 | 135 |
| 层最大差异 | 0.0000e+00 |
| 层平均差异 | 0.0000e+00 |

**gemnet.out_blocks.0.dense_rbf.linear.weight**:
- Shape: [512, 16]
- Max Abs Diff: 0.0000e+00
- Mean Abs Diff: 0.0000e+00
- Max Rel Diff: 0.0000e+00
- Transposed: Yes

**gemnet.out_blocks.0.scale_sum.scale_factor**:
- Shape: []
- Max Abs Diff: 0.0000e+00
- Mean Abs Diff: 0.0000e+00
- Max Rel Diff: 0.0000e+00
- Transposed: No

**gemnet.out_blocks.0.layers.0.linear.weight**:
- Shape: [512, 512]
- Max Abs Diff: 0.0000e+00
- Mean Abs Diff: 0.0000e+00
- Max Rel Diff: 0.0000e+00
- Transposed: Yes

### noise_level_encoding.div_term

| 指标 | 值 |
|------|-----|
| 权重数 | 1 |
| 层最大差异 | 0.0000e+00 |
| 层平均差异 | 0.0000e+00 |

**noise_level_encoding.div_term**:
- Shape: [256]
- Max Abs Diff: 0.0000e+00
- Mean Abs Diff: 0.0000e+00
- Max Rel Diff: 0.0000e+00
- Transposed: No

## 所有层对比

| 层名 | 权重数 | 最大差异 | 平均差异 |
|------|--------|----------|----------|
| fc_atom.bias | 1 | 0.0000e+00 | 0.0000e+00 |
| fc_atom.weight | 1 | 0.0000e+00 | 0.0000e+00 |
| gemnet.angle_edge_emb | 4 | 0.0000e+00 | 0.0000e+00 |
| gemnet.atom_emb | 1 | 0.0000e+00 | 0.0000e+00 |
| gemnet.atom_latent_emb | 2 | 0.0000e+00 | 0.0000e+00 |
| gemnet.cbf_basis3 | 1 | 0.0000e+00 | 0.0000e+00 |
| gemnet.edge_emb | 1 | 0.0000e+00 | 0.0000e+00 |
| gemnet.int_blocks | 108 | 0.0000e+00 | 0.0000e+00 |
| gemnet.lattice_out_blocks | 20 | 0.0000e+00 | 0.0000e+00 |
| gemnet.mlp_cbf3 | 1 | 0.0000e+00 | 0.0000e+00 |
| gemnet.mlp_rbf3 | 1 | 0.0000e+00 | 0.0000e+00 |
| gemnet.mlp_rbf_h | 1 | 0.0000e+00 | 0.0000e+00 |
| gemnet.mlp_rbf_lattice | 1 | 0.0000e+00 | 0.0000e+00 |
| gemnet.mlp_rbf_out | 1 | 0.0000e+00 | 0.0000e+00 |
| gemnet.out_blocks | 135 | 0.0000e+00 | 0.0000e+00 |
| gemnet.radial_basis | 1 | 0.0000e+00 | 0.0000e+00 |
| noise_level_encoding.div_term | 1 | 0.0000e+00 | 0.0000e+00 |

## 结论

✅ **精度完美**

- 最大绝对差异: 0.0000e+00 < 1e-6
- 权重转换完全正确
- 输出应该与 PyTorch 完全一致

---

## 说明

此报告对比了 PyTorch 和 Paddle 的权重差异。

**关于输出对比**:

MatterGen 是一个训练模型，其输出是 loss_dict（包含 loss_coord,
loss_lattice, loss_atom_type 等损失值），而不是直接的预测值。完整的
输出对比需要：

1. 完整的训练环境（噪声调度器、采样器等）
2. 相同的随机种子
3. 相同的噪声添加过程

权重转换的准确性是输出准确性的基础保证。如果权重完全一致
（max_diff < 1e-6），则在相同输入下，输出也应该完全一致。

**权重转换规则**:

- 2D 权重 (Linear): 转置 [out, in] → [in, out]
- 1D 权重 (bias, LayerNorm): 直接复制
- Embedding 权重: 直接复制

---

*报告生成时间: 2026-03-09 11:33:15*