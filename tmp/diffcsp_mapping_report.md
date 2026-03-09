# DiffCSP 模型映射分析报告

**生成时间**: 2026-03-09 10:33:17

## 汇总信息

| 指标 | 数量 |
|------|------|
| PyTorch 总层数 | 78 |
| 匹配的层数 | 67 |
| 需要转置的层数 | 28 |
| 直接复制的层数 | 39 |
| 跳过的层数 | 10 |
| PyTorch 独有层数 | 11 |
| Paddle 独有层数 | 24 |
| 形状不匹配的层数 | 0 |

## 按层类型分组

### atom_latent_emb (2 层)

| PyTorch Key | Paddle Key | Transform | Shape (PT) | Shape (PD) | Status |
|-------------|------------|-----------|-------------|-------------|--------|
| `decoder.atom_latent_emb.bias` | `decoder.atom_latent_emb.bias` | copy | [512] | [512] | matched |
| `decoder.atom_latent_emb.weight` | `decoder.atom_latent_emb.weight` | transpose | [512, 768] | [768, 512] | matched |

### coord_out (1 层)

| PyTorch Key | Paddle Key | Transform | Shape (PT) | Shape (PD) | Status |
|-------------|------------|-----------|-------------|-------------|--------|
| `decoder.coord_out.weight` | `decoder.coord_out.weight` | transpose | [3, 512] | [512, 3] | matched |

### csp_layer (60 层)

| PyTorch Key | Paddle Key | Transform | Shape (PT) | Shape (PD) | Status |
|-------------|------------|-----------|-------------|-------------|--------|
| `decoder.csp_layer_0.edge_mlp.0.bias` | `decoder.csp_layer_0.edge_mlp.0.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_0.edge_mlp.0.weight` | `decoder.csp_layer_0.edge_mlp.0.weight` | transpose | [512, 1801] | [1801, 512] | matched |
| `decoder.csp_layer_0.edge_mlp.2.bias` | `decoder.csp_layer_0.edge_mlp.2.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_0.edge_mlp.2.weight` | `decoder.csp_layer_0.edge_mlp.2.weight` | transpose | [512, 512] | [512, 512] | matched |
| `decoder.csp_layer_0.layer_norm.bias` | `decoder.csp_layer_0.layer_norm.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_0.layer_norm.weight` | `decoder.csp_layer_0.layer_norm.weight` | copy | [512] | [512] | matched |
| `decoder.csp_layer_0.node_mlp.0.bias` | `decoder.csp_layer_0.node_mlp.0.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_0.node_mlp.0.weight` | `decoder.csp_layer_0.node_mlp.0.weight` | transpose | [512, 1024] | [1024, 512] | matched |
| `decoder.csp_layer_0.node_mlp.2.bias` | `decoder.csp_layer_0.node_mlp.2.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_0.node_mlp.2.weight` | `decoder.csp_layer_0.node_mlp.2.weight` | transpose | [512, 512] | [512, 512] | matched |
| `decoder.csp_layer_1.edge_mlp.0.bias` | `decoder.csp_layer_1.edge_mlp.0.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_1.edge_mlp.0.weight` | `decoder.csp_layer_1.edge_mlp.0.weight` | transpose | [512, 1801] | [1801, 512] | matched |
| `decoder.csp_layer_1.edge_mlp.2.bias` | `decoder.csp_layer_1.edge_mlp.2.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_1.edge_mlp.2.weight` | `decoder.csp_layer_1.edge_mlp.2.weight` | transpose | [512, 512] | [512, 512] | matched |
| `decoder.csp_layer_1.layer_norm.bias` | `decoder.csp_layer_1.layer_norm.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_1.layer_norm.weight` | `decoder.csp_layer_1.layer_norm.weight` | copy | [512] | [512] | matched |
| `decoder.csp_layer_1.node_mlp.0.bias` | `decoder.csp_layer_1.node_mlp.0.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_1.node_mlp.0.weight` | `decoder.csp_layer_1.node_mlp.0.weight` | transpose | [512, 1024] | [1024, 512] | matched |
| `decoder.csp_layer_1.node_mlp.2.bias` | `decoder.csp_layer_1.node_mlp.2.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_1.node_mlp.2.weight` | `decoder.csp_layer_1.node_mlp.2.weight` | transpose | [512, 512] | [512, 512] | matched |
| `decoder.csp_layer_2.edge_mlp.0.bias` | `decoder.csp_layer_2.edge_mlp.0.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_2.edge_mlp.0.weight` | `decoder.csp_layer_2.edge_mlp.0.weight` | transpose | [512, 1801] | [1801, 512] | matched |
| `decoder.csp_layer_2.edge_mlp.2.bias` | `decoder.csp_layer_2.edge_mlp.2.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_2.edge_mlp.2.weight` | `decoder.csp_layer_2.edge_mlp.2.weight` | transpose | [512, 512] | [512, 512] | matched |
| `decoder.csp_layer_2.layer_norm.bias` | `decoder.csp_layer_2.layer_norm.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_2.layer_norm.weight` | `decoder.csp_layer_2.layer_norm.weight` | copy | [512] | [512] | matched |
| `decoder.csp_layer_2.node_mlp.0.bias` | `decoder.csp_layer_2.node_mlp.0.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_2.node_mlp.0.weight` | `decoder.csp_layer_2.node_mlp.0.weight` | transpose | [512, 1024] | [1024, 512] | matched |
| `decoder.csp_layer_2.node_mlp.2.bias` | `decoder.csp_layer_2.node_mlp.2.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_2.node_mlp.2.weight` | `decoder.csp_layer_2.node_mlp.2.weight` | transpose | [512, 512] | [512, 512] | matched |
| `decoder.csp_layer_3.edge_mlp.0.bias` | `decoder.csp_layer_3.edge_mlp.0.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_3.edge_mlp.0.weight` | `decoder.csp_layer_3.edge_mlp.0.weight` | transpose | [512, 1801] | [1801, 512] | matched |
| `decoder.csp_layer_3.edge_mlp.2.bias` | `decoder.csp_layer_3.edge_mlp.2.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_3.edge_mlp.2.weight` | `decoder.csp_layer_3.edge_mlp.2.weight` | transpose | [512, 512] | [512, 512] | matched |
| `decoder.csp_layer_3.layer_norm.bias` | `decoder.csp_layer_3.layer_norm.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_3.layer_norm.weight` | `decoder.csp_layer_3.layer_norm.weight` | copy | [512] | [512] | matched |
| `decoder.csp_layer_3.node_mlp.0.bias` | `decoder.csp_layer_3.node_mlp.0.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_3.node_mlp.0.weight` | `decoder.csp_layer_3.node_mlp.0.weight` | transpose | [512, 1024] | [1024, 512] | matched |
| `decoder.csp_layer_3.node_mlp.2.bias` | `decoder.csp_layer_3.node_mlp.2.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_3.node_mlp.2.weight` | `decoder.csp_layer_3.node_mlp.2.weight` | transpose | [512, 512] | [512, 512] | matched |
| `decoder.csp_layer_4.edge_mlp.0.bias` | `decoder.csp_layer_4.edge_mlp.0.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_4.edge_mlp.0.weight` | `decoder.csp_layer_4.edge_mlp.0.weight` | transpose | [512, 1801] | [1801, 512] | matched |
| `decoder.csp_layer_4.edge_mlp.2.bias` | `decoder.csp_layer_4.edge_mlp.2.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_4.edge_mlp.2.weight` | `decoder.csp_layer_4.edge_mlp.2.weight` | transpose | [512, 512] | [512, 512] | matched |
| `decoder.csp_layer_4.layer_norm.bias` | `decoder.csp_layer_4.layer_norm.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_4.layer_norm.weight` | `decoder.csp_layer_4.layer_norm.weight` | copy | [512] | [512] | matched |
| `decoder.csp_layer_4.node_mlp.0.bias` | `decoder.csp_layer_4.node_mlp.0.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_4.node_mlp.0.weight` | `decoder.csp_layer_4.node_mlp.0.weight` | transpose | [512, 1024] | [1024, 512] | matched |
| `decoder.csp_layer_4.node_mlp.2.bias` | `decoder.csp_layer_4.node_mlp.2.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_4.node_mlp.2.weight` | `decoder.csp_layer_4.node_mlp.2.weight` | transpose | [512, 512] | [512, 512] | matched |
| `decoder.csp_layer_5.edge_mlp.0.bias` | `decoder.csp_layer_5.edge_mlp.0.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_5.edge_mlp.0.weight` | `decoder.csp_layer_5.edge_mlp.0.weight` | transpose | [512, 1801] | [1801, 512] | matched |
| `decoder.csp_layer_5.edge_mlp.2.bias` | `decoder.csp_layer_5.edge_mlp.2.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_5.edge_mlp.2.weight` | `decoder.csp_layer_5.edge_mlp.2.weight` | transpose | [512, 512] | [512, 512] | matched |
| `decoder.csp_layer_5.layer_norm.bias` | `decoder.csp_layer_5.layer_norm.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_5.layer_norm.weight` | `decoder.csp_layer_5.layer_norm.weight` | copy | [512] | [512] | matched |
| `decoder.csp_layer_5.node_mlp.0.bias` | `decoder.csp_layer_5.node_mlp.0.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_5.node_mlp.0.weight` | `decoder.csp_layer_5.node_mlp.0.weight` | transpose | [512, 1024] | [1024, 512] | matched |
| `decoder.csp_layer_5.node_mlp.2.bias` | `decoder.csp_layer_5.node_mlp.2.bias` | copy | [512] | [512] | matched |
| `decoder.csp_layer_5.node_mlp.2.weight` | `decoder.csp_layer_5.node_mlp.2.weight` | transpose | [512, 512] | [512, 512] | matched |

### lattice_out (1 层)

| PyTorch Key | Paddle Key | Transform | Shape (PT) | Shape (PD) | Status |
|-------------|------------|-----------|-------------|-------------|--------|
| `decoder.lattice_out.weight` | `decoder.lattice_out.weight` | transpose | [9, 512] | [512, 9] | matched |

### layer_norm (2 层)

| PyTorch Key | Paddle Key | Transform | Shape (PT) | Shape (PD) | Status |
|-------------|------------|-----------|-------------|-------------|--------|
| `decoder.final_layer_norm.bias` | `decoder.final_layer_norm.bias` | copy | [512] | [512] | matched |
| `decoder.final_layer_norm.weight` | `decoder.final_layer_norm.weight` | copy | [512] | [512] | matched |

### node_embedding (1 层)

| PyTorch Key | Paddle Key | Transform | Shape (PT) | Shape (PD) | Status |
|-------------|------------|-----------|-------------|-------------|--------|
| `decoder.node_embedding.weight` | `decoder.node_embedding.weight` | transpose | [512, 100] | [100, 512] | matched |

### paddle_only (24 层)

| PyTorch Key | Paddle Key | Transform | Shape (PT) | Shape (PD) | Status |
|-------------|------------|-----------|-------------|-------------|--------|
| `` | `decoder.csp_layer_0.prop_mlp.0.bias` | skip | [] | [512] | paddle_only |
| `` | `decoder.csp_layer_0.prop_mlp.0.weight` | skip | [] | [512, 512] | paddle_only |
| `` | `decoder.csp_layer_0.prop_mlp.2.bias` | skip | [] | [512] | paddle_only |
| `` | `decoder.csp_layer_0.prop_mlp.2.weight` | skip | [] | [512, 512] | paddle_only |
| `` | `decoder.csp_layer_1.prop_mlp.0.bias` | skip | [] | [512] | paddle_only |
| `` | `decoder.csp_layer_1.prop_mlp.0.weight` | skip | [] | [512, 512] | paddle_only |
| `` | `decoder.csp_layer_1.prop_mlp.2.bias` | skip | [] | [512] | paddle_only |
| `` | `decoder.csp_layer_1.prop_mlp.2.weight` | skip | [] | [512, 512] | paddle_only |
| `` | `decoder.csp_layer_2.prop_mlp.0.bias` | skip | [] | [512] | paddle_only |
| `` | `decoder.csp_layer_2.prop_mlp.0.weight` | skip | [] | [512, 512] | paddle_only |
| `` | `decoder.csp_layer_2.prop_mlp.2.bias` | skip | [] | [512] | paddle_only |
| `` | `decoder.csp_layer_2.prop_mlp.2.weight` | skip | [] | [512, 512] | paddle_only |
| `` | `decoder.csp_layer_3.prop_mlp.0.bias` | skip | [] | [512] | paddle_only |
| `` | `decoder.csp_layer_3.prop_mlp.0.weight` | skip | [] | [512, 512] | paddle_only |
| `` | `decoder.csp_layer_3.prop_mlp.2.bias` | skip | [] | [512] | paddle_only |
| `` | `decoder.csp_layer_3.prop_mlp.2.weight` | skip | [] | [512, 512] | paddle_only |
| `` | `decoder.csp_layer_4.prop_mlp.0.bias` | skip | [] | [512] | paddle_only |
| `` | `decoder.csp_layer_4.prop_mlp.0.weight` | skip | [] | [512, 512] | paddle_only |
| `` | `decoder.csp_layer_4.prop_mlp.2.bias` | skip | [] | [512] | paddle_only |
| `` | `decoder.csp_layer_4.prop_mlp.2.weight` | skip | [] | [512, 512] | paddle_only |
| `` | `decoder.csp_layer_5.prop_mlp.0.bias` | skip | [] | [512] | paddle_only |
| `` | `decoder.csp_layer_5.prop_mlp.0.weight` | skip | [] | [512, 512] | paddle_only |
| `` | `decoder.csp_layer_5.prop_mlp.2.bias` | skip | [] | [512] | paddle_only |
| `` | `decoder.csp_layer_5.prop_mlp.2.weight` | skip | [] | [512, 512] | paddle_only |

### pytorch_only (11 层)

| PyTorch Key | Paddle Key | Transform | Shape (PT) | Shape (PD) | Status |
|-------------|------------|-----------|-------------|-------------|--------|
| `beta_scheduler.alphas` | `` | skip | [1001] | [] | pytorch_only |
| `beta_scheduler.alphas_cumprod` | `` | skip | [1001] | [] | pytorch_only |
| `beta_scheduler.betas` | `` | skip | [1001] | [] | pytorch_only |
| `beta_scheduler.sigmas` | `` | skip | [1001] | [] | pytorch_only |
| `decoder.node_embedding.bias` | `` | skip | [512] | [] | pytorch_only |
| `decoder.type_out.bias` | `` | skip | [100] | [] | pytorch_only |
| `decoder.type_out.weight` | `` | skip | [100, 512] | [] | pytorch_only |
| `sigma_scheduler.sigmas` | `` | skip | [1001] | [] | pytorch_only |
| `sigma_scheduler.sigmas_norm` | `` | skip | [1001] | [] | pytorch_only |
| `type_sigma_scheduler.sigmas` | `` | skip | [1001] | [] | pytorch_only |
| `type_sigma_scheduler.sigmas_norm` | `` | skip | [1001] | [] | pytorch_only |

## 转换规则说明

### 1. Linear 权重转换
```python
# PyTorch: weight.shape = [out_features, in_features]
# Paddle:  weight.shape = [in_features, out_features]
converted_weight = pytorch_weight.T  # 转置
```

### 2. Bias 转换
```python
# 直接复制，形状不变
converted_bias = pytorch_bias
```

### 3. LayerNorm 转换
```python
# 直接复制，形状不变
converted_weight = pytorch_weight
converted_bias = pytorch_bias
```

### 4. Embedding 转换
```python
# 直接复制，形状不变 [num_embeddings, embedding_dim]
converted_weight = pytorch_weight
```

## 特殊说明

### smooth 模式
- `smooth=True`: `node_embedding` 使用 `nn.Linear(max_atoms, hidden_dim)`
- `smooth=False`: `node_embedding` 使用 `nn.Embedding(max_atoms, hidden_dim)`

### Paddle 独有层
- `prop_mlp.*`: Paddle 扩展的属性引导层，推理时 `property_emb=None` 不会被调用
- `*_scheduler.*`: 噪声调度器参数，Paddle 不存储在权重中

### PyTorch 独有层
- `type_out.*`: PyTorch `pred_type=True` 时的原子类型预测层，Paddle `pred_type=False` 跳过

---

*报告生成时间: 2026-03-09 10:33:17*