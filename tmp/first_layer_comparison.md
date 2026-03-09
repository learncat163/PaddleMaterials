# DiffCSP 第一层输出对比报告

**生成时间**: 2026-03-09 11:00:00

## 测试配置

| 项目 | PyTorch | Paddle |
|------|---------|--------|
| Batch Size | 16 | 16 |
| Total Atoms | 210 | 210 |
| Timestep | 1000 | 1000 |

## 第一层 (node_embedding) 输出统计

### PyTorch 输出

| 指标 | 值 |
|------|-----|
| Shape | [210, 512] |
| Mean | 0.001750 |
| Std | 2.858284 |
| Min | -13.167941 |
| Max | 16.260395 |
| Has NaN | ✓ NO |
| Has Inf | ✓ NO |

### Paddle 输出

| 指标 | 值 |
|------|-----|
| Shape | [210, 512] |
| Mean | 0.001750 |
| Std | 2.858284 |

## 对比结果

| 指标 | 值 |
|------|-----|
| Shape Match | ✓ YES |
| Mean Diff | 1.272880e-09 |
| Mean Diff % | 0.0001% |
| Std Diff | 4.770957e-08 |
| Std Diff % | 0.0000% |

## 结论

✅ **第一层输出一致**

- 形状匹配 ✓
- 均值差异 < 0.01% ✓
- 标准差差异 < 0.01% ✓

**PyTorch 和 Paddle 的第一层（node_embedding）输出高度一致！**

---

## 说明

- **第一层 (node_embedding)**: 将原子类型（one-hot 向量）映射到隐藏维度
- **smooth=True 模式**: 使用 `nn.Linear(max_atoms, hidden_dim)` 实现
- **PyTorch**: 直接从 MatInvent checkpoint 运行
- **Paddle**: 使用转换后的权重运行

---

*报告生成时间: 2026-03-09 11:00:00*