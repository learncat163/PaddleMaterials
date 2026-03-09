# DiffCSP 转换验证报告

**生成时间**: 2026-03-09 10:40:22

## 测试配置

| 项目 | 值 |
|------|-----|
| Batch Size | 16 |
| Total Atoms | 210 |
| Timestep | 1000 |

## Paddle 输出统计

| 指标 | pred_l | pred_x |
|------|--------|--------|
| Mean | -0.007249 | 0.002346 |
| Std | 0.938805 | 0.011536 |
| Shape | [16, 3, 3] | [210, 3] |

## PyTorch 输出统计

| 指标 | pred_l | pred_x |
|------|--------|--------|
| Mean | -0.007249 | 0.002346 |
| Std | 0.938805 | 0.011536 |

## 对比结果

| 指标 | 值 |
|------|-----|
| pred_l max_diff | 4.7684e-07 |
| pred_l mean_diff | 8.0951e-08 |
| pred_x max_diff | 1.0431e-07 |
| pred_x mean_diff | 2.7900e-08 |
| **pred_l PASS** | ✓ |
| **pred_x PASS** | ✓ |
| **OVERALL** | ✓ PASS |

## 结论

✅ **验证通过**

Paddle 模型输出与 PyTorch 模型输出一致（max_diff < 1e-4）
权重转换成功！

---

*报告生成时间: 2026-03-09 10:40:22*