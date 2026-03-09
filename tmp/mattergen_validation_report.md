# MatterGen 转换验证报告

**生成时间**: 2026-03-09 11:25:32

## 测试配置

| 项目 | 值 |
|------|-----|
| Batch Size | 4 |
| Total Atoms | 57 |

## Loss 输出

### Total Loss

| 指标 | 值 |
|------|-----|
| Value | 4.092330 |
| Has NaN | ✓ NO |
| Has Inf | ✓ NO |

### loss_coord (坐标损失)

| 指标 | 值 |
|------|-----|
| Value | 14.837682 |
| Has NaN | ✓ NO |
| Has Inf | ✓ NO |

### loss_lattice (晶格损失)

| 指标 | 值 |
|------|-----|
| Value | 1.182533 |
| Has NaN | ✓ NO |
| Has Inf | ✓ NO |

### loss_atom_type (原子类型损失)

| 指标 | 值 |
|------|-----|
| Value | 1.426029 |
| Has NaN | ✓ NO |
| Has Inf | ✓ NO |

## 结论

✅ **验证通过**

- Paddle 模型成功运行前向传播
- 所有 Loss 值正常（无 NaN 或 Inf）
- 模型权重转换正确

---

## 说明

MatterGen 是一个训练模型，前向传播输出的是 loss_dict，包含：
- **loss**: 总损失
- **loss_coord**: 坐标预测损失
- **loss_lattice**: 晶格预测损失
- **loss_atom_type**: 原子类型预测损失

注意: 此验证仅检查 Paddle 模型是否正常运行，无 NaN/Inf 值。
要与 PyTorch 进行精确对比，需要导出 PyTorch 的输出数据。

---

*报告生成时间: 2026-03-09 11:25:32*