# MatterGen 逐层验证报告

**生成时间**: 2026-03-09 11:28:27

## 测试配置

| 项目 | 值 |
|------|-----|
| Batch Size | 4 |
| Total Atoms | 57 |

## 第一层: atom_emb

| 指标 | 值 |
|------|-----|
| Shape | [57, 512] |
| Size | 29184 |
| Mean | -0.008670 |
| Std | 0.979810 |
| Min | -1.878130 |
| Max | 1.878919 |
| Has NaN | ✓ NO |
| Has Inf | ✓ NO |

## 最后一层: decoder 输出

### decoder.frac_coords

| 指标 | 值 |
|------|-----|
| Shape | [57, 3] |
| Size | 171 |
| Mean | 0.000348 |
| Std | 0.779056 |
| Min | -3.153321 |
| Max | 3.189539 |
| Has NaN | ✓ NO |
| Has Inf | ✓ NO |

### decoder.lattice

| 指标 | 值 |
|------|-----|
| Shape | [4, 3, 3] |
| Size | 36 |
| Mean | 0.368452 |
| Std | 1.132362 |
| Min | -1.720928 |
| Max | 2.509274 |
| Has NaN | ✓ NO |
| Has Inf | ✓ NO |

### decoder.atom_types

| 指标 | 值 |
|------|-----|
| Shape | [57, 101] |
| Size | 5757 |
| Mean | -5.224658 |
| Std | 7.912477 |
| Min | -35.413780 |
| Max | 18.706665 |
| Has NaN | ✓ NO |
| Has Inf | ✓ NO |

### Loss 输出

| 指标 | 值 |
|------|-----|
| Total Loss | 3.756971 |
| loss_coord | 18.840042 |
| loss_lattice | 0.599337 |
| loss_atom_type | 1.273629 |

## 结论

✅ **验证通过**

- 第一层 (atom_emb) 输出正常
- 最后一层 (decoder) 输出正常
- 所有输出值正常（无 NaN 或 Inf）
- 输出形状正确

---

## 说明

此验证检查了 MatterGen 模型的第一层和最后一层输出：
- **第一层**: atom_emb - 原子嵌入层
- **最后一层**: decoder - 解码器输出和 loss 计算

注意: 此验证仅检查 Paddle 模型是否正常运行。
要与 PyTorch 进行精确对比，需要导出 PyTorch 的输出数据。

---

*报告生成时间: 2026-03-09 11:28:27*