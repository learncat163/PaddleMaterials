# MatInvent 验证报告生成快速参考

## 快速开始（最常用命令）

```bash
# 进入脚本目录
cd /home/cao/code/github/PaddleMaterials/diff-matinvent/weight/checklist/scripts

# 运行所有验证并生成报告（推荐）
./run_all_verifications.sh --all

# 查看总体汇总报告
cat /home/cao/code/github/PaddleMaterials/diff-matinvent/tmp/compliance_report/summary_compliance_report.md
```

## 报告位置

```
diff-matinvent/tmp/compliance_report/
├── summary_compliance_report.md              # ⭐ 总体汇总报告（首选）
├── diffcsp_compliance_report.md              # DiffCSP 合规性报告
├── mattergen_compliance_report.md            # MatterGen 合规性报告
└── *.json                                    # JSON 格式数据
```

## 命令速查表

### 运行验证

| 场景 | 命令 | 时间 |
|------|------|------|
| 完整验证 | `./run_all_verifications.sh --all` | 10-30分钟 |
| 快速验证 | `./run_all_verifications.sh --fast` | 5-10分钟 |
| 仅生成报告 | `python generate_final_report.py --model all` | 1分钟 |

### 单独验证

| 验证项 | 命令 |
|--------|------|
| 前向 logits | `python verify_forward_logits.py --model all` |
| 训练对齐 | `python verify_training_alignment.py --model all` |
| 采样指标 | `python verify_sampling_metrics.py --model all` |

### 查看报告

| 报告类型 | 命令 |
|----------|------|
| 总体汇总 | `cat .../summary_compliance_report.md` |
| DiffCSP | `cat .../diffcsp_compliance_report.md` |
| MatterGen | `cat .../mattergen_compliance_report.md` |

## 完整路径示例

```bash
# 脚本目录
SCRIPT_DIR="/home/cao/code/github/PaddleMaterials/diff-matinvent/weight/checklist/scripts"

# 报告目录
REPORT_DIR="/home/cao/code/github/PaddleMaterials/diff-matinvent/tmp/compliance_report"

# 运行验证
cd $SCRIPT_DIR
./run_all_verifications.sh --all

# 查看报告
cat $REPORT_DIR/summary_compliance_report.md
```

## 验证要求

| 要求 | 阈值 | 验证内容 |
|------|------|----------|
| 前向 logits | DiffCSP: 1e-4, MatterGen: 1e-6 | 中间层输出精度 |
| 训练对齐 | loss_diff < 1e-3, epochs >= 2 | 训练 loss 一致性 |
| 采样指标 | coord/lattice diff_ratio < 5% | 采样输出质量 |

## 故障排除

### PyTorch 不可用
```bash
# 使用快速模式，跳过 PyTorch
./run_all_verifications.sh --fast
```

### 部分验证失败
```bash
# 单独运行失败的验证
python verify_forward_logits.py --model diffcsp
```

### 重新生成报告
```bash
# 仅生成报告（不运行验证）
python generate_final_report.py --model all
```

## 更多信息

详细文档：
- 使用说明: [README.md](README.md)
- 结构说明: [CHECKLIST_STRUCTURE.md](CHECKLIST_STRUCTURE.md)
- 整合总结: [INTEGRATION_SUMMARY.md](../INTEGRATION_SUMMARY.md)
