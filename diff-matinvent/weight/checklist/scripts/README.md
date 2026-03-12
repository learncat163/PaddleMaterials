# MatInvent 迁移验证脚本

## 概述

本目录包含用于验证 MatInvent (DiffCSP 和 MatterGen) 从 PyTorch 到 PaddlePaddle 迁移正确性的完整验证脚本集。

## 验证要求

脚本集验证以下三个核心要求：

### 1. 单卡前向精度对齐
- **要求**: 前向 logits diff < 1e-4（生成式 1e-6）
- **验证对象**: 中间层输出（first_layer / atom_emb）
- **阈值**:
  - DiffCSP: 1e-4
  - MatterGen: 1e-6（生成式模型）

### 2. 反向对齐
- **要求**: 训练 2 轮以上，loss 一致
- **验证对象**: 训练 loss 差异
- **阈值**: loss_diff < 1e-3

### 3. 生成式模型采样指标
- **要求**: 采样指标保持误差 5% 以内
- **验证对象**: 采样输出质量
- **阈值**:
  - coord_diff_ratio < 5%
  - lattice_diff_ratio < 5%

## 脚本列表

### 主要验证脚本

| 脚本 | 功能 | 用法 |
|------|------|------|
| `verify_forward_logits.py` | 前向 logits 精度验证 | `python verify_forward_logits.py --model diffcsp\|mattergen\|all` |
| `verify_training_alignment.py` | 训练对齐验证 | `python verify_training_alignment.py --model diffcsp\|mattergen\|all` |
| `verify_sampling_metrics.py` | 采样指标验证 | `python verify_sampling_metrics.py --model diffcsp\|mattergen\|all` |
| `generate_final_report.py` | 生成最终合规性报告 | `python generate_final_report.py --model diffcsp\|mattergen\|all` |
| `run_all_verifications.sh` | 运行所有验证的主脚本 | `./run_all_verifications.sh [options]` |

### 原始验证脚本（已复制）

#### DiffCSP 脚本
- `diffcsp_first_layer_pt_runner.py` - PyTorch DiffCSP 第一层推理
- `diffcsp_first_layer_infer_and_diff.py` - DiffCSP 第一层对比
- `diffcsp_final_layer_pt_runner.py` - PyTorch DiffCSP 最终层推理
- `diffcsp_final_layer_infer_and_diff.py` - DiffCSP 最终层对比

#### MatterGen 脚本
- `mattergen_first_layer_pt_runner.py` - PyTorch MatterGen 第一层推理
- `mattergen_first_layer_infer_and_diff.py` - MatterGen 第一层对比
- `mattergen_final_layer_pt_runner.py` - PyTorch MatterGen 最终层推理
- `mattergen_final_layer_infer_and_diff.py` - MatterGen 最终层对比

## 快速开始

### 1. 运行所有验证

```bash
# 运行所有验证（包括 PyTorch 和 Paddle）
./run_all_verifications.sh --all

# 仅运行主验证（跳过 PyTorch 生成）
./run_all_verifications.sh --fast

# 仅运行对比（假设已有 PyTorch 数据）
./run_all_verifications.sh --check-only

# 仅验证特定模型
./run_all_verifications.sh --model diffcsp
```

### 2. 运行单个验证

```bash
# 前向 logits 验证
python verify_forward_logits.py --model diffcsp

# 训练对齐验证
python verify_training_alignment.py --model mattergen

# 采样指标验证
python verify_sampling_metrics.py --model all
```

### 3. 生成合规性报告

```bash
# 生成所有模型的报告
python generate_final_report.py --model all

# 生成特定模型的报告
python generate_final_report.py --model diffcsp
```

## 最终报告位置和生成方法

### 报告文件位置

所有最终报告保存在以下目录：

```
diff-matinvent/tmp/compliance_report/
├── diffcsp_compliance_report.md              # DiffCSP 合规性报告（Markdown）
├── diffcsp_compliance_report.json            # DiffCSP 合规性报告（JSON）
├── mattergen_compliance_report.md            # MatterGen 合规性报告（Markdown）
├── mattergen_compliance_report.json          # MatterGen 合规性报告（JSON）
├── summary_compliance_report.md              # 总体汇总报告（Markdown）
└── summary_compliance_report.json            # 总体汇总报告（JSON）
```

### 生成报告的命令说明

#### 方法 1: 运行完整验证（推荐）

运行完整的验证流程，自动生成所有报告：

```bash
# 进入脚本目录
cd /home/cao/code/github/PaddleMaterials/diff-matinvent/weight/checklist/scripts

# 运行所有验证（包括 PyTorch 和 Paddle）
./run_all_verifications.sh --all

# 或者使用完整路径
/home/cao/code/github/PaddleMaterials/diff-matinvent/weight/checklist/scripts/run_all_verifications.sh --all
```

**说明**:
- 执行时间: 约 10-30 分钟
- 包含所有验证（前向 logits、训练对齐、采样指标）
- 自动生成所有验证数据和最终报告

#### 方法 2: 快速模式（跳过 PyTorch）

如果已有 PyTorch 参考数据，可以跳过 PyTorch 生成：

```bash
# 进入脚本目录
cd /home/cao/code/github/PaddleMaterials/diff-matinvent/weight/checklist/scripts

# 快速模式（仅运行 Paddle 验证）
./run_all_verifications.sh --fast
```

**说明**:
- 执行时间: 约 5-10 分钟
- 跳过 PyTorch 推理/训练/采样
- 使用已生成的 PyTorch 参考数据

#### 方法 3: 仅生成报告（基于已有数据）

如果所有验证数据已存在，仅生成报告：

```bash
# 进入脚本目录
cd /home/cao/code/github/PaddleMaterials/diff-matinvent/weight/checklist/scripts

# 生成所有模型的报告
python generate_final_report.py --model all

# 仅生成 DiffCSP 报告
python generate_final_report.py --model diffcsp

# 仅生成 MatterGen 报告
python generate_final_report.py --model mattergen
```

**说明**:
- 执行时间: 约 1 分钟
- 仅生成报告，不运行验证
- 需要验证数据已存在于 `tmp/` 目录

#### 方法 4: 单独验证后生成报告

分别运行验证，然后生成报告：

```bash
# 进入脚本目录
cd /home/cao/code/github/PaddleMaterials/diff-matinvent/weight/checklist/scripts

# 步骤 1: 运行前向 logits 验证
python verify_forward_logits.py --model all

# 步骤 2: 运行训练对齐验证
python verify_training_alignment.py --model all

# 步骤 3: 运行采样指标验证
python verify_sampling_metrics.py --model all

# 步骤 4: 生成最终报告
python generate_final_report.py --model all
```

**说明**:
- 可以单独运行每个验证
- 便于调试和排查问题
- 最后统一生成报告

### 查看生成的报告

```bash
# 查看总体汇总报告（推荐）
cat /home/cao/code/github/PaddleMaterials/diff-matinvent/tmp/compliance_report/summary_compliance_report.md

# 查看 DiffCSP 合规性报告
cat /home/cao/code/github/PaddleMaterials/diff-matinvent/tmp/compliance_report/diffcsp_compliance_report.md

# 查看 MatterGen 合规性报告
cat /home/cao/code/github/PaddleMaterials/diff-matinvent/tmp/compliance_report/mattergen_compliance_report.md
```

### 报告内容说明

#### 总体汇总报告 (summary_compliance_report.md)
- 所有模型的合规性状态对比
- 验证要求说明
- 总体通过率统计
- MatInvent 迁移是否成功的最终结论

#### 单模型合规性报告 (diffcsp/mattergen_compliance_report.md)
- **验证要求**: 三个核心要求的具体说明
- **合规性评估**: 每个要求的通过/失败状态
  - 前向 logits 精度是否达标
  - 训练对齐是否满足要求
  - 采样指标是否在 5% 以内
- **详细数据**: 具体的数值和阈值对比
- **总体结论**: 该模型是否满足所有要求

### 验证状态检查

```bash
# 检查报告目录是否存在
ls -la /home/cao/code/github/PaddleMaterials/diff-matinvent/tmp/compliance_report/

# 检查验证数据是否存在
ls -la /home/cao/code/github/PaddleMaterials/diff-matinvent/tmp/forward_logits/
ls -la /home/cao/code/github/PaddleMaterials/diff-matinvent/tmp/training_alignment/
ls -la /home/cao/code/github/PaddleMaterials/diff-matinvent/tmp/sampling_metrics/
```

## 输出目录

所有验证结果和报告保存在以下目录：

```
diff-matinvent/tmp/
├── forward_logits/          # 前向 logits 验证结果
│   ├── diffcsp_forward_logits_report.md
│   ├── diffcsp_forward_logits_report.json
│   ├── mattergen_forward_logits_report.md
│   └── mattergen_forward_logits_report.json
├── training_alignment/      # 训练对齐验证结果
│   ├── diffcsp_training_alignment_report.md
│   ├── diffcsp_training_alignment_report.json
│   ├── mattergen_training_alignment_report.md
│   └── mattergen_training_alignment_report.json
├── sampling_metrics/        # 采样指标验证结果
│   ├── diffcsp_sampling_metrics_report.md
│   ├── diffcsp_sampling_metrics_report.json
│   ├── mattergen_sampling_metrics_report.md
│   └── mattergen_sampling_metrics_report.json
└── compliance_report/       # 最终合规性报告
    ├── diffcsp_compliance_report.md
    ├── diffcsp_compliance_report.json
    ├── mattergen_compliance_report.md
    ├── mattergen_compliance_report.json
    ├── summary_compliance_report.md
    └── summary_compliance_report.json
```

## 环境要求

### PaddlePaddle 环境（主环境）
- Python 3.8+
- PaddlePaddle 2.6+
- 项目依赖：`ppmat/`

### PyTorch 环境（用于生成参考数据）
- Python 3.8+
- PyTorch 1.12+
- MatInvent 原始代码：`raw-matinvent/`
- Conda 环境：`matinvent`

## 验证流程

### 1. 前向 Logits 验证流程

1. **生成固定输入**：使用固定随机种子生成确定性测试输入
2. **PyTorch 推理**：通过 subprocess 在 matinvent 环境运行 PyTorch 推理
3. **Paddle 推理**：在 ppmat 环境运行 Paddle 推理
4. **对比分析**：对比中间层输出的数值差异
5. **生成报告**：生成 Markdown 和 JSON 格式的验证报告

### 2. 训练对齐验证流程

1. **生成训练数据**：生成固定训练数据集
2. **PyTorch 训练**：运行 PyTorch 训练（3 轮）
3. **Paddle 训练**：运行 Paddle 训练（3 轮）
4. **对比 Loss**：对比每轮训练的 loss 差异
5. **生成报告**：生成验证报告

### 3. 采样指标验证流程

1. **PyTorch 采样**：运行 PyTorch 模型采样
2. **Paddle 采样**：运行 Paddle 模型采样
3. **计算指标**：计算采样质量指标
4. **对比分析**：对比采样输出质量
5. **生成报告**：生成验证报告

## 配置选项

### 验证配置

可以在脚本中修改以下配置：

```python
# 采样配置
SAMPLING_CONFIG = {
    "num_samples": 2,              # 采样数量
    "num_atoms_list": [[10, 15]],  # 原子数列表
    "num_inference_steps": 20,     # 采样步数
    "batch_size": 2,
}

# 训练配置
TRAINING_CONFIG = {
    "num_epochs": 3,                # 训练轮数
    "num_steps_per_epoch": 5,       # 每轮步数
    "batch_size": 2,                # batch size
    "learning_rate": 1e-4,          # 学习率
    "loss_diff_threshold": 1e-3,    # loss 差异阈值
}

# 阈值配置
THRESHOLDS = {
    "coord_diff_ratio": 0.05,       # 坐标差异比例 5%
    "lattice_diff_ratio": 0.05,     # 晶格差异比例 5%
}
```

## 故障排除

### PyTorch 推理失败

如果 PyTorch 推理失败，检查：
1. `matinvent` conda 环境是否正确配置
2. `raw-matinvent` 目录是否存在
3. PyTorch 模型权重是否已下载

### Paddle 推理失败

如果 Paddle 推理失败，检查：
1. PaddlePaddle 是否正确安装
2. 模型权重是否存在
3. 模型实现是否正确

### 权重文件位置

- **PyTorch 权重**:
  - DiffCSP: `~/.cache/huggingface/hub/models--jwchen25--MatInvent/...`
  - MatterGen: `~/.cache/huggingface/hub/models--microsoft--mattergen/...`

- **Paddle 权重**:
  - DiffCSP: `diff-matinvent/tmp/matinvent_diffcsp_mp20.pdparams`
  - MatterGen: `diff-matinvent/tmp/matinvent_mattergen_mp20.pdparams`

## 参考文档

- [DiffCSP 原始论文](https://arxiv.org/abs/2305.06945)
- [MatterGen 原始论文](https://arxiv.org/abs/2312.03687)
- [MatInvent 项目文档](https://github.com/TransChemAI/MatInvent)

## 更新日志

- 2025-03-12: 初始版本，创建完整验证脚本集
