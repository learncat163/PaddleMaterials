# MatInvent 迁移 Checklist 结构说明

## 目录结构

```
diff-matinvent/weight/checklist/
├── scripts/                      # 验证脚本目录
│   ├── README.md                 # 脚本使用说明
│   │
│   ├── 主要验证脚本
│   ├── verify_forward_logits.py          # 前向 logits 精度验证
│   ├── verify_training_alignment.py      # 训练对齐验证
│   ├── verify_sampling_metrics.py        # 采样指标验证
│   ├── generate_final_report.py          # 最终报告生成
│   └── run_all_verifications.sh          # 主启动脚本
│   │
│   ├── DiffCSP 原始验证脚本（已复制）
│   ├── diffcsp_first_layer_pt_runner.py
│   ├── diffcsp_first_layer_infer_and_diff.py
│   ├── diffcsp_final_layer_pt_runner.py
│   └── diffcsp_final_layer_infer_and_diff.py
│   │
│   └── MatterGen 原始验证脚本（已复制）
│       ├── mattergen_first_layer_pt_runner.py
│       ├── mattergen_first_layer_infer_and_diff.py
│       ├── mattergen_final_layer_pt_runner.py
│       └── mattergen_final_layer_infer_and_diff.py
│
└── CHECKLIST_STRUCTURE.md       # 本文档
```

## 脚本功能说明

### 1. 主要验证脚本

#### verify_forward_logits.py
**功能**: 验证前向 logits 精度
- **要求**: 前向 logits diff < 1e-4（生成式 1e-6）
- **验证对象**: 中间层输出（first_layer / atom_emb）
- **输入**: 固定随机种子的测试输入
- **输出**: 前向 logits 对比报告

**工作流程**:
1. 生成固定输入（seed=42）
2. 运行 PyTorch 推理（subprocess）
3. 运行 Paddle 推理
4. 对比中间层输出差异
5. 生成验证报告

**使用方法**:
```bash
python verify_forward_logits.py --model diffcsp    # 验证 DiffCSP
python verify_forward_logits.py --model mattergen  # 验证 MatterGen
python verify_forward_logits.py --model all        # 验证所有模型
```

#### verify_training_alignment.py
**功能**: 验证训练对齐
- **要求**: 训练 2 轮以上，loss 一致
- **验证对象**: 训练 loss 差异
- **输入**: 固定训练数据
- **输出**: 训练对齐报告

**工作流程**:
1. 生成固定训练数据
2. 运行 PyTorch 训练（3 轮）
3. 运行 Paddle 训练（3 轮）
4. 对比训练 loss 差异
5. 生成验证报告

**使用方法**:
```bash
python verify_training_alignment.py --model diffcsp    # 验证 DiffCSP
python verify_training_alignment.py --model mattergen  # 验证 MatterGen
python verify_training_alignment.py --model all        # 验证所有模型
```

#### verify_sampling_metrics.py
**功能**: 验证采样指标
- **要求**: 采样指标保持误差 5% 以内
- **验证对象**: 采样输出质量
- **输入**: 随机种子
- **输出**: 采样指标报告

**工作流程**:
1. 运行 PyTorch 采样
2. 运行 Paddle 采样
3. 计算采样质量指标
4. 对比采样输出
5. 生成验证报告

**使用方法**:
```bash
python verify_sampling_metrics.py --model diffcsp    # 验证 DiffCSP
python verify_sampling_metrics.py --model mattergen  # 验证 MatterGen
python verify_sampling_metrics.py --model all        # 验证所有模型
```

#### generate_final_report.py
**功能**: 生成最终合规性报告
- **输入**: 所有验证结果
- **输出**: 合规性报告（Markdown + JSON）

**工作流程**:
1. 加载所有验证结果
2. 评估合规性
3. 生成总体报告
4. 生成各模型报告

**使用方法**:
```bash
python generate_final_report.py --model diffcsp    # 生成 DiffCSP 报告
python generate_final_report.py --model mattergen  # 生成 MatterGen 报告
python generate_final_report.py --model all        # 生成所有报告
```

#### run_all_verifications.sh
**功能**: 运行所有验证的主脚本
- **功能**: 自动化运行所有验证
- **选项**:
  - `--fast`: 仅运行主验证
  - `--skip-pytorch`: 跳过 PyTorch 生成
  - `--check-only`: 仅运行对比
  - `--model`: 指定模型

**使用方法**:
```bash
./run_all_verifications.sh --all          # 运行所有验证
./run_all_verifications.sh --fast         # 快速模式
./run_all_verifications.sh --skip-pytorch # 跳过 PyTorch
./run_all_verifications.sh --model diffcsp # 仅验证 DiffCSP
```

### 2. 原始验证脚本（已复制）

#### DiffCSP 脚本

**diffcsp_first_layer_pt_runner.py**:
- 在 matinvent 环境下运行 PyTorch DiffCSP 第一层推理
- 捕获 node_embedding 输出
- 通过 hook 机制收集中间层结果

**diffcsp_first_layer_infer_and_diff.py**:
- 主脚本，协调 PyTorch 和 Paddle 的第一层推理对比
- 生成固定输入
- 运行对比分析
- 生成验证报告

**diffcsp_final_layer_pt_runner.py**:
- 在 matinvent 环境下运行 PyTorch DiffCSP 最终层推理
- 输出 pred_l 和 pred_x

**diffcsp_final_layer_infer_and_diff.py**:
- 主脚本，协调 PyTorch 和 Paddle 的最终层推理对比
- 对比最终输出质量

#### MatterGen 脚本

**mattergen_first_layer_pt_runner.py**:
- 在 matinvent 环境下运行 PyTorch MatterGen 第一层推理
- 捕获 atom_emb 输出
- 使用 MatterGen 包直接加载模型

**mattergen_first_layer_infer_and_diff.py**:
- 主脚本，协调 PyTorch 和 Paddle 的第一层推理对比
- 对比 atom_emb 输出

**mattergen_final_layer_pt_runner.py**:
- 在 matinvent 环境下运行 PyTorch MatterGen 最终层推理
- 输出 pred_lattice, pred_frac_coords, pred_atom_types

**mattergen_final_layer_infer_and_diff.py**:
- 主脚本，协调 PyTorch 和 Paddle 的最终层推理对比
- 对比最终输出质量

## 输出结构

```
diff-matinvent/tmp/
├── forward_logits/                 # 前向 logits 验证结果
│   ├── diffcsp_input.json
│   ├── diffcsp_pytorch_output.json
│   ├── diffcsp_forward_logits_report.md
│   ├── diffcsp_forward_logits_report.json
│   ├── mattergen_input.json
│   ├── mattergen_pytorch_output.json
│   ├── mattergen_forward_logits_report.md
│   └── mattergen_forward_logits_report.json
│
├── training_alignment/             # 训练对齐验证结果
│   ├── diffcsp_training_data.json
│   ├── diffcsp_pytorch_training.py
│   ├── diffcsp_pytorch_training.json
│   ├── diffcsp_training_alignment_report.md
│   ├── diffcsp_training_alignment_report.json
│   ├── mattergen_training_data.json
│   ├── mattergen_pytorch_training.py
│   ├── mattergen_pytorch_training.json
│   ├── mattergen_training_alignment_report.md
│   └── mattergen_training_alignment_report.json
│
├── sampling_metrics/               # 采样指标验证结果
│   ├── diffcsp_pytorch_sampling.py
│   ├── diffcsp_pytorch_sampling.json
│   ├── diffcsp_sampling_metrics_report.md
│   ├── diffcsp_sampling_metrics_report.json
│   ├── mattergen_pytorch_sampling.py
│   ├── mattergen_pytorch_sampling.json
│   ├── mattergen_sampling_metrics_report.md
│   └── mattergen_sampling_metrics_report.json
│
└── compliance_report/              # 最终合规性报告
    ├── diffcsp_compliance_report.md
    ├── diffcsp_compliance_report.json
    ├── mattergen_compliance_report.md
    ├── mattergen_compliance_report.json
    ├── summary_compliance_report.md
    └── summary_compliance_report.json
```

## 验证要求对应

| 要求 | 验证脚本 | 验证内容 | 阈值 |
|------|----------|----------|------|
| 1. 单卡前向精度对齐 | verify_forward_logits.py | 前向 logits diff | DiffCSP: 1e-4<br>MatterGen: 1e-6 |
| 2. 反向对齐 | verify_training_alignment.py | 训练 loss diff | loss_diff < 1e-3<br>num_epochs >= 2 |
| 3. 生成式模型采样指标 | verify_sampling_metrics.py | 采样质量指标 | coord_diff_ratio < 5%<br>lattice_diff_ratio < 5% |

## 最终报告位置和生成方法

### 报告文件位置

所有最终报告保存在以下目录：

```
diff-matinvent/tmp/compliance_report/
├── diffcsp_compliance_report.md              # DiffCSP 合规性报告（Markdown）
├── diffcsp_compliance_report.json            # DiffCSP 合规性报告（JSON）
├── mattergen_compliance_report.md            # MatterGen 合规性报告（Markdown）
├── mattergen_compliance_report.json          # MatterGen 合规性报告（JSON）
├── summary_compliance_report.md              # 总体汇总报告（Markdown）⭐
└── summary_compliance_report.json            # 总体汇总报告（JSON）
```

### 生成报告的完整命令说明

#### 方法 1: 运行完整验证（推荐）

```bash
# 进入脚本目录
cd /home/cao/code/github/PaddleMaterials/diff-matinvent/weight/checklist/scripts

# 运行所有验证（包括 PyTorch 和 Paddle）
./run_all_verifications.sh --all
```

**说明**:
- 执行时间: 约 10-30 分钟
- 包含所有验证（前向 logits、训练对齐、采样指标）
- 自动生成所有验证数据和最终报告

#### 方法 2: 快速模式（跳过 PyTorch）

```bash
# 快速模式（仅运行 Paddle 验证）
./run_all_verifications.sh --fast
```

**说明**:
- 执行时间: 约 5-10 分钟
- 跳过 PyTorch 推理/训练/采样
- 使用已生成的 PyTorch 参考数据

#### 方法 3: 仅生成报告（基于已有数据）

```bash
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

#### 方法 4: 分步验证后生成报告

```bash
# 步骤 1: 运行前向 logits 验证
python verify_forward_logits.py --model all

# 步骤 2: 运行训练对齐验证
python verify_training_alignment.py --model all

# 步骤 3: 运行采样指标验证
python verify_sampling_metrics.py --model all

# 步骤 4: 生成最终报告
python generate_final_report.py --model all
```

### 查看生成的报告

```bash
# 查看总体汇总报告（推荐）⭐
cat /home/cao/code/github/PaddleMaterials/diff-matinvent/tmp/compliance_report/summary_compliance_report.md

# 查看 DiffCSP 合规性报告
cat /home/cao/code/github/PaddleMaterials/diff-matinvent/tmp/compliance_report/diffcsp_compliance_report.md

# 查看 MatterGen 合规性报告
cat /home/cao/code/github/PaddleMaterials/diff-matinvent/tmp/compliance_report/mattergen_compliance_report.md
```

### 报告内容说明

#### 总体汇总报告 (summary_compliance_report.md) ⭐
- 所有模型的合规性状态对比
- 验证要求说明
- 总体通过率统计
- MatInvent 迁移是否成功的最终结论

#### 单模型合规性报告
- 验证要求的具体说明
- 每个要求的通过/失败状态
- 详细的数值和阈值对比
- 该模型是否满足所有要求

## 使用场景

### 场景 1: 完整验证
```bash
# 运行所有验证，生成完整报告
./run_all_verifications.sh --all
```

### 场景 2: 快速验证
```bash
# 跳过 PyTorch 生成，仅验证已有数据
./run_all_verifications.sh --fast
```

### 场景 3: 单独验证
```bash
# 仅验证前向 logits
python verify_forward_logits.py --model diffcsp

# 仅验证训练对齐
python verify_training_alignment.py --model mattergen

# 仅验证采样指标
python verify_sampling_metrics.py --model all
```

### 场景 4: 生成报告
```bash
# 基于已有结果生成报告
python generate_final_report.py --model all
```

## 扩展指南

### 添加新的验证项

1. 创建新的验证脚本 `verify_new_item.py`
2. 在 `run_all_verifications.sh` 中添加调用
3. 在 `generate_final_report.py` 中添加结果处理
4. 更新本文档

### 添加新的模型

1. 在各验证脚本中添加模型处理逻辑
2. 更新阈值配置
3. 测试验证流程
4. 更新文档

## 注意事项

1. **环境隔离**: PyTorch 推理通过 subprocess 在独立环境中运行，避免依赖冲突
2. **固定种子**: 所有验证使用固定随机种子，确保结果可重复
3. **错误处理**: 脚本包含完善的错误处理，PyTorch 不可用时可以部分运行
4. **报告格式**: 所有报告同时生成 Markdown（可读）和 JSON（可解析）格式

## 维护建议

1. **定期更新**: 随着模型更新，定期更新验证脚本
2. **版本控制**: 保持脚本版本与模型版本一致
3. **文档同步**: 更新脚本时同步更新文档
4. **测试验证**: 修改脚本后进行完整测试

## 联系方式

如有问题或建议，请联系项目维护者。
