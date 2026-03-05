# PaddleMaterials Benchmark 脚本说明

本目录包含用于验证 PaddleMaterials 框架 MatInvent 模型迁移精度的基准测试脚本。

## 目录结构

```
diff-matinvent/
├── script/
│   ├── README.md                           # 本文档
│   ├── benchmark_mattergen_paddle.py       # MatterGen 基准测试 (PaddlePaddle)
│   ├── benchmark_diffcsp_paddle.py         # DiffCSP 基准测试 (PaddlePaddle)
│   ├── benchmark_rl_paddle.py              # RL 基准测试 (PaddlePaddle)
│   ├── run_all_benchmarks_paddle.py        # 统一运行脚本 (PaddlePaddle)
│   └── generate_comparison_report.py       # 对比报告生成脚本
└── info/
    ├── mattergen/                          # MatterGen 测试结果
    │   ├── forward_pass_results.json
    │   ├── backward_pass_results.json
    │   ├── sampling_results.json
    │   ├── model_config.json
    │   └── benchmark_summary.json
    ├── diffcsp/                            # DiffCSP 测试结果
    │   └── ...
    ├── rl/                                 # RL 测试结果
    │   └── ...
    └── comparison_report.json              # 对比报告
```

## 迁移精度目标

根据 `.claude/prompts/迁移目标.md`，迁移需要满足以下目标：

### 1. 单卡前向精度对齐
- MatterGen: 前向 logits diff 1e-6 量级（生成式）
- DiffCSP: 前向 logits diff 1e-4 量级

### 2. 反向对齐
- 训练 2 轮以上，loss 一致

### 3. 监督类任务
- metric 误差控制在 1% 以内

### 4. 生成式模型
- 采样指标保持误差 5% 以内

## 使用方法

### 1. 激活环境

```bash
conda activate ppmat
```

### 2. 运行所有测试

```bash
cd /home/cao/code/github/PaddleMaterials
python diff-matinvent/script/run_all_benchmarks_paddle.py --device gpu:0
```

如果使用 CPU：

```bash
python diff-matinvent/script/run_all_benchmarks_paddle.py --device cpu
```

### 3. 单独运行测试

**MatterGen 基准测试:**
```bash
python diff-matinvent/script/benchmark_mattergen_paddle.py --device gpu:0
```

**DiffCSP 基准测试:**
```bash
python diff-matinvent/script/benchmark_diffcsp_paddle.py --device gpu:0
```

**RL 基准测试:**
```bash
python diff-matinvent/script/benchmark_rl_paddle.py
```

### 4. 生成对比报告

```bash
python diff-matinvent/script/generate_comparison_report.py
```

## 输出文件

每个测试脚本会在 `diff-matinvent/info/` 对应目录下生成以下文件：

- `benchmark_YYYYMMDD_HHMMSS.log` - 详细日志
- `forward_pass_results.json` - 前向传播测试结果
- `backward_pass_results.json` - 反向传播测试结果
- `sampling_results.json` - 采样测试结果（如果有）
- `model_config.json` - 模型配置信息
- `benchmark_summary.json` - 测试汇总

统一运行脚本会生成：
- `diff-matinvent/info/benchmark_YYYYMMDD_HHMMSS.log` - 总日志
- `diff-matinvent/info/benchmark_summary_YYYYMMDD_HHMMSS.json` - 总汇总

## 随机种子控制

所有脚本都使用固定的随机种子 (`RANDOM_SEED = 42`) 以确保可重复性：

```python
import paddle
import numpy as np

RANDOM_SEED = 42
paddle.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
```

## 注意事项

1. **不修改项目本体代码**：所有测试脚本都在 `diff-matinvent/script/` 目录下，不修改 `ppmat` 核心代码

2. **conda 环境**：确保在 `ppmat` 环境下运行

3. **设备选择**：默认使用 GPU (gpu:0)，如果需要使用 CPU，指定 `--device cpu`

4. **磁盘空间**：确保有足够的磁盘空间存储输出结果（约 50MB）

## 与原始项目的对比

原始项目 (PyTorch) 的测试结果位于 `raw-matinvent/diff_tmp/` 目录。

运行 PaddlePaddle 测试后，使用 `generate_comparison_report.py` 可以生成与原始结果的对比报告。

## 参考

- 原始项目脚本: `raw-matinvent/diff_info/`
- 原始测试结果: `raw-matinvent/diff_tmp/`
- 当前项目模型: `ppmat/models/`

## 代码来源

本目录的脚本是从 `raw-matinvent/diff_info/` 迁移而来，将 PyTorch 代码替换为 PaddlePaddle 代码，保持相同的测试逻辑和输出格式。

This code is adapted from:
- `raw-matinvent/diff_info/benchmark_mattergen.py`
- `raw-matinvent/diff_info/benchmark_diffcsp.py`
- `raw-matinvent/diff_info/benchmark_rl.py`
- `raw-matinvent/diff_info/run_all_benchmarks.py`
