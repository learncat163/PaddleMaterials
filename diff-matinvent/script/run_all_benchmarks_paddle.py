#!/usr/bin/env python3
"""
统一基准测试运行脚本 (PaddlePaddle版本)

功能：
1. 运行所有基准测试脚本
2. 汇总测试结果
3. 生成统一报告

使用方法：
    conda activate ppmat
    python diff-matinvent/script/run_all_benchmarks_paddle.py [--device DEVICE]

This code is adapted from:
raw-matinvent/diff_info/run_all_benchmarks.py
"""

import os
import sys
import json
import subprocess
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup paths
DIFF_INFO = PROJECT_ROOT / "diff-matinvent"
INFO_OUTPUT = DIFF_INFO / "info"
INFO_OUTPUT.mkdir(parents=True, exist_ok=True)

# Fix random seed for reproducibility
RANDOM_SEED = 42


def setup_logging(timestamp: str) -> logging.Logger:
    """Setup logging configuration."""
    log_file = INFO_OUTPUT / f"benchmark_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def run_benchmark(
    script_name: str,
    device: str,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Run a single benchmark script."""
    logger.info("=" * 60)
    logger.info(f"运行测试: {script_name}")
    logger.info("=" * 60)

    script_path = DIFF_INFO / "script" / script_name

    if not script_path.exists():
        logger.error(f"脚本不存在: {script_path}")
        return {"status": "FAIL", "error": f"Script not found: {script_path}"}

    try:
        result = subprocess.run(
            [sys.executable, str(script_path), f"--device={device}"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode == 0:
            logger.info(f"{script_name} 测试完成")
            return {"status": "PASS"}
        else:
            logger.error(f"{script_name} 测试失败")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            return {
                "status": "FAIL",
                "error": result.stderr,
                "returncode": result.returncode
            }
    except subprocess.TimeoutExpired:
        logger.error(f"{script_name} 测试超时")
        return {"status": "FAIL", "error": "Timeout"}
    except Exception as e:
        logger.error(f"{script_name} 测试异常: {e}")
        return {"status": "FAIL", "error": str(e)}


def load_summary(
    test_name: str,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Load summary from a benchmark test."""
    summary_path = INFO_OUTPUT / test_name / "benchmark_summary.json"

    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"无法加载 {test_name} 汇总文件: {e}")

    return {}


def generate_summary_report(
    results: Dict[str, Dict[str, Any]],
    timestamp: str,
    device: str,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Generate summary report from all test results."""
    logger.info("=" * 60)
    logger.info("生成汇总报告")
    logger.info("=" * 60)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "random_seed": RANDOM_SEED,
        "framework": "PaddlePaddle",
        "tests": {},
    }

    # Add test results
    for test_name, test_result in results.items():
        summary["tests"][test_name] = test_result.get("status", "UNKNOWN")

        # Load detailed summary if available
        detailed_summary = load_summary(test_name, logger)
        if detailed_summary:
            summary[test_name] = detailed_summary

    # Save summary
    summary_file = INFO_OUTPUT / f"benchmark_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"汇总报告已保存到: {summary_file}")

    return summary


def print_summary_report(summary: Dict[str, Any], logger: logging.Logger):
    """Print summary report to console."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("基准测试完成")
    logger.info("=" * 60)
    logger.info("")
    logger.info("结果汇总:")

    # Print test results
    if "tests" in summary:
        logger.info("测试状态:")
        for test_name, status in summary["tests"].items():
            logger.info(f"  - {test_name}: {status}")

    # Print detailed results
    for key, value in summary.items():
        if key not in ["timestamp", "device", "random_seed", "framework", "tests"]:
            logger.info(f"{key} 详情:")
            logger.info(json.dumps(value, indent=2))

    logger.info("")
    logger.info("输出文件位置:")
    logger.info(f"  - diff-matinvent/info/")
    logger.info("")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="运行所有基准测试 (PaddlePaddle)")
    parser.add_argument(
        "--device",
        type=str,
        default="gpu:0",
        help="使用的设备 (gpu:0, gpu:1, 或 cpu，默认为 gpu:0)"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(timestamp)

    logger.info("=" * 60)
    logger.info("基准测试统一运行脚本 (PaddlePaddle)")
    logger.info("=" * 60)
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info(f"设备: {args.device}")
    logger.info(f"时间戳: {timestamp}")
    logger.info("")

    # 检查 conda 环境
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env != "ppmat":
        logger.warning("当前 conda 环境不是 'ppmat'")
        logger.warning("请先运行: conda activate ppmat")
        logger.warning("")

    # 运行基准测试
    results = {}

    benchmarks = [
        ("MatterGen", "benchmark_mattergen_paddle.py"),
        ("DiffCSP", "benchmark_diffcsp_paddle.py"),
        ("强化学习", "benchmark_rl_paddle.py"),
    ]

    for idx, (name, script) in enumerate(benchmarks, 1):
        logger.info(f"测试 {idx}/{len(benchmarks)}: {name} 基准测试")
        result = run_benchmark(script, args.device, logger)
        results[name] = result
        logger.info("")

    # 生成汇总报告
    summary = generate_summary_report(results, timestamp, args.device, logger)

    # 打印汇总报告
    print_summary_report(summary, logger)


if __name__ == "__main__":
    main()
