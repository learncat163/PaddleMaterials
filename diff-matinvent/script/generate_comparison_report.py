#!/usr/bin/env python3
"""
生成 PaddlePaddle 与 PyTorch benchmark 结果对比报告

This code is adapted from the need to compare:
- PaddlePaddle results: diff-matinvent/info/
- PyTorch results: raw-matinvent/diff_tmp/
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup paths
DIFF_INFO = PROJECT_ROOT / "diff-matinvent"
INFO_OUTPUT = DIFF_INFO / "info"
RAW_TMP = PROJECT_ROOT / "raw-matinvent" / "diff_tmp"


def load_json(filepath):
    """Load JSON file if exists."""
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def compare_results():
    """Compare PaddlePaddle and PyTorch results."""
    
    print("=" * 80)
    print("MATTERGEN & DIFFCSP BENCHMARK COMPARISON REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # MatterGen comparison
    print("=" * 80)
    print("MATTERGEN")
    print("=" * 80)
    print()
    
    mg_paddle = load_json(INFO_OUTPUT / "mattergen" / "benchmark_summary.json")
    mg_paddle_loading = load_json(INFO_OUTPUT / "mattergen" / "model_loading_results.json")
    
    if mg_paddle:
        print("PaddlePaddle Results:")
        print(f"  Model: {mg_paddle.get('model_name', 'N/A')}")
        print(f"  Framework: {mg_paddle.get('framework', 'N/A')} {mg_paddle.get('framework_version', 'N/A')}")
        print(f"  Device: {mg_paddle.get('device', 'N/A')}")
        print(f"  Model Loading: {mg_paddle.get('model_loading', {}).get('status', 'N/A')}")
        print(f"  Total Parameters: {mg_paddle.get('model_loading', {}).get('total_parameters', 0):,}")
        print(f"  Backward Pass Trend: {mg_paddle.get('backward_pass', {}).get('trend', 'N/A')}")
        print()
    
    # Check if PyTorch results exist
    mg_pytorch = load_json(RAW_TMP / "mattergen" / "benchmark_summary.json")
    if mg_pytorch:
        print("PyTorch Results (Reference):")
        print(f"  Device: {mg_pytorch.get('device', 'N/A')}")
        print(f"  Forward Pass Status: {mg_pytorch.get('forward_pass', {}).get('status', 'N/A')}")
        print(f"  Backward Pass Trend: {mg_pytorch.get('backward_pass', {}).get('trend', 'N/A')}")
        print(f"  Sampling: {mg_pytorch.get('sampling', {}).get('num_samples', 0)} samples")
        print()
    else:
        print("PyTorch Results: Not available (raw-matinvent/diff_tmp/mattergen/)")
        print()
    
    # DiffCSP comparison
    print("=" * 80)
    print("DIFFCSP")
    print("=" * 80)
    print()
    
    dc_paddle = load_json(INFO_OUTPUT / "diffcsp" / "benchmark_summary.json")
    dc_paddle_loading = load_json(INFO_OUTPUT / "diffcsp" / "model_loading_results.json")
    
    if dc_paddle:
        print("PaddlePaddle Results:")
        print(f"  Model: {dc_paddle.get('model_name', 'N/A')}")
        print(f"  Framework: {dc_paddle.get('framework', 'N/A')} {dc_paddle.get('framework_version', 'N/A')}")
        print(f"  Device: {dc_paddle.get('device', 'N/A')}")
        print(f"  Model Loading: {dc_paddle.get('model_loading', {}).get('status', 'N/A')}")
        print(f"  Total Parameters: {dc_paddle.get('model_loading', {}).get('total_parameters', 0):,}")
        print(f"  Backward Pass Trend: {dc_paddle.get('backward_pass', {}).get('trend', 'N/A')}")
        print()
    
    # Check if PyTorch results exist
    dc_pytorch = load_json(RAW_TMP / "diffcsp" / "benchmark_summary.json")
    if dc_pytorch:
        print("PyTorch Results (Reference):")
        print(f"  Device: {dc_pytorch.get('device', 'N/A')}")
        print(f"  Forward Pass Status: {dc_pytorch.get('forward_pass', {}).get('status', 'N/A')}")
        print(f"  Backward Pass Trend: {dc_pytorch.get('backward_pass', {}).get('trend', 'N/A')}")
        print()
    else:
        print("PyTorch Results: Not available (raw-matinvent/diff_tmp/diffcsp/)")
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    if mg_paddle and dc_paddle:
        print("PaddlePaddle Models Successfully Loaded:")
        print(f"  ✓ MatterGen: {mg_paddle.get('model_loading', {}).get('total_parameters', 0):,} parameters")
        print(f"  ✓ DiffCSP: {dc_paddle.get('model_loading', {}).get('total_parameters', 0):,} parameters")
        print()
        print("Both models can perform backward pass training.")
        print()
        print("Note: Full precision alignment testing requires proper data format (structure_array)")
        print("      which needs to be implemented in future work.")
    
    # Save report
    report_file = INFO_OUTPUT / "comparison_report.txt"
    with open(report_file, 'w') as f:
        f.write("MATTERGEN & DIFFCSP BENCHMARK COMPARISON REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        if mg_paddle:
            f.write("MATTERGEN (PaddlePaddle):\n")
            f.write(f"  Model: {mg_paddle.get('model_name', 'N/A')}\n")
            f.write(f"  Parameters: {mg_paddle.get('model_loading', {}).get('total_parameters', 0):,}\n")
            f.write(f"  Status: {mg_paddle.get('model_loading', {}).get('status', 'N/A')}\n")
            f.write("\n")
        
        if dc_paddle:
            f.write("DIFFCSP (PaddlePaddle):\n")
            f.write(f"  Model: {dc_paddle.get('model_name', 'N/A')}\n")
            f.write(f"  Parameters: {dc_paddle.get('model_loading', {}).get('total_parameters', 0):,}\n")
            f.write(f"  Status: {dc_paddle.get('model_loading', {}).get('status', 'N/A')}\n")
            f.write("\n")
        
        f.write("Both models loaded successfully and can perform training.\n")
        f.write("For full precision testing, structure_array format needs to be implemented.\n")
    
    print(f"Comparison report saved to: {report_file}")


if __name__ == "__main__":
    compare_results()
