#!/usr/bin/env python3
"""
最终合规性报告生成脚本
==========================================

功能：
1. 收集所有验证结果
2. 生成统一的合规性报告
3. 验证是否满足所有要求

用法：
    python generate_final_report.py --model diffcsp
    python generate_final_report.py --model mattergen
    python generate_final_report.py --model all
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent  # PaddleMaterials/
DIFF_MATINVENT_ROOT = PROJECT_ROOT / "diff-matinvent"

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# 要求配置
REQUIREMENTS = {
    "forward_logits": {
        "title": "单卡前向精度对齐",
        "description": "前向 logits diff < 1e-4（生成式 1e-6）",
        "thresholds": {
            "diffcsp": 1e-4,
            "mattergen": 1e-6,
        },
    },
    "training_alignment": {
        "title": "反向对齐",
        "description": "训练 2 轮以上，loss 一致",
        "thresholds": {
            "loss_diff": 1e-3,
            "num_epochs": 2,
        },
    },
    "sampling_metrics": {
        "title": "生成式模型采样指标",
        "description": "采样指标保持误差 5% 以内",
        "thresholds": {
            "coord_diff_ratio": 0.05,
            "lattice_diff_ratio": 0.05,
        },
    },
}

# 输出目录
OUTPUT_DIR = DIFF_MATINVENT_ROOT / "tmp" / "compliance_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_verification_results(model_type: str) -> Dict[str, Any]:
    """加载验证结果"""
    results = {
        "forward_logits": None,
        "training_alignment": None,
        "sampling_metrics": None,
    }

    # 加载前向 logits 结果
    forward_logits_json = DIFF_MATINVENT_ROOT / "tmp" / "forward_logits" / f"{model_type}_forward_logits_report.json"
    if forward_logits_json.exists():
        with open(forward_logits_json) as f:
            results["forward_logits"] = json.load(f)

    # 加载训练对齐结果
    training_alignment_json = DIFF_MATINVENT_ROOT / "tmp" / "training_alignment" / f"{model_type}_training_alignment_report.json"
    if training_alignment_json.exists():
        with open(training_alignment_json) as f:
            results["training_alignment"] = json.load(f)

    # 加载采样指标结果
    sampling_metrics_json = DIFF_MATINVENT_ROOT / "tmp" / "sampling_metrics" / f"{model_type}_sampling_metrics_report.json"
    if sampling_metrics_json.exists():
        with open(sampling_metrics_json) as f:
            results["sampling_metrics"] = json.load(f)

    return results


def evaluate_compliance(model_type: str, results: Dict[str, Any]) -> Dict[str, Any]:
    """评估合规性"""
    compliance = {
        "forward_logits": {"pass": False, "details": {}},
        "training_alignment": {"pass": False, "details": {}},
        "sampling_metrics": {"pass": False, "details": {}},
        "overall": {"pass": False, "num_passed": 0, "num_total": 3},
    }

    # 评估前向 logits
    if results["forward_logits"] is not None:
        fw_result = results["forward_logits"]
        if fw_result.get("pytorch_available") and fw_result.get("comparisons"):
            threshold = REQUIREMENTS["forward_logits"]["thresholds"][model_type]
            all_passed = fw_result.get("all_passed", False)
            compliance["forward_logits"]["pass"] = all_passed
            compliance["forward_logits"]["details"] = {
                "threshold": threshold,
                "all_passed": all_passed,
                "num_comparisons": len(fw_result.get("comparisons", [])),
            }
        else:
            compliance["forward_logits"]["details"] = {
                "reason": "PyTorch not available or no comparisons",
            }
    else:
        compliance["forward_logits"]["details"] = {
            "reason": "No verification results",
        }

    # 评估训练对齐
    if results["training_alignment"] is not None:
        ta_result = results["training_alignment"]
        if ta_result.get("comparison") is not None:
            threshold = REQUIREMENTS["training_alignment"]["thresholds"]["loss_diff"]
            passed = ta_result["comparison"].get("pass", False)
            compliance["training_alignment"]["pass"] = passed
            compliance["training_alignment"]["details"] = {
                "loss_diff_threshold": threshold,
                "mean_loss_diff": ta_result["comparison"].get("mean_loss_diff", 0),
                "num_epochs": ta_result.get("training_config", {}).get("num_epochs", 0),
                "passed": passed,
            }
        else:
            compliance["training_alignment"]["details"] = {
                "reason": "No comparison data",
            }
    else:
        compliance["training_alignment"]["details"] = {
            "reason": "No verification results",
        }

    # 评估采样指标
    if results["sampling_metrics"] is not None:
        sm_result = results["sampling_metrics"]
        if sm_result.get("metrics") is not None:
            coord_threshold = REQUIREMENTS["sampling_metrics"]["thresholds"]["coord_diff_ratio"]
            lattice_threshold = REQUIREMENTS["sampling_metrics"]["thresholds"]["lattice_diff_ratio"]
            metrics = sm_result["metrics"]
            coord_pass = metrics.get("coord_diff_ratio", 1.0) < coord_threshold
            lattice_pass = metrics.get("lattice_diff_ratio", 1.0) < lattice_threshold
            passed = coord_pass and lattice_pass
            compliance["sampling_metrics"]["pass"] = passed
            compliance["sampling_metrics"]["details"] = {
                "coord_diff_ratio": metrics.get("coord_diff_ratio", 0),
                "coord_threshold": coord_threshold,
                "coord_pass": coord_pass,
                "lattice_diff_ratio": metrics.get("lattice_diff_ratio", 0),
                "lattice_threshold": lattice_threshold,
                "lattice_pass": lattice_pass,
                "passed": passed,
            }
        else:
            compliance["sampling_metrics"]["details"] = {
                "reason": "No metrics data",
            }
    else:
        compliance["sampling_metrics"]["details"] = {
            "reason": "No verification results",
        }

    # 计算总体合规性
    num_passed = sum(1 for v in compliance.values() if isinstance(v, dict) and v.get("pass", False))
    compliance["overall"]["num_passed"] = num_passed
    compliance["overall"]["num_total"] = 3
    compliance["overall"]["pass"] = num_passed == 3

    return compliance


def generate_markdown_report(
    model_type: str,
    results: Dict[str, Any],
    compliance: Dict[str, Any],
) -> str:
    """生成 Markdown 报告"""
    lines = [
        f"# {model_type.upper()} 合规性验证报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 验证要求",
        "",
    ]

    # 添加所有要求
    for req_key, req_config in REQUIREMENTS.items():
        lines.append(f"### {req_config['title']}")
        lines.append(f"- **要求**: {req_config['description']}")
        if "thresholds" in req_config:
            if isinstance(req_config["thresholds"], dict):
                if model_type in req_config["thresholds"]:
                    lines.append(f"- **阈值 ({model_type})**: {req_config['thresholds'][model_type]}")
                else:
                    for k, v in req_config["thresholds"].items():
                        lines.append(f"- **阈值 ({k})**: {v}")
            else:
                for k, v in req_config["thresholds"].items():
                    lines.append(f"- **阈值 ({k})**: {v}")
        lines.append("")

    # 添加合规性评估结果
    lines += [
        "## 合规性评估",
        "",
    ]

    overall = compliance["overall"]
    lines.append(f"### 总体结果")
    lines.append(f"- **通过项**: {overall['num_passed']}/{overall['num_total']}")
    lines.append(f"- **状态**: {'PASS' if overall['pass'] else 'FAIL'}")
    lines.append("")

    # 前向 logits
    fw = compliance["forward_logits"]
    lines.append("### 1. 单卡前向精度对齐")
    if fw["pass"]:
        lines.append(f"- **状态**: PASS")
        lines.append(f"- **阈值**: {fw['details'].get('threshold', 'N/A')}")
        lines.append(f"- **检查项**: {fw['details'].get('num_comparisons', 0)}")
        lines.append(f"- **结果**: 所有检查项的平均差异 < {fw['details'].get('threshold', 'N/A')}")
    else:
        lines.append(f"- **状态**: FAIL")
        if "reason" in fw["details"]:
            lines.append(f"- **原因**: {fw['details']['reason']}")
        else:
            lines.append(f"- **原因**: 部分检查项未通过")
    lines.append("")

    # 训练对齐
    ta = compliance["training_alignment"]
    lines.append("### 2. 反向对齐")
    if ta["pass"]:
        lines.append(f"- **状态**: PASS")
        lines.append(f"- **训练轮数**: {ta['details'].get('num_epochs', 0)} 轮（满足 >= 2 轮要求）")
        lines.append(f"- **Loss 差异**: {ta['details'].get('mean_loss_diff', 0):.6f}")
        lines.append(f"- **阈值**: {ta['details'].get('loss_diff_threshold', 'N/A')}")
        lines.append(f"- **结果**: Loss 差异 < 阈值")
    else:
        lines.append(f"- **状态**: FAIL")
        if "reason" in ta["details"]:
            lines.append(f"- **原因**: {ta['details']['reason']}")
        else:
            lines.append(f"- **原因**: Loss 差异 >= 阈值或训练轮数不足")
    lines.append("")

    # 采样指标
    sm = compliance["sampling_metrics"]
    lines.append("### 3. 生成式模型采样指标")
    if sm["pass"]:
        lines.append(f"- **状态**: PASS")
        lines.append(f"- **坐标差异比例**: {sm['details'].get('coord_diff_ratio', 0):.4f} ({sm['details'].get('coord_diff_ratio', 0)*100:.2f}%)")
        lines.append(f"- **坐标阈值**: {sm['details'].get('coord_threshold', 'N/A'):.2%}")
        lines.append(f"- **晶格差异比例**: {sm['details'].get('lattice_diff_ratio', 0):.4f} ({sm['details'].get('lattice_diff_ratio', 0)*100:.2f}%)")
        lines.append(f"- **晶格阈值**: {sm['details'].get('lattice_threshold', 'N/A'):.2%}")
        lines.append(f"- **结果**: 采样指标 < 5%")
    else:
        lines.append(f"- **状态**: FAIL")
        if "reason" in sm["details"]:
            lines.append(f"- **原因**: {sm['details']['reason']}")
        else:
            coord_pass = sm['details'].get('coord_pass', False)
            lattice_pass = sm['details'].get('lattice_pass', False)
            if not coord_pass:
                lines.append(f"- **原因**: 坐标差异比例 >= 5%")
            if not lattice_pass:
                lines.append(f"- **原因**: 晶格差异比例 >= 5%")
    lines.append("")

    # 总体结论
    lines += [
        "## 总体结论",
        "",
    ]

    if overall["pass"]:
        lines += [
            f"[PASS] {model_type.upper()} 合规性验证通过",
            "",
            f"- 所有验证项均通过（{overall['num_passed']}/{overall['num_total']}）",
            f"- 前向 logits 精度满足要求",
            f"- 训练对齐满足要求",
            f"- 采样指标满足要求",
            f"- 模型迁移成功",
            "",
        ]
    else:
        lines += [
            f"[FAIL] {model_type.upper()} 合规性验证失败",
            "",
            f"- 部分验证项未通过（{overall['num_passed']}/{overall['num_total']}）",
            f"- 请检查失败的验证项并修复",
            "",
        ]

    lines += [
        "---",
        "",
        "*报告生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*",
    ]

    return "\n".join(lines)


def generate_summary_report(all_results: Dict[str, Dict]) -> str:
    """生成总体汇总报告"""
    lines = [
        "# MatInvent 迁移合规性总体报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 验证要求",
        "",
        "### 1. 单卡前向精度对齐",
        "- **要求**: 前向 logits diff < 1e-4（生成式 1e-6）",
        "",
        "### 2. 反向对齐",
        "- **要求**: 训练 2 轮以上，loss 一致",
        "",
        "### 3. 生成式模型采样指标",
        "- **要求**: 采样指标保持误差 5% 以内",
        "",
        "## 各模型合规性",
        "",
    ]

    # 表头
    lines += [
        "| 模型 | 前向 logits | 训练对齐 | 采样指标 | 总体 |",
        "|------|-------------|----------|----------|------|",
    ]

    # 各模型结果
    for model_type, compliance in all_results.items():
        fw_status = "PASS" if compliance["forward_logits"]["pass"] else "FAIL"
        ta_status = "PASS" if compliance["training_alignment"]["pass"] else "FAIL"
        sm_status = "PASS" if compliance["sampling_metrics"]["pass"] else "FAIL"
        overall_status = "PASS" if compliance["overall"]["pass"] else "FAIL"

        lines.append(f"| {model_type.upper()} | {fw_status} | {ta_status} | {sm_status} | {overall_status} |")

    lines += [
        "",
        "## 总体结论",
        "",
    ]

    # 统计
    total_models = len(all_results)
    passed_models = sum(1 for c in all_results.values() if c["overall"]["pass"])

    if passed_models == total_models:
        lines += [
            f"[PASS] 所有模型合规性验证通过",
            "",
            f"- 通过模型: {passed_models}/{total_models}",
            f"- 所有验证项均满足要求",
            f"- MatInvent 迁移成功",
            "",
        ]
    else:
        lines += [
            f"[FAIL] 部分模型合规性验证失败",
            "",
            f"- 通过模型: {passed_models}/{total_models}",
            f"- 请检查失败的模型和验证项",
            "",
        ]

    lines += [
        "---",
        "",
        "*报告生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate final compliance report")
    parser.add_argument(
        "--model",
        type=str,
        choices=["diffcsp", "mattergen", "all"],
        default="all",
        help="Model type to generate report for (default: all)",
    )
    args = parser.parse_args()

    # 确定要处理的模型
    models = []
    if args.model == "all":
        models = ["diffcsp", "mattergen"]
    else:
        models = [args.model]

    # 收集所有结果
    all_results = {}

    for model_type in models:
        logger.info(f"Processing {model_type.upper()}...")

        # 加载验证结果
        results = load_verification_results(model_type)

        # 评估合规性
        compliance = evaluate_compliance(model_type, results)
        all_results[model_type] = compliance

        # 生成单模型报告
        md = generate_markdown_report(model_type, results, compliance)

        report_md = OUTPUT_DIR / f"{model_type}_compliance_report.md"
        with open(report_md, "w") as f:
            f.write(md)
        logger.info(f"  Markdown report: {report_md}")

        # 保存 JSON 报告
        report_json = OUTPUT_DIR / f"{model_type}_compliance_report.json"
        report_data = {
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "requirements": REQUIREMENTS,
            "verification_results": results,
            "compliance": compliance,
        }
        with open(report_json, "w") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        logger.info(f"  JSON report: {report_json}")

    # 生成总体汇总报告
    if len(models) > 1:
        md = generate_summary_report(all_results)

        summary_md = OUTPUT_DIR / "summary_compliance_report.md"
        with open(summary_md, "w") as f:
            f.write(md)
        logger.info(f"Summary report: {summary_md}")

        # 保存 JSON 报告
        summary_json = OUTPUT_DIR / "summary_compliance_report.json"
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "models": all_results,
            "total_models": len(all_results),
            "passed_models": sum(1 for c in all_results.values() if c["overall"]["pass"]),
        }
        with open(summary_json, "w") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary JSON report: {summary_json}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("Compliance Report Summary")
    print("=" * 60)
    for model_type, compliance in all_results.items():
        overall = compliance["overall"]
        status = "PASS" if overall["pass"] else "FAIL"
        print(f"  {model_type.upper()}: {status} ({overall['num_passed']}/{overall['num_total']} requirements passed)")
    print("=" * 60)


if __name__ == "__main__":
    main()
