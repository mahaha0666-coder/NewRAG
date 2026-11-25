"""Summarize official MechVQA judge output into paper-ready JSON and Markdown."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .mechvqa_benchmark import MODEL_NAME, load_jsonl


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _result(record: Mapping[str, Any], target_model: str) -> Mapping[str, Any]:
    return record.get("evaluation_result", {}).get(target_model, {})


def summarize_rows(rows: Iterable[Mapping[str, Any]], target_model: str) -> dict[str, Any]:
    records = list(rows)
    successes = sum(float(_result(row, target_model).get("voted_score", 0)) == 1.0 for row in records)
    errors = sum(bool(_result(row, target_model).get("error")) for row in records)
    low, high = wilson_interval(successes, len(records))
    return {
        "count": len(records),
        "correct": successes,
        "accuracy": successes / len(records) if records else 0.0,
        "ci95_wilson": [low, high],
        "error_count": errors,
    }


def group_summary(records: list[dict[str, Any]], field: str, target_model: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = str(record.get("original_record", {}).get("metadata", {}).get(field, "unknown"))
        groups[key].append(record)
    return {
        key: summarize_rows(rows, target_model)
        for key, rows in sorted(groups.items(), key=lambda item: item[0])
    }


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _table(title: str, groups: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = [f"## {title}", "", "| 分组 | 样本数 | 正确数 | 准确率 | 95% Wilson CI | 错误数 |", "|---|---:|---:|---:|---:|---:|"]
    for name, values in groups.items():
        low, high = values["ci95_wilson"]
        lines.append(
            f"| {name} | {values['count']} | {values['correct']} | "
            f"{_percent(values['accuracy'])} | {_percent(low)}–{_percent(high)} | "
            f"{values['error_count']} |"
        )
    lines.append("")
    return lines


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluated-file", required=True)
    parser.add_argument("--phase1-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-model", default=MODEL_NAME)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    evaluated = load_jsonl(args.evaluated_file)
    phase1 = load_jsonl(args.phase1_file)
    target_model = args.target_model
    overall = summarize_rows(evaluated, target_model)
    evidence_valid = sum(
        bool(
            row.get("responses", {})
            .get(target_model, {})
            .get("evidence_validation", {})
            .get("valid")
        )
        for row in phase1
    )
    target_failures = sum(
        bool(row.get("responses", {}).get(target_model, {}).get("response_error"))
        for row in phase1
    )
    total_tokens = sum(
        int(
            row.get("responses", {})
            .get(target_model, {})
            .get("usage", {})
            .get("total_tokens", 0)
        )
        for row in phase1
    )
    latencies = [
        float(row.get("responses", {}).get(target_model, {}).get("latency_seconds", 0))
        for row in phase1
    ]
    summary = {
        "target_model": target_model,
        "judge_model": "deepseek-flash-judge",
        "overall": overall,
        "target_failures": target_failures,
        "evidence_valid_count": evidence_valid,
        "evidence_valid_rate": evidence_valid / len(phase1) if phase1 else 0.0,
        "target_total_tokens": total_tokens,
        "mean_target_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "by_capability": group_summary(evaluated, "capability", target_model),
        "by_subcategory": group_summary(evaluated, "subcategory", target_model),
        "by_difficulty": group_summary(evaluated, "difficulty", target_model),
        "by_language": group_summary(evaluated, "language", target_model),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    low, high = overall["ci95_wilson"]
    lines = [
        "# MechRAG 基线在 MechVQA 公共测试集上的结果",
        "",
        f"- 样本：{overall['count']} 个问答。",
        f"- 正确：{overall['correct']}。",
        f"- 总准确率：{_percent(overall['accuracy'])}。",
        f"- 95% Wilson 置信区间：{_percent(low)}–{_percent(high)}。",
        f"- 目标推理失败：{target_failures}。",
        f"- 证据链校验通过：{evidence_valid}/{len(phase1)}（{_percent(summary['evidence_valid_rate'])}）。",
        "- 目标模型与单一裁判均为 DeepSeek Flash；该结果应标注为同模型自动评审。",
        "",
    ]
    lines.extend(_table("按能力层级", summary["by_capability"]))
    lines.extend(_table("按十个子任务", summary["by_subcategory"]))
    lines.extend(_table("按难度", summary["by_difficulty"]))
    lines.extend(_table("按语言", summary["by_language"]))
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
