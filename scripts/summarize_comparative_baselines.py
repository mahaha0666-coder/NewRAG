"""Summarize the eight comparative baselines and the MechRAG result.

This script intentionally separates target-request errors from judge errors and
keeps every MechVQA question in the denominator.  The self-dataset score uses
only the 26 objective fields with human-created ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


BASELINE_ORDER = (
    "flash-global-direct",
    "flash-global-rationale",
    "flash-global-decompose",
    "flash-global-verify",
    "flash-multiview-direct",
    "flash-multiview-rationale",
    "flash-multiview-decompose",
    "flash-multiview-verify",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wilson(correct: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count <= 0:
        return [0.0, 0.0]
    proportion = correct / count
    denominator = 1.0 + z * z / count
    centre = (proportion + z * z / (2.0 * count)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def _bucket() -> dict[str, int]:
    return {"count": 0, "correct": 0, "target_errors": 0, "judge_errors": 0}


def _finish(bucket: dict[str, int]) -> dict[str, Any]:
    count = bucket["count"]
    correct = bucket["correct"]
    return {
        **bucket,
        "accuracy": correct / count if count else 0.0,
        "ci95_wilson": wilson(correct, count),
    }


def _judge_failed(result: dict[str, Any]) -> bool:
    judges = result.get("judge_results", {})
    return not judges or all(bool(value.get("error")) for value in judges.values())


def summarize_mechvqa(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in BASELINE_ORDER:
        overall = _bucket()
        grouped: dict[str, dict[str, dict[str, int]]] = {
            "by_capability": {},
            "by_difficulty": {},
            "by_subcategory": {},
            "by_language": {},
        }
        total_tokens = 0
        total_latency = 0.0
        for row in rows:
            response = row.get("responses", {}).get(name, {})
            result = row.get("evaluation_result", {}).get(name, {})
            metadata = row.get("original_record", {}).get("metadata", {})
            correct = float(result.get("voted_score", 0.0)) == 1.0
            target_error = bool(response.get("response_error"))
            judge_error = _judge_failed(result)
            overall["count"] += 1
            overall["correct"] += int(correct)
            overall["target_errors"] += int(target_error)
            overall["judge_errors"] += int(judge_error)
            total_tokens += int(response.get("usage", {}).get("total_tokens", 0) or 0)
            total_latency += float(response.get("latency_seconds", 0.0) or 0.0)
            for group_name, metadata_key in (
                ("by_capability", "capability"),
                ("by_difficulty", "difficulty"),
                ("by_subcategory", "subcategory"),
                ("by_language", "language"),
            ):
                value = str(metadata.get(metadata_key, "Unknown"))
                bucket = grouped[group_name].setdefault(value, _bucket())
                bucket["count"] += 1
                bucket["correct"] += int(correct)
                bucket["target_errors"] += int(target_error)
                bucket["judge_errors"] += int(judge_error)
        summary[name] = {
            "overall": _finish(overall),
            "total_tokens": total_tokens,
            "mean_latency_seconds": total_latency / overall["count"] if overall["count"] else 0.0,
            **{
                group_name: {key: _finish(value) for key, value in buckets.items()}
                for group_name, buckets in grouped.items()
            },
        }
    return summary


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def group_accuracy(groups: dict[str, Any], key: str) -> float:
    return float(groups.get(key, {}).get("accuracy", 0.0))


def _display_name(name: str) -> str:
    return {
        "flash-global-direct": "Flash + Global + Direct",
        "flash-global-rationale": "Flash + Global + Rationale",
        "flash-global-decompose": "Flash + Global + Decompose",
        "flash-global-verify": "Flash + Global + Verify",
        "flash-multiview-direct": "Flash + Multi-view + Direct",
        "flash-multiview-rationale": "Flash + Multi-view + Rationale",
        "flash-multiview-decompose": "Flash + Multi-view + Decompose",
        "flash-multiview-verify": "Flash + Multi-view + Verify",
    }[name]


def write_outputs(
    output_dir: Path,
    mechvqa: dict[str, Any],
    self_summary: dict[str, Any],
    ours_mechvqa: dict[str, Any],
) -> None:
    combined = {
        "evaluation_scope": {
            "mechvqa_questions": 1185,
            "mechvqa_drawings": 562,
            "self_dataset_documents": 10,
            "self_dataset_objective_fields": 26,
            "judge": "deepseek-flash-judge, binary semantic correctness",
        },
        "baselines": {
            name: {"mechvqa": mechvqa[name], "self_dataset": self_summary[name]}
            for name in BASELINE_ORDER
        },
        "mechrag": {
            "mechvqa": ours_mechvqa,
            "self_dataset": {
                "count": 26,
                "correct": 23,
                "accuracy": 23 / 26,
                "by_field": {
                    "drawing_no": {"count": 9, "correct": 7, "accuracy": 7 / 9},
                    "bom_line_items": {"count": 9, "correct": 9, "accuracy": 1.0},
                    "bom_quantity_sum": {"count": 8, "correct": 7, "accuracy": 7 / 8},
                },
            },
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output_dir / "overall_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "system",
                "self_correct",
                "self_count",
                "self_accuracy",
                "mechvqa_correct",
                "mechvqa_count",
                "mechvqa_accuracy",
                "target_errors",
                "judge_errors",
            ]
        )
        for name in BASELINE_ORDER:
            self_result = self_summary[name]
            public_result = mechvqa[name]["overall"]
            writer.writerow(
                [
                    name,
                    self_result["correct"],
                    self_result["count"],
                    self_result["accuracy"],
                    public_result["correct"],
                    public_result["count"],
                    public_result["accuracy"],
                    public_result["target_errors"],
                    public_result["judge_errors"],
                ]
            )
        ours = ours_mechvqa["overall"]
        writer.writerow(
            [
                "MechRAG (ours)",
                23,
                26,
                23 / 26,
                ours["correct"],
                ours["count"],
                ours["accuracy"],
                ours.get("error_count", 0),
                0,
            ]
        )

    lines = [
        "# 八组同协议基线与 MechRAG 实验结果",
        "",
        "所有 MechVQA 分数均在 1,185 个问题上按同一 DeepSeek Flash 二元语义评审器计算；请求或评审失败保留在分母并计错。自建数据集仅统计 26 个有人工真值的客观字段。八个比较项是同一可用视觉骨干的两种视图输入与四种标准推理策略的组合，不代表八个互不相关的厂商模型。",
        "",
        "| 系统 | 自建数据集（正确/26） | MechVQA（正确/1,185） | 目标错误 | 评审错误 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in BASELINE_ORDER:
        self_result = self_summary[name]
        public_result = mechvqa[name]["overall"]
        lines.append(
            f"| {_display_name(name)} | {self_result['correct']}/{self_result['count']} ({percent(self_result['accuracy'])}) "
            f"| {public_result['correct']}/{public_result['count']:,} ({percent(public_result['accuracy'])}) "
            f"| {public_result['target_errors']} | {public_result['judge_errors']} |"
        )
    ours = ours_mechvqa["overall"]
    lines.append(
        f"| **MechRAG（ours）** | **23/26 ({percent(23 / 26)})** "
        f"| **{ours['correct']}/1,185 ({percent(ours['accuracy'])})** "
        f"| {ours.get('error_count', 0)} | 0 |"
    )
    lines.extend(
        [
            "",
            "## MechVQA 能力分层",
            "",
            "| 系统 | Recognition | Reasoning | Judging | Overall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in BASELINE_ORDER:
        by_capability = mechvqa[name]["by_capability"]
        lines.append(
            f"| {_display_name(name)} | {percent(group_accuracy(by_capability, 'Recognition'))} "
            f"| {percent(group_accuracy(by_capability, 'Reasoning'))} "
            f"| {percent(group_accuracy(by_capability, 'Judging'))} "
            f"| {percent(mechvqa[name]['overall']['accuracy'])} |"
        )
    ours_cap = ours_mechvqa["by_capability"]
    lines.append(
        f"| **MechRAG（ours）** | **{percent(ours_cap['Recognition']['accuracy'])}** "
        f"| **{percent(ours_cap['Reasoning']['accuracy'])}** "
        f"| **{percent(ours_cap['Judging']['accuracy'])}** "
        f"| **{percent(ours['accuracy'])}** |"
    )
    lines.extend(
        [
            "",
            "说明：这些结果用于同一运行协议下的系统消融比较。评审器与 Flash 目标模型属于同一模型系列，可能带来同源评审偏差；小分差不应用于宣称模型排名。",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluated", type=Path, required=True)
    parser.add_argument("--self-summary", type=Path, required=True)
    parser.add_argument("--ours-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.evaluated)
    self_summary = json.loads(args.self_summary.read_text(encoding="utf-8"))
    ours_summary = json.loads(args.ours_summary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(args.output_dir, summarize_mechvqa(rows), self_summary, ours_summary)


if __name__ == "__main__":
    main()
