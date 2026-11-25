"""Prepare and evaluate a folder of engineering drawing PDFs with DeepSeek."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from mechrag.drawing_pdf import (
    DRAWING_CONDITIONS,
    build_drawing_message_content,
    manifest_json,
    prepare_pdf_dataset,
    response_to_evidence,
)
from mechrag.evidence import EvidenceStore
from mechrag.pipeline import EvidenceFirstPipeline, RetrievalBundle
from mechrag.providers import DEEPSEEK_MODEL, create_deepseek_openai_client
from mechrag.selection import CoverageReport


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render engineering drawing PDFs, ask four evidence-grounded test "
            "questions, and verify every claim-to-region edge."
        )
    )
    parser.add_argument("pdf_dir", help="Folder containing source PDF drawings")
    parser.add_argument("output_dir", help="Output folder for assets and reports")
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--model", default=DEEPSEEK_MODEL)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _answer_text(completion: Any) -> str:
    content = completion.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, Mapping) else str(item)
            for item in content
        )
    return str(content)


def _verification_bundle(document_id: str, units) -> RetrievalBundle:
    condition_scores = {condition.condition_id: 1.0 for condition in DRAWING_CONDITIONS}
    report = CoverageReport(
        condition_scores=condition_scores,
        selected_evidence_ids=tuple(unit.evidence_id for unit in units),
        total_cost=sum(unit.cost for unit in units),
        budget=sum(unit.cost for unit in units),
    )
    return RetrievalBundle(
        plan=type("DrawingPlan", (), {"conditions": DRAWING_CONDITIONS})(),
        candidate_part_ids=(document_id,),
        reranked_parts=(),
        selected_evidence=tuple(units),
        coverage=report,
        prompt="",
    )


def _claims_json(units) -> str:
    claims = []
    for index, (condition, unit) in enumerate(zip(DRAWING_CONDITIONS, units), start=1):
        claims.append(
            {
                "claim_id": f"J{index}",
                "text": unit.summary,
                "condition_ids": [condition.condition_id],
                "evidence_ids": [unit.evidence_id],
            }
        )
    return json.dumps({"claims": claims}, ensure_ascii=False)


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 自建工程图 PDF 测试报告",
        "",
        "每份图纸均执行四类任务：实体识别与去重计数、标题栏/表格/技术要求、",
        "剖视与视图语义、装配与配合关系。`valid` 仅表示证据引用和条件覆盖通过，",
        "不等同于人工标注的答案准确率。",
        "",
    ]
    for row in rows:
        if row.get("error"):
            lines.extend(
                [
                    f"## {row['document_id']}",
                    "",
                    f"- 运行失败：`{row['error_type']}`",
                    f"- 错误：{row['error']}",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                f"## {row['document_id']}",
                "",
                f"- 验证结果：`valid={str(row['verification']['valid']).lower()}`",
                f"- 文本层可用：`{str(row['text_layer_usable']).lower()}`",
                f"- 限制：{'；'.join(row.get('limitations') or ['无'])}",
                "",
            ]
        )
        for result in row["results"]:
            lines.extend(
                [
                    f"### {result['condition_id']} {result['feature_id']}",
                    "",
                    str(result.get("answer", "")),
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = list(
        prepare_pdf_dataset(args.pdf_dir, output_dir / "prepared", dpi=args.dpi)
    )
    if args.limit is not None:
        documents = documents[: args.limit]
    (output_dir / "manifest.json").write_text(
        manifest_json(documents), encoding="utf-8"
    )
    (output_dir / "questions.json").write_text(
        json.dumps(
            [
                {
                    "condition_id": condition.condition_id,
                    "question": condition.description,
                    "property": condition.property_names[0],
                }
                for condition in DRAWING_CONDITIONS
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "prepared_documents": len(documents),
                    "manifest": str((output_dir / "manifest.json").resolve()),
                },
                ensure_ascii=False,
            )
        )
        return 0

    client = create_deepseek_openai_client()
    result_rows = []
    all_evidence = []
    raw_dir = output_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for document in documents:
        try:
            completion = client.chat.completions.create(
                model=args.model,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": build_drawing_message_content(document),
                    }
                ],
            )
            raw_response = _answer_text(completion)
            (raw_dir / f"{document.document_id}.json").write_text(
                raw_response, encoding="utf-8"
            )
            units, payload = response_to_evidence(document, raw_response)
            bundle = _verification_bundle(document.document_id, units)
            verification = EvidenceFirstPipeline(EvidenceStore(units)).verify_response(
                _claims_json(units), bundle
            )
            all_evidence.extend(units)
            result_rows.append(
                {
                    "document_id": document.document_id,
                    "pdf_path": document.pdf_path,
                    "text_layer_usable": document.text_layer_usable,
                    "results": payload["results"],
                    "limitations": payload.get("limitations") or [],
                    "verification": {
                        "valid": verification.valid,
                        "unsupported_claim_ids": list(
                            verification.unsupported_claim_ids
                        ),
                        "unknown_evidence_ids": list(
                            verification.unknown_evidence_ids
                        ),
                        "uncovered_condition_ids": list(
                            verification.uncovered_condition_ids
                        ),
                    },
                }
            )
        except Exception as exc:
            result_rows.append(
                {
                    "document_id": document.document_id,
                    "pdf_path": document.pdf_path,
                    "text_layer_usable": document.text_layer_usable,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    EvidenceStore(all_evidence).save_jsonl(output_dir / "evidence.jsonl")
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in result_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_report(output_dir / "report.md", result_rows)
    print(
        json.dumps(
            {
                "evaluated_documents": len(result_rows),
                "valid_documents": sum(
                    row.get("verification", {}).get("valid", False)
                    for row in result_rows
                ),
                "failed_documents": sum("error" in row for row in result_rows),
                "report": str((output_dir / "report.md").resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
