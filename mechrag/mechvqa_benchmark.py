"""Run the MechRAG evidence-first baseline on the public MechVQA benchmark.

MechVQA supplies a question and one drawing image for each record.  Retrieval
therefore has a single known candidate drawing; the MechRAG baseline still
constructs a local evidence unit, applies budgeted evidence selection, builds
the evidence-contract prompt, and verifies the returned claim/evidence links.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from mimetypes import guess_type
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterable, Mapping

from PIL import Image, ImageOps
from tqdm import tqdm

from .evidence import BoundingBox, EvidenceSource, EvidenceStore, EvidenceUnit, ViewRegion
from .pipeline import EvidenceFirstPipeline
from .providers import DEEPSEEK_MODEL, create_deepseek_openai_client
from .questions import RuleBasedQuestionDecomposer


MODEL_NAME = "mechrag-evidence-deepseek-flash"
VIEW_BOXES = {
    "overview": (0.0, 0.0, 1.0, 1.0),
    "tile_top_left": (0.0, 0.0, 0.58, 0.58),
    "tile_top_right": (0.42, 0.0, 1.0, 0.58),
    "tile_bottom_left": (0.0, 0.42, 0.58, 1.0),
    "tile_bottom_right": (0.42, 0.42, 1.0, 1.0),
}


@dataclass(frozen=True)
class PreparedView:
    view_id: str
    path: str
    bbox: tuple[float, float, float, float]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(output)


def user_question(record: Mapping[str, Any]) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return str(record.get("question", ""))


def record_images(record: Mapping[str, Any]) -> list[str]:
    value = record.get("images", [])
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in value if item]


def _save_jpeg(image: Image.Image, path: Path, *, max_side: int) -> None:
    output = ImageOps.exif_transpose(image).convert("RGB")
    output.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path, format="JPEG", quality=92, optimize=True)


def prepare_views(
    image_path: str | Path,
    output_root: str | Path,
    *,
    max_side: int = 2048,
) -> tuple[PreparedView, ...]:
    """Create one overview and four overlapping local views for a drawing."""

    source = Path(image_path)
    relative_key = f"{source.parent.name}_{source.stem}"
    target_dir = Path(output_root) / relative_key
    expected = tuple(
        PreparedView(view_id, str(target_dir / f"{view_id}.jpg"), bbox)
        for view_id, bbox in VIEW_BOXES.items()
    )
    if all(Path(view.path).is_file() for view in expected):
        return expected

    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        for view in expected:
            target = Path(view.path)
            if target.is_file():
                continue
            if view.view_id == "overview":
                crop = image.copy()
            else:
                x0, y0, x1, y1 = view.bbox
                pixel_box = (
                    round(x0 * width),
                    round(y0 * height),
                    round(x1 * width),
                    round(y1 * height),
                )
                crop = image.crop(pixel_box)
            _save_jpeg(crop, target, max_side=max_side)
    return expected


def _image_data_url(path: str | Path) -> str:
    source = Path(path)
    mime_type = guess_type(source.name)[0] or "image/jpeg"
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object from the target model")
    return payload


def _completion_text(completion: Any) -> str:
    content = completion.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, Mapping) else str(item)
            for item in content
        )
    return str(content or "")


def _usage_dict(completion: Any) -> dict[str, int]:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return {}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def build_bundle(
    *,
    record_idx: int,
    image_path: str,
    question: str,
    views: tuple[PreparedView, ...],
):
    """Build one selected, cross-view evidence unit using the project source."""

    decomposer = RuleBasedQuestionDecomposer()
    plan = decomposer.decompose(question)
    property_names = {
        name for condition in plan.conditions for name in condition.property_names
    }
    part_id = Path(image_path).stem
    unit = EvidenceUnit(
        part_id=part_id,
        feature_id=f"mechvqa_q{record_idx:04d}",
        feature_type="paired_drawing_evidence",
        view_regions=tuple(
            ViewRegion(
                view_id=view.view_id,
                image_uri=view.path,
                bbox=BoundingBox(*view.bbox),
            )
            for view in views
        ),
        geometric_properties={
            name: {"status": "requires visual verification"}
            for name in property_names
        },
        sources=(
            EvidenceSource(
                uri=image_path,
                record_id=f"mechvqa:{record_idx}",
                extractor="mechrag.mechvqa_benchmark",
            ),
        ),
        summary=(
            question
            + " The overview and overlapping local regions are grouped as one "
            "cross-view evidence unit; repeated projections must not be counted twice."
        ),
        family_id=part_id,
        cost=1.0,
    )
    store = EvidenceStore(
        [unit], expected_part_ids=[part_id], universe_complete=True
    )
    pipeline = EvidenceFirstPipeline(store, decomposer=decomposer)
    bundle = pipeline.retrieve(
        question,
        overall_candidate_part_ids=[part_id],
        candidate_limit=1,
        evidence_budget=1.0,
    )
    if not bundle.selected_evidence:
        raise RuntimeError("The evidence selector returned no drawing evidence")
    return pipeline, bundle


def _message_content(prompt: str, views: tuple[PreparedView, ...]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                prompt
                + "\nThe cited evidence unit is supplied below as an overview and four "
                "overlapping local views. Inspect all views, but do not reveal hidden "
                "chain-of-thought. Return exactly one concise claim for each condition. "
                "Keep each claim under 80 words and `answer` under 120 words. Preserve "
                "the exact language of the question."
            ),
        }
    ]
    for view in views:
        content.append(
            {
                "type": "text",
                "text": f"[VIEW:{view.view_id}] normalized_bbox={list(view.bbox)}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(view.path)},
            }
        )
    return content


def _request_with_retries(client: Any, kwargs: dict[str, Any], *, attempts: int = 5):
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # provider exceptions vary across SDK versions
            last_error = exc
            if attempt + 1 == attempts:
                break
            time.sleep(min(20.0, 2.0 ** attempt))
    assert last_error is not None
    raise last_error


def _generate_json_payload(
    client: Any,
    request_kwargs: dict[str, Any],
    *,
    model: str,
) -> tuple[dict[str, Any], str, list[str], dict[str, int]]:
    """Generate strict JSON, repairing only the serialization when necessary."""

    parse_errors: list[str] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    raw_response = ""
    for generation_attempt in range(2):
        completion = _request_with_retries(client, request_kwargs)
        usage = _usage_dict(completion)
        for key in total_usage:
            total_usage[key] += usage.get(key, 0)
        raw_response = _completion_text(completion)
        try:
            return _extract_json(raw_response), raw_response, parse_errors, total_usage
        except Exception as exc:
            parse_errors.append(
                f"generation {generation_attempt + 1}: {type(exc).__name__}: {exc}"
            )
            if not raw_response.strip():
                continue

            repair = _request_with_retries(
                client,
                {
                    "model": model,
                    "temperature": 0,
                    "max_tokens": 1200,
                    "extra_body": {"thinking": {"type": "disabled"}},
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Repair only the JSON serialization of the response below. "
                                "Do not add new engineering facts. Return one strict JSON object "
                                "with `claims` and `answer`.\n\n" + raw_response
                            ),
                        }
                    ],
                },
            )
            repair_usage = _usage_dict(repair)
            for key in total_usage:
                total_usage[key] += repair_usage.get(key, 0)
            repaired = _completion_text(repair)
            try:
                return _extract_json(repaired), repaired, parse_errors, total_usage
            except Exception as repair_exc:
                parse_errors.append(
                    f"repair {generation_attempt + 1}: "
                    f"{type(repair_exc).__name__}: {repair_exc}"
                )
    raise ValueError("Target did not return parseable JSON: " + " | ".join(parse_errors))


def run_record(
    *,
    client: Any,
    record_idx: int,
    record: dict[str, Any],
    image_root: Path,
    prepared_root: Path,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    question = user_question(record)
    images = record_images(record)
    if len(images) != 1:
        raise ValueError(f"Expected exactly one image, found {len(images)}")
    image_path = (image_root / images[0]).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    views = prepare_views(image_path, prepared_root)
    pipeline, bundle = build_bundle(
        record_idx=record_idx,
        image_path=str(image_path),
        question=question,
        views=views,
    )
    payload, raw_response, parse_errors, usage = _generate_json_payload(
        client,
        {
            "model": model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "extra_body": {"thinking": {"type": "disabled"}},
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "user", "content": _message_content(bundle.prompt, views)}
            ],
        },
        model=model,
    )
    model_answer = str(payload.get("answer", "")).strip()
    if not model_answer:
        raise ValueError("Target response contains no non-empty `answer`")
    normalized_response = json.dumps(payload, ensure_ascii=False)
    try:
        verification = pipeline.verify_response(normalized_response, bundle)
        evidence_validation = {
            "valid": verification.valid,
            "unsupported_claim_ids": list(verification.unsupported_claim_ids),
            "unknown_evidence_ids": list(verification.unknown_evidence_ids),
            "uncovered_condition_ids": list(verification.uncovered_condition_ids),
        }
    except Exception as exc:
        evidence_validation = {
            "valid": False,
            "verification_error": f"{type(exc).__name__}: {exc}",
            "unsupported_claim_ids": [],
            "unknown_evidence_ids": [],
            "uncovered_condition_ids": [
                condition.condition_id for condition in bundle.plan.conditions
            ],
        }
    evidence_validation.update(
        {
            "selected_evidence_ids": list(bundle.coverage.selected_evidence_ids),
            "coverage": bundle.coverage.coverage,
            "cost": bundle.coverage.total_cost,
            "budget": bundle.coverage.budget,
        }
    )
    return {
        "model_response": raw_response,
        "model_answer": model_answer,
        "response_error": None,
        "evidence_validation": evidence_validation,
        "format_repair_attempts": parse_errors,
        "prepared_views": [asdict(view) for view in views],
        "usage": usage,
        "latency_seconds": round(time.perf_counter() - started, 3),
    }


def _failure_response(exc: Exception, started: float) -> dict[str, Any]:
    return {
        "model_response": "",
        "model_answer": "",
        "response_error": f"{type(exc).__name__}: {exc}",
        "evidence_validation": {"valid": False},
        "prepared_views": [],
        "usage": {},
        "latency_seconds": round(time.perf_counter() - started, 3),
    }


def _git_commit(repository: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEEPSEEK_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required at runtime")

    input_file = Path(args.input_file).resolve()
    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    prepared_root = output_dir / "prepared_views"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prepared_root.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_file)
    if args.limit is not None:
        records = records[: args.limit]
    client = create_deepseek_openai_client()

    def execute(index: int, record: dict[str, Any]) -> dict[str, Any]:
        checkpoint = checkpoint_dir / f"{index:04d}.json"
        if checkpoint.is_file():
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
            cached_response = cached.get("responses", {}).get(MODEL_NAME, {})
            if not args.retry_errors or not cached_response.get("response_error"):
                return cached
        started = time.perf_counter()
        try:
            response = run_record(
                client=client,
                record_idx=index,
                record=record,
                image_root=image_root,
                prepared_root=prepared_root,
                model=args.model,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:
            response = _failure_response(exc, started)
        row = {
            "record_idx": index,
            "original_record": record,
            "responses": {MODEL_NAME: response},
        }
        temporary = checkpoint.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        temporary.replace(checkpoint)
        return row

    completed: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(execute, index, record): index
            for index, record in enumerate(records)
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="MechRAG baseline"
        ):
            row = future.result()
            completed[int(row["record_idx"])] = row

    phase1_output = output_dir / "responses_mechrag_baseline.jsonl"
    ordered = [completed[index] for index in range(len(records))]
    write_jsonl(ordered, phase1_output)

    evaluated_output = output_dir / "evaluated_mechrag_baseline.jsonl"
    stats_output = output_dir / "official_stats.json"
    config = {
        "input_file": str(input_file),
        "image_root": str(image_root),
        "phase1_output": str(phase1_output),
        "evaluated_output": str(evaluated_output),
        "stats_output": str(stats_output),
        "max_workers": args.judge_workers,
        "append_answer_suffix": False,
        "target_models": [],
        "judge_models": [
            {
                "name": "deepseek-flash-judge",
                "model": args.model,
                "base_url": "https://api.deepseek.com",
                "api_key_env": "DEEPSEEK_API_KEY",
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout": 180,
                "extra_body": {"thinking": {"type": "disabled"}},
            }
        ],
    }
    config_path = output_dir / "official_eval_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    provenance = {
        "benchmark": "MechVQA public evaluation benchmark",
        "benchmark_records": len(records),
        "benchmark_unique_images": len(
            {image for record in records for image in record_images(record)}
        ),
        "benchmark_repository": str(image_root.parent.resolve()),
        "benchmark_commit": _git_commit(image_root.parent),
        "target_system": MODEL_NAME,
        "target_model": args.model,
        "target_temperature": 0,
        "judge_model": args.model,
        "judge_temperature": 0,
        "views_per_question": len(VIEW_BOXES),
        "evidence_budget": 1.0,
        "phase1_output": str(phase1_output),
        "official_eval_config": str(config_path),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    failures = sum(
        bool(row["responses"][MODEL_NAME].get("response_error")) for row in ordered
    )
    evidence_valid = sum(
        bool(
            row["responses"][MODEL_NAME]
            .get("evidence_validation", {})
            .get("valid")
        )
        for row in ordered
    )
    print(
        json.dumps(
            {
                "records": len(ordered),
                "failures": failures,
                "evidence_valid": evidence_valid,
                "phase1_output": str(phase1_output),
                "official_eval_config": str(config_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
