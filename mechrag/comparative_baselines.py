"""Run eight reproducible visual-QA baselines on MechVQA and the self dataset.

The available DeepSeek account exposes one image-capable backbone.  We cross
two visual-input settings with four standard inference settings so every
baseline can be run on both datasets with the same endpoint and deterministic
decoding:

* global/direct: one overview image and a short answer;
* global/rationale: one overview image and a short evidence rationale;
* global/decompose: one overview with explicit question/view decomposition;
* global/verify: one overview with a final visible-evidence consistency check;
* multiview/direct: overview plus four overlapping crops and a short answer;
* multiview/rationale: the five views with a short evidence rationale;
* multiview/decompose: the five views plus explicit question/view decomposition;
* multiview/verify: the five views with a final cross-view consistency check.

These are deliberately labelled as baseline *systems*, not eight independent
foundation-model families.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Iterable, Mapping

from tqdm import tqdm

from .mechvqa_benchmark import (
    _completion_text,
    _extract_json,
    _image_data_url,
    _request_with_retries,
    load_jsonl,
    prepare_views,
    record_images,
    user_question,
    write_jsonl,
)
from .providers import create_deepseek_openai_client


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    model: str
    view_mode: str
    prompt_mode: str


BASELINES = (
    BaselineConfig("flash-global-direct", "deepseek-flash", "global", "direct"),
    BaselineConfig("flash-global-rationale", "deepseek-flash", "global", "rationale"),
    BaselineConfig("flash-global-decompose", "deepseek-flash", "global", "decompose"),
    BaselineConfig("flash-global-verify", "deepseek-flash", "global", "verify"),
    BaselineConfig("flash-multiview-direct", "deepseek-flash", "multiview", "direct"),
    BaselineConfig("flash-multiview-rationale", "deepseek-flash", "multiview", "rationale"),
    BaselineConfig("flash-multiview-decompose", "deepseek-flash", "multiview", "decompose"),
    BaselineConfig("flash-multiview-verify", "deepseek-flash", "multiview", "verify"),
)


def _selected_baselines(names: Iterable[str]) -> tuple[BaselineConfig, ...]:
    requested = {name for name in names if name}
    if not requested:
        return BASELINES
    known = {config.name for config in BASELINES}
    unknown = requested - known
    if unknown:
        raise ValueError(f"Unknown baseline names: {sorted(unknown)}")
    return tuple(config for config in BASELINES if config.name in requested)


def _git_commit(path: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _extract_answer(raw: str) -> tuple[str, dict[str, Any] | None]:
    try:
        payload = _extract_json(raw)
    except Exception:
        payload = None
    if payload is not None:
        answer = str(payload.get("answer", "")).strip()
        if answer:
            return answer, payload
    tagged = re.search(r"<answer>(.*?)</answer>", raw, flags=re.I | re.S)
    if tagged:
        return tagged.group(1).strip(), payload
    return raw.strip(), payload


def _completion_request(
    client: Any,
    *,
    config: BaselineConfig,
    prompt: str,
    image_paths: Iterable[str | Path],
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, image_path in enumerate(image_paths):
        content.append({"type": "text", "text": f"[DRAWING_VIEW_{index + 1}]"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(image_path)},
            }
        )
    started = time.perf_counter()
    completion = _request_with_retries(
        client,
        {
            "model": config.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 700 if config.prompt_mode != "direct" else 400,
            "timeout": 180,
            "extra_body": {"thinking": {"type": "disabled"}},
        },
    )
    raw = _completion_text(completion)
    answer, payload = _extract_answer(raw)
    usage = getattr(completion, "usage", None)
    return {
        "model_response": raw,
        "model_answer": answer,
        "parsed_payload": payload,
        "response_error": None,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "usage": {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        },
    }


def _mechvqa_prompt(question: str, config: BaselineConfig) -> str:
    view_note = (
        "One overview of the engineering drawing is supplied."
        if config.view_mode == "global"
        else (
            "An overview and four overlapping crops of the same engineering drawing "
            "are supplied. Repeated geometry across views is the same physical object "
            "unless the drawing explicitly shows otherwise."
        )
    )
    if config.prompt_mode == "direct":
        method = (
            "Answer directly from the visible drawing. Return only a JSON object "
            'of the form {"answer":"concise answer"}.'
        )
    elif config.prompt_mode == "rationale":
        method = (
            "Identify the decisive visible annotation or geometry and provide a short "
            "evidence rationale. Do not reveal hidden chain-of-thought. Return only "
            'JSON: {"rationale":"brief visible evidence","answer":"concise answer"}.'
        )
    elif config.prompt_mode == "decompose":
        method = (
            "First decompose the question into the requested entity/property, select "
            "the relevant supplied view, and record the decisive visible cue. Do not "
            "reveal hidden chain-of-thought. Return only JSON: "
            '{"observations":["brief visible cue"],"answer":"concise answer"}.'
        )
    else:
        method = (
            "Draft a concise answer, then check it once against the visible drawing "
            "for unsupported attributes, counts, dimensions, and cross-view duplicates. "
            "Return only the verified final JSON: "
            '{"verification":"brief visible consistency check","answer":"concise answer"}.'
        )
    return (
        "You are evaluating a mechanical engineering drawing. "
        + view_note
        + "\nQuestion: "
        + question
        + "\n"
        + method
        + " Preserve the language and units used by the question."
    )


def _empty_mechvqa_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"record_idx": index, "original_record": record, "responses": {}}
        for index, record in enumerate(records)
    ]


def _load_resume_rows(
    output_path: Path, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not output_path.is_file():
        return _empty_mechvqa_rows(records)
    rows = load_jsonl(output_path)
    if len(rows) != len(records):
        raise ValueError(
            f"Resume file has {len(rows)} records, expected {len(records)}: {output_path}"
        )
    return rows


def run_mechvqa(
    *,
    client: Any,
    input_file: Path,
    image_root: Path,
    output_dir: Path,
    baselines: tuple[BaselineConfig, ...],
    workers: int,
    limit: int,
) -> Path:
    records = load_jsonl(input_file)
    if limit > 0:
        records = records[:limit]
    output_path = output_dir / "mechvqa_baseline_responses.jsonl"
    rows = _load_resume_rows(output_path, records)
    selected_names = {config.name for config in baselines}
    for row in rows:
        row["responses"] = {
            name: response
            for name, response in row.get("responses", {}).items()
            if name in selected_names
        }
    prepared_root = output_dir / "prepared_views"

    jobs: list[tuple[int, BaselineConfig]] = []
    for index, row in enumerate(rows):
        for config in baselines:
            prior = row.get("responses", {}).get(config.name)
            if prior is None or prior.get("response_error"):
                jobs.append((index, config))

    lock = threading.Lock()

    def run_one(index: int, config: BaselineConfig) -> tuple[int, str, dict[str, Any]]:
        record = records[index]
        images = record_images(record)
        if not images:
            raise ValueError(f"Record {index} contains no drawing image")
        source = image_root / images[0]
        views = prepare_views(source, prepared_root)
        selected = views[:1] if config.view_mode == "global" else views
        try:
            response = _completion_request(
                client,
                config=config,
                prompt=_mechvqa_prompt(user_question(record), config),
                image_paths=[view.path for view in selected],
            )
        except Exception as exc:
            response = {
                "model_response": "",
                "model_answer": "",
                "parsed_payload": None,
                "response_error": str(exc),
                "latency_seconds": 0.0,
                "usage": {},
            }
        response.update(
            {
                "backbone": config.model,
                "view_mode": config.view_mode,
                "prompt_mode": config.prompt_mode,
                "view_count": len(selected),
            }
        )
        return index, config.name, response

    output_dir.mkdir(parents=True, exist_ok=True)
    if not jobs:
        return output_path
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(run_one, *job) for job in jobs]
        for completed, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="baseline target"), 1
        ):
            index, name, response = future.result()
            with lock:
                rows[index].setdefault("responses", {})[name] = response
                if completed % 20 == 0:
                    write_jsonl(rows, output_path)
    write_jsonl(rows, output_path)
    return output_path


def _self_prompt(config: BaselineConfig) -> str:
    view_note = (
        "One overview image is supplied."
        if config.view_mode == "global"
        else (
            "One overview and four overlapping crops of the same drawing are supplied. "
            "Do not double-count an item repeated in multiple views."
        )
    )
    if config.prompt_mode == "direct":
        method = "Read the requested fields directly."
    elif config.prompt_mode == "rationale":
        method = (
            "Read the requested fields and include one short visible-evidence rationale. "
            "Do not reveal hidden chain-of-thought."
        )
    elif config.prompt_mode == "decompose":
        method = (
            "First locate the title block and BOM, then distinguish BOM row count from "
            "the sum of the quantity column. Include only brief visible observations."
        )
    else:
        method = (
            "Read the requested fields, then verify the title-block identifier, populated "
            "BOM row count, and quantity-column arithmetic once against the visible drawing."
        )
    return f"""You are reading a mechanical engineering drawing. {view_note}
{method}
Return only one JSON object with these keys:
{{"drawing_no":"exact drawing number or UNKNOWN",
  "bom_line_items": integer or null,
  "bom_quantity_sum": integer or null,
  "rationale":"optional short visible evidence"}}
Rules:
- drawing_no is the drawing identifier printed in the title block, not the PDF filename.
- bom_line_items is the number of populated BOM rows.
- bom_quantity_sum is the arithmetic sum of the BOM quantity column.
- Use null/UNKNOWN when the field is unreadable; never estimate.
"""


def _asset_map(document: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(asset.get("role")): str(asset.get("image_path"))
        for asset in document.get("assets", [])
    }


def _self_images(document: Mapping[str, Any], config: BaselineConfig) -> list[str]:
    assets = _asset_map(document)
    overview = assets.get("overview")
    if not overview:
        raise ValueError(f"No overview asset for {document.get('document_id')}")
    if config.view_mode == "global":
        return [overview]
    roles = ("overview", "tile_0_0", "tile_0_1", "tile_1_0", "tile_1_1")
    return [assets[role] for role in roles if assets.get(role)]


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"\d+", str(value).replace(",", ""))
    return int(match.group(0)) if match else None


def _drawing_key(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).upper().replace("＃", "#")
    return re.sub(r"\s+", "", text)


def _self_prediction(response: Mapping[str, Any]) -> dict[str, Any]:
    payload = response.get("parsed_payload")
    if not isinstance(payload, Mapping):
        return {"drawing_no": "", "bom_line_items": None, "bom_quantity_sum": None}
    return {
        "drawing_no": str(payload.get("drawing_no", "")).strip(),
        "bom_line_items": _int_or_none(payload.get("bom_line_items")),
        "bom_quantity_sum": _int_or_none(payload.get("bom_quantity_sum")),
    }


def _score_self(
    rows: list[dict[str, Any]], ground_truth: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    names = [config.name for config in BASELINES]
    summary: dict[str, Any] = {}
    for name in names:
        field_stats = {
            "drawing_no": {"correct": 0, "count": 0},
            "bom_line_items": {"correct": 0, "count": 0},
            "bom_quantity_sum": {"correct": 0, "count": 0},
        }
        errors = 0
        for row in rows:
            truth = ground_truth[str(row["document_id"])]
            response = row.get("responses", {}).get(name, {})
            prediction = response.get("prediction", {})
            errors += int(bool(response.get("response_error")))
            for field, bucket in field_stats.items():
                expected = truth.get(field)
                if expected is None:
                    continue
                bucket["count"] += 1
                if field == "drawing_no":
                    correct = _drawing_key(prediction.get(field)) == _drawing_key(expected)
                else:
                    correct = _int_or_none(prediction.get(field)) == int(expected)
                bucket["correct"] += int(correct)
        total = sum(bucket["count"] for bucket in field_stats.values())
        correct = sum(bucket["correct"] for bucket in field_stats.values())
        for bucket in field_stats.values():
            bucket["accuracy"] = bucket["correct"] / bucket["count"] if bucket["count"] else 0.0
        summary[name] = {
            "count": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "request_errors": errors,
            "by_field": field_stats,
        }
    return summary


def run_self_dataset(
    *,
    client: Any,
    manifest_file: Path,
    ground_truth_file: Path,
    output_dir: Path,
    baselines: tuple[BaselineConfig, ...],
    workers: int,
    limit: int,
) -> tuple[Path, Path]:
    documents = json.loads(manifest_file.read_text(encoding="utf-8"))
    truth = json.loads(ground_truth_file.read_text(encoding="utf-8"))
    if limit > 0:
        documents = documents[:limit]
    response_path = output_dir / "self_dataset_baseline_responses.jsonl"
    if response_path.is_file():
        rows = load_jsonl(response_path)
        if len(rows) != len(documents):
            raise ValueError("Self-dataset resume file length does not match manifest")
    else:
        rows = [
            {"document_id": document["document_id"], "responses": {}}
            for document in documents
        ]
    selected_names = {config.name for config in baselines}
    for row in rows:
        row["responses"] = {
            name: response
            for name, response in row.get("responses", {}).items()
            if name in selected_names
        }

    jobs: list[tuple[int, BaselineConfig]] = []
    for index, row in enumerate(rows):
        for config in baselines:
            prior = row.get("responses", {}).get(config.name)
            if prior is None or prior.get("response_error"):
                jobs.append((index, config))

    def run_one(index: int, config: BaselineConfig) -> tuple[int, str, dict[str, Any]]:
        document = documents[index]
        images = _self_images(document, config)
        try:
            response = _completion_request(
                client,
                config=config,
                prompt=_self_prompt(config),
                image_paths=images,
            )
        except Exception as exc:
            response = {
                "model_response": "",
                "model_answer": "",
                "parsed_payload": None,
                "response_error": str(exc),
                "latency_seconds": 0.0,
                "usage": {},
            }
        response["prediction"] = _self_prediction(response)
        response.update(
            {
                "backbone": config.model,
                "view_mode": config.view_mode,
                "prompt_mode": config.prompt_mode,
                "view_count": len(images),
            }
        )
        return index, config.name, response

    output_dir.mkdir(parents=True, exist_ok=True)
    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(run_one, *job) for job in jobs]
            for completed, future in enumerate(
                tqdm(as_completed(futures), total=len(futures), desc="self target"), 1
            ):
                index, name, response = future.result()
                rows[index].setdefault("responses", {})[name] = response
                if completed % 10 == 0:
                    write_jsonl(rows, response_path)
        write_jsonl(rows, response_path)

    summary_path = output_dir / "self_dataset_baseline_summary.json"
    summary_path.write_text(
        json.dumps(_score_self(rows, truth), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return response_path, summary_path


def _write_official_config(
    *,
    output_dir: Path,
    input_file: Path,
    image_root: Path,
    phase1_output: Path,
    judge_workers: int,
) -> Path:
    config_path = output_dir / "official_eval_config.json"
    payload = {
        "input_file": str(input_file.resolve()),
        "image_root": str(image_root.resolve()),
        "phase1_output": str(phase1_output.resolve()),
        "evaluated_output": str((output_dir / "evaluated_baselines.jsonl").resolve()),
        "stats_output": str((output_dir / "official_stats.json").resolve()),
        "max_workers": judge_workers,
        "append_answer_suffix": False,
        "target_models": [],
        "judge_models": [
            {
                "name": "deepseek-flash-judge",
                "model": "deepseek-flash",
                "base_url": "https://api.deepseek.com",
                "api_key_env": "DEEPSEEK_API_KEY",
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout": 180,
                "extra_body": {"thinking": {"type": "disabled"}},
            }
        ],
    }
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechvqa-input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--self-manifest", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("both", "mechvqa", "self"), default="both")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--judge-workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-config", action="append", default=[])
    args = parser.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY must be set at runtime")
    baselines = _selected_baselines(args.only_config)
    client = create_deepseek_openai_client()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase1_output: Path | None = None
    if args.dataset in {"both", "mechvqa"}:
        phase1_output = run_mechvqa(
            client=client,
            input_file=args.mechvqa_input,
            image_root=args.image_root,
            output_dir=args.output_dir,
            baselines=baselines,
            workers=args.workers,
            limit=args.limit,
        )
        _write_official_config(
            output_dir=args.output_dir,
            input_file=args.mechvqa_input,
            image_root=args.image_root,
            phase1_output=phase1_output,
            judge_workers=args.judge_workers,
        )
    if args.dataset in {"both", "self"}:
        run_self_dataset(
            client=client,
            manifest_file=args.self_manifest,
            ground_truth_file=args.ground_truth,
            output_dir=args.output_dir,
            baselines=baselines,
            workers=args.workers,
            limit=args.limit,
        )

    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_definition": [asdict(config) for config in baselines],
        "temperature": 0,
        "thinking": "disabled",
        "mechvqa_input": str(args.mechvqa_input.resolve()),
        "mechvqa_commit": _git_commit(args.mechvqa_input.parents[2]),
        "self_manifest": str(args.self_manifest.resolve()),
        "self_ground_truth": str(args.ground_truth.resolve()),
        "limit": args.limit,
        "phase1_output": str(phase1_output.resolve()) if phase1_output else None,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
