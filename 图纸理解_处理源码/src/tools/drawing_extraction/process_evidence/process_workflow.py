"""One resumable workflow from drawing observations to a cited process proposal."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from copy import deepcopy
import heapq
import json
from pathlib import Path
import time
from uuid import uuid4

from src.tools.research.artifacts import (
    digest,
    read_json,
    verify_manifest,
    write_new_json,
)

from .order_discovery import discover_order, order_input
from .plan_inputs import verify_drawing
from .plan_order import dependency_layers, order_audit, precedence_paths
from .plan_schema import validate_order_relations
from .process_plan import DEFAULT_ENDPOINT, DEFAULT_MODEL, fingerprint, infer_plan
from .process_plan import recipe as perception_recipe
from .target_grounding import ground_process_targets


def recipe(
    endpoint: str,
    model: str,
    *,
    perception_timeout_seconds: int = 600,
    separate_ordering: bool = False,
) -> dict:
    return {
        "schema": "dwg_process_workflow.v1",
        "perception": perception_recipe(
            endpoint,
            model,
            request_timeout_seconds=perception_timeout_seconds,
            separate_ordering=separate_ordering,
        ),
        # The separate stage also needs room for interleaved weld/cut sequences.
        "order": {
            "max_relations": 8 if separate_ordering else 4,
            "max_tokens": 2048 if separate_ordering else 1024,
        },
        "implementation": {
            name: digest(Path(__file__).with_name(name))
            for name in (
                "process_workflow.py",
                "order_discovery.py",
                "plan_order.py",
                "temporal_evidence.py",
                "target_grounding.py",
            )
        },
    }


def ordered_candidate(
    ids: list[str], relations: list[dict], preference: list[str]
) -> list[str]:
    """Keep the candidate's chosen actions, including its method alternatives.

    Reordering preserves precedence through omitted intermediate actions.
    Missing actions stay visible separately; they are not silently inserted.
    """
    dependency_layers(ids, relations)
    if len(ids) != len(set(ids)) or len(preference) != len(set(preference)):
        raise ValueError("Duplicate operation instances")
    if not set(preference).issubset(ids):
        raise ValueError("Candidate references an unknown operation")
    preferred = list(preference)
    positions = {key: index for index, key in enumerate(preferred)}
    previous: dict[str, set[str]] = {key: set() for key in preferred}
    successors: dict[str, set[str]] = {key: set() for key in preferred}
    for before, after in precedence_paths(ids, relations):
        if before in previous and after in previous:
            previous[after].add(before)
            successors[before].add(after)
    ready = [positions[key] for key in preferred if not previous[key]]
    heapq.heapify(ready)
    sequence = []
    while ready:
        key = preferred[heapq.heappop(ready)]
        sequence.append(key)
        for child in successors[key]:
            previous[child].remove(key)
            if not previous[child]:
                heapq.heappush(ready, positions[child])
    if len(sequence) != len(preferred):
        raise ValueError("Incomplete process ordering")
    return sequence


def compose_result(packet: dict, plan: dict, ordering: dict) -> dict:
    """Publish accepted relations and checked candidates without inventing steps."""
    if (
        plan["source_sha256"] != packet["source_sha256"]
        or packet.get("route_labels_used") is not False
    ):
        raise ValueError("Workflow mixes sources or route labels")
    value = (
        order_input(packet, plan)
        if plan["operations"]
        else {"operations": [], "records": packet["records"]}
    )
    if (
        plan.get("route_labels_used") is not False
        or ordering["source_sha256"] != packet["source_sha256"]
        or ordering.get("route_labels_used") is not False
    ):
        raise ValueError("Ordering mixes sources or route labels")
    if ordering["operations"] != value["operations"]:
        raise ValueError("Ordering changed the immutable operation instances")
    records = {row["id"]: row for row in value["records"]}
    relations = validate_order_relations(
        ordering["order_relations"], [row["id"] for row in value["operations"]], records
    )
    native = {row["id"]: row for row in packet["records"]}
    observations = {row["id"]: row for row in plan.get("observations", [])}

    def enrich(citation: dict) -> dict:
        key = citation["evidence_id"]
        if key in native:
            row = native[key]
            return {
                **citation,
                "source_kind": "native_measurement"
                if row["kind"] == "native_measurement"
                else "native_text",
                "source_handle": row["source_ref"].get("native_handle"),
                "cad_position": row.get("cad_position"),
                "pdf_position": row.get("pdf_position"),
                "scope_status": row.get("scope_status", "unassigned"),
                "literal_quote_verified": True,
            }
        row = observations[key]
        return {
            **citation,
            "source_kind": "drawing_observation",
            "source_handle": None,
            "cad_position": None,
            "pdf_bbox": row["pdf_bbox"],
            "image_id": row["image_id"],
            "page_number": 1,
            "scope_status": "model_observation_needs_review",
            "literal_quote_verified": False,
        }

    operations = deepcopy(plan["operations"])
    for row in [*operations, *relations]:
        row["evidence"] = [
            enrich({"evidence_id": c["evidence_id"], "quote": c["quote"]})
            for c in row["evidence"]
        ]
    ids = [row["id"] for row in operations]
    routes, seen = [], set()
    preferences = plan.get("routes", [])
    for proposed in preferences:
        sequence = ordered_candidate(ids, relations, proposed["operation_ids"])
        if tuple(sequence) in seen:
            continue
        seen.add(tuple(sequence))
        routes.append(
            {
                "rank": len(routes) + 1,
                "operation_ids": sequence,
                "explanation": "按已接受的前置关系组织；未约束处保留模型建议排列。",
                "sequence_origin": "model_preference"
                if sequence == proposed["operation_ids"]
                else "reordered_with_checked_precedence",
                "covers_all_proposed_operations": set(sequence) == set(ids),
                "is_complete_plm_route": False,
            }
        )
    merged = {"operations": operations, "order_relations": relations, "routes": routes}
    audit = order_audit(merged)
    names = {row["id"]: row["name"] for row in operations}
    relation_text = "；".join(
        f"{names[r['before']]}→{names[r['after']]}（{r['scope']}）" for r in relations
    )
    explanation = f"{plan['summary'].rstrip('。')}。提取{len(operations)}项加工动作。"
    if relation_text:
        explanation += f"顺序依据：{relation_text}。"
    if routes:
        explanation += (
            "候选工艺："
            + " → ".join(names[key] for key in routes[0]["operation_ids"])
            + "。"
        )
    unresolved = list(
        dict.fromkeys([*plan.get("unresolved", []), *ordering.get("unresolved", [])])
    )
    if audit["unplaced_operation_ids"]:
        missing_names = "、".join(names[key] for key in audit["unplaced_operation_ids"])
        explanation += f"尚未纳入候选的动作：{missing_names}。"
        unresolved.append(
            f"尚未纳入候选的加工动作：{missing_names}。请结合适用条件确定归入哪条候选，不自动混入替代方案。"
        )
    pending = deepcopy(ordering.get("pending_order_claims", []))
    for row in pending:
        row["relation"]["evidence"] = [enrich(c) for c in row["relation"]["evidence"]]
    return {
        "schema": "dwg_process_workflow.v1",
        "source_sha256": packet["source_sha256"],
        "needs_review": True,
        "summary": plan["summary"],
        "understanding_explanation": explanation,
        "input_records": len(packet["records"]),
        "input_summary": plan.get("input_summary", {}),
        "source_tables": packet.get("tables", []),
        "observations": plan.get("observations", []),
        "images_used": plan.get("images_used", False),
        "ocr_used": False,
        **merged,
        "order_audit": audit,
        "pending_order_claims": pending,
        "order_review": ordering.get("order_review", {"decisions": []}),
        "unresolved": unresolved,
        "route_labels_used": False,
        "accuracy_evaluated": False,
        "frozen_route_model_changed": False,
        "route_scope": "model_candidates_with_checked_precedence",
        "workflow_complete": True,
    }


def _saved_stage(
    root: Path, validate: Callable[[dict], bool]
) -> tuple[dict, Path] | None:
    for path in sorted(root.glob("*/result.json"), reverse=True):
        if not (path.parent / "checksums.json").is_file():
            continue  # A result without its commit manifest is an interrupted stage.
        verify_manifest(path.parent)
        result = read_json(path)
        if validate(result):
            return result, path.parent
    return None


def reuse_perception(case: Path, packet: dict, destination: Path) -> dict:
    """Import an explicitly selected, immutable upstream run without calling it new inference."""
    verify_manifest(case)
    original = read_json(case / "input.json")
    plan = read_json(case / "result.json")
    if fingerprint(original) != fingerprint(packet) or plan.get(
        "input_sha256"
    ) != fingerprint(packet):
        raise ValueError(
            "Saved perception does not have exactly the current engineering inputs"
        )
    order_input(packet, plan)
    if destination.exists():
        verify_manifest(destination)
        if read_json(destination / "result.json") != plan:
            raise ValueError("Imported perception result changed")
        return plan
    destination.mkdir(parents=True)
    write_new_json(destination / "result.json", plan)
    write_new_json(
        destination / "reuse_receipt.json",
        {
            "upstream": str(case.resolve()),
            "input_sha256": fingerprint(packet),
            "upstream_checksums_sha256": digest(case / "checksums.json"),
            "upstream_result_sha256": digest(case / "result.json"),
            "upstream_recipe_sha256": plan.get("recipe_sha256"),
            "new_perception_inference": False,
        },
    )
    write_new_json(
        destination / "checksums.json",
        {p.name: digest(p) for p in destination.iterdir() if p.is_file()},
    )
    return plan


def infer_workflow(
    packet: dict,
    output: Path,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    progress: Callable[[str], None] | None = None,
    verify_source: Callable[[], None] | None = None,
    perception_case: Path | None = None,
    perception_timeout_seconds: int = 600,
    separate_ordering: bool = False,
) -> dict:
    """Resume verified stages after interruption; never mark a partial task ready."""
    started = time.monotonic()
    implementation = recipe(
        endpoint,
        model,
        perception_timeout_seconds=perception_timeout_seconds,
        separate_ordering=separate_ordering,
    )
    identity = {
        "input_sha256": fingerprint(packet),
        "recipe_sha256": fingerprint(implementation),
        "source_sha256": packet["source_sha256"],
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_file = output / "workflow_identity.json"
    if identity_file.exists():
        if read_json(identity_file) != identity:
            raise ValueError("Workflow source or implementation changed")
    else:
        write_new_json(identity_file, identity)
    if not (output / "input.json").exists():
        write_new_json(output / "input.json", packet)
    elif fingerprint(read_json(output / "input.json")) != identity["input_sha256"]:
        raise ValueError("Saved workflow input changed")

    def check() -> None:
        if verify_source:
            verify_source()
        verify_drawing(packet["drawing"])
        if (
            recipe(
                endpoint,
                model,
                perception_timeout_seconds=perception_timeout_seconds,
                separate_ordering=separate_ordering,
            )
            != implementation
            or fingerprint(packet) != identity["input_sha256"]
        ):
            raise ValueError("Workflow identity changed")

    check()
    if (output / "result.json").is_file():
        if (output / "checksums.json").is_file():
            verify_manifest(output)
            return read_json(output / "result.json")
        # Preserve the uncommitted payload and rebuild from verified stages.
        (output / "result.json").rename(
            output / f"uncommitted_result_{uuid4().hex}.json"
        )
    expected_perception = fingerprint(
        perception_recipe(
            endpoint,
            model,
            request_timeout_seconds=perception_timeout_seconds,
            separate_ordering=separate_ordering,
        )
    )
    saved = _saved_stage(
        output / "perception",
        lambda r: (
            r.get("input_sha256") == identity["input_sha256"]
            and r.get("recipe_sha256") == expected_perception
        ),
    )
    perception_reused = saved is not None
    if perception_case is not None:
        perception_path = output / "perception" / "imported"
        plan = reuse_perception(perception_case, packet, perception_path)
        perception_reused = True
    elif saved:
        plan, perception_path = saved
    else:
        perception_path = output / "perception" / uuid4().hex
        plan = infer_plan(
            packet,
            perception_path,
            endpoint=endpoint,
            model=model,
            progress=progress,
            verify_source=check,
            request_timeout_seconds=perception_timeout_seconds,
            separate_ordering=separate_ordering,
        )
    check()
    if progress:
        progress("ordering")
    saved = _saved_stage(
        output / "order",
        lambda r: (
            r.get("source_sha256") == packet["source_sha256"]
            and r.get("operations") == order_input(packet, plan)["operations"]
        ),
    )
    order_reused = saved is not None
    if saved:
        ordering, order_path = saved
    elif plan["operations"]:
        order_path = output / "order" / uuid4().hex
        ordering = discover_order(
            packet,
            plan,
            order_path,
            endpoint=endpoint,
            model=model,
            progress=progress,
            **implementation["order"],
        )
    else:
        order_path = None
        ordering = {
            "source_sha256": packet["source_sha256"],
            "operations": [],
            "order_relations": [],
            "route_labels_used": False,
        }
    check()
    if progress:
        progress("assembling")
    result = compose_result(packet, plan, ordering)
    if packet.get("drawing") and result["operations"]:
        if progress:
            progress("assembling")
        decisions_hash = fingerprint(result["operations"])
        saved = _saved_stage(
            output / "targets",
            lambda r: (
                r.get("input_sha256") == identity["input_sha256"]
                and r.get("decisions_sha256") == decisions_hash
                and r.get("implementation_sha256")
                == implementation["implementation"]["target_grounding.py"]
            ),
        )
        if saved:
            grounding, grounding_path = saved
        else:
            grounding_path = output / "targets" / uuid4().hex
            grounding = ground_process_targets(
                packet, result, grounding_path, endpoint=endpoint, model=model
            )
        by_operation = {row["operation_id"]: row for row in grounding["targets"]}
        for operation in result["operations"]:
            target = by_operation[operation["id"]]
            operation["target_locations"] = target["locations"]
            operation["target_location_reason"] = target["reason"]
        result["target_grounding"] = {
            "status": grounding["status"],
            "seconds": grounding["seconds"],
            "result": str((grounding_path / "result.json").relative_to(output)),
            "reused": saved is not None,
        }
    check()
    result.update(identity)
    result["seconds"] = round(time.monotonic() - started, 3)
    result["timing_seconds"] = {
        "perception_recorded": plan.get("timing_seconds", {}).get(
            "total_from_native_ready", plan["seconds"]
        ),
        "order_recorded": ordering.get("seconds", 0),
        "grounding_recorded": result.get("target_grounding", {}).get("seconds", 0),
        "this_run": result["seconds"],
    }
    result["stages"] = {
        "perception": {
            "result": str((perception_path / "result.json").relative_to(output)),
            "reused": perception_reused,
        },
        "order": {
            "result": str((order_path / "result.json").relative_to(output))
            if order_path
            else None,
            "reused": order_reused,
        },
    }
    check()
    write_new_json(output / "result.json", result)
    write_new_json(
        output / "checksums.json",
        {
            str(p.relative_to(output)): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p != output / "checksums.json"
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Native input.json with a verified same-DWG PDF",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--perception-case",
        type=Path,
        help="Explicitly reuse a verified run with exactly the same native and drawing inputs",
    )
    args = parser.parse_args()
    result = infer_workflow(
        read_json(args.input),
        args.output,
        endpoint=args.endpoint,
        model=args.model,
        perception_case=args.perception_case,
        progress=lambda phase: print(phase, flush=True),
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "operations": len(result["operations"]),
                "routes": len(result["routes"]),
                "seconds": result["seconds"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
