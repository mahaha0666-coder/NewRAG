"""Connect BOM, finite geometry, native dimension witnesses and literal requirements."""

from __future__ import annotations

from collections import Counter
import re
import time
from typing import Any

from ezdxf.tools.text import plain_mtext

from research.dwg_route_validation_20261015.native_ownership_cached import build_ownership_graph
from src.tools.drawing_extraction.native_dwg_batch.artifacts import fingerprint
from src.tools.drawing_extraction.native_dwg_batch.projection import table_input
from src.tools.drawing_extraction.requirement_semantics import parse_requirement_semantics

from .geometry import collect_geometry
from .linking import link_dimensions, link_parts

__all__ = ["build_attribution", "requirement_evidence"]

POLICY = {"route_labels_read": False, "route_models_changed": False, "ocr_used": False,
          "gpu_used": False, "nearest_geometry_assignment": False, "native_block_is_physical_part": False,
          "partial_contour_is_complete_part": False, "coordinate_frame_assumed_confirmed": False,
          "unknown_scope_is_hard_constraint": False, "model_accuracy_claimed": False}


def requirement_evidence(native_ir: dict, raw: dict) -> tuple[list[dict], list[dict]]:
    """Literal actions are evidence mentions, not a generated operation sequence."""
    requirements, views = [], []
    for text in native_ir.get("text_evidence", []):
        if text["kind"] not in {"TEXT", "MTEXT"}:
            continue
        original = text["value"]
        literal = plain_mtext(original) if text["kind"] == "MTEXT" else original
        obj = raw["OBJECTS"][int(text["source_ref"]["json_pointer"].rsplit("/", 1)[1])]
        point = obj.get("ins_pt") or obj.get("insertion_pt") or obj.get("insert")
        view = re.fullmatch(r"\s*([A-Z])[-－—]\1(?:\s*剖[视面]?)?\s*|\s*[A-Z]\s*向\s*", literal)
        if view:
            views.append({"id": "view-label:" + text["handle"], "label": literal,
                          "source_ref": text["source_ref"], "native_origin": point,
                          "status": "label_observed_view_extent_and_parent_cutting_line_unresolved",
                          "view_bbox": None, "physical_scope_confirmed": False})
        semantics = parse_requirement_semantics(literal)
        if semantics["parse_status"] != "structured_candidate":
            continue
        requirements.append({"id": "requirement:" + text["handle"], "raw_text": original,
                             "plain_text": literal, "source_ref": text["source_ref"],
                             "native_origin": point, "localization_status": "native_text_origin_only_no_clause_boxes",
                             "semantics": semantics, "scope_status": "source_drawing_not_entity_bound",
                             "current_part_or_child_or_assembly": "unresolved",
                             "confidence": None, "relation_confidence": None,
                             "hard_constraint_eligible": False,
                             "review_reasons": ["target_class_mentions_do_not_identify_physical_part",
                                                "factory_method_and_transfer_responsibility_not_provided"]})
    return requirements, views


def build_attribution(raw: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    source = evidence["source"]
    ir = evidence.get("native_ir", {})
    if ir and (ir["source_sha256"] != source["sha256"] or ir["response_sha256"] != evidence["response_sha256"]):
        raise ValueError("Native fields belong to a different source/reader response")
    tables = evidence.get("tables", {})
    if tables and any(tables["previews"][key] != value for key, value in (
            ("source_sha256", source["sha256"]), ("response_sha256", evidence["response_sha256"]))):
        raise ValueError("BOM/callout evidence belongs to a different source")
    parts, callouts = link_parts(tables.get("bom", {}), tables.get("callouts", {}), evidence.get("child_catalog_links", []))
    for row in [*parts, *callouts, *ir.get("dimensions", []), *ir.get("text_evidence", [])]:
        ref = row["source_ref"]
        if ref["source_sha256"] != source["sha256"] or ref["response_sha256"] != evidence["response_sha256"]:
            raise ValueError("A nested field/callout belongs to a different source")
    stages, geometry, geometry_audit = {}, [], {}
    started = time.monotonic()
    try:
        projected, _ = table_input(raw)
        graph = build_ownership_graph(projected, source_sha256=source["sha256"], response_sha256=evidence["response_sha256"])
        geometry, geometry_audit = collect_geometry(projected, graph, tables, parts, callouts)
        stages["native_geometry"] = {"status": "completed", "seconds": time.monotonic() - started}
    except (ValueError, KeyError, TypeError, ArithmeticError, IndexError) as error:
        stages["native_geometry"] = {"status": "failed", "seconds": time.monotonic() - started,
                                     "error_type": type(error).__name__, "error": str(error)[:500]}
    dimensions = link_dimensions(parts, callouts, geometry, ir)
    requirements, views = requirement_evidence(ir, raw)
    process_evidence = [{"id": "operation-evidence:" + row["id"], "source_evidence_id": row["id"],
                         "source_ref": row["source_ref"], "literal_clauses": row["semantics"]["clauses"],
                         "target_status": "unresolved_not_admitted_to_hard_route_constraints",
                         "hard_constraint_eligible": False, "historical_reference_used": False}
                        for row in requirements]
    unresolved = [{"kind": "part", "id": row["id"], "reasons": sorted(set(row["review_reasons"])),
                   "source_ref": row["source_ref"]} for row in parts]
    unresolved += [{"kind": "callout", "id": row["id"], "reasons": row["review_reasons"],
                    "source_ref": row["source_ref"]} for row in callouts]
    counts = {"bom_rows": len(parts), "callouts": len(callouts),
              "unique_sequence_cell_joins": sum(row["sequence_match_status"] == "unique" for row in callouts),
              "ambiguous_sequence_joins": sum(row["sequence_match_status"] == "ambiguous" for row in callouts),
              "unmatched_callouts": sum(row["sequence_match_status"] == "unmatched" for row in callouts),
              "parts_with_finite_geometry_candidates": sum(bool(row["geometry_ids"]) for row in parts),
              "geometry_occurrences": len(geometry), "native_dimensions": len(dimensions),
              "explicit_DIMASSOC_relations": sum(len(row["native_geometry_relations"]) for row in dimensions),
              "dimensions_with_conditional_part_candidates": sum(bool(row["part_candidates"]) for row in dimensions),
              "requirement_blocks": len(requirements), "view_label_candidates": len(views),
              "physical_owners_confirmed": 0, "hard_operation_constraints": 0}
    result = {"schema": "native_dwg_attribution.v1", "source": source,
              "response_sha256": evidence["response_sha256"], "input_recipe_sha256": evidence["recipe_sha256"],
              "policy": POLICY, "parts": parts, "callouts": callouts, "geometry": geometry,
              "dimensions": dimensions, "view_candidates": views, "requirements": requirements,
              "operation_evidence": process_evidence, "unresolved": unresolved,
              "native_symbols": evidence.get("proxy_observations", []),
              "geometry_audit": geometry_audit, "counts": counts,
              "review_reason_counts": dict(Counter(reason for row in unresolved for reason in row["reasons"])),
              "stages": stages}
    # Timings are execution metadata. Semantic identity is deterministic across
    # independent processes and intentionally excludes elapsed wall time.
    result["semantic_sha256"] = fingerprint({key: value for key, value in result.items() if key != "stages"})
    return result
