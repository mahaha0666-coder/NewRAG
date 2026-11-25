"""Native storage references, PMI observations and source-bound table evidence."""

from __future__ import annotations

from collections import Counter
import time
from typing import Any

from research.dwg_pdf_visualization_20261031.child_structure import bind_bom_rows
from research.dwg_route_validation_20261015.native_ir import build_ir
from research.dwg_route_validation_20261015.native_title_fields import bind_fields
from research.dwg_route_validation_20261015.native_tdm_callouts import build_callout_topology
from research.dwg_route_validation_20261015.native_tdm_cells import build_tdm_cells
from research.dwg_route_validation_20261015.native_tdm_previews import build_tdm_previews
from research.dwg_route_validation_20261015.parser_probe import handle_value, hex_handle
from research.dwg_route_validation_20261015.title_proxy_probe import (
    CODEPAGE_ENCODINGS, decode_hex_preview, decode_preview,
)

from .catalog import continuation_candidates, link_rows
from .projection import table_input

__all__ = ["build_evidence", "storage_graph"]

POLICY = {"ocr_used": False, "gpu_used": False, "route_labels_read": False,
          "val_test_read": False, "physical_part_ownership_confirmed": False,
          "preview_coordinates_are_pdf_coordinates": False,
          "native_block_is_physical_part": False, "eligible_for_route_training": False}


def storage_graph(raw: dict[str, Any], source_hash: str, response_hash: str) -> dict[str, Any]:
    """Index finite source references once, without expanding block instances.

    This is deliberately a storage graph. Coordinate transforms, visibility and
    physical part attribution cannot be inferred from membership alone.
    """
    classes = {row["number"]: row["dxfname"] for row in raw.get("CLASSES", [])}
    index = {}
    for ordinal, obj in enumerate(raw["OBJECTS"]):
        handle = hex_handle(handle_value(obj.get("handle")))
        if handle is not None:
            if handle in index:
                raise ValueError("Ambiguous native handle identity")
            index[handle] = (ordinal, obj)
    nodes, edges, unresolved = {}, [], []

    def add_node(handle: str) -> dict:
        if handle not in nodes:
            ordinal, obj = index[handle]
            kind = classes.get(obj.get("type")) or obj.get("entity") or obj.get("object")
            # Binary payloads remain intact in the compressed native response.
            fields = {key: value for key, value in obj.items()
                      if key not in {"preview", "unknown_bits", "handle"}}
            nodes[handle] = {"native_handle": handle, "native_kind": kind,
                "native_fields": fields, "coordinate_status": "original_storage_coordinates_not_expanded_WCS",
                "source_ref": {"source_sha256": source_hash, "response_sha256": response_hash,
                               "native_handle": handle, "json_pointer": f"#/OBJECTS/{ordinal}"}}
        return nodes[handle]

    for handle, (ordinal, obj) in index.items():
        if "entity" not in obj and obj.get("object") not in {"BLOCK_HEADER", "LAYER", "GROUP"}:
            continue
        if obj.get("object") == "GROUP" and not obj.get("groups"):
            continue
        add_node(handle)
        refs = [(field, None, obj[field]) for field in ("ownerhandle", "layer", "block_header", "dimstyle")
                if field in obj]
        if obj.get("object") == "BLOCK_HEADER":
            refs.extend(("entities", position, ref) for position, ref in enumerate(obj.get("entities", [])))
        for field, position, ref in refs:
            target = hex_handle(handle_value(ref))
            pointer = f"#/OBJECTS/{ordinal}/{field}" + (f"/{position}" if position is not None else "")
            if target not in index:
                unresolved.append({"source_handle": handle, "field_pointer": pointer,
                                   "raw_reference": ref, "reason": "target_missing_or_reference_unresolved"})
                continue
            add_node(target)
            edges.append({"source_handle": handle, "target_handle": target, "field": field,
                          "field_pointer": pointer, "raw_reference": ref,
                          "status": "explicit_native_storage_reference_not_physical_owner"})
    return {"nodes": list(nodes.values()), "edges": edges, "unresolved_references": unresolved,
            "summary": {"stored_objects": len(raw["OBJECTS"]), "indexed_nodes": len(nodes),
                        "explicit_reference_edges": len(edges), "unresolved_references": len(unresolved),
                        "native_kind_counts": dict(Counter(node["native_kind"] for node in nodes.values()))},
            "policy": POLICY}


def proxy_observations(raw: dict[str, Any], source_hash: str, response_hash: str) -> list[dict]:
    classes = {row["number"]: row["dxfname"] for row in raw.get("CLASSES", [])}
    encoding = CODEPAGE_ENCODINGS.get(raw.get("FILEHEADER", {}).get("codepage"))
    result = []
    for ordinal, obj in enumerate(raw["OBJECTS"]):
        if not obj.get("preview_size") or classes.get(obj.get("type")) in {"TDMBOMITEM", "TDMLABEL"}:
            continue
        row = {"native_kind": classes.get(obj.get("type")),
               "source_ref": {"source_sha256": source_hash, "response_sha256": response_hash,
                              "native_handle": hex_handle(handle_value(obj.get("handle"))),
                              "json_pointer": f"#/OBJECTS/{ordinal}"},
               "status": "unresolved", "physical_scope": "unknown", "confidence": None}
        try:
            if encoding is None:
                raise ValueError("Unsupported declared codepage; bytes retained without replacement decoding")
            data = decode_hex_preview(obj["preview"], obj["preview_size"])
            row["decoded_preview"] = decode_preview(data, encoding=encoding, version=raw["FILEHEADER"]["version"])
            if row["native_kind"] == "TDMTITLE":
                row["closed_cell_fields"] = bind_fields(row["decoded_preview"])
            row["status"] = "native_preview_observation_not_confirmed_engineering_parameter"
        except Exception as error:  # Isolate one custom object; preserve the failure in the artifact.
            row["error_type"] = type(error).__name__
            row["error"] = str(error)[:500]
        result.append(row)
    return result


def build_evidence(raw: dict[str, Any], *, item_code: str, source_hash: str, response_hash: str,
                   catalog: dict, documents: dict) -> dict[str, Any]:
    stages, payload = {}, {"schema": "batch_native_dwg_evidence.v1", "policy": POLICY}
    for name, function in (
        ("native_ir", lambda: build_ir(raw, item_code=item_code, source_sha256=source_hash, response_sha256=response_hash)),
        ("storage_graph", lambda: storage_graph(raw, source_hash, response_hash)),
        ("proxy_observations", lambda: proxy_observations(raw, source_hash, response_hash)),
    ):
        started = time.monotonic()
        try:
            payload[name] = function()
            stages[name] = {"status": "completed", "seconds": time.monotonic() - started}
        except Exception as error:  # A failed stage cannot turn the whole file into a false success.
            stages[name] = {"status": "failed", "seconds": time.monotonic() - started,
                            "error_type": type(error).__name__, "error": str(error)[:500]}
    started = time.monotonic()
    classes = {row["number"]: row["dxfname"] for row in raw.get("CLASSES", [])}
    has_table = any(classes.get(obj.get("type")) in {"TDMBOMITEM", "TDMLABEL"} for obj in raw["OBJECTS"])
    links = []
    if has_table:
        try:
            projected, dictionary_receipt = table_input(raw)
            previews = build_tdm_previews(projected, source_sha256=source_hash, response_sha256=response_hash)
            cells = build_tdm_cells(previews)
            bom = bind_bom_rows(previews, cells)
            continuation_candidates(bom)
            links = link_rows(bom, catalog, documents)
            payload["tables"] = {"previews": previews, "cells": cells, "bom": bom,
                                 "native_dictionary_projection": dictionary_receipt,
                                 "callouts": build_callout_topology(previews, cells)}
            stages["tables"] = {"status": "completed", "seconds": time.monotonic() - started}
        except Exception as error:
            stages["tables"] = {"status": "failed", "seconds": time.monotonic() - started,
                                "error_type": type(error).__name__, "error": str(error)[:500]}
    else:
        stages["tables"] = {"status": "no_supported_native_BOM_class",
                            "other_table_forms_verified_absent": False, "seconds": time.monotonic() - started}
    payload["stages"] = stages
    payload["child_catalog_links"] = links
    return payload
