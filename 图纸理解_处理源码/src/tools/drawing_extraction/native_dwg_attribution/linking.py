"""Join actual sequence cells, keeping remote headers and duplicate rows unresolved."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from research.dwg_route_validation_20261015.native_balloon_probe import finite_line_distance

__all__ = ["link_parts", "link_dimensions", "source_key"]


def source_key(ref: dict[str, Any]) -> tuple[str, str, str, str]:
    return tuple(ref[key] for key in ("source_sha256", "response_sha256", "json_pointer", "native_handle"))


def _sequence_cell(row: dict[str, Any]) -> tuple[str, int, str] | None:
    field = row.get("field_evidence", {}).get("sequence")
    if row.get("header_binding_status") == "connected_literal_header_candidate" and field:
        return field["value_cell_id"], field["column_ordinal"], "connected_header_cell"
    # A translated header is useful evidence, but remains a hypothesis. Do not
    # silently promote it into a confirmed field or assume sequence is column 0.
    candidates = row.get("continuation_evidence", [])
    roles = {tuple(header["header_roles"]) for header in candidates}
    if row.get("continuation_status") != "same_sheet_translated_column_signature_candidate" or len(roles) != 1:
        return None
    signature = next(iter(roles))
    if signature.count("sequence") != 1 or len(signature) != len(row["cells"]):
        return None
    column = signature.index("sequence")
    return row["cells"][column]["cell_id"], column, "translated_header_candidate"


def link_parts(bom: dict[str, Any], topology: dict[str, Any], catalog_links: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return every BOM row and callout, including ambiguous or unmatched ones."""
    parts, by_source, links = [], defaultdict(list), {row["bom_id"]: row for row in catalog_links}
    for row in bom.get("rows", []):
        cell = _sequence_cell(row)
        part = {"id": row["id"], "source_ref": row["source_ref"],
                "fields": row["fields"], "field_candidates": row.get("field_candidates", {}),
                "field_evidence": row.get("field_evidence", {}), "raw_cells": row["cells"],
                "field_binding_status": row["header_binding_status"],
                "continuation_evidence": row.get("continuation_evidence", []),
                "catalog_matches": links.get(row["id"], {}).get("catalog_matches", []),
                "callout_ids": [], "geometry_ids": [], "dimension_ids": [],
                "physical_owner_confirmed": False, "relation_confidence": None,
                "scope_status": "BOM_membership_only", "review_reasons": []}
        if not row["fields"]:
            part["review_reasons"].append("field_header_binding_unconfirmed")
        if any(item["deleted"] == "True" for item in part["catalog_matches"]):
            part["review_reasons"].append("catalog_contains_deleted_material_not_activated")
        parts.append(part)
        if cell:
            by_source[source_key(row["source_ref"])].append((row, part, cell))
    callouts = []
    by_id = {part["id"]: part for part in parts}
    for obj in topology.get("objects", []):
        found = {}
        for literal in (item for preview in obj["previews"] for item in preview["bom_literal_matches"]):
            for candidate in literal["candidates"]:
                for row, part, (cell_id, column, binding) in by_source[source_key(candidate["source_ref"])]:
                    fields = row["fields"] or row.get("field_candidates", {})
                    if fields.get("sequence") != literal["raw_text"]:
                        continue
                    cells = [cell for cell in candidate["cell_locations"] if cell["cell_id"] == cell_id
                             and any(band["band_id"] == row["band"]["band_id"]
                                     and band["column_ordinal"] == column for band in cell["row_bands"])]
                    if cells:
                        found[part["id"]] = {"part_id": part["id"], "binding": binding,
                                             "sequence_cells": cells, "literal_frame_ref": candidate["frame_ref"],
                                             "label_frame_ref": literal["label_frame_ref"]}
        ref = obj["source_ref"]
        identifier = "callout:" + ref["native_handle"]
        callout = {"id": identifier, "source_ref": ref, "previews": obj["previews"],
                   "part_candidates": list(found.values()), "geometry_ids": [],
                   "sequence_match_status": "unique" if len(found) == 1 else "ambiguous" if found else "unmatched",
                   "preview_to_WCS_validated": False, "physical_owner_confirmed": False,
                   "review_reasons": ["preview_coordinate_identity_unverified", "full_physical_boundary_unverified"]}
        callouts.append(callout)
        for part_id in found:
            part = by_id[part_id]
            part["callout_ids"].append(identifier)
            part["scope_status"] = "unique_sequence_cell_reference" if len(found) == 1 else "ambiguous_sequence_reference"
        if len(found) != 1:
            callout["review_reasons"].append("sequence_row_not_unique")
    for part in parts:
        if not part["callout_ids"]:
            part["review_reasons"].append("no_source_qualified_callout")
    return parts, callouts


def link_dimensions(parts: list[dict], callouts: list[dict], geometry: list[dict], native_ir: dict) -> list[dict]:
    """Keep native DIMASSOC facts separate from candidate BOM/physical scopes."""
    by_handle = defaultdict(list)
    for item in geometry:
        if item.get("path_edge_ids") or item.get("kind") == "HATCH_REGION":
            continue
        for ref in item.get("source_refs", [item.get("source_ref")]):
            if ref:
                by_handle[(source_key(ref), ref["native_handle"])].append(item["id"])
    calls = {item["id"]: item for item in callouts}
    geometry_by_id = {item["id"]: item for item in geometry}
    result = []
    for dimension in native_ir.get("dimensions", []):
        relations = [row for row in native_ir.get("relations", []) if row["dimension_id"] == dimension["id"]]
        targets = []
        for relation in relations:
            for witness in relation["source_refs"]:
                ref = witness["geometry_source"]
                # The accepted DIMASSOC relation is direct modelspace only.
                # Reused definition handles cannot identify an INSERT instance.
                targets.extend(by_handle.get((source_key(ref), ref["native_handle"]), []))
        targets = sorted(set(targets))
        candidates = []
        for part in parts:
            hits = sorted(set(part["geometry_ids"]) & set(targets))
            witnesses = []
            coordinates = dimension.get("cad_coordinates", {})
            if dimension.get("native_kind") in {"DIMENSION_LINEAR", "DIMENSION_ALIGNED"} and all(
                    key in coordinates for key in ("xline1_pt", "xline2_pt")):
                for slot in (1, 2):
                    point = coordinates[f"xline{slot}_pt"]
                    contacts = []
                    for identifier in part["geometry_ids"]:
                        item = geometry_by_id[identifier]
                        segments = item.get("native_straight_segments", [])
                        if item["kind"] == "LINE" and item.get("line_role") == "contour_solid_declared":
                            segments = [{"source_ref": item["source_ref"], "path_edge_ids": item["path_edge_ids"],
                                         "start": item["start"], "end": item["end"]}]
                        for segment in segments:
                            if source_key(segment["source_ref"])[:2] != source_key(dimension["source_ref"])[:2]:
                                continue
                            check = finite_line_distance(point, segment["start"], segment["end"])
                            if check["numeric_contact"] and 0 <= check["t_unclamped"] <= 1:
                                contour_scopes = [scope["contour_id"] for scope in part.get("closed_contour_candidates", [])
                                                  if identifier in scope["geometry_ids"]]
                                contacts.append({"geometry_id": identifier, "native_segment": segment, "check": check,
                                                 "scope_id": contour_scopes[0] if len(contour_scopes) == 1 else identifier})
                    witnesses.append({"slot": slot, "point": point,
                                      "source_pointer": dimension["source_ref"]["json_pointer"] + f"/xline{slot}_pt",
                                      "finite_boundary_contacts": contacts})
            common_scopes = (set.intersection(*({contact["scope_id"] for contact in row["finite_boundary_contacts"]}
                                               for row in witnesses)) if len(witnesses) == 2 else set())
            two_ends = bool(common_scopes)
            if two_ends:
                for witness in witnesses:
                    witness["finite_boundary_contacts"] = [contact for contact in witness["finite_boundary_contacts"]
                                                            if contact["scope_id"] in common_scopes]
            if hits or two_ends:
                candidate_ids = set(hits)
                if two_ends:
                    candidate_ids.update(contact["geometry_id"] for row in witnesses for contact in row["finite_boundary_contacts"])
                candidates.append({"part_id": part["id"], "geometry_ids": sorted(candidate_ids),
                                   "native_measurement_origin_witnesses": witnesses,
                                   "common_finite_scope_ids": sorted(common_scopes),
                                   "basis": "explicit_DIMASSOC" if hits else "both_native_origins_on_finite_scope_boundaries",
                                   "status": "conditional_callout_scope_not_physical_owner"})
                part["dimension_ids"].append(dimension["id"])
        targets = sorted(set(targets) | {identifier for item in candidates for identifier in item["geometry_ids"]})
        result.append({**dimension, "native_geometry_relations": relations, "geometry_ids": targets,
                       "part_candidates": candidates, "physical_scope_confirmed": False,
                       "hard_constraint_eligible": False,
                       "review_reasons": ["physical_owner_unverified"] if candidates else ["no_qualified_part_scope"],
                       "callout_ids": [key for key, item in calls.items() if set(item["geometry_ids"]) & set(targets)]})
    return result
