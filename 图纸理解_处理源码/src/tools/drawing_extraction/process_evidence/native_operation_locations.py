"""Attach finite DWG contours to drawing actions with native arrow evidence."""

from __future__ import annotations

from collections import defaultdict
import math
import re

import fitz

from ..native_dwg_understanding.geometry import TOLERANCE, GeometryIndex, bbox
from .plan_inputs import verify_drawing

NATIVE_COORDINATES = "native_modelspace_or_verified_INSERT_affine"


def _arrow_contact(symbol: dict, geometry: GeometryIndex) -> tuple[list, list[dict]] | None:
    if symbol.get("matrix_commands_present"):
        return None
    paths = symbol["paths"]
    endpoints = []
    for path in paths:
        if path["native_kind"] != "POLYLINE":
            continue
        points = list(dict.fromkeys(map(tuple, path["points"])))
        if len(points) == 2 and math.dist(*points) > TOLERANCE:
            endpoints.extend(points)
    contacts = []
    for path in paths:
        if path["native_kind"] != "POLYGON" or not path.get("filled") or len(path["points"]) != 3:
            continue
        matches = [point for point in path["points"] for end in endpoints if math.dist(point, end) <= TOLERANCE]
        if len(matches) != 1:
            continue
        point = matches[0]
        rows = [geometry.rows[identifier] for identifier in geometry.contacts(point)]
        if rows:
            contacts.append((point, rows))
    return contacts[0] if len(contacts) == 1 else None


def _placement_group(rows: list[dict], anchor: dict, geometry: GeometryIndex) -> tuple[list[dict], dict] | None:
    """Find each instance of a rib block attached to the same circular contour."""
    if anchor["kind"] != "LINE" or not anchor.get("root_insert_source_pointer") or len(anchor["path_edge_ids"]) != 1:
        return None
    instances = defaultdict(list)
    for row in rows:
        if len(row["path_edge_ids"]) == 1:
            instances[row["path_edge_ids"][0]].append(row)
    handle = anchor["source_ref"]["native_handle"]
    contacts = defaultdict(list)
    for group in instances.values():
        if not any(r["kind"] == "LINE" and r["source_ref"]["native_handle"] == handle for r in group):
            continue
        caps = [r for r in group if r["kind"] == "ARC"]
        if len(group) != 4 or len(caps) != 1:
            continue
        # Both ends of the curved rib cap must touch one finite circular entity.
        arc = caps[0]["points"]
        circles = set(geometry.contacts(arc[0])) & set(geometry.contacts(arc[-1]))
        for identifier in sorted(circles):
            if geometry.rows[identifier]["kind"] == "CIRCLE":
                contacts[identifier].append(group)
    matches = [
        (groups, geometry.rows[identifier]) for identifier, groups in contacts.items()
        if len(groups) >= 2 and any(anchor in group for group in groups)
    ]
    if len(matches) != 1:
        return None
    groups, circle = matches[0]
    return [row for group in groups for row in group], circle


def _location(
    source: str, drawing: dict, identifier: str, kind: str, label: str,
    basis: str, rows: list[dict], symbol_handle: str, anchor: list | None = None,
) -> dict | None:
    matrix = drawing["cad_to_pdf_matrix44"]
    transform = fitz.Matrix(matrix[0], matrix[1], matrix[4], matrix[5], matrix[12], matrix[13])
    # Use a tight contact window for weld arrows. Long neighboring lines are
    # clipped to that window instead of outlining the entire component.
    if anchor:
        radius = min(drawing["page_points"]) * .011 / math.hypot(matrix[0], matrix[1])
        x, y = anchor[:2]
        box = [x - radius, y - radius, x + radius, y + radius]
    else:
        box = bbox([point for row in rows for point in row["points"]])
        if box is None:
            return None
    frame = fitz.Rect(0, 0, *drawing["page_points"])
    pdf = fitz.Rect(box) * transform
    if pdf.is_empty or not frame.contains(pdf):
        return None
    result = {
        "id": identifier,
        "source_sha256": source,
        "kind": kind,
        "label": label,
        "basis": basis,
        "part_ids": [],
        "pdf_bbox": list(pdf),
        "cad_bounds": box,
        "page_index": 0,
        "native_support_handles": sorted({row["source_ref"]["native_handle"] for row in rows}),
        "native_geometry_ids": sorted({row["id"] for row in rows}),
        "native_symbol_handle": symbol_handle,
        "status": "native_contour_link",
        "physical_owner_confirmed": False,
        "boundary_complete": False,
        "confidence": None,
    }
    if anchor:
        result["native_anchor_cad"] = anchor
    return result


def attach_native_locations(plan: dict, drawing: dict, facts: dict, scene: dict) -> dict:
    source = plan["source_sha256"]
    if any(value != source for value in (drawing["source_sha256"], facts["source"]["sha256"], scene["source"]["sha256"])):
        raise ValueError("Native operation geometry has a different source")
    if scene["base_semantic_sha256"] != facts["semantic_sha256"]:
        raise ValueError("Native operation geometry has a different reader result")
    response = facts["response_sha256"]
    rows = [
        row for row in facts["geometry"]
        if row.get("coordinate_status") == NATIVE_COORDINATES
        and row.get("line_role") == "contour_solid_declared"
        and row["kind"] in {"LINE", "ARC", "CIRCLE"}
        and row.get("source_ref", {}).get("source_sha256") == source
        and row["source_ref"].get("response_sha256") == response
    ]
    geometry = GeometryIndex(rows)
    welds, notes = [], []
    for symbol in scene["symbols"]:
        if symbol["category"] not in {"welding", "note"}:
            continue
        ref = symbol["source_ref"]
        if ref["source_sha256"] != source or ref["response_sha256"] != response:
            continue
        contact = _arrow_contact(symbol, geometry)
        if contact:
            (welds if symbol["category"] == "welding" else notes).append((symbol, *contact))
    if not welds and not notes:
        return plan
    verify_drawing(drawing)
    operations = []
    weld_locations = []
    for symbol, point, contours in welds:
        handle = symbol["source_ref"]["native_handle"]
        mark = symbol["raw_text"] or "无附加文字"
        location = _location(
            source, drawing, handle, "joint", f"焊接接点 · {mark} · {handle}",
            f"DWG 焊接标注 {handle} 的填充箭头与引线共用端点，端点落在有限实体轮廓上。",
            contours, handle, point,
        )
        if location:
            weld_locations.append(location)
    for operation in plan["operations"]:
        action, target, identifier = (operation[key] for key in ("name", "target", "id"))
        locations = (
            [{**item, "id": f"{identifier}:native:{item['id']}"} for item in weld_locations]
            if action == "组焊" or (action == "焊后磨平" and target == "焊缝")
            else []
        )
        if not locations:
            # A literal local instruction identifies the assembly region. The
            # pointer specifically hits a rib edge. The concentric ring is linked
            # only when both ends of each repeated cap touch that same entity.
            quotes = " ".join(row["quote"] for row in operation["evidence"] if row.get("literal_quote_verified"))
            for symbol, point, contours in notes:
                text = re.sub(r"[\s/,，;；]", "", symbol["raw_text"])
                if len(contours) != 1 or len(target) < 3 or target not in quotes:
                    continue
                if target not in text and not (target.endswith("加强筋") and target[:-3] in text and "加强筋" in text):
                    continue
                if not any(word in action for word in ("焊接", "抛光")):
                    continue
                handle = symbol["source_ref"]["native_handle"]
                group = _placement_group(rows, contours[0], geometry)
                if group and target.endswith("加强筋"):
                    ribs, _ = group
                    label = f"{target} · {len(ribs) // 4} 处轮廓"
                    basis = f"DWG 引出标注 {handle} 命中加强筋边线，同块实例的圆弧端点落在同一环形轮廓。"
                    location = _location(source, drawing, f"{identifier}:native:{handle}", "feature", label, basis, ribs, handle)
                elif group and target.endswith("法兰"):
                    _, ring = group
                    label = f"{target} · 与加强筋相接的环形边"
                    basis = f"DWG 引出标注 {handle} 和局部焊接说明指向同一组装区域，高亮与各加强筋圆弧端点相接的环形实体。"
                    location = _location(source, drawing, f"{identifier}:native:{handle}", "joint", label, basis, [ring], handle)
                else:
                    label = f"{target} · 原生引出连接区"
                    basis = f"DWG 引出标注 {handle} 提及该对象，引线端点落在连接区实体边线。"
                    location = _location(source, drawing, f"{identifier}:native:{handle}", "joint", label, basis, contours, handle, point)
                if location:
                    locations.append(location)
        operations.append({**operation, "target_locations": locations} if locations else operation)
    return {**plan, "operations": operations} if operations != plan["operations"] else plan
