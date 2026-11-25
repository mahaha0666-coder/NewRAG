"""Display the pinned original PDF, with independently checked CAD coordinates.

Registration is presentation metadata, not proof of revision or part ownership.
Only a unique frame fit supported by distributed native lines enables overlays.
"""

from __future__ import annotations

import base64
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pymupdf

from .page_views import page_view_boxes, recover_small_sheet

MAX_PDF_BYTES = 32 * 1024 * 1024
MAX_SVG_BYTES = 24 * 1024 * 1024
MAX_PAGES = 8
Point = tuple[float, float]
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Segment:
    start: Point
    end: Point


def _point(value: list[float]) -> Point | None:
    if (len(value) < 2 or not all(math.isfinite(v) for v in value)
            or (len(value) > 2 and abs(value[2]) > 1e-5)):
        return None
    return (float(value[0]), float(value[1]))


def _native_evidence(facts: dict[str, Any]) -> bytes:
    frames: set[Box] = set()
    for symbol in facts.get("native_symbols", []):
        if symbol.get("native_kind") != "TDMFRAME":
            continue
        for entity in symbol.get("decoded_preview", {}).get("entities", []):
            if entity.get("type") != "POLYLINE" or entity.get("matrix_stack"):
                continue
            points = [_point(v["location"]) for v in entity.get("vertices", [])]
            if len(points) != 5 or any(p is None for p in points):
                continue
            if math.dist(points[0], points[-1]) > 1e-5:
                continue
            if any(abs(a[0] - b[0]) > 1e-5 and abs(a[1] - b[1]) > 1e-5
                   for a, b in zip(points, points[1:])):
                continue
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            box = (min(xs), min(ys), max(xs), max(ys))
            if box[2] > box[0] and box[3] > box[1]:
                frames.add(box)
    ordered = sorted(frames, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    frame = ordered[0] if ordered else None
    if len(ordered) > 1:
        first, second = ordered[:2]
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
        if abs(area(first) - area(second)) < area(first) * 1e-6:
            frame = None
    lines = set()
    for row in facts.get("geometry", []):
        if row.get("kind") != "LINE" or row.get("line_role") == "centerline_declared":
            continue
        start, end = _point(row["start"]), _point(row["end"])
        if start is not None and end is not None and math.dist(start, end) > 5:
            lines.add(tuple(sorted((start, end))))
    return json.dumps({"frame": frame, "lines": sorted(lines)}, allow_nan=False).encode()


def _pdf_segments(page: Any) -> list[Segment]:
    result = []
    rotation = page.rotation_matrix
    for path in page.get_drawings():
        if path.get("type") not in {"s", "fs"}:
            continue
        for item in path["items"]:
            if item[0] == "l":
                pairs = [(item[1], item[2])]
            elif item[0] in {"re", "qu"}:
                shape = item[1]
                points = ([shape.tl, shape.tr, shape.br, shape.bl] if item[0] == "re"
                          else [shape.ul, shape.ur, shape.lr, shape.ll])
                pairs = zip(points, points[1:] + points[:1])
            else:
                continue
            for a, b in pairs:
                a, b = a * rotation, b * rotation
                result.append(Segment(tuple(a), tuple(b)))
                if len(result) > 150_000:
                    raise ValueError("PDF vector budget exceeded")
    return result


def _merged_axes(segments: list[Segment], tolerance: float = .1) -> list[list[tuple[float, float, float]]]:
    axes = [defaultdict(list), defaultdict(list)]
    for seg in segments:
        (x1, y1), (x2, y2) = seg.start, seg.end
        if abs(y2 - y1) <= tolerance:
            axes[0][round((y1 + y2) / (2 * tolerance))].append(((y1 + y2) / 2, min(x1, x2), max(x1, x2)))
        elif abs(x2 - x1) <= tolerance:
            axes[1][round((x1 + x2) / (2 * tolerance))].append(((x1 + x2) / 2, min(y1, y2), max(y1, y2)))
    result = []
    for axis in axes:
        merged = []
        for rows in axis.values():
            rows.sort(key=lambda row: row[1])
            coordinate, low, high = rows[0]
            for next_coordinate, next_low, next_high in rows[1:]:
                if next_low <= high + tolerance:
                    high = max(high, next_high)
                else:
                    merged.append((coordinate, low, high))
                    coordinate, low, high = next_coordinate, next_low, next_high
            merged.append((coordinate, low, high))
        result.append(sorted(merged))
    return result


def _pdf_frames(segments: list[Segment], width: float, height: float, aspect: float) -> list[Box]:
    horizontal, vertical = _merged_axes(segments)
    horizontal = [r for r in horizontal if r[2] - r[1] >= width * .65]
    vertical = [r for r in vertical if r[2] - r[1] >= height * .60]
    if max(len(horizontal), len(vertical)) > 128:
        raise ValueError("PDF frame search budget exceeded")
    found = {}
    for index, (bottom, low, high) in enumerate(horizontal):
        for top, next_low, next_high in horizontal[index + 1:]:
            h = top - bottom
            left_limit, right_limit = max(low, next_low), min(high, next_high)
            if h < height * .60 or right_limit - left_limit < width * .65:
                continue
            xs = sorted({x for x, y1, y2 in vertical if y1 <= bottom + .1
                         and y2 >= top - .1 and left_limit - .1 <= x <= right_limit + .1})
            for left_index, left in enumerate(xs):
                for right in xs[left_index + 1:]:
                    if right - left < width * .65 or abs((right - left) / h / aspect - 1) > .003:
                        continue
                    box = (left, bottom, right, top)
                    found[tuple(round(v / .1) for v in box)] = box
    return list(found.values())


def _registration(frame: list[float] | None, native: list, page: Any) -> dict[str, Any]:
    if frame is None or len(native) < 8:
        return {"accepted": False, "reason": "DWG 图框或独立线条证据不足"}
    segments = _pdf_segments(page)
    grid = defaultdict(list)
    tolerance = .8
    for seg in segments:
        for point in (seg.start, seg.end):
            grid[(math.floor(point[0] / tolerance), math.floor(point[1] / tolerance))].append(seg)

    def residual(start: Point, end: Point) -> float:
        x, y = math.floor(start[0] / tolerance), math.floor(start[1] / tolerance)
        return min((min(max(math.dist(start, s.start), math.dist(end, s.end)),
                        max(math.dist(start, s.end), math.dist(end, s.start)))
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    for s in grid.get((x + dx, y + dy), ())), default=math.inf)

    width, height = frame[2] - frame[0], frame[3] - frame[1]
    fits = []
    for box in _pdf_frames(segments, page.rect.width, page.rect.height, width / height):
        sx, sy = (box[2] - box[0]) / width, (box[3] - box[1]) / height
        scale = (sx + sy) / 2
        if min(sx, sy) <= 0 or abs(sx - sy) / scale > .003:
            continue
        tx = (box[0] + box[2] - scale * (frame[0] + frame[2])) / 2
        ty = (box[1] + box[3] + scale * (frame[1] + frame[3])) / 2
        convert = lambda p: (scale * p[0] + tx, -scale * p[1] + ty)
        corners = [(frame[0], frame[1]), (frame[0], frame[3]), (frame[2], frame[1]), (frame[2], frame[3])]
        targets = [(box[0], box[3]), (box[0], box[1]), (box[2], box[3]), (box[2], box[1])]
        frame_error = max(math.dist(convert(p), q) for p, q in zip(corners, targets))
        if frame_error > .6:
            continue
        matched = [(a, b) for a, b in native if residual(convert(a), convert(b)) <= tolerance]
        if len(matched) < 8 or len(matched) / len(native) < .8:
            continue
        points = [p for seg in matched for p in seg]
        spread_x = (max(p[0] for p in points) - min(p[0] for p in points)) / width
        spread_y = (max(p[1] for p in points) - min(p[1] for p in points)) / height
        if min(spread_x, spread_y) < .2:
            continue
        fits.append({"accepted": True, "cad_to_pdf": [scale, 0, 0, -scale, tx, ty],
                     "stage_bounds": [-tx / scale, -ty / scale, page.rect.width / scale, page.rect.height / scale],
                     "matched_lines": len(matched), "checked_lines": len(native),
                     "frame_residual_pt": frame_error, "line_tolerance_pt": tolerance})
    if len(fits) != 1:
        return {"accepted": False, "reason": "坐标配准证据不足" if not fits else "存在多个可能的坐标配准"}
    return fits[0]


@lru_cache(maxsize=2)
def _render_pdf(content: bytes, evidence: bytes) -> dict[str, Any]:
    native = json.loads(evidence)
    pages = []
    used_bytes = 0
    with pymupdf.open(stream=content, filetype="pdf") as document:
        if document.needs_pass or not 0 < len(document) <= MAX_PAGES:
            return {"status": "unavailable", "reason": "PDF 加密、无有效页面或超过当前预览页数限制"}
        for index, page in enumerate(document):
            svg = page.get_svg_image(text_as_path=True).encode()
            used_bytes += len(svg)
            if used_bytes > MAX_SVG_BYTES:
                return {"status": "unavailable", "reason": "PDF 预览内容超过当前显示容量"}
            try:
                registration = _registration(native["frame"], native["lines"], page)
            except ValueError:
                registration = {"accepted": False, "reason": "PDF 几何过于复杂，暂未完成配准"}
            pages.append({"number": index + 1, "width": page.rect.width, "height": page.rect.height,
                          "rotation": page.rotation, "registration": registration, "view_boxes": page_view_boxes(page),
                          "image_url": "data:image/svg+xml;base64," + base64.b64encode(svg).decode("ascii")})
    return {"status": "ready", "pages": pages}


def render_pdf_evidence(content: bytes, facts: dict[str, Any]) -> dict[str, Any]:
    """Render a bounded PDF against this DWG; registration does not prove revision."""
    if len(content) > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds the preview size limit")
    return _render_pdf(content, _native_evidence(facts))


@lru_cache(maxsize=2)
def _render_generated(content: bytes, placement: tuple[float, ...]) -> dict[str, Any]:
    sx, sy, tx, ty, width, height = placement
    with pymupdf.open(stream=content, filetype="pdf") as document:
        if document.needs_pass or len(document) != 1:
            raise ValueError("Expected a single, unencrypted DWG-derived PDF sheet")
        page = document[0]
        if (page.rotation or abs(page.rect.width - width) > .01
                or abs(page.rect.height - height) > .01):
            raise ValueError("Generated PDF page differs from its placement record")
        svg = page.get_svg_image(text_as_path=True).encode()
        if len(svg) > MAX_SVG_BYTES:
            raise ValueError("Generated PDF exceeds the preview capacity")
        boxes = page_view_boxes(page)
        reframe = recover_small_sheet(page, svg) if not boxes else None
        if reframe:
            svg, boxes, (x, y, width, height, zoom) = reframe
            sx, sy = sx * zoom, sy * zoom
            tx, ty = (tx - x) * zoom, (ty - y) * zoom
        registration = {"accepted": True, "basis": "same_source_renderer_transform",
                        "cad_to_pdf": [sx, 0, 0, sy, tx, ty],
                        "stage_bounds": [-tx / sx, ty / sy, width / sx, -height / sy],
                        "physical_ownership_confirmed": False}
        return {"status": "ready", "pages": [{"number": 1, "width": width, "height": height,
                "rotation": 0, "registration": registration, "view_boxes": boxes,
                "display_reframed": bool(reframe),
                "image_url": "data:image/svg+xml;base64," + base64.b64encode(svg).decode("ascii")} ]}


def render_generated_pdf(content: bytes, manifest: dict[str, Any]) -> dict[str, Any]:
    """Use the actual renderer matrix, without estimating a companion-PDF fit."""
    if len(content) > MAX_PDF_BYTES:
        raise ValueError("Generated PDF exceeds the preview size limit")
    matrix = manifest["cad_to_pdf_matrix44"]
    size = manifest["page_points"]
    if (len(matrix) != 16 or len(size) != 2
            or not all(math.isfinite(value) for value in (*matrix, *size))
            or any(abs(matrix[index]) > 1e-10 for index in (1, 2, 3, 4, 6, 7, 8, 9, 11))
            or matrix[15] != 1 or matrix[0] <= 0 or matrix[5] >= 0
            or abs(matrix[0] + matrix[5]) > matrix[0] * 1e-6 or min(size) <= 0):
        raise ValueError("Unsupported generated PDF placement transform")
    return _render_generated(content, (matrix[0], matrix[5], matrix[12], matrix[13], *size))


def load_pdf_surface(project_root: Path, stem: str, facts: dict[str, Any]) -> dict[str, Any]:
    """Recheck original bytes on every open; cache only content-addressed rendering."""
    relative = f"data/raw/cad_current/pdf/{stem}.pdf"
    path = project_root / relative
    if not path.is_file():
        return {"status": "unavailable", "reason": "当前资料中未找到对应 PDF，显示 DWG 结构"}
    try:
        manifest = json.loads((project_root / "data/raw/cad_current/manifest.json").read_bytes())
        matches = [row for row in manifest["payloads"] if row["path"] == relative]
        if len(matches) != 1 or path.stat().st_size > MAX_PDF_BYTES:
            return {"status": "unavailable", "reason": "对应 PDF 未通过当前资料清单校验"}
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if len(content) != matches[0]["bytes"] or digest != matches[0]["sha256"]:
            return {"status": "unavailable", "reason": "对应 PDF 与当前资料清单不一致，请核对原文件"}
        result = _render_pdf(content, _native_evidence(facts))
        return {**result, "filename": f"{stem}.pdf", "sha256": digest,
                "revision_correspondence_confirmed": False}
    except (OSError, ValueError, KeyError, TypeError, pymupdf.FileDataError) as error:
        # Failure is explicit to the viewer; a changed PDF is never served from cache.
        return {"status": "unavailable", "reason": "对应 PDF 暂时无法读取，显示 DWG 结构",
                "failure_kind": type(error).__name__}
