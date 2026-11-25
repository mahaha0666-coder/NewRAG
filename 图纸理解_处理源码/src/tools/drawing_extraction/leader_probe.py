"""Inspect OCR number / vector segment / BOM hypotheses; never assign part ownership."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont


def distance_to_box(point, box):
    return math.hypot(max(box[0] - point[0], 0, point[0] - box[2]),
                      max(box[1] - point[1], 0, point[1] - box[3]))


def overlap(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return area / max(1, min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])))


def point_to_segment_distance(point, segment):
    """Return distance to the finite segment and the nearest segment point."""
    start, end = segment["points"]
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if not length_squared:
        return math.dist(point, start), list(start)
    projection = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    projection = max(0.0, min(1.0, projection))
    nearest = [start[0] + projection * dx, start[1] + projection * dy]
    return math.dist(point, nearest), nearest


def extract_page_segments(drawings, transform) -> list[dict]:
    """Map open PDF line segments into rendered-page pixel coordinates."""
    segments = []
    for path_index, path in enumerate(drawings):
        if path.get("type") != "s" or path.get("closePath"):
            continue
        for item_index, item in enumerate(path["items"]):
            if item[0] != "l":
                continue
            points = [list(point * transform) for point in item[1:3]]
            segments.append({"path_index": path_index, "item_index": item_index,
                             "points": points, "length": math.dist(*points)})
    return segments


def extract_page_fill_marks(drawings, transform) -> list[list[float]]:
    """Pixel boxes of dark filled PDF paths, including leader endpoint dots."""
    marks = []
    for path in drawings:
        fill = path.get("fill")
        if fill is None or max(fill) > 0.25:
            continue
        rect = pymupdf.Rect(path["rect"]) * transform
        if rect.is_empty or rect.is_infinite:
            continue
        marks.append([rect.x0, rect.y0, rect.x1, rect.y1])
    return marks


def endpoint_fill_mark(point, height, marks):
    """Find a compact filled path at a leader tip; it does not prove part ownership."""
    eligible = (box for box in marks
                if 0 < box[2] - box[0] <= .25 * height
                and 0 < box[3] - box[1] <= .25 * height)
    closest = min(eligible, key=lambda box: distance_to_box(point, box), default=None)
    if closest is None or distance_to_box(point, closest) > max(1.0, .04 * height):
        return None
    return closest


def landing_support(segment, box, segments):
    """Return connected short horizontal strokes below a numeric annotation."""
    height = box[3] - box[1]
    near = min(segment["points"], key=lambda p: distance_to_box(p, box))
    support = []
    for stroke in segments:
        if (stroke["path_index"], stroke["item_index"]) == (segment["path_index"], segment["item_index"]):
            continue
        a, b = stroke["points"]
        tolerance = max(1, .04 * height)
        if not (.5 * height <= stroke["length"] <= 2.5 * height):
            continue
        if abs(a[1]-b[1]) > tolerance or min(math.dist(near, a), math.dist(near, b)) > tolerance:
            continue
        if not (box[3] - .1 * height <= (a[1]+b[1])/2 <= box[3] + .7 * height):
            continue
        if min(a[0], b[0]) <= (box[0]+box[2])/2 <= max(a[0], b[0]):
            support.append({"path_index": stroke["path_index"], "item_index": stroke["item_index"],
                            "points": stroke["points"]})
    return support


def strict_leader_matches(box, segments):
    """Find leaders supported at both the item label and target geometry."""
    return _leader_matches(box, segments, target_gap_factor=0.2)


def _leader_starts(box, segments):
    height = box[3] - box[1]
    for segment in segments:
        if segment["length"] < 2 * height:
            continue
        distances = [distance_to_box(point, box) for point in segment["points"]]
        near_index = 0 if distances[0] <= distances[1] else 1
        if distances[near_index] > 0.7 * height:
            continue
        landing = landing_support(segment, box, segments)
        if not landing:
            continue
        yield segment, distances[near_index], near_index, landing


def _leader_matches(box, segments, *, target_gap_factor: float, candidate_only: bool = False):
    height = box[3] - box[1]
    matches = []
    for segment, callout_gap, near_index, landing in _leader_starts(box, segments):
        far_endpoint = segment["points"][1 - near_index]
        target_support = []
        for stroke in segments:
            if (stroke["path_index"], stroke["item_index"]) == (
                    segment["path_index"], segment["item_index"]):
                continue
            if stroke["length"] < 0.5 * height:
                continue
            target_gap, nearest = point_to_segment_distance(far_endpoint, stroke)
            if target_gap <= max(2.0, target_gap_factor * height):
                target_support.append({"path_index": stroke["path_index"],
                                       "item_index": stroke["item_index"],
                                       "points": stroke["points"], "gap_pixels": target_gap,
                                       "nearest_point": nearest})
        if not target_support:
            continue
        match = {"path_index": segment["path_index"],
                 "item_index": segment["item_index"],
                 "points": segment["points"],
                 "gap_pixels": callout_gap,
                 "far_endpoint": far_endpoint,
                 "landing_support": landing,
                 "target_support": target_support}
        if candidate_only:
            match["candidate_only"] = True
        matches.append(match)
    return matches


def _forward_intersection(origin, direction, stroke):
    """Ray distance and point where a leader extension crosses a finite segment."""
    a, b = stroke["points"]
    delta = [b[0] - a[0], b[1] - a[1]]
    cross = direction[0] * delta[1] - direction[1] * delta[0]
    if abs(cross) < 0.5 * stroke["length"]:
        return None
    offset = [a[0] - origin[0], a[1] - origin[1]]
    distance = (offset[0] * delta[1] - offset[1] * delta[0]) / cross
    along = (offset[0] * direction[1] - offset[1] * direction[0]) / cross
    if not (distance > 0 and 0.01 <= along <= 0.99):
        return None
    return distance, [origin[0] + distance * direction[0],
                      origin[1] + distance * direction[1]]


def forward_contour_hypotheses(box, segments):
    """Review-only leaders whose forward extension crosses a contour pair.

    A pair reduces single dimension-line matches, but does not prove the
    crossed strokes belong to the BOM part; never promote these automatically.
    """
    height = box[3] - box[1]
    if height <= 0:
        return []
    matches = []
    for leader, gap, near_index, landing in _leader_starts(box, segments):
        far = leader["points"][1 - near_index]
        near = leader["points"][near_index]
        direction = [(far[0] - near[0]) / leader["length"],
                     (far[1] - near[1]) / leader["length"]]
        crossings = []
        for stroke in segments:
            if (stroke["path_index"], stroke["item_index"]) == (
                    leader["path_index"], leader["item_index"]):
                continue
            if stroke["length"] < 3 * height:
                continue
            hit = _forward_intersection(far, direction, stroke)
            if hit and hit[0] <= 1.65 * height:
                crossings.append((hit[0], hit[1], stroke))
        crossings.sort(key=lambda row: row[0])
        pairs = []
        for distance, point, target in crossings:
            if distance > 1.25 * height:
                break
            target_start, target_end = target["points"]
            target_vector = [target_end[0] - target_start[0],
                             target_end[1] - target_start[1]]
            for peer_distance, peer_point, peer in crossings:
                separation = peer_distance - distance
                if not 0.08 * height <= separation <= 0.4 * height:
                    continue
                peer_start, peer_end = peer["points"]
                peer_vector = [peer_end[0] - peer_start[0],
                               peer_end[1] - peer_start[1]]
                sine = abs(target_vector[0] * peer_vector[1]
                           - target_vector[1] * peer_vector[0]) / (
                               target["length"] * peer["length"])
                if sine > 0.14:
                    continue
                pairs.append((distance, point, target, peer_distance, peer_point, peer))
                break
        if not pairs or pairs[-1][0] - pairs[0][0] > 0.4 * height:
            continue
        distance, point, target, peer_distance, peer_point, peer = pairs[0]
        matches.append({
            "path_index": leader["path_index"], "item_index": leader["item_index"],
            "points": leader["points"], "gap_pixels": gap,
            "far_endpoint": far, "landing_support": landing,
            "target_support": {"path_index": target["path_index"],
                               "item_index": target["item_index"],
                               "points": target["points"], "intersection": point,
                               "forward_gap_pixels": distance},
            "parallel_support": {"path_index": peer["path_index"],
                                 "item_index": peer["item_index"],
                                 "points": peer["points"], "intersection": peer_point,
                                 "forward_gap_pixels": peer_distance},
            "candidate_only": True,
        })
    return matches


def leader_hypothesis_matches(box, segments, target_gap_factor: float = 0.5):
    """Find review-only leader hypotheses with a wider target gap.

    This deliberately does not participate in BOM binding. It is useful when
    raster/OCR coordinates and PDF vector coordinates differ by a few pixels,
    but a wider gap also admits dimension lines and nearby contours.
    """
    if not 0.2 <= target_gap_factor <= 0.8:
        raise ValueError("target_gap_factor must be between 0.2 and 0.8")
    return _leader_matches(box, segments, target_gap_factor=target_gap_factor,
                           candidate_only=True)


def probe(source, pdf_path, output, require_landing=False):
    doc = json.loads(source.read_text())
    if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != doc["source_sha256"]:
        raise ValueError("PDF does not match extraction source")
    output.mkdir(parents=True, exist_ok=False)
    report = {"source_json_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "source_pdf_sha256": doc["source_sha256"], "coordinate_space": "rendered_page_pixels",
              "method": "OCR polygons + nearby original vector segment endpoints",
              "limitations": ["Not a trained detector", "No arrowhead or multi-segment leader validation",
                              "Number equality is not confirmed BOM ownership", "No accuracy ground truth"],
              "parameters": {"endpoint_gap_text_height": 0.6, "minimum_length_text_height": 2,
                             "duplicate_overlap": 0.5, "require_landing": require_landing}, "pages": []}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    with pymupdf.open(pdf_path) as pdf:
        for page in doc["pages"]:
            number = page["page_number"]
            transform = pymupdf.Matrix(*page["pixel_to_pdf_matrix"])
            if transform.invert():
                raise ValueError("Singular coordinate transform")
            paths = pdf[number-1].get_drawings()
            segments = extract_page_segments(paths, transform)
            tables = [loc["bbox"] for b in doc["bom"] for loc in b["locations"]
                      if loc["page_number"] == number and loc.get("bbox")]
            bom = {}
            for entry in doc["bom"]:
                bom.setdefault(str(entry["attributes"]["item_number"]), []).append(entry["id"])
            candidates = []
            for entity in doc["raw_ocr"]:
                text = (entity.get("normalized_text") or "").strip()
                if text not in bom or entity["attributes"].get("localization") != "text_polygon":
                    continue
                for loc in entity["locations"]:
                    box = loc.get("bbox")
                    if loc["page_number"] != number or not box:
                        continue
                    if any(overlap(box, table) > 0.5 for table in tables):
                        continue
                    duplicate = next((c for c in candidates if c["text"] == text and overlap(c["bbox"], box) > 0.5), None)
                    if duplicate:
                        duplicate["ocr_ids"].append(entity["id"])
                        continue
                    candidates.append({"text": text, "bbox": box, "ocr_ids": [entity["id"]]})
            image_path = Path(page["image_path"])
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != page["image_sha256"]:
                raise ValueError("Rendered image hash mismatch")
            with Image.open(image_path) as im:
                canvas = im.convert("RGB")
            draw = ImageDraw.Draw(canvas)
            for i, candidate in enumerate(candidates):
                box = candidate["bbox"]
                height = box[3] - box[1]
                nearby = []
                for segment in segments:
                    if segment["length"] < height * 2:
                        continue
                    distances = [distance_to_box(p, box) for p in segment["points"]]
                    if min(distances) <= height * 0.6:
                        near = 0 if distances[0] <= distances[1] else 1
                        nearby.append({**segment, "gap_pixels": min(distances),
                                       "far_endpoint": segment["points"][1-near]})
                nearby.sort(key=lambda s: s["gap_pixels"])
                rejected = []
                if require_landing:
                    for segment in nearby:
                        segment["landing_support"] = landing_support(segment, box, segments)
                    rejected = [s for s in nearby if not s["landing_support"]]
                    nearby = [s for s in nearby if s["landing_support"]]
                candidate.update({"label": f"N{i+1:02d}", "nearby_segments": nearby,
                                  "bom_candidate_ids": bom[candidate["text"]] if nearby else [],
                                  "status": "geometric_hypothesis" if nearby else "no_leader_evidence",
                                  "part_id": None, "confidence": None, "needs_review": True})
                if require_landing:
                    candidate["rejected_without_landing"] = rejected
                if not nearby:
                    continue
                draw.rectangle(box, outline="#0072B2", width=6)
                draw.text((box[2]+5, box[1]), candidate["label"], font=font, fill="#0072B2")
                for segment in nearby[:3]:
                    draw.line([tuple(p) for p in segment["points"]], fill="#008254", width=5)
                    x, y = segment["far_endpoint"]
                    draw.ellipse((x-9,y-9,x+9,y+9), outline="#BB3377", width=4)
            canvas.save(output / f"page_{number:04d}_leaders.png")
            canvas.thumbnail((1800, 1350))
            canvas.save(output / f"page_{number:04d}_leaders_preview.png")
            report["pages"].append({"page_number": number, "vector_paths": len(paths),
                                    "line_segments": len(segments), "candidates": candidates})
    (output / "leader_candidates.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


__all__ = [
    "distance_to_box", "endpoint_fill_mark", "extract_page_fill_marks",
    "extract_page_segments", "forward_contour_hypotheses", "landing_support",
    "leader_hypothesis_matches", "overlap", "point_to_segment_distance", "probe",
    "strict_leader_matches",
]
