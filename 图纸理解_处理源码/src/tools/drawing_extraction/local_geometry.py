"""Local native-PDF stroke groups are evidence candidates, not physical parts."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import cv2
import pymupdf
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def point_distance(point, polyline):
    points = np.asarray(polyline, dtype=float)
    starts, vectors = points[:-1], np.diff(points, axis=0)
    lengths = np.sum(vectors*vectors, axis=1)
    fractions = np.clip(np.sum((np.asarray(point)-starts)*vectors, axis=1) /
                        np.maximum(lengths, 1e-12), 0, 1)
    return float(np.linalg.norm(starts+fractions[:, None]*vectors-point, axis=1).min())


def stroke_groups(strokes, tolerance=1.0):
    """Join endpoints only; an interior crossing is not an attachment."""
    if not strokes:
        return []
    endpoints = np.array([p for s in strokes for p in (s["points"][0], s["points"][-1])])
    pairs = cKDTree(endpoints).query_pairs(tolerance)
    rows, cols = [], []
    for a, b in pairs:
        rows.extend([a//2, b//2])
        cols.extend([b//2, a//2])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(strokes), len(strokes))).tocsr()
    count, labels = connected_components(graph, directed=False)
    return [[strokes[i] for i in np.flatnonzero(labels == label)] for label in range(count)]


def sample_item(item, transform):
    kind = item[0]
    if kind == "l":
        return [list(p*transform) for p in item[1:3]]
    if kind == "c":
        control = np.array([list(p*transform) for p in item[1:5]])
        return [((1-t)**3*control[0]+3*(1-t)**2*t*control[1]+
                 3*(1-t)*t*t*control[2]+t**3*control[3]).tolist()
                for t in np.linspace(0, 1, 33)]
    if kind == "re":
        r = item[1]
        return [list(p*transform) for p in (r.tl, r.tr, r.br, r.bl, r.tl)]
    if kind == "qu":
        q = item[1]
        return [list(p*transform) for p in (q.ul, q.ur, q.lr, q.ll, q.ul)]
    return None


def extract_strokes(paths, transform, frame, excluded_paths):
    strokes, duplicates, unsupported = [], 0, 0
    keys = {}
    for path_index, path in enumerate(paths):
        if path_index in excluded_paths or path.get("type") not in {"s", "fs"}:
            continue
        for item_index, item in enumerate(path["items"]):
            points = sample_item(item, transform)
            if points is None:
                unsupported += 1
                continue
            a = np.asarray(points)
            low, high = a.min(axis=0), a.max(axis=0)
            if high[0] < frame[0] or high[1] < frame[1] or low[0] > frame[2] or low[1] > frame[3]:
                continue
            if sum(math.dist(a, b) for a, b in zip(points, points[1:])) < .25:
                continue
            # Keep duplicate provenance, without letting repeated CAD strokes dominate groups.
            key_points = tuple(tuple(round(v, 2) for v in p) for p in points)
            key = min(key_points, key_points[::-1])
            ref = {"path_index": path_index, "item_index": item_index, "native_kind": item[0]}
            if key in keys:
                keys[key]["sources"].append(ref)
                duplicates += 1
                continue
            stroke = {"id": f"p{path_index}_i{item_index}", "points": points, "sources": [ref],
                      "width_pdf": path.get("width"), "dashes": path.get("dashes"),
                      "clipped_by_window": bool(low[0] < frame[0] or low[1] < frame[1] or
                                                high[0] > frame[2] or high[1] > frame[3])}
            keys[key] = stroke
            strokes.append(stroke)
    return strokes, {"duplicates_merged": duplicates, "unsupported_items": unsupported}


def group_record(group, query, kind):
    points = np.array([p for stroke in group for p in stroke["points"]])
    digest = hashlib.sha256(json.dumps(sorted(s["id"] for s in group)).encode()).hexdigest()[:12]
    return {"id": f"{kind}_{digest}", "kind": kind, "stroke_ids": [s["id"] for s in group],
            "bbox": [*points.min(axis=0).tolist(), *points.max(axis=0).tolist()],
            "distance_to_query_pixels": min(point_distance(query, s["points"]) for s in group),
            "clipped_by_window": any(s["clipped_by_window"] for s in group),
            "physical_part_id": None, "needs_review": True}


def visible_extent_gate(candidate):
    """Classify geometric extent without promoting a local face or stroke group to a part."""
    if candidate.get("clipped_by_window"):
        status = "visible_partial"
        reasons = ["geometry_crosses_analysis_window"]
    elif candidate.get("kind") == "enclosed_face":
        status = "closed_local_face_not_part"
        reasons = ["local_face_closed", "physical_ownership_unverified"]
    else:
        status = "unverified_stroke_extent"
        reasons = ["stroke_group_not_a_part_contour"]
    return {
        "visible_extent_status": status,
        "closed_local_face": candidate.get("kind") == "enclosed_face" and
                             not candidate.get("clipped_by_window", False),
        "complete_part_contour_established": False,
        "eligible_as_complete_part_training_feature": False,
        "gate_reasons": reasons,
    }


def collinear_runs(strokes, gap=24, tolerance=1):
    lines = [s for s in strokes if len(s["points"]) == 2]
    rows, cols = [], []
    for i, a in enumerate(lines):
        av = np.subtract(a["points"][1], a["points"][0])
        length = np.linalg.norm(av)
        for j in range(i+1, len(lines)):
            b = lines[j]
            bv = np.subtract(b["points"][1], b["points"][0])
            cosine = abs(float(av @ bv))/(length*np.linalg.norm(bv))
            if cosine < math.cos(math.radians(1)):
                continue
            distance = max(abs(av[0]*(p[1]-a["points"][0][1])-av[1]*(p[0]-a["points"][0][0]))/length
                           for p in b["points"])
            endpoint_gap = min(math.dist(p, q) for p in a["points"] for q in b["points"])
            if distance <= tolerance and tolerance < endpoint_gap <= gap:
                rows.extend([i, j])
                cols.extend([j, i])
    if not lines:
        return []
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(lines), len(lines))).tocsr()
    count, labels = connected_components(graph, directed=False)
    return [[lines[i] for i in np.flatnonzero(labels == label)]
            for label in range(count) if np.count_nonzero(labels == label) >= 3]


def dense_parallel_ids(strokes):
    """Detect a local repeated-line pattern, not a confirmed hatch semantic class."""
    lines = [s for s in strokes if len(s["points"]) == 2 and 3 <= math.dist(*s["points"]) <= 100]
    if not lines:
        return set()
    midpoints = np.array([np.mean(s["points"], axis=0) for s in lines])
    directions = np.array([np.subtract(s["points"][1], s["points"][0]) for s in lines], dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    tree = cKDTree(midpoints)
    result = set()
    for i, neighbors in enumerate(tree.query_ball_point(midpoints, 40)):
        aligned = [j for j in neighbors if abs(float(directions[i] @ directions[j])) >= math.cos(math.radians(1))]
        normal = np.array([-directions[i, 1], directions[i, 0]])
        offsets = sorted(float((midpoints[j]-midpoints[i]) @ normal) for j in aligned)
        # Collinear dashed fragments have almost zero normal spread, unlike parallel strokes.
        if len(aligned) >= 5 and offsets[-1]-offsets[0] >= 8:
            result.update(lines[j]["id"] for j in aligned)
    return result


def enclosed_faces(strokes, frame, query, excluded_ids):
    width, height = frame[2]-frame[0], frame[3]-frame[1]
    mask = np.zeros((height, width), dtype=np.uint8)
    raster_strokes = []
    for stroke in strokes:
        if stroke["id"] in excluded_ids:
            continue
        points = np.rint(np.asarray(stroke["points"])-frame[:2]).astype(np.int32)
        cv2.polylines(mask, [points], False, 255, 2)
        low, high = points.min(axis=0), points.max(axis=0)
        raster_strokes.append((stroke["id"], points, low, high))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(255-mask, connectivity=4)
    edge_labels = set(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]).tolist())
    result = []
    for label in range(1, count):
        if label in edge_labels or stats[label, cv2.CC_STAT_AREA] < 30:
            continue
        component = (labels == label).astype(np.uint8)
        contours, hierarchy = cv2.findContours(component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        outer = max((i for i, h in enumerate(hierarchy[0]) if h[3] == -1), key=lambda i: cv2.contourArea(contours[i]))
        contour = contours[outer]
        polygon = (contour[:, 0, :]+np.array(frame[:2])).tolist()
        holes = [(c[:, 0, :]+np.array(frame[:2])).tolist() for i, c in enumerate(contours) if hierarchy[0][i][3] == outer]
        local_query = (float(query[0]-frame[0]), float(query[1]-frame[1]))
        distance = cv2.pointPolygonTest(contour, local_query, True)
        for i, c in enumerate(contours):
            if hierarchy[0][i][3] == outer:
                hole_distance = cv2.pointPolygonTest(c, local_query, True)
                if hole_distance >= 0:
                    distance = -max(hole_distance, 1e-6)
        x, y, w, h, area = (int(v) for v in stats[label])
        boundary = cv2.morphologyEx(
            component, cv2.MORPH_GRADIENT, np.ones((7, 7), dtype=np.uint8),
        )
        left, top = max(0, x-3), max(0, y-3)
        right, bottom = min(width, x+w+3), min(height, y+h+3)
        boundary = boundary[top:bottom, left:right]
        stroke_mask = np.zeros(boundary.shape, dtype=np.uint8)
        touching = []
        for stroke_id, points, low, high in raster_strokes:
            if high[0] < left-2 or high[1] < top-2 or low[0] > right+1 or low[1] > bottom+1:
                continue
            stroke_mask.fill(0)
            cv2.polylines(stroke_mask, [points-np.array([left, top])], False, 1, 2)
            if np.any(stroke_mask & boundary):
                touching.append(stroke_id)
        digest = hashlib.sha256(json.dumps([polygon, holes]).encode()).hexdigest()[:12]
        result.append({"id": f"face_{digest}", "kind": "enclosed_face", "polygon": polygon, "holes": holes,
                       "bbox": [x+frame[0], y+frame[1], x+w+frame[0], y+h+frame[1]],
                       "area_pixels": area, "query_inside": distance >= 0,
                       "distance_to_query_pixels": max(0, -distance), "clipped_by_window": False,
                       "boundary_touching_stroke_ids": sorted(touching),
                       "physical_part_id": None, "needs_review": True})
    return sorted(result, key=lambda f: (f["distance_to_query_pixels"], f["id"]))


def face_exclusions(groups, runs, parallel):
    # Repeated lines may also be end caps or steps. Preserve joined contour evidence.
    protected = {sid for group in groups if len(group["stroke_ids"]) > 1 for sid in group["stroke_ids"]}
    patterns = parallel | {sid for group in runs for sid in group["stroke_ids"]}
    return patterns-protected, protected


def export_local_geometry(source, pdf_path, packets_path, output, items):
    doc = json.loads(source.read_text())
    packets = json.loads(packets_path.read_text())
    pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    if pdf_hash != doc["source_sha256"] or hashlib.sha256(source.read_bytes()).hexdigest() != packets["source_json_sha256"]:
        raise ValueError("Extraction/geometry source mismatch")
    chosen = [p for p in packets["packets"] if p["item_number"] in items]
    if len(chosen) != len(items) or len({p["item_number"] for p in chosen}) != len(items):
        raise ValueError("Expected one packet per requested item")
    output.mkdir(parents=True, exist_ok=False)
    result = {"source_pdf_sha256": pdf_hash, "source_json_sha256": packets["source_json_sha256"],
              "packets_sha256": hashlib.sha256(packets_path.read_bytes()).hexdigest(),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "parameters": {"window_half_size": 240, "endpoint_tolerance_pixels": 1,
                             "bezier_samples": 33, "collinear_gap_pixels": 24,
                             "face_stroke_width_pixels": 2, "face_min_area_pixels": 30,
                             "parallel_min_neighbors": 5, "parallel_radius_pixels": 40},
              "coordinate_space": "original_rendered_page_pixels", "jobs": [],
              "limitations": ["Stroke groups are not physical contours", "No confirmed centerline/hatch classifier",
                              "Bezier distances use sampled curves", "Clipped groups remain incomplete",
                              "Fill-only paths and implicit closePath edges are not reconstructed",
                              "Development drawing, no independent accuracy"]}
    with pymupdf.open(pdf_path) as pdf:
        for packet in chosen:
            if len(packet["candidate_geometry"]) != 1:
                raise ValueError("Ambiguous leader requires review")
            page = next(p for p in doc["pages"] if p["page_number"] == packet["page_number"])
            image_path = Path(page["image_path"])
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != page["image_sha256"]:
                raise ValueError("Source image hash mismatch")
            query = packet["candidate_geometry"][0]["far_endpoint"]
            if not (0 <= query[0] < page["width"] and 0 <= query[1] < page["height"]):
                raise ValueError("Query outside page")
            frame = [max(0, math.floor(query[0])-240), max(0, math.floor(query[1])-240),
                     min(page["width"], math.floor(query[0])+240), min(page["height"], math.floor(query[1])+240)]
            transform = pymupdf.Matrix(*page["pixel_to_pdf_matrix"])
            if transform.invert():
                raise ValueError("Singular page transform")
            excluded = {g["path_index"] for p in packets["packets"] if p["page_number"] == packet["page_number"]
                        for g in p["candidate_geometry"]}
            paths = pdf[packet["page_number"]-1].get_drawings()
            strokes, stats = extract_strokes(paths, transform, frame, excluded)
            groups = [group_record(g, query, "connected") for g in stroke_groups(strokes)]
            runs = [group_record(g, query, "gapped_collinear") for g in collinear_runs(strokes)]
            parallel = dense_parallel_ids(strokes)
            ignored, protected = face_exclusions(groups, runs, parallel)
            faces = enclosed_faces(strokes, frame, query, ignored)
            candidates = sorted(groups, key=lambda g: (g["distance_to_query_pixels"], g["id"]))[:6]
            candidates += sorted(runs, key=lambda g: g["distance_to_query_pixels"])[:3]
            candidates += faces[:4]
            for candidate in candidates:
                candidate["visible_extent_gate"] = visible_extent_gate(candidate)
            by_id = {s["id"]: s for s in strokes}
            with Image.open(image_path) as image:
                if image.size != (page["width"], page["height"]):
                    raise ValueError("Page pixel dimensions mismatch")
                raw = image.convert("RGB").crop(frame)
            raw_path = output / f'{packet["item_number"]}_raw.png'
            raw.save(raw_path)
            pictures = []
            for index, candidate in enumerate(candidates):
                marked = raw.copy()
                draw = ImageDraw.Draw(marked)
                for stroke_id in candidate.get("stroke_ids", []):
                    draw.line([(p[0]-frame[0], p[1]-frame[1]) for p in by_id[stroke_id]["points"]],
                              fill="#0072B2" if candidate["kind"] == "connected" else "#D48400", width=2)
                if candidate["kind"] == "enclosed_face":
                    for ring in [candidate["polygon"], *candidate["holes"]]:
                        points = [(p[0]-frame[0], p[1]-frame[1]) for p in ring]
                        draw.line(points+[points[0]], fill="#008657", width=2)
                x, y = query[0]-frame[0], query[1]-frame[1]
                draw.ellipse((x-5, y-5, x+5, y+5), outline="#C12D76", width=2)
                path = output / f'{packet["item_number"]}_C{index+1}.png'
                marked.save(path)
                candidate["display_label"] = f"C{index+1}"
                pictures.append({"candidate_id": candidate["id"], "path": str(path.resolve()),
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            result["jobs"].append({"item_number": packet["item_number"], "page_number": packet["page_number"],
                                   "query_xy": query, "frame": frame, "excluded_leader_paths": sorted(excluded),
                                   "strokes": strokes, "group_count": len(groups), "run_count": len(runs),
                                   "stats": stats, "candidates": candidates, "candidate_images": pictures,
                                   "face_count": len(faces), "faces": faces,
                                   "face_excluded_stroke_ids": sorted(ignored),
                                   "face_protected_connected_stroke_ids": sorted(protected),
                                   "dense_parallel_stroke_ids": sorted(parallel),
                                   "raw_image": str(raw_path.resolve()),
                                   "raw_image_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()})
    (output / "geometry.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    return result
