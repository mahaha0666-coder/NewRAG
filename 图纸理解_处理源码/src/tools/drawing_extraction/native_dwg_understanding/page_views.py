"""Locate projected views in coordinates of the displayed page, including rotated pages."""

from __future__ import annotations

import cv2
import numpy as np
import pymupdf


def _clear_corner_table(ink: np.ndarray) -> None:
    height, width = ink.shape
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(12, width // 9)), np.uint8))
    _, _, lines, _ = cv2.connectedComponentsWithStats(horizontal)
    rows: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for x, y, w, _, _ in lines[1:]:
        if x > width * .25 and x + w > width * .83 and w > width * .13:
            rows.setdefault((round(x / 6), round((x+w) / 6)), []).append((x, y, x+w))
    for group in rows.values():
        ys = sorted(y for _, y, _ in group)
        if len(ys) < 3 or ys[-1] <= height * .78:
            continue
        left = max(0, min(x for x, _, _ in group) - 2)
        right = min(width, max(r for _, _, r in group) + 3)
        strip = ink[ys[0]:ys[-1]+1,left:right]
        columns = np.mean(strip, axis=0) > .8
        # Require vertical dividers crossing the recurring horizontal rules.
        if np.count_nonzero(np.diff(columns.astype(np.int8), prepend=0) == 1) >= 3:
            end = height if ys[-1] > height * .9 else ys[-1]+3
            ink[max(0,ys[0]-2):end,left:right] = 0


def _rulings(ink: np.ndarray) -> None:
    for turns in range(4):
        _clear_corner_table(np.rot90(ink, turns))

    # Remove the sheet frame and its tick marks without erasing long internal profiles.
    for mask in (ink, ink.T):
        h, w = mask.shape
        rules = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, max(8, round(w*.56))), np.uint8))
        ys = np.flatnonzero(np.any(rules, axis=1))
        upper = ys[ys < h * .12]
        lower = ys[ys > h * .88]
        if len(upper):
            mask[:upper.max()+2] = 0
        if len(lower):
            mask[max(0,lower.min()-1):] = 0


def page_view_boxes(page: pymupdf.Page) -> list[list[float]]:
    """Bounds of printed linework, in displayed PDF points (top-left origin)."""
    rect = page.rect
    scale = 1200 / max(rect.width, rect.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY,
                          alpha=False, annots=False)
    pixels = np.frombuffer(pix.samples_mv, np.uint8).reshape(pix.height, pix.width)
    ink = (pixels < 216).astype(np.uint8)
    _rulings(ink)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    widths, heights, areas = stats[:, 2], stats[:, 3], stats[:, 4]
    keep = (np.maximum(widths, heights) >= 15) & (areas >= 14)
    keep[0] = False
    strokes = keep[labels].astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    joined = cv2.dilate(strokes, kernel)
    _, groups, grouped, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    boxes = []
    for index, (x, y, w, h, _) in enumerate(grouped[1:], start=1):
        if w*h < ink.size*.003 or min(w,h) < 28 or max(w,h) < 56:
            continue
        vertical_margin = w < pix.width*.055 and h > pix.height*.35 and (x < pix.width*.09 or x+w > pix.width*.91)
        horizontal_margin = h < pix.height*.055 and w > pix.width*.35 and (y < pix.height*.09 or y+h > pix.height*.91)
        if vertical_margin or horizontal_margin or (w > pix.width*.9 and h > pix.height*.87):
            continue
        mask = (groups[y:y+h, x:x+w] == index) & (strokes[y:y+h, x:x+w] != 0)
        yy, xx = np.nonzero(mask)
        if len(xx) < max(90, w+h):
            continue
        # Long grid lines have nearly all their ink in a few rows / columns.
        rich_rows = (mask.sum(axis=1) >= 3).sum()
        rich_cols = (mask.sum(axis=0) >= 3).sum()
        if min(rich_rows / h, rich_cols / w) < .16:
            continue
        density = len(xx) / (w * h)
        if density > .13 and len(np.unique(labels[y:y+h,x:x+w][mask])) >= 5:
            horizontal = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                                          np.ones((1, max(12, w//4)), np.uint8))
            vertical = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                                        np.ones((max(12, h//4), 1), np.uint8))
            if not horizontal.any() and not vertical.any():
                continue
        boxes.append([round(v/scale, 3) for v in
                      (x+xx.min(), y+yy.min(), x+xx.max()+1, y+yy.max()+1)])
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def recover_small_sheet(page: pymupdf.Page, svg: bytes):
    """Reframe a dominant tiny printed sheet in a full-extent DWG preview."""
    import xml.etree.ElementTree as ET

    width, height = page.rect.width, page.rect.height
    rects = [list(p["rect"]) for p in page.get_drawings()
             if p["rect"].width < width * .12 and p["rect"].height < height * .12]
    if len(rects) < 100:
        return None
    bounds = np.asarray(rects)
    if not np.isfinite(bounds).all():
        return None
    centers = (bounds[:, :2] + bounds[:, 2:]) / 2
    middle = np.median(centers, axis=0)
    nearby = np.max(np.abs((centers - middle) / [width, height]), axis=1) < .06
    if nearby.mean() < .99:
        return None
    cluster = bounds[nearby]
    low, high = cluster[:, :2].min(axis=0), cluster[:, 2:].max(axis=0)
    extent = max((high - low) / [width, height])
    if not 0 < extent < .04 or min(high - low) <= 0:
        return None
    # Use a display transform only. Preserve the native PDF and original CAD
    # correspondence; the printed pen width needs the inverse magnification.
    pad = max(high - low) * .035
    x, y = low - pad
    w, h = high - low + 2 * pad
    zoom = min(width / w, height / h)
    root = ET.fromstring(svg)
    root.set("viewBox", f"{x} {y} {w} {h}")
    root.set("width", str(w*zoom))
    root.set("height", str(h*zoom))
    for node in root.iter():
        if (pen := node.get("stroke-width")) is not None:
            node.set("stroke-width", str(float(pen) / zoom))
    # MuPDF's SVG reader expects an unprefixed root and XLink href attributes.
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    display = ET.tostring(root, encoding="utf-8")
    with pymupdf.open(stream=display, filetype="svg") as rendered:
        boxes = page_view_boxes(rendered[0])
    if not boxes:
        return None
    return display, boxes, (float(x), float(y), float(w*zoom), float(h*zoom), float(zoom))
