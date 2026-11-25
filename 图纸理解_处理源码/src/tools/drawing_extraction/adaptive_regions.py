"""Deterministic high-resolution crop planning for difficult drawing layouts."""

from __future__ import annotations


def adaptive_region_plan(width: int, height: int, *, max_regions: int = 4) -> list[dict]:
    """Return bounded OCR crops for likely title/BOM and rotated-table areas.

    The global pass decides whether this plan is needed.  These crops are
    intentionally overlapping and source-coordinate preserving; they are not
    semantic guesses and never create a relationship by themselves.
    """
    if width < 1200 or height < 900 or not 1 <= max_regions <= 4:
        raise ValueError("Unsupported page size or region limit")
    candidates = [
        ("adaptive_bom_horizontal", 0.50, 0.34, 0.995, 0.91, 0),
        ("adaptive_bom_vertical", 0.50, 0.34, 0.995, 0.91, 90),
        ("adaptive_lower_sheet", 0.02, 0.54, 0.995, 0.995, 0),
        ("adaptive_lower_sheet_vertical", 0.02, 0.54, 0.995, 0.995, 90),
    ]
    regions = []
    for pass_id, x0, y0, x1, y1, rotation in candidates[:max_regions]:
        frame = [max(0, int(round(x0 * width))), max(0, int(round(y0 * height))),
                 min(width, int(round(x1 * width))), min(height, int(round(y1 * height)))]
        if frame[2] <= frame[0] or frame[3] <= frame[1]:
            continue
        regions.append({"pass_id": pass_id, "frame": frame,
                        "rotation": rotation, "scale": 1.5})
    return regions


def full_page_tile_plan(width: int, height: int, *, tile_side: int = 2200,
                        overlap: float = 0.15) -> list[dict]:
    """Cover a rendered page with bounded OCR tiles in source coordinates.

    A large engineering sheet must not be resized to one detector input: the
    smallest dimension text then disappears before OCR sees it.  The returned
    rectangles cover the whole page, including the right/bottom edge, with a
    measured overlap so tokens crossing tile boundaries are recoverable and
    later deduplication can merge them.
    """
    if width < 1 or height < 1 or not 1200 <= tile_side <= 4000:
        raise ValueError("Unsupported page or OCR tile size")
    if not 0.05 <= overlap < 0.50:
        raise ValueError("overlap must be between 0.05 and 0.50")
    if width <= tile_side and height <= tile_side:
        return [{"pass_id": "full_page", "frame": [0, 0, width, height],
                 "rotation": 0, "scale": 1}]
    step = max(1, int(round(tile_side * (1.0 - overlap))))

    def starts(length: int) -> list[int]:
        if length <= tile_side:
            return [0]
        values = list(range(0, length - tile_side + 1, step))
        final = length - tile_side
        if values[-1] != final:
            values.append(final)
        return values

    regions = []
    for row, y0 in enumerate(starts(height)):
        for column, x0 in enumerate(starts(width)):
            x1, y1 = min(width, x0 + tile_side), min(height, y0 + tile_side)
            regions.append({"pass_id": f"full_tile_r{row:02d}c{column:02d}",
                            "frame": [x0, y0, x1, y1], "rotation": 0, "scale": 1})
    return regions


__all__ = ["adaptive_region_plan", "full_page_tile_plan"]
