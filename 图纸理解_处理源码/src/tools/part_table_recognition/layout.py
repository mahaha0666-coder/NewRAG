"""Bind fresh header readings to ruled columns and distinguish physical headers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .geometry import SourceCrop
from .reader import read_region, spotting_tokens
from .structure import FIELD_ALIASES


def field_key(text: str) -> str | None:
    return FIELD_ALIASES.get("".join(text.split()).casefold())


def read_header(
    crop: SourceCrop, xs: list[int], ys: list[int], *, endpoint: str, output: Path,
    html_header_present: bool,
) -> tuple[list[dict[str, Any]] | None, int | None]:
    physical = list(zip(ys, ys[1:]))
    if not physical:
        return None, None
    # A header is located by its own text, not by the number of recognized rows.
    candidates = [len(physical) - 1, 0] if len(physical) > 1 else [0]
    for candidate in candidates:
        y0, y1 = physical[candidate]
        image = crop.image[y0:y1 + 1, xs[0]:xs[-1] + 1]
        result = read_region(image, task="spotting", endpoint=endpoint,
                             output=output / f"header-{candidate:03d}.json")
        labels: dict[int, list[dict[str, Any]]] = {}
        for token in spotting_tokens(result["content"], (image.shape[1], image.shape[0])):
            x0, _, x1, _ = token["bbox"]
            center = (x0 + x1) / 2 + xs[0]
            column = next((index for index in range(len(xs) - 1) if xs[index] <= center < xs[index + 1]), None)
            if column is not None:
                labels.setdefault(column, []).append(token)
        values = {index: "".join(token["text"] for token in sorted(tokens, key=lambda value: (value["bbox"][1], value["bbox"][0])))
                  for index, tokens in labels.items()}
        known = {field_key(value) for value in values.values()} - {None}
        if len(known) < 3 or not {"item_no", "part_code"}.issubset(known):
            continue
        columns = []
        for index, (x0, x1) in enumerate(zip(xs, xs[1:])):
            label = values.get(index, "")
            if not field_key(label):
                reread = read_region(crop.image[y0:y1 + 1, x0:x1 + 1], task="ocr", endpoint=endpoint,
                                     output=output / f"header-{candidate:03d}-column-{index:02d}.json")
                label = reread["content"].strip() or label
            columns.append({"key": field_key(label) or f"column_{index}", "label": label,
                "bbox": crop.page_box([x0, ys[0], x1, ys[-1]]),
                "header_source_box": crop.page_box([x0, y0, x1, y1]),
                "header_raw_ref": str((output / f"header-{candidate:03d}.json").name)})
        return columns, candidate
    return None, None


def share_columns(
    xs: list[int], ys: list[int], crop: SourceCrop, reference: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Reuse only a current-page header with independently matching ruling widths."""
    previous_xs = reference["xs"]
    if len(xs) != len(previous_xs) or len(xs) < 3:
        return None
    widths = np.diff(xs) / (xs[-1] - xs[0])
    previous = np.diff(previous_xs) / (previous_xs[-1] - previous_xs[0])
    if np.max(np.abs(widths - previous)) > .015:
        return None
    return [{**column, "bbox": crop.page_box([xs[index], ys[0], xs[index + 1], ys[-1]]),
             "header_relation": {"kind": "same_page_continuation_header", "table_id": reference["table_id"]}}
            for index, column in enumerate(reference["columns"])]
