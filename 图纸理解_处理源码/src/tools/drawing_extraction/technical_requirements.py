"""Locate numbered technical requirements without guessing their part ownership."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from .document_schema import DrawingDocument, Entity, Location, stable_id
from .document_tables import PART_NUMBER, normalize_part_number


NUMBERED_LINE = re.compile(
    r"^\s*([1-9]\d?)\s*(?:[.．、)）]\s*|\s+)(\S[\s\S]{1,})$"
)
COMPACT_NUMBERED_LINE = re.compile(r"^\s*([1-9])([\u3400-\u9fff]{2}[\s\S]{2,})$")
COMPACT_NUMBERED_PREFIX = re.compile(r"^\s*[1-9]\d?[\u3400-\u9fff]")
STANDARD_REFERENCE = re.compile(r"^(?:GB(?:/T)?|ISO|DIN|JIS)[0-9A-Za-z./+\-]+$", re.I)
HEADING = "技术要求"
HEADING_PUNCTUATION = "：:。.;；"
CONDITIONAL_HEADING = re.compile(
    r"^若此件(?P<condition>外购|自制)[,，]则相应技术要求如下[：:]?$"
)


def _box(entity: Entity, page_number: int) -> tuple[float, float, float, float] | None:
    for location in entity.locations:
        if location.page_number == page_number and location.bbox is not None:
            return location.bbox
    return None


def _same_position(first: tuple[float, float, float, float],
                   second: tuple[float, float, float, float]) -> bool:
    overlap_x = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    overlap_y = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return (overlap_x >= .7 * min(first[2] - first[0], second[2] - second[0])
            and overlap_y >= .7 * min(first[3] - first[1], second[3] - second[1]))


def _normalized_text(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def _heading_applicability(text: str) -> str | None:
    normalized = _normalized_text(text)
    if normalized.rstrip(HEADING_PUNCTUATION) == HEADING:
        return "unconditional"
    match = CONDITIONAL_HEADING.fullmatch(normalized)
    if match is None:
        return None
    return "external_purchase" if match["condition"] == "外购" else "self_manufacture"


def _trusted_heading(entity: Entity, observations: list[Entity],
                     page_number: int) -> bool:
    if _heading_applicability(entity.raw_text or "") is None:
        return False
    if (entity.kind == "raw_ocr"
            and entity.attributes.get("localization") == "text_polygon"):
        return True
    if (entity.kind == "regions"
            and entity.attributes.get("label") == "paragraph_title"):
        return True
    if not (entity.kind == "regions" and entity.attributes.get("label") == "text"):
        return False
    box = _box(entity, page_number)
    if box is None:
        return False
    passes = {
        str(other.attributes["recognition_pass"])
        for other in observations
        if other.kind == "regions" and other.attributes.get("label") == "text"
        and _heading_applicability(other.raw_text or "") == _heading_applicability(entity.raw_text or "")
        and other.attributes.get("recognition_pass")
        and (other_box := _box(other, page_number)) is not None
        and _same_position(box, other_box)
    }
    return len(passes) >= 2


def _looks_numbered(text: str) -> bool:
    return bool(NUMBERED_LINE.fullmatch(text) or COMPACT_NUMBERED_PREFIX.match(text))


def _numbered_requirement(text: str) -> re.Match[str] | None:
    match = NUMBERED_LINE.fullmatch(text) or COMPACT_NUMBERED_LINE.fullmatch(text)
    if match is None or re.search(r"[A-Za-z\u3400-\u9fffØΦ∅⌀±]", match[2]) is None:
        return None
    if any(_looks_numbered(line.strip()) for line in text.splitlines()[1:]):
        return None
    return match


def _part_number_only(text: str) -> bool:
    normalized = normalize_part_number(text)
    return bool(PART_NUMBER.fullmatch(normalized)
                and not STANDARD_REFERENCE.fullmatch(normalized))


def _center_inside(container: tuple[float, float, float, float],
                   candidate: tuple[float, float, float, float]) -> bool:
    center_x = (candidate[0] + candidate[2]) / 2
    center_y = (candidate[1] + candidate[3]) / 2
    return container[0] <= center_x <= container[2] and container[1] <= center_y <= container[3]


def _crosses_into_adjacent_table(
    anchor: tuple[float, float, float, float],
    candidate: tuple[float, float, float, float],
    tables: list[tuple[float, float, float, float]],
) -> bool:
    return any(_center_inside(table, candidate) and not _center_inside(table, anchor)
               for table in tables)


def _requirement_table_boxes(
    title_box: tuple[float, float, float, float],
    tables: list[tuple[float, float, float, float]],
    observations: list[Entity],
    page_number: int,
) -> set[tuple[float, float, float, float]]:
    """Identify coarse layout tables that swallowed a numbered-note block.

    Paddle sometimes merges the requirements area and adjacent title block into
    one table region.  We only waive table exclusion when at least two localized
    numbered lines sit in that table immediately below the exact heading, or
    follow an aligned first item that starts just above the table boundary.
    """

    title_height = title_box[3] - title_box[1]
    output = set()
    for table in tables:
        if _center_inside(table, title_box):
            output.add(table)
            continue
        numbered_boxes = []
        preceding_boxes = []
        for observation in observations:
            if not (observation.kind == "raw_ocr"
                    and observation.attributes.get("localization") == "text_polygon"):
                continue
            match = _numbered_requirement((observation.raw_text or "").strip())
            if match is None or _part_number_only(match[2]):
                continue
            box = _box(observation, page_number)
            if box is None:
                continue
            if _center_inside(table, box):
                numbered_boxes.append((int(match[1]), box))
            elif title_box[3] <= box[1] < table[1]:
                preceding_boxes.append((int(match[1]), box))
        if len({number for number, _ in numbered_boxes}) < 2:
            continue
        first_number, first = min(numbered_boxes, key=lambda item: (item[1][1], item[1][0]))
        horizontally_aligned = (
            title_box[0] - 2 * title_height <= first[0]
            and first[0] <= title_box[2] + 2 * title_height
        )
        immediately_below = 0 <= first[1] - title_box[3] <= max(2 * title_height, 40)
        if horizontally_aligned and immediately_below:
            output.add(table)
            continue
        if not horizontally_aligned or first_number != 2 or 3 not in {
            number for number, _ in numbered_boxes
        }:
            continue
        for number, preceding in preceding_boxes:
            preceding_height = preceding[3] - preceding[1]
            if (number == 1
                    and title_box[0] - 2 * title_height <= preceding[0]
                    <= title_box[2] + 2 * title_height
                    and preceding[1] - title_box[3] <= max(2 * title_height, 40)
                    and abs(first[0] - preceding[0]) <= max(1.5 * title_height, 40)
                    and 0 <= first[1] - preceding[3]
                    <= max(2 * preceding_height, 2 * title_height)):
                output.add(table)
                break
    return output


def _same_pass_continuations(
    primary: Entity,
    observations: list[Entity],
    page_number: int,
    next_numbered_axis: float | None,
    excluded_tables: list[tuple[float, float, float, float]],
) -> list[Entity]:
    """Return localized continuation lines without crossing into another block."""

    primary_box = _box(primary, page_number)
    recognition_pass = primary.attributes.get("recognition_pass")
    if primary_box is None or not recognition_pass:
        return []
    line_height = primary_box[3] - primary_box[1]
    previous_box = primary_box
    continuations = []
    candidates = []
    for observation in observations:
        if observation.id == primary.id:
            continue
        if not (observation.kind == "raw_ocr"
                and observation.attributes.get("localization") == "text_polygon"
                and observation.attributes.get("recognition_pass") == recognition_pass):
            continue
        text = (observation.raw_text or "").strip()
        box = _box(observation, page_number)
        if not text or box is None or _looks_numbered(text):
            continue
        if any(_center_inside(table, box) for table in excluded_tables):
            continue
        candidates.append((box, observation))
    ordered_candidates = sorted(
        candidates,
        key=lambda item: (item[0][1], item[0][0], item[1].id),
    )
    for box, observation in ordered_candidates:
        center_y = (box[1] + box[3]) / 2
        if next_numbered_axis is not None and center_y >= next_numbered_axis:
            break
        vertical_gap = box[1] - previous_box[3]
        aligned = abs(box[0] - primary_box[0]) <= max(1.5 * line_height, 32)
        if vertical_gap < -0.35 * line_height:
            continue
        if vertical_gap > max(1.25 * line_height, 40):
            if box[1] > previous_box[3]:
                break
            continue
        if not aligned:
            continue
        continuations.append(observation)
        previous_box = box
    return continuations


def _overlap_height(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    return max(0.0, min(first[3], second[3]) - max(first[1], second[1]))


def _text_overlap(current: str, fragment: str) -> tuple[int, int] | None:
    """Locate an exact suffix/prefix overlap, allowing one noisy leading glyph."""

    current_text = _normalized_text(current)
    fragment_text = _normalized_text(fragment)
    for prefix_noise in (0, 1):
        available = len(fragment_text) - prefix_noise
        for length in range(min(len(current_text), available), 1, -1):
            overlap = fragment_text[prefix_noise:prefix_noise + length]
            minimum = 2 if any("\u3400" <= char <= "\u9fff" for char in overlap) else 4
            if length >= minimum and current_text.endswith(overlap):
                return prefix_noise, length
    return None


def _has_near_overlap_conflict(current: str, fragment: str) -> bool:
    """Return true for a suffix/prefix overlap containing one disputed glyph."""

    current_text = _normalized_text(current)
    fragment_text = _normalized_text(fragment)
    for prefix_noise in (0, 1):
        available = len(fragment_text) - prefix_noise
        for length in range(min(len(current_text), available), 3, -1):
            left = current_text[-length:]
            right = fragment_text[prefix_noise:prefix_noise + length]
            differences = sum(first != second for first, second in zip(left, right))
            if differences == 1:
                return True
    return False


def _raw_suffix_after_normalized_prefix(text: str, prefix_length: int) -> str:
    """Trim a normalized prefix while preserving the remaining raw characters."""

    if prefix_length <= 0:
        return text
    normalized_count = 0
    for index, char in enumerate(text):
        for normalized_char in unicodedata.normalize("NFKC", char):
            if normalized_char.isspace():
                continue
            normalized_count += 1
            if normalized_count == prefix_length:
                return text[index + 1:]
    return ""


def _same_line_fragments(
    primary: Entity,
    observations: list[Entity],
    page_number: int,
    *, reject_tall_regions: bool = False,
) -> tuple[str, list[Entity], list[Entity]]:
    """Recover text split horizontally across overlapping Paddle input windows.

    Exact text overlap is required unless two localized polygons are separated by
    only a small fraction of their line height.  Conflicting overlapping readings
    are returned for audit and never used to overwrite the primary observation.
    """

    primary_box = _box(primary, page_number)
    if primary_box is None:
        return primary.raw_text or "", [], []
    line_height = primary_box[3] - primary_box[1]
    localized_heights = sorted(
        box[3] - box[1]
        for observation in observations
        if observation.kind == "raw_ocr"
        and observation.attributes.get("localization") == "text_polygon"
        and (box := _box(observation, page_number)) is not None
        and _overlap_height(primary_box, box) >= 0.5 * min(line_height, box[3] - box[1])
    )
    if localized_heights:
        line_height = min(line_height, localized_heights[len(localized_heights) // 2])
    table_boxes = [_box(entity, page_number) for entity in observations
                   if entity.kind == "regions" and entity.attributes.get("label") == "table"
                   and _box(entity, page_number) is not None]
    current_text = primary.raw_text or ""
    current_right = primary_box[2]
    used: list[Entity] = []
    conflicts: list[Entity] = []
    remaining = []
    for observation in observations:
        if observation.id == primary.id:
            continue
        is_localized_text = (
            observation.kind == "raw_ocr"
            and observation.attributes.get("localization") == "text_polygon"
        ) or (
            observation.kind == "regions"
            and observation.attributes.get("label") == "text"
        )
        text = (observation.raw_text or "").strip()
        box = _box(observation, page_number)
        if not is_localized_text or not text or "\n" in text or box is None:
            continue
        if (reject_tall_regions and box[3] - box[1]
                > 1.5 * (primary_box[3] - primary_box[1])):
            continue
        if _looks_numbered(text):
            continue
        overlap = _overlap_height(primary_box, box)
        if overlap < 0.55 * min(line_height, box[3] - box[1]):
            continue
        if box[0] <= primary_box[0] + max(0.25 * line_height, 16):
            continue
        if box[2] <= current_right + max(0.05 * line_height, 4):
            continue
        remaining.append((box, observation))

    while remaining:
        ranked = []
        for box, observation in remaining:
            if box[2] <= current_right + max(0.05 * line_height, 4):
                continue
            overlap = _text_overlap(current_text, observation.raw_text or "")
            gap = box[0] - current_right
            close_gap = bool(used) and 0 <= gap <= max(0.35 * line_height, 16)
            if (used and current_text.rstrip().endswith("。")
                    and (overlap is not None or close_gap)
                    and _crosses_into_adjacent_table(primary_box, box, table_boxes)):
                if observation.id not in {conflict.id for conflict in conflicts}:
                    conflicts.append(observation)
                continue
            if overlap is not None:
                added_text = _raw_suffix_after_normalized_prefix(
                    observation.raw_text or "", overlap[0] + overlap[1]
                )
                if not _normalized_text(added_text):
                    continue
            if overlap is None and not close_gap:
                conflict_ids = {conflict.id for conflict in conflicts}
                if (
                    box[0] < current_right
                    and _has_near_overlap_conflict(
                        current_text, observation.raw_text or ""
                    )
                    and observation.id not in conflict_ids
                ):
                    conflicts.append(observation)
                continue
            ranked.append((
                overlap is not None,
                observation.kind == "raw_ocr",
                overlap[1] if overlap else 0,
                -box[2],
                observation.id,
                box,
                observation,
                overlap,
            ))
        if not ranked:
            break
        _, _, _, _, _, box, selected, overlap = max(ranked)
        if overlap is None:
            current_text = f"{current_text}{selected.raw_text or ''}"
        else:
            prefix_noise, overlap_length = overlap
            suffix = _raw_suffix_after_normalized_prefix(
                selected.raw_text or "", prefix_noise + overlap_length
            )
            current_text = (
                f"{current_text}{suffix}"
            )
        current_right = max(current_right, box[2])
        used.append(selected)
        remaining = [item for item in remaining if item[1].id != selected.id]
    return current_text, used, conflicts


def _record_issue(document: DrawingDocument, issue: dict) -> None:
    if issue not in document.issues:
        document.issues.append(issue)


def _supported_duplicate_cluster(
    candidates: list[Entity], page_number: int, line_width: float,
) -> list[Entity] | None:
    """Keep a repeated OCR line only when independent passes resolve its position."""

    threshold = max(20, line_width * .8)
    clusters: list[list[Entity]] = []
    for entity in sorted(candidates, key=lambda item: _box(item, page_number)[1]):
        start = _box(entity, page_number)[1]
        if not clusters or start - _box(clusters[-1][0], page_number)[1] > threshold:
            clusters.append([entity])
        else:
            clusters[-1].append(entity)
    if len(clusters) < 2:
        return None
    ranked = sorted(clusters, key=lambda cluster: len({
        str(entity.attributes["recognition_pass"])
        for entity in cluster
        if entity.kind == "raw_ocr" and entity.attributes.get("recognition_pass")
    }), reverse=True)
    support = [len({
        str(entity.attributes["recognition_pass"])
        for entity in cluster
        if entity.kind == "raw_ocr" and entity.attributes.get("recognition_pass")
    }) for cluster in ranked]
    if support[0] < 2 or support[0] <= support[1]:
        return None
    return ranked[0]


def _has_single_line_reading(
    observation: Entity, observations: list[Entity], page_number: int,
    number: int,
) -> bool:
    """Use a localized single line for stitching when a tile OCR also spans lines."""

    box = _box(observation, page_number)
    for other in observations:
        if other.id == observation.id or "\n" in (other.raw_text or ""):
            continue
        text = (other.raw_text or "").strip()
        match = (_numbered_requirement(text)
                 if NUMBERED_LINE.fullmatch(text) else None)
        other_box = _box(other, page_number)
        if (match is not None and int(match[1]) == number and other_box is not None
                and abs(other_box[1] - box[1]) <= max(20, .35 * (box[3] - box[1]))
                and abs(other_box[0] - box[0]) <= max(32, .35 * (box[2] - box[0]))):
            return True
    return False


def _supported_compact_entry(
    entry: tuple[float, float, int, Entity],
    regular: list[tuple[float, float, int, Entity]],
    line_width: float,
    page_height: float,
) -> bool:
    if any(other[2] == entry[2] for other in regular):
        return False
    alignment_tolerance = max(1.4 * line_width, 32)
    span_limit = max(8 * line_width, page_height * .08)
    previous = [other for other in regular
                if other[2] == entry[2] - 1 and other[0] < entry[0]
                and abs(other[1] - entry[1]) <= alignment_tolerance]
    following = [other for other in regular
                 if other[2] == entry[2] + 1 and entry[0] < other[0]
                 and abs(other[1] - entry[1]) <= alignment_tolerance]
    return any(next_entry[0] - prior[0] <= span_limit
               for prior in previous for next_entry in following)


def extract_ocr_technical_requirements(document: DrawingDocument) -> int:
    """Preserve numbered note text and conflicting OCR readings as review candidates."""
    added = 0
    for page in document.pages:
        observations = [entity for entity in document.regions + document.raw_ocr
                        if _box(entity, page.page_number) is not None]
        headings = [entity for entity in observations
                    if _trusted_heading(entity, observations, page.page_number)]
        selected_headings = []
        for heading in sorted(headings, key=lambda entity: (
            _heading_applicability(entity.raw_text or "") == "unconditional",
            entity.kind == "regions" and entity.attributes.get("label") == "text",
            entity.kind != "regions", entity.id,
        )):
            box = _box(heading, page.page_number)
            if not any(_same_position(box, _box(other, page.page_number))
                       for other in selected_headings):
                selected_headings.append(heading)
        for heading in selected_headings:
            applicability = _heading_applicability(heading.raw_text or "")
            fallback_heading = (heading.kind == "regions"
                                and heading.attributes.get("label") == "text")
            title_box = _box(heading, page.page_number)
            title_width, title_height = title_box[2] - title_box[0], title_box[3] - title_box[1]
            vertical = title_height >= 1.4 * title_width
            line_width = title_width if vertical else title_height
            matches = defaultdict(list)
            bridged_numbers = set()
            positioned = []
            table_candidates = []
            vertical_columns = {"left": [], "right": []}
            tables = [_box(region, page.page_number) for region in document.regions
                      if region.attributes.get("label") == "table"
                      and _box(region, page.page_number) is not None]
            requirement_tables = _requirement_table_boxes(
                title_box, tables, observations, page.page_number,
            )
            tables = [table for table in tables if table not in requirement_tables]
            for observation in observations:
                if not (observation.kind == "regions" and observation.attributes.get("label") == "text"
                        or observation.kind == "raw_ocr" and observation.attributes.get("localization")
                        == "text_polygon"):
                    continue
                match = _numbered_requirement((observation.raw_text or "").strip())
                if not match:
                    continue
                if _part_number_only(match[2]):
                    merged, fragments, _ = _same_line_fragments(
                        observation, observations, page.page_number,
                    )
                    if (not fragments or _part_number_only(
                            (_numbered_requirement(merged) or match)[2]
                    )):
                        continue
                if ("\n" in (observation.raw_text or "")
                        and _has_single_line_reading(
                            observation, observations, page.page_number, int(match[1])
                        )):
                    continue
                box = _box(observation, page.page_number)
                if vertical:
                    if not (box[3] - box[1] >= 1.4 * (box[2] - box[0])
                            and box[1] <= title_box[3] and box[3] >= title_box[1]):
                        continue
                    reach = max(20 * line_width, page.width * .12)
                    if title_box[0] - reach <= box[2] <= title_box[0]:
                        side = "left"
                        axis = -(box[0] + box[2]) / 2
                    elif title_box[2] <= box[0] <= title_box[2] + reach:
                        side = "right"
                        axis = (box[0] + box[2]) / 2
                    else:
                        continue
                    alignment = box[1]
                else:
                    if not (title_box[3] <= (box[1] + box[3]) / 2
                            <= min(page.height, title_box[3] + max(20 * line_width, page.height * .12))
                            and title_box[0] - page.width * .12 <= box[0] <= title_box[2]):
                        continue
                    axis, alignment = (box[1] + box[3]) / 2, box[0]
                center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                entry = (axis, alignment, int(match[1]), observation)
                covering_tables = [table for table in tables
                                   if table[0] <= center_x <= table[2]
                                   and table[1] <= center_y <= table[3]]
                if covering_tables and not vertical:
                    table_candidates.append(entry)
                    continue
                if vertical:
                    vertical_columns[side].append(entry)
                else:
                    positioned.append(entry)
            regular = [entry for entry in positioned if not COMPACT_NUMBERED_LINE.fullmatch(
                (entry[3].raw_text or "").strip()
            )]
            positioned = [entry for entry in positioned
                          if entry in regular or _supported_compact_entry(
                              entry, regular, line_width, page.height,
                          )]
            if vertical:
                valid_columns = {}
                for side, column in vertical_columns.items():
                    column.sort(key=lambda item: (item[0], item[1], item[3].id))
                    if column:
                        first_gap = (title_box[0] + column[0][0] if side == "left"
                                     else column[0][0] - title_box[2])
                        if first_gap <= max(3.5 * line_width, page.width * .06):
                            valid_columns[side] = column
                if len(valid_columns) > 1:
                    starts_at_one = [side for side, column in valid_columns.items()
                                     if column[0][2] == 1]
                    if len(starts_at_one) == 1:
                        valid_columns = {starts_at_one[0]: valid_columns[starts_at_one[0]]}
                    else:
                        _record_issue(document, {
                            "stage": "technical_requirement_direction_ambiguous",
                            "page": page.page_number, "heading_id": heading.id,
                            "candidate_entity_ids": sorted({
                                entry[3].id for column in valid_columns.values() for entry in column
                            }),
                        })
                        continue
                vertical_side, positioned = next(iter(valid_columns.items()), (None, []))
            positioned.sort(key=lambda item: (item[0], item[1], item[3].id))
            first_gap = (((title_box[0] + positioned[0][0]) if vertical_side == "left"
                          else (positioned[0][0] - title_box[2])) if vertical
                         else (positioned[0][0] - title_box[3])) if positioned else 0
            page_span = page.width if vertical else page.height
            if positioned and first_gap <= max(3.5 * line_width, page_span * .06):
                anchor = positioned[0][1]
                anchor_box = _box(positioned[0][3], page.page_number)
                previous_axis = positioned[0][0]
                previous_observation = positioned[0][3]
                previous_number = positioned[0][2]
                for axis, alignment, number, observation in positioned:
                    previous_box = _box(previous_observation, page.page_number)
                    current_box = _box(observation, page.page_number)
                    adjacent_height = min(
                        previous_box[3] - previous_box[1],
                        current_box[3] - current_box[1],
                    )
                    max_numbered_gap = max(
                        2.8 * line_width, page_span * .035,
                        min(3 * adjacent_height, page_span * .08),
                    )
                    bridged = False
                    if axis - previous_axis > max_numbered_gap:
                        if vertical:
                            break
                        bridge_end = max(
                            (
                                _box(entity, page.page_number)[3]
                                for candidate in matches[previous_number]
                                for entity in _same_pass_continuations(
                                    candidate, observations, page.page_number, axis, tables,
                                )
                            ),
                            default=None,
                        )
                        if (bridge_end is None or not -0.35 * line_width
                                <= current_box[1] - bridge_end <= max(1.25 * line_width, 40)):
                            break
                        bridged = True
                    if vertical:
                        aligned = min(
                            abs(current_box[1] - anchor_box[1]),
                            abs(current_box[3] - anchor_box[3]),
                        ) <= max(1.4 * line_width, 32)
                    else:
                        aligned = abs(alignment - anchor) <= max(1.4 * line_width, 32)
                    if not aligned:
                        continue
                    matches[number].append(observation)
                    if bridged:
                        bridged_numbers.add(number)
                    previous_axis = axis
                    previous_observation = observation
                    previous_number = number
            if not vertical and matches and table_candidates:
                previous_number = max(matches)
                previous_entries = matches[previous_number]
                previous = max(previous_entries, key=lambda entity: _box(entity, page.page_number)[3])
                previous_box = _box(previous, page.page_number)
                next_entries = [entry for entry in table_candidates
                                if entry[2] == previous_number + 1
                                and abs(entry[1] - anchor) <= max(1.4 * line_width, 32)]
                support = {str(entry[3].attributes.get("recognition_pass"))
                           for entry in next_entries
                           if entry[3].attributes.get("recognition_pass")}
                for axis, _, number, observation in next_entries:
                    box = _box(observation, page.page_number)
                    line_height = previous_box[3] - previous_box[1]
                    table_borders = [table[1] for table in tables
                                     if table[0] <= (box[0] + box[2]) / 2 <= table[2]
                                     and table[1] <= (box[1] + box[3]) / 2 <= table[3]]
                    if (len(support) >= 2
                            and 0 <= axis - (previous_box[1] + previous_box[3]) / 2
                            <= max(2 * line_height, page.height * .035)
                            and any(-line_height <= border - previous_box[3] <= line_height
                                    for border in table_borders)):
                        matches[number].append(observation)
            if not matches:
                _record_issue(document, {"stage": "technical_requirement_ocr",
                                         "page": page.page_number, "heading_id": heading.id,
                                         "error": "No numbered text below heading"})
                continue
            if fallback_heading:
                supporting = [entity for entity in headings
                              if entity.kind == "regions"
                              and entity.attributes.get("label") == "text"
                              and _same_position(title_box, _box(entity, page.page_number))]
                _record_issue(document, {
                    "stage": "technical_requirement_layout_heading_review",
                    "page": page.page_number, "heading_id": heading.id,
                    "supporting_heading_ids": sorted(entity.id for entity in supporting),
                    "raw_result_refs": sorted({ref for entity in supporting
                                               for ref in entity.raw_result_refs}),
                })
            missing = sorted(set(range(1, max(matches) + 1)) - set(matches))
            has_sequence_anchor = 1 in matches or (2 in matches and len(matches) >= 2)
            if missing and has_sequence_anchor and max(matches) <= 9:
                _record_issue(document, {"stage": "technical_requirement_sequence",
                                         "page": page.page_number, "heading_id": heading.id,
                                         "missing_item_numbers": missing})
            for number, candidates in sorted(matches.items()):
                starts = [(box[2] if vertical else box[1]) for entity in candidates
                          if (box := _box(entity, page.page_number))]
                if max(starts) - min(starts) > max(20, line_width * .8):
                    selected = _supported_duplicate_cluster(
                        candidates, page.page_number, line_width,
                    )
                    _record_issue(document, {"stage": "technical_requirement_duplicate_number",
                                             "page": page.page_number, "heading_id": heading.id,
                                             "item_number": number,
                                             "entity_ids": [entity.id for entity in candidates]})
                    if selected is None:
                        continue
                    candidates = selected
                by_text = defaultdict(list)
                for entity in candidates:
                    by_text[_normalized_text(entity.raw_text or "")].append(entity)
                variants = sorted(by_text.values(), key=lambda group: (
                    -len({str(entity.attributes.get("recognition_pass", entity.id)) for entity in group}),
                    -max(len(entity.raw_text or "") for entity in group),
                    -any(entity.kind == "raw_ocr" for entity in group),
                    min(entity.id for entity in group),
                ))
                primary = max(variants[0], key=lambda entity: (
                    len(entity.raw_text or ""), entity.kind == "raw_ocr"))
                longest = max(candidates, key=lambda entity: len(entity.raw_text or ""))
                if (_normalized_text(longest.raw_text or "").startswith(
                        _normalized_text(primary.raw_text or ""))
                        and len(longest.raw_text or "") >= len(primary.raw_text or "") * 1.5):
                    primary = longest
                next_axes = [
                    (candidate_box[1] + candidate_box[3]) / 2
                    for later_number, later_candidates in matches.items()
                    if later_number > number
                    for candidate in later_candidates
                    if (candidate_box := _box(candidate, page.page_number)) is not None
                ]
                continuations = ([] if vertical else _same_pass_continuations(
                    primary,
                    observations,
                    page.page_number,
                    min(next_axes) if next_axes else None,
                    tables,
                ))
                same_line_text, same_line_fragments, fragment_conflicts = (
                    _same_line_fragments(
                        primary, observations, page.page_number,
                        reject_tall_regions=fallback_heading,
                    )
                    if not vertical else (primary.raw_text or "", [], [])
                )
                if fragment_conflicts:
                    _record_issue(document, {
                        "stage": "technical_requirement_same_line_conflict",
                        "page": page.page_number,
                        "heading_id": heading.id,
                        "item_number": number,
                        "primary_entity_id": primary.id,
                        "conflicting_entity_ids": [
                            entity.id for entity in fragment_conflicts
                        ],
                    })
                evidence_entities = [
                    *candidates, *same_line_fragments, *fragment_conflicts, *continuations,
                ]
                if bridged_numbers and number >= min(bridged_numbers) and any(
                    note.attributes.get("item_number") == str(number)
                    and note.attributes.get("applicability_condition") == applicability
                    and any(candidate.id in note.evidence_ids for candidate in candidates)
                    for note in document.technical_notes
                ):
                    continue
                primary_box = _box(primary, page.page_number)
                post_terminal_cross_table_conflicts = {
                    entity.id for entity in fragment_conflicts
                    if same_line_text.rstrip().endswith("。")
                    and _crosses_into_adjacent_table(
                        primary_box, _box(entity, page.page_number), tables)
                }
                boxes = [_box(entity, page.page_number) for entity in evidence_entities
                         if entity.id not in post_terminal_cross_table_conflicts]
                raw_text = "\n".join(
                    [same_line_text, *(entity.raw_text or "" for entity in continuations)]
                )
                if _part_number_only((_numbered_requirement(raw_text) or
                                      _numbered_requirement(primary.raw_text or ""))[2]):
                    continue
                location = Location(page_number=page.page_number,
                                    bbox=(min(box[0] for box in boxes), min(box[1] for box in boxes),
                                          max(box[2] for box in boxes), max(box[3] for box in boxes)),
                                    localization_source="paddle_spotting" if primary.kind == "raw_ocr"
                                    else "paddle_layout")
                entity = Entity(
                    id=stable_id(document.source_sha256, "technical_notes",
                                 [page.page_number, heading.id, number]),
                    kind="technical_notes", raw_text=raw_text,
                    normalized_text=unicodedata.normalize("NFKC", raw_text),
                    attributes={"note_type": "explicit_technical_requirement_candidate",
                                "item_number": str(number), "heading_id": heading.id,
                                "applicability_condition": applicability,
                                "applicability_status": (
                                    "unresolved" if applicability != "unconditional"
                                    else "not_conditional"
                                ),
                                "scope_status": "unassigned", "eligible_for_training": False,
                                "multiline_merged": bool(continuations),
                                "continuation_evidence_ids": [entity.id for entity in continuations],
                                "same_line_merged": bool(same_line_fragments),
                                "same_line_evidence_ids": [
                                    entity.id for entity in same_line_fragments
                                ],
                                "same_line_fragment_conflicts": [
                                    {"entity_id": entity.id,
                                     "text": entity.raw_text,
                                     "recognition_pass": entity.attributes.get(
                                         "recognition_pass"
                                     )}
                                    for entity in fragment_conflicts
                                ],
                                "ocr_variant_disagreement": bool(
                                    len(variants) > 1 or fragment_conflicts
                                ),
                                "ocr_variants": [{"text": group[0].raw_text,
                                                  "evidence_ids": [entity.id for entity in group]}
                                                 for group in variants]},
                    locations=[location], evidence_ids=sorted({
                        heading.id, *(entity.id for entity in evidence_entities),
                    }),
                    raw_result_refs=sorted({ref for entity in [heading, *evidence_entities]
                                            for ref in entity.raw_result_refs}),
                    method="paddle_ocr_numbered_requirement_v1")
                if entity.id not in {note.id for note in document.technical_notes}:
                    document.technical_notes.append(entity)
                    added += 1
    document.configuration["ocr_technical_requirements"] = {
        "method": "heading_anchored_numbered_lines_v1",
        "candidates": sum(note.method == "paddle_ocr_numbered_requirement_v1"
                          for note in document.technical_notes),
        "ownership_verified": False, "eligible_for_training": False,
    }
    return added
