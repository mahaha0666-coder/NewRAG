"""Run the official page pipeline in its isolated Paddle environment."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from importlib.metadata import version
import math
from threading import Lock
import time
from pathlib import Path
import sys
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
DEFAULT_MAX_NEW_TOKENS = 4096
MAX_RETRY_NEW_TOKENS = 16384
DEFAULT_MAX_PASS_ATTEMPTS = 3
DETAIL_WINDOWS = ((0, 0), (0.45, 0), (0, 0.45), (0.45, 0.45))
REGION_LEDGER_POLICY = "region_ledger_v6"
FULL_CONTENT_REGION_LEDGER_POLICY = "region_ledger_v11"
MAX_TARGETED_REGIONS = 12
MAX_GAP_REGIONS = 3
MAX_TARGETED_REFINEMENT_DEPTH = 2
CONTENT_SPOTTING_MAX_SIDE = 1200
MAX_CONTENT_SPOTTING_TILES = 24
MAX_CONTENT_SPOTTING_PASSES = 96
SHORT_TARGETED_MAX_SIDE = 1200
SHORT_TARGETED_NEW_TOKENS = 1024
REPETITIVE_TAIL_LENGTH = 160
REPETITIVE_LEAF_PADDING = 48
LAYOUT_NO_COMPLETION_LEAF_PADDING = 48
NO_RECORDED_COMPLETION_ERROR = "ValueError: Pipeline returned no recorded model completion"
LAYOUT_NO_CALLBACK_LIMIT_PER_TILE = 8
LAYOUT_RECOVERY_SOFT_SECONDS = 900
COARSE_LAYOUT_MIN_IMAGE_FRACTION = 0.75
COARSE_LAYOUT_ENGINEERING_TERMS = ("技术要求", "材料", "公差", "TECHNICAL", "MATERIAL")


def parallel_endpoint_ready(endpoint: str, model: str, *, timeout: float = 2.0) -> tuple[bool, str]:
    """Check that the secondary service advertises the exact requested model."""
    try:
        parsed = urlsplit(endpoint)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            return False, "invalid_endpoint"
        models_url = urlunsplit((parsed.scheme, parsed.netloc,
                                 parsed.path.rstrip("/") + "/models", "", ""))
        request = Request(models_url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return False, "models_request_failed"
            body = response.read(65537)
        if len(body) > 65536:
            return False, "invalid_models_response"
        payload = json.loads(body)
    except (OSError, ValueError):
        return False, "models_request_failed"
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return False, "invalid_models_response"
    if not any(isinstance(item, dict) and item.get("id") == model for item in models):
        return False, "model_id_mismatch"
    return True, "ready"


def available_parallel_endpoint(endpoint: str | None, model: str) -> str | None:
    if not endpoint:
        return None
    ready, reason = parallel_endpoint_ready(endpoint, model)
    if ready:
        return endpoint
    print(f"Secondary VLM unavailable ({reason}); using serial inference",
          file=sys.stderr, flush=True)
    return None


def retry_token_schedule(max_new_tokens: int, attempts: int) -> list[int]:
    """Increase generation budget within the 32k context-safe output cap."""
    if not 1024 <= max_new_tokens <= MAX_RETRY_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1024 and {MAX_RETRY_NEW_TOKENS}")
    if not 1 <= attempts <= 5:
        raise ValueError("attempts must be between 1 and 5")
    values = []
    for index in range(attempts):
        values.append(min(MAX_RETRY_NEW_TOKENS, max_new_tokens * (2 ** index)))
    return values


def repetitive_completion_tail(content: str | None) -> bool:
    """Flag only a long, low-diversity periodic tail in an incomplete response."""

    if not content or len(content) < REPETITIVE_TAIL_LENGTH:
        return False
    tail = content[-REPETITIVE_TAIL_LENGTH:]
    if len(set(tail)) > 3:
        return False
    return any(
        all(char == tail[index % period] for index, char in enumerate(tail))
        for period in range(1, 5)
    )


def collect_model_responses(futures, target: Path, records: list[dict]) -> list[str]:
    """Preserve termination metadata and suppress known-incomplete content."""
    results: list[str] = []
    for future in futures:
        response = future.result()
        path = target.with_name(target.stem + f"_completion_{len(records):04d}.json")
        raw = response.model_dump(mode="json")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(raw, stream, ensure_ascii=True, indent=2)
        choice = response.choices[0]
        guard = raw.get("stream_guard") or {}
        finish_reason = ("client_abort" if guard.get("status") == "aborted"
                         else choice.finish_reason)
        complete = finish_reason == "stop"
        content = choice.message.content
        usage = raw.get("usage") or {}
        records.append({
            "raw": path.name,
            "finish_reason": finish_reason,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "content_discarded": not complete,
            "repetitive_tail": (not complete and (
                guard.get("status") == "aborted" or repetitive_completion_tail(content))),
            **({"stream_guard": guard} if guard else {}),
        })
        results.append(content if complete else "")
    return results


def completion_summary(records: list[dict]) -> dict:
    """Return compact counters without rereading large raw response files."""
    finish_reasons = Counter(record["finish_reason"] for record in records)
    return {
        "count": len(records),
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "prompt_tokens": sum(record.get("prompt_tokens") or 0 for record in records),
        "completion_tokens": sum(record.get("completion_tokens") or 0 for record in records),
        "discarded_incomplete": sum(bool(record.get("content_discarded")) for record in records),
        **({"usage_complete": all(record.get("stream_guard", {}).get("usage_complete", True)
                                  for record in records)}
           if any("stream_guard" in record for record in records) else {}),
    }


def detail_passes(width: int, height: int, *, layout: bool) -> list[tuple[list[int], bool, str]]:
    """Build the four overlapping windows used for spotting or layout recovery."""
    prefix = "layout_detail" if layout else "detail"
    return [
        ([int(x0 * width), int(y0 * height),
          min(width, int((x0 + .55) * width)),
          min(height, int((y0 + .55) * height))], layout, f"{prefix}_{index}")
        for index, (x0, y0) in enumerate(DETAIL_WINDOWS)
    ]


def spotting_recovery_frames(frame: list[int], pass_id: str) -> list[tuple[list[int], bool, str]]:
    """Split one failed spotting tile into overlapping, page-coordinate tiles."""
    x0, y0, x1, y1 = frame
    return [([x0 + box[0], y0 + box[1], x0 + box[2], y0 + box[3]],
             False, f"{pass_id}_sub_{index}")
            for index, (box, _, _) in enumerate(detail_passes(x1 - x0, y1 - y0, layout=False), 1)]


def spotting_recovery_plan(width: int, height: int,
                           detail_results: list[tuple[list[int], bool, str, bool]]) -> dict:
    """A recovered page requires every original tile or all of its subtiles."""
    tiles = [{"pass_id": pass_id, "frame": frame, "status": "ok" if complete else "failed"}
             for frame, _, pass_id, complete in detail_results]
    expected = [(frame, layout, name)
                for frame, layout, name in detail_passes(width, height, layout=False)]
    covered = [(frame, layout, name) for frame, layout, name, _ in detail_results] == expected
    return {"full_page_frame": [0, 0, width, height],
            "strategy": "four_overlapping_spotting_tiles_with_one_level_of_refinement",
            "tiles": tiles, "complete": covered and all(tile["status"] == "ok" for tile in tiles)}


def should_recover_layout(layout_succeeded: bool, attempts: list[dict],
                          unmatched_part_numbers: int) -> bool:
    """Recover a missing full layout or a truncated layout with BOM gaps."""
    return (not layout_succeeded
            or (unmatched_part_numbers >= 2
                and any(attempt["status"] in {"retrying", "failed"}
                        for attempt in attempts)))


def validate_spotting_result(result: dict) -> None:
    """Reject responses the page adapter would otherwise silently truncate."""

    wrapped = result.get("res", result)
    if not isinstance(wrapped, dict):
        raise ValueError("Spotting response has no result object")
    spotting = wrapped.get("spotting_res") or []
    groups = [spotting] if isinstance(spotting, dict) else spotting
    if not isinstance(groups, list):
        raise ValueError("Spotting response has malformed groups")
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("Spotting response has malformed group")
        texts = group.get("rec_texts") or []
        polygons = group.get("rec_polys") or []
        if (not isinstance(texts, list) or not isinstance(polygons, list)
                or len(texts) != len(polygons)):
            raise ValueError("Spotting text and polygon counts differ")


def empty_layout_detector_result(result: dict, frame: list[int]) -> bool:
    """Recognize an official zero-block layout result that needs no VLM call."""

    if set(result) != {"res"} or not isinstance(result["res"], dict):
        return False
    wrapped = result["res"]
    settings = wrapped.get("model_settings")
    detection = wrapped.get("layout_det_res")
    width, height = wrapped.get("width"), wrapped.get("height")
    return bool(
        type(width) is int and type(height) is int
        and width == frame[2] - frame[0] and height == frame[3] - frame[1]
        and isinstance(settings, dict)
        and settings.get("use_layout_detection") is True
        and isinstance(detection, dict) and detection.get("boxes") == []
        and wrapped.get("parsing_res_list") == []
    )


def run_pass(
    *, pipeline, source, page_image: Path, target: Path, payload: dict, completions: list,
    frame: list[int], layout: bool, pass_id: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_attempts: int = DEFAULT_MAX_PASS_ATTEMPTS,
    first_max_new_tokens: int | None = None,
) -> bool:
    """Run one frame until every completion stops cleanly or retries are exhausted."""
    image_path = page_image
    if frame != [0, 0, source.width, source.height]:
        image_path = target.with_name(target.stem + f"_{pass_id}.png")
        source.crop(frame).save(image_path)
    try:
        token_schedule = retry_token_schedule(max_new_tokens, max_attempts)
        if first_max_new_tokens is not None:
            if not 1024 <= first_max_new_tokens < max_new_tokens:
                raise ValueError("first_max_new_tokens must be smaller than max_new_tokens")
            token_schedule.insert(0, first_max_new_tokens)
        for attempt_number, attempt_tokens in enumerate(token_schedule, 1):
            attempt_started = time.monotonic()
            before_completion = len(completions)
            local_results = []
            error = None
            empty_without_generation = False
            try:
                for result in pipeline.predict(str(image_path), use_layout_detection=layout,
                        prompt_label=None if layout else "spotting", max_new_tokens=attempt_tokens,
                        temperature=0, use_doc_orientation_classify=False,
                        use_doc_unwarping=False):
                    raw_result = result.json
                    if not isinstance(raw_result, dict):
                        raise ValueError("Pipeline returned malformed page result")
                    if not layout:
                        validate_spotting_result(raw_result)
                    local_results.append(raw_result)
                if not local_results:
                    raise ValueError("Pipeline returned no page result")
                if len(completions) == before_completion:
                    if (layout and len(local_results) == 1
                            and empty_layout_detector_result(local_results[0], frame)):
                        empty_without_generation = True
                    else:
                        raise ValueError("Pipeline returned no recorded model completion")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            new_completions = completions[before_completion:]
            for record in new_completions:
                record.update(pass_id=pass_id, frame=frame, layout=layout,
                              attempt_number=attempt_number, max_new_tokens=attempt_tokens)
            unfinished = [row for row in new_completions if row["finish_reason"] != "stop"]
            repetition_detected = any(row.get("repetitive_tail") for row in unfinished)
            attempt = {"pass_id": pass_id, "frame": frame, "layout": layout,
                       "attempt_number": attempt_number, "max_new_tokens": attempt_tokens,
                       "elapsed_seconds": round(time.monotonic() - attempt_started, 3)}
            if error is None and not unfinished:
                payload["results"].extend(local_results)
                payload["frames"].extend([frame] * len(local_results))
                payload.setdefault("pass_ids", []).extend([pass_id] * len(local_results))
                attempt.update(status="ok", result_count=len(local_results))
                if empty_without_generation:
                    source_white = source_frame_is_white(source, frame)
                    attempt.update(
                        completion_policy="zero_layout_blocks_no_vlm_generation",
                        source_frame_exact_white=source_white,
                    )
                    if not source_white:
                        payload.setdefault("layout_empty_detector_review_regions", []).append({
                            "pass_id": pass_id,
                            "frame": frame,
                            "source_image_ref": page_image.name,
                            "source_check": "every_rgb_pixel_exact_white",
                            "source_frame_exact_white": False,
                            "reason": "zero_layout_blocks_on_nonblank_source",
                            "status": "review_required",
                            "route_hard_constraint_eligible": False,
                        })
                payload["attempts"].append(attempt)
                return True
            can_retry = (attempt_number < len(token_schedule)
                         and (first_max_new_tokens is None or not repetition_detected)
                         and (first_max_new_tokens is None
                              or error is None or bool(unfinished)))
            status = "retrying" if can_retry else "failed"
            if error is not None:
                attempt.update(status=status,
                               error=error, discarded_result_count=len(local_results))
                if error == NO_RECORDED_COMPLETION_ERROR and layout and local_results:
                    refs = []
                    block_counts = []
                    for result_index, result in enumerate(local_results):
                        raw_path = target.with_name(
                            f"{target.stem}_{pass_id}_unverified_{attempt_number:02d}_"
                            f"{result_index:02d}.json"
                        )
                        with raw_path.open("x", encoding="utf-8") as stream:
                            json.dump(result, stream, ensure_ascii=True, indent=2)
                        refs.append(raw_path.name)
                        wrapped = result.get("res", result)
                        blocks = wrapped.get("parsing_res_list") if isinstance(wrapped, dict) else None
                        block_counts.append(len(blocks) if isinstance(blocks, list) else None)
                    attempt.update(
                        unverified_result_refs=refs,
                        unverified_layout_block_counts=block_counts,
                    )
            else:
                attempt.update(status=status,
                               result_count=len(local_results),
                               incomplete_completions=[row["raw"] for row in unfinished],
                               discarded_result_count=len(local_results))
            if repetition_detected:
                attempt["repetition_detected"] = True
            payload["attempts"].append(attempt)
            if not can_retry:
                reason = ("repetitive incomplete completion" if repetition_detected
                          else "incomplete completion after retries")
                payload["errors"].append({**attempt, "error": error or reason})
                return False
    finally:
        if image_path != page_image:
            image_path.unlink(missing_ok=True)


def adaptive_layout_tiles(
    width: int,
    height: int,
    *,
    max_side: int = 1800,
    overlap_fraction: float = 0.08,
) -> list[list[int]]:
    """Cover a failed full-page layout with size-driven, overlapping tiles."""

    if width <= 0 or height <= 0 or max_side < 256:
        raise ValueError("Invalid page or tile dimensions")
    columns = max(1, math.ceil(width / max_side))
    rows = max(1, math.ceil(height / max_side))
    tile_width = math.ceil(width / columns)
    tile_height = math.ceil(height / rows)
    overlap_x = int(round(tile_width * overlap_fraction))
    overlap_y = int(round(tile_height * overlap_fraction))
    frames = []
    for row in range(rows):
        for column in range(columns):
            x0 = max(0, column * tile_width - overlap_x)
            y0 = max(0, row * tile_height - overlap_y)
            x1 = min(width, (column + 1) * tile_width + overlap_x)
            y1 = min(height, (row + 1) * tile_height + overlap_y)
            frames.append([x0, y0, x1, y1])
    return frames


def content_frame(image, *, max_audit_side: int = 1800) -> list[int]:
    """Return a padded page-coordinate frame around visible content.

    Some source PDFs use a media box far larger than the actual drawing. Sending
    that whitespace to the layout model makes both the document and its text too
    small to parse. The returned frame never changes the document coordinate
    system; it only defines the image window sent to the model.
    """

    width, height = image.size
    scale = min(1.0, max_audit_side / max(width, height))
    audit_width = max(1, int(round(width * scale)))
    audit_height = max(1, int(round(height * scale)))
    gray = cv2.cvtColor(
        np.array(image.resize((audit_width, audit_height)).convert("RGB")),
        cv2.COLOR_RGB2GRAY,
    )
    mask = (gray < 245).astype(np.uint8)
    if not np.any(mask):
        return [0, 0, width, height]

    # Ignore isolated antialiasing specks while retaining thin dimensions and
    # frame lines. The projection threshold is intentionally only two pixels.
    columns = np.flatnonzero(mask.sum(axis=0) >= 2)
    rows = np.flatnonzero(mask.sum(axis=1) >= 2)
    if not len(columns) or not len(rows):
        return [0, 0, width, height]
    x0 = int(math.floor(columns[0] / scale))
    y0 = int(math.floor(rows[0] / scale))
    x1 = int(math.ceil((columns[-1] + 1) / scale))
    y1 = int(math.ceil((rows[-1] + 1) / scale))
    content_width = max(1, x1 - x0)
    content_height = max(1, y1 - y0)
    padding = max(16, int(round(max(content_width, content_height) * 0.02)))
    return [
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    ]


def outside_processing_ink_pixels(image, frame: list[int], *, max_audit_side: int = 1800) -> int:
    """Count dark pixels excluded by content framing at a bounded audit scale."""

    width, height = image.size
    scale = min(1.0, max_audit_side / max(width, height))
    audit_width = max(1, int(round(width * scale)))
    audit_height = max(1, int(round(height * scale)))
    gray = cv2.cvtColor(
        np.array(image.resize((audit_width, audit_height)).convert("RGB")),
        cv2.COLOR_RGB2GRAY,
    )
    ink = gray < 230
    inside = np.zeros_like(ink, dtype=np.uint8)
    x0, y0, x1, y1 = frame
    inside[
        max(0, int(math.floor(y0 * scale))):min(audit_height, int(math.ceil(y1 * scale))),
        max(0, int(math.floor(x0 * scale))):min(audit_width, int(math.ceil(x1 * scale))),
    ] = 1
    return int(np.count_nonzero(ink & (inside == 0)))


def tiles_within(frame: list[int], *, max_side: int = 1800) -> list[list[int]]:
    """Create adaptive page-coordinate tiles inside an arbitrary frame."""

    x0, y0, x1, y1 = frame
    return [
        [x0 + tile[0], y0 + tile[1], x0 + tile[2], y0 + tile[3]]
        for tile in adaptive_layout_tiles(x1 - x0, y1 - y0, max_side=max_side)
    ]


def content_spotting_tiles(frame: list[int]) -> list[list[int]]:
    """Cover visible content with overlapping crops at the render's native DPI."""

    return tiles_within(frame, max_side=CONTENT_SPOTTING_MAX_SIDE)


def targeted_refinement_frames(
    frame: list[int], depth: int, *, minimum_first_dimension: int = 512,
) -> list[list[int]]:
    """Return strictly smaller tiles for a truncated targeted pass.

    Two levels reduce the largest preflight title/BOM block from roughly 1,160
    pixels to at most 320 pixels. Returning no frames is an explicit stop
    condition, so a failed response cannot trigger an identical-input retry or
    unbounded recursion.
    """

    if not 0 <= depth < MAX_TARGETED_REFINEMENT_DEPTH:
        return []
    width = frame[2] - frame[0]
    height = frame[3] - frame[1]
    max_dimension = max(width, height)
    minimum_dimension = minimum_first_dimension if depth == 0 else 256
    if max_dimension <= minimum_dimension:
        return []
    if depth == 0:
        max_side = max(320, min(700, math.ceil(max_dimension / 2)))
    else:
        # Account for adaptive_layout_tiles' 8% overlap and keep the actual
        # second-level crops near or below 320 pixels.
        max_side = max(256, min(296, math.ceil(max_dimension / 2)))
    children = tiles_within(frame, max_side=max_side)
    if len(children) == 1 and children[0] == frame:
        return []
    return children


def expanded_targeted_leaf_frame(
    frame: list[int], page_size: tuple[int, int],
    padding: int = REPETITIVE_LEAF_PADDING,
) -> list[int]:
    """Expand a failed leaf within the original page's pixel coordinates."""

    width, height = page_size
    return [
        max(0, frame[0] - padding),
        max(0, frame[1] - padding),
        min(width, frame[2] + padding),
        min(height, frame[3] + padding),
    ]


def run_targeted_with_refinement(
    *, pipeline, full, page_image: Path, target: Path, payload: dict,
    completions: list, frame: list[int], pass_id: str, max_new_tokens: int,
    depth: int = 0, pass_budget: dict[str, int] | None = None,
    minimum_first_dimension: int = 512,
) -> dict:
    """Run a targeted pass and recover truncation with bounded smaller tiles."""

    if pass_budget is not None:
        lock = pass_budget.get("lock")
        if lock is not None:
            with lock:
                permitted = pass_budget["remaining"] > 0
                if permitted:
                    pass_budget["remaining"] -= 1
        else:
            permitted = pass_budget["remaining"] > 0
            if permitted:
                pass_budget["remaining"] -= 1
        if not permitted:
            return {
                "pass_id": pass_id, "frame": frame, "depth": depth,
                "complete": False, "children": [],
                "skipped_reason": "content_spotting_pass_budget_exhausted",
            }
    complete = run_pass(
        pipeline=pipeline,
        source=full,
        page_image=page_image,
        target=target,
        payload=payload,
        completions=completions,
        frame=frame,
        layout=False,
        pass_id=pass_id,
        max_new_tokens=min(max_new_tokens, 8192),
        max_attempts=1,
        first_max_new_tokens=(
            SHORT_TARGETED_NEW_TOKENS
            if (max(frame[2] - frame[0], frame[3] - frame[1]) <= SHORT_TARGETED_MAX_SIDE
                and max_new_tokens > SHORT_TARGETED_NEW_TOKENS)
            else None
        ),
    )
    node = {
        "pass_id": pass_id,
        "frame": frame,
        "depth": depth,
        "complete": complete,
        "children": [],
    }
    if complete:
        return node

    child_frames = targeted_refinement_frames(
        frame, depth, minimum_first_dimension=minimum_first_dimension,
    )
    repeated = any(
        attempt.get("pass_id") == pass_id and attempt.get("repetition_detected")
        for attempt in payload.get("attempts") or []
    )
    if not child_frames and repeated and pass_id.startswith("targeted_"):
        expanded = expanded_targeted_leaf_frame(frame, full.size)
        if expanded != frame:
            recovery_pass_id = f"{pass_id}_expanded_{REPETITIVE_LEAF_PADDING}"
            recovered = run_pass(
                pipeline=pipeline, source=full, page_image=page_image,
                target=target, payload=payload, completions=completions,
                frame=expanded, layout=False, pass_id=recovery_pass_id,
                max_new_tokens=min(max_new_tokens, 8192), max_attempts=1,
                first_max_new_tokens=(
                    SHORT_TARGETED_NEW_TOKENS
                    if max_new_tokens > SHORT_TARGETED_NEW_TOKENS else None
                ),
            )
            node["children"].append({
                "pass_id": recovery_pass_id, "frame": expanded,
                "depth": depth + 1, "complete": recovered, "children": [],
                "recovery_kind": "expanded_source_leaf",
                "failed_source_frame": frame,
            })
            node["complete"] = recovered
            if recovered:
                for error in payload["errors"]:
                    if error.get("pass_id") == pass_id:
                        error.update(resolved=True, recovered_by=[recovery_pass_id])
            return node

    for child_index, child_frame in enumerate(child_frames, start=1):
        node["children"].append(run_targeted_with_refinement(
            pipeline=pipeline,
            full=full,
            page_image=page_image,
            target=target,
            payload=payload,
            completions=completions,
            frame=child_frame,
            pass_id=f"{pass_id}_sub_{child_index:02d}",
            max_new_tokens=max_new_tokens,
            depth=depth + 1,
            pass_budget=pass_budget,
            minimum_first_dimension=minimum_first_dimension,
        ))
    complete = bool(node["children"]) and all(
        child["complete"] for child in node["children"]
    )
    node["complete"] = complete
    if complete:
        for error in payload["errors"]:
            if error.get("pass_id") == pass_id:
                error.update(
                    resolved=True,
                    recovered_by=[child["pass_id"] for child in node["children"]],
                )
    return node


def accepted_spotting_frames(node: dict) -> list[list[int]]:
    """Report only successful leaf crops; a failed parent has no accepted result."""

    if node["children"]:
        return [frame for child in node["children"]
                for frame in accepted_spotting_frames(child)]
    return [node["frame"]] if node["complete"] else []


def run_parallel_spotting_tasks(
    *, pipelines: tuple, full, page_image: Path, target: Path,
    payload: dict, completions: list, tasks: list[tuple[str, list[int]]],
    max_new_tokens: int, minimum_first_dimension: int = 512,
    pass_budget: dict | None = None,
) -> list[dict]:
    """Run independent crops on isolated lanes and publish them in task order."""

    def run_task(lane_index: int, pass_id: str, frame: list[int]):
        pipeline = pipelines[lane_index]
        task_target = target.with_name(f"{target.stem}_lane_{lane_index}_{pass_id}.json")
        task_payload = {"results": [], "frames": [], "pass_ids": [],
                        "attempts": [], "errors": [], "warnings": []}
        task_completions = []
        recognizer = pipeline.paddlex_pipeline.vl_rec_model
        recognizer._doc_vlm_genai_collect_responses = (
            lambda futures: collect_model_responses(futures, task_target, task_completions)
        )
        result = run_targeted_with_refinement(
            pipeline=pipeline, full=full, page_image=page_image,
            target=task_target, payload=task_payload,
            completions=task_completions, frame=frame, pass_id=pass_id,
            max_new_tokens=max_new_tokens, pass_budget=pass_budget,
            minimum_first_dimension=minimum_first_dimension,
        )
        return result, task_payload, task_completions

    with (ThreadPoolExecutor(max_workers=1) as first,
          ThreadPoolExecutor(max_workers=1) as second):
        lanes = (first, second)
        futures = [lanes[index % 2].submit(run_task, index % 2, pass_id, frame)
                   for index, (pass_id, frame) in enumerate(tasks)]
        results = []
        for future in futures:
            result, task_payload, task_completions = future.result()
            for key in ("results", "frames", "pass_ids", "attempts", "errors", "warnings"):
                payload[key].extend(task_payload[key])
            completions.extend(task_completions)
            results.append(result)
    return results


def source_frame_is_white(full, frame: list[int]) -> bool:
    """Only exact white RGB source pixels establish an empty layout region."""

    if full.mode != "RGB":
        return False
    return all(low == 255 for low, _high in full.crop(tuple(frame)).getextrema())


def layout_stop_reason(budget: dict) -> str | None:
    if budget["no_callback_count"] >= LAYOUT_NO_CALLBACK_LIMIT_PER_TILE:
        return "layout_no_callback_budget_exhausted"
    if time.monotonic() - budget["page_started"] >= LAYOUT_RECOVERY_SOFT_SECONDS:
        return "layout_soft_time_budget_exhausted"
    return None


def record_layout_no_callback(payload: dict, pass_id: str, budget: dict) -> None:
    error = next((row for row in reversed(payload["errors"])
                  if row.get("pass_id") == pass_id), None)
    if (error is not None and error.get("error") == NO_RECORDED_COMPLETION_ERROR
            and error.get("discarded_result_count", 0) > 0):
        budget["no_callback_count"] += 1
        budget["page_budget"]["total_no_callback_count"] += 1


def layout_recovery_counts(nodes: list[dict]) -> tuple[int, int]:
    skipped = blank = 0
    for node in nodes:
        skipped += bool(node.get("skipped_reason") or node.get("recovery_skipped_reason"))
        blank += node.get("recovery_kind") == "source_verified_blank"
        child_skipped, child_blank = layout_recovery_counts(node["children"])
        skipped += child_skipped
        blank += child_blank
    return skipped, blank


def run_layout_recovery_tile(
    *, pipeline, full, page_image: Path, target: Path, payload: dict,
    completions: list, frame: list[int], pass_id: str, max_new_tokens: int,
    depth: int = 0, budget: dict | None = None,
) -> dict:
    """Recover an incomplete layout response with smaller, source-density crops."""

    if budget is not None:
        if source_frame_is_white(full, frame):
            return {
                "pass_id": pass_id, "frame": frame, "depth": depth,
                "complete": True, "children": [],
                "recovery_kind": "source_verified_blank",
                "source_image_ref": page_image.name,
                "source_check": "every_rgb_pixel_exact_white",
                "pixel_count": (frame[2] - frame[0]) * (frame[3] - frame[1]),
                "nonwhite_pixel_count": 0,
            }
        stop_reason = layout_stop_reason(budget)
        if stop_reason is not None:
            payload["errors"].append({
                "pass_id": pass_id, "frame": frame, "status": "skipped",
                "error": stop_reason, "resolved": False,
            })
            return {"pass_id": pass_id, "frame": frame, "depth": depth,
                    "complete": False, "children": [], "skipped_reason": stop_reason}
    complete = run_pass(
        pipeline=pipeline, source=full, page_image=page_image,
        target=target, payload=payload, completions=completions,
        frame=frame, layout=True, pass_id=pass_id,
        max_new_tokens=max_new_tokens, max_attempts=1,
    )
    node = {"pass_id": pass_id, "frame": frame, "depth": depth,
            "complete": complete, "children": []}
    if budget is not None and not complete:
        record_layout_no_callback(payload, pass_id, budget)
    if complete:
        return node

    children = tiles_within(frame, max_side=800 if depth == 0 else 400) if depth < 2 else []
    if children and not (len(children) == 1 and children[0] == frame):
        node["children"] = [run_layout_recovery_tile(
            pipeline=pipeline, full=full, page_image=page_image, target=target,
            payload=payload, completions=completions, frame=child_frame,
            pass_id=f"{pass_id}_sub_{index:02d}",
            max_new_tokens=max_new_tokens, depth=depth + 1, budget=budget,
        ) for index, child_frame in enumerate(children, start=1)]
        node["complete"] = all(child["complete"] for child in node["children"])
        if node["complete"]:
            for error in payload["errors"]:
                if error.get("pass_id") == pass_id:
                    error.update(resolved=True,
                                 recovered_by=[child["pass_id"] for child in node["children"]])
        return node

    failed = next((error for error in reversed(payload["errors"])
                   if error.get("pass_id") == pass_id), None)
    if (failed is not None and failed.get("error") == NO_RECORDED_COMPLETION_ERROR
            and failed.get("discarded_result_count", 0) > 0):
        stop_reason = layout_stop_reason(budget) if budget is not None else None
        if stop_reason is not None:
            node["recovery_skipped_reason"] = stop_reason
            return node
        expanded = expanded_targeted_leaf_frame(
            frame, full.size, padding=LAYOUT_NO_COMPLETION_LEAF_PADDING,
        )
        if expanded != frame:
            recovery_pass_id = f"{pass_id}_expanded_{LAYOUT_NO_COMPLETION_LEAF_PADDING}"
            recovered = run_pass(
                pipeline=pipeline, source=full, page_image=page_image,
                target=target, payload=payload, completions=completions,
                frame=expanded, layout=True, pass_id=recovery_pass_id,
                max_new_tokens=max_new_tokens, max_attempts=1,
            )
            node["children"] = [{
                "pass_id": recovery_pass_id, "frame": expanded,
                "depth": depth + 1, "complete": recovered, "children": [],
                "recovery_kind": "expanded_no_completion_layout_leaf",
                "failed_source_frame": frame,
            }]
            node["complete"] = recovered
            if budget is not None and not recovered:
                record_layout_no_callback(payload, recovery_pass_id, budget)
            if recovered:
                failed.update(resolved=True, recovered_by=[recovery_pass_id])
    return node


def layout_regions(payload: dict, width: int, height: int) -> list[dict]:
    """Map Paddle layout blocks from crop coordinates back to page pixels."""

    regions = []
    for result, frame, pass_id in zip(
        payload.get("results") or [],
        payload.get("frames") or [],
        payload.get("pass_ids") or [],
    ):
        if not str(pass_id).startswith("layout") or len(frame) != 4:
            continue
        wrapped = result.get("res", result)
        for block_index, block in enumerate(wrapped.get("parsing_res_list") or []):
            bbox = block.get("block_bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                local = [float(value) for value in bbox]
            except (TypeError, ValueError):
                continue
            page_bbox = [
                max(0.0, min(float(width), local[0] + frame[0])),
                max(0.0, min(float(height), local[1] + frame[1])),
                max(0.0, min(float(width), local[2] + frame[0])),
                max(0.0, min(float(height), local[3] + frame[1])),
            ]
            if page_bbox[2] - page_bbox[0] < 4 or page_bbox[3] - page_bbox[1] < 4:
                continue
            content = str(block.get("block_content") or "")
            regions.append({
                "id": f"{pass_id}_block_{block_index:04d}",
                "source_pass_id": pass_id,
                "label": str(block.get("block_label") or "unknown"),
                "bbox": [round(value, 3) for value in page_bbox],
                "content_preview": content[:240],
                "content_length": len(content),
            })
    return regions


def coarse_layout_recovery_candidate(payload: dict, frame: list[int]) -> dict | None:
    """Find a stopped full layout that swallowed distinct engineering blocks."""

    full_results = [
        (index, result) for index, (pass_id, result) in enumerate(zip(
            payload.get("pass_ids") or [], payload.get("results") or [],
        )) if pass_id == "layout_full"
    ]
    if len(full_results) != 1:
        return None
    result_index, result = full_results[0]
    wrapped = result.get("res", result)
    blocks = wrapped.get("parsing_res_list") or []
    if not isinstance(blocks, list):
        return None
    images = [(index, block) for index, block in enumerate(blocks)
              if isinstance(block, dict) and block.get("block_label") == "image"]
    if len(images) != 1:
        return None
    image_index, image = images[0]
    bbox = image.get("block_bbox")
    width, height = frame[2] - frame[0], frame[3] - frame[1]
    if (not isinstance(bbox, list) or len(bbox) != 4 or width <= 0 or height <= 0):
        return None
    try:
        image_box = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if not (0 <= image_box[0] < image_box[2] <= width
            and 0 <= image_box[1] < image_box[3] <= height):
        return None
    image_fraction = ((image_box[2] - image_box[0])
                      * (image_box[3] - image_box[1]) / (width * height))
    if image_fraction < COARSE_LAYOUT_MIN_IMAGE_FRACTION:
        return None

    nested = []
    for index, block in enumerate(blocks):
        if (index == image_index or not isinstance(block, dict)
                or block.get("block_label") not in {
                    "text", "paragraph_title", "figure_title", "table", "aside_text",
                } or not str(block.get("block_content") or "").strip()):
            continue
        box = block.get("block_bbox")
        if not isinstance(box, list) or len(box) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(value) for value in box)
        except (TypeError, ValueError):
            continue
        if (image_box[0] <= x0 < x1 <= image_box[2]
                and image_box[1] <= y0 < y1 <= image_box[3]
                and all(_bbox_iou([x0, y0, x1, y1], prior["box"]) < .5
                        for prior in nested)):
            nested.append({"block_index": index, "box": [x0, y0, x1, y1],
                           "label": block["block_label"],
                           "content": str(block["block_content"])})
    text = "\n".join([str(image.get("block_content") or ""),
                      *(block["content"] for block in nested)]).upper()
    if (len(nested) < 2
            or not (any(term.upper() in text for term in COARSE_LAYOUT_ENGINEERING_TERMS)
                    or any(block["label"] == "table" for block in nested))):
        return None
    return {
        "reason": "single_large_image_contains_distinct_engineering_blocks",
        "full_result_index": result_index,
        "image_block_index": image_index,
        "image_area_fraction": round(image_fraction, 6),
        "nested_block_indices": [block["block_index"] for block in nested],
        "source_frame": frame,
    }


def _expanded_bbox(bbox: list[float], width: int, height: int, padding: int = 12) -> list[int]:
    return [
        max(0, int(math.floor(bbox[0])) - padding),
        max(0, int(math.floor(bbox[1])) - padding),
        min(width, int(math.ceil(bbox[2])) + padding),
        min(height, int(math.ceil(bbox[3])) + padding),
    ]


def _bbox_iou(left: list[int], right: list[int]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _bbox_coverage(inner: list[int], outer: list[int]) -> float:
    """Return the fraction of ``inner`` covered by ``outer``."""

    x0, y0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x1, y1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area = max(0, inner[2] - inner[0]) * max(0, inner[3] - inner[1])
    return intersection / area if area else 0.0


def build_region_ledger(
    image,
    regions: list[dict],
    processing_frame: list[int] | None = None,
    *,
    spotting_frames: list[list[int]] | None = None,
    include_coverage_fallback: bool = True,
    include_all_views: bool = False,
) -> tuple[list[dict], dict]:
    """Select bounded high-risk regions and explicitly account for residual ink."""

    width, height = image.size
    processing_frame = processing_frame or [0, 0, width, height]
    px0, py0, px1, py1 = processing_frame
    processing = image.crop(processing_frame)
    processing_width, processing_height = processing.size
    scale = min(1.0, 1600.0 / max(processing_width, processing_height))
    small_width = max(1, int(round(processing_width * scale)))
    small_height = max(1, int(round(processing_height * scale)))
    gray = cv2.cvtColor(
        np.array(processing.resize((small_width, small_height)).convert("RGB")),
        cv2.COLOR_RGB2GRAY,
    )
    ink = (gray < 230).astype(np.uint8)
    covered = np.zeros_like(ink)
    for region in regions:
        x0, y0, x1, y1 = _expanded_bbox(region["bbox"], width, height, padding=8)
        x0, x1 = max(px0, x0), min(px1, x1)
        y0, y1 = max(py0, y0), min(py1, y1)
        if x0 >= x1 or y0 >= y1:
            continue
        cv2.rectangle(
            covered,
            (int(round((x0 - px0) * scale)), int(round((y0 - py0) * scale))),
            (int(round((x1 - px0) * scale)), int(round((y1 - py0) * scale))),
            1,
            thickness=-1,
        )

    proposals = []
    image_regions = sorted(
        (region for region in regions if region["label"] == "image"),
        key=lambda region: (
            (region["bbox"][2] - region["bbox"][0])
            * (region["bbox"][3] - region["bbox"][1])
        ),
        reverse=True,
    )[:MAX_TARGETED_REGIONS if include_all_views else 2]
    for region in regions:
        content = region["content_preview"]
        reasons = []
        if region["label"] == "table":
            reasons.append("table")
        if "技术要求" in content or "TECHNICAL" in content.upper():
            reasons.append("technical_requirement")
        if region in image_regions:
            reasons.append("major_view" if region in image_regions[:2] else "view_detail")
        bbox = region["bbox"]
        if (bbox[0] >= px0 + processing_width * 0.55
                and bbox[1] >= py0 + processing_height * 0.55):
            reasons.append("title_or_bom_quadrant")
        if reasons:
            proposals.append({
                "region_id": region["id"],
                "frame": _expanded_bbox(bbox, width, height),
                "reasons": sorted(set(reasons)),
                "source": "layout_block",
            })

    spotted = np.zeros_like(ink)
    for x0, y0, x1, y1 in spotting_frames or []:
        x0, x1 = max(px0, x0), min(px1, x1)
        y0, y1 = max(py0, y0), min(py1, y1)
        if x0 >= x1 or y0 >= y1:
            continue
        cv2.rectangle(
            spotted,
            (int(round((x0 - px0) * scale)), int(round((y0 - py0) * scale))),
            (int(round((x1 - px0) * scale)), int(round((y1 - py0) * scale))),
            1,
            thickness=-1,
        )

    residual = ink & (covered == 0) & (spotted == 0)
    grouping = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    grouped = cv2.dilate(residual, grouping, iterations=1)
    _, _, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
    minimum_area = max(40, int(small_width * small_height * 0.00025))
    gaps = []
    for x, y, component_width, component_height, area in stats[1:]:
        if area < minimum_area or component_width < 4 or component_height < 4:
            continue
        gaps.append({
            "frame": [
                max(px0, px0 + int(math.floor(x / scale)) - 12),
                max(py0, py0 + int(math.floor(y / scale)) - 12),
                min(px1, px0 + int(math.ceil((x + component_width) / scale)) + 12),
                min(py1, py0 + int(math.ceil((y + component_height) / scale)) + 12),
            ],
            "ink_component_area": int(area),
        })
    gaps.sort(key=lambda row: row["ink_component_area"], reverse=True)
    ink_pixels = int(np.count_nonzero(ink))
    covered_ink = int(np.count_nonzero(ink & covered))
    layout_coverage = covered_ink / ink_pixels if ink_pixels else 1.0
    if include_coverage_fallback and (layout_coverage < 0.98 or len(gaps) > MAX_GAP_REGIONS):
        # A line drawing often forms one connected component, so connected
        # components alone cannot prove page coverage. Add conditional spotting
        # windows that cover the actual content frame, not the PDF media box.
        coverage_proposals = []
        for index, frame in enumerate(tiles_within(processing_frame), start=1):
            coverage_proposals.append({
                "region_id": f"coverage_tile_{index:03d}",
                "frame": frame,
                "reasons": ["low_layout_ink_coverage"],
                "source": "content_coverage_fallback",
            })
        # Coverage is a gate, while high-risk blocks are optional refinements.
        # Reserve the target budget for complete content coverage first.
        proposals = coverage_proposals + proposals
    for index, gap in enumerate(gaps, start=1):
        proposals.append({
            "region_id": f"uncovered_ink_{index:03d}",
            "frame": gap["frame"],
            "reasons": ["uncovered_ink"],
            "source": "ink_gap_detector",
        })

    selected = []
    for proposal in proposals:
        if proposal["frame"][2] - proposal["frame"][0] < 32:
            continue
        if proposal["frame"][3] - proposal["frame"][1] < 32:
            continue
        covering = next(
            (row for row in selected
             if row["source"] == "content_coverage_fallback"
             and _bbox_coverage(proposal["frame"], row["frame"]) >= 0.98),
            None,
        )
        if covering is not None and proposal["source"] == "ink_gap_detector":
            covering.setdefault("covered_region_ids", []).append(proposal["region_id"])
            covering["reasons"] = sorted(set(covering["reasons"] + proposal["reasons"]))
            continue
        duplicate = next(
            (row for row in selected if _bbox_iou(row["frame"], proposal["frame"]) >= 0.70),
            None,
        )
        if duplicate is not None:
            duplicate["reasons"] = sorted(set(duplicate["reasons"] + proposal["reasons"]))
            duplicate.setdefault("merged_region_ids", []).append(proposal["region_id"])
            continue
        selected.append(proposal)
        if len(selected) >= MAX_TARGETED_REGIONS:
            break

    target_covered = np.zeros_like(ink)
    for row in selected:
        x0, y0, x1, y1 = row["frame"]
        cv2.rectangle(
            target_covered,
            (int(round((x0 - px0) * scale)), int(round((y0 - py0) * scale))),
            (int(round((x1 - px0) * scale)), int(round((y1 - py0) * scale))),
            1,
            thickness=-1,
        )
    remaining = ink & (covered == 0) & (target_covered == 0) & (spotted == 0)
    _, _, remaining_stats, _ = cv2.connectedComponentsWithStats(remaining, 8)
    remaining_gaps = []
    for x, y, component_width, component_height, area in remaining_stats[1:]:
        if area < minimum_area or component_width < 4 or component_height < 4:
            continue
        remaining_gaps.append({
            "frame": [
                max(px0, px0 + int(math.floor(x / scale)) - 12),
                max(py0, py0 + int(math.floor(y / scale)) - 12),
                min(px1, px0 + int(math.ceil((x + component_width) / scale)) + 12),
                min(py1, py0 + int(math.ceil((y + component_height) / scale)) + 12),
            ],
            "ink_component_area": int(area),
        })
    ledger = {
        "schema_version": "page_region_coverage.v3" if spotting_frames is not None
                          else "page_region_coverage.v2",
        "page_size": [width, height],
        "processing_frame": processing_frame,
        "processing_area_ratio": round(
            processing_width * processing_height / (width * height), 6
        ),
        "layout_region_count": len(regions),
        "layout_regions": regions,
        "ink_pixel_count_at_audit_scale": ink_pixels,
        "layout_covered_ink_pixels": covered_ink,
        "layout_ink_coverage": round(covered_ink / ink_pixels, 6) if ink_pixels else 1.0,
        "uncovered_ink_component_count": len(gaps),
        "selected_region_count": len(selected),
        "selected_regions": selected,
        "target_covered_ink_pixels": int(np.count_nonzero(ink & target_covered)),
        "final_covered_ink_pixels": int(np.count_nonzero(
            ink & ((covered | target_covered | spotted) > 0)
        )),
        "final_ink_coverage": round(
            np.count_nonzero(ink & ((covered | target_covered | spotted) > 0)) / ink_pixels, 6
        ) if ink_pixels else 1.0,
        "unselected_coverage_gaps": remaining_gaps,
        "all_detected_gaps_accounted": True,
    }
    if spotting_frames is not None:
        ledger["spotting_covered_ink_pixels"] = int(np.count_nonzero(ink & spotted))
        ledger["spotting_ink_coverage"] = round(
            np.count_nonzero(ink & spotted) / ink_pixels, 6
        ) if ink_pixels else 1.0
    return selected, ledger


def run_region_ledger_page(
    *,
    pipeline,
    full,
    page_image: Path,
    target: Path,
    payload: dict,
    completions: list,
    max_new_tokens: int,
    full_content_spotting: bool = False,
    parallel_pipelines: tuple | None = None,
) -> None:
    width, height = full.size
    processing_frame = content_frame(full)
    layout_recovery_tokens = min(8192, max_new_tokens * 2)
    layout_budget = {"page_started": time.monotonic(), "total_no_callback_count": 0}
    full_layout = run_pass(
        pipeline=pipeline,
        source=full,
        page_image=page_image,
        target=target,
        payload=payload,
        completions=completions,
        frame=processing_frame,
        layout=True,
        pass_id="layout_full",
        max_new_tokens=max_new_tokens,
        max_attempts=1,
    )
    if full_content_spotting and not full_layout:
        record_layout_no_callback(payload, "layout_full", {
            "no_callback_count": 0, "page_budget": layout_budget,
        })
    coarse_layout = (coarse_layout_recovery_candidate(payload, processing_frame)
                     if full_layout else None)
    layout_tiles = []
    if not full_layout or coarse_layout is not None:
        layout_frames = tiles_within(processing_frame, max_side=1400)
        if len(layout_frames) == 1 and layout_frames[0] == processing_frame:
            layout_frames = tiles_within(
                processing_frame,
                max_side=max(256, min(800, math.ceil(max(width, height) / 2))),
            )
        for index, frame in enumerate(layout_frames, start=1):
            pass_id = f"layout_tile_{index:03d}"
            tile_budget = ({"page_started": layout_budget["page_started"],
                            "page_budget": layout_budget, "no_callback_count": 0}
                           if full_content_spotting else None)
            layout_tiles.append(run_layout_recovery_tile(
                pipeline=pipeline,
                full=full,
                page_image=page_image,
                target=target,
                payload=payload,
                completions=completions,
                frame=frame,
                pass_id=pass_id,
                max_new_tokens=layout_recovery_tokens,
                budget=tile_budget,
            ))
            if tile_budget is not None:
                layout_budget.setdefault("per_tile_no_callback_counts", {})[pass_id] = (
                    tile_budget["no_callback_count"]
                )
        if layout_tiles and all(tile["complete"] for tile in layout_tiles):
            for error in payload["errors"]:
                if error.get("pass_id") == "layout_full":
                    error.update(resolved=True,
                                 recovered_by=[tile["pass_id"] for tile in layout_tiles])
    layout_elapsed_seconds = round(time.monotonic() - layout_budget["page_started"], 3)
    skipped_layout_tiles, verified_blank_tiles = layout_recovery_counts(layout_tiles)
    stopped_subtrees = sum(
        layout_recovery_counts([tile])[0] > 0 for tile in layout_tiles
    )

    content_tiles = content_spotting_tiles(processing_frame) if full_content_spotting else []
    content_budget = {"remaining": MAX_CONTENT_SPOTTING_PASSES}
    content_results = []
    content_started = time.monotonic()
    if full_content_spotting:
        if parallel_pipelines is not None:
            content_budget["lock"] = Lock()
            content_results = run_parallel_spotting_tasks(
                pipelines=parallel_pipelines, full=full, page_image=page_image,
                target=target, payload=payload, completions=completions,
                tasks=[(f"content_{index:03d}", frame)
                       for index, frame in enumerate(content_tiles[:MAX_CONTENT_SPOTTING_TILES], 1)],
                max_new_tokens=max_new_tokens, pass_budget=content_budget,
            )
            content_results.extend(
                {"pass_id": f"content_{index:03d}", "frame": frame,
                 "depth": 0, "complete": False, "children": [],
                 "skipped_reason": "content_spotting_tile_limit_exceeded"}
                for index, frame in enumerate(content_tiles[MAX_CONTENT_SPOTTING_TILES:],
                                               MAX_CONTENT_SPOTTING_TILES + 1)
            )
        else:
            for index, frame in enumerate(content_tiles, start=1):
                pass_id = f"content_{index:03d}"
                if index > MAX_CONTENT_SPOTTING_TILES:
                    content_results.append({
                        "pass_id": pass_id, "frame": frame, "depth": 0,
                        "complete": False, "children": [],
                        "skipped_reason": "content_spotting_tile_limit_exceeded",
                    })
                    continue
                content_results.append(run_targeted_with_refinement(
                    pipeline=pipeline,
                    full=full,
                    page_image=page_image,
                    target=target,
                    payload=payload,
                    completions=completions,
                    frame=frame,
                    pass_id=pass_id,
                    max_new_tokens=max_new_tokens,
                    pass_budget=content_budget,
                ))
    content_elapsed_seconds = round(time.monotonic() - content_started, 3)
    accepted_frames = [frame for result in content_results
                       for frame in accepted_spotting_frames(result)]
    content_complete = all(result["complete"] for result in content_results)

    regions = layout_regions(payload, width, height)
    selected, ledger = build_region_ledger(
        full, regions, processing_frame,
        spotting_frames=accepted_frames if full_content_spotting else None,
        include_coverage_fallback=not full_content_spotting,
        include_all_views=full_content_spotting,
    )
    task_results = []
    targeted_started = time.monotonic()
    if parallel_pipelines is not None:
        task_results = run_parallel_spotting_tasks(
            pipelines=parallel_pipelines, full=full, page_image=page_image,
            target=target, payload=payload, completions=completions,
            tasks=[(f"targeted_{index:03d}", region["frame"])
                   for index, region in enumerate(selected, 1)],
            max_new_tokens=max_new_tokens,
            minimum_first_dimension=256 if full_content_spotting else 512,
        )
        for result, region in zip(task_results, selected):
            result.update({"region_id": region["region_id"],
                           "reasons": region["reasons"]})
    else:
        for index, region in enumerate(selected, start=1):
            pass_id = f"targeted_{index:03d}"
            result = run_targeted_with_refinement(
                pipeline=pipeline,
                full=full,
                page_image=page_image,
                target=target,
                payload=payload,
                completions=completions,
                frame=region["frame"],
                pass_id=pass_id,
                max_new_tokens=max_new_tokens,
                minimum_first_dimension=256 if full_content_spotting else 512,
            )
            result.update({
                "region_id": region["region_id"],
                "reasons": region["reasons"],
            })
            task_results.append(result)
    payload["stage_timings_seconds"] = {
        "layout": layout_elapsed_seconds,
        "content_spotting": content_elapsed_seconds,
        "targeted_regions": round(time.monotonic() - targeted_started, 3),
    }
    ledger.update({
        "full_layout_complete": full_layout,
        "layout_empty_detector_review_regions": list(
            payload.get("layout_empty_detector_review_regions") or []
        ),
        "coarse_layout_recovery": (
            {**coarse_layout, "triggered": True,
             "complete": bool(layout_tiles) and all(tile["complete"] for tile in layout_tiles)}
            if coarse_layout is not None else {"triggered": False}
        ),
        "layout_tiles": layout_tiles,
        "targeted_results": task_results,
        "incomplete_targeted_region_count": sum(not row["complete"] for row in task_results),
    })
    if full_content_spotting:
        ledger["layout_recovery_budget"] = {
            "per_tile_no_callback_limit": LAYOUT_NO_CALLBACK_LIMIT_PER_TILE,
            "page_soft_seconds": LAYOUT_RECOVERY_SOFT_SECONDS,
            "elapsed_seconds": layout_elapsed_seconds,
            "total_no_callback_count": layout_budget["total_no_callback_count"],
            "per_tile_no_callback_counts": layout_budget.get(
                "per_tile_no_callback_counts", {}
            ),
            "skipped_tile_count": skipped_layout_tiles,
            "stopped_subtree_count": stopped_subtrees,
            "verified_blank_tile_count": verified_blank_tiles,
        }
    excluded_ink = 0
    if full_content_spotting:
        excluded_ink = outside_processing_ink_pixels(full, processing_frame)
        ledger["content_spotting"] = {
            "strategy": "adaptive_overlapping_content_tiles_with_bounded_refinement",
            "planned_tile_count": len(content_tiles),
            "completed_tile_count": sum(result["complete"] for result in content_results),
            "incomplete_tile_count": sum(not result["complete"] for result in content_results),
            "accepted_leaf_frame_count": len(accepted_frames),
            "pass_count": MAX_CONTENT_SPOTTING_PASSES - content_budget["remaining"],
            "max_initial_tiles": MAX_CONTENT_SPOTTING_TILES,
            "max_passes": MAX_CONTENT_SPOTTING_PASSES,
            "base_tile_side": CONTENT_SPOTTING_MAX_SIDE,
            "overlap_fraction": 0.08,
            "outside_processing_ink_pixel_count_at_audit_scale": excluded_ink,
            "complete": content_complete,
            "tiles": content_results,
        }
        ledger["all_detected_gaps_accounted"] = bool(
            content_complete and not excluded_ink and not ledger["unselected_coverage_gaps"]
        )
    payload["coverage_ledger"] = ledger
    has_usable_results = bool(payload["results"])
    has_gap = bool(ledger["unselected_coverage_gaps"])
    has_incomplete = bool(
        any(not error.get("resolved") for error in payload["errors"])
        or ledger["layout_empty_detector_review_regions"]
        or ledger["incomplete_targeted_region_count"]
        or (coarse_layout is not None
            and not ledger["coarse_layout_recovery"]["complete"])
        or (full_content_spotting and not content_complete)
        or (full_content_spotting and excluded_ink > 0)
    )
    payload["status"] = "failed" if not has_usable_results else "partial" if has_gap or has_incomplete else "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="PaddleOCR-VL-1.6")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-pass-attempts", type=int, default=1)
    parser.add_argument("--full-content-spotting", action="store_true")
    parser.add_argument("--stream-guard", action="store_true",
                        help="Abort repetitive streams and recover with native-resolution crops")
    parser.add_argument("--parallel-endpoint",
                        help="Second VLM endpoint for independent spotting crops")
    args = parser.parse_args()
    if not 1024 <= args.max_new_tokens <= MAX_RETRY_NEW_TOKENS:
        parser.error(f"--max-new-tokens must be between 1024 and {MAX_RETRY_NEW_TOKENS}")
    if args.max_pass_attempts != 1:
        parser.error("Region ledger policy forbids deterministic content retries")
    if (args.parallel_endpoint is not None
            and args.parallel_endpoint.rstrip("/") == args.endpoint.rstrip("/")):
        parser.error("--parallel-endpoint must differ from --endpoint")
    parallel_endpoint = available_parallel_endpoint(args.parallel_endpoint, args.model)
    from paddleocr import PaddleOCRVL
    pipeline_options = dict(
        pipeline_version="v1.6", device="cpu", vl_rec_backend="vllm-server",
        vl_rec_server_url=args.endpoint, vl_rec_api_model_name=args.model,
        vl_rec_max_concurrency=1, use_doc_orientation_classify=False,
        use_doc_unwarping=False, use_ocr_for_image_block=True, use_queues=False,
    )
    pipeline = PaddleOCRVL(**pipeline_options)
    parallel_pipeline = (PaddleOCRVL(**{**pipeline_options,
                                       "vl_rec_server_url": parallel_endpoint})
                         if parallel_endpoint else None)
    # This small adapter targets the installed, pinned official collector API.
    if version("paddlex") != "3.7.2":
        raise RuntimeError("Completion recorder requires tested PaddleX 3.7.2")
    recognizer = pipeline.paddlex_pipeline.vl_rec_model
    if not callable(getattr(recognizer, "_doc_vlm_genai_collect_responses", None)):
        raise RuntimeError("Official completion collector API has changed")
    if parallel_pipeline is not None and not callable(getattr(
        parallel_pipeline.paddlex_pipeline.vl_rec_model,
        "_doc_vlm_genai_collect_responses", None,
    )):
        raise RuntimeError("Official completion collector API has changed")
    if args.stream_guard:
        from src.tools.drawing_extraction.stream_guard import install_stream_guard
        install_stream_guard(recognizer)
        if parallel_pipeline is not None:
            install_stream_guard(parallel_pipeline.paddlex_pipeline.vl_rec_model)
    for page in json.loads(args.manifest.read_text()):
        target = Path(page["output"])
        if target.exists():
            raise FileExistsError(target)
        from PIL import Image
        payload = {"status": "ok", "results": [], "frames": [], "pass_ids": [],
                   "errors": [], "warnings": [], "attempts": [], "generation_policy": {
                       "version": (FULL_CONTENT_REGION_LEDGER_POLICY
                                   if args.full_content_spotting else REGION_LEDGER_POLICY),
                       "max_new_tokens": args.max_new_tokens,
                       "layout_recovery_max_new_tokens": min(8192, args.max_new_tokens * 2),
                       "layout_recovery_max_depth": 2,
                       "full_layout_passes": 1,
                       "full_spotting_passes": 0,
                       "fixed_detail_passes": 0,
                       "max_targeted_regions": MAX_TARGETED_REGIONS,
                       "max_gap_regions": MAX_GAP_REGIONS,
                       "max_targeted_refinement_depth": MAX_TARGETED_REFINEMENT_DEPTH,
                       "short_targeted_max_side": SHORT_TARGETED_MAX_SIDE,
                       "short_targeted_first_max_new_tokens": SHORT_TARGETED_NEW_TOKENS,
                       "repetitive_tail_period_limit": 4,
                       "repetitive_tail_length": REPETITIVE_TAIL_LENGTH,
                       "repetitive_leaf_padding_pixels": REPETITIVE_LEAF_PADDING,
                       "layout_no_completion_leaf_padding_pixels": LAYOUT_NO_COMPLETION_LEAF_PADDING,
                       "coarse_layout_min_image_fraction": COARSE_LAYOUT_MIN_IMAGE_FRACTION,
                       "coarse_layout_recovery_requires_nested_engineering_blocks": True,
                       "max_pass_attempts": 1,
                       "spotting_lanes": 2 if parallel_pipeline is not None else 1,
                       "attempt_timing_kind": "monotonic_model_request_seconds",
                       "layout_recovery_condition": "full_layout_incomplete_only",
                       "empty_layout_detector_policy": (
                           "strict_zero_blocks_review_nonblank_source"
                       ),
                       "content_frame_policy": "visible_ink_bbox_with_page_coordinate_back_mapping",
                       "target_condition": "layout high-risk blocks plus conditional content coverage",
                       "incomplete_completion_content_discarded": True}}
        if args.full_content_spotting:
            payload["generation_policy"].update({
                "content_spotting_strategy": "adaptive_overlapping_tiles",
                "content_spotting_base_tile_side": CONTENT_SPOTTING_MAX_SIDE,
                "content_spotting_max_initial_tiles": MAX_CONTENT_SPOTTING_TILES,
                "content_spotting_max_passes": MAX_CONTENT_SPOTTING_PASSES,
                "target_condition": "all layout views and high-risk blocks plus residual ink gaps",
            })
        completions = []
        if args.stream_guard:
            from src.tools.drawing_extraction.stream_guard import POLICY
            payload["generation_policy"]["stream_guard"] = POLICY
        recognizer._doc_vlm_genai_collect_responses = lambda futures: collect_model_responses(futures, target, completions)
        with Image.open(page["image"]) as opened:
            full = opened.convert("RGB")
            run_region_ledger_page(
                pipeline=pipeline,
                full=full,
                page_image=Path(page["image"]),
                target=target,
                payload=payload,
                completions=completions,
                max_new_tokens=args.max_new_tokens,
                full_content_spotting=args.full_content_spotting,
                parallel_pipelines=(pipeline, parallel_pipeline)
                    if parallel_pipeline is not None else None,
            )
        payload["completions"] = completions
        payload["completion_summary"] = completion_summary(completions)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, indent=2)
        print(target, payload["status"], flush=True)


if __name__ == "__main__":
    main()
