"""Bounded original-density OCR requests with retained source transforms."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openai import APIError, OpenAI

from .stream_guard import RepetitionGuard

MAX_PIXELS = 1003520
OVERLAP = 0.12


def tile_boxes(width: int, height: int, max_pixels: int = MAX_PIXELS) -> list[list[int]]:
    """Bound aligned processor pixels, without resampling the source image."""
    if min(width, height) <= 0 or max_pixels < 28 * 28:
        raise ValueError("Invalid image dimensions or pixel budget")
    edge = max(28, int(math.sqrt(max_pixels)) // 28 * 28)
    tile_w = min(width, edge)
    tile_h = min(height, max_pixels // (math.ceil(tile_w / 28) * 28) // 28 * 28)
    if height <= edge:
        tile_h = height
        tile_w = min(width, max_pixels // (math.ceil(tile_h / 28) * 28) // 28 * 28)

    def starts(length: int, extent: int) -> list[int]:
        result = list(range(0, max(1, length - extent + 1), max(1, int(extent * (1 - OVERLAP)))))
        if result[-1] != max(0, length - extent):
            result.append(max(0, length - extent))
        return result

    return [[x, y, min(width, x + tile_w), min(height, y + tile_h)]
            for y in starts(height, tile_h) for x in starts(width, tile_w)]


def read_tile(image: np.ndarray, *, endpoint: str, output: Path,
              task: str = "spotting", timeout_seconds: float = 90) -> dict[str, Any]:
    """One attempt only; the region scheduler owns retries and acceptance."""
    if task not in {"spotting", "ocr"} or len(tile_boxes(image.shape[1], image.shape[0])) != 1:
        raise ValueError("Tile must fit the processor pixel budget")
    if not 0 < timeout_seconds <= 90:
        raise ValueError("Request timeout must be between zero and 90 seconds")
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Unable to encode source crop")
    png_bytes = png.tobytes()
    output.with_suffix(".png").write_bytes(png_bytes)
    started = time.monotonic()
    result: dict[str, Any] = {
        "status": "incomplete", "content": "", "raw_content": "",
        "image_size": [image.shape[1], image.shape[0]],
        "image_sha256": hashlib.sha256(png_bytes).hexdigest(),
        "task": task, "max_pixels": MAX_PIXELS, "max_tokens": 4096,
        "finish_reason": None, "confidence": None,
    }
    guard = RepetitionGuard()
    parts: list[str] = []
    try:
        with OpenAI(base_url=endpoint, api_key="local", timeout=timeout_seconds, max_retries=0) as client:
            with client.chat.completions.create(
                model="PaddleOCR-VL-1.6", temperature=0, max_tokens=4096,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")}},
                    {"type": "text", "text": "Spotting:" if task == "spotting" else "OCR:"},
                ]}], stream=True, stream_options={"include_usage": True},
                extra_body={"skip_special_tokens": task != "spotting",
                            "mm_processor_kwargs": {"min_pixels": 112896, "max_pixels": MAX_PIXELS}},
            ) as stream:
                for chunk in stream:
                    if chunk.usage is not None:
                        result["usage"] = chunk.usage.model_dump()
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta.content or ""
                    parts.append(delta)
                    if choice.finish_reason is not None:
                        result["finish_reason"] = choice.finish_reason
                    reason = guard.feed(delta)
                    if reason and result["finish_reason"] is None:
                        result.update(finish_reason="client_abort", review_reason=reason)
                        break
        if result["finish_reason"] == "stop":
            result.update(status="completed", content="".join(parts))
        else:
            result.setdefault("review_reason", "recognition_incomplete")
    except APIError as error:
        result.update(status="failed", review_reason="recognition_request_failed",
                      diagnostic=f"{type(error).__name__}: {error}")
    result.update(raw_content="".join(parts), elapsed_seconds=round(time.monotonic() - started, 3))
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
