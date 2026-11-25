"""Fresh primary-reader requests with retained images and termination metadata."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from urllib.request import Request, urlopen

import cv2
import numpy as np


def read_region(image: np.ndarray, *, task: str, endpoint: str, output: Path) -> dict[str, Any]:
    if task not in {"table", "spotting", "ocr"}:
        raise ValueError("Unsupported recognition task")
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output.with_suffix(".png")), image)
    encoded = base64.b64encode(output.with_suffix(".png").read_bytes()).decode("ascii")
    payload = {
        "model": "PaddleOCR-VL-1.6", "temperature": 0, "max_tokens": 8192,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
            {"type": "text", "text": {"table": "Table Recognition:", "spotting": "Spotting:", "ocr": "OCR:"}[task]},
        ]}],
        "mm_processor_kwargs": {"min_pixels": 112896, "max_pixels": 1605632},
        "skip_special_tokens": task != "spotting",
    }
    started = time.monotonic()
    request = Request(endpoint.rstrip("/") + "/chat/completions",
                      data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        raw = json.load(response)
    choice = raw["choices"][0]
    complete = choice.get("finish_reason") == "stop"
    result = {"task": task, "elapsed_seconds": round(time.monotonic() - started, 3),
              "image_size": [image.shape[1], image.shape[0]], "response": raw,
              "status": "completed" if complete else "incomplete",
              "content": choice["message"]["content"] if complete else ""}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if task == "table" and "<fcel>" in result["content"]:
        repo = Path(__file__).resolve().parents[3]
        subprocess.run([str(repo / ".venv-paddle-document/bin/python"),
                        str(repo / "scripts/parts_reading_repair/convert_table.py"), str(output)],
                       check=True, capture_output=True, timeout=60)
        result = json.loads(output.read_text(encoding="utf-8"))
    elif task == "table":
        result["table_html"] = result["content"]
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def spotting_tokens(content: str, size: tuple[int, int]) -> list[dict[str, Any]]:
    # Paddle's location tokens use the same 0..1000 frame as its official adapter.
    width, height = size
    tokens = []
    last_end = 0
    for block in re.finditer(r"(?:<\|LOC_\d+\|>\s*)+", content):
        values = [int(value) for value in re.findall(r"<\|LOC_(\d+)\|>", block.group())]
        text = re.sub(r"<\|[^>]*\|>", "", content[last_end:block.start()]).strip()
        last_end = block.end()
        if len(values) != 8:
            continue
        xs = [value * width / 1000 for value in values[::2]]
        ys = [value * height / 1000 for value in values[1::2]]
        if text and all(0 <= value <= 1000 for value in values):
            tokens.append({"text": text, "bbox": [min(xs), min(ys), max(xs), max(ys)]})
    return tokens
