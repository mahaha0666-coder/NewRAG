"""Shared local-model identity, bounded requests and complete JSON parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from src.tools.drawing_extraction.visual_runtime import image_content
from src.tools.research.artifacts import digest

from .prompts import SYSTEM


def payload(
    model: str,
    prompt: str,
    paths: list[Path],
    tokens: int,
    *,
    schema: dict[str, Any] | None = None,
    image_ids: list[str] | None = None,
) -> dict[str, Any]:
    if image_ids is not None and (
        len(image_ids) != len(paths)
        or any(not isinstance(value, str) or not value.strip() for value in image_ids)
        or len(set(image_ids)) != len(image_ids)
    ):
        raise ValueError("Image IDs must uniquely match the supplied image paths")
    content = [{"type": "text", "text": prompt}]
    for index, block in enumerate(image_content(paths), start=1):
        caption = (
            f"图片标识 image_id={image_ids[index - 1]}，与图片清单中的同名项对应："
            if image_ids is not None
            else f"图片顺序 {index}，对应图片清单第 {index} 项："
        )
        content.extend(
            [
                {"type": "text", "text": caption},
                block,
            ]
        )
    result = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if schema is not None:
        result["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "source_evidence_review", "schema": schema},
        }
    return result


def model_identity(
    endpoint: str, session: requests.Session, model: str
) -> dict[str, Any]:
    url = urlparse(endpoint)
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "localhost", "::1"}
        or url.username
        or url.password
    ):
        raise ValueError(
            "This source-bearing review accepts loopback HTTP endpoints only"
        )
    response = session.get(endpoint + "/models", timeout=(3, 8), allow_redirects=False)
    response.raise_for_status()
    served = next((row for row in response.json()["data"] if row["id"] == model), None)
    if served is None:
        raise ValueError("Requested model is not served by this endpoint")
    root = Path(served["root"])
    identities = {}
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "chat_template.jinja",
    ):
        if (root / name).is_file():
            identities[name] = digest(root / name)
    return {"endpoint": endpoint, "served": served, "configuration_sha256": identities}


def parse_json_response(response: requests.Response) -> dict[str, Any]:
    """Parse a complete JSON object; domain schemas impose their own fields."""
    if response.status_code != 200:
        raise ValueError(f"Model HTTP {response.status_code}")
    value = response.json()
    if value["choices"][0]["finish_reason"] != "stop":
        raise ValueError("Truncated response")
    content = value["choices"][0]["message"]["content"]
    if (
        isinstance(content, str)
        and content.strip().startswith("```json\n")
        and content.strip().endswith("\n```")
    ):
        content = content.strip()[8:-4]
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("Response must be a JSON object")
    return result


def parse_response(response: requests.Response) -> dict[str, Any]:
    """Keep the existing review-required contract for drawing interpretation."""
    result = parse_json_response(response)
    if result.get("needs_review") is not True:
        raise ValueError("Response must be a review-required JSON object")
    return result
