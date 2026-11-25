"""Bounded local inference transport and execution evidence."""

import base64
import hashlib
import json
import time
from pathlib import Path

import requests

from src.tools.drawing_extraction.visual_validation import parse_completion, validate_response


def response_usage(raw_response: str) -> dict[str, int]:
    """Extract stable token counters while preserving the complete raw response."""
    try:
        usage = json.loads(raw_response).get("usage", {})
    except (AttributeError, json.JSONDecodeError, TypeError):
        return {}
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {key: usage[key] for key in keys if isinstance(usage.get(key), int)}


def image_content(paths):
    return [{"type": "image_url", "image_url": {"url": "data:image/png;base64,"+
             base64.b64encode(Path(p).read_bytes()).decode()}} for p in paths]


def wait_for_model(session, endpoint, output):
    deadline = time.monotonic()+360
    while True:
        try:
            response = session.get(endpoint+"/models", timeout=5, allow_redirects=False)
            if response.status_code == 200:
                path = output / "served_models.json"
                if path.exists():
                    path = output / f"served_models_resume_{time.time_ns()}.json"
                with path.open("x") as stream:
                    stream.write(response.text)
                return
        except requests.RequestException:
            pass
        if time.monotonic() > deadline:
            raise RuntimeError("Model startup timed out")
        time.sleep(2)


def implementation_fingerprint():
    names = ("visual_runtime.py", "visual_validation.py", "visual_prompts.py", "visual_experiments.py")
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}


def execute_request(session, endpoint, payload, metadata, validator=lambda value: value, *, response_parser=validate_response,
                    request_timeout=(10, 180)):
    record = {**metadata, "status": "failed", "needs_review": True, "physical_part_id": None,
              "request_sha256": hashlib.sha256(json.dumps(payload).encode()).hexdigest()}
    start = time.monotonic()
    try:
        response = session.post(endpoint+"/chat/completions", json=payload, timeout=request_timeout, allow_redirects=False)
        record.update(http_status=response.status_code, raw_response=response.text)
        usage = response_usage(response.text)
        if usage:
            record["usage"] = usage
        record["proposal"] = validator(response_parser(response))
        record["status"] = "unverified_response"
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as error:
        record["error"] = str(error)
    record["elapsed_seconds"] = time.monotonic()-start
    return record


def validate_cached_record(record, request_sha256, validator):
    if record.get("request_sha256") != request_sha256:
        raise ValueError("Saved request does not match frozen payload")
    if record.get("status") == "unverified_response":
        if record.get("http_status") != 200:
            raise ValueError("Cached successful response has invalid HTTP status")
        proposal = validator(parse_completion(json.loads(record["raw_response"])))
        if proposal != record.get("proposal"):
            raise ValueError("Cached proposal differs from raw model response")
    elif record.get("status") != "failed":
        raise ValueError("Unexpected cached response status")
    return record


__all__ = [
    "execute_request", "image_content", "implementation_fingerprint", "response_usage",
    "validate_cached_record", "wait_for_model",
]
