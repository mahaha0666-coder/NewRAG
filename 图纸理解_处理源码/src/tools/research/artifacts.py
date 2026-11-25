"""Small, dependency-free helpers for immutable experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

__all__ = ["digest", "read_json", "contained_path", "verify_manifest", "write_new_json"]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def contained_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Expected a contained relative path: {name}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes its root: {name}")
    return path


def verify_manifest(root: Path) -> dict[str, str]:
    manifest = read_json(root / "checksums.json")
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError(f"Empty or invalid checksums: {root}")
    for name, expected in manifest.items():
        if digest(contained_path(root, name)) != expected:
            raise ValueError(f"Frozen artifact changed: {root.name}/{name}")
    return manifest


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # CAD readers can retain undecodable source bytes as lone surrogate code
    # points. Escape them in JSON instead of dropping bytes or leaving an empty
    # artifact after a UnicodeEncodeError; valid UTF-8 output remains identical.
    payload = serialized.encode("utf-8", errors="backslashreplace")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
