"""Atomic, compact batch artifacts and content identities."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

__all__ = ["digest", "fingerprint", "read_json", "save_json", "save_csv", "save_gzip", "pack_file"]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False, prefix=".writing-") as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def save_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=path.parent,
                                     delete=False, prefix=".writing-") as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def save_gzip(path: Path, value: Any) -> None:
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    temporary = path.with_name("." + path.name + ".writing")
    temporary.write_bytes(gzip.compress(data, compresslevel=3, mtime=0))
    temporary.replace(path)


def pack_file(source: Path, destination: Path) -> None:
    """Keep the parser's exact bytes; compression never repairs its JSON."""
    import shutil

    temporary = destination.with_name("." + destination.name + ".writing")
    with source.open("rb") as inp, temporary.open("wb") as output:
        with gzip.GzipFile(fileobj=output, mode="wb", compresslevel=3, mtime=0, filename="") as packed:
            shutil.copyfileobj(inp, packed)
    temporary.replace(destination)
