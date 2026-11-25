"""Paths and file identity shared by native extraction and PDF presentation."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEM = "12502370454"
OUT = ROOT / "outputs/dwg_pdf_visualization_20261031" / STEM
EVIDENCE = ROOT / "outputs/dwg_data_audit_20261031/cases" / STEM

__all__ = ["EVIDENCE", "OUT", "ROOT", "STEM", "digest", "work_dir"]


def work_dir(output: Path) -> Path:
    """Derived vector inputs, manifests and logs; raw evidence stays upstream."""
    return output / "evidence"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
