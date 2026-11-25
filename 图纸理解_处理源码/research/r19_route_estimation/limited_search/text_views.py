"""Deterministic, label-free text views shared by training and inference."""

from __future__ import annotations

import hashlib
import json
import re

__all__ = ["input_digest", "text_views"]


def input_digest(row: dict) -> str:
    return hashlib.sha256(json.dumps({key: row.get(key, "") for key in
        ("base_text", "evidence_text", "source_sha256", "analysis_sha256")},
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def text_views(row: dict) -> list[str]:
    base = row["base_text"].replace("\\#", "#")
    base = re.sub(r"(?<!\d)\d{10,14}(?!\d)", " ", base)
    base = re.sub(r"[A-Za-z][A-Za-z0-9.]*[#＃][A-Za-z0-9.-]+", " ", base)
    lines = list(dict.fromkeys(line.strip() for line in base.splitlines() if line.strip()))
    semantic = [line for line in lines if re.search(r"[\u4e00-\u9fff]{2,}", line)]
    notes = re.sub(r"\[[^\]]*\]", " ", row.get("evidence_text", ""))
    return ["\n".join(semantic), notes, "\n".join(lines)]
