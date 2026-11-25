"""Presentation evidence for matching native view titles to their source marks."""
from __future__ import annotations

import gzip
import hashlib
import logging
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

from research.dwg_route_study_20261105.scene_input import SceneInput
from src.tools.drawing_extraction.native_dwg_batch.projection import load_native

_LIMIT = 256 * 1024 * 1024
_CAPTION = re.compile(r"([A-Z]{1,2})(?:[-－—–]\1|向)")
_VIEW_TITLE = re.compile(r"主视图|正视图|俯视图|仰视图|左视图|右视图|后视图|轴测图|等轴测图|局部放大图")
_DETAIL = re.compile(r"I|II|III|IV|V|VI|VII|VIII|IX|X")
_SCALE = re.compile(r"(?:[1-9]\d*(?:\.\d+)?|0\.\d*[1-9]\d*):(?:[1-9]\d*(?:\.\d+)?|0\.\d*[1-9]\d*)")


def _clean(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split()).upper()


def _possible_title(text: str) -> bool:
    return bool(_CAPTION.fullmatch(text) or _VIEW_TITLE.fullmatch(text) or _DETAIL.fullmatch(text))


_LOG = logging.getLogger(__name__)


def point(value) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2 and all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in value[:2]
    ):
        if len(value) > 2 and (not isinstance(value[2], (int, float)) or not abs(value[2]) < 1e-6):
            return None
        return list(value[:2])
    return None


def _extract(raw: bytes):
    document, _, _ = load_native(raw)
    # Many parts have no cross-view titles. Avoid building their full insert index.
    if not any(_possible_title(_clean(str(obj.get(field, ""))))
               for obj in document["OBJECTS"] if obj.get("entity") in {"TEXT","MTEXT"}
               for field in ("text","text_value")):
        return []
    scene = SceneInput(document)
    marks = []
    for handle, obj in scene.index.items():
        kind = obj.get("entity")
        if kind not in {"TEXT", "MTEXT", "LEADER", "UNKNOWN_ENT", "XTCUTSYMBOL"}:
            continue
        if obj.get("extrusion", [0, 0, 1]) != [0, 0, 1]:
            continue
        if not scene.modelspace(handle) or not scene.print_state(handle)[0]:
            continue
        ref = format(handle, "X")
        if kind in {"UNKNOWN_ENT", "XTCUTSYMBOL"}:
            # The section symbol's decoded identity and geometry are checked by the consumer.
            marks.append({"role": "proxy", "ref": ref})
        elif kind == "LEADER":
            pts = scene.value(handle,"points")
            if scene.value(handle,"arrowhead_on") == 1 and isinstance(pts,list) and len(pts)==2:
                a,b = map(point,pts)
                if a and b and a != b:
                    marks.append({"role":"arrow","ref":ref,"tip":a,"tail":b})
        else:
            text = scene.value(handle, "text" if kind == "MTEXT" else "text_value")
            if not isinstance(text,str):
                continue
            label = _clean(text)
            loc = point(scene.value(handle,"ins_pt"))
            height = scene.value(handle,"text_height" if kind == "MTEXT" else "height")
            if not loc or not isinstance(height,(int,float)) or not 0 < height < math.inf:
                continue
            base = {"ref": ref, "point": loc, "height": height, "raw": text.strip()}
            if _VIEW_TITLE.fullmatch(label):
                marks.append({**base, "role": "view_title", "label": label})
            elif _DETAIL.fullmatch(label):
                marks.append({**base, "role": "detail_index", "key": label})
            elif _SCALE.fullmatch(label):
                marks.append({**base, "role": "scale", "label": label})
            match = _CAPTION.fullmatch(label)
            if match or re.fullmatch("[A-Z]{1,2}",text.strip().upper()):
                marks.append({"role": "caption" if match else "letter", "ref":ref,
                              "key":match[1] if match else label, "label":label,
                              "point":loc,"height":height,"raw":text.strip()})
    return marks


@lru_cache(maxsize=12)
def _read_verified(path: Path, expected_sha: str, identity: tuple):
    with gzip.open(path, "rb") as stream:
        raw = stream.read(_LIMIT + 1)
    if len(raw) > _LIMIT or hashlib.sha256(raw).hexdigest() != expected_sha:
        return []
    return _extract(raw)


def read_view_marks(directory: Path | None, response_sha: str | None, symbols: list) -> list:
    if not directory or not response_sha:
        return []
    path = directory / "native_response.json.gz"
    try:
        stat = path.stat()
        rows = _read_verified(path, response_sha, (stat.st_ino, stat.st_size, stat.st_ctime_ns))
    except FileNotFoundError:
        return []
    except (OSError, EOFError, ValueError, KeyError, TypeError, RecursionError) as error:
        _LOG.info("View marks unavailable (%s)", type(error).__name__)
        return []
    verified = {row["ref"] for row in rows if row["role"] == "proxy"}
    result = [row for row in rows if row["role"] != "proxy"]
    for symbol in symbols:
        ref = symbol["source_ref"].get("native_handle")
        if ref not in verified or symbol.get("native_kind") != "XTCUTSYMBOL":
            continue
        texts = symbol.get("texts", [])
        if len(texts) != 2:
            continue
        keys = [t["text"].strip().upper() for t in texts]
        pts = [point(t.get("point")) for t in texts]
        if keys[0] == keys[1] and re.fullmatch("[A-Z]{1,2}",keys[0]) and all(pts):
            # Each end of a native cut has a shared corner between its cut
            # stroke and direction stroke. Use only an unambiguous pair.
            endpoints = Counter()
            for path in symbol.get("paths", []):
                line = path.get("points", [])
                if path.get("native_kind") == "POLYLINE" and len(line) == 2:
                    for vertex in line:
                        xy = point(vertex)
                        if xy:
                            endpoints[tuple(xy)] += 1
            corners = [list(xy) for xy,n in endpoints.items() if n == 2]
            result.append({"role":"cut","ref":ref,"key":keys[0],"points":pts,
                           "axis":corners if len(corners) == 2 else None})
    return result
