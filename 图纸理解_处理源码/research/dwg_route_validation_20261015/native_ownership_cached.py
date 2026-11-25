"""Call-local field-evidence cache around the unchanged ownership algorithm.

The frozen builder reads field evidence; it mutates graph nodes/edges instead.
Its slot-reference branch creates a new evidence dictionary before selecting an
array member. Reusing container evidence therefore avoids copying whole arrays
per member. The original function's code object is executed with private globals;
neither its module globals nor the input are patched. Returned evidence is local
to one invocation and must remain read-only, as required by its record digest.
"""

from __future__ import annotations

from types import FunctionType
from typing import Any

from research.dwg_route_validation_20261015 import native_cad_ownership


def build_ownership_graph(parsed: dict[str, Any], *, source_sha256: str,
                          response_sha256: str) -> dict[str, Any]:
    """Execute the frozen builder verbatim with a fresh, isolated field cache."""
    original = native_cad_ownership.build_ownership_graph
    field_evidence = original.__globals__["_field_evidence"]
    cache: dict[tuple[int, str, str], tuple[dict, dict]] = {}

    def cached_evidence(obj: dict, field: str, pointer: str) -> dict:
        key = id(obj), field, pointer
        if key not in cache:
            # Retaining obj prevents identity reuse during this invocation.
            cache[key] = obj, field_evidence(obj, field, pointer)
        return cache[key][1]

    private_globals = {**original.__globals__, "_field_evidence": cached_evidence}
    builder = FunctionType(original.__code__, private_globals, original.__name__,
                           original.__defaults__, original.__closure__)
    builder.__kwdefaults__ = original.__kwdefaults__
    try:
        return builder(parsed, source_sha256=source_sha256, response_sha256=response_sha256)
    finally:
        cache.clear()
