"""Isolate invalid native objects and disambiguate literal CAD dictionary keys.

The original reader bytes are never modified. A non-finite numeric observation
quarantines its entire object, retaining the original array index and token
offsets. It is never replaced with a plausible coordinate or value.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from research.dwg_route_validation_20261015.duplicate_audit import SIDECAR, lossless_projection
from research.dwg_route_validation_20261015.native_relation_features import FORBIDDEN
from research.dwg_route_validation_20261015.parser_probe import handle_value

__all__ = ["load_native", "table_input"]

# Quoted strings are matched first: a literal note containing "-nan" or "inf"
# must stay untouched. All remaining JSON syntax is still strictly validated.
TOKENS = re.compile(
    rb'(?P<string>"(?:[^"\\]|\\.)*")'
    rb'|(?P<constant>(?<![\w.])[+-]?(?:infinity|nan|inf)(?![\w.]))'
    rb'|(?P<number>-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)',
    re.IGNORECASE,
)


def load_native(raw: bytes) -> tuple[dict[str, Any], list[dict], list[dict]]:
    """Use the strict loader; on known non-finite numbers isolate whole objects."""
    try:
        parsed, duplicates = lossless_projection(raw)
        return parsed, duplicates, []
    except (ValueError, UnicodeError) as original_error:
        prefix = f"__batch_invalid_number_{hashlib.sha256(raw).hexdigest()}_"
        if prefix.encode() in raw:
            raise ValueError("Source collides with quarantine marker") from original_error
        tokens: dict[str, dict] = {}

        def substitute(match: re.Match[bytes]) -> bytes:
            literal = match.group()
            if match.lastgroup == "string":
                return literal
            invalid = match.lastgroup == "constant"
            if match.lastgroup == "number" and any(char in literal for char in (b".", b"e", b"E")):
                invalid = not math.isfinite(float(literal))
            if not invalid:
                return literal
            marker = f"{prefix}{len(tokens)}__"
            tokens[marker] = {"raw_token": literal.decode("ascii"),
                              "byte_start": match.start(), "byte_end": match.end()}
            return json.dumps(marker).encode("ascii")

        diagnostic_input = TOKENS.sub(substitute, raw)
        if not tokens:
            raise original_error
        parsed, duplicates = lossless_projection(diagnostic_input)

    def markers(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set().union(*(markers(key) | markers(child) for key, child in value.items()))
        if isinstance(value, list):
            return set().union(*(markers(child) for child in value))
        return {value} if isinstance(value, str) and value in tokens else set()

    # An invalid header, class declaration or outer metadata cannot be safely
    # isolated as one drawing object. No recovery is attempted in that case.
    if markers({key: value for key, value in parsed.items() if key != "OBJECTS"}):
        raise ValueError("Non-finite native metadata outside OBJECTS cannot be isolated")
    quarantined, consumed = [], set()
    for ordinal, obj in enumerate(parsed["OBJECTS"]):
        invalid = markers(obj)
        if not invalid:
            continue
        consumed.update(invalid)
        quarantined.append({"object_index": ordinal, "json_pointer": f"#/OBJECTS/{ordinal}",
                            "status": "whole_object_quarantined_not_used_for_geometry_or_fields",
                            "reason": "native_reader_emitted_nonfinite_numeric_observation",
                            "invalid_numeric_tokens": sorted((tokens[key] for key in invalid),
                                                             key=lambda token: token["byte_start"])})
        # Keep source JSON pointers stable. No native identity, coordinate,
        # field, or relationship from this object is passed to consumers.
        parsed["OBJECTS"][ordinal] = {"__batch_quarantined_object_index__": ordinal}
    if consumed != set(tokens):
        raise ValueError("Non-finite token could not be attributed to a native object")
    return parsed, duplicates, quarantined


def table_input(raw: dict[str, Any]) -> tuple[dict[str, Any], list[dict]]:
    """Represent verified DICTIONARY item names as strings, not feature keys.

    A native dictionary can literally contain the CAD name ``label``. The
    frozen feature guard still checks every other key and value. Only entries
    with the native DICTIONARY shape and valid handle references are projected;
    no supervision or arbitrary mapping is removed from the guard's input.
    """
    objects, receipt = list(raw["OBJECTS"]), []
    for ordinal, obj in enumerate(objects):
        items = obj.get("items")
        repeated = items.get(SIDECAR, {}) if isinstance(items, dict) else {}
        if (obj.get("object") != "DICTIONARY" or obj.get("type") != 42
                or not isinstance(items, dict)
                or not FORBIDDEN.intersection(set(items) | set(repeated))):
            continue
        if "items" in obj.get(SIDECAR, {}):
            raise ValueError("Repeated native dictionary items require a separate review")
        entries = []
        for key in dict.fromkeys([*items, *repeated]):
            if key == SIDECAR:
                continue
            refs = repeated[key] if key in repeated else [items[key]]
            if not isinstance(key, str) or not isinstance(refs, list) or any(
                    not isinstance(ref, list) or handle_value(ref) is None for ref in refs):
                raise ValueError("Native dictionary items must be literal names and handle references")
            if key in repeated:
                entries.append({"native_name": key, "native_handle_reference_occurrences": refs,
                                "status": "repeated_native_name_no_reference_selected"})
            else:
                entries.append({"native_name": key, "native_handle_reference": refs[0]})
        objects[ordinal] = {**obj, "items": entries}
        receipt.append({"json_pointer": f"#/OBJECTS/{ordinal}/items", "native_handle": obj["handle"],
                        "native_object_type": "DICTIONARY", "entry_count": len(entries),
                        "repeated_names_preserved_without_selection": len(repeated),
                        "reason": "literal_CAD_entry_names_are_not_route_supervision_fields",
                        "original_entries_retained_in_native_response": True})
    return {**raw, "OBJECTS": objects}, receipt
