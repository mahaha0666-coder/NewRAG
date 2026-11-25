"""Admit native observations through explicit modelspace paths and print flags.

CAD reachability and global layer flags qualify a document observation. They do
not prove a printed PDF match, revision, physical part, or manufacturing scope.
All excluded observations remain in the immutable source document and audit.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
from typing import Any

from research.dwg_route_study_20261105.observations import label_free
from research.dwg_route_validation_20261015.native_dimension_display import strict_handle
from research.dwg_route_validation_20261015.native_layer_roles import _field_evidence
from research.dwg_route_validation_20261015.parser_probe import object_kind

MAX_PATHS = 10_000
MAX_DEPTH = 16
_MISSING = object()


class SceneInput:
    """A bounded, call-local membership index; no full geometry graph expansion."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.objects = raw["OBJECTS"]
        self.index: dict[int, dict] = {}
        self.ordinals: dict[int, int] = {}
        self.fields: dict[tuple[int, str], Any] = {}
        self.parents: dict[int, set[int]] = defaultdict(set)
        self.paths: dict[int, list[dict]] = defaultdict(list)
        self.diagnostics: Counter[str] = Counter()
        version = raw.get("FILEHEADER", {}).get("version", "")
        self.layer_bitmask_supported = (
            isinstance(version, str) and version.startswith("AC")
            and version[2:].isdigit() and int(version[2:]) >= 1015
        )
        for ordinal, obj in enumerate(self.objects):
            handle = strict_handle(obj.get("handle"), reference=False)
            if handle is None:
                continue
            if handle in self.index:
                raise ValueError("Duplicate native scene identity")
            self.index[handle], self.ordinals[handle] = obj, ordinal
        # Reverse membership is used to detect an implicit modelspace flag that
        # contradicts an explicitly supplied non-space block member list.
        for handle, obj in self.index.items():
            if object_kind(obj) == "BLOCK_HEADER":
                members = self.value(handle, "entities")
                if isinstance(members, list):
                    for ref in members:
                        member = strict_handle(ref, reference=True, origin=handle)
                        if member is not None:
                            self.parents[member].add(handle)
        self._expand()

    def value(self, handle: int, key: str) -> Any:
        cache_key = handle, key
        if cache_key not in self.fields:
            obj = self.index[handle]
            evidence = _field_evidence(obj, key, f"#/OBJECTS/{self.ordinals[handle]}")
            self.fields[cache_key] = (
                evidence["raw_value"] if evidence["present_in_projection"]
                and not evidence["conflicting"] else _MISSING
            )
        return self.fields[cache_key]

    def reference(self, handle: int, key: str) -> int | None:
        value = self.value(handle, key)
        return strict_handle(value, reference=True, origin=handle) if value is not _MISSING else None

    def modelspace(self, handle: int) -> bool:
        if self.value(handle, "entmode") != 2:
            return False
        owners = set(self.parents[handle])
        if self.value(handle, "ownerhandle") is not _MISSING:
            owner = self.reference(handle, "ownerhandle")
            if owner is None:
                return False
            owners.add(owner)
        return all(
            owner in self.index and object_kind(self.index[owner]) == "BLOCK_HEADER"
            and isinstance(self.value(owner, "name"), str)
            and self.value(owner, "name").casefold() == "*model_space" for owner in owners
        )

    def print_state(self, handle: int) -> tuple[bool, str, int | None]:
        invisible = self.value(handle, "invisible")
        if invisible is _MISSING:
            return False, "entity_visibility_unknown", None
        if type(invisible) is not int or invisible != 0:
            return False, "entity_hidden_or_visibility_invalid", None
        layer = self.reference(handle, "layer")
        if layer not in self.index or object_kind(self.index[layer]) != "LAYER":
            return False, "layer_reference_unknown", layer
        flags = self.value(layer, "flag0")
        if not self.layer_bitmask_supported or type(flags) is not int or flags < 0:
            return False, "layer_flags_unknown", layer
        # Pinned LibreDWG dwg.spec, LAYER, R2000+: 1=frozen, 2=off,
        # 4=frozen only in NEW viewports, 8=locked, 16=plotflag.
        # Do not confuse internal flag0 with DXF flag70, or lock with hiding.
        if flags & 1:
            return False, "layer_globally_frozen", layer
        if flags & 2:
            return False, "layer_off", layer
        if not flags & 16:
            return False, "layer_nonplot", layer
        return True, "global_print_flags_allow_candidate", layer

    def _definition(self, handle: int) -> int | None:
        kind = object_kind(self.index[handle]) or ""
        field = "block_header" if kind == "INSERT" else "block" if kind.startswith("DIMENSION_") else None
        if field is None:
            return None
        block = self.reference(handle, field)
        if block not in self.index or object_kind(self.index[block]) != "BLOCK_HEADER":
            self.diagnostics["unresolved_instance_definition"] += 1
            return None
        # An external definition's body is not established by local membership.
        if any(self.value(block, key) != 0 for key in ("blkisxref", "xrefoverlaid", "is_xref_dep")):
            self.diagnostics["external_or_unknown_definition"] += 1
            return None
        if self.value(block, "xref_pname") != "":
            self.diagnostics["external_or_unknown_definition"] += 1
            return None
        return block

    def _expand(self) -> None:
        queue: deque[tuple[int, tuple[int, ...], tuple[int, ...]]] = deque()
        for handle, obj in self.index.items():
            kind = object_kind(obj) or ""
            if kind == "INSERT" or kind.startswith("DIMENSION_"):
                if self.modelspace(handle) and self.print_state(handle)[0]:
                    queue.append((handle, (), ()))
        paths = 0
        while queue:
            source, path, definitions = queue.popleft()
            block = self._definition(source)
            if block is None:
                continue
            if block in definitions:
                self.diagnostics["instance_cycle"] += 1
                continue
            if len(path) >= MAX_DEPTH or paths >= MAX_PATHS:
                raise ValueError("Native scene admission exceeded its fixed path budget")
            paths += 1
            next_path, next_definitions = (*path, source), (*definitions, block)
            members = self.value(block, "entities")
            if not isinstance(members, list):
                self.diagnostics["definition_members_unknown"] += 1
                continue
            handles = [strict_handle(ref, reference=True, origin=block) for ref in members]
            counts = Counter(handles)
            for member in handles:
                if member not in self.index or counts[member] != 1:
                    self.diagnostics["unresolved_or_duplicate_member"] += 1
                    continue
                if self.value(member, "entmode") != 0 or self.reference(member, "ownerhandle") != block:
                    self.diagnostics["nonreciprocal_member"] += 1
                    continue
                if not self.print_state(member)[0]:
                    continue
                self.paths[member].append({"instance_handles": list(next_path),
                                           "definition_handles": list(next_definitions)})
                kind = object_kind(self.index[member]) or ""
                if kind == "INSERT" or kind.startswith("DIMENSION_"):
                    queue.append((member, next_path, next_definitions))

    def qualify(self, source_ref: dict) -> dict:
        handle = strict_handle(self.objects[int(source_ref["json_pointer"].split("/")[2])].get("handle"),
                               reference=False)
        if handle not in self.index or source_ref["native_handle"] != f"{handle:X}":
            raise ValueError("Native observation identity changed")
        allowed, reason, layer = self.print_state(handle)
        direct, paths = self.modelspace(handle), self.paths[handle]
        accepted = allowed and (direct or bool(paths))
        if allowed and not accepted:
            reason = "modelspace_path_not_established"
        return {"admitted": accepted, "reason": reason,
                "basis": "direct_modelspace" if direct else "explicit_active_block_membership" if paths else "unresolved",
                "entity_handle": f"{handle:X}", "layer_handle": f"{layer:X}" if layer is not None else None,
                "instance_paths": deepcopy(paths), "final_PDF_visibility_confirmed": False,
                "same_revision_confirmed": False, "physical_owner_confirmed": False}


def qualify_document(document: dict, raw: dict) -> tuple[dict, dict]:
    """Filter a new input snapshot, never edit its source or original observations."""
    label_free(document)
    scene = SceneInput(raw)
    retained, checks = [], []
    for ordinal, field in enumerate(document["fields"]):
        if field["source_ref"]["source_sha256"] != document["dwg_source_sha256"]:
            raise ValueError("Native scene observation refers to a different source")
        check = scene.qualify(field["source_ref"])
        checks.append({"field_ordinal": ordinal, "kind": field["kind"], "category": field["category"],
                       "raw_text": field["raw_text"], "source_ref": field["source_ref"], **check})
        if check["admitted"]:
            retained.append({**deepcopy(field), "scene_admission": check})
    result = {**deepcopy(document), "fields": retained,
              "available": document["available"] and bool(retained),
              "scene_input_policy": "explicit_modelspace_path_global_print_flags.v1"}
    audit = {"item_code": document["item_code"], "before": len(checks), "after": len(retained),
             "reasons": dict(Counter(row["reason"] for row in checks if not row["admitted"])),
             "scene_diagnostics": dict(scene.diagnostics), "checks": checks,
             "route_labels_read": False, "PDF_sheet_membership_confirmed": False}
    return result, audit
