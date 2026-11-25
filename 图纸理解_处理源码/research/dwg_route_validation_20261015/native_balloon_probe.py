"""Check native balloon vectors and declared reactor contacts, without images.

An exact row string and a geometry contact remain separate candidate relations;
neither establishes the manufacturing object or a physical part boundary.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import time

import ezdxf
import ezdxf.proxygraphic
import ezdxf.tools.binarydata

from research.dwg_route_validation_20261015 import native_balloon_vectors
from research.dwg_route_validation_20261015.duplicate_audit import lossless_projection
from research.dwg_route_validation_20261015.native_dwg_consumer_probe import (
    BASE, ITEM_CODES, REPO, digest, encode, file_guard, write_new,
)
from research.dwg_route_validation_20261015.native_insert_transform_probe import (
    ABS_TOL, REL_TOL, package_pins,
)
from src.tools.research.source_graph import closure

OWNERSHIP = BASE / "native_cad_ownership_001"
COORDINATES = BASE / "native_insert_transform_001"
PINS = {
    str((OWNERSHIP / "PROTOCOL.json").relative_to(REPO)): "df1400365d30b36c16d3b17190de91cd5bd8f57c6940abae9210fbc4faf813d7",
    str((OWNERSHIP / "result.json").relative_to(REPO)): "b07d51ea44a5ee47c50fc86d89c2fd59ac4114e62c69c95fb7ae9b1b80873f35",
    str((OWNERSHIP / "evidence.json.gz").relative_to(REPO)): "f186bd9272016eff2fc6b96bcb6b634db550970a38bda6927b4ff88ff6cc6b96",
    str((COORDINATES / "PROTOCOL.json").relative_to(REPO)): "ff6d45d712107c627ba522ad90795d18dc4ea9a48940952bafd305494f7aa7cd",
    str((COORDINATES / "result.json").relative_to(REPO)): "4927e73621e53f59f529aa2534a73e56b307ffe1310d9a092be984563930338e",
    str((COORDINATES / "evidence.json.gz").relative_to(REPO)): "afbceb7b627f87304679aae7ca5652c8f212466cb08f4d4b34332cd4f80686eb",
}
SCHEMA = "native_balloon_probe.v1"
READ_LIMIT = 256 * 1024 * 1024
OUTPUT_LIMIT = 10 * 1024 * 1024
POLICY = {
    "source_count": 6, "selection": "unchanged exposed FIT native DWGs",
    "native_preview_vectors_not_raster": True, "independent_struct_comparison": True,
    "contact_scope": "declared reactor LINE or verified members of declared reactor INSERT",
    "anchors": ["terminal_circle_center", "terminal_segment_free_endpoint"],
    "finite_line_distance": "clamped Euclidean 3D projection; retain all contacts",
    "abs_tolerance_native_units": ABS_TOL, "relative_tolerance": REL_TOL,
    "nearest_neighbor_assignment": False, "opaque_plugin_payload_decoded": False,
    "exact_sequence_match_is_candidate_only": True, "physical_part_ownership_validated": False,
    "final_visibility_validated": False, "route_labels_read": False, "query_inputs_read": False,
    "val_test_read": False, "pdf_or_images_read": False, "ocr_calls": 0, "reader_calls": 0,
    "route_training_performed": False, "gpu_used": False,
    "max_read_bytes": READ_LIMIT, "max_output_bytes": OUTPUT_LIMIT,
}


def library_pins() -> dict[str, str]:
    paths = [Path(ezdxf.proxygraphic.__file__), Path(ezdxf.tools.binarydata.__file__)]
    return {**package_pins(), **{str(path): digest(path) for path in paths}}


def output_path(path: Path) -> Path:
    path = path.resolve()
    if path.parent != BASE.resolve() or not path.name.startswith("native_balloon_"):
        raise ValueError("Choose a new native_balloon_* immediate output child")
    return path


def verify(protocol: dict) -> None:
    if (protocol.get("schema_version") != SCHEMA or protocol.get("policy") != POLICY
            or protocol.get("python_version") != platform.python_version()
            or protocol.get("ezdxf_version") != ezdxf.__version__
            or protocol.get("package_pins") != library_pins()):
        raise ValueError("Frozen native balloon policy/environment changed")
    for name, pin in PINS.items():
        if digest(REPO / name) != pin:
            raise ValueError("Frozen native ownership or coordinate evidence changed")
    parent = json.loads((OWNERSHIP / "PROTOCOL.json").read_bytes())
    expected = {**parent["source_pins"], **PINS}
    if (protocol["source_pins"] != expected or protocol["inventory"] != parent["inventory"]
            or protocol["item_codes"] != ITEM_CODES):
        raise ValueError("Native balloon source read boundary changed")
    required = {str(Path(__file__).relative_to(REPO)),
                str(Path(native_balloon_vectors.__file__).relative_to(REPO))}
    if not required.issubset(protocol["code_pins"]):
        raise ValueError("Native balloon code closure missing")
    for name, pin in {**protocol["source_pins"], **protocol["code_pins"]}.items():
        if digest(REPO / name) != pin:
            raise ValueError(f"Frozen balloon source/code changed: {name}")


def freeze(output: Path) -> dict:
    output = output_path(output)
    if output.exists():
        raise ValueError("Require a new native balloon output directory")
    for name, pin in PINS.items():
        if digest(REPO / name) != pin:
            raise ValueError("Unchanged native source evidence is required")
    parent = json.loads((OWNERSHIP / "PROTOCOL.json").read_bytes())
    code = {str(path.relative_to(REPO)): digest(path)
            for path in sorted(closure([Path(__file__)], REPO))}
    protocol = {"schema_version": SCHEMA, "policy": POLICY,
                "python_version": platform.python_version(), "ezdxf_version": ezdxf.__version__,
                "package_pins": library_pins(), "source_pins": {**parent["source_pins"], **PINS},
                "inventory": parent["inventory"], "item_codes": ITEM_CODES, "code_pins": code}
    verify(protocol)
    output.mkdir(exist_ok=False)
    write_new(output / "PROTOCOL.json", encode(protocol))
    write_new(output / "producer.executed.py", Path(__file__).read_bytes())
    return {"protocol_sha256": digest(output / "PROTOCOL.json")}


def bounded_gzip(path: Path) -> dict:
    with gzip.open(path, "rb") as stream:
        raw = stream.read(READ_LIMIT + 1)
    if len(raw) > READ_LIMIT:
        raise ValueError("Native evidence exceeds frozen read limit")
    return json.loads(raw)


def point(value: object) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3 or any(
            type(v) not in {float, int} or not math.isfinite(v) for v in value):
        raise ValueError("Finite explicit native XYZ required")
    return list(value)


def finite_line_distance(p: list[float], a: list[float], b: list[float]) -> dict:
    """A nearby extension never substitutes for finite-segment contact."""
    p, a, b = point(p), point(a), point(b)
    v = [y - x for x, y in zip(a, b, strict=True)]
    length2 = sum(x * x for x in v)
    if length2 == 0:
        raise ValueError("Degenerate LINE cannot supply a directed segment")
    t = sum((x - y) * z for x, y, z in zip(p, a, v, strict=True)) / length2
    clamped = max(0., min(1., t))
    distance = math.dist(p, [x + clamped * z for x, z in zip(a, v, strict=True)])
    tolerance = ABS_TOL + REL_TOL * max(abs(x) for q in (p, a, b) for x in q)
    return {"distance_native_units": distance, "t_unclamped": t, "t_clamped": clamped,
            "tolerance_native_units": tolerance, "numeric_contact": distance <= tolerance}


def independent_preview_check(binary: bytes, balloon: dict) -> dict:
    """Compare direct struct values to the library; do not use its ByteStream."""
    if len(binary) < 8 or struct.unpack_from("<I", binary)[0] != len(binary):
        raise ValueError("Incomplete native graphic stream")
    records, offset, counts = [], 8, Counter()
    indexed = {row["frame_index"]: row for row in [*balloon["geometry"], *balloon["texts"]]}
    while offset < len(binary):
        if offset + 8 > len(binary):
            raise ValueError("Incomplete native command header")
        size, opcode = struct.unpack_from("<II", binary, offset)
        if size < 8 or offset + size > len(binary):
            raise ValueError("Incomplete native command")
        frame = len(records)
        payload = binary[offset + 8:offset + size]
        record = {"index": frame, "offset": offset, "size": size, "opcode": opcode}
        if opcode == 6:
            count = struct.unpack_from("<I", payload)[0]
            if len(payload) != 4 + count * 24:
                raise ValueError("Independent POLYLINE extent mismatch")
            coords = struct.unpack_from(f"<{count * 3}d", payload, 4)
            actual = [point(coords[i:i + 3]) for i in range(0, len(coords), 3)]
            expected = [row["location"] for row in indexed[frame]["vertices"]]
            if actual != expected:
                raise ValueError("Independent POLYLINE coordinates differ")
            record["vertices"] = actual
            counts["polyline_records"] += 1
        elif opcode == 2:
            values = struct.unpack("<7d", payload)
            attributes = indexed[frame]["attributes"]
            if (point(values[:3]) != attributes["center"] or values[3] != attributes["radius"]
                    or point(values[4:]) != attributes["extrusion"]):
                raise ValueError("Independent CIRCLE fields differ")
            record.update(center=list(values[:3]), radius=values[3], normal=list(values[4:]))
            counts["circle_records"] += 1
        elif opcode == 11:
            values = struct.unpack_from("<9d", payload)
            text_end = payload.index(b"\0", 72)
            text = payload[72:text_end].decode("gbk", errors="strict")
            actual = indexed[frame]
            if (text != actual["raw_text"] or point(values[:3]) != actual["raw_start_point"]
                    or point(values[3:6]) != actual["raw_normal"]
                    or point(values[6:]) != actual["raw_text_direction"]):
                raise ValueError("Independent TEXT2 literal/coordinates differ")
            record.update(raw_text=text, insert=list(values[:3]), normal=list(values[3:6]),
                          direction=list(values[6:]), text_byte_start=offset + 80,
                          text_byte_end=offset + 8 + text_end)
            counts["text_records"] += 1
        records.append(record)
        offset += size
    expected = [{key: frame[key] for key in ("index", "offset", "size", "opcode")}
                for frame in balloon["frames"]]
    if [{key: frame[key] for key in ("index", "offset", "size", "opcode")} for frame in records] != expected:
        raise ValueError("Independent frame layout differs")
    return {"status": "independent_struct_fields_equal", "counts": dict(counts), "records": records}


def terminal_anchors(balloon: dict) -> list[dict]:
    """Describe a same-object chain and terminal circle, without target semantics."""
    geometry = balloon["geometry"]
    lines = [row for row in geometry if row["type"] == "POLYLINE"]
    circles = [row for row in geometry if row["type"] == "CIRCLE"]
    if len(lines) != 2 or len(circles) != 1 or len(geometry) != 3:
        return []
    endpoints = [[point(v["location"]) for v in row["vertices"]] for row in lines]
    if any(len(points) != 2 or points[0] == points[1] for points in endpoints):
        return []
    shared = {tuple(p) for p in endpoints[0]} & {tuple(p) for p in endpoints[1]}
    if len(shared) != 1:
        return []
    circle = circles[0]
    center, radius = point(circle["attributes"]["center"]), circle["attributes"]["radius"]
    contacts = []
    for row, points in zip(lines, endpoints, strict=True):
        for slot, p in enumerate(points):
            if tuple(p) in shared:
                continue
            residual = abs(math.dist(center, p) - radius)
            tolerance = ABS_TOL + REL_TOL * max(abs(x) for q in (p, center) for x in q)
            if residual <= tolerance:
                contacts.append({"kind": "terminal_segment_free_endpoint", "point": p,
                                 "frame_index": row["frame_index"], "vertex_slot": slot,
                                 "circle_radius": radius, "circle_frame_index": circle["frame_index"],
                                 "circle_boundary_residual_native_units": residual,
                                 "tolerance_native_units": tolerance})
    if len(contacts) != 1:
        return []
    return [{"kind": "terminal_circle_center", "point": center,
             "frame_index": circle["frame_index"], "circle_radius": radius}, *contacts]


def line_catalog(graph: dict, coordinates: dict) -> list[dict]:
    """Include only explicit modelspace LINEs and previously verified instances."""
    result = []
    for node in graph["objects"]:
        if (node["native_kind"] != "LINE" or node["cad_context"]["cad_space"] != "modelspace"
                or node["cad_context"]["reasons"]):
            continue
        fields = [node["field_evidence"].get(name, {}) for name in ("start", "end")]
        if any(ev.get("conflicting") or not ev.get("occurrence_count") for ev in fields):
            continue
        result.append({"source_ref": node["source_ref"], "path_edge_ids": [],
                       "start": point(fields[0]["raw_value"]), "end": point(fields[1]["raw_value"]),
                       "root_insert_source_pointer": None})
    grouped = defaultdict(dict)
    for row in coordinates["comparison"]["line_endpoint_evidence"]:
        key = (tuple(row["path_edge_ids"]), row["source_ref"]["json_pointer"])
        if row["field"] in grouped[key]:
            raise ValueError("Repeated cached LINE endpoint")
        grouped[key][row["field"]] = row
    edges = {row["id"]: row for row in graph["reference_edges"]}
    for (path, _), pair in grouped.items():
        if set(pair) != {"start", "end"} or not path:
            raise ValueError("Incomplete previously verified LINE occurrence")
        start, end = pair["start"], pair["end"]
        if start["source_ref"] != end["source_ref"]:
            raise ValueError("Cached LINE endpoint source identities differ")
        result.append({"source_ref": start["source_ref"], "path_edge_ids": list(path),
                       "start": point(start["modelspace_point"]), "end": point(end["modelspace_point"]),
                       "root_insert_source_pointer": edges[path[0]]["source_ref"]["json_pointer"]})
    return result


def contact_audit(balloon: dict, graph: dict, lines: list[dict]) -> dict:
    anchors = terminal_anchors(balloon)
    if balloon["cad_context"]["cad_space"] != "modelspace" or balloon["cad_context"]["reasons"]:
        return {"anchors": anchors, "declared_reactor_contact_checks": [],
                "status": "unknown_preview_modelspace_transform", "physical_owner": "unknown",
                "final_visibility": "unknown", "geometry_target_confirmed": False,
                "manufacturing_evidence_admitted": False}
    nodes = {row["source_ref"]["json_pointer"]: row for row in graph["objects"]}
    results = []
    for edge in balloon["incoming_reactors"]:
        node = nodes[edge["source_ref"]["json_pointer"]]
        supported = edge["status"] == "declared_native_reactor_reference"
        selected = [line for line in lines if supported and (
            node["native_kind"] == "LINE" and line["source_ref"] == node["source_ref"]
            or node["native_kind"] == "INSERT" and line["root_insert_source_pointer"] == node["source_ref"]["json_pointer"])]
        checks = []
        for anchor in anchors:
            contacts, distances, degenerate = [], [], 0
            for line in selected:
                try:
                    check = finite_line_distance(anchor["point"], line["start"], line["end"])
                except ValueError:
                    degenerate += 1
                    continue
                distances.append(check["distance_native_units"])
                if check["numeric_contact"]:
                    contacts.append({**line, **check})
            checks.append({"anchor": anchor, "evaluated_line_occurrences": len(distances),
                           "degenerate_lines_skipped": degenerate, "contacts": contacts,
                           "minimum_distance_diagnostic_only": min(distances) if distances else None,
                           "status": "native_reactor_declared_geometry_contact_candidate" if contacts else "unknown"})
        results.append({"reactor_source_ref": edge["source_ref"], "reactor_source_kind": node["native_kind"],
                        "reactor_slot_pointer": edge["source_pointer"], "reference_status": edge["status"],
                        "eligible_line_occurrences": len(selected), "anchor_checks": checks})
    return {"anchors": anchors, "declared_reactor_contact_checks": results,
            "physical_owner": "unknown", "final_visibility": "unknown",
            "geometry_target_confirmed": False, "manufacturing_evidence_admitted": False}


def compute(protocol: dict) -> tuple[dict, bytes]:
    graphs = bounded_gzip(OWNERSHIP / "evidence.json.gz")["cases"]
    expected = json.loads((OWNERSHIP / "result.json").read_bytes())["cases"]
    coordinates = bounded_gzip(COORDINATES / "evidence.json.gz")["cases"]
    if any([case["item_code"] for case in cases] != ITEM_CODES for cases in (graphs, expected, coordinates)):
        raise ValueError("All six original cases must remain")
    cases, evidence = [], []
    for source, case, pin, coords in zip(protocol["inventory"], graphs, expected, coordinates, strict=True):
        graph = case["graph"]
        if hashlib.sha256(encode(graph)).hexdigest() != pin["graph_sha256"]:
            raise ValueError("Native ownership graph changed")
        with gzip.open(REPO / source["archive_path"], "rb") as stream:
            raw = stream.read(READ_LIMIT + 1)
        if len(raw) > READ_LIMIT or hashlib.sha256(raw).hexdigest() != source["raw_response_sha256"]:
            raise ValueError("Original native response changed")
        parsed, _ = lossless_projection(raw)
        recovered = native_balloon_vectors.build_balloon_vectors(parsed, graph, coords["attribute_rows"])
        lines = line_catalog(graph, coords) if recovered["balloons"] else []
        counts, audits = Counter(), []
        for balloon in recovered["balloons"]:
            counts["selected_objects"] += 1
            if balloon["status"] != "native_vector_evidence":
                counts["unknown_preview_objects"] += 1
                continue
            position = int(balloon["source_ref"]["json_pointer"].split("/")[-1])
            binary = bytes.fromhex(parsed["OBJECTS"][position]["preview"])
            independent = independent_preview_check(binary, balloon)
            contacts_audit = contact_audit(balloon, graph, lines)
            audits.append({"source_ref": balloon["source_ref"], "independent_struct_check": independent,
                           "contact_audit": contacts_audit})
            counts["independently_checked_previews"] += 1
            counts["previews_with_terminal_anchors"] += bool(contacts_audit["anchors"])
            counts["previews_with_declared_reactors"] += any(
                edge["status"] == "declared_native_reactor_reference" for edge in balloon["incoming_reactors"])
            counts["previews_with_exact_row_candidate"] += any(
                row["status"] == "exact_string_candidate_only" for row in balloon["label_row_candidates"])
            contacts = [check for edge in contacts_audit["declared_reactor_contact_checks"]
                        for check in edge["anchor_checks"] if check["contacts"]]
            counts["previews_with_numeric_reactor_contact_candidate"] += bool(contacts)
            counts["numeric_contact_occurrences"] += sum(len(check["contacts"]) for check in contacts)
        evidence.append({"item_code": source["item_code"], "recovered": recovered, "verification_audits": audits})
        cases.append({"item_code": source["item_code"], "source_sha256": source["source_sha256"],
                      "counts": dict(counts), "source_summary": recovered["summary"],
                      "physical_part_bindings_confirmed": 0})
    compressed = gzip.compress(encode({"schema_version": SCHEMA, "cases": evidence}), mtime=0)
    result = {"schema_version": SCHEMA, "policy": POLICY, "denominator": 6, "cases": cases,
              "totals": dict(sum((Counter(case["counts"]) for case in cases), Counter())),
              "evidence_gzip_sha256": hashlib.sha256(compressed).hexdigest(),
              "route_accuracy_measured": False, "physical_part_accuracy_measured": False}
    return result, compressed


def run(output: Path, *, replay: bool) -> dict:
    output = output_path(output)
    protocol = json.loads((output / "PROTOCOL.json").read_bytes())
    started = time.monotonic()
    with file_guard(protocol, output) as audit:
        verify(protocol)
        result, evidence = compute(protocol)
        verify(protocol)
    payload = encode(result)
    if replay:
        original = json.loads((output / "execution.json").read_bytes())
        if (original["pid"] == os.getpid() or original["protocol_sha256"] != digest(output / "PROTOCOL.json")
                or original["result_sha256"] != digest(output / "result.json")
                or (output / "result.json").read_bytes() != payload
                or (output / "evidence.json.gz").read_bytes() != evidence):
            raise ValueError("Independent native balloon replay differs")
    else:
        retained = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
        if retained + len(payload) + len(evidence) + 65536 > OUTPUT_LIMIT:
            raise ValueError("Native balloon output exceeds frozen bound")
        write_new(output / "result.json", payload)
        write_new(output / "evidence.json.gz", evidence)
    receipt = {"schema_version": SCHEMA, "pid": os.getpid(), "elapsed_seconds": time.monotonic() - started,
               "protocol_sha256": digest(output / "PROTOCOL.json"), "result_sha256": digest(output / "result.json"),
               "evidence_gzip_sha256": hashlib.sha256(evidence).hexdigest(),
               "independent_byte_equal_replay": replay, "observed_pinned_repo_reads": sorted(audit["observed"]),
               "policy": POLICY}
    write_new(output / ("replay.json" if replay else "execution.json"), encode(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "replay"))
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    receipt = freeze(options.output) if options.action == "freeze" else run(
        options.output, replay=options.action == "replay")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
