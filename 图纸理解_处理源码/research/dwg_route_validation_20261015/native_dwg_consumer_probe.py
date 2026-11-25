"""Immutable fixed-six DWG input consumption; no labels, heads or images.

Freeze, run and replay are separate processes. Only the new pipeline is pickled;
the original DWGs and cached reader responses are never copied or reread by a
reader. C1 adds finite native measurement numbers to the identical C0 channels.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time

import numpy as np
import scipy
from scipy.sparse import csr_matrix, hstack
import sklearn
from sklearn.feature_extraction import DictVectorizer

from research.dwg_route_validation_20261015 import native_dwg_inputs, native_ir
from research.dwg_route_validation_20261015.duplicate_audit import lossless_projection
from research.r19_route_estimation.limited_search.models import TextFeatures
from src.tools.research.source_graph import closure

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs/dwg_route_validation_20261015"
PARENT = BASE / "native_layer_role_001/PROTOCOL.json"
PARENT_SHA = "7913f2c03d1c247f585003439fef3ab49482a9403844275445e63291de25b5ee"
SCHEMA = "native_dwg_consumer_probe.v1"
RAW_LIMIT = 128 * 1024 * 1024
OUTPUT_LIMIT = 10 * 1024 * 1024
TEXT_PARAMETERS = {"embedding_path": None, "text_weights": [1., .5, .5], "embedding_weight": 1.}
TEXT_SOURCE_SHA = {
    "models.py": "390696f337f5c3355393ba2571149f3bb60252cea8b7571e6c4aa911716c1c03",
    "text_views.py": "d07db26990aae5f012b8e2d61b678a75eb547e1b08bea070dbeaf1630e0b089f",
}
SAFE_FIELDS = ("item_code", "group_id", "fit_index", "source_path", "source_sha256",
               "source_bytes", "archive_path", "archive_sha256", "raw_response_sha256")
ITEM_CODES = ["12500520015", "12501052343", "12810010463", "12500180941", "125002312743", "12502370454"]
POLICY = {
    "source_count": 6, "selection": "Unchanged previously exposed six FIT groups from pinned layer protocol",
    "modes": ["C0", "C1"], "difference": "Only finite native raw_measurement numerical columns",
    "measurement_prefix": "native_measurement/", "max_decompressed_response_bytes": RAW_LIMIT,
    "max_output_bytes": OUTPUT_LIMIT, "text_parameters": TEXT_PARAMETERS,
    "source_kind": "native_dwg", "units": "unknown", "physical_scope": "unknown",
    "route_training_performed": False, "route_predictions_performed": False,
    "route_accuracy_measured": False, "baseline_equivalence_claimed": False,
    "route_labels_read": False, "query_inputs_read": False, "val_test_read": False,
    "reader_calls": 0, "ocr_calls": 0, "gpu_used": False,
    "images_or_pdf_required": False, "embeddings_or_old_feature_caches_used": False,
}
_ACTIVE_AUDIT: dict | None = None


def encode(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)


def versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "pickle_protocol": 5}


def output_path(path: Path) -> Path:
    path = path.resolve()
    if path.parent != BASE.resolve() or not path.name.startswith("native_dwg_consumer_"):
        raise ValueError("Choose a new native_dwg_consumer_* immediate output child")
    return path


def _audit(event: str, args: tuple) -> None:
    state = _ACTIVE_AUDIT
    if state is None:
        return
    if (event.startswith(("socket.", "subprocess.", "os.exec", "os.spawn", "os.posix_spawn"))
            or event in {"os.system", "os.fork", "os.forkpty"}):
        raise PermissionError("Network and subprocess calls are forbidden in this native computation")
    if event != "open" or isinstance(args[0], int):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    mode, flags = args[1:3]
    writing = (isinstance(mode, str) and any(char in mode for char in "wax+")) or (
        isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
    if writing:
        if path not in state["writes"]:
            raise PermissionError(f"Native computation cannot write this path: {path}")
        return
    if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif", ".tiff", ".bmp", ".svg"}:
        raise PermissionError("Image/PDF reads are forbidden in this native computation")
    if path in state["reads"]:
        state["observed"].add(str(path.relative_to(REPO)))
        return
    if path.is_relative_to(REPO):
        if path.suffix == ".py" and path in state["code"]:
            return
        raise PermissionError(f"Unpinned repository data/cache/model read: {path}")
    if path.suffix.lower() in {".pkl", ".pickle", ".joblib", ".pt", ".pth", ".onnx", ".npz", ".npy", ".safetensors"}:
        raise PermissionError("External model/cache reads are forbidden")
    if not any(path.is_relative_to(prefix) for prefix in state["external"]):
        raise PermissionError(f"Only installed package/stdlib reads are allowed outside the repository: {path}")


sys.addaudithook(_audit)


@contextmanager
def file_guard(protocol: dict, output: Path):
    global _ACTIVE_AUDIT
    if _ACTIVE_AUDIT is not None:
        raise RuntimeError("Native file guards cannot be nested")
    own = {output / name for name in ("PROTOCOL.json", "pipeline.pkl", "result.json", "evidence.json.gz", "execution.json", "replay.json")}
    state = {"reads": {REPO / name for name in protocol["source_pins"]} | own,
             "writes": {output / "pipeline.pkl"},
             "code": {REPO / name for name in protocol["code_pins"]}, "observed": set(),
             "external": {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}}
    _ACTIVE_AUDIT = state
    try:
        yield state
    finally:
        _ACTIVE_AUDIT = None


def verify(protocol: dict) -> None:
    if protocol.get("schema_version") != SCHEMA or protocol.get("policy") != POLICY or protocol.get("versions") != versions():
        raise ValueError("Frozen protocol, policy or installed versions changed")
    rows = protocol["inventory"]
    if (len(rows) != 6 or len({row["group_id"] for row in rows}) != 6
            or [row["item_code"] for row in rows] != protocol["item_codes"] or protocol["item_codes"] != ITEM_CODES
            or any(set(row) != set(SAFE_FIELDS) for row in rows)):
        raise ValueError("All six native-only source records must be preserved")
    expected_sources = {row[field]: row[pin] for row in rows for field, pin in
                        (("source_path", "source_sha256"), ("archive_path", "archive_sha256"))}
    if protocol["source_pins"] != expected_sources or protocol["parent_protocol_sha256"] != PARENT_SHA:
        raise ValueError("Fixed source metadata and byte pins disagree")
    for row in rows:
        code = row["item_code"]
        if (row["source_path"] != f"data/frozen/route_drawing_ready_20260909/assets/dwg/{code}.dwg"
                or row["archive_path"] != f"outputs/dwg_route_validation_20261015/train_fit_native_probe_001/cases/{code}/stdout.json.gz"):
            raise ValueError("Only the six original DWGs and archived native responses are allowed")
    for name, expected in {**protocol["source_pins"], **protocol["code_pins"]}.items():
        if digest(REPO / name) != expected:
            raise ValueError(f"Frozen source/code changed: {name}")
    expected = sha(encode({"code_pins": protocol["code_pins"], "schema": SCHEMA,
                           "consumer_schema": native_dwg_inputs.SCHEMA,
                           "feature_schema": native_dwg_inputs.FEATURE_SCHEMA, "policy": POLICY}))
    if expected != protocol["pipeline_code_fingerprint"]:
        raise ValueError("Native cache/code fingerprint changed")


def freeze(output: Path) -> dict:
    output = output_path(output)
    if output.exists() or digest(PARENT) != PARENT_SHA:
        raise ValueError("Require a new output and the exact fixed-six parent protocol")
    parent = read(PARENT)
    inventory = [{key: row[key] for key in SAFE_FIELDS} for row in parent["inventory"]]
    code = closure([Path(__file__)], REPO)
    for name, expected in TEXT_SOURCE_SHA.items():
        relative = Path("research/r19_route_estimation/limited_search") / name
        for path in (REPO / relative, REPO / "release/delivery-20261007-v1/baseline/snapshot" / relative):
            if digest(path) != expected:
                raise ValueError("TextFeatures differs from the frozen delivery source")
            code.add(path)
    pins = {str(path.relative_to(REPO)): digest(path) for path in sorted(code)}
    protocol = {"schema_version": SCHEMA, "policy": POLICY, "versions": versions(),
                "parent_protocol": str(PARENT.relative_to(REPO)), "parent_protocol_sha256": PARENT_SHA,
                "item_codes": parent["item_codes"], "inventory": inventory,
                "source_pins": {row[field]: row[pin] for row in inventory for field, pin in
                                (("source_path", "source_sha256"), ("archive_path", "archive_sha256"))},
                "code_pins": pins,
                "visibility_definition_prior_review": {"artifact": "native_layer_role_001/line_style_source_review.json",
                    "sha256": "a2816d1ad3036510072c0bb028af222db8568a4fc53f380cd7a40d94f5204dfe",
                    "meaning": "Previously reviewed LAYER flag0 frozen=1/off=2; not reopened as a runtime input"}}
    protocol["pipeline_code_fingerprint"] = sha(encode({"code_pins": pins, "schema": SCHEMA,
        "consumer_schema": native_dwg_inputs.SCHEMA, "feature_schema": native_dwg_inputs.FEATURE_SCHEMA, "policy": POLICY}))
    verify(protocol)
    output.mkdir(exist_ok=False)
    write_new(output / "PROTOCOL.json", encode(protocol))
    write_new(output / "producer.executed.py", Path(__file__).read_bytes())
    return {"item_codes": protocol["item_codes"], "protocol_sha256": digest(output / "PROTOCOL.json")}


def matrix(value) -> csr_matrix:
    result = csr_matrix(value, dtype=np.float64, copy=True)
    result.sum_duplicates()
    result.sort_indices()
    result.eliminate_zeros()
    if not np.isfinite(result.data).all():
        raise ValueError("Native feature matrix must be finite")
    return result


def matrix_hash(value) -> str:
    value = matrix(value)
    hasher = hashlib.sha256(encode({"shape": list(value.shape), "dtype": "float64"}))
    for data, dtype in ((value.indptr, "<i8"), (value.indices, "<i8"), (value.data, "<f8")):
        hasher.update(np.asarray(data, dtype=dtype).tobytes())
    return hasher.hexdigest()


def transform(pipeline: dict, rows: list[dict], maps: dict) -> dict:
    names = pipeline["native_feature_names"]
    known = set(names)
    if any(set(values) - known for mode in POLICY["modes"] for values in maps[mode]):
        raise ValueError("Unseen native keys cannot be silently discarded")
    text = matrix(pipeline["text"].transform(rows))
    native = {mode: matrix(pipeline["native"].transform(maps[mode])) for mode in POLICY["modes"]}
    combined = {mode: matrix(hstack((text, native[mode]), format="csr")) for mode in POLICY["modes"]}
    return {"text": text, "native": native, "combined": combined}


def compute(protocol: dict, output: Path, *, replay: bool) -> tuple[dict, bytes]:
    records, rows, cases = [], [], []
    maps = {mode: [] for mode in POLICY["modes"]}
    for source in protocol["inventory"]:
        with gzip.open(REPO / source["archive_path"], "rb") as stream:
            raw = stream.read(RAW_LIMIT + 1)
        if len(raw) > RAW_LIMIT or sha(raw) != source["raw_response_sha256"]:
            raise ValueError("Cached native response exceeded its bound or changed")
        parsed, duplicates = lossless_projection(raw)
        ir = native_ir.build_ir(parsed, item_code=source["item_code"],
            source_sha256=source["source_sha256"], response_sha256=source["raw_response_sha256"])
        before = encode(ir)
        record = native_dwg_inputs.build_native_input(parsed, ir,
            source_sha256=source["source_sha256"], response_sha256=source["raw_response_sha256"])
        if before != encode(ir):
            raise ValueError("Native consumer mutated the rebuilt IR")
        records.append(record)
        rows.append({"item_code": source["item_code"], "group_id": source["group_id"], "split": "train",
                     "source_kind": "native_dwg", "source_sha256": source["source_sha256"],
                     "base_text": native_dwg_inputs.sanitized_predictive_text(record),
                     "evidence_text": "", "analysis_sha256": ""})
        fingerprints = {}
        for mode in POLICY["modes"]:
            maps[mode].append(native_dwg_inputs.predictive_features(record, mode=mode))
            fingerprints[mode] = sha(encode({"native_cache_fingerprint": native_dwg_inputs.cache_fingerprint(record, mode=mode),
                                            "pipeline_code_fingerprint": protocol["pipeline_code_fingerprint"]}))
        if {key: value for key, value in maps["C1"][-1].items() if not key.startswith("native_measurement/")} != maps["C0"][-1]:
            raise ValueError("C0/C1 differ outside native measurement features")
        cases.append({"item_code": source["item_code"], "group_id": source["group_id"],
            "source_sha256": source["source_sha256"], "response_sha256": source["raw_response_sha256"],
            "record_sha256": record["record_sha256"], "native_cache_fingerprints": fingerprints,
            "accepted_text": len(record["accepted_text"]), "deferred_text": len(record["deferred_text"]),
            "body_text_sha256": sha(record["body_text"].encode("utf-8")),
            "predictive_body_text_sha256": sha(rows[-1]["base_text"].encode("utf-8")),
            "dimensions": len(record["dimensions"]),
            "measurement_status_counts": dict(Counter(node["raw_measurement_status"] for node in record["dimensions"])),
            "finite_raw_measurements_preserved": sum(node["raw_measurement_status"] == "finite_native_raw" for node in record["dimensions"]),
            "source_nonhidden_dimensions": sum(node["visibility"]["source_declared_not_hidden"] for node in record["dimensions"]),
            "dimensions_actually_consumed": int(sum(value for key, value in maps["C0"][-1].items() if key.startswith("native_dimension_kind/"))),
            "finite_raw_measurements_actually_consumed": int(sum(value for key, value in maps["C1"][-1].items() if key.startswith("native_measurement/") and key.endswith("/count"))),
            "blank_user_text_preserved": sum(node["raw_user_text"] == "" and node["raw_user_text_evidence"]["occurrence_count"] > 0
                                               and not node["raw_user_text_evidence"]["conflicting"] for node in record["dimensions"]),
            "direct_relations": len(ir["relations"]), "duplicate_field_groups_preserved": len(duplicates),
            "physical_scope": "unknown", "displayed_dimension_text": "unknown", "units": "unknown"})
    text = TextFeatures.fit(rows, TEXT_PARAMETERS)
    vectorizer = DictVectorizer(dtype=np.float64, sparse=True, sort=True).fit(maps["C1"])
    pipeline = {"schema_version": SCHEMA, "text": text, "native": vectorizer,
                "native_feature_names": vectorizer.get_feature_names_out().tolist(),
                "text_parameters": TEXT_PARAMETERS, "pipeline_code_fingerprint": protocol["pipeline_code_fingerprint"]}
    fresh = transform(pipeline, rows, maps)
    pickle_path = output / "pipeline.pkl"
    if replay:
        execution = read(output / "execution.json")
        if (execution["pid"] == os.getpid() or execution["protocol_sha256"] != digest(output / "PROTOCOL.json")
                or digest(pickle_path) != execution["pipeline_pickle_sha256"]):
            raise ValueError("Replay must verify the original pinned pipeline bytes before unpickling")
    if not replay:
        blob = pickle.dumps(pipeline, protocol=5)
        if len(blob) > OUTPUT_LIMIT:
            raise ValueError("Pipeline exceeds the complete output budget")
        write_new(pickle_path, blob)
    loaded = pickle.loads(pickle_path.read_bytes())
    if (loaded["schema_version"] != SCHEMA or loaded["pipeline_code_fingerprint"] != protocol["pipeline_code_fingerprint"]
            or loaded["native_feature_names"] != pipeline["native_feature_names"]
            or loaded["text_parameters"] != TEXT_PARAMETERS or loaded["text"].embedding_path is not None):
        raise ValueError("Reloaded native pipeline contract changed")
    restored = transform(loaded, rows, maps)
    for name in ("text", "C0", "C1"):
        left = fresh["text"] if name == "text" else fresh["combined"][name]
        right = restored["text"] if name == "text" else restored["combined"][name]
        if matrix_hash(left) != matrix_hash(right):
            raise ValueError("Saved pipeline does not reproduce the actual feature matrix")
    names = pipeline["native_feature_names"]
    common = [index for index, name in enumerate(names) if not name.startswith("native_measurement/")]
    if matrix_hash(fresh["native"]["C0"][:, common]) != matrix_hash(fresh["native"]["C1"][:, common]):
        raise ValueError("Common native columns changed")
    delta = matrix(fresh["native"]["C1"] - fresh["native"]["C0"])
    if not delta.nnz:
        raise ValueError("The fixed-six measurement probe must produce an actual numerical difference")
    changed = sorted({names[index] for index in delta.indices})
    changed_items = [row["item_code"] for index, row in enumerate(rows) if delta.indptr[index + 1] > delta.indptr[index]]
    if any(not name.startswith("native_measurement/") for name in changed):
        raise ValueError("A nonmeasurement column differs between C0 and C1")
    for mode in POLICY["modes"]:
        if matrix_hash(fresh["combined"][mode][:, :fresh["text"].shape[1]]) != matrix_hash(fresh["text"]):
            raise ValueError("Appending native fields changed the frozen text block")
    evidence = gzip.compress(encode({"schema_version": SCHEMA, "records": records, "native_rows": rows,
                                    "feature_maps": maps, "native_feature_names": names}), mtime=0)
    result = {"schema_version": SCHEMA, "policy": POLICY, "denominator": 6, "cases": cases,
              "pipeline_code_fingerprint": protocol["pipeline_code_fingerprint"], "pipeline_pickle_sha256": digest(pickle_path),
              "evidence_gzip_sha256": sha(evidence), "text_matrix_sha256": matrix_hash(fresh["text"]),
              "text_shape": list(fresh["text"].shape), "native_feature_names": names,
              "matrix_sha256": {mode: matrix_hash(fresh["combined"][mode]) for mode in POLICY["modes"]},
              "matrix_shape": {mode: list(fresh["combined"][mode].shape) for mode in POLICY["modes"]},
              "changed_native_columns": changed, "text_blocks_bit_identical": True,
              "changed_rows": len(changed_items), "changed_item_codes": changed_items,
              "summary": {key: sum(case[key] for case in cases) for key in (
                  "accepted_text", "deferred_text", "dimensions", "finite_raw_measurements_preserved",
                  "source_nonhidden_dimensions", "dimensions_actually_consumed",
                  "finite_raw_measurements_actually_consumed", "blank_user_text_preserved", "direct_relations")},
              "common_native_columns_bit_identical": True, "saved_pipeline_reload_exact": True,
              "feature_fit_sources": 6, "text_and_native_schema_shared": True,
              "file_audit_active": True, "network_or_subprocess_calls": 0}
    return result, evidence


def run(output: Path, *, replay: bool) -> dict:
    output = output_path(output)
    protocol = read(output / "PROTOCOL.json")
    started = time.monotonic()
    with file_guard(protocol, output) as audit:
        verify(protocol)
        result, evidence = compute(protocol, output, replay=replay)
        verify(protocol)
    serialized = encode(result)
    if replay:
        execution = read(output / "execution.json")
        if execution["pid"] == os.getpid() or (output / "result.json").read_bytes() != serialized or (output / "evidence.json.gz").read_bytes() != evidence:
            raise ValueError("Independent native replay is not byte-identical")
        if digest(output / "result.json") != execution["result_sha256"]:
            raise ValueError("Original native result changed")
    else:
        retained = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
        if retained + len(serialized) + len(evidence) + 65536 > OUTPUT_LIMIT:
            raise ValueError("Complete native probe outputs exceed 10MiB")
        write_new(output / "evidence.json.gz", evidence)
        write_new(output / "result.json", serialized)
    receipt = {"schema_version": SCHEMA, "pid": os.getpid(), "elapsed_seconds": time.monotonic() - started,
               "versions": versions(), "protocol_sha256": digest(output / "PROTOCOL.json"),
               "result_sha256": digest(output / "result.json"), "evidence_gzip_sha256": sha(evidence),
               "pipeline_pickle_sha256": digest(output / "pipeline.pkl"), "independent_byte_equal_replay": replay,
               "observed_pinned_repo_reads": sorted(audit["observed"]), "policy": POLICY}
    write_new(output / ("replay.json" if replay else "execution.json"), encode(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "replay"))
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    result = freeze(options.output) if options.action == "freeze" else run(options.output, replay=options.action == "replay")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
