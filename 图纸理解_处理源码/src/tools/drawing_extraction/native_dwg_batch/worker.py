"""One isolated CPU task; invoked by the batch runner, never by a web request."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from .artifacts import digest, fingerprint, pack_file, read_json, save_gzip, save_json

__all__ = ["process_case"]


def process_case(snapshot: dict, case: dict, destination: Path) -> dict:
    started = time.monotonic()
    source = Path(case["path"])
    destination.mkdir(parents=True, exist_ok=True)
    result = {"schema": "native_dwg_batch_case.v1", "item_code": case["item_code"],
              "source": case, "recipe_sha256": snapshot["recipe_sha256"], "status": "failed",
              "artifacts": [], "stages": {}, "counts": {}, "reader_seconds": None,
              "gpu_used": False, "route_labels_read": False}

    def artifact(path: Path) -> None:
        result["artifacts"].append({"file": path.name, "sha256": digest(path), "bytes": path.stat().st_size})

    try:
        if digest(source) != case["sha256"]:
            raise ValueError("Original DWG differs from the frozen batch inventory")
        with source.open("rb") as stream:
            prefix = stream.read(256)
        if not re.fullmatch(rb"AC[0-9]{4}", prefix[:6]):
            result.update(status="unsupported_container", error_type="UnsupportedContainer",
                          error="非标准DWG文件头；需要原CAD导出可读取文件",
                          esafenet_marker_present=b"E-SafeNet" in prefix)
            return result
        binary = Path(snapshot["reader"]["path"])
        if digest(binary) != snapshot["reader"]["sha256"]:
            raise ValueError("Pinned native reader changed")
        with tempfile.TemporaryDirectory(dir=destination, prefix="reader-") as temporary:
            raw_path = Path(temporary) / "response.json"
            stderr = destination / "reader.stderr.txt"
            command = [str(binary), "-O", "JSON", str(source)]
            read_started = time.monotonic()
            try:
                with raw_path.open("wb") as stdout, stderr.open("wb") as errors:
                    completed = subprocess.run(command, stdout=stdout, stderr=errors,
                                               timeout=snapshot["reader_timeout"], check=False)
            except subprocess.TimeoutExpired:
                packed = destination / "native_response.json.gz"
                pack_file(raw_path, packed)
                artifact(packed)
                artifact(stderr)
                raise
            result["reader_seconds"] = time.monotonic() - read_started
            raw_hash = digest(raw_path)
            packed = destination / "native_response.json.gz"
            pack_file(raw_path, packed)
            artifact(packed)
            artifact(stderr)
            result["reader_receipt"] = {"command": command, "exit_code": completed.returncode,
                "raw_response_sha256": raw_hash, "raw_response_bytes": raw_path.stat().st_size,
                "stderr_bytes": stderr.stat().st_size, "reader_sha256": snapshot["reader"]["sha256"]}
            if completed.returncode:
                result.update(status="reader_failed", error_type="NativeReaderExit",
                              error=f"原生读取器返回 {completed.returncode}；原始响应和错误记录已保留")
                return result
            from .evidence import build_evidence
            from .projection import load_native

            parse_started = time.monotonic()
            parsed, duplicates, quarantined = load_native(raw_path.read_bytes())
            if parsed.get("FILEHEADER", {}).get("version") != prefix[:6].decode("ascii"):
                raise ValueError("Native response version differs from the original DWG header")
            result["lossless_json_seconds"] = time.monotonic() - parse_started
            result["duplicate_field_groups"] = len(duplicates)
            result["response_sha256"] = raw_hash
            evidence = build_evidence(parsed, item_code=case["item_code"], source_hash=case["sha256"],
                                      response_hash=raw_hash, catalog=snapshot["catalog"],
                                      documents=snapshot["documents"])
            evidence["source"] = case
            evidence["response_sha256"] = raw_hash
            evidence["recipe_sha256"] = snapshot["recipe_sha256"]
            evidence["quarantined_native_objects"] = quarantined
            evidence["stages"]["native_json_projection"] = {
                "status": "quarantined_objects" if quarantined else "completed",
                "seconds": result["lossless_json_seconds"], "quarantined_objects": len(quarantined),
                "error": "含非有限数值的原生对象整体隔离；不用于几何、字段或关系" if quarantined else "",
            }
            if quarantined:
                quarantine_path = destination / "quarantined_objects.json"
                save_json(quarantine_path, {"source": case, "response_sha256": raw_hash,
                                           "recipe_sha256": snapshot["recipe_sha256"], "objects": quarantined})
                artifact(quarantine_path)
            output = destination / "native_evidence.json.gz"
            save_gzip(output, evidence)
            artifact(output)
            children = destination / "child_catalog_links.json"
            save_json(children, {"source": case, "recipe_sha256": snapshot["recipe_sha256"],
                                 "rows": evidence["child_catalog_links"]})
            artifact(children)
            links = evidence["child_catalog_links"]
            result["stages"] = evidence["stages"]
            ir = evidence.get("native_ir", {})
            graph = evidence.get("storage_graph", {})
            result["counts"] = {
                "native_objects": len(parsed["OBJECTS"]),
                "dimensions": len(ir.get("dimensions", [])),
                "text_observations": len(ir.get("text_evidence", [])),
                "native_storage_nodes": len(graph.get("nodes", [])),
                "native_storage_edges": len(graph.get("edges", [])),
                "dimension_geometry_edges": len(ir.get("relations", [])),
                "bom_rows": len(links),
                "bom_connected_header_rows": sum(bool(row["fields"]) for row in links),
                "bom_continuation_candidate_rows": sum(bool(row["field_candidates"]) for row in links),
                "bom_catalog_matched_rows": sum(bool(row["catalog_matches"]) for row in links),
                "proxy_observations": len(evidence.get("proxy_observations", [])),
                "unresolved_proxy_observations": sum(row["status"] == "unresolved"
                                                     for row in evidence.get("proxy_observations", [])),
                "quarantined_native_objects": len(quarantined),
            }
            failures = [name for name, stage in result["stages"].items() if stage["status"] == "failed"]
            result["status"] = "partial" if (failures or quarantined
                                             or result["counts"]["unresolved_proxy_observations"]) else "parsed"
            result["failed_stages"] = failures
    except subprocess.TimeoutExpired:
        result.update(status="timeout", error_type="ReaderTimeout", error="原生读取超时；可单图重试")
    except Exception as error:  # Task boundary: all failures are persisted, never reported as success.
        result.update(status="failed", error_type=type(error).__name__, error=str(error)[:800])
    finally:
        result["source_unchanged"] = source.is_file() and digest(source) == case["sha256"]
        if not result["source_unchanged"]:
            result.update(status="source_changed", error="原图在处理期间发生变化，结果不得复用")
        result["seconds"] = time.monotonic() - started
        save_json(destination / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.nice(5)
    snapshot = read_json(args.snapshot)
    if fingerprint({k: v for k, v in snapshot.items() if k != "recipe_sha256"}) != snapshot["recipe_sha256"]:
        raise ValueError("Batch snapshot content hash changed")
    result = process_case(snapshot, snapshot["cases"][args.index], args.output)
    print(json.dumps({"item_code": result["item_code"], "status": result["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
