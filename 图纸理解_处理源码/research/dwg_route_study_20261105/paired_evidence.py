"""CPU-only, immutable DWG -> vector PDF -> source-linked evidence bundle.

Reuse a hash-verified native reader response; never read a route label, invoke
OCR, alter the delivery model, or overwrite an earlier experiment directory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import ezdxf
import pymupdf as fitz

from research.dwg_pdf_visualization_20261031.render_native import pixel_audit, render
from research.dwg_route_study_20261105.paired_facts import build_bundle
from src.tools.drawing_extraction.native_dwg_batch.projection import load_native
from src.tools.research.artifacts import (
    contained_path, digest, read_json, verify_manifest, write_new_json,
)
from src.tools.research.source_graph import closure

REPO = Path(__file__).resolve().parents[2]
CONVERTER = Path("/home/userroot/miniconda3/envs/dwg-free/bin/dwg2dxf")
BASELINE = REPO / "release/delivery-20261007-v1/baseline/model/model.joblib"
BASELINE_SHA256 = "2d9c27bfc3a49861b58cf696998a51013b1aa03e6e88b51b79b464ca00a8f2f8"


def load_case(case: Path) -> tuple[dict, Path, Path, dict, list]:
    result = read_json(case / "result.json")
    if result["status"] != "parsed" or not result["source_unchanged"] or result["failed_stages"]:
        raise ValueError("A complete, unchanged native source receipt is required")
    source = Path(result["source"]["path"])
    if digest(source) != result["source"]["sha256"]:
        raise ValueError("Original DWG changed after native reading")
    response = contained_path(case, "native_response.json.gz")
    artifacts = {row["file"]: row for row in result["artifacts"]}
    if digest(response) != artifacts[response.name]["sha256"]:
        raise ValueError("Compressed native response changed")
    raw = gzip.decompress(response.read_bytes())
    raw_hash = hashlib.sha256(raw).hexdigest()
    if (raw_hash != result["response_sha256"]
            or raw_hash != result["reader_receipt"]["raw_response_sha256"]
            or len(raw) != result["reader_receipt"]["raw_response_bytes"]):
        raise ValueError("Reader bytes differ from the pinned native source")
    parsed, duplicates, quarantined = load_native(raw)
    if quarantined:
        raise ValueError("Quarantined objects require review; do not claim a complete vector export")
    return result, source, response, parsed, duplicates


def _pins(paths: list[Path]) -> list[dict]:
    return [{"path": str(path.resolve()), "sha256": digest(path)} for path in paths]


def _check_pins(pins: list[dict]) -> None:
    for pin in pins:
        if digest(Path(pin["path"])) != pin["sha256"]:
            raise ValueError(f"Frozen source or code changed: {pin['path']}")


def convert(source: Path, output: Path) -> tuple[Path, dict]:
    destination = output / "evidence" / f"{source.stem}.dxf"
    destination.parent.mkdir(parents=True, exist_ok=False)
    command = [str(CONVERTER), "--as", "r2004", "-y", "-o", str(destination), str(source)]
    pins = _pins([source, CONVERTER])
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, capture_output=True, timeout=120, check=False)
    except subprocess.TimeoutExpired as error:
        (destination.parent / "conversion.stderr").write_bytes(error.stderr or b"")
        (destination.parent / "conversion.stdout").write_bytes(error.stdout or b"")
        raise RuntimeError("DWG converter timed out; no completed bundle was published") from error
    receipt = {"command": command, "exit_code": completed.returncode,
               "seconds": time.perf_counter() - started, "inputs": pins}
    (destination.parent / "conversion.stdout").write_bytes(completed.stdout)
    (destination.parent / "conversion.stderr").write_bytes(completed.stderr)
    write_new_json(destination.parent / "conversion.json", receipt)
    _check_pins(pins)
    if completed.returncode or not destination.is_file():
        raise RuntimeError("DWG conversion failed; see evidence/conversion.stderr")
    return destination, receipt


def run(case: Path, output: Path) -> dict:
    started = time.perf_counter()
    code_pins = _pins(sorted(closure([Path(__file__).resolve()], REPO)))
    result, source, response, raw, duplicates = load_case(case)
    if digest(BASELINE) != BASELINE_SHA256:
        raise ValueError("Delivery baseline identity changed")
    inputs = _pins([case / "result.json", source, response, BASELINE])
    output.mkdir(parents=True, exist_ok=False)
    source_seconds = time.perf_counter() - started
    try:
        dxf, conversion = convert(source, output)
        manifest = render(source, response, dxf, output, cad_bounds=None,
                          filename_stem=source.stem, audit_entities=True)
        pdf = output / "evidence" / f"{source.stem}_原生矢量.pdf"
        before_pair = time.perf_counter()
        bundle = build_bundle(raw, manifest, item_code=result["item_code"],
                              source_sha256=result["source"]["sha256"],
                              response_sha256=result["response_sha256"], pdf_sha256=digest(pdf))
        pairing_seconds = time.perf_counter() - before_pair
        write_new_json(output / "bundle.json", bundle)
        _check_pins(inputs)
        _check_pins(code_pins)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
        receipt = {"schema": "same_origin_dwg_pdf_receipt.v1", "case": str(case),
                   "inputs": inputs, "code": code_pins, "git_head_before_run": head,
                   "code_hashes_include_uncommitted_source": True,
                   "environment": {"python": platform.python_version(), "ezdxf": ezdxf.__version__,
                                   "pymupdf": fitz.VersionBind, "platform": platform.platform()},
                   "pdf_file": str(pdf.relative_to(output)), "summary": bundle["summary"],
                   "timing_seconds": {"source_checks_and_code_pins": source_seconds,
                                      "DWG_to_DXF": conversion["seconds"],
                                      "vector_render_and_extent_audit": manifest["render_elapsed_seconds"],
                                      "evidence_pairing": pairing_seconds,
                                      "total": time.perf_counter() - started},
                   "prior_native_read_reused": True, "ocr_invoked": False, "gpu_used": False,
                   "val_test_read": False, "route_labels_read": False,
                   "delivery_baseline_unchanged": True, "route_prediction_improvement_evaluated": False}
        write_new_json(output / "receipt.json", receipt)
        files = sorted(path for path in output.rglob("*") if path.is_file())
        write_new_json(output / "checksums.json", {str(path.relative_to(output)): digest(path)
                                                   for path in files})
    except Exception as error:
        write_new_json(output / "FAILED.json", {"status": "failed_not_complete",
                                                "error_type": type(error).__name__, "reason": str(error)})
        raise
    return receipt


def verify(output: Path) -> dict:
    """Rebuild evidence in a new process, without rerendering or changing inputs."""
    verify_manifest(output)
    receipt = read_json(output / "receipt.json")
    _check_pins(receipt["inputs"])
    _check_pins(receipt["code"])
    case = Path(receipt["case"])
    result, _, _, raw, _ = load_case(case)
    manifest = read_json(output / "evidence/render_manifest.json")
    pdf = contained_path(output, receipt["pdf_file"])
    rebuilt = build_bundle(raw, manifest, item_code=result["item_code"],
                           source_sha256=result["source"]["sha256"],
                           response_sha256=result["response_sha256"], pdf_sha256=digest(pdf))
    if rebuilt != read_json(output / "bundle.json") or rebuilt["summary"] != receipt["summary"]:
        raise ValueError("Independent native/PDF evidence reconstruction differs")
    with fitz.open(pdf) as doc:
        if (len(doc) != 1 or [doc[0].rect.width, doc[0].rect.height] != manifest["page_points"]
                or len(doc[0].get_drawings()) != manifest["vector_paths"]
                or len(doc[0].get_images()) != manifest["embedded_bitmap_count"]):
            raise ValueError("Rendered PDF no longer matches its evidence manifest")
        if pixel_audit(doc[0]) != manifest["pixel_audit"]:
            raise ValueError("Independent PDF pixel check differs")
    if rebuilt["summary"]["inverse_max_error_cad"] is None or rebuilt["summary"]["inverse_max_error_cad"] > 1e-7:
        raise ValueError("Coordinate inverse consistency failed")
    return {"status": "passed_for_same_origin_source_and_coordinates", "output": str(output),
            "summary": rebuilt["summary"], "route_prediction_improvement_evaluated": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "verify"))
    parser.add_argument("--case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run":
        if args.case is None:
            parser.error("--case is required for run")
        result = run(args.case.resolve(), args.output.resolve())
    else:
        result = verify(args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
