"""Share one source-verified DWG-derived PDF between display and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tools.drawing_extraction.native_dwg_batch.artifacts import digest, read_json
from src.tools.drawing_extraction.native_dwg_understanding.pdf_surface import render_generated_pdf


def generate(attempt: Path) -> dict[str, Any]:
    from research.dwg_route_study_20261105.paired_evidence import run as render_pair

    pair = attempt / "pair"
    receipt = render_pair(attempt / "native", pair)
    return pair_record(pair, receipt)


def pair_record(pair: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    """Inventory a completed conversion without copying its source or geometry."""
    paths = [receipt["pdf_file"], "receipt.json", "bundle.json", "evidence/render_manifest.json"]
    return {"directory": "pair", "artifacts": {name: digest(pair / name) for name in paths},
            "seconds": receipt["timing_seconds"]["total"]}


def verified_pair(attempt: Path, result: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    record = result["generated_pdf"]
    if record["directory"] != "pair":
        raise ValueError("Unexpected generated PDF directory")
    pair = attempt / "pair"
    if pair.is_symlink():
        raise ValueError("Generated PDF directory cannot be linked")
    for name, expected in record["artifacts"].items():
        path = pair / name
        if (path.is_symlink() or not path.resolve().is_relative_to(pair.resolve())
                or digest(path) != expected):
            raise ValueError("Generated PDF evidence changed")
    receipt = read_json(pair / "receipt.json")
    required = {receipt["pdf_file"], "receipt.json", "bundle.json", "evidence/render_manifest.json"}
    if set(record["artifacts"]) != required:
        raise ValueError("Generated PDF evidence inventory is incomplete")
    pdf = pair / receipt["pdf_file"]
    manifest = read_json(pair / "evidence/render_manifest.json")
    bundle = read_json(pair / "bundle.json")
    source_hash = result["source"]["sha256"]
    if (manifest["source"]["sha256"] != source_hash
            or bundle["source"]["sha256"] != source_hash
            or bundle["derived_pdf"]["derived_from_sha256"] != source_hash
            or bundle["derived_pdf"]["sha256"] != record["artifacts"][receipt["pdf_file"]]
            or not manifest["automatic_full_extent"]):
        raise ValueError("Generated PDF does not belong to the uploaded DWG")
    return pair, pdf, manifest


def load_surface(attempt: Path, result: dict[str, Any]) -> dict[str, Any] | None:
    if not result.get("generated_pdf"):
        return None
    _, pdf, manifest = verified_pair(attempt, result)
    return {**render_generated_pdf(pdf.read_bytes(), manifest),
            "filename": Path(result["source"]["item_code"]).name + ".pdf",
            "sha256": digest(pdf), "origin": "generated_from_dwg",
            "dwg_source_sha256": result["source"]["sha256"],
            "revision_correspondence_confirmed": True,
            "revision_confirmation_basis": "same_source_conversion",
            "evidence_independence_confirmed": False}
