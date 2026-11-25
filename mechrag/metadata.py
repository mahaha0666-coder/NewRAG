"""Convert the text extracted from MechRAG metadata PDFs into evidence units."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .evidence import EvidenceSource, EvidenceStore, EvidenceUnit


NUMBER = r"([-+]?\d+(?:\.\d+)?)"
MODEL_REF = r"(?:this|the)\s+model(?:_\d+)?"

NUMERIC_PATTERNS = {
    "bounding_box_volume": rf"(?:volume of {MODEL_REF}\s+bounding box|bounding box volume)(?:\s+(?:is|=))?\s*{NUMBER}",
    "surface_area": rf"surface area(?: of {MODEL_REF})?\s+(?:is|=)\s*{NUMBER}",
    "gaussian_curvature_variance": rf"variance of (?:the )?gaussian curvature(?: of {MODEL_REF})?\s+(?:is|=)\s*{NUMBER}",
    "mean_curvature_variance": rf"variance of (?:the )?mean curvature(?: of {MODEL_REF})?\s+(?:is|=)\s*{NUMBER}",
    "flat_surface_ratio": rf"flat (?:plane|surface)(?: of {MODEL_REF})?\s+(?:occupies|is|=)\s*{NUMBER}\s*%",
    "curved_surface_ratio": rf"curved surface(?: of {MODEL_REF})?\s+(?:occupies|is|=)\s*{NUMBER}\s*%",
    "straight_edge_ratio": rf"straight edge(?: of {MODEL_REF})?\s+(?:occupies|is|=)\s*{NUMBER}\s*%",
    "curved_edge_ratio": rf"curved edge(?: of {MODEL_REF})?\s+(?:occupies|is|=)\s*{NUMBER}\s*%",
    "hole_count": rf"total number of (?:round )?holes?(?: of {MODEL_REF})?\s+(?:is|=)\s*{NUMBER}",
    "volume": rf"\bvolume of {MODEL_REF}\s+(?:is|=)\s*{NUMBER}",
}


def _part_id(source_uri: str, text: str) -> str:
    match = re.search(r"model_\d+", Path(source_uri).stem, re.I)
    if not match:
        match = re.search(r"model_\d+", text, re.I)
    return match.group(0) if match else Path(source_uri).stem


def _first_group(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.I | re.S)
    return match.group(1).strip() if match else None


def parse_metadata_text(
    source_uri: str,
    text: str,
    *,
    page: int | None = None,
    family_id: str | None = None,
) -> list[EvidenceUnit]:
    """Create small citable records instead of one long document embedding."""

    part_id = _part_id(source_uri, text)
    source = EvidenceSource(
        uri=source_uri,
        page=page,
        extractor="mechrag.metadata.parse_metadata_text",
    )
    units = []
    for property_name, pattern in NUMERIC_PATTERNS.items():
        value = _first_group(pattern, text)
        if value is None:
            continue
        number = float(value)
        if number.is_integer():
            number = int(number)
        units.append(
            EvidenceUnit(
                part_id=part_id,
                feature_id=f"property_{property_name}",
                feature_type="geometric_property",
                geometric_properties={property_name: number},
                sources=(source,),
                summary=f"{part_id}: {property_name}={number}",
                family_id=family_id,
            )
        )

    symmetry = None
    if re.search(rf"{MODEL_REF}\s+has\s+no\s+symmetry", text, re.I):
        symmetry = "none"
    if symmetry is None:
        symmetry = _first_group(
            rf"{MODEL_REF}\s+is\s+([^\n.]*symmetr(?:ical|y))",
            text,
        )
    if symmetry is None:
        symmetry = _first_group(
            rf"(?<!axis of )symmetry(?: of {MODEL_REF})?\s+is\s+([^\n.]+)",
            text,
        )
    if symmetry is not None:
        units.append(
            EvidenceUnit(
                part_id=part_id,
                feature_id="property_symmetry",
                feature_type="geometric_property",
                geometric_properties={"symmetry": symmetry},
                sources=(source,),
                summary=f"{part_id}: symmetry={symmetry}",
                family_id=family_id,
            )
        )

    symmetry_axis = _first_group(
        rf"axis of symmetry of {MODEL_REF}\s+is\s+([^\n]+)", text
    )
    if symmetry_axis is not None:
        symmetry_axis = symmetry_axis.rstrip(".").strip()
        units.append(
            EvidenceUnit(
                part_id=part_id,
                feature_id="property_symmetry_axis",
                feature_type="geometric_property",
                geometric_properties={"symmetry_axis": symmetry_axis},
                sources=(source,),
                summary=f"{part_id}: symmetry_axis={symmetry_axis}",
                family_id=family_id,
            )
        )

    # Modelling operations are split into individually retrievable facts.  The
    # regex intentionally keeps the original wording as the evidence payload.
    operations = re.findall(
        r"(?:set the (?:direction|origin|scale|extent|operation)[^.\n]*|"
        r"create\s*a line ending at[^.\n]*)",
        text,
        re.I,
    )
    for index, operation in enumerate(operations, start=1):
        operation_type = (
            "sketch_operation"
            if re.search(r"create\s*a line", operation, re.I)
            else "extrude_operation"
        )
        units.append(
            EvidenceUnit(
                part_id=part_id,
                feature_id=f"operation_{index:04d}",
                feature_type=operation_type,
                geometric_properties={
                    "modelling_history": operation.strip(),
                    "sequence_index": index,
                },
                sources=(source,),
                summary=operation.strip(),
                family_id=family_id,
            )
        )
    return units


def build_text_evidence_store(
    records: Iterable[tuple[str, str]],
    *,
    expected_part_ids: Iterable[str] = (),
    universe_complete: bool = True,
) -> EvidenceStore:
    units = []
    discovered_parts = set(expected_part_ids)
    for source_uri, text in records:
        discovered_parts.add(_part_id(source_uri, text))
        parsed = parse_metadata_text(source_uri, text)
        units.extend(parsed)
    return EvidenceStore(
        units,
        expected_part_ids=discovered_parts,
        universe_complete=universe_complete,
    )
