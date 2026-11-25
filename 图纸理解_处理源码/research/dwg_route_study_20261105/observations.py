"""Preserve native observations without promoting ownership or manufacturing facts.

This is a document-level research input, not the admitted PDF/geometry relation
channel. It never assigns a symbol to a part or interprets it as an operation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from research.r19_route_estimation.limited_search.models import TextFeatures
from research.r19_route_estimation.limited_search.text_views import input_digest
from research.r19_route_estimation.limited_search.visual_model import VisualFeatures
from src.tools.research.artifacts import digest, read_json

SCHEMA = "native_dwg_document_observations.v1"
FORBIDDEN = frozenset({"operations", "target", "target_route", "training_labels", "labels",
                       "ground_truth", "expected_route", "query_answers", "gt"})
TITLE_FIELDS = frozenset({"common_name", "material", "specification"})
BOM_FIELDS = frozenset({"name", "material", "specification", "quantity", "remark"})


def label_free(value: Any) -> None:
    if isinstance(value, Mapping):
        if FORBIDDEN.intersection(value):
            raise ValueError("Route answers cannot enter document observations")
        for child in value.values():
            label_free(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            label_free(child)


def redact(text: str, identities: Sequence[str]) -> str:
    """Exclude document/item identifiers while preserving short engineering strings."""
    for identity in sorted(set(identities), key=lambda item: (-len(item), item)):
        if len(identity) >= 4:
            text = text.replace(identity, " ")
    text = re.sub(r"[A-Za-z][A-Za-z0-9.]*[#＃][A-Za-z0-9.-]+", " ", text)
    return re.sub(r"(?<!\d)\d{8,}(?!\d)", " ", text).strip()


def make_document(row: dict, ir: dict, status: str) -> dict:
    """Only source-labelled values; IDs and geometry stay evidence, not features."""
    label_free(row)
    code = row["item_code"]
    if ir["source"]["item_code"] != code:
        raise ValueError("Native document identity differs from the fixed sample")
    native_sha = ir["source"]["sha256"]
    drawing = ir.get("drawing_fields", {})
    identities = [code]
    for field in ("item_code", "drawing_number"):
        identities.extend(str(value["value"]) for value in drawing.get(field, []))
    for part in ir.get("parts", []):
        for field in ("drawing_number", "item_code"):
            identities.append(str(part.get("fields", {}).get(field, "")))
            identities.append(str(part.get("field_candidates", {}).get(field, "")))

    title_codes = {str(value["value"]).strip() for value in drawing.get("item_code", [])}
    title_identity = "matches" if title_codes == {code} else "unknown" if not title_codes else "conflict"
    fields: list[dict[str, Any]] = []

    def add(kind: str, category: str, value: str, ref: dict, state: str,
            evidence: dict | None = None) -> None:
        if not isinstance(value, str):
            raise ValueError("Native text values must not be coerced to numbers")
        if ref.get("source_sha256") != native_sha or not ref.get("json_pointer"):
            raise ValueError("Observation lost its original native source")
        text = redact(value, identities)
        fields.append({"kind": kind, "category": category, "raw_text": value,
                       "feature_text": text, "status": state, "scope": "unresolved",
                       "source_ref": ref, "evidence": evidence or {},
                       "physical_owner_confirmed": False, "hard_constraint_eligible": False})

    for field in sorted(TITLE_FIELDS):
        for value in drawing.get(field, []):
            add("title_candidate", field, value["value"], value["source_ref"], value["status"])
    for part in ir.get("parts", []):
        for container, state in ((part.get("fields", {}), "native_cell_binding"),
                                  (part.get("field_candidates", {}), "candidate_field_binding")):
            for field in sorted(BOM_FIELDS.intersection(container)):
                value = container[field]
                if value:
                    add("bom_row", field, value, part["source_ref"], state,
                        {"row_id": part["id"], "field": field})
    for dimension in ir.get("dimensions", []):
        # Native numerical measurement is not a substitute for the displayed value.
        if dimension.get("display_text_verified"):
            value, state = dimension["display_text"], "native_display_text"
        else:
            value, state = dimension.get("raw_text", ""), "unverified_raw_dimension"
        add("dimension", dimension["category"], value, dimension["source_ref"], state,
            {"native_kind": dimension["native_kind"], "unit": dimension.get("unit", "unknown"),
             "geometry_relation_count": len(dimension.get("native_geometry_relations", [])),
             "part_candidates": dimension.get("part_candidates", [])})
    for symbol in ir.get("symbols", []):
        add("symbol", symbol["category"], symbol.get("raw_text", ""), symbol["source_ref"],
            "native_symbol_observation", {"native_kind": symbol["native_kind"],
                                          "part_candidates": symbol.get("part_candidates", [])})
    for requirement in ir.get("requirements", []):
        add("requirement", "technical_note", requirement["raw_text"], requirement["source_ref"],
            "literal_clause_scope_unresolved")
    result = {"schema": SCHEMA, "item_code": code, "pdf_source_sha256": row["source_sha256"],
              "baseline_input_sha256": input_digest(row), "dwg_source_sha256": native_sha,
              "dwg_status": status, "title_identity": title_identity,
              "available": status in {"analyzed", "partial"} and title_identity != "conflict",
              "same_revision_status": "not_independently_verified",
              "fields": fields, "hard_operation_constraints": 0}
    label_free(result)
    return result


def missing_document(row: dict, status: str) -> dict:
    label_free(row)
    return {"schema": SCHEMA, "item_code": row["item_code"], "available": False,
            "baseline_input_sha256": input_digest(row), "pdf_source_sha256": row["source_sha256"],
            "dwg_status": status, "title_identity": "unknown", "fields": [],
            "same_revision_status": "not_independently_verified", "hard_operation_constraints": 0}


def raw_text(document: dict) -> str:
    return "\n".join(field["feature_text"] for field in document["fields"] if field["feature_text"])


def typed_values(document: dict) -> dict[str, float]:
    """Observed field roles and values, not inferred holes, methods, or part owners."""
    result: Counter[str] = Counter()
    for field in document["fields"]:
        prefix = f"{field['kind']}/{field['category']}/{field['status']}"
        result[prefix + "/observed"] += 1
        # Keep the same literal values as the raw arm, now tagged with their role.
        for token in re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+|[+-]?\d+(?:\.\d+)?|[Ø±°×]", field["feature_text"]):
            result[prefix + "/value=" + token] += 1
        if field["kind"] == "dimension":
            evidence = field["evidence"]
            result[prefix + "/native_kind=" + evidence["native_kind"]] += 1
            result[prefix + "/unit=" + str(evidence["unit"])] += 1
            if evidence["geometry_relation_count"]:
                result[prefix + "/has_explicit_geometry_relation_scope_unknown"] += 1
    return {key: math.log1p(value) for key, value in result.items()}


@dataclass
class DocumentFeatures:
    base: TextFeatures | VisualFeatures
    raw_vectorizer: TfidfVectorizer
    typed_vectorizer: DictVectorizer | None
    documents_path: str
    documents_sha256: str
    fitting_statistics: dict[str, Any]

    @staticmethod
    def documents(rows: Sequence[dict], path: str, sha256: str) -> list[dict]:
        source = Path(path)
        if digest(source) != sha256:
            raise ValueError("Frozen native document observations changed")
        documents = read_json(source)
        result = []
        for row in rows:
            label_free(row)
            document = documents[row["item_code"]]
            label_free(document)
            if (document["schema"] != SCHEMA or document["item_code"] != row["item_code"]
                    or document["baseline_input_sha256"] != input_digest(row)
                    or document["pdf_source_sha256"] != row["source_sha256"]
                    or document["hard_operation_constraints"] != 0):
                raise ValueError("Native observations do not belong to this frozen input")
            result.append(document if document["available"] else {**document, "fields": []})
        return result

    @classmethod
    def fit(cls, rows: list[dict], parameters: dict) -> DocumentFeatures:
        mode = parameters["dwg_document_mode"]
        if mode not in {"raw", "typed"}:
            raise ValueError("Unknown fixed native observation arm")
        path, sha256 = parameters["dwg_documents_path"], parameters["dwg_documents_sha256"]
        documents = cls.documents(rows, path, sha256)
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3),
                                     max_features=12000, sublinear_tf=True)
        vectorizer.fit([raw_text(document) for document in documents] + [" "])
        typed = DictVectorizer(dtype=float) if mode == "typed" else None
        if typed is not None:
            typed.fit([typed_values(document) for document in documents])
        factory = VisualFeatures if parameters.get("visual_path") else TextFeatures
        statistics = {"mode": mode, "fitting_codes": [row["item_code"] for row in rows],
                      "available_count": sum(document["available"] for document in documents),
                      "query_statistics_used": False, "identifiers_used_as_features": False,
                      "raw_columns": len(vectorizer.vocabulary_),
                      "typed_columns": len(typed.vocabulary_) if typed is not None else 0}
        return cls(factory.fit(rows, parameters), vectorizer, typed, path, sha256, statistics)

    def transform(self, rows: Sequence[dict]) -> csr_matrix:
        documents = self.documents(rows, self.documents_path, self.documents_sha256)
        blocks = [self.base.transform(rows),
                  self.raw_vectorizer.transform([raw_text(document) for document in documents])]
        if self.typed_vectorizer is not None:
            blocks.append(normalize(self.typed_vectorizer.transform([typed_values(document) for document in documents])))
        # Original base block is bit-identical. New text has its own fitted encoder;
        # no stale embedding is claimed to encode the newly extracted DWG content.
        return hstack(blocks, format="csr")
