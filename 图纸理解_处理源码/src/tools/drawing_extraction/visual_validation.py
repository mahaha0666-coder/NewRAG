"""Validate untrusted visual proposals without certifying physical truth."""

import json

def validate_response(response):
    if response.status_code != 200:
        raise ValueError(f"Model HTTP {response.status_code}")
    return parse_completion(response.json())


def parse_completion(payload):
    choice = payload["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError("Truncated response")
    value = json.loads(choice["message"]["content"])
    if not isinstance(value, dict) or value.get("needs_review") is not True:
        raise ValueError("Response must be a review-required JSON object")
    return value


def validate_anchor(value, target_id):
    if not isinstance(value, dict):
        raise ValueError("Observation must be a JSON object")
    if value.get("target_id") != target_id:
        raise ValueError("Model changed the query identity")
    if not any(value.get("target_visible") is v for v in (True, False, None)) or "target_visible" not in value:
        raise ValueError("Invalid target visibility")
    if value.get("axis_in_image") not in {"horizontal", "vertical", "oblique", "curved", "mixed", "unknown"}:
        raise ValueError("Invalid drawing-plane axis")
    for key in ("visible_outline", "axis_evidence"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"Missing observation field: {key}")
    for key in ("adjacent_shapes", "uncertain_interpretations"):
        if not isinstance(value.get(key), list) or not all(isinstance(v, str) for v in value[key]):
            raise ValueError(f"Invalid observation list: {key}")
    if value.get("needs_review") is not True:
        raise ValueError("Observation cannot bypass review")
    return value


def validate_geometry(value, target_id, candidates):
    validate_anchor(value, target_id)
    for field, allowed in (("selected_region_id", {"connected", "enclosed_face"}),
                           ("axis_evidence_id", {"gapped_collinear"})):
        if field not in value:
            raise ValueError(f"Missing candidate reference: {field}")
        ref = value[field]
        if ref is not None and (not isinstance(ref, str) or ref not in candidates or candidates[ref]["kind"] not in allowed):
            raise ValueError(f"Invalid candidate reference: {field}")
    return value


def audit_anchor_pair(raw, marked):
    """Flag instability, never select an answer or certify physical correctness."""
    issues = []
    if raw.get("target_id") != marked.get("target_id"):
        issues.append("query_identity_mismatch")
    for name, record in (("raw", raw), ("marked", marked)):
        try:
            if record.get("status") != "unverified_response":
                raise ValueError("Failed inference")
            validate_anchor(record["proposal"], record["target_id"])
        except (ValueError, KeyError, TypeError):
            issues.append(f"{name}_invalid_response")
    if not issues:
        left, right = raw["proposal"], marked["proposal"]
        for field in ("axis_in_image", "target_visible"):
            if left[field] != right[field]:
                issues.append(f"{field}_disagreement")
        if left["axis_in_image"] == "unknown" or right["axis_in_image"] == "unknown":
            issues.append("axis_unresolved")
        if left["target_visible"] is not True or right["target_visible"] is not True:
            issues.append("target_visibility_unresolved")
    return {"target_id": raw.get("target_id"), "issues": issues,
            "status": "review_required" if issues else "no_checked_conflict_not_verified",
            "checked_fields": ["target_id", "axis_in_image", "target_visible"],
            "semantic_outline_checked": False, "selected_arm": None,
            "needs_review": True, "physical_part_id": None}

__all__ = ["validate_response", "parse_completion", "validate_anchor", "validate_geometry", "audit_anchor_pair"]
