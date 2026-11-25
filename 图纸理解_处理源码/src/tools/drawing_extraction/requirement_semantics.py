"""Parse localized requirement text without asserting physical ownership.

The parser records only literal manufacturing semantics.  A textual target
such as ``焊缝`` or ``外表面`` is a target class, not a detected drawing
entity.  Callers must keep the result unresolved until a separate relation
binds the requirement to a localized object.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from pydantic import JsonValue


@dataclass(frozen=True)
class _NamedPattern:
    code: str
    pattern: re.Pattern[str]


_NUMBERED_PREFIX = re.compile(r"^\s*[1-9]\d?\s*(?:[.．、)）]\s*|\s+)")
_CLAUSE_SEPARATOR = re.compile(r"[;；。]+")
_STANDARD = re.compile(
    r"(?P<family>ISO|GB\s*/?\s*T|GB|JB\s*/?\s*T|DIN|EN)\s*"
    r"(?P<number>\d+(?:-\d+)*)"
    r"(?:\s*[-—–]\s*(?P<grade>[A-Za-z0-9]+(?:\s*/\s*[A-Za-z0-9]+)*))?",
    re.IGNORECASE,
)
_INSPECTION_COVERAGE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*%\s*(?P<method>VT|PT|MT|UT|RT)?",
    re.IGNORECASE,
)
_HARDNESS = re.compile(
    r"(?P<scale>HBW?|HBS|HRC|HV)\s*(?P<lower>\d+(?:\.\d+)?)"
    r"(?:\s*[-~～]\s*(?P<upper>\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)
_THICKNESS = re.compile(r"厚度\s*(?P<value>\d+(?:\.\d+)?)", re.IGNORECASE)
_CHAMFER = re.compile(r"(?:未注)?倒角\s*C\s*(?P<value>\d+(?:\.\d+)?)", re.IGNORECASE)
_BEND_RADIUS = re.compile(
    r"(?:未注)?折弯(?:内角)?\s*R\s*(?P<value>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
WELDING_ACTION_PATTERN = re.compile(
    r"满焊|配焊|自动焊|塞焊|焊接"
    r"(?!后|前|尺寸公差|公差|检验|检测|探伤|缺陷|质量|等级|标准|方法|融痕|痕|疤|处|部位|符号|要求)"
)


_ACTION_PATTERNS = (
    _NamedPattern("joining_welding", WELDING_ACTION_PATTERN),
    _NamedPattern("automatic_welding", re.compile(r"自动焊")),
    _NamedPattern("inspection", re.compile(r"检验|检测|探伤|\b(?:VT|PT|MT|UT|RT)\b", re.IGNORECASE)),
    _NamedPattern("grind_flush", re.compile(r"磨平|修平|打磨")),
    _NamedPattern("edge_finishing", re.compile(r"锐[边棱]倒钝|去(?:除)?(?:尖角)?毛刺|去锐边毛刺")),
    _NamedPattern("chamfering", re.compile(r"(?:未注)?倒角\s*C\s*\d+(?:\.\d+)?", re.IGNORECASE)),
    _NamedPattern("polishing", re.compile(r"抛光")),
    _NamedPattern("mechanical_brushing", re.compile(r"机械拉丝|拉丝")),
    _NamedPattern("heat_treatment", re.compile(r"热处理|调质|淬火|退火|回火")),
    _NamedPattern("surface_treatment", re.compile(r"防锈|镀铬|喷塑|喷漆|涂装|钝化|镀锌")),
    _NamedPattern("machining", re.compile(r"机加工|精加工|配车|车削|铣削|镗削|磨削")),
    _NamedPattern(
        "forming_or_straightening",
        re.compile(r"折弯|弯曲|调平|整形|校正|压平"),
    ),
    _NamedPattern("fit_or_match", re.compile(r"配作|配车|紧配|贴紧")),
)

_REQUIREMENT_PATTERNS = (
    _NamedPattern("general_tolerance", re.compile(r"未注(?:机加工)?(?:尺寸|形位)?公差|ISO\s*2768|GB\s*/?\s*T\s*1804", re.IGNORECASE)),
    _NamedPattern("welding_tolerance", re.compile(r"焊接(?:尺寸)?公差|ISO\s*13920", re.IGNORECASE)),
    _NamedPattern("welding_quality", re.compile(r"焊缝|焊接缺陷|假焊|裂纹|ISO\s*5817", re.IGNORECASE)),
    _NamedPattern("inspection_requirement", re.compile(r"检验|检测|探伤|\d+\s*%\s*(?:VT|PT|MT|UT|RT)", re.IGNORECASE)),
    _NamedPattern("edge_quality", re.compile(r"锐[边棱]倒钝|去(?:除)?(?:尖角)?毛刺|去锐边毛刺")),
    _NamedPattern("default_chamfer", re.compile(r"(?:未注)?倒角\s*C\s*\d+(?:\.\d+)?", re.IGNORECASE)),
    _NamedPattern(
        "default_bend_radius",
        re.compile(r"(?:未注)?折弯(?:内角)?\s*R\s*\d+(?:\.\d+)?", re.IGNORECASE),
    ),
    _NamedPattern("surface_finish", re.compile(r"抛光|拉丝|亚光|粗糙度|\bRa\s*\d", re.IGNORECASE)),
    _NamedPattern("surface_treatment", re.compile(r"防锈|镀铬|喷塑|喷漆|涂装|钝化|镀锌")),
    _NamedPattern("heat_treatment_or_hardness", re.compile(r"热处理|调质|淬火|退火|回火|\b(?:HBW?|HBS|HRC|HV)\s*\d", re.IGNORECASE)),
    _NamedPattern("dimensional_or_geometric_alignment", re.compile(r"同一平面|对称|平面度|同轴度|垂直度|平行度|位置度")),
    _NamedPattern("fit_requirement", re.compile(r"配作|配车|紧配|贴紧")),
    _NamedPattern("appearance_defect_limit", re.compile(r"不允许有|不得有|无明显|影响外观|缺陷")),
)

_TARGET_PATTERNS = (
    _NamedPattern("current_drawing_subject", re.compile(r"本件|本零件|本图所示零件|整件")),
    _NamedPattern("weld_seam", re.compile(r"焊缝|对接焊缝|角焊缝|焊疤")),
    _NamedPattern("outer_surface", re.compile(r"外表面")),
    _NamedPattern("inner_surface", re.compile(r"内表面")),
    _NamedPattern("machined_dimension", re.compile(r"机加工(?:未注)?尺寸|机加工尺寸")),
    _NamedPattern("mounting_surface", re.compile(r"安装面")),
    _NamedPattern("edge", re.compile(r"锐边|锐棱|尖角|毛刺|倒角")),
    _NamedPattern("hole", re.compile(r"开孔|通孔|盲孔|螺纹孔")),
    _NamedPattern("internal_diameter", re.compile(r"内径")),
    _NamedPattern("joint", re.compile(r"连接处|拼接处|接触处")),
    _NamedPattern("pipe", re.compile(r"管道|管件|管路")),
    _NamedPattern("bend_inner_corner", re.compile(r"折弯内角")),
    _NamedPattern("specified_region", re.compile(r"指定区域|局部")),
)

_SCOPE_PATTERNS = (
    _NamedPattern("current_part", re.compile(r"本件|本零件|本图所示零件|整件")),
    _NamedPattern("child_part", re.compile(r"所有.{0,8}零件|全部.{0,8}零件|各零件|子件")),
    _NamedPattern("assembly_context", re.compile(r"总成|组件|装配时|安装时|配焊|配钻|配作")),
)

_TEMPORAL_PATTERNS = (
    _NamedPattern("after_welding", re.compile(r"焊后|焊接(?:完成)?后")),
    _NamedPattern("after_grinding", re.compile(r"磨平后|打磨后|磨削后")),
    _NamedPattern("after_machining", re.compile(r"机?加工后|精加工后")),
    _NamedPattern("before_welding", re.compile(r"焊前|焊接前")),
    _NamedPattern("before_machining", re.compile(r"机?加工前|精加工前")),
    _NamedPattern("after_assembly", re.compile(r"装配后|安装后")),
)


def _normalized_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def _matches(text: str, patterns: tuple[_NamedPattern, ...]) -> list[dict[str, JsonValue]]:
    output: list[dict[str, JsonValue]] = []
    for entry in patterns:
        match = entry.pattern.search(text)
        if match is None:
            continue
        output.append({
            "code": entry.code,
            "evidence_text": match.group(0),
            "span": [match.start(), match.end()],
        })
    return output


def _standards(text: str) -> list[dict[str, JsonValue]]:
    output: list[dict[str, JsonValue]] = []
    for match in _STANDARD.finditer(text):
        family = re.sub(r"\s+", "", match.group("family")).upper()
        grade = re.sub(r"\s+", "", match.group("grade") or "") or None
        evidence_end = match.end()
        if grade and text[match.end():].startswith("%"):
            joined_coverage = re.fullmatch(r"([A-Za-z]+)(\d+)", grade)
            if joined_coverage is not None:
                grade = joined_coverage.group(1)
                evidence_end -= len(joined_coverage.group(2))
        output.append({
            "designation": f"{family}{match.group('number')}",
            "grade_or_class": grade,
            "evidence_text": text[match.start():evidence_end],
            "span": [match.start(), evidence_end],
        })
    return output


def _parameters(text: str) -> list[dict[str, JsonValue]]:
    output: list[dict[str, JsonValue]] = []
    for match in _INSPECTION_COVERAGE.finditer(text):
        output.append({
            "kind": "inspection_coverage",
            "value": float(match.group("value")),
            "unit": "percent",
            "method": (match.group("method") or "").upper() or None,
            "evidence_text": match.group(0),
            "span": [match.start(), match.end()],
        })
    for match in _HARDNESS.finditer(text):
        upper = match.group("upper")
        trailing_separator = bool(
            upper is None and re.match(r"\s*[-~～]", text[match.end():])
        )
        output.append({
            "kind": "hardness",
            "scale": match.group("scale").upper(),
            "lower": float(match.group("lower")),
            "upper": float(upper) if upper is not None else None,
            "range_complete": not trailing_separator,
            "evidence_text": match.group(0),
            "span": [match.start(), match.end()],
        })
    for match in _THICKNESS.finditer(text):
        output.append({
            "kind": "thickness",
            "value": float(match.group("value")),
            "unit": None,
            "evidence_text": match.group(0),
            "span": [match.start(), match.end()],
        })
    for match in _CHAMFER.finditer(text):
        output.append({
            "kind": "chamfer_size",
            "value": float(match.group("value")),
            "unit": None,
            "evidence_text": match.group(0),
            "span": [match.start(), match.end()],
        })
    for match in _BEND_RADIUS.finditer(text):
        output.append({
            "kind": "bend_radius",
            "value": float(match.group("value")),
            "unit": None,
            "evidence_text": match.group(0),
            "span": [match.start(), match.end()],
        })
    return output


def _possible_truncation(text: str) -> list[str]:
    compact = text.rstrip()
    reasons = []
    if re.search(r"(?:按|为|厚度|硬度|[-—–《])\s*$", compact):
        reasons.append("terminal_incomplete_phrase")
    if compact.count("《") != compact.count("》"):
        reasons.append("unbalanced_book_title_mark")
    return reasons


def parse_requirement_semantics(text: str | None) -> dict[str, JsonValue]:
    """Return literal semantics while leaving object ownership unresolved."""

    raw = (text or "").strip()
    normalized = _normalized_text(raw)
    raw_body = _NUMBERED_PREFIX.sub("", raw, count=1)
    body = _NUMBERED_PREFIX.sub("", normalized, count=1)
    clauses = [value.strip() for value in _CLAUSE_SEPARATOR.split(body) if value.strip()]
    clause_rows = []
    for clause in clauses or ([body] if body else []):
        actions = _matches(clause, _ACTION_PATTERNS)
        requirement_types = _matches(clause, _REQUIREMENT_PATTERNS)
        targets = _matches(clause, _TARGET_PATTERNS)
        explicit_scopes = _matches(clause, _SCOPE_PATTERNS)
        temporal = _matches(clause, _TEMPORAL_PATTERNS)
        standards = _standards(clause)
        parameters = _parameters(clause)
        if not any((actions, requirement_types, targets, explicit_scopes,
                    temporal, standards, parameters)):
            continue
        target_codes = {str(row["code"]) for row in targets}
        linguistic_scopes = {str(row["code"]) for row in explicit_scopes}
        if target_codes - {"current_drawing_subject"}:
            linguistic_scopes.add("local_feature")
        clause_rows.append({
            "text": clause,
            "action_mentions": actions,
            "requirement_types": requirement_types,
            "standards": standards,
            "parameters": parameters,
            "temporal_conditions": temporal,
            "target_class_mentions": targets,
            "linguistic_scope_candidates": sorted(linguistic_scopes),
            "binding_status": (
                "target_class_only_not_entity_bound" if targets
                else "no_target_class_observed"
            ),
        })
    return {
        "schema_version": "manufacturing_requirement_semantics.v2",
        "raw_text": raw,
        "normalized_text": normalized,
        "raw_body_text": raw_body,
        "body_text": body,
        "clauses": clause_rows,
        "possible_truncation_reasons": _possible_truncation(body),
        "parse_status": "structured_candidate" if clause_rows else "no_semantic_match",
        "physical_target_binding_status": "unresolved",
        "hard_constraint_eligible": False,
    }


__all__ = ["WELDING_ACTION_PATTERN", "parse_requirement_semantics"]
