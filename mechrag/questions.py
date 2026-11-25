"""Question decomposition into conditions that evidence must cover."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence


@dataclass(frozen=True)
class QueryCondition:
    condition_id: str
    description: str
    property_names: tuple[str, ...]
    weight: float = 1.0
    scope: str = "local"
    exhaustive: bool = False


@dataclass(frozen=True)
class QueryPlan:
    question: str
    conditions: tuple[QueryCondition, ...]
    aggregate_operation: str | None = None
    aggregate_property: str | None = None

    @property
    def requires_exhaustive_scan(self) -> bool:
        return any(condition.exhaustive for condition in self.conditions)


DEFAULT_PROPERTY_PATTERNS: Mapping[str, Sequence[str]] = {
    "bounding_box_volume": (
        r"bounding\s+box\s+volume",
        r"包围盒体积",
    ),
    "volume": (r"(?<!box\s)\bvolume\b", r"(?<!包围盒)体积"),
    "surface_area": (r"surface\s+area", r"表面积"),
    "hole_count": (r"(?:number|count)\s+of\s+(?:round\s+)?holes?", r"孔(?:洞)?数量", r"几个孔"),
    "hole_type": (r"through\s+hole", r"blind\s+hole", r"通孔", r"盲孔"),
    "coaxial": (r"coaxial", r"同轴"),
    "symmetry": (r"symmetr", r"对称"),
    "gaussian_curvature_variance": (r"gaussian\s+curvature", r"高斯曲率"),
    "mean_curvature_variance": (r"mean\s+curvature", r"平均曲率"),
    "flat_surface_ratio": (r"flat\s+(?:plane|surface)", r"平面占比"),
    "curved_surface_ratio": (r"curved\s+surface", r"曲面占比"),
    "modelling_history": (r"modell?ing\s+history", r"建模历史"),
    "manufacturing_process": (r"manufacturing\s+process", r"制造工艺", r"加工工艺"),
}


class RuleBasedQuestionDecomposer:
    """Deterministic baseline; aliases can be extended for a project ontology."""

    def __init__(
        self, property_patterns: Mapping[str, Sequence[str]] = DEFAULT_PROPERTY_PATTERNS
    ) -> None:
        self._patterns = {
            name: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
            for name, patterns in property_patterns.items()
        }

    def decompose(self, question: str) -> QueryPlan:
        normalized = " ".join(question.split())
        properties = [
            name
            for name, patterns in self._patterns.items()
            if any(pattern.search(normalized) for pattern in patterns)
        ]

        operation = self._operation(normalized)
        global_scope = bool(
            operation
            or re.search(
                r"\b(all|every|global|list)\b|所有|全部|全局|排序|排名",
                normalized,
                re.IGNORECASE,
            )
        )
        conditions = []
        if not properties:
            conditions.append(
                QueryCondition(
                    condition_id="C1",
                    description=normalized,
                    property_names=(),
                    scope="global" if global_scope else "local",
                    exhaustive=False,
                )
            )
        else:
            for index, property_name in enumerate(properties, start=1):
                exhaustive = global_scope and operation is not None
                conditions.append(
                    QueryCondition(
                        condition_id=f"C{index}",
                        description=f"Verify {property_name} for the requested scope",
                        property_names=(property_name,),
                        scope="global" if global_scope else "local",
                        exhaustive=exhaustive,
                    )
                )

        aggregate_property = properties[0] if operation and len(properties) == 1 else None
        return QueryPlan(
            question=question,
            conditions=tuple(conditions),
            aggregate_operation=operation if aggregate_property else None,
            aggregate_property=aggregate_property,
        )

    @staticmethod
    def _operation(question: str) -> str | None:
        if re.search(r"how many different|number of different|多少种|几种", question, re.I):
            return "count_distinct"
        if re.search(r"\b(max(?:imum)?|largest|greatest|highest)\b|最大|最高", question, re.I):
            return "max"
        if re.search(r"\b(min(?:imum)?|smallest|lowest)\b|最小|最低", question, re.I):
            return "min"
        if re.search(r"descending|降序", question, re.I):
            return "sort_desc"
        if re.search(r"ascending|升序", question, re.I):
            return "sort_asc"
        return None
