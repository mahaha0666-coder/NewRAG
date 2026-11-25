"""A conservative literal check for claimed explicit Chinese action ordering.

This checks the position of words, not manufacturing feasibility or ownership.
An unsupported grammar stays unresolved rather than becoming a process rule.
"""

from __future__ import annotations

import re

# These are transcription anchors, never mappings from requirements to processes.
# In particular, 磨平 is not an alias for 抛光.
ACTION_ANCHORS = (
    ("焊接", ("焊接", "组焊", "焊")),
    ("组焊", ("组焊", "焊接", "焊")),
    ("磨平", ("磨平",)),
    ("抛光", ("抛光",)),
    ("拉丝", ("拉丝",)),
    ("钻孔", ("钻孔", "钻")),
    ("钻底孔", ("钻底孔", "钻")),
    ("攻丝", ("攻丝",)),
    ("镗孔", ("镗孔", "镗")),
    ("铰孔", ("铰孔", "铰")),
    ("车削", ("车削", "车")),
    ("铣削", ("铣削", "铣")),
    ("下料", ("下料",)),
    ("切割", ("切割",)),
    ("倒角", ("倒角",)),
    ("倒钝", ("倒钝",)),
    ("去毛刺", ("去毛刺",)),
    ("清洗", ("清洗",)),
    ("装配", ("装配",)),
    ("淬火", ("淬火",)),
    ("回火", ("回火",)),
    ("退火", ("退火",)),
    ("折弯", ("折弯",)),
    ("喷塑", ("喷塑",)),
)
CONNECTOR = re.compile(
    r"之后|以后|随后|然后|最后|后(?!盖|板|面|端|部|侧|梁|壁|座|轮|轴)|再|才"
)
NEGATION = re.compile(
    r"不(?:需|必|用|要求|得|应|进行)|无需|禁止|严禁|不得|并非|不是|未规定"
)


def _actions(name: str) -> tuple[str, ...]:
    # “焊后磨平” names the smoothing action; “磨平焊缝” does not name welding.
    name = re.split(r"之后|以后|后", name)[-1]
    found = {
        anchor for word, anchors in ACTION_ANCHORS if word in name for anchor in anchors
    }
    if not found:
        # Preserve an exact model action as a literal anchor for other vocabularies.
        found = {name} if len(name) >= 2 else set()
    return tuple(sorted(found, key=lambda value: (-len(value), value)))


def literal_temporal_support(
    relation: dict, operations: dict[str, dict]
) -> dict | None:
    """Require action words on opposite sides of a source temporal connector.

    This is necessary evidence for our supported grammar, not a sufficient test
    of physical scope. The separate semantic review remains required.
    """
    before = _actions(operations[relation["before"]]["name"])
    after = _actions(operations[relation["after"]]["name"])
    # Same action at different stages needs object-specific disambiguation.
    if set(before) & set(after):
        targets = [operations[relation[key]]["target"] for key in ("before", "after")]
        if targets[0] == targets[1] or any(len(target) < 2 for target in targets):
            return None
    else:
        targets = None
    for citation in relation["evidence"]:
        quote = citation["quote"]
        if NEGATION.search(quote):
            continue
        for match in CONNECTOR.finditer(quote):
            left, right = quote[: match.start()], quote[match.end() :]
            if targets and not (targets[0] in left and targets[1] in right):
                continue
            first = next((word for word in before if word in left), None)
            second = next((word for word in after if word in right), None)
            if first and second:
                return {
                    "evidence_id": citation["evidence_id"],
                    "quote": quote,
                    "before_action": first,
                    "connector": match.group(),
                    "after_action": second,
                    "connector_span": list(match.span()),
                    "physical_scope_verified": False,
                }
    return None
