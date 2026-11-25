"""Source-based colors for native vectors; no inferred physical-part identity."""

from __future__ import annotations

from collections import Counter
from copy import copy
from dataclasses import dataclass

from ezdxf.addons.drawing import Frontend
from ezdxf.addons.drawing.mtext_complex import ComplexMTextRenderer
from ezdxf.addons.drawing.properties import Properties
from ezdxf.document import Drawing
from ezdxf.entities import DXFGraphic, MText
from ezdxf.tools.text import MTextContext
from ezdxf.tools import text_layout


@dataclass(frozen=True)
class LineStyle:
    label: str
    color: str
    description: str


STYLES = {
    "contour": LineStyle("轮廓实线", "#2563EB", "原生轮廓图层；不是已确认的部件分割"),
    "dimension": LineStyle("尺寸与公差", "#00908D", "尺寸对象、尺寸显示块及原生标注图层"),
    "center": LineStyle("中心线", "#CC7C0B", "原生中心线图层或CENTER线型"),
    "hatch": LineStyle("剖面线", "#398A67", "HATCH实体及原生剖面线图层"),
    "leader": LineStyle("引线与焊接", "#8446C8", "LEADER、件号引线及焊接符号原生预览"),
    "gdt": LineStyle("形位与基准", "#BC367B", "原生形位符号及基准对象预览"),
    "section": LineStyle("剖切标识", "#0085B6", "原生剖切符号预览"),
    "hidden": LineStyle("虚线与辅助线", "#61758D", "虚线、细实线或双点划线图层"),
    "text": LineStyle("文字与说明", "#183451", "文字图层、TEXT及MTEXT"),
    "neutral": LineStyle("图框 / 明细 / 未分类", "#78869A", "图框、表格与缺少类别依据的实体"),
}
PROXY_ROLES = {
    "XTWELDSYMBOL": "leader", "TDMLABEL": "leader", "XTDOUBLETEXT": "leader",
    "XTSHAPEPOSITION": "gdt", "XTBENCHMARK": "gdt", "XTCUTSYMBOL": "section",
    "XTCHAMFER": "dimension", "TDMFRAME": "neutral", "TDMTITLE": "neutral",
    "TDMBOMITEM": "neutral",
}


class _SemanticTextRenderer(ComplexMTextRenderer):
    def new_text_properties(self, properties: Properties, ctx: MTextContext) -> Properties:
        # ACI/inline colors must not bypass the requested display palette.
        resolved = copy(properties)
        resolved.font = ctx.font_face
        return resolved


class ColoredFrontend(Frontend):
    def draw_complex_mtext(self, mtext: MText, properties: Properties) -> None:
        renderer = _SemanticTextRenderer(self.ctx, self.pipeline, properties)
        engine = renderer.layout_engine(mtext)
        engine.place(align=text_layout.LayoutAlignment(mtext.dxf.attachment_point))
        engine.render(mtext.ucs().matrix)


class NativeLineColorizer:
    """Modify display properties only; source entities/layers remain unchanged."""

    def __init__(self, document: Drawing, proxies: list[dict]) -> None:
        self.document = document
        self.proxy_roles = {handle: PROXY_ROLES.get(obj["native_kind"], "neutral")
                            for obj in proxies for handle in obj["emitted_entity_handles"]}
        self.render_counts: Counter[str] = Counter()

    def classify(self, entity: DXFGraphic, properties: Properties) -> tuple[str, str]:
        original = entity.source_of_copy or entity
        handle = original.dxf.get("handle", "")
        if handle in self.proxy_roles:
            return self.proxy_roles[handle], "native_custom_object_class"
        kind = entity.dxftype()
        if kind in {"LEADER", "MLEADER"}:
            return "leader", "native_entity_type"
        if kind == "DIMENSION":
            return "dimension", "native_entity_type"
        if kind == "HATCH":
            return "hatch", "native_entity_type"
        owner = self.document.entitydb.get(original.dxf.get("owner", ""))
        if owner is not None and owner.dxftype() == "BLOCK_RECORD" and owner.dxf.name.startswith("*D"):
            return "dimension", "native_dimension_display_block"
        layer = properties.layer
        if "中心" in layer or "CENTER" in properties.linetype_name.upper():
            return "center", "native_layer_or_linetype"
        if "剖面" in layer:
            return "hatch", "native_layer_name"
        if "标注" in layer:
            return "dimension", "native_layer_name_candidate"
        if "虚线" in layer or "细线" in layer or "细实线" in layer or "点划" in layer:
            return "hidden", "native_layer_name"
        if kind in {"TEXT", "MTEXT", "ATTRIB", "ATTDEF"}:
            return "text", "native_entity_type"
        if "文字" in layer:
            return "text", "native_layer_name"
        if "轮廓" in layer:
            return "contour", "native_layer_name"
        return "neutral", "unclassified_no_semantic_guess"

    def __call__(self, entity: DXFGraphic, properties: Properties) -> None:
        role, _ = self.classify(entity, properties)
        properties.color = STYLES[role].color
        properties.layer = STYLES[role].label
        self.render_counts[role] += 1

    def report(self) -> dict:
        return {"basis": "native_layer_entity_type_dimension_block_or_custom_object",
                "physical_part_identity_inferred": False,
                "dimension_extension_arrow_separately_confirmed": False,
                "render_entity_visit_counts": dict(self.render_counts),
                "styles": {role: {"label": style.label, "color": style.color,
                                   "description": style.description}
                           for role, style in STYLES.items()}}
