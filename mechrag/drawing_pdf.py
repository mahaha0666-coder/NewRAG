"""PDF drawing preparation and evidence conversion for MechRAG.

Engineering drawings in the user's dataset have little or no usable PDF text
layer.  This module therefore renders every page, creates high-resolution
local crops, and converts a vision model's structured observations into the
same :class:`EvidenceUnit` records used by the evidence-first pipeline.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image
from pypdf import PdfReader

from .evidence import BoundingBox, EvidenceSource, EvidenceUnit, ViewRegion
from .questions import QueryCondition


DRAWING_CONDITIONS: tuple[QueryCondition, ...] = (
    QueryCondition(
        "C1",
        "识别指定实体并按物理实体计数；同一特征在不同视图中不得重复计数",
        ("entity_identification_and_count",),
    ),
    QueryCondition(
        "C2",
        "提取并解释标题栏、参数表、技术要求和物料清单中的可见内容",
        ("document_text_and_tables",),
    ),
    QueryCondition(
        "C3",
        "根据主视图、剖视图、局部详图和视图组织推断结构与语义关系",
        ("drawing_structure_and_semantics",),
    ),
    QueryCondition(
        "C4",
        "根据图中明确证据推断装配、连接、配合、相对位置、功能作用和零件依赖",
        ("assembly_relationships",),
    ),
)


PROPERTY_BY_CONDITION = {
    condition.condition_id: condition.property_names[0]
    for condition in DRAWING_CONDITIONS
}


@dataclass(frozen=True)
class DrawingAsset:
    """One model-visible image tied back to a PDF page region."""

    asset_id: str
    document_id: str
    pdf_path: str
    page_number: int
    image_path: str
    full_page_bbox: tuple[float, float, float, float]
    role: str
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "document_id": self.document_id,
            "pdf_path": self.pdf_path,
            "page_number": self.page_number,
            "image_path": self.image_path,
            "full_page_bbox": list(self.full_page_bbox),
            "role": self.role,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class DrawingDocument:
    document_id: str
    pdf_path: str
    page_count: int
    extracted_text_chars: int
    text_layer_usable: bool
    assets: tuple[DrawingAsset, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "pdf_path": self.pdf_path,
            "page_count": self.page_count,
            "extracted_text_chars": self.extracted_text_chars,
            "text_layer_usable": self.text_layer_usable,
            "assets": [asset.to_dict() for asset in self.assets],
        }


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "document"


def _text_layer_status(reader: PdfReader) -> tuple[int, bool]:
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    compact = "".join(text.split())
    if not compact:
        return 0, False
    replacement_ratio = compact.count("\ufffd") / len(compact)
    chinese_or_ascii = sum(
        character.isascii() or "\u4e00" <= character <= "\u9fff"
        for character in compact
    )
    readable_ratio = chinese_or_ascii / len(compact)
    # A timestamp or watermark string is technically extractable but is not a
    # useful drawing text layer.  Require enough content to plausibly include
    # a title block or notes.
    return (
        len(compact),
        len(compact) >= 100 and replacement_ratio < 0.05 and readable_ratio >= 0.8,
    )


def _render_page(
    pdf_path: Path,
    page_number: int,
    output_path: Path,
    *,
    dpi: int,
    renderer: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = output_path.with_suffix("")
    command = [
        renderer,
        "-f",
        str(page_number),
        "-l",
        str(page_number),
        "-singlefile",
        "-png",
        "-r",
        str(dpi),
        str(pdf_path),
        str(prefix),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"Failed to render {pdf_path.name} page {page_number}: "
            f"{completed.stderr.strip()}"
        )


def _dark_content_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    """Find the main dark-ink cluster while ignoring pale diagonal watermarks."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - notebook extras include numpy
        raise ImportError("Drawing crop detection requires numpy") from exc

    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    grid_rows = min(72, max(12, height // 64))
    grid_cols = min(96, max(12, width // 64))
    row_edges = np.linspace(0, height, grid_rows + 1, dtype=int)
    col_edges = np.linspace(0, width, grid_cols + 1, dtype=int)
    active = np.zeros((grid_rows, grid_cols), dtype=bool)
    weights = np.zeros((grid_rows, grid_cols), dtype=int)

    # Values below 175 retain drawing strokes and discard the light watermark.
    dark = gray < 175
    for row in range(grid_rows):
        for col in range(grid_cols):
            block = dark[
                row_edges[row] : row_edges[row + 1],
                col_edges[col] : col_edges[col + 1],
            ]
            count = int(block.sum())
            weights[row, col] = count
            active[row, col] = count >= max(6, int(block.size * 0.0015))

    # Join nearby drawing elements before connected-component selection.
    expanded = active.copy()
    for _ in range(2):
        prior = expanded.copy()
        for row_shift in (-1, 0, 1):
            for col_shift in (-1, 0, 1):
                shifted = np.roll(np.roll(prior, row_shift, axis=0), col_shift, axis=1)
                if row_shift < 0:
                    shifted[row_shift:, :] = False
                elif row_shift > 0:
                    shifted[:row_shift, :] = False
                if col_shift < 0:
                    shifted[:, col_shift:] = False
                elif col_shift > 0:
                    shifted[:, :col_shift] = False
                expanded |= shifted

    seen = np.zeros_like(expanded)
    components: list[tuple[int, int, int, int, int]] = []
    for start_row, start_col in zip(*np.nonzero(expanded)):
        if seen[start_row, start_col]:
            continue
        stack = [(int(start_row), int(start_col))]
        seen[start_row, start_col] = True
        cells = []
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                other_row, other_col = row + delta_row, col + delta_col
                if (
                    0 <= other_row < grid_rows
                    and 0 <= other_col < grid_cols
                    and expanded[other_row, other_col]
                    and not seen[other_row, other_col]
                ):
                    seen[other_row, other_col] = True
                    stack.append((other_row, other_col))
        rows = [cell[0] for cell in cells]
        cols = [cell[1] for cell in cells]
        weight = sum(weights[row, col] for row, col in cells)
        components.append((min(rows), min(cols), max(rows), max(cols), weight))

    if not components:
        return (0, 0, width, height)
    row_min, col_min, row_max, col_max, _ = max(
        components,
        key=lambda component: (component[4], (component[2] - component[0] + 1) * (component[3] - component[1] + 1)),
    )
    left = col_edges[max(0, col_min - 1)]
    top = row_edges[max(0, row_min - 1)]
    right = col_edges[min(grid_cols, col_max + 2)]
    bottom = row_edges[min(grid_rows, row_max + 2)]
    return int(left), int(top), int(right), int(bottom)


def _normalized_bbox(
    bbox: Sequence[int | float], width: int, height: int
) -> tuple[float, float, float, float]:
    left, top, right, bottom = map(float, bbox)
    return (left / width, top / height, right / width, bottom / height)


def _save_model_image(
    image: Image.Image,
    path: Path,
    *,
    max_dimension: int = 2400,
) -> tuple[int, int]:
    converted = image.convert("RGB")
    converted.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    converted.save(path, "JPEG", quality=92, optimize=True)
    return converted.size


def _page_assets(
    pdf_path: Path,
    page_number: int,
    rendered_path: Path,
    assets_dir: Path,
) -> list[DrawingAsset]:
    document_id = pdf_path.stem
    with Image.open(rendered_path) as page_image:
        page_image = page_image.convert("RGB")
        page_width, page_height = page_image.size
        page_prefix = f"{_safe_id(document_id)}_p{page_number}"
        assets: list[DrawingAsset] = []

        overview_path = assets_dir / f"{page_prefix}_overview.jpg"
        overview_width, overview_height = _save_model_image(
            page_image, overview_path, max_dimension=2048
        )
        assets.append(
            DrawingAsset(
                asset_id=f"{page_prefix}_overview",
                document_id=document_id,
                pdf_path=str(pdf_path.resolve()),
                page_number=page_number,
                image_path=str(overview_path.resolve()),
                full_page_bbox=(0.0, 0.0, 1.0, 1.0),
                role="overview",
                width=overview_width,
                height=overview_height,
            )
        )

        content_pixels = _dark_content_bbox(page_image)
        left, top, right, bottom = content_pixels
        content_bbox = _normalized_bbox(content_pixels, page_width, page_height)
        content_image = page_image.crop(content_pixels)
        content_path = assets_dir / f"{page_prefix}_content.jpg"
        content_width, content_height = _save_model_image(content_image, content_path)
        assets.append(
            DrawingAsset(
                asset_id=f"{page_prefix}_content",
                document_id=document_id,
                pdf_path=str(pdf_path.resolve()),
                page_number=page_number,
                image_path=str(content_path.resolve()),
                full_page_bbox=content_bbox,
                role="content",
                width=content_width,
                height=content_height,
            )
        )

        # Four overlapping quadrants retain small title-block, BOM, dimension,
        # and sectional-view text that becomes illegible in a whole-page image.
        crop_width, crop_height = content_image.size
        if crop_width >= 1400 or crop_height >= 1400:
            x_ranges = ((0, int(crop_width * 0.58)), (int(crop_width * 0.42), crop_width))
            y_ranges = ((0, int(crop_height * 0.58)), (int(crop_height * 0.42), crop_height))
            for row, (tile_top, tile_bottom) in enumerate(y_ranges):
                for col, (tile_left, tile_right) in enumerate(x_ranges):
                    tile = content_image.crop((tile_left, tile_top, tile_right, tile_bottom))
                    tile_path = assets_dir / f"{page_prefix}_tile_{row}_{col}.jpg"
                    tile_width, tile_height = _save_model_image(tile, tile_path)
                    full_left = left + tile_left
                    full_top = top + tile_top
                    full_right = left + tile_right
                    full_bottom = top + tile_bottom
                    assets.append(
                        DrawingAsset(
                            asset_id=f"{page_prefix}_tile_{row}_{col}",
                            document_id=document_id,
                            pdf_path=str(pdf_path.resolve()),
                            page_number=page_number,
                            image_path=str(tile_path.resolve()),
                            full_page_bbox=_normalized_bbox(
                                (full_left, full_top, full_right, full_bottom),
                                page_width,
                                page_height,
                            ),
                            role=f"tile_{row}_{col}",
                            width=tile_width,
                            height=tile_height,
                        )
                    )
        return assets


def prepare_pdf_dataset(
    pdf_dir: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 240,
    renderer: str = "pdftoppm",
) -> tuple[DrawingDocument, ...]:
    """Render a PDF folder and return model-ready, provenance-aware assets."""

    source_dir = Path(pdf_dir)
    destination = Path(output_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"PDF dataset directory not found: {source_dir}")
    renderer_path = shutil.which(renderer)
    if renderer_path is None:
        raise FileNotFoundError(f"PDF renderer not found on PATH: {renderer}")

    documents = []
    for pdf_path in sorted(source_dir.rglob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        text_chars, text_layer_usable = _text_layer_status(reader)
        assets: list[DrawingAsset] = []
        for page_number in range(1, len(reader.pages) + 1):
            rendered_path = (
                destination / "rendered" / f"{_safe_id(pdf_path.stem)}_p{page_number}.png"
            )
            _render_page(
                pdf_path,
                page_number,
                rendered_path,
                dpi=dpi,
                renderer=renderer_path,
            )
            assets.extend(
                _page_assets(
                    pdf_path,
                    page_number,
                    rendered_path,
                    destination / "assets",
                )
            )
        documents.append(
            DrawingDocument(
                document_id=pdf_path.stem,
                pdf_path=str(pdf_path.resolve()),
                page_count=len(reader.pages),
                extracted_text_chars=text_chars,
                text_layer_usable=text_layer_usable,
                assets=tuple(assets),
            )
        )
    return tuple(documents)


def image_data_url(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def build_drawing_prompt(document: DrawingDocument) -> str:
    asset_manifest = [
        {
            "asset_id": asset.asset_id,
            "page": asset.page_number,
            "role": asset.role,
            "full_page_bbox": list(asset.full_page_bbox),
        }
        for asset in document.assets
    ]
    condition_manifest = [
        {
            "condition_id": condition.condition_id,
            "description": condition.description,
            "property": condition.property_names[0],
        }
        for condition in DRAWING_CONDITIONS
    ]
    return f"""你是一名机械工程图审查员。以下所有图片均来自同一份 PDF 图纸
{document.document_id}，包括整页图和用于读取小字的重叠局部图。

图像资产及其在原始整页中的归一化位置：
{json.dumps(asset_manifest, ensure_ascii=False, indent=2)}

需要回答的验证条件：
{json.dumps(condition_manifest, ensure_ascii=False, indent=2)}

严格规则：
1. 只能依据图中可见证据；模糊、缺失或不能从单张零件图推出的内容必须明确写“不可验证”。
2. 同一个孔、零件或结构在主视、俯视、剖视和局部详图中重复出现时，只计一个物理实体。
3. 区分“BOM 行项目数”和“按数量列求和后的实例总数”，并说明采用哪一种。
4. 装配/配合结论必须引用尺寸、公差、紧固件、焊缝、BOM 或剖视关系；不得仅凭外形相似猜测。
5. 每个结果必须给出至少一个 evidence_regions。bbox 是相对于所引用 asset 的
   [x_min,y_min,x_max,y_max] 归一化坐标，取值范围 0 到 1。
6. 每个 condition_id 恰好返回一项；feature_id 分别固定为
   entity_inventory、document_text、drawing_structure、assembly_relations。
7. 返回严格 JSON，不要使用 Markdown 代码块，也不要在 JSON 外添加文字。

返回结构：
{{
  "document_id": "{document.document_id}",
  "results": [
    {{
      "condition_id": "C1",
      "feature_id": "entity_inventory",
      "answer": "中文结论",
      "facts": {{"结构化字段": "值"}},
      "evidence_regions": [
        {{"asset_id": "资产ID", "bbox": [0.0,0.0,1.0,1.0], "note": "该区域支持什么"}}
      ],
      "confidence": 0.0
    }}
  ],
  "limitations": ["无法验证的原因"]
}}
"""


def build_drawing_message_content(document: DrawingDocument) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": build_drawing_prompt(document)}
    ]
    for asset in document.assets:
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"ASSET {asset.asset_id}: page={asset.page_number}, "
                        f"role={asset.role}, full_page_bbox={asset.full_page_bbox}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(asset.image_path)},
                },
            ]
        )
    return content


def _clean_json_response(response: str) -> Mapping[str, Any]:
    stripped = response.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    payload = json.loads(stripped)
    if not isinstance(payload, Mapping):
        raise ValueError("Drawing response must be a JSON object")
    return payload


def _to_full_page_bbox(
    asset_bbox: Sequence[float], local_bbox: Sequence[float]
) -> BoundingBox:
    if len(asset_bbox) != 4 or len(local_bbox) != 4:
        raise ValueError("Every bounding box must contain four coordinates")
    asset_left, asset_top, asset_right, asset_bottom = map(float, asset_bbox)
    left, top, right, bottom = map(float, local_bbox)
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError(f"Invalid local bounding box: {local_bbox}")
    width = asset_right - asset_left
    height = asset_bottom - asset_top
    return BoundingBox(
        asset_left + left * width,
        asset_top + top * height,
        asset_left + right * width,
        asset_top + bottom * height,
    )


def response_to_evidence(
    document: DrawingDocument,
    response: str | Mapping[str, Any],
) -> tuple[tuple[EvidenceUnit, ...], Mapping[str, Any]]:
    """Validate model JSON and convert it to four citable evidence units."""

    payload = _clean_json_response(response) if isinstance(response, str) else response
    if str(payload.get("document_id")) != document.document_id:
        raise ValueError("Response document_id does not match the requested document")
    asset_map = {asset.asset_id: asset for asset in document.assets}
    overview_by_page = {
        asset.page_number: asset
        for asset in document.assets
        if asset.role == "overview"
    }
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Response must contain a results array")
    if len(results) != len(PROPERTY_BY_CONDITION):
        raise ValueError("Response must contain exactly four condition results")
    by_condition = {str(item.get("condition_id")): item for item in results}
    missing = set(PROPERTY_BY_CONDITION) - set(by_condition)
    extra = set(by_condition) - set(PROPERTY_BY_CONDITION)
    if missing or extra:
        raise ValueError(f"Unexpected condition coverage; missing={missing}, extra={extra}")

    units = []
    expected_feature_ids = {
        "C1": "entity_inventory",
        "C2": "document_text",
        "C3": "drawing_structure",
        "C4": "assembly_relations",
    }
    for condition_id in ("C1", "C2", "C3", "C4"):
        item = by_condition[condition_id]
        feature_id = str(item.get("feature_id", ""))
        if feature_id != expected_feature_ids[condition_id]:
            raise ValueError(
                f"{condition_id} must use feature_id={expected_feature_ids[condition_id]}"
            )
        regions = []
        sources = []
        for index, region in enumerate(item.get("evidence_regions") or []):
            asset_id = str(region.get("asset_id", ""))
            if asset_id not in asset_map:
                raise ValueError(f"Unknown asset_id in response: {asset_id}")
            asset = asset_map[asset_id]
            bbox = _to_full_page_bbox(asset.full_page_bbox, region.get("bbox", []))
            page_image = overview_by_page.get(asset.page_number)
            if page_image is None:
                raise ValueError(f"Missing overview asset for page {asset.page_number}")
            regions.append(
                ViewRegion(
                    view_id=f"page_{asset.page_number}:{asset.role}:{index + 1}",
                    # The transformed bbox is in full-page coordinates, so the
                    # URI must point at the full-page overview rather than at
                    # the local tile used to read the feature.
                    image_uri=page_image.image_path,
                    bbox=bbox,
                )
            )
            sources.append(
                EvidenceSource(
                    uri=document.pdf_path,
                    page=asset.page_number,
                    record_id=f"{condition_id}:{index + 1}",
                    extractor="deepseek-flash-vision",
                )
            )
        if not regions:
            raise ValueError(f"{condition_id} contains no evidence region")
        property_name = PROPERTY_BY_CONDITION[condition_id]
        confidence = float(item.get("confidence", 0.0))
        if not 0 <= confidence <= 1:
            raise ValueError(f"{condition_id} confidence must be in [0, 1]")
        value = {
            "answer": str(item.get("answer", "")),
            "facts": item.get("facts") or {},
            "confidence": confidence,
        }
        units.append(
            EvidenceUnit(
                part_id=document.document_id,
                feature_id=feature_id,
                feature_type=property_name,
                view_regions=tuple(regions),
                geometric_properties={property_name: value},
                sources=tuple(dict.fromkeys(sources)),
                summary=str(item.get("answer", "")),
                family_id=document.document_id,
                cost=max(1.0, len(regions) * 0.5),
            )
        )
    return tuple(units), payload


def manifest_json(documents: Iterable[DrawingDocument]) -> str:
    return json.dumps(
        [document.to_dict() for document in documents],
        ensure_ascii=False,
        indent=2,
    )
