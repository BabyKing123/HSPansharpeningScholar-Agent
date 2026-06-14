"""本模块作用：从论文 PDF 中识别可能包含图表的页面，并将整页渲染为图片。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rag.parser import build_document_id


VISUAL_TEXT_PATTERN = re.compile(
    r"\b(?:figure|fig\.?|table)\b|图表|模型图|结构图|实验结果|可视化|消融|曲线|柱状图|"
    r"流程图|结果图|对比图|指标表|表\s*\d+|图\s*\d+",
    flags=re.IGNORECASE,
)

CAPTION_PATTERN = re.compile(
    r"^\s*(?:fig(?:ure)?\.?\s*\d+|table\s*\d+|图\s*\d+|表\s*\d+|模型图|结构图|实验结果|可视化)",
    flags=re.IGNORECASE,
)


def _load_fitz() -> Any:
    """延迟导入 PyMuPDF，避免未安装依赖时影响纯文本功能。"""

    try:
        import fitz
    except ImportError as exc:
        raise ImportError("缺少 PyMuPDF 依赖，请先执行 `pip install -r requirements.txt`。") from exc
    return fitz


def _safe_relative_path(path: Path) -> str:
    """尽量保存相对工作区路径，失败时保存绝对路径。"""

    resolved_path = path.expanduser().resolve()
    try:
        return str(resolved_path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved_path)


def normalize_caption_line(line: str, max_length: int = 500) -> str:
    """清洗 PDF 页面中的候选图表标题行。"""

    cleaned_line = re.sub(r"\s+", " ", line).strip()
    if len(cleaned_line) <= max_length:
        return cleaned_line
    return cleaned_line[:max_length].rstrip() + "..."


def extract_caption_from_page_text(page_text: str, max_lines: int = 6) -> str:
    """从页面文本中提取可能的 Figure/Table/图/表 标题。"""

    captions: list[str] = []
    for raw_line in page_text.splitlines():
        line = normalize_caption_line(raw_line)
        if not line:
            continue
        if CAPTION_PATTERN.search(line):
            captions.append(line)
        if len(captions) >= max_lines:
            break
    return " ".join(captions)


def page_has_image_block(page: Any) -> bool:
    """判断页面文本块中是否包含图片 block。"""

    try:
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") == 1:
                return True
    except Exception:
        return False
    return False


def page_has_image_object(page: Any) -> bool:
    """判断页面中是否存在明显图片对象。"""

    try:
        return bool(page.get_images(full=True))
    except Exception:
        return False


def page_has_vector_drawings(page: Any, min_drawings: int = 8) -> bool:
    """用绘图对象数量兜底识别矢量图表页。"""

    try:
        return len(page.get_drawings()) >= min_drawings
    except Exception:
        return False


def should_render_page(page: Any, page_text: str) -> bool:
    """判断页面是否可能包含图、表、模型结构图、实验曲线或可视化结果。"""

    if VISUAL_TEXT_PATTERN.search(page_text):
        return True
    if "图" in page_text or "表" in page_text:
        return True
    if page_has_image_block(page):
        return True
    if page_has_image_object(page):
        return True
    if page_has_vector_drawings(page):
        return True
    return False


def render_page_to_png(page: Any, output_path: Path, zoom: float) -> None:
    """将 PDF 整页渲染成 PNG。"""

    fitz = _load_fitz()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = fitz.Matrix(max(zoom, 0.5), max(zoom, 0.5))
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    pixmap.save(str(output_path))


def extract_visual_items_from_pdf(
    pdf_path: str | Path,
    image_dir: str | Path,
    *,
    zoom: float = 2.0,
    max_pages_per_paper: int = 30,
) -> list[dict[str, Any]]:
    """识别并渲染单篇 PDF 中的候选图表页。

    输入：
        pdf_path: 待处理 PDF 路径。
        image_dir: 图表页图片输出根目录。
        zoom: 页面渲染倍率。
        max_pages_per_paper: 每篇论文最多处理的候选图表页数。
    输出：
        VisualItem 字典列表。
    异常：
        单页失败会被记录在 item 的 error 字段中；PDF 打开失败会抛出 RuntimeError。
    """

    fitz = _load_fitz()
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF 文件不存在：{path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"不是 PDF 文件：{path}")

    document_id = build_document_id(path)
    document_image_dir = Path(image_dir).expanduser().resolve() / document_id
    items: list[dict[str, Any]] = []

    try:
        document = fitz.open(str(path))
    except Exception as exc:
        raise RuntimeError(f"无法打开 PDF 文件进行图表解析：{path.name}") from exc

    try:
        for page_index in range(document.page_count):
            if len(items) >= max(max_pages_per_paper, 0):
                break
            try:
                page = document.load_page(page_index)
                raw_page_text = page.get_text("text") or ""
                if not should_render_page(page, raw_page_text):
                    continue

                page_number = page_index + 1
                image_path = document_image_dir / f"page_{page_number:03d}.png"
                item: dict[str, Any] = {
                    "visual_id": f"{document_id}_page_{page_number:03d}",
                    "document_id": document_id,
                    "file_name": path.name,
                    "source_path": str(path),
                    "page_number": page_number,
                    "image_path": _safe_relative_path(image_path),
                    "caption": extract_caption_from_page_text(raw_page_text),
                    "raw_page_text": normalize_caption_line(raw_page_text, max_length=3000),
                }
                try:
                    render_page_to_png(page, image_path, zoom=zoom)
                except Exception as exc:
                    item["render_error"] = str(exc)
                items.append(item)
            except Exception as exc:
                items.append(
                    {
                        "visual_id": f"{document_id}_page_{page_index + 1:03d}",
                        "document_id": document_id,
                        "file_name": path.name,
                        "source_path": str(path),
                        "page_number": page_index + 1,
                        "image_path": "",
                        "caption": "",
                        "raw_page_text": "",
                        "extract_error": str(exc),
                    }
                )
    finally:
        document.close()

    return items
