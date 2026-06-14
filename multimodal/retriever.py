"""本模块作用：基于关键词从视觉摘要索引中检索相关图表页。"""

from __future__ import annotations

import re
from typing import Any


VISUAL_QUERY_TERMS = [
    "图",
    "图表",
    "表格",
    "Figure",
    "Fig",
    "Table",
    "曲线",
    "柱状图",
    "模型图",
    "结构图",
    "architecture",
    "ablation",
    "消融",
    "实验表",
    "可视化",
]

RETRIEVAL_FIELDS = [
    "file_name",
    "caption",
    "visual_type",
    "summary",
    "technical_details",
    "relevance_to_pansharpening",
    "keywords",
]

STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "which",
    "论文",
    "哪些",
    "什么",
    "如何",
    "说明",
    "这些",
    "中的",
    "以及",
}


def is_visual_query(question: str) -> bool:
    """判断用户问题是否明显在询问图表、表格、结构图或可视化内容。"""

    lowered_question = question.lower()
    return any(term.lower() in lowered_question for term in VISUAL_QUERY_TERMS)


def extract_query_terms(question: str) -> list[str]:
    """从问题中提取中英文检索词。"""

    lowered = question.lower()
    english_terms = re.findall(r"[a-z][a-z0-9_\-]{2,}", lowered)
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", question)
    metric_terms = re.findall(r"\b(?:PSNR|SSIM|SAM|ERGAS|RMSE|QNR|SCC|Q2n)\b", question, flags=re.IGNORECASE)
    terms: list[str] = []
    seen: set[str] = set()
    for term in metric_terms + english_terms + chinese_terms:
        cleaned = term.strip().lower()
        if not cleaned or cleaned in STOP_WORDS or cleaned in seen:
            continue
        seen.add(cleaned)
        terms.append(cleaned)
    return terms


def stringify_field(value: Any) -> str:
    """把图表索引字段转为可检索文本。"""

    if isinstance(value, list):
        return " ".join(str(item) for item in value if str(item).strip())
    if isinstance(value, dict):
        return " ".join(str(item) for item in value.values() if str(item).strip())
    return str(value or "")


def build_item_text(item: dict[str, Any]) -> str:
    """拼接单个视觉条目的检索字段。"""

    return " ".join(stringify_field(item.get(field, "")) for field in RETRIEVAL_FIELDS).lower()


def score_visual_item(question: str, item: dict[str, Any]) -> float:
    """计算问题与图表摘要条目的关键词相关性。"""

    item_text = build_item_text(item)
    if not item_text.strip():
        return 0.0

    score = 0.0
    terms = extract_query_terms(question)
    lowered_question = question.lower()

    for term in terms:
        hit_count = item_text.count(term)
        if hit_count <= 0:
            continue
        score += min(hit_count, 4) * (2.0 if len(term) >= 5 else 1.2)

    for visual_term in VISUAL_QUERY_TERMS:
        if visual_term.lower() in lowered_question:
            score += 1.5
            visual_type = str(item.get("visual_type", "")).lower()
            caption = str(item.get("caption", "")).lower()
            if visual_term.lower() in visual_type or visual_term.lower() in caption:
                score += 2.0

    file_name = str(item.get("file_name", "")).lower()
    if file_name and any(term in file_name for term in terms):
        score += 3.0

    if re.search(r"psnr|ssim|sam|ergas|rmse|qnr|指标|metric", lowered_question):
        if re.search(r"psnr|ssim|sam|ergas|rmse|qnr|metric|指标", item_text):
            score += 4.0
    if re.search(r"architecture|模型图|结构图|网络|框架", lowered_question):
        if str(item.get("visual_type", "")).lower() == "architecture":
            score += 5.0
    if re.search(r"ablation|消融", lowered_question):
        if re.search(r"ablation|消融", item_text):
            score += 4.0

    return score


def retrieve_visual_items(
    question: str,
    visual_items: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """根据用户问题检索相关视觉摘要条目。

    输入：
        question: 用户问题。
        visual_items: load_vision_index 返回的条目列表。
        top_k: 最多返回数量。
    输出：
        相关视觉条目列表。索引为空或异常输入时返回空列表。
    异常：
        无。
    """

    if not question.strip() or not visual_items or top_k <= 0:
        return []

    scored_items: list[tuple[float, int, dict[str, Any]]] = []
    for index, item in enumerate(visual_items):
        if not isinstance(item, dict):
            continue
        score = score_visual_item(question, item)
        if score <= 0:
            continue
        enriched_item = dict(item)
        enriched_item["retrieval_score"] = round(score, 3)
        scored_items.append((score, index, enriched_item))

    scored_items.sort(key=lambda value: (-value[0], value[1]))
    return [item for _, _, item in scored_items[:top_k]]


def truncate_text(text: str, max_length: int = 220) -> str:
    """裁剪过长图表摘要。"""

    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[:max_length].rstrip() + "..."


def format_visual_items_for_context(items: list[dict[str, Any]], max_items: int | None = None) -> str:
    """将视觉条目整理成可放入 prompt 或 Markdown 的证据文本。"""

    selected_items = items[:max_items] if max_items is not None else items
    lines: list[str] = []
    for index, item in enumerate(selected_items, start=1):
        lines.extend(
            [
                f"[图表{index}]",
                f"论文：{item.get('file_name', '未知论文')}",
                f"页码：{item.get('page_number', '未知页码')}",
                f"类型：{item.get('visual_type', 'unknown')}",
                f"caption：{item.get('caption', '')}",
                f"摘要：{truncate_text(str(item.get('summary', '')))}",
                f"技术细节：{truncate_text(str(item.get('technical_details', '')))}",
                f"相关性：{truncate_text(str(item.get('relevance_to_pansharpening', '')))}",
                f"图片路径：{item.get('image_path', '')}",
                "",
            ]
        )
    return "\n".join(lines).strip()
