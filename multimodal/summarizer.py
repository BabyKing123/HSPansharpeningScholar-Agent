"""本模块作用：调用视觉语言模型，为论文图表页生成结构化摘要。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from llm_dashscope import DashScopeClient, parse_first_json_object


VISION_SUMMARY_PROMPT = """你是科研论文图表理解助手。请阅读论文截图，重点识别其中的模型图、实验图、结果图、消融表格、指标表格、算法流程图或可视化结果。请输出 JSON，字段包括：

visual_type
caption
main_content
technical_details
relevance_to_pansharpening
keywords

注意：
* 如果图中包含模型结构，请说明模块、输入输出和信息流。
* 如果图中包含实验表格，请尽量说明涉及的数据集、指标、对比方法和主要结论。
* 如果图中包含曲线图，请说明横纵轴、趋势和结论。
* 如果无法确定细节，不要编造，写“无法从图中确定”。
* 输出必须是 JSON object。
"""

ALLOWED_VISUAL_TYPES = {
    "figure",
    "table",
    "architecture",
    "chart",
    "algorithm",
    "result",
    "unknown",
}


def normalize_visual_type(value: Any) -> str:
    """把模型输出的图表类型归一到稳定枚举。"""

    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    if any(term in text for term in ["architecture", "framework", "network", "model", "结构", "模型"]):
        return "architecture"
    if any(term in text for term in ["table", "表", "ablation", "消融"]):
        return "table"
    if any(term in text for term in ["chart", "curve", "plot", "bar", "曲线", "柱状"]):
        return "chart"
    if any(term in text for term in ["algorithm", "flow", "pipeline", "流程", "算法"]):
        return "algorithm"
    if any(term in text for term in ["result", "visual", "comparison", "结果", "可视化", "对比"]):
        return "result"
    if text in ALLOWED_VISUAL_TYPES:
        return text
    return "figure"


def normalize_keywords(value: Any, fallback_text: str = "", max_items: int = 12) -> list[str]:
    """规范化关键词字段。"""

    raw_items: list[str] = []
    if isinstance(value, list):
        raw_items = [str(item) for item in value]
    elif isinstance(value, str):
        raw_items = re.split(r"[,，;；、\n]+", value)

    if not raw_items and fallback_text:
        raw_items = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}", fallback_text)

    keywords: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        cleaned = re.sub(r"\s+", " ", str(item)).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        keywords.append(cleaned)
        if len(keywords) >= max_items:
            break
    return keywords


def ensure_text(value: Any, default: str = "无法从图中确定") -> str:
    """把模型字段转换为非空字符串。"""

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text or default


def build_fallback_summary(item: dict[str, Any], reason: str) -> dict[str, Any]:
    """在视觉模型不可用或解析失败时构造保守摘要。"""

    caption = ensure_text(item.get("caption"), default="")
    raw_page_text = ensure_text(item.get("raw_page_text"), default="")
    fallback_text = caption or raw_page_text[:500] or "未生成视觉摘要。"
    enriched_item = dict(item)
    enriched_item.update(
        {
            "visual_type": normalize_visual_type(caption or raw_page_text),
            "summary": fallback_text,
            "technical_details": "无法从图中确定",
            "relevance_to_pansharpening": "无法从图中确定",
            "keywords": normalize_keywords([], fallback_text=fallback_text),
            "summary_error": reason,
        }
    )
    return enriched_item


def summarize_visual_item(
    client: DashScopeClient | None,
    item: dict[str, Any],
    *,
    model_name: str,
    temperature: float = 0.1,
    max_tokens: int = 1200,
) -> dict[str, Any]:
    """为单个 VisualItem 生成结构化摘要。

    输入：
        client: DashScopeClient；为空时返回占位摘要。
        item: extractor 产生的 VisualItem。
        model_name: 视觉语言模型名称。
    输出：
        带 visual_type、summary、technical_details 等字段的 VisualItem。
    异常：
        无。内部捕获模型调用与 JSON 解析失败并写入 summary_error。
    """

    image_path = str(item.get("image_path", "")).strip()
    if client is None:
        return build_fallback_summary(item, "未配置 DashScope 客户端，已跳过视觉摘要生成。")
    if not image_path:
        return build_fallback_summary(item, "图像路径为空，已跳过视觉摘要生成。")
    if not Path(image_path).expanduser().exists():
        return build_fallback_summary(item, f"图像文件不存在：{image_path}")

    prompt = (
        f"{VISION_SUMMARY_PROMPT}\n\n"
        f"论文文件名：{item.get('file_name', '')}\n"
        f"页码：{item.get('page_number', '')}\n"
        f"候选 caption：{item.get('caption', '')}\n"
        f"页面文本片段：{item.get('raw_page_text', '')[:1200]}\n"
    )

    try:
        raw_text = client.chat_multimodal(
            model=model_name,
            prompt=prompt,
            image_paths=[image_path],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        return build_fallback_summary(item, f"视觉模型调用失败：{exc}")

    json_data = parse_first_json_object(raw_text)
    if json_data is None:
        return build_fallback_summary(item, "视觉模型输出非 JSON，已使用页面文本兜底。")

    caption = ensure_text(json_data.get("caption"), default=str(item.get("caption", "") or ""))
    main_content = ensure_text(json_data.get("main_content"))
    technical_details = ensure_text(json_data.get("technical_details"))
    relevance = ensure_text(json_data.get("relevance_to_pansharpening"))
    fallback_keyword_text = " ".join([caption, main_content, technical_details, relevance])

    enriched_item = dict(item)
    enriched_item.update(
        {
            "caption": caption,
            "visual_type": normalize_visual_type(json_data.get("visual_type")),
            "summary": main_content,
            "technical_details": technical_details,
            "relevance_to_pansharpening": relevance,
            "keywords": normalize_keywords(json_data.get("keywords"), fallback_text=fallback_keyword_text),
        }
    )
    return enriched_item
