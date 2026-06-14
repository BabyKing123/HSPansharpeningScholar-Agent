"""本模块作用：提供 HSPansharpeningScholar-Agent 的命令行主流程，完成本地建库、检索问答与单篇论文分析，并支持 DashScope 大模型增强。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from config import OUTPUT_DIR, get_app_config
from core.agent import (
    AgentAnswer,
    EmbeddingIndexResponse,
    GraphIndexResponse,
    HSPansharpeningScholarAgent,
    KnowledgeBaseState,
    PaperAnalysisResponse,
    PaperComparisonResponse,
    ReviewOutlineResponse,
    WorkflowResponse,
)
from core.workflow import (
    WorkflowRunResult,
    WorkflowState,
    build_default_workflow_plan,
    export_workflow_result,
    format_workflow_run_result,
    mark_step_status,
)
from llm_dashscope import DashScopeClient, parse_first_json_object
from multimodal import (
    extract_visual_items_from_pdf,
    format_visual_items_for_context,
    get_vision_status,
    is_visual_query,
    load_vision_index,
    retrieve_visual_items,
    save_vision_index,
    summarize_visual_item,
    vision_index_exists,
)
from tools.analyze_tool import StructuredPaperAnalysis, format_analysis_result
from tools.compare_tool import ComparisonPaperSummary, MultiPaperComparison, format_comparison_result
from tools.outline_tool import ReviewOutline, build_outline_sections, format_review_outline


DEFAULT_WORKFLOW_TOPIC = "基于指定论文生成高光谱全色锐化综述"


def ensure_directories(paths: list[Path]) -> None:
    """创建运行所需目录。"""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def initialize_llm_client(config: dict[str, Path | str | int | float | bool]) -> DashScopeClient | None:
    """根据配置初始化 DashScope 客户端。"""

    api_key = str(config.get("dashscope_api_key", "")).strip()
    base_url = str(config.get("dashscope_base_url", "")).strip()
    timeout_sec = int(config.get("dashscope_timeout_sec", 45))
    # 未配置 API Key 时保持“纯本地规则模式”，避免启动时报错中断。
    if not api_key:
        return None
    try:
        return DashScopeClient(api_key=api_key, base_url=base_url, timeout_sec=timeout_sec)
    except Exception:
        # 初始化失败时不抛出异常，回退到本地规则模式继续可用。
        return None


def show_startup_info(
    config: dict[str, Path | str | int | float | bool],
    knowledge_base: KnowledgeBaseState,
    llm_client: DashScopeClient | None,
    embedding_status: EmbeddingIndexResponse | None,
    graph_status: GraphIndexResponse | None,
) -> None:
    """输出启动信息。"""

    llm_enabled = llm_client is not None
    answer_model = str(config.get("dashscope_answer_model", "qwen-plus"))
    analysis_model = str(config.get("dashscope_analysis_model", "qwen-max"))

    print("=" * 60)
    print(f"启动项目：{config['project_name']}")
    print(f"项目目录：{config['base_dir']}")
    print(f"论文目录：{config['raw_papers_dir']}")
    print(f"发现 PDF 数量：{len(knowledge_base.pdf_files)}")
    print(f"解析成功论文数：{len(knowledge_base.documents)}")
    print(f"可检索片段数：{len(knowledge_base.chunk_records)}")
    print(f"解析失败论文数：{len(knowledge_base.parse_errors)}")
    print(f"大模型增强：{'已启用' if llm_enabled else '未启用'}")
    if llm_enabled:
        print(f"问答模型：{answer_model}")
        print(f"分析模型：{analysis_model}")
        print(f"向量模型：{config['dashscope_embedding_model']}")
    if embedding_status is not None:
        print(f"向量索引：{embedding_status.status_message}")
        if embedding_status.vector_count > 0:
            print(f"向量数量：{embedding_status.vector_count}")
    if graph_status is not None:
        print(f"图谱索引：{graph_status.status_message}")
        if graph_status.entity_count > 0:
            print(
                f"图谱统计：实体 {graph_status.entity_count} | 关系 {graph_status.relation_count} | "
                f"图谱社群 {graph_status.community_count} | 图谱社群摘要 {graph_status.community_summary_count}"
            )
    vision_status = get_vision_status(
        Path(config.get("vision_index_path", config["processed_data_dir"] / "vision_index.json")),
        Path(config.get("vision_image_dir", config["processed_data_dir"] / "vision")),
    )
    print(f"多模态图表理解：{'已启用' if config.get('vision_enabled', True) else '未启用'}")
    print(
        f"图表索引：{'已构建' if vision_status['index_exists'] else '未构建'} | "
        f"图表页 {vision_status['visual_item_count']} | 论文 {vision_status['paper_count']}"
    )
    print("当前模式：命令行最小检索、混合检索问答、单篇分析、多篇比较、综述提纲与多步工作流。")
    print("领域提醒：实体类型、关键词规则和 GraphRAG 抽取逻辑已面向全色锐化/高光谱全色锐化；旧索引请删除后执行 rebuild_index 与 rebuild_graph。")
    print("可用命令：")
    print("- 直接输入问题：执行检索问答（自动在 Hybrid RAG / GraphRAG 之间路由）")
    print("- ask :: 问题：执行论文检索问答")
    print("- build_index：构建第四周本地向量索引")
    print("- rebuild_index：强制重建本地向量索引")
    print("- build_graph：构建本地 GraphRAG 图谱索引")
    print("- rebuild_graph：强制重建本地图谱索引")
    print("- graph_status：查看图谱索引状态")
    print("- build_vision：构建多模态图表索引")
    print("- rebuild_vision：强制重建多模态图表索引")
    print("- vision_status：查看多模态图表索引状态")
    print("- figures：列出已识别的图表页")
    print("- vision :: 问题 / 看图 :: 问题：直接基于图表摘要提问")
    print("- papers：查看当前可分析论文列表")
    print("- analyze：分析第一篇论文（启用大模型时自动增强）")
    print("- analyze 1：按序号分析论文")
    print("- analyze 文件名关键词：按文件名或文档编号分析论文")
    print("- compare：比较前两篇论文")
    print("- compare 1,2：比较指定论文")
    print("- compare 1,2 :: 空间-光谱融合机制：按主题比较指定论文")
    print("- outline 高光谱全色锐化中的空间-光谱融合机制综述：生成最小综述提纲")
    print("- outline 1,2,3 :: 高光谱全色锐化中的注意力与门控机制综述：基于指定论文生成提纲")
    print("- workflow 高光谱全色锐化中的 zero-shot 与 diffusion 方法综述：执行多步工作流（启用大模型时自动增强比较与提纲）")
    print("- workflow 1,2,3：基于指定论文执行默认综述工作流")
    print("- workflow 1,2,3 :: 基于指定论文生成高光谱全色锐化综述：基于指定论文执行多步工作流")
    print("- help：再次查看命令说明")
    print("- exit：退出程序")
    print("=" * 60)


def show_parse_error_summary(parse_errors: list[dict[str, str]], max_items: int = 3) -> None:
    """输出 PDF 解析失败摘要信息。"""

    if not parse_errors:
        return

    print("以下 PDF 在建库时解析失败，已自动跳过：")
    for error_item in parse_errors[:max_items]:
        print(f"- 文件：{error_item['file_name']}")
        print(f"  原因：{error_item['error_message']}")


def format_page_numbers(page_numbers: list[int]) -> str:
    """将页码列表转换为可展示文本。"""

    if not page_numbers:
        return "未知页码"
    return "、".join(str(number) for number in page_numbers)


def build_answer_context(result: AgentAnswer) -> str:
    """将召回来源整理为模型可读上下文。"""

    context_lines: list[str] = []
    for index, source in enumerate(result.sources, start=1):
        # 将来源信息统一格式化，方便模型按来源编号引用证据。
        page_text = format_page_numbers(source.get("page_numbers", []))
        file_name = str(source.get("file_name", "未知论文"))
        snippet = str(source.get("snippet", ""))
        context_lines.append(f"[来源{index}] 论文：{file_name} | 页码：{page_text} | 片段：{snippet}")
    return "\n".join(context_lines)


def load_visual_items_safely(index_path: Path) -> list[dict[str, Any]]:
    """安全加载视觉索引，失败时返回空列表。"""

    try:
        return load_vision_index(index_path)
    except Exception:
        return []


def get_question_visual_items(
    question: str,
    *,
    vision_enabled: bool,
    vision_index_path: Path,
    top_k: int,
    require_visual_query: bool = True,
) -> list[dict[str, Any]]:
    """按问题检索图表摘要证据。"""

    if not vision_enabled:
        return []
    if require_visual_query and not is_visual_query(question):
        return []
    visual_items = load_visual_items_safely(vision_index_path)
    if not visual_items:
        return []
    return retrieve_visual_items(question, visual_items, top_k=top_k)


def build_visual_source_entries(visual_items: list[dict[str, Any]]) -> list[dict[str, object]]:
    """将视觉摘要条目转换为问答来源项。"""

    sources: list[dict[str, object]] = []
    for item in visual_items:
        summary = str(item.get("summary", "") or item.get("caption", "") or "")
        sources.append(
            {
                "source_type": "visual_summary",
                "file_name": str(item.get("file_name", "未知论文")),
                "source_path": str(item.get("source_path", "")),
                "page_numbers": [int(item.get("page_number", 0))] if str(item.get("page_number", "")).isdigit() else [],
                "chunk_id": str(item.get("visual_id", "")),
                "chunk_index": 0,
                "score": float(item.get("retrieval_score", 0.0) or 0.0),
                "snippet": f"图表摘要：{summary}",
                "matched_terms": [str(keyword) for keyword in item.get("keywords", []) if str(keyword).strip()]
                if isinstance(item.get("keywords", []), list)
                else [],
            }
        )
    return sources


def merge_visual_sources(result: AgentAnswer, visual_items: list[dict[str, Any]]) -> None:
    """把图表摘要来源附加到问答结果中，便于 CLI 展示来源页码。"""

    if not visual_items:
        return
    existing_ids = {str(source.get("chunk_id", "")) for source in result.sources}
    for source in build_visual_source_entries(visual_items):
        if str(source.get("chunk_id", "")) in existing_ids:
            continue
        result.sources.append(source)
        existing_ids.add(str(source.get("chunk_id", "")))


def append_visual_summary_to_answer(result: AgentAnswer, visual_items: list[dict[str, Any]]) -> AgentAnswer:
    """无 LLM 时把图表摘要作为保守补充追加到规则答案。"""

    if not visual_items:
        return result
    visual_lines = [
        "",
        "图表摘要补充（来自 vision_index，不等同于论文原文）：",
    ]
    for index, item in enumerate(visual_items, start=1):
        caption = str(item.get("caption", "")).strip()
        summary = str(item.get("summary", "")).strip()
        page_number = item.get("page_number", "未知页码")
        visual_lines.append(
            f"{index}. 《{item.get('file_name', '未知论文')}》第 {page_number} 页"
            f"（{item.get('visual_type', 'unknown')}）：{caption or summary[:120]}"
        )
    result.model_answer = result.model_answer.rstrip() + "\n" + "\n".join(visual_lines)
    merge_visual_sources(result, visual_items)
    return result


def format_visual_items_section(
    items: list[dict[str, Any]],
    *,
    title: str = "图表与视觉内容分析",
    max_items: int = 8,
) -> str:
    """将视觉条目整理为可附加到分析、比较、提纲和报告中的文本。"""

    if not items:
        return ""
    lines = [f"{title}："]
    for item in items[:max_items]:
        caption = str(item.get("caption", "")).strip()
        summary = str(item.get("summary", "")).strip()
        details = str(item.get("technical_details", "")).strip()
        image_path = str(item.get("image_path", "")).strip()
        lines.extend(
            [
                f"- 论文：{item.get('file_name', '未知论文')}，第 {item.get('page_number', '未知页码')} 页，{caption}",
                f"  类型：{item.get('visual_type', 'unknown')}",
                f"  摘要：{summary or '未生成摘要'}",
                f"  技术细节：{details or '无法从图中确定'}",
                f"  图片路径：{image_path}",
            ]
        )
    return "\n".join(lines)


def select_visual_items_for_documents(
    documents: list[Any],
    *,
    vision_enabled: bool,
    vision_index_path: Path,
    max_items_per_document: int = 4,
) -> list[dict[str, Any]]:
    """按论文列表筛选 vision_index 中对应图表页。"""

    if not vision_enabled or not documents:
        return []
    visual_items = load_visual_items_safely(vision_index_path)
    if not visual_items:
        return []

    selected_items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for document in documents:
        document_id = str(getattr(document, "document_id", ""))
        file_name = str(getattr(document, "file_name", ""))
        matched = [
            item
            for item in visual_items
            if str(item.get("document_id", "")) == document_id or str(item.get("file_name", "")) == file_name
        ]
        for item in matched[:max_items_per_document]:
            visual_id = str(item.get("visual_id", ""))
            if visual_id in seen_ids:
                continue
            selected_items.append(item)
            seen_ids.add(visual_id)
    return selected_items


def build_vision_index_for_papers(
    *,
    agent: HSPansharpeningScholarAgent,
    client: DashScopeClient | None,
    model_name: str,
    image_dir: Path,
    index_path: Path,
    zoom: float,
    max_pages_per_paper: int,
    force_rebuild: bool,
) -> dict[str, Any]:
    """扫描 PDF、渲染图表页、生成视觉摘要并保存 vision_index。"""

    if not force_rebuild and vision_index_exists(index_path):
        status = get_vision_status(index_path, image_dir)
        status["status_message"] = "已存在多模态图表索引；如需重建请执行 rebuild_vision。"
        status["warnings"] = []
        return status

    knowledge_base = agent.ensure_knowledge_base_ready()
    all_items: list[dict[str, Any]] = []
    warnings: list[str] = []
    image_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in knowledge_base.pdf_files:
        try:
            extracted_items = extract_visual_items_from_pdf(
                pdf_path,
                image_dir,
                zoom=zoom,
                max_pages_per_paper=max_pages_per_paper,
            )
        except Exception as exc:
            warnings.append(f"{Path(pdf_path).name} 图表页解析失败：{exc}")
            continue

        for item in extracted_items:
            try:
                all_items.append(
                    summarize_visual_item(
                        client,
                        item,
                        model_name=model_name,
                    )
                )
            except Exception as exc:
                fallback_item = dict(item)
                fallback_item["summary_error"] = f"视觉摘要生成失败：{exc}"
                all_items.append(fallback_item)

    save_vision_index(all_items, index_path)
    status = get_vision_status(index_path, image_dir)
    status["status_message"] = f"已完成多模态图表索引构建，图表页数量：{len(all_items)}。"
    if client is None:
        warnings.append("未配置可用 DashScope 客户端，已仅保存图表页与页面文本兜底摘要。")
    status["warnings"] = warnings
    return status


def display_vision_status(status: dict[str, Any], *, vision_enabled: bool, model_name: str) -> None:
    """展示视觉索引状态。"""

    print("\n多模态图表理解状态：")
    print(f"是否启用：{'是' if vision_enabled else '否'}")
    print(f"视觉模型：{model_name}")
    print(f"索引路径：{status.get('index_path', '')}")
    print(f"图片目录：{status.get('image_dir', '')}")
    print(f"索引是否存在：{'是' if status.get('index_exists') else '否'}")
    print(f"已索引论文数量：{status.get('paper_count', 0)}")
    print(f"图表页数量：{status.get('visual_item_count', 0)}")
    print(f"图片文件数量：{status.get('image_count', 0)}")
    if status.get("status_message"):
        print(f"状态：{status['status_message']}")
    warnings = status.get("warnings", [])
    if warnings:
        print("提示：")
        for warning in warnings[:8]:
            print(f"- {warning}")


def display_figures(index_path: Path) -> None:
    """列出 vision_index 中的图表页。"""

    visual_items = load_visual_items_safely(index_path)
    print("\n已识别图表页：")
    if not visual_items:
        print("当前没有可用图表索引，请先执行 rebuild_vision。")
        return
    for index, item in enumerate(visual_items, start=1):
        caption = str(item.get("caption", "")).strip()
        summary = str(item.get("summary", "")).strip()
        preview = (caption or summary or "无摘要").replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:80].rstrip() + "..."
        print(
            f"{index}. {item.get('file_name', '未知论文')} | "
            f"页码：{item.get('page_number', '未知')} | "
            f"类型：{item.get('visual_type', 'unknown')} | {preview}"
        )


def parse_vision_question(user_input: str) -> str:
    """解析 vision :: question 或 看图 :: question 命令。"""

    if "::" in user_input:
        return user_input.split("::", maxsplit=1)[1].strip()
    parts = user_input.strip().split(maxsplit=1)
    if len(parts) >= 2:
        return parts[1].strip()
    return ""


def parse_ask_question(user_input: str) -> str:
    """解析 ask :: question 或 ask question 命令。"""

    return parse_vision_question(user_input)


def answer_visual_question(
    *,
    question: str,
    client: DashScopeClient | None,
    model_name: str,
    vision_index_path: Path,
    top_k: int,
) -> tuple[str, list[dict[str, Any]], str | None]:
    """直接基于图表摘要回答用户问题。"""

    visual_items = load_visual_items_safely(vision_index_path)
    if not visual_items:
        return "当前没有可用图表索引，请先执行 rebuild_vision。", [], None

    retrieved_items = retrieve_visual_items(question, visual_items, top_k=top_k)
    if not retrieved_items:
        return "当前 vision_index 中没有检索到相关图表摘要。", [], None

    visual_context = format_visual_items_for_context(retrieved_items)
    if client is None:
        return (
            "当前未启用大模型，先返回检索到的图表摘要：\n"
            + visual_context,
            retrieved_items,
            None,
        )

    prompt = (
        "你是科研论文图表问答助手。请仅基于以下图表摘要回答，不要编造。"
        "图表摘要来自视觉模型，不等同于论文原文；回答中必须注明论文名、页码和 caption（如果有）。\n\n"
        f"用户问题：{question}\n\n"
        f"图表摘要证据：\n{visual_context}\n\n"
        "请用中文回答，先给结论，再列出依据来源。"
    )
    try:
        answer = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200,
        )
    except Exception as exc:
        return "视觉问答大模型调用失败，已返回图表摘要：\n" + visual_context, retrieved_items, str(exc)

    return answer, retrieved_items, None


def enhance_answer_with_llm(
    result: AgentAnswer,
    client: DashScopeClient | None,
    model_name: str,
    visual_items: list[dict[str, Any]] | None = None,
) -> tuple[AgentAnswer, str | None]:
    """在已有检索结果基础上，使用大模型增强回答文本。"""

    visual_items = visual_items or []
    # 没有模型或没有来源证据时，不进行增强，保持最小闭环稳定。
    if client is None:
        return append_visual_summary_to_answer(result, visual_items), None
    if not result.sources and not visual_items:
        return result, None

    system_prompt = (
        "你是面向全色锐化、高光谱全色锐化和遥感图像融合方向的论文科研助教。"
        "你只能基于给定来源回答，不要编造来源中不存在的结论。"
        "回答必须包含简洁结论，并尽量保留来源编号引用。"
        "请重点关注任务类型、输入模态、空间-光谱建模、损失函数、退化模型、数据集、指标、实验结论和对用户当前模型的启示。"
        "图表证据是视觉模型生成的图表摘要，不等同于论文原文；使用时必须注明论文名和页码。"
    )
    text_context = build_answer_context(result) or "无可用文本证据。"
    visual_context = format_visual_items_for_context(visual_items) if visual_items else "未检索到相关图表摘要。"
    user_prompt = (
        f"用户问题：{result.question}\n\n"
        f"【文本证据】\n{text_context}\n\n"
        f"【图表证据（图表摘要，不等同于论文原文）】\n{visual_context}\n\n"
        "请输出中文回答，结构为：\n"
        "1) 直接回答\n"
        "2) 关键依据（用 [来源1] 这样的编号表示）\n"
        "3) 图表证据补充（如果使用图表摘要，请注明论文名和页码）\n"
        "4) 不确定性提示（如果有）"
    )

    try:
        llm_answer = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=900,
        )
    except Exception as exc:
        return append_visual_summary_to_answer(result, visual_items), f"问答大模型调用失败，已回退最小规则回答：{exc}"

    result.model_answer = llm_answer
    merge_visual_sources(result, visual_items)
    return result, None


def build_analysis_input_for_llm(full_text: str, max_chars: int = 18000) -> str:
    """构建用于结构化分析的大模型输入文本。"""

    text = full_text.strip()
    if not text:
        return text

    lowered = text.lower()
    reference_markers = ["\nreferences", "\nreference", "\nbibliography", "\nacknowledgments"]
    cut_positions: list[int] = []
    for marker in reference_markers:
        index = lowered.find(marker)
        if index != -1:
            cut_positions.append(index)
    if cut_positions:
        text = text[: min(cut_positions)].strip()

    if len(text) <= max_chars:
        return text

    # 文本过长时保留前后文，兼顾背景信息与结论信息。
    head = text[: int(max_chars * 0.65)]
    tail = text[-int(max_chars * 0.35) :]
    return f"{head}\n\n[...中间内容已省略...]\n\n{tail}"


MISSING_FIELD_TEXT = "原文未稳定提取出该字段，请查看依据片段"
VALID_UNCLEAR_FIELD_TEXT = "原文未明确说明"


def normalize_extracted_text(text: str) -> str:
    """压缩模型或规则抽取结果中的多余空白。"""

    return re.sub(r"\s+", " ", str(text or "")).strip()


def is_bad_extracted_field(text: str) -> bool:
    """判断结构化字段是否为空、过短或明显像被截断的残句。"""

    cleaned_text = normalize_extracted_text(text)
    if not cleaned_text:
        return True
    if cleaned_text in {VALID_UNCLEAR_FIELD_TEXT, MISSING_FIELD_TEXT}:
        return False
    if any(marker in cleaned_text for marker in ["文本中未明确识别", "建议查看", "建议补充", "人工补充"]):
        return True
    if len(cleaned_text) < 8:
        return True
    if cleaned_text.endswith(("...", "…", ",", "，", ";", "；", ":", "：", "-", "—")):
        return True

    lowered_text = cleaned_text.lower().strip()
    stripped_tail = lowered_text.rstrip(".。;；,，:：")
    bad_tail_terms = (
        "to address",
        "fo",
        "and",
        "or",
        "of",
        "for",
        "with",
        "without",
        "to",
        "from",
        "by",
        "in",
        "on",
        "as",
        "the",
        "a",
        "an",
        "such as",
        "due to",
        "resulting in",
    )
    if any(stripped_tail.endswith(f" {term}") or stripped_tail == term for term in bad_tail_terms):
        return True
    if cleaned_text.count("(") != cleaned_text.count(")") or cleaned_text.count("[") != cleaned_text.count("]"):
        return True

    has_chinese = re.search(r"[\u4e00-\u9fff]", cleaned_text) is not None
    english_word_count = len(re.findall(r"[A-Za-z]+", cleaned_text))
    if not has_chinese and english_word_count >= 4:
        return True
    if re.match(r"^[a-z]", cleaned_text) and english_word_count >= 4:
        return True
    return False


def choose_analysis_field(
    json_data: dict[str, Any],
    fallback_analysis: StructuredPaperAnalysis,
    key: str,
) -> tuple[str, str]:
    """在 LLM 字段、规则字段和稳定兜底文本之间选择最终展示值。"""

    llm_value = normalize_extracted_text(str(json_data.get(key, "")))
    if not is_bad_extracted_field(llm_value):
        return llm_value, "llm"

    fallback_value = normalize_extracted_text(str(getattr(fallback_analysis, key, "")))
    if not is_bad_extracted_field(fallback_value):
        return fallback_value, "fallback"

    return MISSING_FIELD_TEXT, "fallback"


def pick_text_field(data: dict[str, Any], key: str, fallback: str) -> str:
    """从 JSON 结果中读取文本字段，不合法时回退。"""

    value = data.get(key, "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def normalize_evidence_map(data: dict[str, Any]) -> dict[str, list[str]]:
    """将模型输出的 evidence_map 规范为固定结构。"""

    field_keys = [
        "research_question",
        "research_object",
        "methods",
        "data_source",
        "key_findings",
        "limitations",
        "implications",
    ]
    evidence_map: dict[str, list[str]] = {}
    raw_map = data.get("evidence_map", {})
    if not isinstance(raw_map, dict):
        raw_map = {}

    for field_key in field_keys:
        raw_value = raw_map.get(field_key, [])
        # 兼容模型可能返回的字符串或列表两种结构，统一为字符串列表。
        if isinstance(raw_value, list):
            normalized_list = [str(item).strip() for item in raw_value if str(item).strip()]
        elif isinstance(raw_value, str):
            normalized_list = [raw_value.strip()] if raw_value.strip() else []
        else:
            normalized_list = []
        evidence_map[field_key] = normalized_list
    return evidence_map


def enhance_analysis_with_llm(
    agent: HSPansharpeningScholarAgent,
    target: str | None,
    client: DashScopeClient | None,
    model_name: str,
) -> tuple[PaperAnalysisResponse, str | None]:
    """执行单篇论文分析，并在可用时使用大模型增强结构化结果。"""

    base_result = agent.analyze_paper(target)
    if client is None or base_result.analysis is None:
        return base_result, None

    document = agent.find_document(target)
    if document is None:
        return base_result, None

    llm_input = build_analysis_input_for_llm(document.full_text)
    if not llm_input:
        return base_result, None

    system_prompt = (
        "你是全色锐化领域的论文分析助手。"
        "请仅根据给定论文内容提取结构化结果，不要编造。"
        "所有字段必须用中文完整句概括，不允许直接复制残缺原文。"
        "输出必须是 JSON 对象。"
    )
    user_prompt = (
        "请对以下论文内容进行结构化提取，并严格输出 JSON。"
        "字段必须包含：\n"
        "research_question, research_object, methods, data_source, key_findings, limitations, implications, evidence_map\n"
        "其中 evidence_map 是对象，键为上述七个字段名，值为字符串数组（每项是依据片段）。\n\n"
        "输出规则：\n"
        "1. 所有字段必须使用中文概括，且必须是完整句子。\n"
        "2. 不允许直接复制英文残句、半句话或被截断的原文。\n"
        "3. 如果原文没有明确说明，字段值写“原文未明确说明”。\n"
        "4. key_findings 必须总结实验结论、性能优势或理论结论，不要只写泛泛的 results。\n"
        "5. implications 必须结合高光谱全色锐化、空间-光谱融合、PAN/LRHS 条件建模、扩散模型或用户研究方向给出启示。\n\n"
        f"论文文件名：{document.file_name}\n"
        f"文档编号：{document.document_id}\n\n"
        f"论文内容：\n{llm_input}"
    )

    try:
        raw_text = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=3000,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        return base_result, f"学术分析大模型调用失败，已回退规则分析：{exc}"

    json_data = parse_first_json_object(raw_text)
    if json_data is None:
        return base_result, "学术分析大模型输出非 JSON，已回退规则分析。"

    fallback_analysis = base_result.analysis
    field_keys = [
        "research_question",
        "research_object",
        "methods",
        "data_source",
        "key_findings",
        "limitations",
        "implications",
    ]
    selected_values: dict[str, str] = {}
    selected_sources: dict[str, str] = {}
    for field_key in field_keys:
        selected_values[field_key], selected_sources[field_key] = choose_analysis_field(
            json_data,
            fallback_analysis,
            field_key,
        )

    llm_evidence_map = normalize_evidence_map(json_data)
    merged_evidence_map: dict[str, list[str]] = {}
    for field_key in field_keys:
        fallback_evidence = fallback_analysis.evidence_map.get(field_key, [])
        if selected_sources[field_key] == "llm" and llm_evidence_map.get(field_key):
            merged_evidence_map[field_key] = llm_evidence_map[field_key]
        else:
            merged_evidence_map[field_key] = fallback_evidence

    # 逐字段合并：LLM 字段通过质量校验才展示，否则回退到规则结果或稳定提示。
    merged_analysis = StructuredPaperAnalysis(
        file_name=fallback_analysis.file_name,
        document_id=fallback_analysis.document_id,
        research_question=selected_values["research_question"],
        research_object=selected_values["research_object"],
        methods=selected_values["methods"],
        data_source=selected_values["data_source"],
        key_findings=selected_values["key_findings"],
        limitations=selected_values["limitations"],
        implications=selected_values["implications"],
        evidence_map=merged_evidence_map,
    )

    enhanced_result = PaperAnalysisResponse(
        target=base_result.target,
        status_message=f"{base_result.status_message}（已使用大模型增强：{model_name}）",
        analysis=merged_analysis,
        formatted_output=format_analysis_result(merged_analysis),
    )
    return enhanced_result, None


def augment_analysis_with_visuals(
    *,
    agent: HSPansharpeningScholarAgent,
    result: PaperAnalysisResponse,
    target: str | None,
    vision_enabled: bool,
    vision_index_path: Path,
) -> PaperAnalysisResponse:
    """在单篇论文分析结果后追加图表摘要章节。"""

    if result.analysis is None:
        return result
    try:
        document = agent.find_document(target)
    except Exception:
        document = None
    if document is None:
        return result

    visual_items = select_visual_items_for_documents(
        [document],
        vision_enabled=vision_enabled,
        vision_index_path=vision_index_path,
        max_items_per_document=6,
    )
    section_text = format_visual_items_section(visual_items)
    if not section_text:
        return result
    result.formatted_output = result.formatted_output.rstrip() + "\n\n" + section_text
    result.status_message += "（已加入图表摘要）"
    return result


COMPARE_SECTION_MARKERS = {
    "abstract_intro": ["abstract", "introduction", "引言", "摘要"],
    "methods": ["method", "methodology", "proposed", "approach", "方法", "模型", "网络"],
    "experiments": ["experiment", "experimental", "result", "evaluation", "ablation", "实验", "结果", "评价"],
    "conclusion": ["conclusion", "discussion", "future work", "结论", "讨论"],
    "references": ["references", "reference", "bibliography", "参考文献"],
}


def find_compare_marker(text: str, markers: list[str], start: int = 0) -> int:
    """查找比较输入片段的章节标记位置。"""

    lowered_text = text.lower()
    positions = [lowered_text.find(marker.lower(), start) for marker in markers]
    positions = [position for position in positions if position >= 0]
    return min(positions) if positions else -1


def extract_compare_section(
    text: str,
    start_markers: list[str],
    end_markers: list[str],
    *,
    fallback_start: int,
    max_chars: int = 2200,
) -> str:
    """从论文正文中提取比较所需的章节窗口。"""

    start_position = find_compare_marker(text, start_markers)
    if start_position < 0:
        return text[fallback_start : fallback_start + max_chars].strip()

    end_position = len(text)
    for marker in end_markers:
        candidate_position = find_compare_marker(text, [marker], start=start_position + 1)
        if candidate_position > start_position:
            end_position = min(end_position, candidate_position)

    section_text = text[start_position:end_position].strip()
    if len(section_text) < 300:
        return text[fallback_start : fallback_start + max_chars].strip()
    return section_text[:max_chars].strip()


def build_compare_paper_context(document: Any) -> dict[str, str]:
    """为 LLM 比较构造单篇论文的分区输入。"""

    text = build_analysis_input_for_llm(str(document.full_text), max_chars=16000)
    text_length = len(text)
    method_fallback = int(text_length * 0.25)
    experiment_fallback = int(text_length * 0.55)
    conclusion_fallback = max(0, text_length - 2600)

    abstract_intro = extract_compare_section(
        text,
        COMPARE_SECTION_MARKERS["abstract_intro"],
        COMPARE_SECTION_MARKERS["methods"] + COMPARE_SECTION_MARKERS["experiments"],
        fallback_start=0,
    )
    methods = extract_compare_section(
        text,
        COMPARE_SECTION_MARKERS["methods"],
        COMPARE_SECTION_MARKERS["experiments"] + COMPARE_SECTION_MARKERS["conclusion"],
        fallback_start=method_fallback,
    )
    experiments = extract_compare_section(
        text,
        COMPARE_SECTION_MARKERS["experiments"],
        COMPARE_SECTION_MARKERS["conclusion"] + COMPARE_SECTION_MARKERS["references"],
        fallback_start=experiment_fallback,
    )
    conclusion = extract_compare_section(
        text,
        COMPARE_SECTION_MARKERS["conclusion"],
        COMPARE_SECTION_MARKERS["references"],
        fallback_start=conclusion_fallback,
    )

    return {
        "file_name": str(document.file_name),
        "document_id": str(document.document_id),
        "abstract_intro": abstract_intro,
        "methods": methods,
        "experiments": experiments,
        "conclusion": conclusion,
    }


def build_comparison_prompt_context(paper_contexts: list[dict[str, str]]) -> str:
    """将多篇论文的分区片段整理成模型输入。"""

    lines: list[str] = []
    for index, paper in enumerate(paper_contexts, start=1):
        lines.extend(
            [
                f"【论文{index}】",
                f"file_name: {paper['file_name']}",
                f"document_id: {paper['document_id']}",
                "摘要/引言片段：",
                paper["abstract_intro"],
                "方法片段：",
                paper["methods"],
                "实验片段：",
                paper["experiments"],
                "结论片段：",
                paper["conclusion"],
                "",
            ]
        )
    return "\n".join(lines)


def normalize_json_text_list(value: Any) -> list[str]:
    """将模型 JSON 中的字符串、数组或对象数组统一为文本列表。"""

    if isinstance(value, str):
        cleaned_text = normalize_extracted_text(value)
        return [cleaned_text] if cleaned_text else []
    if not isinstance(value, list):
        return []

    items: list[str] = []
    for raw_item in value:
        if isinstance(raw_item, dict):
            preferred_keys = ["summary", "insight", "comparison", "text", "value", "finding"]
            text_parts = [str(raw_item[key]).strip() for key in preferred_keys if str(raw_item.get(key, "")).strip()]
            if not text_parts:
                text_parts = [str(item).strip() for item in raw_item.values() if str(item).strip()]
            cleaned_text = normalize_extracted_text("；".join(text_parts))
        else:
            cleaned_text = normalize_extracted_text(str(raw_item))
        if cleaned_text and not is_bad_extracted_field(cleaned_text):
            items.append(cleaned_text)
    return items


def find_summary_item_for_document(
    summary_items: list[Any],
    document: Any,
    index: int,
) -> dict[str, Any]:
    """按 document_id、文件名或序号匹配 LLM 返回的单篇摘要。"""

    for item in summary_items:
        if not isinstance(item, dict):
            continue
        if str(item.get("document_id", "")).strip() == str(document.document_id):
            return item
        if str(item.get("file_name", "")).strip() == str(document.file_name):
            return item
    if index < len(summary_items) and isinstance(summary_items[index], dict):
        return summary_items[index]
    return {}


def pick_compare_summary_field(item: dict[str, Any], key: str) -> str:
    """读取 LLM 单篇比较摘要字段，并过滤残句。"""

    value = normalize_extracted_text(str(item.get(key, "")))
    if not is_bad_extracted_field(value):
        return value
    return VALID_UNCLEAR_FIELD_TEXT


def build_llm_comparison_result(
    json_data: dict[str, Any],
    documents: list[Any],
    topic_hint: str,
) -> MultiPaperComparison:
    """将 LLM JSON 比较结果映射为项目内的比较结果对象。"""

    summary_items = json_data.get("paper_summaries", [])
    if not isinstance(summary_items, list):
        summary_items = []

    paper_summaries: list[ComparisonPaperSummary] = []
    for index, document in enumerate(documents):
        summary_item = find_summary_item_for_document(summary_items, document, index)
        paper_summaries.append(
            ComparisonPaperSummary(
                file_name=str(document.file_name),
                document_id=str(document.document_id),
                research_question=pick_compare_summary_field(summary_item, "research_question"),
                research_object=pick_compare_summary_field(summary_item, "research_object"),
                methods=pick_compare_summary_field(summary_item, "methods"),
                data_source=pick_compare_summary_field(summary_item, "data_source"),
                key_findings=pick_compare_summary_field(summary_item, "key_findings"),
                limitations=pick_compare_summary_field(summary_item, "limitations"),
                implications=pick_compare_summary_field(summary_item, "implications"),
            )
        )

    actual_topic = normalize_extracted_text(str(json_data.get("topic_hint", ""))) or topic_hint
    dataset_items = normalize_json_text_list(json_data.get("dataset_comparison"))
    metric_items = normalize_json_text_list(json_data.get("metric_comparison"))
    insight_items = normalize_json_text_list(json_data.get("integrated_insights"))
    relevance_items = normalize_json_text_list(json_data.get("relevance_to_user_research"))
    task_type_items = normalize_json_text_list(json_data.get("task_type_comparison"))
    common_theme_items = normalize_json_text_list(json_data.get("common_themes"))
    if not common_theme_items:
        if actual_topic and actual_topic != "未指定比较主题":
            common_theme_items = [f"本次比较聚焦“{actual_topic}”。"]
        elif task_type_items:
            common_theme_items = task_type_items[:1]

    return MultiPaperComparison(
        topic_hint=actual_topic or "未指定比较主题",
        paper_summaries=paper_summaries,
        common_themes=common_theme_items,
        task_type_comparison=task_type_items,
        modality_comparison=normalize_json_text_list(json_data.get("modality_comparison")),
        method_comparison=normalize_json_text_list(json_data.get("method_comparison")),
        spatial_modeling_comparison=normalize_json_text_list(json_data.get("spatial_modeling_comparison")),
        spectral_modeling_comparison=normalize_json_text_list(json_data.get("spectral_modeling_comparison")),
        prior_or_degradation_modeling=normalize_json_text_list(json_data.get("prior_or_degradation_modeling")),
        dataset_comparison=dataset_items,
        metric_comparison=metric_items,
        strengths_and_limitations=normalize_json_text_list(json_data.get("strengths_and_limitations")),
        relevance_to_user_research=relevance_items,
        data_comparison=dataset_items + metric_items,
        finding_comparison=normalize_json_text_list(json_data.get("strengths_and_limitations")),
        integrated_implications=insight_items or relevance_items,
    )


def comparison_has_llm_content(comparison: MultiPaperComparison) -> bool:
    """判断 LLM 比较结果是否包含足够可展示的比较维度。"""

    def has_meaningful_item(items: list[str]) -> bool:
        return any(item and item.strip() and VALID_UNCLEAR_FIELD_TEXT not in item for item in items)

    comparison_lists = [
        comparison.task_type_comparison,
        comparison.modality_comparison,
        comparison.method_comparison,
        comparison.spatial_modeling_comparison,
        comparison.spectral_modeling_comparison,
        comparison.prior_or_degradation_modeling,
        comparison.dataset_comparison,
        comparison.metric_comparison,
        comparison.strengths_and_limitations,
        comparison.relevance_to_user_research,
        comparison.integrated_implications,
    ]
    return any(has_meaningful_item(items) for items in comparison_lists)


def enhance_comparison_with_llm(
    agent: HSPansharpeningScholarAgent,
    targets: list[str] | None,
    topic_hint: str,
    client: DashScopeClient | None,
    model_name: str,
    default_count: int = 2,
) -> tuple[PaperComparisonResponse, str | None]:
    """执行多篇论文比较，并在可用时使用大模型生成领域化比较结果。"""

    if client is None:
        return agent.compare_papers(targets, topic_hint=topic_hint), None

    knowledge_base = agent.ensure_knowledge_base_ready()
    if len(knowledge_base.documents) < 2:
        return agent.compare_papers(targets, topic_hint=topic_hint), None

    if targets:
        documents = agent.find_documents(targets, default_count=default_count)
    else:
        documents = agent.find_relevant_documents_for_topic(topic_hint, default_count=default_count)
    if len(documents) < 2:
        return agent.compare_papers(targets, topic_hint=topic_hint), None

    paper_contexts = build_comparison_prompt_context(
        [build_compare_paper_context(document) for document in documents]
    )
    cleaned_topic = topic_hint.strip() or "未指定比较主题"

    system_prompt = (
        "你是遥感图像融合与高光谱全色锐化方向的论文比较助手。"
        "请仅依据给定论文片段进行比较，不要编造。"
        "所有输出必须是中文完整句，避免 method、results、model 这类无意义泛词。"
        "输出必须是 JSON 对象。"
    )
    user_prompt = (
        f"比较主题提示：{cleaned_topic}\n\n"
        "请比较以下论文，并严格输出 JSON。顶层字段必须包含：\n"
        "topic_hint, paper_summaries, task_type_comparison, modality_comparison, method_comparison, "
        "spatial_modeling_comparison, spectral_modeling_comparison, prior_or_degradation_modeling, "
        "dataset_comparison, metric_comparison, strengths_and_limitations, relevance_to_user_research, integrated_insights\n\n"
        "paper_summaries 必须是数组，每篇论文对象包含：file_name, document_id, research_question, "
        "research_object, methods, data_source, key_findings, limitations, implications。\n"
        "比较维度字段除 topic_hint 和 paper_summaries 外都用字符串数组，每条必须是完整中文概括。\n"
        "重点比较任务类型、输入模态、空间信息建模、光谱信息建模、注意力/门控/先验/退化模型、数据集和指标。"
        "relevance_to_user_research 必须面向用户当前的高光谱全色锐化、PAN/LRHS 条件建模、扩散模型或空间-光谱融合研究给出启示。"
        "如果某项原文未明确说明，写“原文未明确说明”，不要用标题或正文残片充当结论。\n\n"
        f"论文片段：\n{paper_contexts}"
    )

    try:
        raw_text = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=3500,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        fallback_result = agent.compare_papers(targets, topic_hint=topic_hint)
        return fallback_result, f"多篇比较大模型调用失败，已回退规则比较：{exc}"

    json_data = parse_first_json_object(raw_text)
    if json_data is None:
        fallback_result = agent.compare_papers(targets, topic_hint=topic_hint)
        return fallback_result, "多篇比较大模型输出非 JSON，已回退规则比较。"

    comparison = build_llm_comparison_result(json_data, documents, cleaned_topic)
    if not comparison_has_llm_content(comparison):
        fallback_result = agent.compare_papers(targets, topic_hint=topic_hint)
        return fallback_result, "多篇比较大模型输出缺少有效比较维度，已回退规则比较。"

    enhanced_result = PaperComparisonResponse(
        targets=[document.file_name for document in documents],
        status_message=f"已使用大模型增强完成 {len(documents)} 篇论文的比较（{model_name}）。",
        comparison=comparison,
        formatted_output=format_comparison_result(comparison),
    )
    return enhanced_result, None


def augment_comparison_with_visuals(
    *,
    agent: HSPansharpeningScholarAgent,
    result: PaperComparisonResponse,
    vision_enabled: bool,
    vision_index_path: Path,
) -> PaperComparisonResponse:
    """在多篇比较结果后追加图表证据补充。"""

    if result.comparison is None:
        return result
    documents: list[Any] = []
    for target in result.targets:
        try:
            document = agent.find_document(target)
        except Exception:
            document = None
        if document is not None:
            documents.append(document)

    visual_items = select_visual_items_for_documents(
        documents,
        vision_enabled=vision_enabled,
        vision_index_path=vision_index_path,
        max_items_per_document=4,
    )
    section_text = format_visual_items_section(visual_items, title="图表证据补充", max_items=10)
    if not section_text:
        return result
    result.formatted_output = result.formatted_output.rstrip() + "\n\n" + section_text
    result.status_message += "（已加入图表摘要）"
    return result


def build_outline_from_comparison(topic: str, comparison: MultiPaperComparison) -> ReviewOutlineResponse:
    """基于比较结果生成综述提纲响应。"""

    outline = ReviewOutline(
        topic=topic,
        source_papers=[item.file_name for item in comparison.paper_summaries],
        sections=build_outline_sections(topic, comparison),
    )
    return ReviewOutlineResponse(
        topic=topic,
        status_message="已基于大模型增强比较结果生成综述提纲。",
        outline=outline,
        formatted_output=format_review_outline(outline),
    )


def enhance_outline_with_llm(
    agent: HSPansharpeningScholarAgent,
    targets: list[str] | None,
    topic: str,
    client: DashScopeClient | None,
    model_name: str,
) -> tuple[ReviewOutlineResponse, str | None]:
    """生成综述提纲，并在可用时复用大模型增强比较结果。"""

    cleaned_topic = topic.strip()
    if not cleaned_topic:
        raise ValueError("综述主题不能为空。")

    if client is None:
        return agent.generate_review_outline(topic=cleaned_topic, targets=targets), None

    knowledge_base = agent.ensure_knowledge_base_ready()
    if len(knowledge_base.documents) < 2:
        return agent.generate_review_outline(topic=cleaned_topic, targets=targets), None

    if targets:
        documents = agent.find_documents(targets, default_count=3)
    else:
        documents = agent.find_relevant_documents_for_topic(cleaned_topic, default_count=3)
    if len(documents) < 2:
        return agent.generate_review_outline(topic=cleaned_topic, targets=targets), None

    comparison_result, warning = enhance_comparison_with_llm(
        agent=agent,
        targets=[document.document_id for document in documents],
        topic_hint=cleaned_topic,
        client=client,
        model_name=model_name,
        default_count=3,
    )
    if comparison_result.comparison is None:
        fallback_result = agent.generate_review_outline(topic=cleaned_topic, targets=[document.document_id for document in documents])
        return fallback_result, warning

    return build_outline_from_comparison(cleaned_topic, comparison_result.comparison), warning


def augment_outline_with_visuals(
    *,
    agent: HSPansharpeningScholarAgent,
    result: ReviewOutlineResponse,
    vision_enabled: bool,
    vision_index_path: Path,
) -> ReviewOutlineResponse:
    """在综述提纲后追加可由图表摘要支持的小节建议。"""

    if result.outline is None:
        return result
    documents: list[Any] = []
    for file_name in result.outline.source_papers:
        try:
            document = agent.find_document(file_name)
        except Exception:
            document = None
        if document is not None:
            documents.append(document)

    visual_items = select_visual_items_for_documents(
        documents,
        vision_enabled=vision_enabled,
        vision_index_path=vision_index_path,
        max_items_per_document=3,
    )
    section_text = format_visual_items_section(
        visual_items,
        title="图表摘要可支持的提纲补充",
        max_items=8,
    )
    if not section_text:
        return result
    result.formatted_output = result.formatted_output.rstrip() + "\n\n" + section_text
    result.status_message += "（已加入图表摘要）"
    return result


def enhance_workflow_with_llm(
    agent: HSPansharpeningScholarAgent,
    topic: str,
    targets: list[str] | None,
    client: DashScopeClient | None,
    model_name: str,
    vision_enabled: bool,
    vision_index_path: Path,
    output_dir: str | Path | None = None,
) -> tuple[WorkflowResponse, str | None]:
    """执行 workflow，并在可用时使用大模型增强比较与提纲内容。"""

    cleaned_topic = topic.strip()
    if not cleaned_topic:
        raise ValueError("工作流主题不能为空。")

    knowledge_base = agent.ensure_knowledge_base_ready()
    if len(knowledge_base.documents) < 2:
        return agent.run_review_workflow(topic=cleaned_topic, targets=targets, output_dir=output_dir), None

    documents = agent.find_documents(targets, default_count=3)
    if len(documents) < 2:
        return agent.run_review_workflow(topic=cleaned_topic, targets=targets, output_dir=output_dir), None

    selected_document_ids = [document.document_id for document in documents]
    warning: str | None = None
    if client is None:
        comparison_result = agent.compare_papers(targets=selected_document_ids, topic_hint=cleaned_topic)
    else:
        comparison_result, warning = enhance_comparison_with_llm(
            agent=agent,
            targets=selected_document_ids,
            topic_hint=cleaned_topic,
            client=client,
            model_name=model_name,
            default_count=3,
        )

    if comparison_result.comparison is not None:
        outline_result = build_outline_from_comparison(cleaned_topic, comparison_result.comparison)
    else:
        comparison_result = agent.compare_papers(targets=selected_document_ids, topic_hint=cleaned_topic)
        outline_result = agent.generate_review_outline(topic=cleaned_topic, targets=selected_document_ids)

    plan = build_default_workflow_plan(topic=cleaned_topic, targets=selected_document_ids)
    state = WorkflowState(topic=cleaned_topic, targets=selected_document_ids)
    mark_step_status(plan, "select_papers", "completed")
    state.selected_papers = [document.file_name for document in documents]
    state.step_logs.append(f"步骤 1：已选中 {len(documents)} 篇论文。")

    graph_text, graph_log = agent.retrieve_workflow_graph_text(cleaned_topic, documents)
    mark_step_status(plan, "retrieve_graphrag", "completed")
    state.graph_text = graph_text
    state.step_logs.append(graph_log)

    visual_items = select_visual_items_for_documents(
        documents,
        vision_enabled=vision_enabled,
        vision_index_path=vision_index_path,
        max_items_per_document=4,
    )
    state.vision_text = format_visual_items_section(visual_items, max_items=12)
    if state.vision_text:
        state.step_logs.append(f"步骤 2：已加载图表摘要辅助发现（图表页 {len(visual_items)} 条）。")
    elif vision_enabled:
        state.step_logs.append("步骤 2：图表摘要辅助跳过，当前未命中可用 vision_index 证据。")

    mark_step_status(plan, "compare_papers", "completed")
    state.comparison_text = comparison_result.formatted_output
    used_llm = warning is None and "大模型增强" in comparison_result.status_message
    if used_llm:
        state.step_logs.append(f"步骤 3：已使用大模型增强完成多篇论文比较（{model_name}）。")
    else:
        state.step_logs.append("步骤 3：已完成多篇论文比较。")
    if warning:
        state.step_logs.append(f"提示：{warning}")

    mark_step_status(plan, "generate_outline", "completed")
    state.outline_text = outline_result.formatted_output
    if used_llm:
        state.step_logs.append("步骤 4：已基于大模型增强比较结果生成综述提纲。")
    else:
        state.step_logs.append("步骤 4：已生成综述提纲。")

    export_step_log = "步骤 5：已导出 Markdown 报告。"
    artifact = export_workflow_result(
        output_dir=str(Path(output_dir).expanduser().resolve()) if output_dir is not None else str(OUTPUT_DIR),
        topic=cleaned_topic,
        selected_papers=state.selected_papers,
        comparison_text=state.comparison_text,
        outline_text=state.outline_text,
        step_logs=state.step_logs + [export_step_log],
        graph_text=state.graph_text,
        vision_text=state.vision_text,
    )
    mark_step_status(plan, "export_markdown", "completed")
    state.export_artifact = artifact
    state.step_logs.append(f"步骤 5：已导出 Markdown 报告到 {artifact.output_path}")

    status_suffix = f"（已使用大模型增强：{model_name}）" if used_llm else ""
    workflow_result = WorkflowRunResult(
        status_message=f"已完成主题“{cleaned_topic}”的多步工作流{status_suffix}。",
        plan=plan,
        state=state,
        formatted_output="",
    )
    workflow_result.formatted_output = format_workflow_run_result(workflow_result)

    return (
        WorkflowResponse(
            topic=cleaned_topic,
            status_message=workflow_result.status_message,
            workflow_result=workflow_result,
            formatted_output=workflow_result.formatted_output,
        ),
        warning,
    )


def display_answer(result: AgentAnswer) -> None:
    """以命令行方式展示一次问答结果。"""

    print(f"\n检索路由：{result.retrieval_mode}")
    print("\n模型回答：")
    print(result.model_answer)

    print("\n来源依据：")
    if not result.sources:
        print("当前没有可展示的来源依据。")
        return

    for index, source in enumerate(result.sources, start=1):
        print(f"{index}. 论文：{source['file_name']}")
        print(f"   页码：{format_page_numbers(source['page_numbers'])}")
        print(f"   片段编号：{source['chunk_id']}")
        if source.get("source_type"):
            print(f"   来源类型：{source['source_type']}")
        print(f"   匹配词：{', '.join(source['matched_terms']) if source['matched_terms'] else '无'}")
        print(f"   相关分数：{source['score']}")
        print(f"   片段内容：{source['snippet']}")
        print(f"   文件路径：{source['source_path']}")


def display_analysis(result: PaperAnalysisResponse) -> None:
    """以命令行方式展示单篇论文结构化分析结果。"""

    print("\n学术分析：")
    print(result.status_message)
    print(result.formatted_output)


def display_comparison(result: PaperComparisonResponse) -> None:
    """以命令行方式展示多篇论文比较结果。"""

    print("\n多篇比较：")
    print(result.status_message)
    print(result.formatted_output)


def display_outline(result: ReviewOutlineResponse) -> None:
    """以命令行方式展示综述提纲生成结果。"""

    print("\n综述提纲：")
    print(result.status_message)
    outline_text = result.formatted_output
    if outline_text.startswith("综述提纲：\n"):
        outline_text = "\n".join(outline_text.splitlines()[1:])
    print(outline_text)


def display_workflow(result: WorkflowResponse) -> None:
    """以命令行方式展示多步工作流执行结果。"""

    print("\n多步工作流：")
    print(result.status_message)
    print(result.formatted_output)


def display_embedding_index_status(result: EmbeddingIndexResponse) -> None:
    """以命令行方式展示向量索引状态。"""

    print("\n向量索引：")
    print(result.status_message)
    print(f"索引路径：{result.index_path}")
    print(f"向量数量：{result.vector_count}")
    print(f"是否缓存加载：{'是' if result.loaded_from_cache else '否'}")


def display_graph_index_status(result: GraphIndexResponse) -> None:
    """以命令行方式展示图谱索引状态。"""

    print("\n图谱索引：")
    print(result.status_message)
    print(f"索引路径：{result.index_path}")
    print(f"实体数量：{result.entity_count}")
    print(f"关系数量：{result.relation_count}")
    print(f"图谱社群数量：{result.community_count}")
    print(f"图谱社群摘要数量：{result.community_summary_count}")
    print(f"是否缓存加载：{'是' if result.loaded_from_cache else '否'}")
    print(f"source_fingerprint 是否匹配：{'是' if result.source_fingerprint_match else '否'}")
    if result.build_warnings:
        print("构建提示：")
        for warning in result.build_warnings[:5]:
            print(f"- {warning}")


def display_paper_list(agent: HSPansharpeningScholarAgent) -> None:
    """展示当前可用论文列表。"""

    paper_items = agent.list_available_papers()
    print("\n当前可用论文：")
    if not paper_items:
        print("当前没有可用论文。")
        return

    for item in paper_items:
        short_document_id = str(item["document_id"])
        print(
            f"{item['index']}. [{short_document_id}] {item['file_name']} | "
            f"页数：{item['total_pages']} | 字符数：{item['total_characters']}"
        )


def show_cli_help() -> None:
    """输出命令行帮助信息。"""

    print("\n命令说明：")
    print("- 直接输入问题：执行论文检索问答（自动在 Hybrid RAG / GraphRAG 之间路由）")
    print("- ask :: 问题：执行论文检索问答")
    print("- build_index：构建本地向量索引")
    print("- rebuild_index：强制重建本地向量索引")
    print("- build_graph：构建本地图谱索引")
    print("- rebuild_graph：强制重建本地图谱索引")
    print("- graph_status：查看图谱索引状态")
    print("- build_vision：构建多模态图表索引")
    print("- rebuild_vision：强制重建多模态图表索引")
    print("- vision_status：查看多模态图表索引状态")
    print("- figures：列出已识别的图表页")
    print("- vision :: 问题 / 看图 :: 问题：直接基于图表摘要提问")
    print("- papers：查看论文列表")
    print("- analyze：分析第一篇论文")
    print("- analyze 1：分析第 1 篇论文")
    print("- analyze 关键词：按文件名或文档编号模糊匹配分析")
    print("- compare：比较前两篇论文")
    print("- compare 1,2：比较指定论文")
    print("- compare 1,2 :: 空间-光谱融合机制：按主题比较指定论文")
    print("- outline 高光谱全色锐化中的空间-光谱融合机制综述：基于默认论文生成综述提纲")
    print("- outline 1,2,3 :: 高光谱全色锐化中的注意力与门控机制综述：基于指定论文生成综述提纲")
    print("- workflow 高光谱全色锐化中的 zero-shot 与 diffusion 方法综述：基于默认论文执行多步工作流（启用大模型时自动增强）")
    print("- workflow 1,2,3：基于指定论文执行默认综述工作流")
    print("- workflow 1,2,3 :: 基于指定论文生成高光谱全色锐化综述：基于指定论文执行多步工作流")
    print("- help：查看帮助")
    print("- exit：退出程序")


def parse_analyze_target(user_input: str) -> str | None:
    """从命令行输入中解析分析目标。"""

    parts = user_input.strip().split(maxsplit=1)
    if len(parts) < 2:
        return None
    return parts[1].strip() or None


def parse_target_list(target_text: str) -> list[str]:
    """将命令中的目标论文文本解析为目标列表。"""

    cleaned_text = target_text.replace("，", ",").strip()
    if not cleaned_text:
        return []
    if "," in cleaned_text:
        return [item.strip() for item in cleaned_text.split(",") if item.strip()]
    return [item.strip() for item in cleaned_text.split() if item.strip()]


def parse_compare_request(user_input: str) -> tuple[list[str] | None, str]:
    """从 compare 命令中解析目标论文列表与可选比较主题。"""

    parts = user_input.strip().split(maxsplit=1)
    if len(parts) < 2:
        return None, "未指定比较主题"

    payload = parts[1].strip()
    if "::" in payload:
        target_text, topic_hint = payload.split("::", maxsplit=1)
        targets = parse_target_list(target_text)
        return (targets or None), (topic_hint.strip() or "未指定比较主题")

    targets = parse_target_list(payload)
    return (targets or None), "未指定比较主题"


def parse_outline_request(user_input: str) -> tuple[list[str] | None, str]:
    """从 outline 命令中解析目标论文列表与综述主题。"""

    parts = user_input.strip().split(maxsplit=1)
    if len(parts) < 2:
        return None, ""

    payload = parts[1].strip()
    if "::" not in payload:
        return None, payload

    target_text, topic = payload.split("::", maxsplit=1)
    targets = parse_target_list(target_text)
    return (targets or None), topic.strip()


def is_numeric_target_shortcut(target_text: str) -> bool:
    """判断是否为 workflow 1,2,3 这类目标论文简写。"""

    targets = parse_target_list(target_text)
    return bool(targets) and all(re.fullmatch(r"\d+", target) for target in targets)


def parse_workflow_request(user_input: str) -> tuple[list[str] | None, str]:
    """从 workflow 命令中解析目标论文列表与工作流主题。"""

    parts = user_input.strip().split(maxsplit=1)
    if len(parts) < 2:
        return None, ""

    payload = parts[1].strip()
    if "::" in payload:
        target_text, topic = payload.split("::", maxsplit=1)
        targets = parse_target_list(target_text)
        return (targets or None), (topic.strip() or DEFAULT_WORKFLOW_TOPIC)

    if is_numeric_target_shortcut(payload):
        return parse_target_list(payload), DEFAULT_WORKFLOW_TOPIC

    return None, payload


def run_cli_chat(
    agent: HSPansharpeningScholarAgent,
    llm_client: DashScopeClient | None,
    answer_model: str,
    analysis_model: str,
    vision_model: str,
    embedding_model: str,
    embedding_dimensions: int,
    processed_data_dir: Path,
    graph_index_path: Path,
    graph_extraction_model: str,
    graph_summary_model: str,
    vision_enabled: bool,
    vision_index_path: Path,
    vision_image_dir: Path,
    vision_render_zoom: float,
    vision_max_pages_per_paper: int,
    vision_top_k: int,
) -> None:
    """启动命令行交互循环。"""

    while True:
        try:
            user_input = input("\n请输入你的问题或命令：").strip()
        except EOFError:
            print("\n检测到输入结束，程序已退出。")
            break
        except KeyboardInterrupt:
            print("\n检测到手动中断，程序已退出。")
            break

        normalized_input = user_input.lower()
        if normalized_input in {"exit", "quit", "q", "退出", "结束"}:
            print("程序已退出，欢迎下次继续使用。")
            break

        if not user_input:
            print("请输入有效问题或命令，输入 help 查看说明。")
            continue

        if normalized_input in {"help", "帮助"}:
            show_cli_help()
            continue

        if normalized_input in {"papers", "list", "论文", "论文列表"}:
            try:
                display_paper_list(agent)
            except Exception as exc:
                print(f"读取论文列表失败：{exc}")
            continue

        if normalized_input in {"build_index", "index"}:
            try:
                embedding_result = agent.prepare_embedding_index(
                    client=llm_client,
                    model_name=embedding_model,
                    dimensions=embedding_dimensions,
                    processed_data_dir=processed_data_dir,
                    build_if_missing=True,
                    force_rebuild=False,
                )
            except Exception as exc:
                print(f"向量索引构建失败：{exc}")
                continue

            display_embedding_index_status(embedding_result)
            continue

        if normalized_input == "rebuild_index":
            try:
                embedding_result = agent.prepare_embedding_index(
                    client=llm_client,
                    model_name=embedding_model,
                    dimensions=embedding_dimensions,
                    processed_data_dir=processed_data_dir,
                    build_if_missing=True,
                    force_rebuild=True,
                )
            except Exception as exc:
                print(f"向量索引重建失败：{exc}")
                continue

            display_embedding_index_status(embedding_result)
            continue

        if normalized_input == "build_graph":
            try:
                graph_result = agent.prepare_graph_index(
                    client=llm_client,
                    processed_data_dir=processed_data_dir,
                    graph_index_path=graph_index_path,
                    extraction_model=graph_extraction_model,
                    summary_model=graph_summary_model,
                    embedding_model=embedding_model,
                    embedding_dimensions=embedding_dimensions,
                    build_if_missing=True,
                    force_rebuild=False,
                )
            except Exception as exc:
                print(f"图谱索引构建失败：{exc}")
                continue

            display_graph_index_status(graph_result)
            continue

        if normalized_input == "rebuild_graph":
            try:
                graph_result = agent.prepare_graph_index(
                    client=llm_client,
                    processed_data_dir=processed_data_dir,
                    graph_index_path=graph_index_path,
                    extraction_model=graph_extraction_model,
                    summary_model=graph_summary_model,
                    embedding_model=embedding_model,
                    embedding_dimensions=embedding_dimensions,
                    build_if_missing=True,
                    force_rebuild=True,
                )
            except Exception as exc:
                print(f"图谱索引重建失败：{exc}")
                continue

            display_graph_index_status(graph_result)
            continue

        if normalized_input == "graph_status":
            try:
                graph_result = agent.get_graph_index_status(
                    graph_index_path=graph_index_path,
                    processed_data_dir=processed_data_dir,
                )
            except Exception as exc:
                print(f"读取图谱索引状态失败：{exc}")
                continue

            display_graph_index_status(graph_result)
            continue

        if normalized_input == "build_vision":
            if not vision_enabled:
                print("多模态图表理解当前已通过 VISION_ENABLED 关闭。")
                continue
            try:
                status = build_vision_index_for_papers(
                    agent=agent,
                    client=llm_client,
                    model_name=vision_model,
                    image_dir=vision_image_dir,
                    index_path=vision_index_path,
                    zoom=vision_render_zoom,
                    max_pages_per_paper=vision_max_pages_per_paper,
                    force_rebuild=False,
                )
            except Exception as exc:
                print(f"多模态图表索引构建失败：{exc}")
                continue
            display_vision_status(status, vision_enabled=vision_enabled, model_name=vision_model)
            continue

        if normalized_input == "rebuild_vision":
            if not vision_enabled:
                print("多模态图表理解当前已通过 VISION_ENABLED 关闭。")
                continue
            try:
                status = build_vision_index_for_papers(
                    agent=agent,
                    client=llm_client,
                    model_name=vision_model,
                    image_dir=vision_image_dir,
                    index_path=vision_index_path,
                    zoom=vision_render_zoom,
                    max_pages_per_paper=vision_max_pages_per_paper,
                    force_rebuild=True,
                )
            except Exception as exc:
                print(f"多模态图表索引重建失败：{exc}")
                continue
            display_vision_status(status, vision_enabled=vision_enabled, model_name=vision_model)
            continue

        if normalized_input == "vision_status":
            status = get_vision_status(vision_index_path, vision_image_dir)
            display_vision_status(status, vision_enabled=vision_enabled, model_name=vision_model)
            continue

        if normalized_input in {"figures", "图表", "图表列表"}:
            display_figures(vision_index_path)
            continue

        if normalized_input.startswith("vision") or user_input.startswith("看图"):
            if not vision_enabled:
                print("多模态图表理解当前已通过 VISION_ENABLED 关闭。")
                continue
            vision_question = parse_vision_question(user_input)
            if not vision_question:
                print("请输入图表问题，例如：vision :: 哪些论文的实验表格报告了 PSNR、SSIM、SAM 和 ERGAS？")
                continue
            answer_text, visual_items, warning = answer_visual_question(
                question=vision_question,
                client=llm_client,
                model_name=answer_model,
                vision_index_path=vision_index_path,
                top_k=vision_top_k,
            )
            if warning:
                print(f"提示：{warning}")
            print("\n图表问答：")
            print(answer_text)
            if visual_items:
                print("\n图表来源：")
                for index, item in enumerate(visual_items, start=1):
                    print(
                        f"{index}. {item.get('file_name', '未知论文')} | "
                        f"页码：{item.get('page_number', '未知')} | "
                        f"caption：{item.get('caption', '')}"
                    )
            continue

        if normalized_input.startswith("ask") or user_input.startswith("问答"):
            ask_question = parse_ask_question(user_input)
            if not ask_question:
                print("请输入问题，例如：ask :: 这些论文如何保持光谱一致性？")
                continue
            try:
                answer_result = agent.answer(ask_question)
                visual_items = get_question_visual_items(
                    ask_question,
                    vision_enabled=vision_enabled,
                    vision_index_path=vision_index_path,
                    top_k=vision_top_k,
                    require_visual_query=True,
                )
                answer_result, warning = enhance_answer_with_llm(
                    answer_result,
                    llm_client,
                    answer_model,
                    visual_items=visual_items,
                )
            except ValueError as exc:
                print(f"输入有误：{exc}")
                continue
            except Exception as exc:
                print(f"问答过程出现异常：{exc}")
                continue

            if warning:
                print(f"提示：{warning}")
            display_answer(answer_result)
            continue

        if normalized_input.startswith("analyze") or user_input.startswith("分析"):
            # 分析命令支持：默认第一篇、序号、关键词三种入口。
            analyze_target = parse_analyze_target(user_input)
            try:
                analysis_result, warning = enhance_analysis_with_llm(
                    agent=agent,
                    target=analyze_target,
                    client=llm_client,
                    model_name=analysis_model,
                )
                analysis_result = augment_analysis_with_visuals(
                    agent=agent,
                    result=analysis_result,
                    target=analyze_target,
                    vision_enabled=vision_enabled,
                    vision_index_path=vision_index_path,
                )
            except Exception as exc:
                print(f"学术分析过程出现异常：{exc}")
                continue

            if warning:
                print(f"提示：{warning}")
            display_analysis(analysis_result)
            continue

        if normalized_input.startswith("compare") or user_input.startswith("对比"):
            compare_targets, compare_topic_hint = parse_compare_request(user_input)
            try:
                comparison_result, warning = enhance_comparison_with_llm(
                    agent=agent,
                    targets=compare_targets,
                    topic_hint=compare_topic_hint,
                    client=llm_client,
                    model_name=analysis_model,
                )
                comparison_result = augment_comparison_with_visuals(
                    agent=agent,
                    result=comparison_result,
                    vision_enabled=vision_enabled,
                    vision_index_path=vision_index_path,
                )
            except ValueError as exc:
                print(f"多篇比较输入有误：{exc}")
                continue
            except Exception as exc:
                print(f"多篇比较过程出现异常：{exc}")
                continue

            if warning:
                print(f"提示：{warning}")
            display_comparison(comparison_result)
            continue

        if normalized_input.startswith("outline") or user_input.startswith("提纲"):
            outline_targets, outline_topic = parse_outline_request(user_input)
            if not outline_topic:
                print("请输入综述主题，例如：outline 高光谱全色锐化中的空间-光谱融合机制综述")
                continue

            try:
                outline_result, warning = enhance_outline_with_llm(
                    agent=agent,
                    topic=outline_topic,
                    targets=outline_targets,
                    client=llm_client,
                    model_name=analysis_model,
                )
                outline_result = augment_outline_with_visuals(
                    agent=agent,
                    result=outline_result,
                    vision_enabled=vision_enabled,
                    vision_index_path=vision_index_path,
                )
            except ValueError as exc:
                print(f"综述提纲输入有误：{exc}")
                continue
            except Exception as exc:
                print(f"综述提纲生成过程出现异常：{exc}")
                continue

            if warning:
                print(f"提示：{warning}")
            display_outline(outline_result)
            continue

        if normalized_input.startswith("workflow") or user_input.startswith("流程"):
            workflow_targets, workflow_topic = parse_workflow_request(user_input)
            if not workflow_topic:
                print("请输入工作流主题，例如：workflow 高光谱全色锐化中的 zero-shot 与 diffusion 方法综述")
                continue

            try:
                workflow_result, warning = enhance_workflow_with_llm(
                    agent=agent,
                    topic=workflow_topic,
                    targets=workflow_targets,
                    client=llm_client,
                    model_name=analysis_model,
                    vision_enabled=vision_enabled,
                    vision_index_path=vision_index_path,
                )
            except ValueError as exc:
                print(f"多步工作流输入有误：{exc}")
                continue
            except Exception as exc:
                print(f"多步工作流执行过程出现异常：{exc}")
                continue

            if warning:
                print(f"提示：{warning}")
            display_workflow(workflow_result)
            continue

        try:
            # 普通输入按问答处理：先本地检索，再按配置进行大模型增强。
            answer_result = agent.answer(user_input)
            visual_items = get_question_visual_items(
                user_input,
                vision_enabled=vision_enabled,
                vision_index_path=vision_index_path,
                top_k=vision_top_k,
                require_visual_query=True,
            )
            answer_result, warning = enhance_answer_with_llm(
                answer_result,
                llm_client,
                answer_model,
                visual_items=visual_items,
            )
        except ValueError as exc:
            print(f"输入有误：{exc}")
            continue
        except Exception as exc:
            print(f"问答过程出现异常：{exc}")
            continue

        if warning:
            print(f"提示：{warning}")
        display_answer(answer_result)


def main() -> None:
    """运行命令行主入口。"""

    config = get_app_config()
    # 启动时先确保基础目录存在，避免后续读写路径失败。
    ensure_directories(
        [
            config["data_dir"],
            config["raw_papers_dir"],
            config["processed_data_dir"],
            config["output_dir"],
            config["vision_image_dir"],
        ]
    )

    llm_client = initialize_llm_client(config)
    answer_model = str(config.get("dashscope_answer_model", "qwen3.7-plus"))
    analysis_model = str(config.get("dashscope_analysis_model", "deepseek-v4-pro"))
    vision_model = str(config.get("dashscope_vision_model", "qwen3.7-plus"))
    embedding_model = str(config.get("dashscope_embedding_model", "text-embedding-v4"))
    embedding_dimensions = int(config.get("dashscope_embedding_dimensions", 128))
    graph_enabled = bool(config.get("graph_enabled", True))
    graph_index_path = Path(config.get("graph_index_path", config["processed_data_dir"] / "graph_index.json"))
    graph_extraction_model = str(config.get("graph_extraction_model", analysis_model))
    graph_summary_model = str(config.get("graph_summary_model", answer_model))
    vision_enabled = bool(config.get("vision_enabled", True))
    vision_index_path = Path(config.get("vision_index_path", config["processed_data_dir"] / "vision_index.json"))
    vision_image_dir = Path(config.get("vision_image_dir", config["processed_data_dir"] / "vision"))
    vision_render_zoom = float(config.get("vision_render_zoom", 2.0))
    vision_max_pages_per_paper = int(config.get("vision_max_pages_per_paper", 30))
    vision_top_k = int(config.get("vision_top_k", 3))

    agent = HSPansharpeningScholarAgent(raw_papers_dir=config["raw_papers_dir"])
    try:
        knowledge_base = agent.build_knowledge_base()
    except ImportError as exc:
        print(f"依赖缺失：{exc}")
        return
    except Exception as exc:
        print(f"知识库构建失败：{exc}")
        return

    embedding_status: EmbeddingIndexResponse | None = None
    graph_status: GraphIndexResponse | None = None
    if llm_client is not None and knowledge_base.chunk_records:
        try:
            embedding_status = agent.prepare_embedding_index(
                client=llm_client,
                model_name=embedding_model,
                dimensions=embedding_dimensions,
                processed_data_dir=config["processed_data_dir"],
                build_if_missing=False,
                force_rebuild=False,
            )
        except Exception as exc:
            embedding_status = EmbeddingIndexResponse(
                status_message=f"向量索引初始化失败：{exc}",
                index_path="",
                vector_count=0,
                loaded_from_cache=False,
            )

    if graph_enabled and knowledge_base.chunk_records:
        try:
            graph_status = agent.prepare_graph_index(
                client=llm_client,
                processed_data_dir=config["processed_data_dir"],
                graph_index_path=graph_index_path,
                extraction_model=graph_extraction_model,
                summary_model=graph_summary_model,
                embedding_model=embedding_model,
                embedding_dimensions=embedding_dimensions,
                build_if_missing=False,
                force_rebuild=False,
            )
        except Exception as exc:
            graph_status = GraphIndexResponse(
                status_message=f"图谱索引初始化失败：{exc}",
                index_path=str(graph_index_path),
                entity_count=0,
                relation_count=0,
                community_count=0,
                community_summary_count=0,
                loaded_from_cache=False,
                source_fingerprint_match=False,
                build_warnings=[],
            )

    show_startup_info(config, knowledge_base, llm_client, embedding_status, graph_status)
    show_parse_error_summary(knowledge_base.parse_errors)

    if not knowledge_base.pdf_files:
        print("当前未在 data/raw_papers/ 发现 PDF，暂时无法进入交互。")
        print("请先放入论文文件后重新运行。")
        return

    if not knowledge_base.documents:
        print("当前没有可用论文被成功解析，暂时无法进行问答或分析。")
        print("请检查 PDF 是否可被正常解析，或补充新的论文后再试。")
        return

    if not knowledge_base.chunk_records:
        print("提示：当前没有可检索片段，问答可能无法返回有效结果。")

    run_cli_chat(
        agent=agent,
        llm_client=llm_client,
        answer_model=answer_model,
        analysis_model=analysis_model,
        vision_model=vision_model,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        processed_data_dir=config["processed_data_dir"],
        graph_index_path=graph_index_path,
        graph_extraction_model=graph_extraction_model,
        graph_summary_model=graph_summary_model,
        vision_enabled=vision_enabled,
        vision_index_path=vision_index_path,
        vision_image_dir=vision_image_dir,
        vision_render_zoom=vision_render_zoom,
        vision_max_pages_per_paper=vision_max_pages_per_paper,
        vision_top_k=vision_top_k,
    )


if __name__ == "__main__":
    main()
