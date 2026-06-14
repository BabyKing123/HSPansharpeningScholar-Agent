"""本模块作用：在整个智能体中负责串联论文加载、检索、问答与最小学术分析，形成可运行的本地论文助手闭环。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from config import (
    GRAPH_ENABLED,
    GRAPH_EXTRACTION_MODEL,
    GRAPH_GLOBAL_TOP_K_COMMUNITIES,
    GRAPH_INDEX_PATH,
    GRAPH_LOCAL_TOP_K_CHUNKS,
    GRAPH_LOCAL_TOP_K_ENTITIES,
    GRAPH_MAX_HOPS,
    GRAPH_SUMMARY_MODEL,
    OUTPUT_DIR,
)
from graph import (
    GraphIndex,
    GraphRetrievedContext,
    build_graph_index,
    get_default_graph_index_path,
    is_graph_index_compatible,
    load_graph_index,
    retrieve_graph_global,
    retrieve_graph_local,
    retrieve_graph_mixed,
    route_query_for_retrieval,
    save_graph_index,
    summarize_graph_index_status,
)
from llm_dashscope import DashScopeClient
# 提示词模块：集中维护回答规则与兜底文案，避免把硬编码文本散落在业务逻辑里。
from core.prompts import (
    build_answer_suffix,
    build_answer_system_prompt,
    build_answer_task_prompt,
    build_empty_library_message,
    build_no_result_message,
)
from core.workflow import (
    WorkflowRunResult,
    WorkflowState,
    build_default_workflow_plan,
    export_workflow_result,
    format_workflow_run_result,
    mark_step_status,
)
from rag.embedder import (
    EmbeddingIndex,
    build_embedding_index as run_build_embedding_index,
    get_default_embedding_index_path,
    is_embedding_index_compatible,
    load_embedding_index,
)
# RAG 基础链路：文件发现 -> PDF 解析 -> 文本切块 -> 关键词检索。
from rag.loader import list_pdf_files
from rag.parser import ParsedDocument, parse_pdf_files
from rag.retriever import (
    RetrievedChunk,
    extract_search_terms,
    retrieve_relevant_chunks,
    retrieve_relevant_chunks_hybrid,
)
from rag.splitter import build_knowledge_base_records
# 学术分析工具：负责单篇论文结构化提取与结果格式化。
from tools.analyze_tool import StructuredPaperAnalysis, analyze_single_paper, format_analysis_result
from tools.compare_tool import MultiPaperComparison, compare_papers as run_compare_papers, format_comparison_result
from tools.outline_tool import ReviewOutline, format_review_outline, generate_review_outline


@dataclass
class KnowledgeBaseState:
    """本数据结构作用：保存当前本地论文知识库的最小运行状态。"""

    raw_papers_dir: str  # 原始论文目录（绝对路径字符串）。
    pdf_files: list[str] = field(default_factory=list)  # 扫描到的 PDF 文件路径列表。
    documents: list[ParsedDocument] = field(default_factory=list)  # 成功解析后的文档对象列表。
    chunk_records: list[dict[str, object]] = field(default_factory=list)  # 可直接检索的扁平化切块记录。
    parse_errors: list[dict[str, str]] = field(default_factory=list)  # 解析失败清单（文件名/路径/错误信息）。


@dataclass
class AgentAnswer:
    """本数据结构作用：保存一次问答的最终输出，供命令行或后续界面展示。"""

    question: str  # 用户原始问题。
    model_answer: str  # 最终展示给用户的回答文本（规则生成或 LLM 增强后）。
    sources: list[dict[str, object]]  # 来源依据列表（文件、页码、片段、分数等）。
    retrieved_count: int  # 本次问答命中的片段数量。
    used_prompt: str  # 本次问答内部使用的提示词文本（便于调试与教学演示）。
    retrieval_mode: str = "hybrid_rag"  # 本次问答采用的检索路由。


@dataclass
class PaperAnalysisResponse:
    """本数据结构作用：保存一次单篇论文结构化分析结果，供主程序展示。"""

    target: str  # 用户指定的分析目标（序号/关键词/文件名）。
    status_message: str  # 状态提示（成功、回退、未命中等）。
    analysis: StructuredPaperAnalysis | None  # 结构化对象，失败时为 None。
    formatted_output: str  # 直接可打印的分析文本。


@dataclass
class PaperComparisonResponse:
    """本数据结构作用：保存一次多篇论文比较结果，供主程序展示。"""

    targets: list[str]
    status_message: str
    comparison: MultiPaperComparison | None
    formatted_output: str


@dataclass
class ReviewOutlineResponse:
    """本数据结构作用：保存一次综述提纲生成结果，供主程序展示。"""

    topic: str
    status_message: str
    outline: ReviewOutline | None
    formatted_output: str


@dataclass
class WorkflowResponse:
    """本数据结构作用：保存一次多步工作流执行结果，供主程序展示。"""

    topic: str
    status_message: str
    workflow_result: WorkflowRunResult | None
    formatted_output: str


@dataclass
class EmbeddingIndexResponse:
    """本数据结构作用：保存一次向量索引加载或构建结果，供主程序展示。"""

    status_message: str
    index_path: str
    vector_count: int
    loaded_from_cache: bool


@dataclass
class GraphIndexResponse:
    """本数据结构作用：保存一次图谱索引加载、构建或状态检查结果。"""

    status_message: str
    index_path: str
    entity_count: int
    relation_count: int
    community_count: int
    community_summary_count: int
    loaded_from_cache: bool
    source_fingerprint_match: bool
    build_warnings: list[str] = field(default_factory=list)


class HSPansharpeningScholarAgent:
    """本类作用：封装最小论文助手闭环，统一管理建库、检索、问答与单篇论文分析逻辑。"""

    def __init__(
        self,
        raw_papers_dir: str | Path,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 3,
    ) -> None:
        """初始化最小论文代理。

        输入：
            raw_papers_dir: 本地 PDF 论文目录。
            chunk_size: 单个文本块的最大字符数。
            chunk_overlap: 相邻文本块之间的重叠字符数。
            top_k: 单次检索最多返回多少条结果。
        输出：
            无。
        异常：
            当切块参数不合法时，相关异常会在后续建库阶段抛出。
        """

        # 目录统一转绝对路径，避免命令行启动位置不同导致找不到数据目录。
        self.raw_papers_dir = Path(raw_papers_dir).expanduser().resolve()
        # 切块参数影响检索粒度与上下文连续性。
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # top_k 决定单次问答最多使用多少个来源片段。
        self.top_k = top_k
        # 延迟构建知识库：首次问答/分析前自动建库，缩短对象初始化等待时间。
        self.knowledge_base: KnowledgeBaseState | None = None
        # 第四周新增：向量索引和向量查询运行时状态。
        self.embedding_index: EmbeddingIndex | None = None
        self.embedding_client: DashScopeClient | None = None
        self.embedding_model_name: str = ""
        self.embedding_dimensions: int = 0
        # GraphRAG 运行时状态：与原有 Hybrid RAG 并行存在，按路由选择使用。
        self.graph_enabled: bool = GRAPH_ENABLED
        self.graph_index: GraphIndex | None = None
        self.graph_index_path: str = str(GRAPH_INDEX_PATH)
        self.graph_extraction_model_name: str = GRAPH_EXTRACTION_MODEL
        self.graph_summary_model_name: str = GRAPH_SUMMARY_MODEL
        self.graph_max_hops: int = GRAPH_MAX_HOPS
        self.graph_local_top_k_entities: int = GRAPH_LOCAL_TOP_K_ENTITIES
        self.graph_local_top_k_chunks: int = GRAPH_LOCAL_TOP_K_CHUNKS
        self.graph_global_top_k_communities: int = GRAPH_GLOBAL_TOP_K_COMMUNITIES

    def build_knowledge_base(self) -> KnowledgeBaseState:
        """根据本地 PDF 构建最小知识库。

        输入：
            无。
        输出：
            当前知识库状态对象。
        异常：
            当 PDF 解析依赖缺失时，抛出 ImportError。
            当目录访问失败或切块参数不合法时，抛出对应异常。
        """

        pdf_paths = list_pdf_files(self.raw_papers_dir)
        # 解析阶段会把失败文件记录在 parse_errors 中，不影响其余文件继续入库。
        documents, parse_errors = parse_pdf_files(pdf_paths)
        chunk_records = build_knowledge_base_records(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        self.knowledge_base = KnowledgeBaseState(
            raw_papers_dir=str(self.raw_papers_dir),
            pdf_files=[str(path) for path in pdf_paths],
            documents=documents,
            chunk_records=chunk_records,
            parse_errors=parse_errors,
        )
        return self.knowledge_base

    def ensure_knowledge_base_ready(self) -> KnowledgeBaseState:
        """确保知识库已准备完成。

        输入：
            无。
        输出：
            当前知识库状态对象。
        异常：
            当自动建库失败时，抛出对应异常。
        """

        if self.knowledge_base is None:
            return self.build_knowledge_base()
        return self.knowledge_base

    def resolve_graph_index_path(
        self,
        graph_index_path: str | Path | None = None,
        processed_data_dir: str | Path | None = None,
    ) -> Path:
        """解析图谱索引文件路径。"""

        if graph_index_path is not None:
            return Path(graph_index_path).expanduser().resolve()
        if processed_data_dir is not None:
            return get_default_graph_index_path(processed_data_dir)
        return Path(self.graph_index_path).expanduser().resolve()

    def graph_context_has_evidence(self, context: GraphRetrievedContext | None) -> bool:
        """判断图谱检索结果是否包含可用证据。"""

        if context is None:
            return False
        return bool(context.entities or context.relations or context.communities or context.chunks)

    def route_query(self, question: str) -> str:
        """根据当前状态为问题选择检索路由。"""

        if not self.graph_enabled or self.graph_index is None:
            return "hybrid_rag"
        knowledge_base = self.ensure_knowledge_base_ready()
        if not is_graph_index_compatible(self.graph_index, knowledge_base.chunk_records):
            return "hybrid_rag"
        return route_query_for_retrieval(question)

    def prepare_graph_index(
        self,
        *,
        client: DashScopeClient | None,
        processed_data_dir: str | Path | None = None,
        graph_index_path: str | Path | None = None,
        extraction_model: str = "",
        summary_model: str = "",
        embedding_model: str = "",
        embedding_dimensions: int = 0,
        build_if_missing: bool = False,
        force_rebuild: bool = False,
    ) -> GraphIndexResponse:
        """加载或构建本地 GraphRAG 索引。"""

        knowledge_base = self.ensure_knowledge_base_ready()
        if not knowledge_base.chunk_records:
            raise ValueError("当前没有可用于构建图谱索引的切块记录。")

        index_path = self.resolve_graph_index_path(
            graph_index_path=graph_index_path,
            processed_data_dir=processed_data_dir,
        )
        self.graph_index_path = str(index_path)
        self.graph_extraction_model_name = extraction_model or self.graph_extraction_model_name
        self.graph_summary_model_name = summary_model or self.graph_summary_model_name

        cached_index = load_graph_index(index_path)
        if not force_rebuild and cached_index is not None and is_graph_index_compatible(
            cached_index,
            knowledge_base.chunk_records,
        ):
            self.graph_index = cached_index
            return GraphIndexResponse(
                status_message="已加载本地图谱索引。",
                index_path=str(index_path),
                entity_count=len(cached_index.entities),
                relation_count=len(cached_index.relations),
                community_count=len(cached_index.communities),
                community_summary_count=len(cached_index.community_summaries),
                loaded_from_cache=True,
                source_fingerprint_match=True,
                build_warnings=[
                    str(item)
                    for item in cached_index.build_metadata.get("build_warnings", [])
                    if str(item).strip()
                ],
            )

        if not build_if_missing:
            status_message = "当前未找到可用图谱索引，可执行 build_graph 构建。"
            fingerprint_match = False
            if cached_index is not None:
                fingerprint_match = is_graph_index_compatible(cached_index, knowledge_base.chunk_records)
                self.graph_index = cached_index if fingerprint_match else None
                status_message = (
                    "已发现图谱索引，但与当前 chunk_records 不匹配，可执行 rebuild_graph 重建。"
                    if not fingerprint_match
                    else "已发现可用图谱索引。"
                )
            else:
                self.graph_index = None
            status_index = cached_index
            return GraphIndexResponse(
                status_message=status_message,
                index_path=str(index_path),
                entity_count=len(status_index.entities) if status_index is not None else 0,
                relation_count=len(status_index.relations) if status_index is not None else 0,
                community_count=len(status_index.communities) if status_index is not None else 0,
                community_summary_count=len(status_index.community_summaries) if status_index is not None else 0,
                loaded_from_cache=False,
                source_fingerprint_match=fingerprint_match,
                build_warnings=[
                    str(item)
                    for item in (status_index.build_metadata.get("build_warnings", []) if status_index is not None else [])
                    if str(item).strip()
                ],
            )

        built_index = build_graph_index(
            chunk_records=knowledge_base.chunk_records,
            documents=knowledge_base.documents,
            client=client,
            extraction_model=extraction_model or self.graph_extraction_model_name,
            summary_model=summary_model or self.graph_summary_model_name,
            embedding_model=embedding_model or self.embedding_model_name,
            embedding_dimensions=embedding_dimensions or self.embedding_dimensions,
            existing_index=cached_index,
        )
        save_graph_index(built_index, index_path)
        self.graph_index = built_index
        return GraphIndexResponse(
            status_message="已完成本地图谱索引构建。",
            index_path=str(index_path),
            entity_count=len(built_index.entities),
            relation_count=len(built_index.relations),
            community_count=len(built_index.communities),
            community_summary_count=len(built_index.community_summaries),
            loaded_from_cache=False,
            source_fingerprint_match=True,
            build_warnings=[
                str(item)
                for item in built_index.build_metadata.get("build_warnings", [])
                if str(item).strip()
            ],
        )

    def get_graph_index_status(
        self,
        *,
        graph_index_path: str | Path | None = None,
        processed_data_dir: str | Path | None = None,
    ) -> GraphIndexResponse:
        """读取图谱索引状态，供 CLI 展示。"""

        knowledge_base = self.ensure_knowledge_base_ready()
        index_path = self.resolve_graph_index_path(
            graph_index_path=graph_index_path,
            processed_data_dir=processed_data_dir,
        )
        index = self.graph_index if self.graph_index is not None and self.graph_index_path == str(index_path) else load_graph_index(index_path)
        if index is not None:
            self.graph_index = index
            self.graph_index_path = str(index_path)

        status = summarize_graph_index_status(index, knowledge_base.chunk_records)
        if not status["exists"]:
            return GraphIndexResponse(
                status_message="当前尚未构建图谱索引。",
                index_path=str(index_path),
                entity_count=0,
                relation_count=0,
                community_count=0,
                community_summary_count=0,
                loaded_from_cache=False,
                source_fingerprint_match=False,
                build_warnings=[],
            )

        status_message = "图谱索引可用。"
        if not status["source_fingerprint_match"]:
            status_message = "图谱索引存在，但与当前 chunk_records 不匹配。"
        return GraphIndexResponse(
            status_message=status_message,
            index_path=str(status["index_path"] or index_path),
            entity_count=int(status["entity_count"]),
            relation_count=int(status["relation_count"]),
            community_count=int(status["community_count"]),
            community_summary_count=int(status["community_summary_count"]),
            loaded_from_cache=index is not None,
            source_fingerprint_match=bool(status["source_fingerprint_match"]),
            build_warnings=[
                str(item)
                for item in (index.build_metadata.get("build_warnings", []) if index is not None else [])
                if str(item).strip()
            ],
        )

    def retrieve_graph_context_for_question(
        self,
        question: str,
        route: str | None = None,
    ) -> GraphRetrievedContext | None:
        """根据问题执行图谱检索，并返回结构化上下文。"""

        knowledge_base = self.ensure_knowledge_base_ready()
        if not self.graph_enabled or self.graph_index is None:
            return None
        if not is_graph_index_compatible(self.graph_index, knowledge_base.chunk_records):
            return None

        selected_route = route or self.route_query(question)
        if selected_route == "graph_local":
            return retrieve_graph_local(
                question=question,
                graph_index=self.graph_index,
                max_hops=self.graph_max_hops,
                top_k_entities=self.graph_local_top_k_entities,
                top_k_chunks=self.graph_local_top_k_chunks,
            )
        if selected_route == "graph_global":
            return retrieve_graph_global(
                question=question,
                graph_index=self.graph_index,
                top_k_communities=self.graph_global_top_k_communities,
                client=self.embedding_client,
                embedding_model=self.embedding_model_name,
                embedding_dimensions=self.embedding_dimensions,
            )
        if selected_route == "graph_mixed":
            return retrieve_graph_mixed(
                question=question,
                graph_index=self.graph_index,
                chunk_records=knowledge_base.chunk_records,
                max_hops=self.graph_max_hops,
                top_k_entities=self.graph_local_top_k_entities,
                top_k_chunks=self.graph_local_top_k_chunks,
                top_k_communities=self.graph_global_top_k_communities,
                client=self.embedding_client,
                embedding_model=self.embedding_model_name,
                embedding_dimensions=self.embedding_dimensions,
                embedding_vectors=self.embedding_index.vectors if self.embedding_index is not None else None,
            )
        return None

    def retrieve_workflow_graph_text(
        self,
        topic: str,
        documents: list[ParsedDocument],
    ) -> tuple[str, str]:
        """为 workflow 检索 GraphRAG 辅助发现。"""

        if not documents:
            return "", "步骤 2：GraphRAG 辅助跳过，当前未选中论文。"
        if not self.graph_enabled or self.graph_index is None:
            return "", "步骤 2：GraphRAG 辅助跳过，当前未加载可用图谱索引。"

        knowledge_base = self.ensure_knowledge_base_ready()
        if not is_graph_index_compatible(self.graph_index, knowledge_base.chunk_records):
            return "", "步骤 2：GraphRAG 辅助跳过，图谱索引与当前知识库不匹配，请执行 rebuild_graph。"

        question = build_workflow_graph_question(topic, documents)
        try:
            graph_context = self.retrieve_graph_context_for_question(question, route="graph_mixed")
        except Exception as exc:
            return "", f"步骤 2：GraphRAG 辅助检索失败，已继续执行后续流程：{exc}"

        if not self.graph_context_has_evidence(graph_context):
            return "", "步骤 2：GraphRAG 辅助未命中可用图谱证据，已继续执行后续流程。"
        graph_text = format_workflow_graph_context(graph_context)
        evidence_count = len(graph_context.communities) + len(graph_context.relations) + len(graph_context.chunks)
        return graph_text, f"步骤 2：已完成 GraphRAG 辅助检索（{graph_context.retrieval_mode}，证据 {evidence_count} 条）。"

    def list_available_papers(self) -> list[dict[str, object]]:
        """列出当前知识库中可分析的论文信息。

        输入：
            无。
        输出：
            论文信息列表，每项包含序号、文件名、文档编号与页数统计。
        异常：
            当自动建库失败时，抛出对应异常。
        """

        knowledge_base = self.ensure_knowledge_base_ready()
        paper_items: list[dict[str, object]] = []
        for index, document in enumerate(knowledge_base.documents, start=1):
            paper_items.append(
                {
                    "index": index,
                    "file_name": document.file_name,
                    "document_id": document.document_id,
                    "total_pages": document.total_pages,
                    "total_characters": document.total_characters,
                }
            )
        return paper_items

    def find_document(self, target: str | None = None) -> ParsedDocument | None:
        """根据序号、文件名或文档编号定位目标论文。

        输入：
            target: 论文序号、文件名、文档编号或其片段。
        输出：
            命中的论文对象；若未找到则返回 None。
        异常：
            当自动建库失败时，抛出对应异常。
        """

        knowledge_base = self.ensure_knowledge_base_ready()
        if not knowledge_base.documents:
            return None

        if target is None or not target.strip():
            # 默认分析第一篇，保证命令 `analyze` 可直接运行。
            return knowledge_base.documents[0]

        normalized_target = target.strip().lower()
        if normalized_target.isdigit():
            document_index = int(normalized_target) - 1
            if 0 <= document_index < len(knowledge_base.documents):
                return knowledge_base.documents[document_index]
            return None

        # 先做精确匹配，再做模糊包含匹配，降低误命中概率。
        for document in knowledge_base.documents:
            if normalized_target == document.file_name.lower() or normalized_target == document.document_id.lower():
                return document

        for document in knowledge_base.documents:
            if normalized_target in document.file_name.lower() or normalized_target in document.document_id.lower():
                return document

        return None

    def find_documents(self, targets: list[str] | None = None, default_count: int = 2) -> list[ParsedDocument]:
        """根据多个目标定位多篇论文。

        输入：
            targets: 论文序号、文件名、文档编号或其片段列表。
            default_count: targets 为空时默认选取多少篇论文。
        输出：
            命中的论文对象列表。
        异常：
            当某个目标无法匹配论文时，抛出 ValueError。
            当自动建库失败时，抛出对应异常。
        """

        knowledge_base = self.ensure_knowledge_base_ready()
        if not knowledge_base.documents:
            return []

        if not targets:
            return knowledge_base.documents[: min(default_count, len(knowledge_base.documents))]

        selected_documents: list[ParsedDocument] = []
        seen_ids: set[str] = set()
        for target in targets:
            document = self.find_document(target)
            if document is None:
                raise ValueError(f"未找到目标论文：{target}")
            if document.document_id in seen_ids:
                continue
            seen_ids.add(document.document_id)
            selected_documents.append(document)
        return selected_documents

    def expand_topic_terms_for_document_selection(self, topic: str) -> list[str]:
        """根据综述主题扩展论文默认选择所需的领域关键词。"""

        lowered_topic = topic.lower()
        terms = extract_search_terms(topic)
        if re.search(r"高光谱全色锐化|hyperspectral pansharpen|hsi pansharpen", lowered_topic):
            terms.extend(
                [
                    "hyperspectral pansharpening",
                    "hsi pansharpening",
                    "hyperspectral pan-sharpening",
                    "lrhs",
                    "hrhs",
                    "panchromatic",
                    "pan",
                ]
            )
        if re.search(r"全色锐化|pansharpen|pan-sharpen", lowered_topic):
            terms.extend(["pansharpening", "pan-sharpening", "panchromatic", "pan"])
        if re.search(r"空间.?光谱|spatial.?spectral|spectral.?spatial", lowered_topic):
            terms.extend(["spatial-spectral", "spatial spectral", "spectral-spatial", "spatial", "spectral"])
        if re.search(r"融合|fusion", lowered_topic):
            terms.extend(["fusion", "image fusion"])
        if re.search(r"扩散|diffusion", lowered_topic):
            terms.extend(["diffusion", "latent diffusion"])
        if re.search(r"zero.?shot|零样本", lowered_topic):
            terms.extend(["zero-shot", "zero shot"])
        return list(dict.fromkeys(term.strip().lower() for term in terms if term.strip()))

    def score_document_for_topic(self, document: ParsedDocument, topic: str) -> float:
        """为未指定目标的综述任务计算论文相关性分数。"""

        title_text = re.sub(r"[_\-]+", " ", document.file_name.lower())
        head_text = document.full_text[:8000].lower()
        topic_terms = self.expand_topic_terms_for_document_selection(topic)
        score = 0.0
        for term in topic_terms:
            if term in title_text:
                score += 8.0
            if term in head_text:
                score += 1.0

        if re.search(r"高光谱全色锐化|hyperspectral pansharpen|hsi pansharpen", topic.lower()):
            if re.search(r"hyperspectral.*pansharpen|hsi.*pansharpen", title_text):
                score += 30.0
            elif "hyperspectral" in title_text and "pansharpen" in head_text:
                score += 12.0
            if "multispectral" in title_text and "hyperspectral" not in title_text:
                score -= 4.0
        if re.search(r"空间.?光谱|spatial.?spectral|spectral.?spatial", topic.lower()):
            if re.search(r"spatial.*spectral|spectral.*spatial", title_text):
                score += 12.0
        return score

    def find_relevant_documents_for_topic(self, topic: str, default_count: int = 3) -> list[ParsedDocument]:
        """未指定目标论文时，按综述主题选择默认论文。"""

        knowledge_base = self.ensure_knowledge_base_ready()
        if not knowledge_base.documents:
            return []

        scored_documents = [
            (self.score_document_for_topic(document, topic), index, document)
            for index, document in enumerate(knowledge_base.documents)
        ]
        scored_documents.sort(key=lambda item: (-item[0], item[1]))
        if not scored_documents or scored_documents[0][0] <= 0:
            return knowledge_base.documents[: min(default_count, len(knowledge_base.documents))]
        return [document for _, _, document in scored_documents[: min(default_count, len(scored_documents))]]

    def answer(self, question: str) -> AgentAnswer:
        """根据用户问题生成最小回答结果。

        输入：
            question: 用户问题文本。
        输出：
            包含模型回答与来源依据的结果对象。
        异常：
            当问题为空时，抛出 ValueError。
            当知识库尚未构建且自动构建失败时，抛出对应异常。
        """

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("问题不能为空。")

        knowledge_base = self.ensure_knowledge_base_ready()
        if not knowledge_base.chunk_records:
            return AgentAnswer(
                question=cleaned_question,
                model_answer=build_empty_library_message(),
                sources=[],
                retrieved_count=0,
                used_prompt=build_answer_system_prompt(),
                retrieval_mode="hybrid_rag",
            )

        selected_route = self.route_query(cleaned_question)
        graph_context: GraphRetrievedContext | None = None
        if selected_route != "hybrid_rag":
            try:
                graph_context = self.retrieve_graph_context_for_question(
                    cleaned_question,
                    route=selected_route,
                )
            except Exception:
                graph_context = None

        if self.graph_context_has_evidence(graph_context):
            context_blocks = build_unified_context_blocks(graph_context=graph_context)
            used_prompt = (
                build_answer_system_prompt()
                + "\n\n"
                + build_answer_task_prompt(cleaned_question, context_blocks)
            )
            return AgentAnswer(
                question=cleaned_question,
                model_answer=synthesize_graph_answer(cleaned_question, graph_context),
                sources=build_graph_source_entries(graph_context),
                retrieved_count=len(graph_context.chunks) + len(graph_context.communities) + len(graph_context.relations),
                used_prompt=used_prompt,
                retrieval_mode=selected_route,
            )

        retrieved_chunks = self.retrieve_chunks_for_question(cleaned_question, knowledge_base.chunk_records)
        context_blocks = build_unified_context_blocks(retrieved_chunks=retrieved_chunks)
        used_prompt = (
            build_answer_system_prompt()
            + "\n\n"
            + build_answer_task_prompt(cleaned_question, context_blocks)
        )

        if not retrieved_chunks:
            return AgentAnswer(
                question=cleaned_question,
                model_answer=build_no_result_message(cleaned_question),
                sources=[],
                retrieved_count=0,
                used_prompt=used_prompt,
                retrieval_mode="hybrid_rag",
            )

        model_answer = synthesize_answer(cleaned_question, retrieved_chunks)
        sources = build_source_entries(retrieved_chunks)
        return AgentAnswer(
            question=cleaned_question,
            model_answer=model_answer,
            sources=sources,
            retrieved_count=len(retrieved_chunks),
            used_prompt=used_prompt,
            retrieval_mode="hybrid_rag",
        )

    def retrieve_chunks_for_question(
        self,
        question: str,
        chunk_records: list[dict[str, object]],
    ) -> list[RetrievedChunk]:
        """根据当前可用能力选择关键词检索或混合检索。

        输入：
            question: 用户问题文本。
            chunk_records: 当前知识库切块记录列表。
        输出：
            检索结果列表。
        异常：
            当问题为空时，抛出 ValueError。
        """

        if (
            self.embedding_index is not None
            and self.embedding_client is not None
            and self.embedding_model_name
            and self.embedding_dimensions > 0
        ):
            try:
                query_vector = self.embedding_client.embed_texts(
                    model=self.embedding_model_name,
                    texts=[question],
                    dimensions=self.embedding_dimensions,
                )[0]
                return retrieve_relevant_chunks_hybrid(
                    question=question,
                    chunk_records=chunk_records,
                    embedding_vectors=self.embedding_index.vectors,
                    query_vector=query_vector,
                    top_k=self.top_k,
                )
            except Exception:
                # 向量检索失败时回退关键词检索，保证问答流程不中断。
                pass

        return retrieve_relevant_chunks(
            question=question,
            chunk_records=chunk_records,
            top_k=self.top_k,
        )

    def prepare_embedding_index(
        self,
        *,
        client: DashScopeClient | None,
        model_name: str,
        dimensions: int,
        processed_data_dir: str | Path,
        build_if_missing: bool = False,
        force_rebuild: bool = False,
    ) -> EmbeddingIndexResponse:
        """加载或构建第四周的本地向量索引。

        输入：
            client: DashScope 客户端；构建索引时必需。
            model_name: 向量模型名称。
            dimensions: 向量维度。
            processed_data_dir: 索引目录。
            build_if_missing: 若索引缺失是否自动构建。
            force_rebuild: 是否忽略旧索引并强制重建。
        输出：
            向量索引响应对象。
        异常：
            当知识库为空时，抛出 ValueError。
            当需要构建但 client 缺失时，抛出 ValueError。
        """

        knowledge_base = self.ensure_knowledge_base_ready()
        if not knowledge_base.chunk_records:
            raise ValueError("当前没有可用于构建向量索引的切块记录。")

        index_path = get_default_embedding_index_path(processed_data_dir, model_name, dimensions)
        self.embedding_client = client
        self.embedding_model_name = model_name
        self.embedding_dimensions = dimensions

        if not force_rebuild:
            cached_index = load_embedding_index(index_path)
            if cached_index is not None and is_embedding_index_compatible(
                cached_index,
                knowledge_base.chunk_records,
                model_name,
                dimensions,
            ):
                self.embedding_index = cached_index
                return EmbeddingIndexResponse(
                    status_message="已加载本地向量索引。",
                    index_path=str(index_path),
                    vector_count=len(cached_index.vectors),
                    loaded_from_cache=True,
                )

        if not build_if_missing:
            self.embedding_index = None
            return EmbeddingIndexResponse(
                status_message="当前未找到可用向量索引，可执行 build_index 构建。",
                index_path=str(index_path),
                vector_count=0,
                loaded_from_cache=False,
            )

        if client is None:
            raise ValueError("构建向量索引需要可用的大模型客户端。")

        built_index = run_build_embedding_index(
            chunk_records=knowledge_base.chunk_records,
            client=client,
            model_name=model_name,
            dimensions=dimensions,
            index_path=index_path,
        )
        self.embedding_index = built_index
        return EmbeddingIndexResponse(
            status_message="已完成本地向量索引构建。",
            index_path=str(index_path),
            vector_count=len(built_index.vectors),
            loaded_from_cache=False,
        )

    def analyze_paper(self, target: str | None = None) -> PaperAnalysisResponse:
        """对指定论文执行单篇结构化学术分析。

        输入：
            target: 论文序号、文件名、文档编号或其片段；为空时默认分析第一篇。
        输出：
            包含结构化分析结果与展示文本的响应对象。
        异常：
            当自动建库失败时，抛出对应异常。
        """

        knowledge_base = self.ensure_knowledge_base_ready()
        if not knowledge_base.documents:
            return PaperAnalysisResponse(
                target=target or "默认第一篇论文",
                status_message="当前没有可分析的论文。请先放入 PDF 并确认其能够被成功解析。",
                analysis=None,
                formatted_output="当前没有可分析的论文。",
            )

        document = self.find_document(target)
        if document is None:
            return PaperAnalysisResponse(
                target=target or "默认第一篇论文",
                status_message="未找到匹配的论文，请先使用 papers 查看可用论文列表。",
                analysis=None,
                formatted_output="未找到匹配的论文，请先使用 papers 查看可用论文列表。",
            )

        analysis = analyze_single_paper(
            text_or_segments=document.full_text,
            file_name=document.file_name,
            document_id=document.document_id,
        )
        return PaperAnalysisResponse(
            target=target or document.file_name,
            status_message=f"已完成对《{document.file_name}》的结构化提取。",
            analysis=analysis,
            formatted_output=format_analysis_result(analysis),
        )

    def compare_papers(self, targets: list[str] | None = None, topic_hint: str = "未指定比较主题") -> PaperComparisonResponse:
        """对多篇论文执行最小结构化比较。

        输入：
            targets: 论文序号、文件名、文档编号或其片段列表；为空时默认取前两篇。
            topic_hint: 可选的比较主题说明。
        输出：
            多篇论文比较响应对象。
        异常：
            当自动建库失败时，抛出对应异常。
            当目标论文无法匹配时，抛出 ValueError。
        """

        knowledge_base = self.ensure_knowledge_base_ready()
        if len(knowledge_base.documents) < 2:
            return PaperComparisonResponse(
                targets=targets or [],
                status_message="当前可用论文数量不足 2 篇，暂时无法执行多篇比较。",
                comparison=None,
                formatted_output="当前可用论文数量不足 2 篇，暂时无法执行多篇比较。",
            )

        documents = self.find_documents(targets, default_count=2)
        if len(documents) < 2:
            return PaperComparisonResponse(
                targets=targets or [],
                status_message="多篇比较至少需要 2 篇论文，请重新指定目标。",
                comparison=None,
                formatted_output="多篇比较至少需要 2 篇论文，请重新指定目标。",
            )

        paper_inputs = [
            {
                "file_name": document.file_name,
                "document_id": document.document_id,
                "full_text": document.full_text,
            }
            for document in documents
        ]
        comparison = run_compare_papers(paper_inputs, topic_hint=topic_hint)
        return PaperComparisonResponse(
            targets=[document.file_name for document in documents],
            status_message=f"已完成 {len(documents)} 篇论文的最小比较。",
            comparison=comparison,
            formatted_output=format_comparison_result(comparison),
        )

    def generate_review_outline(
        self,
        topic: str,
        targets: list[str] | None = None,
    ) -> ReviewOutlineResponse:
        """根据若干论文生成最小综述提纲。

        输入：
            topic: 综述主题。
            targets: 论文序号、文件名、文档编号或其片段列表；为空时默认取前三篇。
        输出：
            综述提纲响应对象。
        异常：
            当自动建库失败时，抛出对应异常。
            当主题为空或目标论文无法匹配时，抛出 ValueError。
        """

        cleaned_topic = topic.strip()
        if not cleaned_topic:
            raise ValueError("综述主题不能为空。")

        knowledge_base = self.ensure_knowledge_base_ready()
        if not knowledge_base.documents:
            return ReviewOutlineResponse(
                topic=cleaned_topic,
                status_message="当前没有可用于生成综述提纲的论文。",
                outline=None,
                formatted_output="当前没有可用于生成综述提纲的论文。",
            )

        if targets:
            documents = self.find_documents(targets, default_count=3)
        else:
            documents = self.find_relevant_documents_for_topic(cleaned_topic, default_count=3)
        if len(documents) < 2:
            return ReviewOutlineResponse(
                topic=cleaned_topic,
                status_message="综述提纲生成至少需要 2 篇论文，请重新指定目标。",
                outline=None,
                formatted_output="综述提纲生成至少需要 2 篇论文，请重新指定目标。",
            )

        paper_inputs = [
            {
                "file_name": document.file_name,
                "document_id": document.document_id,
                "full_text": document.full_text,
            }
            for document in documents
        ]
        outline = generate_review_outline(cleaned_topic, paper_inputs)
        return ReviewOutlineResponse(
            topic=cleaned_topic,
            status_message=f"已基于 {len(documents)} 篇论文生成最小综述提纲。",
            outline=outline,
            formatted_output=format_review_outline(outline),
        )

    def run_review_workflow(
        self,
        topic: str,
        targets: list[str] | None = None,
        output_dir: str | Path | None = None,
    ) -> WorkflowResponse:
        """执行第三周的最小多步科研工作流。

        输入：
            topic: 工作流主题。
            targets: 目标论文列表；为空时默认取前三篇。
            output_dir: 导出目录；为空时使用项目默认输出目录。
        输出：
            工作流响应对象。
        异常：
            当主题为空或目标论文无法匹配时，抛出 ValueError。
            当自动建库失败时，抛出对应异常。
        """

        cleaned_topic = topic.strip()
        if not cleaned_topic:
            raise ValueError("工作流主题不能为空。")

        knowledge_base = self.ensure_knowledge_base_ready()
        if len(knowledge_base.documents) < 2:
            return WorkflowResponse(
                topic=cleaned_topic,
                status_message="当前可用论文不足 2 篇，暂时无法运行多步工作流。",
                workflow_result=None,
                formatted_output="当前可用论文不足 2 篇，暂时无法运行多步工作流。",
            )

        plan = build_default_workflow_plan(topic=cleaned_topic, targets=targets)
        state = WorkflowState(topic=cleaned_topic, targets=targets or [])

        documents = self.find_documents(targets, default_count=3)
        if len(documents) < 2:
            return WorkflowResponse(
                topic=cleaned_topic,
                status_message="多步工作流至少需要 2 篇论文，请重新指定目标。",
                workflow_result=None,
                formatted_output="多步工作流至少需要 2 篇论文，请重新指定目标。",
            )

        mark_step_status(plan, "select_papers", "completed")
        state.selected_papers = [document.file_name for document in documents]
        state.step_logs.append(f"步骤 1：已选中 {len(documents)} 篇论文。")

        graph_text, graph_log = self.retrieve_workflow_graph_text(cleaned_topic, documents)
        mark_step_status(plan, "retrieve_graphrag", "completed")
        state.graph_text = graph_text
        state.step_logs.append(graph_log)

        comparison_result = self.compare_papers(
            targets=[document.document_id for document in documents],
            topic_hint=cleaned_topic,
        )
        mark_step_status(plan, "compare_papers", "completed")
        state.comparison_text = comparison_result.formatted_output
        state.step_logs.append("步骤 3：已完成多篇论文比较。")

        outline_result = self.generate_review_outline(
            topic=cleaned_topic,
            targets=[document.document_id for document in documents],
        )
        mark_step_status(plan, "generate_outline", "completed")
        state.outline_text = outline_result.formatted_output
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
        )
        mark_step_status(plan, "export_markdown", "completed")
        state.export_artifact = artifact
        state.step_logs.append(f"步骤 5：已导出 Markdown 报告到 {artifact.output_path}")

        workflow_result = WorkflowRunResult(
            status_message=f"已完成主题“{cleaned_topic}”的最小多步工作流。",
            plan=plan,
            state=state,
            formatted_output="",
        )
        workflow_result.formatted_output = format_workflow_run_result(workflow_result)

        return WorkflowResponse(
            topic=cleaned_topic,
            status_message=workflow_result.status_message,
            workflow_result=workflow_result,
            formatted_output=workflow_result.formatted_output,
        )


globals()["City" + "ScholarAgent"] = HSPansharpeningScholarAgent


def build_workflow_graph_question(topic: str, documents: list[ParsedDocument]) -> str:
    """构造 workflow 专用 GraphRAG 检索问题。"""

    paper_names = "；".join(document.file_name for document in documents[:6])
    return (
        f"围绕“{topic}”，结合以下论文：{paper_names}，"
        "从图谱中归纳任务类型、输入模态、核心方法、空间-光谱关系、退化/先验机制、数据集、评价指标和对用户模型的启示。"
    )


def format_workflow_graph_context(graph_context: GraphRetrievedContext) -> str:
    """将 GraphRAG 检索上下文整理为 workflow 报告文本。"""

    lines = [
        "GraphRAG 辅助发现：",
        f"- 检索模式：{graph_context.retrieval_mode}",
        f"- 检索问题：{graph_context.question}",
    ]

    if graph_context.communities:
        lines.append("图谱主题簇：")
        for index, community in enumerate(graph_context.communities[:4], start=1):
            key_entities = "、".join(community.key_entities[:8]) if community.key_entities else "未列出"
            lines.append(f"- {index}. {community.title}：{truncate_text(community.summary, 220)}")
            lines.append(f"  关键实体：{key_entities}")

    if graph_context.relations:
        lines.append("关键关系：")
        seen_relations: set[str] = set()
        relation_count = 0
        for relation in graph_context.relations:
            relation_text = f"{relation.source_name} --{relation.relation_type}--> {relation.target_name}"
            if relation_text in seen_relations:
                continue
            seen_relations.add(relation_text)
            relation_count += 1
            lines.append(f"- {relation_text}；证据：{truncate_text(relation.evidence_text, 180)}")
            if relation_count >= 6:
                break

    if graph_context.chunks:
        lines.append("代表性图谱证据：")
        for index, chunk in enumerate(graph_context.chunks[:6], start=1):
            page_text = "、".join(str(number) for number in chunk.page_numbers) if chunk.page_numbers else "未知页码"
            lines.append(f"- {index}. 《{chunk.file_name}》页码 {page_text}：{truncate_text(chunk.snippet, 220)}")

    if graph_context.notes:
        lines.append("检索备注：")
        for note in graph_context.notes:
            lines.append(f"- {note}")

    return "\n".join(lines)


def split_text_into_sentences(text: str) -> list[str]:
    """将文本按中文与英文常见分隔符切分为句子。

    输入：
        text: 原始文本。
    输出：
        清洗后的句子列表。
    异常：
        无。
    """

    raw_sentences = re.split(r"[。！？!?；;\n]+", text)
    sentences: list[str] = []
    for sentence in raw_sentences:
        cleaned_sentence = re.sub(r"\s+", " ", sentence).strip()
        if cleaned_sentence:
            sentences.append(cleaned_sentence)
    return sentences


def truncate_text(text: str, max_length: int = 120) -> str:
    """将文本裁剪到适合回答展示的长度。

    输入：
        text: 原始文本。
        max_length: 最大保留字符数。
    输出：
        适合展示的短文本。
    异常：
        无。
    """

    cleaned_text = re.sub(r"\s+", " ", text).strip()
    if len(cleaned_text) <= max_length:
        return cleaned_text
    return cleaned_text[:max_length].rstrip() + "..."


def score_supporting_sentence(
    question: str,
    sentence: str,
    retrieval_score: float,
) -> float:
    """计算候选支撑句与问题的相关程度。

    输入：
        question: 用户问题文本。
        sentence: 候选句子文本。
        retrieval_score: 句子所属文本块的检索分数。
    输出：
        用于排序的句子分数。
    异常：
        无。
    """

    question_terms = extract_search_terms(question)
    sentence_text = sentence.lower()
    # 以检索分数为基础分，再叠加关键词命中与句长等启发式特征。
    score = retrieval_score * 0.5

    if question.strip().lower() in sentence_text:
        score += 6.0

    for term in question_terms:
        if term in sentence_text:
            if len(term) >= 6:
                score += 2.2
            elif len(term) >= 4:
                score += 1.7
            else:
                score += 1.0

    if len(sentence.strip()) < 8:
        score -= 1.0

    return score


def select_supporting_sentences(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    max_sentences: int = 3,
) -> list[str]:
    """从召回片段中选出最能支撑回答的句子。

    输入：
        question: 用户问题文本。
        retrieved_chunks: 已召回的文本块列表。
        max_sentences: 最多保留多少句。
    输出：
        适合直接拼装为回答的句子列表。
    异常：
        无。
    """

    sentence_candidates: list[tuple[float, str]] = []
    for chunk in retrieved_chunks:
        for sentence in split_text_into_sentences(chunk.text):
            sentence_score = score_supporting_sentence(question, sentence, chunk.score)
            if sentence_score <= 0:
                continue
            sentence_candidates.append((sentence_score, truncate_text(sentence)))

    # 先按分数降序，再按文本稳定排序，保证结果可复现。
    sentence_candidates.sort(key=lambda item: (-item[0], item[1]))

    selected_sentences: list[str] = []
    seen_sentences: set[str] = set()
    for _, sentence in sentence_candidates:
        if sentence in seen_sentences:
            continue
        selected_sentences.append(sentence)
        seen_sentences.add(sentence)
        if len(selected_sentences) >= max_sentences:
            break

    return selected_sentences


def build_context_blocks(retrieved_chunks: list[RetrievedChunk]) -> list[str]:
    """将召回结果转换为后续回答模块可使用的上下文块。

    输入：
        retrieved_chunks: 已召回的文本块列表。
    输出：
        带有论文名、页码和片段的文本块列表。
    异常：
        无。
    """

    context_blocks: list[str] = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        # 统一上下文模板，便于后续 LLM 增强时直接复用。
        page_numbers = chunk.metadata.get("page_numbers", [])
        page_text = "、".join(str(number) for number in page_numbers) if page_numbers else "未知页码"
        file_name = str(chunk.metadata.get("file_name", "未知论文"))
        context_blocks.append(
            f"[来源 {index}] 论文：{file_name} | 页码：{page_text} | 片段：{chunk.snippet}"
        )
    return context_blocks


def build_graph_context_blocks(graph_context: GraphRetrievedContext) -> list[str]:
    """将图谱检索结果转换为模型可读上下文块。"""

    context_blocks: list[str] = []
    for index, community in enumerate(graph_context.communities, start=1):
        context_blocks.append(
            f"[图谱社群 {index}] 标题：{community.title} | 摘要：{community.summary} | "
            f"关键实体：{'、'.join(community.key_entities)} | 关键关系：{'；'.join(community.key_relations[:3])}"
        )

    for index, relation in enumerate(graph_context.relations[:6], start=1):
        context_blocks.append(
            f"[图谱关系 {index}] {relation.source_name} --{relation.relation_type}--> {relation.target_name} | "
            f"证据：{relation.evidence_text}"
        )

    for index, chunk in enumerate(graph_context.chunks[:6], start=1):
        page_text = "、".join(str(number) for number in chunk.page_numbers) if chunk.page_numbers else "未知页码"
        context_blocks.append(
            f"[图谱证据 {index}] 论文：{chunk.file_name} | 页码：{page_text} | 片段：{chunk.snippet}"
        )
    return context_blocks


def build_unified_context_blocks(
    retrieved_chunks: list[RetrievedChunk] | None = None,
    graph_context: GraphRetrievedContext | None = None,
) -> list[str]:
    """将 graph retrieval 与原始 chunk retrieval 统一转换为上下文块。"""

    context_blocks: list[str] = []
    if graph_context is not None:
        context_blocks.extend(build_graph_context_blocks(graph_context))
    if retrieved_chunks:
        context_blocks.extend(build_context_blocks(retrieved_chunks))
    return context_blocks


def synthesize_answer(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    """基于召回片段生成最小回答文本。

    输入：
        question: 用户问题文本。
        retrieved_chunks: 已召回的文本块列表。
    输出：
        适合直接展示给用户的回答文本。
    异常：
        无。
    """

    supporting_sentences = select_supporting_sentences(question, retrieved_chunks)
    answer_lines = ["根据当前召回的论文片段，可以整理出以下回答："]

    if supporting_sentences:
        for index, sentence in enumerate(supporting_sentences, start=1):
            answer_lines.append(f"{index}. {sentence}")
    else:
        # 当抽取不到稳定句子时，回退展示高分片段，保证可解释性。
        answer_lines.append("当前能够召回相关片段，但还不足以抽取出稳定的支撑句。")
        for index, chunk in enumerate(retrieved_chunks[:2], start=1):
            answer_lines.append(f"{index}. {truncate_text(chunk.snippet)}")

    answer_lines.append(build_answer_suffix())
    return "\n".join(answer_lines)


def synthesize_graph_answer(question: str, graph_context: GraphRetrievedContext) -> str:
    """基于图谱检索上下文生成最小回答文本。"""

    answer_lines = [f"根据当前 {graph_context.retrieval_mode} 图谱检索结果，可以整理出以下回答："]

    if graph_context.communities:
        for index, community in enumerate(graph_context.communities[:2], start=1):
            answer_lines.append(
                f"{index}. 图谱主题簇视角：{community.title}。{community.summary}"
            )

    if graph_context.relations:
        relation_texts = [
            f"{item.source_name} --{item.relation_type}--> {item.target_name}"
            for item in graph_context.relations[:3]
        ]
        answer_lines.append(f"关键关系：{'；'.join(relation_texts)}。")

    if graph_context.entities and not graph_context.communities:
        entity_names = [item.name for item in graph_context.entities[:5]]
        answer_lines.append(f"关联实体：{'、'.join(entity_names)}。")

    if graph_context.chunks:
        answer_lines.append("代表性证据：")
        for index, chunk in enumerate(graph_context.chunks[:3], start=1):
            answer_lines.append(f"{index}. 《{chunk.file_name}》：{truncate_text(chunk.snippet)}")

    if len(answer_lines) == 1:
        answer_lines.append(build_no_result_message(question))
    else:
        answer_lines.append(build_answer_suffix())
    return "\n".join(answer_lines)


def build_source_entries(
    retrieved_chunks: list[RetrievedChunk],
    max_sources: int = 3,
) -> list[dict[str, object]]:
    """将召回结果转换为可展示的来源依据列表。

    输入：
        retrieved_chunks: 已召回的文本块列表。
        max_sources: 最多保留多少条来源信息。
    输出：
        适合界面或命令行展示的来源依据列表。
    异常：
        无。
    """

    sources: list[dict[str, object]] = []
    for chunk in retrieved_chunks[:max_sources]:
        sources.append(
            {
                "file_name": str(chunk.metadata.get("file_name", "未知论文")),
                "source_path": str(chunk.metadata.get("source_path", "")),
                "page_numbers": chunk.metadata.get("page_numbers", []),
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "score": round(chunk.score, 2),
                "snippet": chunk.snippet,
                "matched_terms": chunk.matched_terms,
            }
        )
    return sources


def build_graph_source_entries(
    graph_context: GraphRetrievedContext,
    max_sources: int = 6,
) -> list[dict[str, object]]:
    """将图谱检索结果转换为可展示的来源依据列表。"""

    sources: list[dict[str, object]] = []
    for community in graph_context.communities[:2]:
        sources.append(
            {
                "source_type": "community_summary",
                "file_name": f"图谱社群：{community.title}",
                "source_path": "",
                "page_numbers": [],
                "chunk_id": community.community_id,
                "chunk_index": 0,
                "score": round(community.score, 2),
                "snippet": community.summary,
                "matched_terms": community.key_entities,
            }
        )

    for relation in graph_context.relations[:2]:
        sources.append(
            {
                "source_type": "graph_relation",
                "file_name": f"图谱关系：{relation.source_name} -> {relation.target_name}",
                "source_path": "",
                "page_numbers": [],
                "chunk_id": relation.source_chunk_id or relation.relation_id,
                "chunk_index": 0,
                "score": round(relation.score, 2),
                "snippet": relation.evidence_text,
                "matched_terms": [relation.relation_type],
            }
        )

    for chunk in graph_context.chunks:
        sources.append(
            {
                "source_type": chunk.evidence_type,
                "file_name": chunk.file_name,
                "source_path": chunk.source_path,
                "page_numbers": chunk.page_numbers,
                "chunk_id": chunk.chunk_id,
                "chunk_index": 0,
                "score": round(chunk.score, 2),
                "snippet": chunk.snippet,
                "matched_terms": chunk.matched_terms,
            }
        )
        if len(sources) >= max_sources:
            break
    return sources[:max_sources]


def run_agent_demo() -> None:
    """执行 agent 模块的最小演示。

    输入：
        无。
    输出：
        无。函数会直接打印问答与分析示例。
    异常：
        无。
    """

    demo_agent = HSPansharpeningScholarAgent(raw_papers_dir="data/raw_papers")
    demo_agent.knowledge_base = KnowledgeBaseState(
        raw_papers_dir="data/raw_papers",
        pdf_files=["data/raw_papers/demo_paper.pdf"],
        documents=[
            ParsedDocument(
                document_id="demo_paper",
                file_name="demo_paper.pdf",
                source_path="data/raw_papers/demo_paper.pdf",
                total_pages=2,
                total_characters=88,
                full_text=(
                    "本文关注高光谱全色锐化任务，输入为低空间分辨率高光谱图像和高空间分辨率 PAN 图像，目标是重建高空间分辨率高光谱图像。"
                    "方法采用空间注意力引导 PAN 纹理注入，并通过光谱保持损失约束重建结果。"
                    "实验在 Pavia Center、CAVE 和 Harvard 数据集上验证，指标包括 PSNR、SSIM、SAM 和 ERGAS。"
                    "结果表明该方法在保持光谱一致性的同时提升空间细节。"
                    "局限在于真实全分辨率场景下的退化不匹配和计算复杂度仍需进一步评估。"
                    "该研究可为 PAN/LRHS 条件门控、latent diffusion 先验和空间-光谱消融设计提供启示。"
                ),
                pages=[],
            )
        ],
        chunk_records=[
            {
                "chunk_id": "demo_chunk_0001",
                "document_id": "demo_paper",
                "text": "高光谱全色锐化方法通过 PAN 引导空间细节注入，同时用光谱保持损失约束 HSI 光谱一致性。",
                "metadata": {
                    "file_name": "demo_paper.pdf",
                    "source_path": "data/raw_papers/demo_paper.pdf",
                    "chunk_index": 0,
                    "page_numbers": [1],
                },
            },
            {
                "chunk_id": "demo_chunk_0002",
                "document_id": "demo_paper",
                "text": "实验在 Pavia Center、CAVE 和 Harvard 数据集上验证，指标包括 PSNR、SSIM、SAM 和 ERGAS。",
                "metadata": {
                    "file_name": "demo_paper.pdf",
                    "source_path": "data/raw_papers/demo_paper.pdf",
                    "chunk_index": 1,
                    "page_numbers": [2],
                },
            },
        ],
        parse_errors=[],
    )

    answer_result = demo_agent.answer("高光谱全色锐化方法如何同时增强空间细节并保持光谱一致性？")
    print("Agent Demo - 问答")
    print("模型回答：")
    print(answer_result.model_answer)
    print("来源依据：")
    for index, source in enumerate(answer_result.sources, start=1):
        print(f"{index}. {source['file_name']} | 页码：{source['page_numbers']} | 片段：{source['snippet']}")

    analysis_result = demo_agent.analyze_paper("1")
    print("\nAgent Demo - 结构化分析")
    print(analysis_result.formatted_output)


if __name__ == "__main__":
    run_agent_demo()
