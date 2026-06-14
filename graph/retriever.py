"""本模块作用：提供 local/global/mixed 三类 GraphRAG 检索，并返回结构化上下文对象。"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

from graph.normalizer import normalize_entity_alias
from graph.store import GraphChunk, GraphCommunitySummary, GraphIndex
from llm_dashscope import DashScopeClient
from rag.retriever import (
    RetrievedChunk,
    cosine_similarity,
    extract_search_terms,
    normalize_search_text,
    retrieve_relevant_chunks,
    retrieve_relevant_chunks_hybrid,
)


@dataclass
class GraphRetrievedEntity:
    """本数据结构作用：保存图谱检索命中的实体信息。"""

    entity_id: str
    name: str
    entity_type: str
    description: str
    aliases: list[str]
    source_chunk_ids: list[str]
    source_paper_ids: list[str]
    score: float
    hop_distance: int


@dataclass
class GraphRetrievedRelation:
    """本数据结构作用：保存图谱检索命中的关系信息。"""

    relation_id: str
    source_entity_id: str
    target_entity_id: str
    source_name: str
    target_name: str
    relation_type: str
    evidence_text: str
    source_chunk_id: str
    source_paper_id: str
    score: float


@dataclass
class GraphRetrievedCommunity:
    """本数据结构作用：保存图谱检索命中的图谱社群摘要信息。"""

    community_id: str
    title: str
    summary: str
    key_entities: list[str]
    key_relations: list[str]
    representative_chunks: list[str]
    score: float


@dataclass
class GraphRetrievedChunk:
    """本数据结构作用：保存图谱检索阶段引用的 chunk 证据信息。"""

    chunk_id: str
    file_name: str
    source_path: str
    page_numbers: list[int]
    snippet: str
    text: str
    score: float
    evidence_type: str
    matched_terms: list[str] = field(default_factory=list)


@dataclass
class GraphRetrievedPaper:
    """本数据结构作用：保存图谱检索命中的论文信息。"""

    paper_id: str
    file_name: str
    source_path: str
    score: float


@dataclass
class GraphRetrievedContext:
    """本数据结构作用：保存一次图谱检索的统一上下文结果。"""

    question: str
    retrieval_mode: str
    seed_entity_ids: list[str] = field(default_factory=list)
    entities: list[GraphRetrievedEntity] = field(default_factory=list)
    relations: list[GraphRetrievedRelation] = field(default_factory=list)
    communities: list[GraphRetrievedCommunity] = field(default_factory=list)
    chunks: list[GraphRetrievedChunk] = field(default_factory=list)
    papers: list[GraphRetrievedPaper] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def truncate_text(text: str, max_chars: int = 180) -> str:
    """将文本裁剪到适合 CLI 展示的长度。"""

    cleaned_text = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned_text) <= max_chars:
        return cleaned_text
    return cleaned_text[:max_chars].rstrip() + "..."


def score_entity_match(question: str, entity_name: str, aliases: list[str], normalized_name: str) -> float:
    """根据问题文本为实体匹配打分。"""

    normalized_question = normalize_search_text(question)
    query_terms = extract_search_terms(question)
    entity_aliases = [entity_name, normalized_name, *aliases]
    score = 0.0

    for alias in entity_aliases:
        normalized_alias = normalize_entity_alias(alias)
        if not normalized_alias:
            continue
        if normalized_alias == normalized_question:
            score += 8.0
        if normalized_alias in normalized_question:
            score += 5.0 + min(len(normalized_alias) / 10.0, 2.5)
        if normalized_question and normalized_question in normalized_alias:
            score += 2.5
        for term in query_terms:
            if not term:
                continue
            if term == normalized_alias:
                score += 3.0
            elif term in normalized_alias or normalized_alias in term:
                score += 1.5

    return score


def chunk_to_retrieved_chunk(
    chunk: GraphChunk,
    score: float,
    evidence_type: str,
    matched_terms: list[str] | None = None,
) -> GraphRetrievedChunk:
    """将图谱 chunk 节点转换为检索证据对象。"""

    return GraphRetrievedChunk(
        chunk_id=chunk.chunk_id,
        file_name=chunk.file_name,
        source_path=chunk.source_path,
        page_numbers=chunk.page_numbers,
        snippet=truncate_text(chunk.text),
        text=chunk.text,
        score=score,
        evidence_type=evidence_type,
        matched_terms=matched_terms or [],
    )


def retrieved_chunk_to_graph_chunk(chunk: RetrievedChunk, evidence_type: str = "hybrid_chunk") -> GraphRetrievedChunk:
    """将原有 RAG 检索结果转换为图谱统一 chunk 证据对象。"""

    return GraphRetrievedChunk(
        chunk_id=chunk.chunk_id,
        file_name=str(chunk.metadata.get("file_name", "")),
        source_path=str(chunk.metadata.get("source_path", "")),
        page_numbers=[int(item) for item in chunk.metadata.get("page_numbers", [])],
        snippet=chunk.snippet,
        text=chunk.text,
        score=chunk.score,
        evidence_type=evidence_type,
        matched_terms=chunk.matched_terms,
    )


def build_entity_relation_adjacency(index: GraphIndex) -> dict[str, list[str]]:
    """根据关系边构建实体邻接表。"""

    adjacency: dict[str, list[str]] = {entity_id: [] for entity_id in index.entities}
    for relation in index.relations.values():
        if relation.source in adjacency:
            adjacency[relation.source].append(relation.target)
        if relation.target in adjacency:
            adjacency[relation.target].append(relation.source)
    return adjacency


def retrieve_graph_local(
    *,
    question: str,
    graph_index: GraphIndex,
    max_hops: int = 2,
    top_k_entities: int = 5,
    top_k_chunks: int = 5,
) -> GraphRetrievedContext:
    """执行局部实体关系型图谱检索。"""

    context = GraphRetrievedContext(question=question, retrieval_mode="graph_local")
    if not graph_index.entities:
        context.notes.append("当前图谱中没有实体节点。")
        return context

    entity_scores: list[tuple[float, str]] = []
    for entity in graph_index.entities.values():
        score = score_entity_match(
            question=question,
            entity_name=entity.name,
            aliases=entity.aliases,
            normalized_name=entity.normalized_name,
        )
        if score <= 0:
            continue
        entity_scores.append((score, entity.entity_id))

    entity_scores.sort(key=lambda item: (-item[0], item[1]))
    seed_items = entity_scores[:top_k_entities]
    if not seed_items:
        context.notes.append("未在图谱中匹配到足够稳定的种子实体。")
        return context

    context.seed_entity_ids = [entity_id for _, entity_id in seed_items]
    adjacency = build_entity_relation_adjacency(graph_index)
    entity_state: dict[str, tuple[float, int]] = {}
    bfs_queue: deque[tuple[str, int, float]] = deque()
    for score, entity_id in seed_items:
        entity_state[entity_id] = (score, 0)
        bfs_queue.append((entity_id, 0, score))

    while bfs_queue:
        current_entity_id, current_hop, current_score = bfs_queue.popleft()
        if current_hop >= max_hops:
            continue
        for neighbor_id in adjacency.get(current_entity_id, []):
            neighbor_score = current_score * 0.72
            neighbor_hop = current_hop + 1
            previous_state = entity_state.get(neighbor_id)
            if previous_state is not None and previous_state[0] >= neighbor_score:
                continue
            entity_state[neighbor_id] = (neighbor_score, neighbor_hop)
            bfs_queue.append((neighbor_id, neighbor_hop, neighbor_score))

    ranked_entity_items = sorted(
        entity_state.items(),
        key=lambda item: (-item[1][0], item[1][1], item[0]),
    )
    selected_entity_ids = [entity_id for entity_id, _ in ranked_entity_items[: max(top_k_entities * 2, 8)]]
    selected_entity_set = set(selected_entity_ids)

    chunk_score_map: dict[str, float] = {}
    paper_score_map: dict[str, float] = {}
    for entity_id, (score, hop_distance) in ranked_entity_items:
        if entity_id not in selected_entity_set:
            continue
        entity = graph_index.entities.get(entity_id)
        if entity is None:
            continue
        context.entities.append(
            GraphRetrievedEntity(
                entity_id=entity.entity_id,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                aliases=entity.aliases,
                source_chunk_ids=entity.source_chunk_ids,
                source_paper_ids=entity.source_paper_ids,
                score=score,
                hop_distance=hop_distance,
            )
        )
        for chunk_id in entity.source_chunk_ids:
            chunk_score_map[chunk_id] = max(chunk_score_map.get(chunk_id, 0.0), score)
        for paper_id in entity.source_paper_ids:
            paper_score_map[paper_id] = max(paper_score_map.get(paper_id, 0.0), score)

    for relation in graph_index.relations.values():
        if relation.source not in selected_entity_set and relation.target not in selected_entity_set:
            continue
        source_entity = graph_index.entities.get(relation.source)
        target_entity = graph_index.entities.get(relation.target)
        if source_entity is None or target_entity is None:
            continue
        relation_score = max(
            entity_state.get(relation.source, (0.0, 99))[0],
            entity_state.get(relation.target, (0.0, 99))[0],
        )
        context.relations.append(
            GraphRetrievedRelation(
                relation_id=relation.relation_id,
                source_entity_id=relation.source,
                target_entity_id=relation.target,
                source_name=source_entity.name,
                target_name=target_entity.name,
                relation_type=relation.relation_type,
                evidence_text=relation.evidence_text,
                source_chunk_id=relation.source_chunk_id,
                source_paper_id=relation.source_paper_id,
                score=relation_score,
            )
        )
        if relation.source_chunk_id:
            chunk_score_map[relation.source_chunk_id] = max(chunk_score_map.get(relation.source_chunk_id, 0.0), relation_score + 0.4)
        if relation.source_paper_id:
            paper_score_map[relation.source_paper_id] = max(paper_score_map.get(relation.source_paper_id, 0.0), relation_score + 0.2)

    ranked_chunk_items = sorted(chunk_score_map.items(), key=lambda item: (-item[1], item[0]))
    for chunk_id, score in ranked_chunk_items[:top_k_chunks]:
        chunk = graph_index.chunks.get(chunk_id)
        if chunk is None:
            continue
        context.chunks.append(chunk_to_retrieved_chunk(chunk, score=score, evidence_type="graph_local"))

    community_scores: dict[str, float] = {}
    for community_id, community in graph_index.communities.items():
        overlap_entity_ids = selected_entity_set & set(community.entity_ids)
        if not overlap_entity_ids:
            continue
        community_score = sum(entity_state.get(entity_id, (0.0, 99))[0] for entity_id in overlap_entity_ids)
        community_scores[community_id] = community_score

    for community_id, score in sorted(community_scores.items(), key=lambda item: (-item[1], item[0]))[:3]:
        summary = graph_index.community_summaries.get(community_id)
        if summary is None:
            continue
        context.communities.append(
            GraphRetrievedCommunity(
                community_id=summary.community_id,
                title=summary.title,
                summary=summary.summary,
                key_entities=summary.key_entities,
                key_relations=summary.key_relations,
                representative_chunks=summary.representative_chunks,
                score=score,
            )
        )

    for paper_id, score in sorted(paper_score_map.items(), key=lambda item: (-item[1], item[0]))[:5]:
        paper = graph_index.papers.get(paper_id)
        if paper is None:
            continue
        context.papers.append(
            GraphRetrievedPaper(
                paper_id=paper.paper_id,
                file_name=paper.file_name,
                source_path=paper.source_path,
                score=score,
            )
        )

    if not context.chunks:
        context.notes.append("图谱局部检索命中了实体，但没有找到足够稳定的 chunk 证据。")
    return context


def score_community_summary_lexical(question: str, summary: GraphCommunitySummary) -> float:
    """根据问题与图谱社群摘要文本的词项重叠打分。"""

    combined_text = normalize_search_text(
        " ".join(
            [
                summary.title,
                summary.summary,
                " ".join(summary.key_entities),
                " ".join(summary.key_relations),
            ]
        )
    )
    score = 0.0
    for term in extract_search_terms(question):
        if term and term in combined_text:
            score += 1.0 + min(len(term) / 6.0, 2.0)
    return score


def build_query_embedding(
    question: str,
    client: DashScopeClient | None,
    embedding_model: str,
    embedding_dimensions: int,
) -> list[float]:
    """在可用时构建查询向量。"""

    if client is None or not embedding_model.strip() or embedding_dimensions <= 0:
        return []
    try:
        return client.embed_texts(
            model=embedding_model,
            texts=[question],
            dimensions=embedding_dimensions,
        )[0]
    except Exception:
        return []


def retrieve_graph_global(
    *,
    question: str,
    graph_index: GraphIndex,
    top_k_communities: int = 4,
    client: DashScopeClient | None = None,
    embedding_model: str = "",
    embedding_dimensions: int = 0,
) -> GraphRetrievedContext:
    """执行全局综述型图谱检索。"""

    context = GraphRetrievedContext(question=question, retrieval_mode="graph_global")
    if not graph_index.community_summaries:
        context.notes.append("当前图谱中没有可用的图谱社群摘要。")
        return context

    query_vector = build_query_embedding(
        question=question,
        client=client,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )
    community_scores: list[tuple[float, str]] = []
    for community_id, summary in graph_index.community_summaries.items():
        lexical_score = score_community_summary_lexical(question, summary)
        semantic_score = 0.0
        if query_vector and summary.embedding_vector:
            semantic_score = cosine_similarity(query_vector, summary.embedding_vector)
        if lexical_score <= 0 and semantic_score <= 0:
            continue
        final_score = min(lexical_score / 10.0, 1.0) * 0.55 + semantic_score * 0.45
        community_scores.append((final_score, community_id))

    community_scores.sort(key=lambda item: (-item[0], item[1]))
    if not community_scores and graph_index.community_summaries:
        fallback_community_ids = sorted(graph_index.communities, key=lambda item: item)[:top_k_communities]
        community_scores = [(0.1, community_id) for community_id in fallback_community_ids]

    chunk_score_map: dict[str, float] = {}
    paper_score_map: dict[str, float] = {}
    entity_score_map: dict[str, float] = {}
    for score, community_id in community_scores[:top_k_communities]:
        summary = graph_index.community_summaries.get(community_id)
        community = graph_index.communities.get(community_id)
        if summary is None or community is None:
            continue
        context.communities.append(
            GraphRetrievedCommunity(
                community_id=summary.community_id,
                title=summary.title,
                summary=summary.summary,
                key_entities=summary.key_entities,
                key_relations=summary.key_relations,
                representative_chunks=summary.representative_chunks,
                score=score,
            )
        )
        for chunk_id in summary.representative_chunks or community.chunk_ids[:4]:
            chunk_score_map[chunk_id] = max(chunk_score_map.get(chunk_id, 0.0), score)
        for paper_id in community.paper_ids:
            paper_score_map[paper_id] = max(paper_score_map.get(paper_id, 0.0), score)
        for entity_id in community.entity_ids[:8]:
            entity_score_map[entity_id] = max(entity_score_map.get(entity_id, 0.0), score)

    for entity_id, score in sorted(entity_score_map.items(), key=lambda item: (-item[1], item[0]))[:10]:
        entity = graph_index.entities.get(entity_id)
        if entity is None:
            continue
        context.entities.append(
            GraphRetrievedEntity(
                entity_id=entity.entity_id,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                aliases=entity.aliases,
                source_chunk_ids=entity.source_chunk_ids,
                source_paper_ids=entity.source_paper_ids,
                score=score,
                hop_distance=0,
            )
        )

    for chunk_id, score in sorted(chunk_score_map.items(), key=lambda item: (-item[1], item[0]))[:8]:
        chunk = graph_index.chunks.get(chunk_id)
        if chunk is None:
            continue
        context.chunks.append(chunk_to_retrieved_chunk(chunk, score=score, evidence_type="graph_global"))

    for paper_id, score in sorted(paper_score_map.items(), key=lambda item: (-item[1], item[0]))[:6]:
        paper = graph_index.papers.get(paper_id)
        if paper is None:
            continue
        context.papers.append(
            GraphRetrievedPaper(
                paper_id=paper.paper_id,
                file_name=paper.file_name,
                source_path=paper.source_path,
                score=score,
            )
        )
    return context


def retrieve_support_chunks(
    *,
    question: str,
    chunk_records: list[dict[str, object]],
    client: DashScopeClient | None,
    embedding_model: str,
    embedding_dimensions: int,
    embedding_vectors: dict[str, list[float]] | None,
    top_k: int,
) -> list[GraphRetrievedChunk]:
    """按现有 Hybrid RAG 逻辑补充原始 chunk 证据。"""

    if not chunk_records:
        return []

    if client is not None and embedding_vectors and embedding_model.strip() and embedding_dimensions > 0:
        try:
            query_vector = client.embed_texts(
                model=embedding_model,
                texts=[question],
                dimensions=embedding_dimensions,
            )[0]
            chunk_results = retrieve_relevant_chunks_hybrid(
                question=question,
                chunk_records=chunk_records,
                embedding_vectors=embedding_vectors,
                query_vector=query_vector,
                top_k=top_k,
            )
            return [retrieved_chunk_to_graph_chunk(item) for item in chunk_results]
        except Exception:
            pass

    chunk_results = retrieve_relevant_chunks(
        question=question,
        chunk_records=chunk_records,
        top_k=top_k,
    )
    return [retrieved_chunk_to_graph_chunk(item) for item in chunk_results]


def merge_graph_contexts(
    question: str,
    retrieval_mode: str,
    contexts: list[GraphRetrievedContext],
) -> GraphRetrievedContext:
    """将多个图谱检索上下文合并为一个统一结果。"""

    merged_context = GraphRetrievedContext(question=question, retrieval_mode=retrieval_mode)
    merged_context.seed_entity_ids = sorted(
        {
            entity_id
            for context in contexts
            for entity_id in context.seed_entity_ids
            if entity_id
        }
    )

    entity_map: dict[str, GraphRetrievedEntity] = {}
    relation_map: dict[str, GraphRetrievedRelation] = {}
    community_map: dict[str, GraphRetrievedCommunity] = {}
    chunk_map: dict[str, GraphRetrievedChunk] = {}
    paper_map: dict[str, GraphRetrievedPaper] = {}

    for context in contexts:
        merged_context.notes.extend(context.notes)

        for entity in context.entities:
            existing = entity_map.get(entity.entity_id)
            if existing is None or existing.score < entity.score:
                entity_map[entity.entity_id] = entity

        for relation in context.relations:
            existing = relation_map.get(relation.relation_id)
            if existing is None or existing.score < relation.score:
                relation_map[relation.relation_id] = relation

        for community in context.communities:
            existing = community_map.get(community.community_id)
            if existing is None or existing.score < community.score:
                community_map[community.community_id] = community

        for chunk in context.chunks:
            existing = chunk_map.get(chunk.chunk_id)
            if existing is None or existing.score < chunk.score:
                chunk_map[chunk.chunk_id] = chunk

        for paper in context.papers:
            existing = paper_map.get(paper.paper_id)
            if existing is None or existing.score < paper.score:
                paper_map[paper.paper_id] = paper

    merged_context.entities = sorted(entity_map.values(), key=lambda item: (-item.score, item.hop_distance, item.name))
    merged_context.relations = sorted(relation_map.values(), key=lambda item: (-item.score, item.relation_type, item.relation_id))
    merged_context.communities = sorted(community_map.values(), key=lambda item: (-item.score, item.community_id))
    merged_context.chunks = sorted(chunk_map.values(), key=lambda item: (-item.score, item.chunk_id))
    merged_context.papers = sorted(paper_map.values(), key=lambda item: (-item.score, item.paper_id))
    merged_context.notes = [note for index, note in enumerate(merged_context.notes) if note and note not in merged_context.notes[:index]]
    return merged_context


def retrieve_graph_mixed(
    *,
    question: str,
    graph_index: GraphIndex,
    chunk_records: list[dict[str, object]],
    max_hops: int = 2,
    top_k_entities: int = 5,
    top_k_chunks: int = 5,
    top_k_communities: int = 4,
    client: DashScopeClient | None = None,
    embedding_model: str = "",
    embedding_dimensions: int = 0,
    embedding_vectors: dict[str, list[float]] | None = None,
) -> GraphRetrievedContext:
    """执行混合图谱检索：先 global，再 local，最后补充原始 chunk。"""

    global_context = retrieve_graph_global(
        question=question,
        graph_index=graph_index,
        top_k_communities=top_k_communities,
        client=client,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )
    local_context = retrieve_graph_local(
        question=question,
        graph_index=graph_index,
        max_hops=max_hops,
        top_k_entities=top_k_entities,
        top_k_chunks=top_k_chunks,
    )
    support_chunks = retrieve_support_chunks(
        question=question,
        chunk_records=chunk_records,
        client=client,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        embedding_vectors=embedding_vectors,
        top_k=top_k_chunks,
    )
    support_context = GraphRetrievedContext(
        question=question,
        retrieval_mode="hybrid_rag",
        chunks=support_chunks,
    )
    return merge_graph_contexts(
        question=question,
        retrieval_mode="graph_mixed",
        contexts=[global_context, local_context, support_context],
    )
