"""本模块作用：将现有 chunk_records 增量构建为可持久化的本地 GraphRAG 索引。"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from graph.community import build_graph_communities
from graph.extractor import build_chunk_text_hash, extract_chunk_graph_data
from graph.normalizer import EntityNormalizationResult, normalize_extracted_entities
from graph.store import (
    GRAPH_INDEX_VERSION,
    ChunkExtractionEntity,
    ChunkExtractionRecord,
    GraphChunk,
    GraphEntity,
    GraphIndex,
    GraphPaper,
    GraphRelation,
    build_graph_source_fingerprint,
)
from graph.summarizer import summarize_graph_communities
from llm_dashscope import DashScopeClient
from rag.parser import ParsedDocument


def build_relation_id(
    source_entity_id: str,
    target_entity_id: str,
    relation_type: str,
    source_chunk_id: str,
    evidence_text: str,
) -> str:
    """根据关系关键字段生成稳定关系编号。"""

    digest_source = "|".join(
        [
            source_entity_id,
            target_entity_id,
            relation_type,
            source_chunk_id,
            evidence_text,
        ]
    )
    return f"relation_{hashlib.sha1(digest_source.encode('utf-8', errors='surrogatepass')).hexdigest()[:12]}"


def build_paper_nodes(
    documents: list[ParsedDocument] | None,
    chunk_records: list[dict[str, object]],
) -> dict[str, GraphPaper]:
    """从文档元信息与 chunk_records 构建论文节点。"""

    papers: dict[str, GraphPaper] = {}

    for document in documents or []:
        papers[document.document_id] = GraphPaper(
            paper_id=document.document_id,
            document_id=document.document_id,
            file_name=document.file_name,
            source_path=document.source_path,
            metadata={
                "total_pages": document.total_pages,
                "total_characters": document.total_characters,
            },
        )

    for chunk_record in chunk_records:
        document_id = str(chunk_record.get("document_id", "")).strip()
        metadata = chunk_record.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        if document_id not in papers:
            papers[document_id] = GraphPaper(
                paper_id=document_id,
                document_id=document_id,
                file_name=str(metadata.get("file_name", "")),
                source_path=str(metadata.get("source_path", "")),
                metadata={},
            )
    return papers


def build_chunk_nodes(
    chunk_records: list[dict[str, object]],
    papers: dict[str, GraphPaper],
) -> dict[str, GraphChunk]:
    """从现有 chunk_records 构建 chunk 节点。"""

    chunks: dict[str, GraphChunk] = {}
    for chunk_record in chunk_records:
        metadata = chunk_record.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        chunk_id = str(chunk_record.get("chunk_id", ""))
        document_id = str(chunk_record.get("document_id", ""))
        text = str(chunk_record.get("text", ""))
        chunk_node = GraphChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            paper_id=document_id,
            file_name=str(metadata.get("file_name", "")),
            source_path=str(metadata.get("source_path", "")),
            text=text,
            text_hash=build_chunk_text_hash(text),
            chunk_index=int(metadata.get("chunk_index", 0)),
            page_numbers=[int(item) for item in metadata.get("page_numbers", [])],
        )
        chunks[chunk_id] = chunk_node
        if document_id in papers:
            papers[document_id].chunk_ids.append(chunk_id)
    return chunks


def collect_raw_entities(chunk_extractions: dict[str, ChunkExtractionRecord]) -> list[ChunkExtractionEntity]:
    """汇总所有 chunk 的原始实体抽取结果。"""

    raw_entities: list[ChunkExtractionEntity] = []
    for extraction in chunk_extractions.values():
        raw_entities.extend(extraction.entities)
    return raw_entities


def resolve_relation_target(
    raw_reference: str,
    normalization_result: EntityNormalizationResult,
) -> str:
    """将原始实体引用解析为规范实体编号。"""

    if raw_reference in normalization_result.raw_to_canonical:
        return normalization_result.raw_to_canonical[raw_reference]
    return normalization_result.normalized_name_map.get(raw_reference, "")


def build_relation_nodes(
    *,
    chunk_extractions: dict[str, ChunkExtractionRecord],
    normalization_result: EntityNormalizationResult,
    chunk_to_paper_map: dict[str, str],
    chunks: dict[str, GraphChunk],
) -> dict[str, GraphRelation]:
    """基于归一化结果构建规范关系边。"""

    relations: dict[str, GraphRelation] = {}
    seen_relation_keys: set[tuple[str, str, str, str]] = set()

    for extraction in chunk_extractions.values():
        for raw_relation in extraction.relations:
            source_entity_id = resolve_relation_target(raw_relation.source, normalization_result)
            target_entity_id = resolve_relation_target(raw_relation.target, normalization_result)
            if not source_entity_id or not target_entity_id or source_entity_id == target_entity_id:
                continue

            relation_key = (
                source_entity_id,
                target_entity_id,
                raw_relation.relation_type,
                raw_relation.source_chunk_id,
            )
            if relation_key in seen_relation_keys:
                continue
            seen_relation_keys.add(relation_key)

            relation_id = build_relation_id(
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
                relation_type=raw_relation.relation_type,
                source_chunk_id=raw_relation.source_chunk_id,
                evidence_text=raw_relation.evidence_text,
            )
            relation = GraphRelation(
                relation_id=relation_id,
                source=source_entity_id,
                target=target_entity_id,
                relation_type=raw_relation.relation_type,
                evidence_text=raw_relation.evidence_text,
                source_chunk_id=raw_relation.source_chunk_id,
                source_paper_id=chunk_to_paper_map.get(raw_relation.source_chunk_id, ""),
                weight=1.0,
            )
            relations[relation_id] = relation
            if raw_relation.source_chunk_id in chunks:
                chunks[raw_relation.source_chunk_id].relation_ids.append(relation_id)
    return relations


def attach_entities_to_chunks(
    chunks: dict[str, GraphChunk],
    chunk_extractions: dict[str, ChunkExtractionRecord],
    normalization_result: EntityNormalizationResult,
) -> None:
    """将规范实体链接回 chunk 节点。"""

    for extraction in chunk_extractions.values():
        chunk = chunks.get(extraction.chunk_id)
        if chunk is None:
            continue
        chunk.keywords = extraction.keywords
        chunk.claim_texts = extraction.claims
        linked_entity_ids = {
            normalization_result.raw_to_canonical.get(item.entity_id, "")
            for item in extraction.entities
        }
        chunk.entity_ids = sorted({item for item in linked_entity_ids if item})
        chunk.relation_ids = sorted(set(chunk.relation_ids))


def build_incremental_chunk_extractions(
    *,
    chunks: dict[str, GraphChunk],
    client: DashScopeClient | None,
    extraction_model: str,
    existing_index: GraphIndex | None,
) -> tuple[dict[str, ChunkExtractionRecord], dict[str, int], list[str]]:
    """增量构建或复用 chunk 级抽取缓存。"""

    existing_extractions = existing_index.chunk_extractions if existing_index is not None else {}
    chunk_extractions: dict[str, ChunkExtractionRecord] = {}
    stats = {
        "reused_chunk_extractions": 0,
        "new_chunk_extractions": 0,
        "failed_chunk_extractions": 0,
    }
    warnings: list[str] = []

    for chunk in sorted(chunks.values(), key=lambda item: item.chunk_id):
        cached_extraction = existing_extractions.get(chunk.chunk_id)
        if cached_extraction is not None and cached_extraction.text_hash == chunk.text_hash:
            chunk_extractions[chunk.chunk_id] = cached_extraction
            stats["reused_chunk_extractions"] += 1
            continue

        extraction = extract_chunk_graph_data(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            client=client,
            model_name=extraction_model,
        )
        chunk_extractions[chunk.chunk_id] = extraction
        stats["new_chunk_extractions"] += 1
        if extraction.error_message:
            stats["failed_chunk_extractions"] += 1
            warnings.append(f"{chunk.chunk_id}: {extraction.error_message}")

    return chunk_extractions, stats, warnings


def build_graph_index(
    *,
    chunk_records: list[dict[str, object]],
    documents: list[ParsedDocument] | None = None,
    client: DashScopeClient | None = None,
    extraction_model: str = "",
    summary_model: str = "",
    embedding_model: str = "",
    embedding_dimensions: int = 0,
    existing_index: GraphIndex | None = None,
) -> GraphIndex:
    """根据当前知识库切块构建 GraphRAG 索引。"""

    papers = build_paper_nodes(documents, chunk_records)
    chunks = build_chunk_nodes(chunk_records, papers)
    chunk_to_paper_map = {chunk_id: chunk.paper_id for chunk_id, chunk in chunks.items()}

    chunk_extractions, extraction_stats, extraction_warnings = build_incremental_chunk_extractions(
        chunks=chunks,
        client=client,
        extraction_model=extraction_model,
        existing_index=existing_index,
    )

    raw_entities = collect_raw_entities(chunk_extractions)
    normalization_result = normalize_extracted_entities(raw_entities, chunk_to_paper_map)
    attach_entities_to_chunks(chunks, chunk_extractions, normalization_result)
    relations = build_relation_nodes(
        chunk_extractions=chunk_extractions,
        normalization_result=normalization_result,
        chunk_to_paper_map=chunk_to_paper_map,
        chunks=chunks,
    )

    graph_index = GraphIndex(
        graph_version=GRAPH_INDEX_VERSION,
        created_at=datetime.now().isoformat(timespec="seconds"),
        source_fingerprint=build_graph_source_fingerprint(chunk_records),
        papers={paper_id: paper for paper_id, paper in papers.items()},
        chunks={chunk_id: chunk for chunk_id, chunk in chunks.items()},
        entities={entity_id: entity for entity_id, entity in normalization_result.entities.items()},
        relations=relations,
        chunk_extractions=chunk_extractions,
        normalization_debug={
            "merge_logs": normalization_result.merge_logs,
            "raw_entity_count": len(raw_entities),
            "canonical_entity_count": len(normalization_result.entities),
        },
        build_metadata={
            **extraction_stats,
            "build_warnings": extraction_warnings[:20],
        },
    )
    graph_index.communities = build_graph_communities(graph_index)
    graph_index.community_summaries = summarize_graph_communities(
        graph_index,
        client=client,
        model_name=summary_model,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )
    return graph_index
