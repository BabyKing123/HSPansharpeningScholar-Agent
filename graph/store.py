"""本模块作用：定义 GraphRAG 本地索引的数据结构，并提供 JSON 持久化与兼容性判断能力。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

GRAPH_INDEX_VERSION = "2.0"


@dataclass
class ChunkExtractionEntity:
    """本数据结构作用：保存单个 chunk 的原始实体抽取结果。"""

    entity_id: str
    name: str
    normalized_name: str
    entity_type: str
    description: str
    source_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class ChunkExtractionRelation:
    """本数据结构作用：保存单个 chunk 的原始关系抽取结果。"""

    relation_id: str
    source: str
    target: str
    relation_type: str
    evidence_text: str
    source_chunk_id: str


@dataclass
class ChunkExtractionRecord:
    """本数据结构作用：保存单个 chunk 的完整抽取结果与缓存信息。"""

    chunk_id: str
    text_hash: str
    entities: list[ChunkExtractionEntity] = field(default_factory=list)
    relations: list[ChunkExtractionRelation] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    extraction_method: str = "rule"
    error_message: str = ""


@dataclass
class GraphPaper:
    """本数据结构作用：保存论文节点的最小图谱信息。"""

    paper_id: str
    document_id: str
    file_name: str
    source_path: str
    chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class GraphChunk:
    """本数据结构作用：保存 chunk 节点的最小图谱信息。"""

    chunk_id: str
    document_id: str
    paper_id: str
    file_name: str
    source_path: str
    text: str
    text_hash: str
    chunk_index: int = 0
    page_numbers: list[int] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    relation_ids: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    claim_texts: list[str] = field(default_factory=list)


@dataclass
class GraphEntity:
    """本数据结构作用：保存归一化后的实体节点信息。"""

    entity_id: str
    name: str
    normalized_name: str
    entity_type: str
    description: str
    aliases: list[str] = field(default_factory=list)
    source_chunk_ids: list[str] = field(default_factory=list)
    source_paper_ids: list[str] = field(default_factory=list)
    mention_count: int = 0


@dataclass
class GraphRelation:
    """本数据结构作用：保存实体间关系边信息。"""

    relation_id: str
    source: str
    target: str
    relation_type: str
    evidence_text: str
    source_chunk_id: str
    source_paper_id: str
    weight: float = 1.0


@dataclass
class GraphCommunity:
    """本数据结构作用：保存图谱社群划分结果。"""

    community_id: str
    entity_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    paper_ids: list[str] = field(default_factory=list)
    relation_ids: list[str] = field(default_factory=list)
    size: int = 0


@dataclass
class GraphCommunitySummary:
    """本数据结构作用：保存图谱社群级摘要结果，供 global retrieval 使用。"""

    community_id: str
    title: str
    summary: str
    key_entities: list[str] = field(default_factory=list)
    key_relations: list[str] = field(default_factory=list)
    representative_chunks: list[str] = field(default_factory=list)
    embedding_vector: list[float] = field(default_factory=list)


@dataclass
class GraphIndex:
    """本数据结构作用：保存整个本地 GraphRAG 索引。"""

    graph_version: str
    created_at: str
    source_fingerprint: str
    papers: dict[str, GraphPaper] = field(default_factory=dict)
    chunks: dict[str, GraphChunk] = field(default_factory=dict)
    entities: dict[str, GraphEntity] = field(default_factory=dict)
    relations: dict[str, GraphRelation] = field(default_factory=dict)
    communities: dict[str, GraphCommunity] = field(default_factory=dict)
    community_summaries: dict[str, GraphCommunitySummary] = field(default_factory=dict)
    chunk_extractions: dict[str, ChunkExtractionRecord] = field(default_factory=dict)
    normalization_debug: dict[str, object] = field(default_factory=dict)
    build_metadata: dict[str, object] = field(default_factory=dict)
    index_path: str = ""


def get_default_graph_index_path(processed_data_dir: str | Path) -> Path:
    """返回默认图谱索引文件路径。"""

    target_dir = Path(processed_data_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / "graph_index.json"


def build_graph_source_fingerprint(chunk_records: list[dict[str, object]]) -> str:
    """为当前 chunk_records 生成稳定图谱指纹。"""

    fingerprint_items: list[str] = []
    for chunk_record in chunk_records:
        metadata = chunk_record.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        fingerprint_items.append(
            "|".join(
                [
                    str(chunk_record.get("chunk_id", "")),
                    str(chunk_record.get("document_id", "")),
                    str(metadata.get("chunk_index", "")),
                    str(metadata.get("page_numbers", [])),
                    str(chunk_record.get("text", "")),
                ]
            )
        )
    fingerprint_source = "\n".join(fingerprint_items)
    return hashlib.sha256(fingerprint_source.encode("utf-8", errors="surrogatepass")).hexdigest()


def graph_index_to_payload(index: GraphIndex) -> dict[str, Any]:
    """将图谱索引对象转换为可序列化字典。"""

    return {
        "graph_version": index.graph_version,
        "created_at": index.created_at,
        "source_fingerprint": index.source_fingerprint,
        "papers": {key: asdict(value) for key, value in index.papers.items()},
        "chunks": {key: asdict(value) for key, value in index.chunks.items()},
        "entities": {key: asdict(value) for key, value in index.entities.items()},
        "relations": {key: asdict(value) for key, value in index.relations.items()},
        "communities": {key: asdict(value) for key, value in index.communities.items()},
        "community_summaries": {key: asdict(value) for key, value in index.community_summaries.items()},
        "chunk_extractions": {key: asdict(value) for key, value in index.chunk_extractions.items()},
        "normalization_debug": index.normalization_debug,
        "build_metadata": index.build_metadata,
    }


def _load_chunk_extraction_entity(payload: dict[str, Any]) -> ChunkExtractionEntity:
    """从字典恢复单个抽取实体。"""

    return ChunkExtractionEntity(
        entity_id=str(payload.get("entity_id", "")),
        name=str(payload.get("name", "")),
        normalized_name=str(payload.get("normalized_name", "")),
        entity_type=str(payload.get("entity_type", "Other")),
        description=str(payload.get("description", "")),
        source_chunk_ids=[str(item) for item in payload.get("source_chunk_ids", [])],
    )


def _load_chunk_extraction_relation(payload: dict[str, Any]) -> ChunkExtractionRelation:
    """从字典恢复单个抽取关系。"""

    return ChunkExtractionRelation(
        relation_id=str(payload.get("relation_id", "")),
        source=str(payload.get("source", "")),
        target=str(payload.get("target", "")),
        relation_type=str(payload.get("relation_type", "related_to")),
        evidence_text=str(payload.get("evidence_text", "")),
        source_chunk_id=str(payload.get("source_chunk_id", "")),
    )


def _load_chunk_extraction_record(payload: dict[str, Any]) -> ChunkExtractionRecord:
    """从字典恢复单个 chunk 的抽取缓存。"""

    return ChunkExtractionRecord(
        chunk_id=str(payload.get("chunk_id", "")),
        text_hash=str(payload.get("text_hash", "")),
        entities=[
            _load_chunk_extraction_entity(item)
            for item in payload.get("entities", [])
            if isinstance(item, dict)
        ],
        relations=[
            _load_chunk_extraction_relation(item)
            for item in payload.get("relations", [])
            if isinstance(item, dict)
        ],
        claims=[str(item) for item in payload.get("claims", []) if str(item).strip()],
        keywords=[str(item) for item in payload.get("keywords", []) if str(item).strip()],
        extraction_method=str(payload.get("extraction_method", "rule")),
        error_message=str(payload.get("error_message", "")),
    )


def _load_graph_paper(payload: dict[str, Any]) -> GraphPaper:
    """从字典恢复论文节点。"""

    return GraphPaper(
        paper_id=str(payload.get("paper_id", "")),
        document_id=str(payload.get("document_id", "")),
        file_name=str(payload.get("file_name", "")),
        source_path=str(payload.get("source_path", "")),
        chunk_ids=[str(item) for item in payload.get("chunk_ids", [])],
        metadata=payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {},
    )


def _load_graph_chunk(payload: dict[str, Any]) -> GraphChunk:
    """从字典恢复 chunk 节点。"""

    return GraphChunk(
        chunk_id=str(payload.get("chunk_id", "")),
        document_id=str(payload.get("document_id", "")),
        paper_id=str(payload.get("paper_id", "")),
        file_name=str(payload.get("file_name", "")),
        source_path=str(payload.get("source_path", "")),
        text=str(payload.get("text", "")),
        text_hash=str(payload.get("text_hash", "")),
        chunk_index=int(payload.get("chunk_index", 0)),
        page_numbers=[int(item) for item in payload.get("page_numbers", [])],
        entity_ids=[str(item) for item in payload.get("entity_ids", [])],
        relation_ids=[str(item) for item in payload.get("relation_ids", [])],
        keywords=[str(item) for item in payload.get("keywords", []) if str(item).strip()],
        claim_texts=[str(item) for item in payload.get("claim_texts", []) if str(item).strip()],
    )


def _load_graph_entity(payload: dict[str, Any]) -> GraphEntity:
    """从字典恢复实体节点。"""

    return GraphEntity(
        entity_id=str(payload.get("entity_id", "")),
        name=str(payload.get("name", "")),
        normalized_name=str(payload.get("normalized_name", "")),
        entity_type=str(payload.get("entity_type", "Other")),
        description=str(payload.get("description", "")),
        aliases=[str(item) for item in payload.get("aliases", []) if str(item).strip()],
        source_chunk_ids=[str(item) for item in payload.get("source_chunk_ids", [])],
        source_paper_ids=[str(item) for item in payload.get("source_paper_ids", [])],
        mention_count=int(payload.get("mention_count", 0)),
    )


def _load_graph_relation(payload: dict[str, Any]) -> GraphRelation:
    """从字典恢复关系边。"""

    return GraphRelation(
        relation_id=str(payload.get("relation_id", "")),
        source=str(payload.get("source", "")),
        target=str(payload.get("target", "")),
        relation_type=str(payload.get("relation_type", "related_to")),
        evidence_text=str(payload.get("evidence_text", "")),
        source_chunk_id=str(payload.get("source_chunk_id", "")),
        source_paper_id=str(payload.get("source_paper_id", "")),
        weight=float(payload.get("weight", 1.0)),
    )


def _load_graph_community(payload: dict[str, Any]) -> GraphCommunity:
    """从字典恢复图谱社群信息。"""

    return GraphCommunity(
        community_id=str(payload.get("community_id", "")),
        entity_ids=[str(item) for item in payload.get("entity_ids", [])],
        chunk_ids=[str(item) for item in payload.get("chunk_ids", [])],
        paper_ids=[str(item) for item in payload.get("paper_ids", [])],
        relation_ids=[str(item) for item in payload.get("relation_ids", [])],
        size=int(payload.get("size", 0)),
    )


def _load_graph_community_summary(payload: dict[str, Any]) -> GraphCommunitySummary:
    """从字典恢复图谱社群摘要。"""

    return GraphCommunitySummary(
        community_id=str(payload.get("community_id", "")),
        title=str(payload.get("title", "")),
        summary=str(payload.get("summary", "")),
        key_entities=[str(item) for item in payload.get("key_entities", []) if str(item).strip()],
        key_relations=[str(item) for item in payload.get("key_relations", []) if str(item).strip()],
        representative_chunks=[str(item) for item in payload.get("representative_chunks", []) if str(item).strip()],
        embedding_vector=[float(item) for item in payload.get("embedding_vector", [])],
    )


def save_graph_index(index: GraphIndex, index_path: str | Path) -> None:
    """将图谱索引保存到本地 JSON 文件。"""

    path = Path(index_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    index.index_path = str(path)
    path.write_bytes(
        json.dumps(graph_index_to_payload(index), ensure_ascii=False, indent=2).encode(
            "utf-8",
            errors="surrogatepass",
        )
    )


def load_graph_index(index_path: str | Path) -> GraphIndex | None:
    """从本地 JSON 文件加载图谱索引。"""

    path = Path(index_path).expanduser().resolve()
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_bytes().decode("utf-8", errors="surrogatepass"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        raise ValueError("图谱索引文件格式不正确。")

    return GraphIndex(
        graph_version=str(payload.get("graph_version", "")),
        created_at=str(payload.get("created_at", "")),
        source_fingerprint=str(payload.get("source_fingerprint", "")),
        papers={
            str(key): _load_graph_paper(value)
            for key, value in payload.get("papers", {}).items()
            if isinstance(value, dict)
        },
        chunks={
            str(key): _load_graph_chunk(value)
            for key, value in payload.get("chunks", {}).items()
            if isinstance(value, dict)
        },
        entities={
            str(key): _load_graph_entity(value)
            for key, value in payload.get("entities", {}).items()
            if isinstance(value, dict)
        },
        relations={
            str(key): _load_graph_relation(value)
            for key, value in payload.get("relations", {}).items()
            if isinstance(value, dict)
        },
        communities={
            str(key): _load_graph_community(value)
            for key, value in payload.get("communities", {}).items()
            if isinstance(value, dict)
        },
        community_summaries={
            str(key): _load_graph_community_summary(value)
            for key, value in payload.get("community_summaries", {}).items()
            if isinstance(value, dict)
        },
        chunk_extractions={
            str(key): _load_chunk_extraction_record(value)
            for key, value in payload.get("chunk_extractions", {}).items()
            if isinstance(value, dict)
        },
        normalization_debug=payload.get("normalization_debug", {})
        if isinstance(payload.get("normalization_debug", {}), dict)
        else {},
        build_metadata=payload.get("build_metadata", {})
        if isinstance(payload.get("build_metadata", {}), dict)
        else {},
        index_path=str(path),
    )


def is_graph_index_compatible(index: GraphIndex, chunk_records: list[dict[str, object]]) -> bool:
    """判断图谱索引是否与当前知识库兼容。"""

    if not index.graph_version or not index.graph_version.startswith("2."):
        return False
    return index.source_fingerprint == build_graph_source_fingerprint(chunk_records)


def summarize_graph_index_status(
    index: GraphIndex | None,
    chunk_records: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """返回图谱索引状态统计，便于 CLI 展示。"""

    if index is None:
        return {
            "exists": False,
            "entity_count": 0,
            "relation_count": 0,
            "community_count": 0,
            "community_summary_count": 0,
            "source_fingerprint_match": False,
            "index_path": "",
        }

    fingerprint_match = False
    if chunk_records is not None:
        fingerprint_match = is_graph_index_compatible(index, chunk_records)

    return {
        "exists": True,
        "entity_count": len(index.entities),
        "relation_count": len(index.relations),
        "community_count": len(index.communities),
        "community_summary_count": len(index.community_summaries),
        "source_fingerprint_match": fingerprint_match,
        "index_path": index.index_path,
    }
