"""本模块作用：对 chunk 级抽取出的实体执行最小归一化、去重与可解释合并。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from graph.store import ChunkExtractionEntity, GraphEntity


ENTITY_TYPE_PRIORITY = [
    "ResearchProblem",
    "Task",
    "Modality",
    "Method",
    "ModelArchitecture",
    "Module",
    "LossFunction",
    "DegradationModel",
    "Prior",
    "Dataset",
    "Metric",
    "Baseline",
    "Finding",
    "Limitation",
    "Institution",
    "TimePeriod",
    "Concept",
    "Other",
]


@dataclass
class EntityNormalizationResult:
    """本数据结构作用：保存实体归一化结果与调试信息。"""

    entities: dict[str, GraphEntity] = field(default_factory=dict)
    raw_to_canonical: dict[str, str] = field(default_factory=dict)
    normalized_name_map: dict[str, str] = field(default_factory=dict)
    merge_logs: list[str] = field(default_factory=list)


def normalize_entity_name(name: str) -> str:
    """对实体名称做最小可解释归一化。"""

    cleaned_name = re.sub(r"[\(\)\[\]{}]+", " ", name or "")
    cleaned_name = cleaned_name.replace("/", " ").replace("_", " ").replace("-", " ")
    cleaned_name = re.sub(r"[^\w\u4e00-\u9fff\s]+", " ", cleaned_name.lower())
    cleaned_name = re.sub(r"\s+", " ", cleaned_name).strip()
    return cleaned_name


def normalize_entity_alias(name: str) -> str:
    """对别名做适度归一化，便于规则匹配。"""

    cleaned_alias = normalize_entity_name(name)
    if cleaned_alias:
        return cleaned_alias
    return re.sub(r"\s+", " ", (name or "").lower()).strip()


def build_entity_id(normalized_name: str) -> str:
    """根据归一化名称生成稳定实体编号。"""

    digest = hashlib.sha1(normalized_name.encode("utf-8", errors="surrogatepass")).hexdigest()[:12]
    return f"entity_{digest}"


def choose_entity_type(entity_types: list[str]) -> str:
    """从多个候选实体类型中选出更稳定的一个。"""

    priority_map = {name: index for index, name in enumerate(ENTITY_TYPE_PRIORITY)}
    cleaned_types = [item for item in entity_types if item]
    if not cleaned_types:
        return "Other"
    return min(cleaned_types, key=lambda item: priority_map.get(item, len(priority_map)))


def build_acronym(text: str) -> str:
    """根据英文长名称构造缩写，用于最小缩写归并。"""

    english_words = re.findall(r"[A-Za-z]+", text or "")
    if len(english_words) < 2:
        return ""
    return "".join(word[0] for word in english_words).upper()


def is_short_abbreviation(name: str) -> bool:
    """判断实体名是否更像英文缩写。"""

    compact_name = re.sub(r"[^A-Za-z]", "", name or "")
    return 2 <= len(compact_name) <= 8 and compact_name.isupper()


def calculate_description_overlap(description_a: str, description_b: str) -> float:
    """计算两个实体描述的最小词项重叠率。"""

    tokens_a = set(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", (description_a or "").lower()))
    tokens_b = set(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", (description_b or "").lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    overlap_count = len(tokens_a & tokens_b)
    return overlap_count / max(min(len(tokens_a), len(tokens_b)), 1)


def should_merge_abbreviation(candidate: GraphEntity, target: GraphEntity) -> bool:
    """判断缩写实体是否应并入长名称实体。"""

    if not is_short_abbreviation(candidate.name):
        return False
    if candidate.entity_id == target.entity_id:
        return False

    candidate_acronym = re.sub(r"[^A-Za-z]", "", candidate.name).upper()
    target_acronym = build_acronym(target.name)
    if not target_acronym or candidate_acronym != target_acronym:
        return False

    if candidate.entity_type != target.entity_type and candidate.entity_type != "Other":
        return False

    same_paper = bool(set(candidate.source_paper_ids) & set(target.source_paper_ids))
    same_chunk = bool(set(candidate.source_chunk_ids) & set(target.source_chunk_ids))
    description_overlap = calculate_description_overlap(candidate.description, target.description)
    return same_paper or same_chunk or description_overlap >= 0.3


def merge_graph_entity(target: GraphEntity, source: GraphEntity) -> None:
    """将一个实体合并到另一个规范实体中。"""

    if len(source.name) > len(target.name):
        target.name = source.name

    if source.description and len(source.description) > len(target.description):
        target.description = source.description

    target.entity_type = choose_entity_type([target.entity_type, source.entity_type])
    target.aliases = sorted(set(target.aliases + source.aliases + [target.name, source.name]))
    target.source_chunk_ids = sorted(set(target.source_chunk_ids + source.source_chunk_ids))
    target.source_paper_ids = sorted(set(target.source_paper_ids + source.source_paper_ids))
    target.mention_count += source.mention_count


def build_initial_entity(
    group_entities: list[ChunkExtractionEntity],
    normalized_name: str,
    paper_id_lookup: dict[str, str],
) -> GraphEntity:
    """基于同一归一化名称的原始实体构造初始规范实体。"""

    aliases = sorted({item.name.strip() for item in group_entities if item.name.strip()})
    descriptions = [item.description.strip() for item in group_entities if item.description.strip()]
    source_chunk_ids = sorted(
        {
            chunk_id
            for item in group_entities
            for chunk_id in item.source_chunk_ids
            if str(chunk_id).strip()
        }
    )
    source_paper_ids = sorted(
        {
            paper_id_lookup.get(chunk_id, "")
            for chunk_id in source_chunk_ids
            if paper_id_lookup.get(chunk_id, "")
        }
    )
    preferred_name = max(aliases, key=len) if aliases else normalized_name
    return GraphEntity(
        entity_id=build_entity_id(normalized_name),
        name=preferred_name,
        normalized_name=normalized_name,
        entity_type=choose_entity_type([item.entity_type for item in group_entities]),
        description=max(descriptions, key=len) if descriptions else "",
        aliases=aliases,
        source_chunk_ids=source_chunk_ids,
        source_paper_ids=source_paper_ids,
        mention_count=len(group_entities),
    )


def normalize_extracted_entities(
    raw_entities: list[ChunkExtractionEntity],
    chunk_to_paper_map: dict[str, str],
) -> EntityNormalizationResult:
    """对原始抽取实体进行规则归一化与去重。"""

    result = EntityNormalizationResult()
    if not raw_entities:
        return result

    grouped_entities: dict[str, list[ChunkExtractionEntity]] = {}
    raw_normalized_names: dict[str, str] = {}
    for item in raw_entities:
        normalized_name = normalize_entity_name(item.normalized_name or item.name)
        if not normalized_name:
            normalized_name = normalize_entity_alias(item.name)
        if not normalized_name:
            continue
        raw_normalized_names[item.entity_id] = normalized_name
        grouped_entities.setdefault(normalized_name, []).append(item)

    for normalized_name, group_items in grouped_entities.items():
        canonical_entity = build_initial_entity(group_items, normalized_name, chunk_to_paper_map)
        result.entities[canonical_entity.entity_id] = canonical_entity
        result.normalized_name_map[normalized_name] = canonical_entity.entity_id
        for raw_entity in group_items:
            result.raw_to_canonical[raw_entity.entity_id] = canonical_entity.entity_id

    entity_items = sorted(result.entities.values(), key=lambda item: (-len(item.aliases), -item.mention_count, item.name))
    merged_alias_map: dict[str, str] = {}
    removed_entity_ids: set[str] = set()
    for candidate in entity_items:
        if candidate.entity_id in removed_entity_ids:
            continue
        for target in entity_items:
            if target.entity_id in removed_entity_ids:
                continue
            if not should_merge_abbreviation(candidate, target):
                continue
            merge_graph_entity(target, candidate)
            merged_alias_map[candidate.entity_id] = target.entity_id
            removed_entity_ids.add(candidate.entity_id)
            result.merge_logs.append(
                f"缩写归并：{candidate.name} -> {target.name}（归因：缩写与长名称共现）"
            )
            break

    for removed_entity_id in removed_entity_ids:
        result.entities.pop(removed_entity_id, None)

    for raw_entity_id, canonical_id in list(result.raw_to_canonical.items()):
        if canonical_id in merged_alias_map:
            result.raw_to_canonical[raw_entity_id] = merged_alias_map[canonical_id]

    result.normalized_name_map = {
        normalized_name: merged_alias_map.get(entity_id, entity_id)
        for normalized_name, entity_id in result.normalized_name_map.items()
    }

    deduplicated_entities: dict[str, GraphEntity] = {}
    for entity in result.entities.values():
        entity.aliases = sorted({alias for alias in entity.aliases if alias})
        entity.source_chunk_ids = sorted(set(entity.source_chunk_ids))
        entity.source_paper_ids = sorted(set(entity.source_paper_ids))
        deduplicated_entities[entity.entity_id] = entity
    result.entities = deduplicated_entities
    return result
