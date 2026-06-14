"""本模块作用：为图谱社群生成本地可持久化摘要，并在可用时补充向量表示。"""

from __future__ import annotations

import re

from graph.store import GraphCommunity, GraphCommunitySummary, GraphIndex
from llm_dashscope import DashScopeClient, parse_first_json_object


def truncate_text(text: str, max_chars: int = 180) -> str:
    """将长文本裁剪到摘要展示所需长度。"""

    cleaned_text = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned_text) <= max_chars:
        return cleaned_text
    return cleaned_text[:max_chars].rstrip() + "..."


def rank_community_entity_names(index: GraphIndex, community: GraphCommunity, top_k: int = 6) -> list[str]:
    """返回图谱社群内更具代表性的实体名称。"""

    entity_items = []
    for entity_id in community.entity_ids:
        entity = index.entities.get(entity_id)
        if entity is None:
            continue
        entity_items.append((entity.mention_count, len(entity.source_paper_ids), entity.name))
    entity_items.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[2] for item in entity_items[:top_k]]


def rank_community_relations(index: GraphIndex, community: GraphCommunity, top_k: int = 6) -> list[str]:
    """返回图谱社群内更具代表性的关系描述。"""

    relation_items: list[tuple[float, str]] = []
    for relation_id in community.relation_ids:
        relation = index.relations.get(relation_id)
        if relation is None:
            continue
        source_entity = index.entities.get(relation.source)
        target_entity = index.entities.get(relation.target)
        if source_entity is None or target_entity is None:
            continue
        relation_items.append(
            (
                relation.weight,
                f"{source_entity.name} --{relation.relation_type}--> {target_entity.name}",
            )
        )
    relation_items.sort(key=lambda item: (-item[0], item[1]))
    unique_relations: list[str] = []
    seen_relations: set[str] = set()
    for _, relation_text in relation_items:
        if relation_text in seen_relations:
            continue
        seen_relations.add(relation_text)
        unique_relations.append(relation_text)
        if len(unique_relations) >= top_k:
            break
    return unique_relations


def rank_representative_chunk_ids(index: GraphIndex, community: GraphCommunity, top_k: int = 4) -> list[str]:
    """返回图谱社群最具代表性的 chunk 列表。"""

    scored_chunks: list[tuple[int, int, str]] = []
    for chunk_id in community.chunk_ids:
        chunk = index.chunks.get(chunk_id)
        if chunk is None:
            continue
        scored_chunks.append((len(chunk.entity_ids), len(chunk.relation_ids), chunk_id))
    scored_chunks.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[2] for item in scored_chunks[:top_k]]


def build_summary_messages(
    community: GraphCommunity,
    entity_names: list[str],
    relation_texts: list[str],
    chunk_texts: list[str],
) -> list[dict[str, str]]:
    """构建图谱社群摘要大模型提示词。"""

    system_prompt = (
        "你是遥感图像融合论文 GraphRAG 的图谱社群摘要助手。"
        "请仅根据给定图谱社群中的实体、关系和 chunk 证据，生成一个保守、可检索的中文摘要。"
        "摘要应优先围绕方法族、任务类型、输入模态、数据集、指标、空间-光谱建模机制、退化模型、先验、zero-shot 或 diffusion 主题组织。"
        "输出必须是严格 JSON 对象。"
    )
    user_prompt = (
        "请根据下列图谱社群材料输出 JSON：\n"
        "{\n"
        '  "title": "图谱主题簇标题",\n'
        '  "summary": "2-4 句摘要",\n'
        '  "key_entities": ["实体名"],\n'
        '  "key_relations": ["实体A --relation--> 实体B"],\n'
        '  "representative_chunks": ["chunk_id"]\n'
        "}\n"
        "要求：\n"
        "1. 不要编造未出现的信息。\n"
        "2. title 和 summary 要便于后续按全色锐化、HSI-MSI fusion、空间-光谱融合、退化模型、数据集或指标检索。\n"
        "3. key_entities 不超过 6 个，key_relations 不超过 6 个。\n"
        f"community_id：{community.community_id}\n"
        f"实体：{entity_names}\n"
        f"关系：{relation_texts}\n"
        f"代表性 chunk 证据：\n" + "\n".join(chunk_texts)
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_rule_based_summary(index: GraphIndex, community: GraphCommunity) -> GraphCommunitySummary:
    """在 LLM 不可用时，构造规则版图谱社群摘要。"""

    entity_names = rank_community_entity_names(index, community)
    relation_texts = rank_community_relations(index, community)
    representative_chunks = rank_representative_chunk_ids(index, community)

    if entity_names:
        title = f"图谱主题簇：{'、'.join(entity_names[:3])}"
    else:
        title = f"图谱主题簇：{community.community_id}"

    summary_lines = []
    if entity_names:
        summary_lines.append(f"该图谱主题簇主要围绕 {', '.join(entity_names[:5])} 展开。")
    if relation_texts:
        summary_lines.append(f"核心关系包括：{'；'.join(relation_texts[:3])}。")
    if community.paper_ids:
        summary_lines.append(f"相关论文数量为 {len(community.paper_ids)} 篇。")
    if not summary_lines:
        summary_lines.append("该图谱主题簇当前证据较少，建议回看代表性 chunk。")

    return GraphCommunitySummary(
        community_id=community.community_id,
        title=title,
        summary="".join(summary_lines),
        key_entities=entity_names[:6],
        key_relations=relation_texts[:6],
        representative_chunks=representative_chunks,
    )


def embed_summary_if_available(
    summary: GraphCommunitySummary,
    client: DashScopeClient | None,
    embedding_model: str,
    embedding_dimensions: int,
) -> None:
    """若向量能力可用，则为图谱社群摘要补充 embedding。"""

    if client is None or not embedding_model.strip() or embedding_dimensions <= 0:
        return
    try:
        vector = client.embed_texts(
            model=embedding_model,
            texts=[f"{summary.title}\n{summary.summary}"],
            dimensions=embedding_dimensions,
        )[0]
    except Exception:
        return
    summary.embedding_vector = vector


def summarize_single_community(
    index: GraphIndex,
    community: GraphCommunity,
    client: DashScopeClient | None,
    model_name: str,
    embedding_model: str,
    embedding_dimensions: int,
) -> GraphCommunitySummary:
    """为单个图谱社群生成摘要对象。"""

    rule_summary = build_rule_based_summary(index, community)
    if client is None or not model_name.strip():
        embed_summary_if_available(rule_summary, client, embedding_model, embedding_dimensions)
        return rule_summary

    representative_chunk_ids = rank_representative_chunk_ids(index, community)
    chunk_texts = []
    for chunk_id in representative_chunk_ids:
        chunk = index.chunks.get(chunk_id)
        if chunk is None:
            continue
        chunk_texts.append(f"{chunk_id}: {truncate_text(chunk.text)}")

    try:
        raw_text = client.chat(
            model=model_name,
            messages=build_summary_messages(
                community=community,
                entity_names=rank_community_entity_names(index, community),
                relation_texts=rank_community_relations(index, community),
                chunk_texts=chunk_texts,
            ),
            temperature=0.2,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        payload = parse_first_json_object(raw_text)
        if payload is None:
            embed_summary_if_available(rule_summary, client, embedding_model, embedding_dimensions)
            return rule_summary

        summary = GraphCommunitySummary(
            community_id=community.community_id,
            title=str(payload.get("title", "")).strip() or rule_summary.title,
            summary=str(payload.get("summary", "")).strip() or rule_summary.summary,
            key_entities=[
                str(item).strip()
                for item in payload.get("key_entities", rule_summary.key_entities)
                if str(item).strip()
            ][:6]
            or rule_summary.key_entities,
            key_relations=[
                str(item).strip()
                for item in payload.get("key_relations", rule_summary.key_relations)
                if str(item).strip()
            ][:6]
            or rule_summary.key_relations,
            representative_chunks=[
                str(item).strip()
                for item in payload.get("representative_chunks", rule_summary.representative_chunks)
                if str(item).strip()
            ][:4]
            or rule_summary.representative_chunks,
        )
        embed_summary_if_available(summary, client, embedding_model, embedding_dimensions)
        return summary
    except Exception:
        embed_summary_if_available(rule_summary, client, embedding_model, embedding_dimensions)
        return rule_summary


def summarize_graph_communities(
    index: GraphIndex,
    client: DashScopeClient | None,
    model_name: str,
    embedding_model: str = "",
    embedding_dimensions: int = 0,
) -> dict[str, GraphCommunitySummary]:
    """批量生成图谱社群摘要。"""

    summaries: dict[str, GraphCommunitySummary] = {}
    for community in sorted(index.communities.values(), key=lambda item: (-item.size, item.community_id)):
        summaries[community.community_id] = summarize_single_community(
            index=index,
            community=community,
            client=client,
            model_name=model_name,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
        )
    return summaries
