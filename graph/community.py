"""本模块作用：基于实体关系子图进行图谱社群划分，并在失败时回退到连通分量分组。"""

from __future__ import annotations

from collections import defaultdict

from graph.store import GraphCommunity, GraphIndex


def build_entity_adjacency(index: GraphIndex) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], list[str]]]:
    """根据实体关系边构建实体邻接表。"""

    adjacency: dict[str, dict[str, float]] = {entity_id: {} for entity_id in index.entities}
    relation_lookup: dict[tuple[str, str], list[str]] = defaultdict(list)

    for relation in index.relations.values():
        if relation.source not in index.entities or relation.target not in index.entities:
            continue
        if relation.source == relation.target:
            continue
        adjacency.setdefault(relation.source, {})
        adjacency.setdefault(relation.target, {})
        adjacency[relation.source][relation.target] = adjacency[relation.source].get(relation.target, 0.0) + relation.weight
        adjacency[relation.target][relation.source] = adjacency[relation.target].get(relation.source, 0.0) + relation.weight
        relation_lookup[(relation.source, relation.target)].append(relation.relation_id)
        relation_lookup[(relation.target, relation.source)].append(relation.relation_id)
    return adjacency, relation_lookup


def detect_communities_with_louvain(adjacency: dict[str, dict[str, float]]) -> list[list[str]]:
    """优先尝试使用 Louvain 算法进行图谱社群发现。"""

    try:
        import networkx as nx
        from community import community_louvain
    except Exception as exc:  # pragma: no cover - 依赖缺失时自然回退
        raise RuntimeError(f"Louvain 依赖不可用：{exc}") from exc

    graph = nx.Graph()
    for source_id, neighbors in adjacency.items():
        graph.add_node(source_id)
        for target_id, weight in neighbors.items():
            if source_id >= target_id:
                continue
            graph.add_edge(source_id, target_id, weight=weight)

    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        return [[node] for node in graph.nodes()]

    partition = community_louvain.best_partition(graph, weight="weight", random_state=42)
    grouped_nodes: dict[int, list[str]] = defaultdict(list)
    for node_id, group_id in partition.items():
        grouped_nodes[int(group_id)].append(str(node_id))
    return [sorted(nodes) for nodes in grouped_nodes.values()]


def detect_communities_with_connected_components(adjacency: dict[str, dict[str, float]]) -> list[list[str]]:
    """使用连通分量作为图谱社群发现回退方案。"""

    visited_nodes: set[str] = set()
    communities: list[list[str]] = []

    for node_id in adjacency:
        if node_id in visited_nodes:
            continue
        stack = [node_id]
        component_nodes: list[str] = []
        while stack:
            current_node = stack.pop()
            if current_node in visited_nodes:
                continue
            visited_nodes.add(current_node)
            component_nodes.append(current_node)
            for neighbor in adjacency.get(current_node, {}):
                if neighbor not in visited_nodes:
                    stack.append(neighbor)
        communities.append(sorted(component_nodes))
    return communities


def build_graph_communities(index: GraphIndex) -> dict[str, GraphCommunity]:
    """根据当前图谱索引构建图谱社群划分结果。"""

    adjacency, relation_lookup = build_entity_adjacency(index)
    if not adjacency:
        return {}

    try:
        grouped_entities = detect_communities_with_louvain(adjacency)
        build_method = "louvain"
    except Exception:
        grouped_entities = detect_communities_with_connected_components(adjacency)
        build_method = "connected_components"

    ranked_groups = sorted(
        grouped_entities,
        key=lambda items: (-len(items), items),
    )

    communities: dict[str, GraphCommunity] = {}
    for index_number, entity_ids in enumerate(ranked_groups, start=1):
        relation_ids: set[str] = set()
        chunk_ids: set[str] = set()
        paper_ids: set[str] = set()

        entity_id_set = set(entity_ids)
        for entity_id in entity_ids:
            entity = index.entities.get(entity_id)
            if entity is None:
                continue
            chunk_ids.update(entity.source_chunk_ids)
            paper_ids.update(entity.source_paper_ids)

        for relation in index.relations.values():
            if relation.source in entity_id_set and relation.target in entity_id_set:
                relation_ids.add(relation.relation_id)
                if relation.source_chunk_id:
                    chunk_ids.add(relation.source_chunk_id)
                if relation.source_paper_id:
                    paper_ids.add(relation.source_paper_id)

        community_id = f"community_{index_number:04d}"
        communities[community_id] = GraphCommunity(
            community_id=community_id,
            entity_ids=sorted(entity_ids),
            chunk_ids=sorted(chunk_ids),
            paper_ids=sorted(paper_ids),
            relation_ids=sorted(relation_ids),
            size=len(entity_ids),
        )

    if communities:
        index.build_metadata["community_build_method"] = build_method
    return communities
