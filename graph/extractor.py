"""本模块作用：基于 DashScope 或规则方法，从 chunk 中抽取实体、关系与可追溯证据。"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from graph.normalizer import normalize_entity_name
from graph.store import (
    ChunkExtractionEntity,
    ChunkExtractionRecord,
    ChunkExtractionRelation,
)
from llm_dashscope import DashScopeClient, parse_first_json_object
from rag.retriever import extract_search_terms


GRAPH_ENTITY_TYPES = [
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

GRAPH_RELATION_TYPES = [
    "addresses_task",
    "uses_modality",
    "uses_method",
    "contains_module",
    "uses_loss",
    "uses_degradation_model",
    "uses_prior",
    "evaluates_on",
    "reports_metric",
    "compares_with",
    "improves",
    "preserves_spectral_quality",
    "enhances_spatial_detail",
    "limited_by",
    "finds",
    "mentions",
    "related_to",
]


def build_chunk_text_hash(text: str) -> str:
    """为 chunk 文本生成稳定哈希。"""

    return hashlib.sha1((text or "").encode("utf-8", errors="surrogatepass")).hexdigest()


def truncate_text(text: str, max_chars: int = 240) -> str:
    """将文本裁剪到抽取或证据展示需要的长度。"""

    cleaned_text = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned_text) <= max_chars:
        return cleaned_text
    return cleaned_text[:max_chars].rstrip() + "..."


def split_into_sentences(text: str) -> list[str]:
    """将 chunk 文本切分为句子。"""

    raw_items = re.split(r"[。！？!?；;\n]+", text or "")
    return [re.sub(r"\s+", " ", item).strip() for item in raw_items if re.sub(r"\s+", " ", item).strip()]


def infer_entity_type(name: str, context_text: str = "") -> str:
    """根据名称与上下文做最小实体类型判断。"""

    lowered_name = (name or "").lower()
    lowered_context = (context_text or "").lower()
    merged_text = f"{lowered_name} {lowered_context}"

    if "loss" not in lowered_name and re.search(r"\b(psnr|ssim|sam|ergas|rmse|qnr|d_lambda|d_s|q2n|scc|cc)\b|指标|评价指标", lowered_name):
        return "Metric"
    if re.search(r"\b(pan|panchromatic|ms|msi|hsi|lrhs|hrhs|lms|hs|hyperspectral image|multispectral image)\b|全色图像|高光谱图像|多光谱图像|低分辨率高光谱|高分辨率高光谱", lowered_name):
        return "Modality"
    if re.search(r"\b(cave|harvard|pavia|washington dc mall|botswana|chikusei|worldview|gaofen|quickbird|ikonos|aviris|rosis|moffett|salinas)\b|数据集|基准集", lowered_name):
        return "Dataset"
    if re.search(r"\b(loss|l1|l2|sam loss|spectral loss|spatial loss|reconstruction loss|degradation consistency loss)\b|损失|光谱损失|空间损失|重建损失|退化一致性", lowered_name):
        return "LossFunction"
    if re.search(r"\b(psf|srf|blur kernel|blur|downsampling|spectral response|wald protocol|degradation model)\b|退化模型|点扩散函数|光谱响应|模糊核|下采样", lowered_name):
        return "DegradationModel"
    if re.search(r"\b(prior|zero-shot prior|image prior|latent prior|degradation prior|zero shot)\b|先验|零样本|潜空间先验|退化先验", lowered_name):
        return "Prior"
    if re.search(r"\b(cnn|transformer|u-net|unet|diffusion|vae|autoencoder|neural operator|network|architecture)\b|网络结构|扩散模型|自编码器|神经算子", lowered_name):
        return "ModelArchitecture"
    if re.search(r"\b(attention|gate|gating|spatial attention|channel attention|spectral attention|arconv|wavelet block|module|block)\b|注意力|门控|模块|小波块|通道注意力|光谱注意力|空间注意力", lowered_name):
        return "Module"
    if re.search(r"\b(pansharpen\w*|pan-sharpen\w*|hyperspectral pansharpen\w*|hsi-msi fusion|hsi super-resolution|super-resolution|image fusion|task)\b|全色锐化|高光谱全色锐化|图像融合|超分辨率|任务", lowered_name):
        return "Task"
    if re.search(r"\b(method|model|framework|algorithm|approach|unfolding|tensor decomposition|wavelet|optimization)\b|方法|模型|框架|算法|优化|张量分解|小波|展开", merged_text):
        return "Method"
    if re.search(r"\b(baseline|comparison method|state-of-the-art|sota)\b|基线|对比方法", merged_text):
        return "Baseline"
    if re.search(r"\b(limitation|future work|runtime|complexity|generalization|robustness)\b|局限|不足|未来工作|复杂度|泛化|鲁棒", merged_text):
        return "Limitation"
    if re.search(r"\b(finding|result|effect|impact|improvement)\b|结论|结果|发现|影响|提升|改善", merged_text):
        return "Finding"
    if re.search(r"\b(university|institute|laboratory|academy)\b|大学|学院|研究院|实验室", merged_text):
        return "Institution"
    if re.search(r"\b(19\d{2}|20\d{2}|period|year|decade)\b|\d{4}年|时期|阶段", merged_text):
        return "TimePeriod"
    if re.search(r"\b(problem|challenge|question|issue|ill-posed)\b|问题|挑战|难点|病态", merged_text):
        return "ResearchProblem"
    if re.search(r"\b(concept|theory|mechanism)\b|概念|理论|机制", merged_text):
        return "Concept"
    return "Other"


def infer_relation_type(sentence: str) -> str:
    """根据证据句做最小关系类型判断。"""

    lowered_sentence = sentence.lower()
    if re.search(r"\b(evaluate|experiment on|tested on)\b|在.*数据集|在.*实验|评估于|测试于", lowered_sentence):
        return "evaluates_on"
    if re.search(r"\b(psnr|ssim|sam|ergas|rmse|qnr|d_lambda|d_s|q2n|scc|metric)\b|指标|评价", lowered_sentence):
        return "reports_metric"
    if re.search(r"\b(preserve|spectral fidelity|spectral consistency|spectral distortion)\b|光谱保真|光谱一致|光谱失真|保持光谱", lowered_sentence):
        return "preserves_spectral_quality"
    if re.search(r"\b(enhance spatial|spatial detail|high-frequency|texture|edge)\b|增强空间|空间细节|高频|纹理|边缘", lowered_sentence):
        return "enhances_spatial_detail"
    if re.search(r"\b(compare|versus|than)\b|比较|对比|优于", lowered_sentence):
        return "compares_with"
    if re.search(r"\b(improve|improved|enhance|enhanced|outperform)\b|提升|改善|优于", lowered_sentence):
        return "improves"
    if re.search(r"\b(limit|limited|constraint)\b|受限于|局限于|限制", lowered_sentence):
        return "limited_by"
    if re.search(r"\b(loss|l1|l2|sam loss|spectral loss|reconstruction loss)\b|损失|光谱损失|重建损失", lowered_sentence):
        return "uses_loss"
    if re.search(r"\b(psf|srf|blur|downsampling|spectral response|wald protocol|degradation)\b|退化|模糊|下采样|光谱响应|wald", lowered_sentence):
        return "uses_degradation_model"
    if re.search(r"\b(prior|zero-shot|zero shot|latent prior|guided by pan|pan guidance)\b|先验|零样本|PAN.*引导|全色.*引导", lowered_sentence):
        return "uses_prior"
    if re.search(r"\b(use|uses|using|input|guided by)\b.*\b(pan|hsi|msi|lrhs|hrhs|lms|panchromatic|hyperspectral|multispectral)\b|输入.*(pan|hsi|msi|lrhs|hrhs|lms)|使用.*(pan|hsi|msi|lrhs|hrhs|lms)", lowered_sentence):
        return "uses_modality"
    if re.search(r"\b(contain|consist|module|block|attention|gate|gating)\b|包含|模块|注意力|门控", lowered_sentence):
        return "contains_module"
    if re.search(r"\b(focus on|address|study|investigate|target)\b|关注|聚焦|探讨|研究|面向", lowered_sentence):
        return "addresses_task"
    if re.search(r"\b(find|show|suggest|indicate|reveal)\b|表明|发现|指出|说明", lowered_sentence):
        return "finds"
    if re.search(r"\b(use|using|adopt|apply|based on)\b|采用|使用|基于", lowered_sentence):
        return "uses_method"
    if re.search(r"\b(mention|introduce|describe)\b|提到|描述|涉及", lowered_sentence):
        return "mentions"
    return "related_to"


def build_entity_id(chunk_id: str, index: int) -> str:
    """为 chunk 内实体生成稳定局部编号。"""

    return f"{chunk_id}_entity_{index:02d}"


def build_relation_id(chunk_id: str, index: int) -> str:
    """为 chunk 内关系生成稳定局部编号。"""

    return f"{chunk_id}_relation_{index:02d}"


def extract_candidate_entities_rule(chunk_id: str, text: str) -> list[ChunkExtractionEntity]:
    """使用规则方式从 chunk 中提取保守实体。"""

    candidates: list[str] = []
    seen_names: set[str] = set()

    domain_patterns: list[str] = []
    for pattern in [
        r"\b(?:PAN|MS|MSI|HSI|LRHS|HRHS|LMS|HS)\b",
        r"\b(?:PSNR|SSIM|SAM|ERGAS|RMSE|QNR|D_lambda|D_s|Q2n|SCC)\b",
        r"\b(?:CAVE|Harvard|Pavia(?: Center| University)?|Washington DC Mall|Botswana|Chikusei|WorldView(?:-2|-3)?|GaoFen(?:-2)?|QuickBird|IKONOS|AVIRIS|ROSIS|Moffett|Salinas)\b",
        r"\b(?:pansharpening|hyperspectral pansharpening|HSI-MSI fusion|HSI super-resolution|latent diffusion|zero-shot|spatial-spectral fusion|tensor decomposition|neural operator|degradation model|Wald protocol)\b",
        r"\b(?:Transformer|CNN|U-Net|UNet|Diffusion|VAE|Autoencoder|attention|gate|gating|ARConv|wavelet block|SAM loss|spectral loss|spatial loss|reconstruction loss)\b",
    ]:
        domain_patterns.extend(match.group(0) for match in re.finditer(pattern, text or "", flags=re.IGNORECASE))

    english_patterns = re.findall(
        r"\b(?:[A-Z]{2,8}|[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,4})\b",
        text or "",
    )
    chinese_patterns = re.findall(
        r"[\u4e00-\u9fff]{2,20}(?:方法|模型|框架|网络|数据集|指标|全色锐化|高光谱|多光谱|模块|注意力|门控|扩散|先验|退化|损失|结论|发现|问题|机制)",
        text or "",
    )
    year_patterns = re.findall(r"(?:19|20)\d{2}(?:年)?", text or "")

    for candidate in domain_patterns + english_patterns + chinese_patterns + year_patterns:
        cleaned_name = re.sub(r"\s+", " ", candidate).strip(" ,.;:()[]{}")
        normalized_name = normalize_entity_name(cleaned_name)
        if not cleaned_name or not normalized_name or normalized_name in seen_names:
            continue
        if normalized_name in {"the", "this", "that", "these", "those", "proposed", "method", "network", "model", "figure", "table"}:
            continue
        if len(cleaned_name) < 2:
            continue
        seen_names.add(normalized_name)
        candidates.append(cleaned_name)

    entities: list[ChunkExtractionEntity] = []
    for index, candidate in enumerate(candidates, start=1):
        entity_type = infer_entity_type(candidate, text)
        description = truncate_text(candidate if candidate in text else text)
        entities.append(
            ChunkExtractionEntity(
                entity_id=build_entity_id(chunk_id, index),
                name=candidate,
                normalized_name=normalize_entity_name(candidate),
                entity_type=entity_type,
                description=description,
                source_chunk_ids=[chunk_id],
            )
        )
        if len(entities) >= 12:
            break
    return entities


def choose_relation_pair(
    entities: list[ChunkExtractionEntity],
    relation_type: str,
) -> tuple[ChunkExtractionEntity, ChunkExtractionEntity] | None:
    """根据关系类型，为规则抽取选择更合理的实体对。"""

    if len(entities) < 2:
        return None

    def pick_by_type(target_type: str) -> tuple[ChunkExtractionEntity | None, ChunkExtractionEntity | None]:
        return pick_by_types({target_type})

    def pick_by_types(target_types: set[str]) -> tuple[ChunkExtractionEntity | None, ChunkExtractionEntity | None]:
        source_candidate: ChunkExtractionEntity | None = None
        target_candidate: ChunkExtractionEntity | None = None
        for item in entities:
            if item.entity_type in target_types and target_candidate is None:
                target_candidate = item
            elif source_candidate is None:
                source_candidate = item
        if source_candidate is None or target_candidate is None or source_candidate.entity_id == target_candidate.entity_id:
            return None, None
        return source_candidate, target_candidate

    if relation_type == "uses_method":
        source_item, target_item = pick_by_types({"Method", "ModelArchitecture"})
        if source_item and target_item:
            return source_item, target_item
    if relation_type == "uses_modality":
        source_item, target_item = pick_by_type("Modality")
        if source_item and target_item:
            return source_item, target_item
    if relation_type == "contains_module":
        source_item, target_item = pick_by_type("Module")
        if source_item and target_item:
            return source_item, target_item
    if relation_type == "uses_loss":
        source_item, target_item = pick_by_type("LossFunction")
        if source_item and target_item:
            return source_item, target_item
    if relation_type == "uses_degradation_model":
        source_item, target_item = pick_by_type("DegradationModel")
        if source_item and target_item:
            return source_item, target_item
    if relation_type == "uses_prior":
        source_item, target_item = pick_by_type("Prior")
        if source_item and target_item:
            return source_item, target_item
    if relation_type == "addresses_task":
        source_item, target_item = pick_by_types({"Task", "ResearchProblem"})
        if source_item and target_item:
            return source_item, target_item
    if relation_type == "evaluates_on":
        source_item, target_item = pick_by_type("Dataset")
        if source_item and target_item:
            return source_item, target_item
    if relation_type == "reports_metric":
        source_item, target_item = pick_by_type("Metric")
        if source_item and target_item:
            return source_item, target_item
    return entities[0], entities[1]


def extract_relations_rule(
    chunk_id: str,
    text: str,
    entities: list[ChunkExtractionEntity],
) -> list[ChunkExtractionRelation]:
    """使用规则方式从 chunk 中提取保守关系。"""

    entity_lookup = {item.name: item for item in entities}
    relations: list[ChunkExtractionRelation] = []
    seen_relation_keys: set[tuple[str, str, str, str]] = set()

    for sentence in split_into_sentences(text):
        sentence_entities = [
            entity_lookup[name]
            for name in entity_lookup
            if name in sentence
        ]
        if len(sentence_entities) < 2:
            continue

        relation_type = infer_relation_type(sentence)
        pair = choose_relation_pair(sentence_entities, relation_type)
        if pair is None:
            continue
        source_entity, target_entity = pair
        relation_key = (
            source_entity.entity_id,
            target_entity.entity_id,
            relation_type,
            sentence,
        )
        if relation_key in seen_relation_keys:
            continue
        seen_relation_keys.add(relation_key)
        relations.append(
            ChunkExtractionRelation(
                relation_id=build_relation_id(chunk_id, len(relations) + 1),
                source=source_entity.entity_id,
                target=target_entity.entity_id,
                relation_type=relation_type,
                evidence_text=truncate_text(sentence, max_chars=160),
                source_chunk_id=chunk_id,
            )
        )
        if len(relations) >= 6:
            break
    return relations


def sanitize_keywords(keywords: list[Any], text: str) -> list[str]:
    """规范化关键词列表。"""

    normalized_keywords: list[str] = []
    seen_keywords: set[str] = set()
    fallback_keywords = extract_search_terms(text)[:8]
    raw_keywords = keywords or fallback_keywords
    for item in raw_keywords:
        keyword = re.sub(r"\s+", " ", str(item)).strip()
        if not keyword:
            continue
        lowered_keyword = keyword.lower()
        if lowered_keyword in seen_keywords:
            continue
        seen_keywords.add(lowered_keyword)
        normalized_keywords.append(keyword)
        if len(normalized_keywords) >= 10:
            break
    return normalized_keywords


def sanitize_claims(claims: list[Any]) -> list[str]:
    """规范化 claim 列表。"""

    normalized_claims: list[str] = []
    seen_claims: set[str] = set()
    for item in claims or []:
        claim_text = truncate_text(str(item), max_chars=200)
        if not claim_text:
            continue
        lowered_claim = claim_text.lower()
        if lowered_claim in seen_claims:
            continue
        seen_claims.add(lowered_claim)
        normalized_claims.append(claim_text)
        if len(normalized_claims) >= 8:
            break
    return normalized_claims


def sanitize_entities(
    chunk_id: str,
    raw_entities: list[Any],
) -> list[ChunkExtractionEntity]:
    """校验并规范化模型输出实体。"""

    entities: list[ChunkExtractionEntity] = []
    seen_entity_names: set[str] = set()
    for index, item in enumerate(raw_entities or [], start=1):
        if not isinstance(item, dict):
            continue
        name = re.sub(r"\s+", " ", str(item.get("name", ""))).strip()
        normalized_name = normalize_entity_name(str(item.get("normalized_name", "")) or name)
        if not name or not normalized_name:
            continue
        if normalized_name in seen_entity_names:
            continue
        seen_entity_names.add(normalized_name)
        entity_type = str(item.get("entity_type", "Other")).strip() or "Other"
        if entity_type not in GRAPH_ENTITY_TYPES:
            entity_type = infer_entity_type(name, str(item.get("description", "")))
        entities.append(
            ChunkExtractionEntity(
                entity_id=str(item.get("entity_id", "")).strip() or build_entity_id(chunk_id, index),
                name=name,
                normalized_name=normalized_name,
                entity_type=entity_type,
                description=truncate_text(str(item.get("description", "")), max_chars=180),
                source_chunk_ids=[chunk_id],
            )
        )
        if len(entities) >= 12:
            break
    return entities


def sanitize_relations(
    chunk_id: str,
    raw_relations: list[Any],
    entities: list[ChunkExtractionEntity],
) -> list[ChunkExtractionRelation]:
    """校验并规范化模型输出关系。"""

    entity_id_map = {item.entity_id: item for item in entities}
    entity_name_map = {
        normalize_entity_name(item.name): item.entity_id for item in entities
    }
    relations: list[ChunkExtractionRelation] = []
    seen_relation_keys: set[tuple[str, str, str, str]] = set()

    for index, item in enumerate(raw_relations or [], start=1):
        if not isinstance(item, dict):
            continue

        raw_source = str(item.get("source", "")).strip()
        raw_target = str(item.get("target", "")).strip()
        source_id = raw_source if raw_source in entity_id_map else entity_name_map.get(normalize_entity_name(raw_source), "")
        target_id = raw_target if raw_target in entity_id_map else entity_name_map.get(normalize_entity_name(raw_target), "")
        if not source_id or not target_id or source_id == target_id:
            continue

        relation_type = str(item.get("relation_type", "related_to")).strip() or "related_to"
        if relation_type not in GRAPH_RELATION_TYPES:
            relation_type = "related_to"

        evidence_text = truncate_text(str(item.get("evidence_text", "")), max_chars=180)
        if not evidence_text:
            continue

        relation_key = (source_id, target_id, relation_type, evidence_text)
        if relation_key in seen_relation_keys:
            continue
        seen_relation_keys.add(relation_key)
        relations.append(
            ChunkExtractionRelation(
                relation_id=str(item.get("relation_id", "")).strip() or build_relation_id(chunk_id, index),
                source=source_id,
                target=target_id,
                relation_type=relation_type,
                evidence_text=evidence_text,
                source_chunk_id=chunk_id,
            )
        )
        if len(relations) >= 10:
            break
    return relations


def build_graph_extraction_messages(chunk_id: str, text: str) -> list[dict[str, str]]:
    """构建 GraphRAG 结构化抽取提示词。"""

    system_prompt = (
        "你是论文图谱抽取助手。"
        "你的任务是仅基于当前 chunk 文本，保守地抽取实体、实体关系、claims 和 keywords。"
        "不要猜测 chunk 中没有明确出现的信息。"
        "关系必须能够被当前 chunk 的原文短证据支撑。"
        "输出必须是严格 JSON 对象。"
    )
    user_prompt = (
        "请对以下论文 chunk 做结构化抽取，并严格输出 JSON 对象。\n"
        "字段结构必须为：\n"
        "{\n"
        '  "entities": [\n'
        "    {\n"
        '      "entity_id": "chunk_local_id",\n'
        '      "name": "实体原文名",\n'
        '      "normalized_name": "归一化后的名称",\n'
        '      "entity_type": "ResearchProblem|Task|Modality|Method|ModelArchitecture|Module|LossFunction|DegradationModel|Prior|Dataset|Metric|Baseline|Finding|Limitation|Institution|TimePeriod|Concept|Other",\n'
        '      "description": "一句保守描述"\n'
        "    }\n"
        "  ],\n"
        '  "relations": [\n'
        "    {\n"
        '      "relation_id": "chunk_relation_id",\n'
        '      "source": "必须引用 entities 中的 entity_id 或实体名",\n'
        '      "target": "必须引用 entities 中的 entity_id 或实体名",\n'
        '      "relation_type": "addresses_task|uses_modality|uses_method|contains_module|uses_loss|uses_degradation_model|uses_prior|evaluates_on|reports_metric|compares_with|improves|preserves_spectral_quality|enhances_spatial_detail|limited_by|finds|mentions|related_to",\n'
        '      "evidence_text": "来自当前 chunk 的短证据"\n'
        "    }\n"
        "  ],\n"
        '  "claims": ["可选，保守抽取的论断"],\n'
        '  "keywords": ["可选，最多 10 个关键词"]\n'
        "}\n"
        "规则：\n"
        "1. 实体尽量保守，不要超过 12 个。\n"
        "2. 关系必须能在 chunk 中直接找到证据，不确定就不要输出。\n"
        "3. 若没有明确关系，可输出空数组。\n"
        "4. 实体名称保留原文写法，但 normalized_name 需要是便于去重的归一化文本。\n"
        "5. 优先识别全色锐化/高光谱全色锐化论文中的任务、模态、方法、网络模块、损失函数、退化模型、先验、数据集、指标、baseline、结论和局限。\n"
        f"当前 chunk_id：{chunk_id}\n"
        f"当前 chunk 文本：\n{text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def extract_chunk_graph_data_with_rule(
    chunk_id: str,
    text: str,
    error_message: str = "",
) -> ChunkExtractionRecord:
    """基于规则构造单个 chunk 的图谱抽取结果。"""

    entities = extract_candidate_entities_rule(chunk_id, text)
    relations = extract_relations_rule(chunk_id, text, entities)
    claims = [truncate_text(sentence, max_chars=180) for sentence in split_into_sentences(text)[:3]]
    return ChunkExtractionRecord(
        chunk_id=chunk_id,
        text_hash=build_chunk_text_hash(text),
        entities=entities,
        relations=relations,
        claims=sanitize_claims(claims),
        keywords=sanitize_keywords([], text),
        extraction_method="rule",
        error_message=error_message,
    )


def extract_chunk_graph_data(
    *,
    chunk_id: str,
    text: str,
    client: DashScopeClient | None,
    model_name: str,
) -> ChunkExtractionRecord:
    """从单个 chunk 中抽取图谱实体、关系、claims 与关键词。"""

    if not text.strip():
        return ChunkExtractionRecord(
            chunk_id=chunk_id,
            text_hash=build_chunk_text_hash(text),
            extraction_method="empty",
        )

    if client is None or not model_name.strip():
        return extract_chunk_graph_data_with_rule(chunk_id, text)

    try:
        raw_text = client.chat(
            model=model_name,
            messages=build_graph_extraction_messages(chunk_id, text),
            temperature=0.1,
            max_tokens=1600,
            response_format={"type": "json_object"},
        )
        payload = parse_first_json_object(raw_text)
        if payload is None:
            return extract_chunk_graph_data_with_rule(
                chunk_id,
                text,
                error_message="大模型抽取输出非 JSON，已回退规则抽取。",
            )

        entities = sanitize_entities(chunk_id, payload.get("entities", []))
        relations = sanitize_relations(chunk_id, payload.get("relations", []), entities)
        claims = sanitize_claims(payload.get("claims", []))
        keywords = sanitize_keywords(payload.get("keywords", []), text)

        return ChunkExtractionRecord(
            chunk_id=chunk_id,
            text_hash=build_chunk_text_hash(text),
            entities=entities,
            relations=relations,
            claims=claims,
            keywords=keywords,
            extraction_method="llm",
        )
    except Exception as exc:
        return extract_chunk_graph_data_with_rule(
            chunk_id,
            text,
            error_message=f"大模型抽取失败，已回退规则抽取：{exc}",
        )
