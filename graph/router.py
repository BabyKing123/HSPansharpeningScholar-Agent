"""本模块作用：根据中文学术问答特征，为检索层选择更合适的 RAG 路由。"""

from __future__ import annotations

import re


GLOBAL_TRIGGER_TERMS = {
    "总体",
    "整体",
    "趋势",
    "方向",
    "综述",
    "共性",
    "差异",
    "主要问题",
    "主要方法",
    "总结",
    "现状",
    "格局",
    "演进",
    "脉络",
    "方法族",
    "任务类型",
    "数据集",
    "评价指标",
    "空间-光谱",
    "zero-shot",
    "diffusion",
}

LOCAL_TRIGGER_TERMS = {
    "什么是",
    "关系",
    "如何",
    "为什么",
    "作用",
    "影响",
    "机制",
    "区别",
    "方法",
    "指标",
    "模态",
    "PAN",
    "pan",
    "HSI",
    "hsi",
    "MSI",
    "msi",
    "LRHS",
    "lrhs",
    "HRHS",
    "hrhs",
    "注意力",
    "门控",
    "先验",
    "退化",
    "损失",
    "论文",
    "概念",
}

MIXED_TRIGGER_TERMS = {
    "代表性论文",
    "举例",
    "并请",
    "同时请",
    "结合论文",
    "具体应用",
    "具体方法",
    "在整体研究中的位置",
    "并说明",
    "我的模型",
    "用户模型",
    "可以借鉴",
    "启示",
    "baseline",
    "消融",
    "gate",
    "attention",
    "prior",
    "diffusion",
    "latent space",
    "zero-shot",
    "spatial-spectral fusion",
}


def contains_any_term(text: str, terms: set[str]) -> bool:
    """判断文本中是否包含任一触发词。"""

    return any(term in text for term in terms)


def count_trigger_terms(text: str, terms: set[str]) -> int:
    """统计文本命中的触发词数量。"""

    return sum(1 for term in terms if term in text)


def looks_like_global_summary_question(question: str) -> bool:
    """判断问题是否更像整体总结或趋势综述类问题。"""

    normalized_question = question.strip()
    if not normalized_question:
        return False
    if contains_any_term(normalized_question, GLOBAL_TRIGGER_TERMS):
        return True
    return bool(re.search(r"(研究|文献).*(总体|整体|趋势|方向|综述)", normalized_question))


def looks_like_local_relation_question(question: str) -> bool:
    """判断问题是否更像局部实体关系类问题。"""

    normalized_question = question.strip()
    if not normalized_question:
        return False
    if contains_any_term(normalized_question, LOCAL_TRIGGER_TERMS):
        return True
    return bool(re.search(r"(某|该|这个|这种).*(方法|指标|模态|数据集|概念|论文|模块|损失)", normalized_question))


def looks_like_mixed_question(question: str) -> bool:
    """判断问题是否同时包含全局与局部证据需求。"""

    normalized_question = question.strip()
    if not normalized_question:
        return False
    if contains_any_term(normalized_question, MIXED_TRIGGER_TERMS):
        return True
    has_global_intent = count_trigger_terms(normalized_question, GLOBAL_TRIGGER_TERMS) > 0
    has_local_intent = count_trigger_terms(normalized_question, LOCAL_TRIGGER_TERMS) > 0
    asks_for_examples = bool(re.search(r"(举例|案例|论文|证据|应用)", normalized_question))
    return has_global_intent and (has_local_intent or asks_for_examples)


def route_query_for_retrieval(question: str) -> str:
    """根据问题文本选择 hybrid_rag、graph_local、graph_global 或 graph_mixed。"""

    cleaned_question = question.strip()
    if not cleaned_question:
        return "hybrid_rag"

    if looks_like_mixed_question(cleaned_question):
        return "graph_mixed"
    if looks_like_global_summary_question(cleaned_question):
        return "graph_global"
    if looks_like_local_relation_question(cleaned_question):
        return "graph_local"
    return "hybrid_rag"
