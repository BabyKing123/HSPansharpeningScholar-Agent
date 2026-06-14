"""本模块作用：在整个智能体中负责对多篇论文做最小结构化比较，帮助系统完成“多工具调用”中的论文对比任务。"""

import re
from collections import Counter
from dataclasses import dataclass, field

from tools.analyze_tool import StructuredPaperAnalysis, analyze_single_paper, is_weak_field_value


COMPARE_STOP_WORDS = {
    "研究",
    "方法",
    "结果",
    "数据",
    "问题",
    "对象",
    "分析",
    "影响",
    "表明",
    "指出",
    "采用",
    "基于",
    "本文",
    "论文",
    "以及",
    "进行",
    "不同",
    "相关",
    "模型",
    "图像",
    "影像",
    "融合",
    "数据集",
    "实验",
    "性能",
    "网络",
    "框架",
    "任务",
    "提出",
    "文章",
    "对于",
    "and",
    "the",
    "this",
    "that",
    "these",
    "those",
    "into",
    "from",
    "have",
    "been",
    "also",
    "more",
    "such",
    "with",
    "using",
    "based",
    "method",
    "methods",
    "result",
    "results",
    "model",
    "models",
    "paper",
    "papers",
    "study",
    "studies",
    "image",
    "images",
    "data",
    "dataset",
    "datasets",
    "fusion",
    "task",
    "tasks",
    "proposed",
    "approach",
    "approaches",
    "network",
    "networks",
    "framework",
    "performance",
    "experiment",
    "experiments",
    "table",
    "figure",
    "abstract",
    "introduction",
    "conclusion",
    "hyperspectral",
    "remote",
    "sensing",
    "pansharpening",
    "panchromatic",
    "spatial",
    "spectral",
    "are",
    "was",
    "were",
    "their",
    "them",
    "both",
    "between",
    "across",
    "through",
    "moreover",
    "extract",
    "features",
    "feature",
    "capability",
    "superior",
    "optimal",
    "quality",
    "resolution",
    "multispectral",
    "hyperspectral",
    "panchromatic",
    "spatial",
    "spectral",
    "remote",
    "sensing",
}


@dataclass
class ComparisonPaperSummary:
    """本数据结构作用：保存单篇论文在比较任务中的精简摘要。"""

    file_name: str
    document_id: str
    research_question: str
    research_object: str
    methods: str
    data_source: str
    key_findings: str
    limitations: str
    implications: str


@dataclass
class MultiPaperComparison:
    """本数据结构作用：保存多篇论文比较任务的结构化结果。"""

    topic_hint: str
    paper_summaries: list[ComparisonPaperSummary] = field(default_factory=list)
    common_themes: list[str] = field(default_factory=list)
    task_type_comparison: list[str] = field(default_factory=list)
    modality_comparison: list[str] = field(default_factory=list)
    method_comparison: list[str] = field(default_factory=list)
    spatial_modeling_comparison: list[str] = field(default_factory=list)
    spectral_modeling_comparison: list[str] = field(default_factory=list)
    prior_or_degradation_modeling: list[str] = field(default_factory=list)
    dataset_comparison: list[str] = field(default_factory=list)
    metric_comparison: list[str] = field(default_factory=list)
    strengths_and_limitations: list[str] = field(default_factory=list)
    relevance_to_user_research: list[str] = field(default_factory=list)
    data_comparison: list[str] = field(default_factory=list)
    finding_comparison: list[str] = field(default_factory=list)
    integrated_implications: list[str] = field(default_factory=list)


def normalize_compare_text(text: str) -> str:
    """对比较任务中的文本做基础清洗。

    输入：
        text: 原始文本。
    输出：
        统一空白后的文本。
    异常：
        无。
    """

    cleaned_text = re.sub(r"\s+", " ", text).strip()
    return cleaned_text


def truncate_compare_text(text: str, max_length: int = 360) -> str:
    """将比较结果中的长文本裁剪为适合展示的长度。

    输入：
        text: 原始文本。
        max_length: 最大保留字符数。
    输出：
        裁剪后的文本。
    异常：
        无。
    """

    cleaned_text = normalize_compare_text(text)
    if len(cleaned_text) <= max_length:
        return cleaned_text
    return cleaned_text[:max_length].rstrip() + "..."


def is_unhelpful_compare_text(text: str) -> bool:
    """判断比较字段是否不适合直接展示。"""

    cleaned_text = normalize_compare_text(text)
    if not cleaned_text:
        return True
    if is_weak_field_value(cleaned_text):
        return True
    return any(
        marker in cleaned_text
        for marker in [
            "未稳定识别",
            "建议查看",
            "人工补充",
            "未能稳定",
        ]
    )


def display_field_or_unclear(text: str, max_length: int = 300, default_text: str = "原文未明确说明") -> str:
    """返回适合比较输出的字段文本。"""

    if is_unhelpful_compare_text(text):
        return default_text
    return truncate_compare_text(text, max_length=max_length)


def extract_compare_keywords(text: str) -> list[str]:
    """从比较文本中提取最小关键词。

    输入：
        text: 原始文本。
    输出：
        去重后的关键词列表。
    异常：
        无。
    """

    normalized_text = normalize_compare_text(text).lower()
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", normalized_text)
    english_terms = re.findall(r"[a-z]{3,}", normalized_text)

    keywords: list[str] = []
    seen_terms: set[str] = set()
    for term in chinese_terms + english_terms:
        if term in COMPARE_STOP_WORDS or term in seen_terms:
            continue
        seen_terms.add(term)
        keywords.append(term)
    return keywords


def build_paper_summary(analysis: StructuredPaperAnalysis) -> ComparisonPaperSummary:
    """将单篇论文分析结果转换为多篇比较所需的摘要结构。

    输入：
        analysis: 单篇论文结构化分析结果。
    输出：
        适用于比较任务的单篇摘要对象。
    异常：
        无。
    """

    return ComparisonPaperSummary(
        file_name=analysis.file_name,
        document_id=analysis.document_id,
        research_question=analysis.research_question,
        research_object=analysis.research_object,
        methods=analysis.methods,
        data_source=analysis.data_source,
        key_findings=analysis.key_findings,
        limitations=analysis.limitations,
        implications=analysis.implications,
    )


def collect_common_themes(analyses: list[StructuredPaperAnalysis], max_items: int = 4) -> list[str]:
    """从多篇论文中提取共同主题。

    输入：
        analyses: 多篇论文的结构化分析结果。
        max_items: 最多返回多少条共同主题。
    输出：
        共同主题列表。
    异常：
        无。
    """

    task_labels: list[str] = []
    for analysis in analyses:
        merged_text = merge_analysis_fields(analysis)
        if re.search(r"hsi-msi|高光谱.*多光谱", merged_text):
            task_labels.append("高光谱-多光谱融合")
        elif re.search(r"hyperspectral pansharpen|hsi pansharpen|高光谱全色锐化", merged_text):
            task_labels.append("高光谱全色锐化")
        elif re.search(r"pansharpen|pan-sharpen|全色锐化", merged_text):
            task_labels.append("多光谱/遥感全色锐化")
        elif re.search(r"super-resolution|超分", merged_text):
            task_labels.append("高光谱超分辨率")

    task_counter = Counter(task_labels)
    if task_counter:
        ordered_tasks = [task for task, _ in task_counter.most_common()]
        theme_items = [
            f"选中论文覆盖{ '、'.join(ordered_tasks) }等相关任务。",
            "共同核心问题是提升空间分辨率，同时尽量保持光谱或辐射一致性。",
        ]
        if len(ordered_tasks) > 1:
            theme_items.append("这些论文适合用于比较不同融合任务之间的方法迁移关系。")
        return theme_items[:max_items]

    counter: Counter[str] = Counter()
    for analysis in analyses:
        merged_text = " ".join(
            [
                analysis.research_question,
                analysis.research_object,
                analysis.key_findings,
                analysis.implications,
            ]
        )
        counter.update(extract_compare_keywords(merged_text))

    common_terms = [term for term, count in counter.items() if count >= 2 and len(term) >= 4][:max_items]
    if not common_terms:
        return ["当前几篇论文的共同主题不够集中，更适合按方法、对象和发现分别比较。"]

    dataset_terms = {"cave", "harvard", "pavia", "chikusei", "botswana", "worldview", "gaofen", "quickbird"}
    metric_terms = {"psnr", "ssim", "sam", "ergas", "rmse", "qnr", "scc", "q2n"}
    theme_items: list[str] = []
    for term in common_terms:
        if term in dataset_terms:
            theme_items.append(f"多篇论文共同使用或比较 {term.upper() if len(term) <= 4 else term.title()} 数据集。")
        elif term in metric_terms:
            theme_items.append(f"多篇论文共同报告 {term.upper()} 等评价指标。")
        else:
            theme_items.append(f"多篇论文都涉及“{term}”相关主题。")
    return theme_items


def build_method_comparison(analyses: list[StructuredPaperAnalysis]) -> list[str]:
    """生成多篇论文的方法对比结果。

    输入：
        analyses: 多篇论文的结构化分析结果。
    输出：
        方法对比文本列表。
    异常：
        无。
    """

    return [
        f"《{item.file_name}》：{display_field_or_unclear(item.methods, default_text='原文未明确说明方法结构')}"
        for item in analyses
    ]


def build_data_comparison(analyses: list[StructuredPaperAnalysis]) -> list[str]:
    """生成多篇论文的数据来源对比结果。

    输入：
        analyses: 多篇论文的结构化分析结果。
    输出：
        数据来源对比文本列表。
    异常：
        无。
    """

    return [
        f"《{item.file_name}》：{display_field_or_unclear(item.data_source, default_text='原文未明确说明数据集或实验设置')}"
        for item in analyses
    ]


def build_finding_comparison(analyses: list[StructuredPaperAnalysis]) -> list[str]:
    """生成多篇论文的主要发现对比结果。

    输入：
        analyses: 多篇论文的结构化分析结果。
    输出：
        主要发现对比文本列表。
    异常：
        无。
    """

    return [
        f"《{item.file_name}》：{display_field_or_unclear(item.key_findings, default_text='原文未明确说明主要结论')}"
        for item in analyses
    ]


def build_integrated_implications(analyses: list[StructuredPaperAnalysis], max_items: int = 4) -> list[str]:
    """汇总多篇论文对全色锐化任务的综合启示。

    输入：
        analyses: 多篇论文的结构化分析结果。
        max_items: 最多返回多少条综合启示。
    输出：
        综合启示列表。
    异常：
        无。
    """

    implication_items: list[str] = []
    seen_items: set[str] = set()
    for analysis in analyses:
        if is_unhelpful_compare_text(analysis.implications):
            continue
        candidate = truncate_compare_text(analysis.implications, max_length=150).rstrip("。；;,.，")
        if candidate in seen_items:
            continue
        seen_items.add(candidate)
        implication_items.append(f"《{analysis.file_name}》提示：{candidate}。")
        if len(implication_items) >= max_items:
            break

    if implication_items:
        return implication_items
    return ["当前未能稳定归纳出综合启示，建议回看单篇分析中的“启示”字段。"]


def merge_analysis_fields(analysis: StructuredPaperAnalysis) -> str:
    """合并单篇分析字段，供规则比较做轻量领域识别。"""

    return " ".join(
        [
            analysis.file_name,
            analysis.research_question,
            analysis.research_object,
            analysis.methods,
            analysis.data_source,
            analysis.key_findings,
            analysis.implications,
        ]
    ).lower()


def merge_analysis_fields_original(analysis: StructuredPaperAnalysis) -> str:
    """合并单篇分析字段并保留原始大小写，供数据集和指标展示使用。"""

    return " ".join(
        [
            analysis.file_name,
            analysis.research_question,
            analysis.research_object,
            analysis.methods,
            analysis.data_source,
            analysis.key_findings,
            analysis.implications,
        ]
    )


def build_task_type_comparison(analyses: list[StructuredPaperAnalysis]) -> list[str]:
    """根据规则分析结果补充任务类型比较。"""

    items: list[str] = []
    for analysis in analyses:
        merged_text = merge_analysis_fields(analysis)
        if re.search(r"hyperspectral pansharpen|hsi pansharpen|高光谱全色锐化", merged_text):
            task_type = "HSI pansharpening / 高光谱全色锐化"
        elif re.search(r"hsi-msi|hyperspectral and multispectral|高光谱.*多光谱|多光谱.*高光谱", merged_text):
            task_type = "HSI-MSI fusion / 高光谱-多光谱融合"
        elif re.search(r"super-resolution|super resolution|高光谱.*超分|hsi super", merged_text):
            task_type = "HSI super-resolution / 高光谱超分辨率"
        elif re.search(r"latent.*restoration|latent hsi|潜空间.*复原|潜空间.*恢复", merged_text):
            task_type = "latent HSI restoration / 潜空间高光谱复原"
        elif re.search(r"pansharpen|pan-sharpen|全色锐化", merged_text):
            task_type = "MSI pansharpening 或通用 pansharpening"
        elif re.search(r"image fusion|multimodal fusion|图像融合|遥感图像融合", merged_text):
            task_type = "general image fusion / 遥感图像融合"
        else:
            task_type = truncate_compare_text(analysis.research_question, max_length=220)
        items.append(f"《{analysis.file_name}》主要对应{task_type}。")
    return items


def build_modality_comparison(analyses: list[StructuredPaperAnalysis]) -> list[str]:
    """根据文本线索补充输入模态比较。"""

    items: list[str] = []
    for analysis in analyses:
        merged_text = merge_analysis_fields(analysis)
        modalities: list[str] = []
        if re.search(r"hyperspectral.*pansharpen|hsi.*pansharpen|高光谱全色锐化", merged_text):
            modalities.extend(["PAN/全色图像", "LRHS/低分辨率高光谱图像", "HSI/高光谱图像"])
        elif re.search(r"pansharpen|pan-sharpen|全色锐化", merged_text):
            modalities.extend(["PAN/全色图像", "MSI/多光谱图像"])
        if re.search(r"\bpan\b|panchromatic|全色", merged_text):
            modalities.append("PAN/全色图像")
        if re.search(r"\blrhs\b|low-resolution hyperspectral|低分辨率高光谱", merged_text):
            modalities.append("LRHS/低分辨率高光谱图像")
        if re.search(r"\bhsi\b|hyperspectral|高光谱", merged_text):
            modalities.append("HSI/高光谱图像")
        if re.search(r"\bmsi\b|multispectral|多光谱", merged_text):
            modalities.append("MSI/多光谱图像")
        if re.search(r"\bhrhs\b|high-resolution hyperspectral|高分辨率高光谱", merged_text):
            modalities.append("HRHS/高空间分辨率高光谱图像")
        if re.search(r"\blms\b|low-resolution multispectral|低分辨率多光谱", merged_text):
            modalities.append("LMS/低分辨率多光谱图像")
        if re.search(r"\bms\b|multispectral", merged_text) and "MSI/多光谱图像" not in modalities:
            modalities.append("MS/多光谱图像")
        unique_modalities = list(dict.fromkeys(modalities))
        modality_text = "、".join(unique_modalities) if unique_modalities else "原文未明确说明"
        items.append(f"《{analysis.file_name}》涉及的输入模态：{modality_text}。")
    return items


def build_spatial_modeling_comparison(analyses: list[StructuredPaperAnalysis]) -> list[str]:
    """根据规则分析结果补充空间信息建模比较。"""

    items: list[str] = []
    for analysis in analyses:
        merged_text = merge_analysis_fields(analysis)
        method_text = display_field_or_unclear(analysis.methods, max_length=220, default_text="")
        cues: list[str] = []
        if re.search(r"\bpan\b|panchromatic|全色", merged_text):
            cues.append("利用 PAN/全色图像提供高空间分辨率细节")
        if re.search(r"high-frequency|frequency|wavelet|高频|频域|小波", merged_text):
            cues.append("通过高频或频域分量强化纹理与边缘")
        if re.search(r"transformer|attention|注意力", merged_text):
            cues.append("用 Transformer/注意力建模长程空间依赖或跨模态特征交互")
        if re.search(r"convolution|cnn|卷积|encoder|decoder|编码|解码", merged_text):
            cues.append("用卷积或编码-解码结构提取局部空间结构")
        if method_text and (cues or re.search(r"spatial|空间|纹理|细节|结构", merged_text)):
            cue_text = "；".join(cues) if cues else "围绕空间结构恢复建模"
            items.append(f"《{analysis.file_name}》空间建模：{cue_text}。方法线索：{method_text}")
        else:
            items.append(f"《{analysis.file_name}》未稳定提取出明确的空间信息建模描述。")
    return items


def build_spectral_modeling_comparison(analyses: list[StructuredPaperAnalysis]) -> list[str]:
    """根据规则分析结果补充光谱信息建模比较。"""

    items: list[str] = []
    for analysis in analyses:
        merged_text = merge_analysis_fields(analysis)
        method_text = display_field_or_unclear(analysis.methods, max_length=220, default_text="")
        cues: list[str] = []
        if re.search(r"\bhsi\b|hyperspectral|lrhs|高光谱", merged_text):
            cues.append("以 HSI/LRHS 的光谱信息约束融合结果")
        if re.search(r"spectral fidelity|spectral consistency|spectral distortion|sam|光谱保真|光谱一致|光谱失真", merged_text):
            cues.append("关注光谱保真、光谱一致性或 SAM 等失真约束")
        if re.search(r"tensor|low-rank|decomposition|张量|低秩|分解", merged_text):
            cues.append("用张量/低秩结构表达空间-光谱相关性")
        if re.search(r"vae|autoencoder|latent|diffusion|潜空间|自编码|扩散", merged_text):
            cues.append("在潜空间或生成模型中维持光谱表达")
        if method_text and (cues or re.search(r"spectral|光谱|保真|一致性", merged_text)):
            cue_text = "；".join(cues) if cues else "围绕光谱信息保持建模"
            items.append(f"《{analysis.file_name}》光谱建模：{cue_text}。方法线索：{method_text}")
        else:
            items.append(f"《{analysis.file_name}》未稳定提取出明确的光谱信息建模描述。")
    return items


def build_prior_or_degradation_comparison(analyses: list[StructuredPaperAnalysis]) -> list[str]:
    """根据文本线索补充注意力、门控、先验和退化模型比较。"""

    items: list[str] = []
    for analysis in analyses:
        merged_text = merge_analysis_fields(analysis)
        matched_terms: list[str] = []
        for pattern, label in [
            (r"attention|注意力", "注意力机制"),
            (r"gating|gate|门控", "门控机制"),
            (r"prior|先验", "先验约束"),
            (r"degradation|退化|blur|downsampling|psf|srf", "退化建模"),
            (r"tensor|low-rank|decomposition|张量|低秩|分解", "张量/低秩先验"),
            (r"implicit neural representation|\binr\b|neural operator|隐式", "隐式表示/神经算子先验"),
            (r"bayesian|variational|optimization|模型驱动|优化", "模型驱动优化"),
            (r"zero-shot|zero shot|零样本", "zero-shot 建模"),
            (r"latent|潜空间", "latent space 表示"),
            (r"vae|autoencoder|自编码", "VAE/autoencoder 表示学习"),
            (r"u-net|unet", "U-Net 层级结构"),
            (r"arconv|adaptive rectangular convolution", "ARConv 或自适应卷积"),
            (r"diffusion|扩散", "扩散模型"),
        ]:
            if re.search(pattern, merged_text):
                matched_terms.append(label)
        if matched_terms:
            items.append(f"《{analysis.file_name}》涉及：{'、'.join(matched_terms)}。")
        else:
            items.append(f"《{analysis.file_name}》原文未明确说明注意力、门控、先验或退化建模。")
    return items


def build_metric_comparison(analyses: list[StructuredPaperAnalysis]) -> list[str]:
    """从主要结论和数据字段中识别常见全色锐化评价指标。"""

    items: list[str] = []
    for analysis in analyses:
        merged_text = merge_analysis_fields(analysis)
        metrics = re.findall(r"\b(?:PSNR|SSIM|SAM|ERGAS|RMSE|QNR|D_LAMBDA|D_S|Q2N|CC|SCC)\b", merged_text, flags=re.IGNORECASE)
        unique_metrics = []
        for metric in metrics:
            normalized_metric = metric.upper().replace("D_LAMBDA", "D_lambda").replace("D_S", "D_s").replace("Q2N", "Q2n")
            if normalized_metric not in unique_metrics:
                unique_metrics.append(normalized_metric)
        metric_text = "、".join(unique_metrics) if unique_metrics else "原文未明确说明"
        items.append(f"《{analysis.file_name}》涉及的评价指标：{metric_text}。")
    return items


def build_dataset_comparison(analyses: list[StructuredPaperAnalysis]) -> list[str]:
    """从数据字段中识别常见数据集和实验协议。"""

    dataset_pattern = re.compile(
        r"\b(?:CAVE|Harvard|Pavia(?: Center| University)?|Washington DC Mall|Botswana|Chikusei|WorldView-2|WorldView-3|WorldView|GaoFen-2|GaoFen|QuickBird|IKONOS|AVIRIS|ROSIS|Moffett|Salinas|Wald protocol|reduced-resolution|full-resolution)\b",
        flags=re.IGNORECASE,
    )
    items: list[str] = []
    for analysis in analyses:
        merged_text = merge_analysis_fields_original(analysis)
        matches = dataset_pattern.findall(merged_text)
        unique_items: list[str] = []
        for item in matches:
            normalized_item = item.strip()
            if normalized_item.lower() == "wald protocol":
                normalized_item = "Wald protocol"
            if normalized_item.lower() == "reduced-resolution":
                normalized_item = "reduced-resolution"
            if normalized_item.lower() == "full-resolution":
                normalized_item = "full-resolution"
            if normalized_item not in unique_items:
                unique_items.append(normalized_item)
        if unique_items:
            dataset_text = "、".join(unique_items)
        else:
            dataset_text = display_field_or_unclear(
                analysis.data_source,
                max_length=260,
                default_text="原文未明确说明",
            )
        items.append(f"《{analysis.file_name}》数据集/实验协议：{dataset_text}。")
    return items


def build_strengths_and_limitations(analyses: list[StructuredPaperAnalysis]) -> list[str]:
    """生成规则 fallback 下的优势与局限比较。"""

    return [
        (
            f"《{analysis.file_name}》主要结论："
            f"{display_field_or_unclear(analysis.key_findings, max_length=260, default_text='原文未明确说明')}"
            f"；局限性："
            f"{display_field_or_unclear(analysis.limitations, max_length=220, default_text='原文未明确说明')}"
        )
        for analysis in analyses
    ]


def compare_papers(
    papers: list[dict[str, str]],
    topic_hint: str = "未指定比较主题",
) -> MultiPaperComparison:
    """对多篇论文执行最小结构化比较。

    输入：
        papers: 待比较论文列表，每项至少包含 file_name、document_id 和 full_text。
        topic_hint: 可选的比较主题提示。
    输出：
        多篇论文比较结果对象。
    异常：
        当论文数量不足两篇时，抛出 ValueError。
        当论文文本为空时，可能抛出 ValueError。
    """

    if len(papers) < 2:
        raise ValueError("多篇比较至少需要 2 篇论文。")

    analyses: list[StructuredPaperAnalysis] = []
    summaries: list[ComparisonPaperSummary] = []
    for paper in papers:
        analysis = analyze_single_paper(
            text_or_segments=paper["full_text"],
            file_name=paper["file_name"],
            document_id=paper["document_id"],
        )
        analyses.append(analysis)
        summaries.append(build_paper_summary(analysis))

    data_comparison = build_data_comparison(analyses)
    dataset_comparison = build_dataset_comparison(analyses)
    integrated_implications = build_integrated_implications(analyses)

    return MultiPaperComparison(
        topic_hint=topic_hint,
        paper_summaries=summaries,
        common_themes=collect_common_themes(analyses),
        task_type_comparison=build_task_type_comparison(analyses),
        modality_comparison=build_modality_comparison(analyses),
        method_comparison=build_method_comparison(analyses),
        spatial_modeling_comparison=build_spatial_modeling_comparison(analyses),
        spectral_modeling_comparison=build_spectral_modeling_comparison(analyses),
        prior_or_degradation_modeling=build_prior_or_degradation_comparison(analyses),
        dataset_comparison=dataset_comparison,
        metric_comparison=build_metric_comparison(analyses),
        strengths_and_limitations=build_strengths_and_limitations(analyses),
        relevance_to_user_research=integrated_implications,
        data_comparison=data_comparison,
        finding_comparison=build_finding_comparison(analyses),
        integrated_implications=integrated_implications,
    )


def format_comparison_result(result: MultiPaperComparison) -> str:
    """将多篇论文比较结果整理为适合命令行展示的文本。

    输入：
        result: 多篇论文比较结果对象。
    输出：
        可直接打印的比较结果文本。
    异常：
        无。
    """

    def append_items(title: str, items: list[str], default_text: str = "未稳定提取出该维度。") -> None:
        lines.append(title)
        cleaned_items = [truncate_compare_text(item, max_length=700) for item in items if item and item.strip()]
        if not cleaned_items:
            lines.append(f"- {default_text}")
            return
        for item in cleaned_items:
            lines.append(f"- {item}")

    lines = [
        "多篇论文比较结果：",
        f"比较主题：{result.topic_hint}",
        f"纳入论文数量：{len(result.paper_summaries)}",
        "",
        "一、单篇论文概括：",
    ]

    for index, summary in enumerate(result.paper_summaries, start=1):
        lines.append(f"{index}. 《{summary.file_name}》")
        lines.append(f"   研究问题：{truncate_compare_text(summary.research_question, max_length=700)}")
        lines.append(f"   研究对象：{truncate_compare_text(summary.research_object, max_length=700)}")
        lines.append(f"   方法：{truncate_compare_text(summary.methods, max_length=700)}")
        lines.append(f"   主要结论：{truncate_compare_text(summary.key_findings, max_length=700)}")

    lines.append("")
    lines.append("二、面向遥感图像融合/全色锐化的比较：")
    append_items("任务类型比较：", result.task_type_comparison)
    append_items("输入模态比较：", result.modality_comparison)
    append_items("核心方法比较：", result.method_comparison)
    append_items("空间信息建模比较：", result.spatial_modeling_comparison)
    append_items("光谱信息建模比较：", result.spectral_modeling_comparison)
    append_items("退化模型/先验建模比较：", result.prior_or_degradation_modeling)
    append_items("数据集与实验协议比较：", result.dataset_comparison or result.data_comparison)
    append_items("指标与主要结论比较：", result.metric_comparison)

    lines.append("")
    lines.append("三、综合判断：")
    append_items("共同主题：", result.common_themes)
    append_items("优势与局限：", result.strengths_and_limitations or result.finding_comparison)
    append_items("对用户高光谱全色锐化研究的综合启示：", result.relevance_to_user_research or result.integrated_implications)
    append_items("可整合的研究思路：", result.integrated_implications)

    return "\n".join(lines)


def run_compare_demo() -> None:
    """执行 compare_tool 模块的最小演示。

    输入：
        无。
    输出：
        无。函数会直接打印多篇比较结果。
    异常：
        无。
    """

    demo_papers = [
        {
            "file_name": "hsi_pansharpening_attention.pdf",
            "document_id": "paper_a",
            "full_text": (
                "本文关注高光谱全色锐化方法，输入为 LRHS 与 PAN，目标是重建 HRHS。"
                "方法通过空间注意力增强 PAN 纹理注入，并使用光谱保持损失约束 HSI 光谱一致性。"
                "实验在 Pavia Center、CAVE 和 Harvard 数据集上验证，指标包括 PSNR、SSIM、SAM 和 ERGAS。"
                "结果表明该方法能够提升空间细节并减少光谱失真。"
                "论文建议后续探索 PAN/LRHS 条件门控和退化一致性约束。"
            ),
        },
        {
            "file_name": "hsi_msi_fusion_tensor_transformer.pdf",
            "document_id": "paper_b",
            "full_text": (
                "研究聚焦 HSI-MSI fusion，使用张量分解与 Transformer 建模空间-光谱相关性。"
                "输入包括低空间分辨率 HSI 和高空间分辨率 MSI，用于恢复高空间分辨率 HSI。"
                "实验在 CAVE、Harvard 和 Pavia University 数据集上进行，并比较 PSNR、SAM、ERGAS 与 RMSE。"
                "结果显示该框架在光谱保真和空间结构恢复上优于若干 baseline。"
                "局限是训练数据依赖较强，真实退化模型不匹配时泛化能力仍需验证。"
            ),
        },
    ]

    result = compare_papers(demo_papers, topic_hint="高光谱全色锐化与 HSI-MSI fusion 方法比较")
    print(format_comparison_result(result))


if __name__ == "__main__":
    run_compare_demo()
