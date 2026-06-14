"""本模块作用：在整个智能体中负责根据若干论文生成最小综述提纲，帮助系统完成“工具调用与模块化”阶段的提纲生成任务。"""

from dataclasses import dataclass, field

from tools.compare_tool import MultiPaperComparison, compare_papers


@dataclass
class ReviewOutlineSection:
    """本数据结构作用：保存综述提纲中的单个章节。"""

    title: str
    bullets: list[str] = field(default_factory=list)


@dataclass
class ReviewOutline:
    """本数据结构作用：保存综述提纲生成任务的结构化结果。"""

    topic: str
    source_papers: list[str] = field(default_factory=list)
    sections: list[ReviewOutlineSection] = field(default_factory=list)


def build_section(
    title: str,
    bullets: list[str],
    default_bullet: str,
    max_items: int = 4,
) -> ReviewOutlineSection:
    """构建综述提纲中的单个章节。

    输入：
        title: 章节标题。
        bullets: 候选要点列表。
        default_bullet: 当候选为空时使用的默认要点。
        max_items: 最多保留多少条要点。
    输出：
        单个综述章节对象。
    异常：
        无。
    """

    cleaned_items = [item.strip() for item in bullets if item and item.strip()]
    if not cleaned_items:
        cleaned_items = [default_bullet]
    return ReviewOutlineSection(title=title, bullets=cleaned_items[:max_items])


def build_outline_sections(topic: str, comparison: MultiPaperComparison) -> list[ReviewOutlineSection]:
    """根据多篇比较结果生成综述提纲章节。

    输入：
        topic: 综述主题。
        comparison: 多篇论文比较结果。
    输出：
        综述提纲章节列表。
    异常：
        无。
    """

    sections = [
        build_section(
            title="研究背景与问题提出",
            bullets=[
                f"围绕“{topic}”梳理 PAN 与 LRHS/HSI 在空间分辨率和光谱分辨率上的互补性。",
                *comparison.common_themes,
            ],
            default_bullet="说明为什么需要同时利用 PAN 的空间细节和 HSI/LRHS 的光谱信息。",
        ),
        build_section(
            title="任务定义与输入模态",
            bullets=comparison.task_type_comparison + comparison.modality_comparison,
            default_bullet="区分 MSI pansharpening、HSI pansharpening、HSI-MSI fusion、HSI super-resolution 和 general image fusion。",
        ),
        build_section(
            title="传统方法与模型驱动方法",
            bullets=[
                "梳理 CS、MRA、VO、Bayesian/model-based、tensor decomposition 和 degradation consistency 等传统或模型驱动路线。",
                *comparison.prior_or_degradation_modeling,
            ],
            default_bullet="比较传统注入、分解、优化和物理退化约束方法的适用边界。",
        ),
        build_section(
            title="深度网络方法",
            bullets=comparison.method_comparison,
            default_bullet="整理 CNN、Transformer、attention/gate、unfolding、diffusion 与 zero-shot 等深度方法。",
        ),
        build_section(
            title="空间信息建模",
            bullets=comparison.spatial_modeling_comparison,
            default_bullet="比较 PAN 引导、纹理增强、高频注入、频域/小波、空间注意力和卷积结构。",
        ),
        build_section(
            title="光谱信息保持",
            bullets=comparison.spectral_modeling_comparison,
            default_bullet="比较通道注意力、光谱约束、低秩/张量结构、光谱保持损失和 SAM 等光谱建模方式。",
        ),
        build_section(
            title="退化模型与物理约束",
            bullets=comparison.prior_or_degradation_modeling,
            default_bullet="讨论 PSF、SRF、blur kernel、downsampling、spectral response 与 Wald protocol 等退化设定。",
        ),
        build_section(
            title="数据集、实验协议与评价指标",
            bullets=comparison.dataset_comparison + comparison.metric_comparison + comparison.finding_comparison,
            default_bullet="整理 CAVE、Harvard、Pavia、Chikusei、WorldView、GaoFen 等数据集和 PSNR、SSIM、SAM、ERGAS、QNR 等指标。",
        ),
        build_section(
            title="局限与后续研究方向",
            bullets=[
                f"《{item.file_name}》的局限：{item.limitations}"
                for item in comparison.paper_summaries
            ],
            default_bullet="可从真实退化不匹配、全分辨率验证、泛化能力、计算成本和训练数据依赖等角度总结局限。",
        ),
        build_section(
            title="对用户当前模型的启示",
            bullets=comparison.relevance_to_user_research or comparison.integrated_implications,
            default_bullet="总结对 latent diffusion、zero-shot prior、PAN/LRHS 条件门控、U-Net 层级调制、ARConv 和消融设计的启示。",
        ),
    ]
    return sections


def generate_review_outline(topic: str, papers: list[dict[str, str]]) -> ReviewOutline:
    """根据若干论文生成最小综述提纲。

    输入：
        topic: 综述主题。
        papers: 参与提纲生成的论文列表，每项至少包含 file_name、document_id 和 full_text。
    输出：
        综述提纲对象。
    异常：
        当 topic 为空时，抛出 ValueError。
        当 papers 数量不足两篇时，抛出 ValueError。
    """

    cleaned_topic = topic.strip()
    if not cleaned_topic:
        raise ValueError("综述主题不能为空。")
    if len(papers) < 2:
        raise ValueError("生成综述提纲时至少需要 2 篇论文。")

    comparison = compare_papers(papers=papers, topic_hint=cleaned_topic)

    return ReviewOutline(
        topic=cleaned_topic,
        source_papers=[item.file_name for item in comparison.paper_summaries],
        sections=build_outline_sections(cleaned_topic, comparison),
    )


def format_review_outline(result: ReviewOutline) -> str:
    """将综述提纲整理为适合命令行展示的文本。

    输入：
        result: 综述提纲对象。
    输出：
        可直接打印的提纲文本。
    异常：
        无。
    """

    lines = [
        "综述提纲：",
        f"主题：{result.topic}",
        f"参考论文：{', '.join(result.source_papers)}",
    ]

    for index, section in enumerate(result.sections, start=1):
        lines.append(f"{index}. {section.title}")
        for bullet in section.bullets:
            lines.append(f"- {bullet}")

    return "\n".join(lines)


def run_outline_demo() -> None:
    """执行 outline_tool 模块的最小演示。

    输入：
        无。
    输出：
        无。函数会直接打印综述提纲。
    异常：
        无。
    """

    demo_papers = [
        {
            "file_name": "hsi_pansharpening_attention.pdf",
            "document_id": "paper_a",
            "full_text": (
                "本文关注高光谱全色锐化任务，输入为 LRHS 与 PAN。"
                "方法采用空间注意力引导 PAN 纹理注入，并用光谱保持损失约束重建结果。"
                "实验在 Pavia Center、CAVE 和 Harvard 数据集上验证，指标包括 PSNR、SSIM、SAM 和 ERGAS。"
                "结果表明该方法在保持光谱一致性的同时提升空间细节。"
            ),
        },
        {
            "file_name": "hsi_msi_fusion_transformer.pdf",
            "document_id": "paper_b",
            "full_text": (
                "研究聚焦 HSI-MSI fusion，输入为低空间分辨率 HSI 与高空间分辨率 MSI。"
                "方法使用张量分解和 Transformer 建模空间-光谱相关性。"
                "实验在 CAVE、Harvard 和 Pavia University 数据集上进行。"
                "结果显示该框架在光谱保真和空间结构恢复上优于若干 baseline。"
            ),
        },
    ]

    result = generate_review_outline("高光谱全色锐化中的空间-光谱融合机制综述", demo_papers)
    print(format_review_outline(result))


if __name__ == "__main__":
    run_outline_demo()
