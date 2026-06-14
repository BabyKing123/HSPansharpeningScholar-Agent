"""本模块作用：在整个智能体中负责将多步流程的中间结果导出为 Markdown 文件，支撑第三周的最小工作流闭环。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExportArtifact:
    """本数据结构作用：保存一次导出任务的结果信息。"""

    output_path: str
    title: str
    content_length: int


def sanitize_file_name(name: str, fallback_name: str = "workflow_report") -> str:
    """将任意标题清洗为适合文件名使用的字符串。

    输入：
        name: 原始标题文本。
        fallback_name: 当标题为空时使用的默认文件名。
    输出：
        清洗后的文件名主体。
    异常：
        无。
    """

    cleaned_name = re.sub(r"[^\w\u4e00-\u9fff\- ]+", "_", name.strip())
    cleaned_name = re.sub(r"\s+", "_", cleaned_name)
    cleaned_name = cleaned_name.strip("_")
    return cleaned_name or fallback_name


def ensure_output_dir(output_dir: str | Path) -> Path:
    """确保导出目录存在。

    输入：
        output_dir: 导出目录路径。
    输出：
        已确保存在的目录对象。
    异常：
        当目录创建失败时，抛出 OSError。
    """

    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def escape_table_cell(text: str) -> str:
    """转义 Markdown 表格单元格。"""

    return text.replace("|", "\\|").replace("\n", "<br>").strip()


def render_summary_table(rows: list[dict[str, str]]) -> list[str]:
    """把单篇论文概括渲染为 Markdown 表格。"""

    if not rows:
        return []
    headers = ["论文", "研究问题", "研究对象", "方法", "主要结论"]
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    table_lines = [header_line, divider_line]
    field_map = {
        "论文": "file_name",
        "研究问题": "研究问题",
        "研究对象": "研究对象",
        "方法": "方法",
        "主要结论": "主要结论",
    }
    for row in rows:
        cells = [escape_table_cell(row.get(field_map[header], "")) for header in headers]
        table_lines.append("| " + " | ".join(cells) + " |")
    return table_lines


def render_comparison_markdown(comparison_text: str) -> list[str]:
    """将比较模块的纯文本输出转换为更可读的 Markdown。"""

    source_lines = [line.rstrip() for line in comparison_text.splitlines()]
    rendered_lines: list[str] = []
    index = 0
    while index < len(source_lines):
        raw_line = source_lines[index]
        line = raw_line.strip()
        if not line:
            rendered_lines.append("")
            index += 1
            continue
        if line == "多篇论文比较结果：":
            index += 1
            continue
        if line.startswith(("比较主题：", "纳入论文数量：")):
            rendered_lines.append(f"- {line}")
            index += 1
            continue
        if line == "一、单篇论文概括：":
            rendered_lines.append("### 一、单篇论文概括")
            rows: list[dict[str, str]] = []
            current_row: dict[str, str] | None = None
            index += 1
            while index < len(source_lines):
                summary_line = source_lines[index].strip()
                if re.match(r"^[二三四五六七八九十]+、", summary_line):
                    break
                paper_match = re.match(r"^\d+\.\s*《(.+)》$", summary_line)
                if paper_match:
                    if current_row is not None:
                        rows.append(current_row)
                    current_row = {"file_name": paper_match.group(1)}
                    index += 1
                    continue
                field_match = re.match(r"^(研究问题|研究对象|方法|主要结论)：(.+)$", summary_line)
                if field_match and current_row is not None:
                    current_row[field_match.group(1)] = field_match.group(2).strip()
                index += 1
            if current_row is not None:
                rows.append(current_row)
            rendered_lines.extend(render_summary_table(rows))
            rendered_lines.append("")
            continue
        if re.match(r"^[一二三四五六七八九十]+、", line):
            rendered_lines.append(f"### {line.rstrip('：')}")
        elif line.endswith("：") and not line.startswith("-"):
            rendered_lines.append(f"#### {line.rstrip('：')}")
        else:
            rendered_lines.append(line)
        index += 1
    return rendered_lines


def render_outline_markdown(outline_text: str) -> list[str]:
    """将综述提纲纯文本输出转换为 Markdown。"""

    rendered_lines: list[str] = []
    for raw_line in outline_text.splitlines():
        line = raw_line.strip()
        if not line or line == "综述提纲：":
            continue
        if line.startswith(("主题：", "参考论文：")):
            rendered_lines.append(f"- {line}")
        elif re.match(r"^\d+\.\s+", line):
            rendered_lines.append("")
            rendered_lines.append(f"### {line}")
        else:
            rendered_lines.append(line)
    return rendered_lines


def build_workflow_markdown(
    topic: str,
    selected_papers: list[str],
    comparison_text: str,
    outline_text: str,
    step_logs: list[str],
    graph_text: str = "",
    vision_text: str = "",
) -> str:
    """将第三周工作流结果整理为 Markdown 文本。

    输入：
        topic: 工作流主题。
        selected_papers: 纳入论文列表。
        comparison_text: 多篇比较展示文本。
        outline_text: 综述提纲展示文本。
        step_logs: 工作流步骤日志。
        graph_text: GraphRAG 辅助发现文本。
        vision_text: 图表摘要辅助发现文本。
    输出：
        Markdown 格式文本。
    异常：
        无。
    """

    lines = [
        f"# {topic}",
        "",
        "## 工作流摘要",
        "",
        f"- 主题：{topic}",
        f"- 纳入论文数量：{len(selected_papers)}",
        "",
        "## 纳入论文",
        "",
    ]

    for paper in selected_papers:
        lines.append(f"- {paper}")

    lines.extend(
        [
            "",
            "## 工作流步骤日志",
            "",
        ]
    )
    for step_log in step_logs:
        lines.append(f"- {step_log}")

    if graph_text.strip():
        lines.extend(["", "## GraphRAG 辅助发现", ""])
        for raw_line in graph_text.splitlines():
            line = raw_line.strip()
            if not line or line == "GraphRAG 辅助发现：":
                continue
            if line.endswith("：") and not line.startswith("-"):
                lines.append(f"### {line.rstrip('：')}")
            else:
                lines.append(line)

    if vision_text.strip():
        lines.extend(["", "## 图表与视觉内容分析", ""])
        for raw_line in vision_text.splitlines():
            line = raw_line.strip()
            if not line or line == "图表与视觉内容分析：":
                continue
            if line.endswith("：") and not line.startswith("-"):
                lines.append(f"### {line.rstrip('：')}")
            else:
                lines.append(line)

    lines.extend(["", "## 多篇论文比较", ""])
    lines.extend(render_comparison_markdown(comparison_text))
    lines.extend(["", "## 综述提纲", ""])
    lines.extend(render_outline_markdown(outline_text))
    lines.append("")
    return "\n".join(lines)


def export_markdown_report(
    output_dir: str | Path,
    title: str,
    markdown_text: str,
) -> ExportArtifact:
    """将 Markdown 内容写入输出目录。

    输入：
        output_dir: 输出目录。
        title: 导出标题。
        markdown_text: 待写入的 Markdown 内容。
    输出：
        导出结果对象。
    异常：
        当文件写入失败时，抛出 OSError。
    """

    target_dir = ensure_output_dir(output_dir)
    file_name = sanitize_file_name(title) + ".md"
    output_path = target_dir / file_name
    output_path.write_text(markdown_text, encoding="utf-8")
    return ExportArtifact(
        output_path=str(output_path),
        title=title,
        content_length=len(markdown_text),
    )


def run_export_demo() -> None:
    """执行 export_tool 模块的最小演示。

    输入：
        无。
    输出：
        无。函数会直接打印导出结果。
    异常：
        无。
    """

    markdown_text = build_workflow_markdown(
        topic="高光谱全色锐化中的空间-光谱融合机制综述",
        selected_papers=["hsi_pansharpening_attention.pdf", "hsi_msi_fusion_transformer.pdf"],
        comparison_text="这里是多篇比较结果。",
        outline_text="这里是综述提纲。",
        step_logs=[
            "步骤 1：已选中 2 篇论文。",
            "步骤 2：已完成多篇比较。",
            "步骤 3：已生成综述提纲。",
        ],
    )
    artifact = export_markdown_report("outputs", "高光谱全色锐化中的空间-光谱融合机制综述", markdown_text)
    print(f"导出成功：{artifact.output_path}")


if __name__ == "__main__":
    run_export_demo()
