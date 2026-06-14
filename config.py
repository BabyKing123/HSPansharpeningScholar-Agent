"""本模块作用：集中管理 HSPansharpeningScholar-Agent 的基础配置，为整个智能体提供统一目录、模型与运行参数。"""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_NAME = "HSPansharpeningScholar-Agent"
DATA_DIR = BASE_DIR / "data"
RAW_PAPERS_DIR = DATA_DIR / "raw_papers"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-3ba29d5b5ef64c1ba1dcc54e3e7194c4").strip()
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
).strip()

# 按场景分配模型：问答优先性价比，结构化分析优先准确性。
DASHSCOPE_ANSWER_MODEL = os.getenv("DASHSCOPE_ANSWER_MODEL", "qwen3.7-plus").strip()
DASHSCOPE_ANALYSIS_MODEL = os.getenv("DASHSCOPE_ANALYSIS_MODEL", "deepseek-v4-pro").strip()
DASHSCOPE_EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4").strip()


def _read_int_env(name: str, default_value: int) -> int:
    """读取整型环境变量，并在异常时回退默认值。"""

    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default_value
    try:
        return int(raw_value)
    except ValueError:
        return default_value


def _read_float_env(name: str, default_value: float) -> float:
    """读取浮点型环境变量，并在异常时回退默认值。"""

    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default_value
    try:
        return float(raw_value)
    except ValueError:
        return default_value  


def _read_bool_env(name: str, default_value: bool) -> bool:
    """读取布尔环境变量，并兼容常见真值与假值写法。"""

    raw_value = os.getenv(name, "").strip().lower()
    if not raw_value:
        return default_value
    if raw_value in {"1", "true", "yes", "y", "on"}:
        return True
    if raw_value in {"0", "false", "no", "n", "off"}:
        return False
    return default_value


DASHSCOPE_TIMEOUT_SEC = _read_int_env("DASHSCOPE_TIMEOUT_SEC", 600)
DASHSCOPE_EMBEDDING_DIMENSIONS = _read_int_env("DASHSCOPE_EMBEDDING_DIMENSIONS", 128)

GRAPH_ENABLED = _read_bool_env("GRAPH_ENABLED", True)
GRAPH_INDEX_PATH = Path(os.getenv("GRAPH_INDEX_PATH", str(PROCESSED_DATA_DIR / "graph_index.json"))).expanduser().resolve()
GRAPH_EXTRACTION_MODEL = os.getenv("GRAPH_EXTRACTION_MODEL", DASHSCOPE_ANALYSIS_MODEL).strip()
GRAPH_SUMMARY_MODEL = os.getenv("GRAPH_SUMMARY_MODEL", DASHSCOPE_ANSWER_MODEL).strip()
GRAPH_MAX_HOPS = _read_int_env("GRAPH_MAX_HOPS", 2)
GRAPH_LOCAL_TOP_K_ENTITIES = _read_int_env("GRAPH_LOCAL_TOP_K_ENTITIES", 5)
GRAPH_LOCAL_TOP_K_CHUNKS = _read_int_env("GRAPH_LOCAL_TOP_K_CHUNKS", 5)
GRAPH_GLOBAL_TOP_K_COMMUNITIES = _read_int_env("GRAPH_GLOBAL_TOP_K_COMMUNITIES", 4)

VISION_ENABLED = _read_bool_env("VISION_ENABLED", True)
VISION_INDEX_PATH = Path(
    os.getenv("VISION_INDEX_PATH", str(PROCESSED_DATA_DIR / "vision_index.json"))
).expanduser().resolve()
# 这里选择多模态模型的视觉模型，用于图像检索和分析
VISION_IMAGE_DIR = Path(os.getenv("VISION_IMAGE_DIR", str(PROCESSED_DATA_DIR / "vision"))).expanduser().resolve()
DASHSCOPE_VISION_MODEL = os.getenv("DASHSCOPE_VISION_MODEL", "qwen3.7-plus").strip()
VISION_RENDER_ZOOM = _read_float_env("VISION_RENDER_ZOOM", 2.0)
VISION_MAX_PAGES_PER_PAPER = _read_int_env("VISION_MAX_PAGES_PER_PAPER", 30)
VISION_TOP_K = _read_int_env("VISION_TOP_K", 3)


def get_app_config() -> dict[str, Path | str | int | float | bool]:
    """返回应用当前使用的基础配置。

    输入：
        无。
    输出：
        包含项目目录、模型参数和 API 开关的配置字典。
    异常：
        无。
    """

    return {
        "project_name": PROJECT_NAME,
        "base_dir": BASE_DIR,
        "data_dir": DATA_DIR,
        "raw_papers_dir": RAW_PAPERS_DIR,
        "processed_data_dir": PROCESSED_DATA_DIR,
        "output_dir": OUTPUT_DIR,
        "dashscope_api_key": DASHSCOPE_API_KEY,
        "dashscope_base_url": DASHSCOPE_BASE_URL,
        "dashscope_answer_model": DASHSCOPE_ANSWER_MODEL,
        "dashscope_analysis_model": DASHSCOPE_ANALYSIS_MODEL,
        "dashscope_embedding_model": DASHSCOPE_EMBEDDING_MODEL,
        "dashscope_embedding_dimensions": DASHSCOPE_EMBEDDING_DIMENSIONS,
        "dashscope_timeout_sec": DASHSCOPE_TIMEOUT_SEC,
        "graph_enabled": GRAPH_ENABLED,
        "graph_index_path": GRAPH_INDEX_PATH,
        "graph_extraction_model": GRAPH_EXTRACTION_MODEL,
        "graph_summary_model": GRAPH_SUMMARY_MODEL,
        "graph_max_hops": GRAPH_MAX_HOPS,
        "graph_local_top_k_entities": GRAPH_LOCAL_TOP_K_ENTITIES,
        "graph_local_top_k_chunks": GRAPH_LOCAL_TOP_K_CHUNKS,
        "graph_global_top_k_communities": GRAPH_GLOBAL_TOP_K_COMMUNITIES,
        "vision_enabled": VISION_ENABLED,
        "vision_index_path": VISION_INDEX_PATH,
        "vision_image_dir": VISION_IMAGE_DIR,
        "dashscope_vision_model": DASHSCOPE_VISION_MODEL,
        "vision_render_zoom": VISION_RENDER_ZOOM,
        "vision_max_pages_per_paper": VISION_MAX_PAGES_PER_PAPER,
        "vision_top_k": VISION_TOP_K,
        "llm_enabled": bool(DASHSCOPE_API_KEY),
    }
