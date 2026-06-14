"""本模块作用：保存、加载和统计论文图表视觉索引。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VISION_INDEX_VERSION = 1


def vision_index_exists(index_path: str | Path) -> bool:
    """判断 vision_index.json 是否存在。"""

    return Path(index_path).expanduser().exists()


def save_vision_index(items: list[dict[str, Any]], index_path: str | Path) -> None:
    """以 UTF-8 JSON 保存视觉索引。"""

    path = Path(index_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": VISION_INDEX_VERSION,
        "item_count": len(items),
        "items": items,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vision_index(index_path: str | Path) -> list[dict[str, Any]]:
    """加载视觉索引；文件缺失、为空或格式异常时返回空列表。"""

    path = Path(index_path).expanduser()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        items = payload.get("items", [])
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def get_vision_status(index_path: str | Path, image_dir: str | Path) -> dict[str, Any]:
    """返回视觉索引与图片目录状态。"""

    path = Path(index_path).expanduser().resolve()
    image_root = Path(image_dir).expanduser().resolve()
    items = load_vision_index(path)
    document_ids = {str(item.get("document_id", "")).strip() for item in items if item.get("document_id")}
    indexed_files = {str(item.get("file_name", "")).strip() for item in items if item.get("file_name")}
    image_count = 0
    if image_root.exists():
        try:
            image_count = len(list(image_root.rglob("*.png")))
        except Exception:
            image_count = 0

    return {
        "enabled_index_path": str(path),
        "index_path": str(path),
        "image_dir": str(image_root),
        "index_exists": path.exists(),
        "image_dir_exists": image_root.exists(),
        "paper_count": len(document_ids),
        "indexed_file_count": len(indexed_files),
        "visual_item_count": len(items),
        "image_count": image_count,
    }
