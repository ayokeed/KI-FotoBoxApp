# pipeline/assets.py  (NEW FILE)
from __future__ import annotations

import os
import json
from typing import Dict, Any, List

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def _scan_files_without_ext(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return ["none"]
    files = [
        os.path.splitext(f)[0]
        for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTS)
    ]
    if "none" not in files:
        files.append("none")
    return sorted(set(files))


def _load_meta_keys(meta_file: str) -> List[str]:
    if not os.path.isfile(meta_file):
        return ["none"]
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        keys = list(meta.keys())
        if "none" not in keys:
            keys.append("none")
        return sorted(set(keys))
    except Exception:
        return ["none"]


def load_all_assets(asset_dirs: Dict[str, str]) -> Dict[str, Any]:
    """
    Assets-only scan (no ML imports).
    Keeps your existing "options list" format so you can reuse prompts:
      {
        "backgrounds": ["beach","space","none",...],
        "effects": [...],
        "hats": [...],
        "glasses": [...],
        "masks": [...]
      }
    """
    all_assets: Dict[str, Any] = {}

    for category in ["backgrounds", "effects"]:
        all_assets[category] = _scan_files_without_ext(asset_dirs.get(category, ""))

    for category in ["hats", "glasses", "masks"]:
        meta_file = os.path.join(asset_dirs.get(category, ""), f"{category}_metadata.json")
        all_assets[category] = _load_meta_keys(meta_file)

    return all_assets
