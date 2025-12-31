# pipeline/assets.py  (NEW FILE)
from __future__ import annotations

import os
import json
from typing import Dict, Any, List

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def load_all_assets(asset_dirs: Dict[str, str]) -> Dict[str, Any]:
    """
    Loads asset options for all categories.
    - backgrounds/effects: scan filenames (no extensions)
    - hats/glasses/masks: load keys from *_metadata.json
    Always adds "none".
    This module is intentionally lightweight: no ML imports.
    """
    all_assets: Dict[str, Any] = {}

    # backgrounds + effects by scanning files
    for category in ["backgrounds", "effects"]:
        directory = asset_dirs.get(category, "")
        if not directory or not os.path.isdir(directory):
            all_assets[category] = ["none"]
            continue

        files = [
            os.path.splitext(f)[0]
            for f in os.listdir(directory)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        if "none" not in files:
            files.append("none")
        all_assets[category] = sorted(set(files))

    # hats + glasses + masks by metadata json keys
    for category in ["hats", "glasses", "masks"]:
        meta_dir = asset_dirs.get(category, "")
        meta_file = os.path.join(meta_dir, f"{category}_metadata.json")

        if meta_dir and os.path.isfile(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                keys = list(meta.keys())
                if "none" not in keys:
                    keys.append("none")
                all_assets[category] = sorted(set(keys))
            except Exception:
                all_assets[category] = ["none"]
        else:
            all_assets[category] = ["none"]

    return all_assets