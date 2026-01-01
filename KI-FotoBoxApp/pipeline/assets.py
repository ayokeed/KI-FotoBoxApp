# pipeline/assets.py
from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Optional, Tuple

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

def _mime_for_file(path: str) -> str:
    _, ext = os.path.splitext(path)
    return MIME_BY_EXT.get(ext.lower(), "application/octet-stream")

def _first_existing_file(base_dir: str, stem: str) -> Optional[str]:
    """
    Try stem + allowed extensions; return first existing path or None.
    """
    for ext in IMAGE_EXTS:
        p = os.path.join(base_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None

def _extract_file_candidate(meta_value: Any) -> Optional[str]:
    """
    Try common keys in metadata entries to find an image filename/path.
    Returns a string (maybe relative) or None.
    """
    if isinstance(meta_value, str):
        return meta_value
    if not isinstance(meta_value, dict):
        return None

    for key in ("file", "filename", "path", "image", "img", "asset", "src"):
        v = meta_value.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def load_all_assets(asset_dirs: Dict[str, str]) -> Dict[str, Any]:
    """
    Loads asset options for all categories.
    - backgrounds/effects: scan filenames (stems)
    - hats/glasses/masks: load keys from *_metadata.json
    Always adds "none".
    Lightweight: no ML imports.
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
            if f.lower().endswith(IMAGE_EXTS)
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
                keys = list(meta.keys()) if isinstance(meta, dict) else []
                if "none" not in keys:
                    keys.append("none")
                all_assets[category] = sorted(set(keys))
            except Exception:
                all_assets[category] = ["none"]
        else:
            all_assets[category] = ["none"]

    return all_assets

def build_asset_preview_index(asset_dirs: Dict[str, str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Builds:
      {category: {asset_id: {"path": "/abs/file.png", "mime": "image/png"}}}
    Used by GET /assets/{category}/{asset_id}.

    IMPORTANT:
    - asset_id stays the SAME IDs your pipeline expects (stems / metadata keys),
      so /process keeps working without changes.
    - For metadata-based assets, we try to resolve the underlying image file:
        - if metadata entry contains a filename/path -> use it
        - else fall back to looking for <asset_id>.(png/jpg/jpeg/webp) in the category folder
    """
    index: Dict[str, Dict[str, Dict[str, str]]] = {
        "backgrounds": {},
        "effects": {},
        "hats": {},
        "glasses": {},
        "masks": {},
    }

    # backgrounds/effects: map stem -> file path
    for category in ["backgrounds", "effects"]:
        directory = asset_dirs.get(category, "")
        if not directory or not os.path.isdir(directory):
            continue
        for fn in os.listdir(directory):
            if not fn.lower().endswith(IMAGE_EXTS):
                continue
            stem, _ = os.path.splitext(fn)
            full = os.path.join(directory, fn)
            index[category][stem] = {"path": full, "mime": _mime_for_file(full)}

    # hats/glasses/masks: resolve via metadata if possible
    for category in ["hats", "glasses", "masks"]:
        meta_dir = asset_dirs.get(category, "")
        if not meta_dir or not os.path.isdir(meta_dir):
            continue

        meta_file = os.path.join(meta_dir, f"{category}_metadata.json")
        meta: Any = None
        if os.path.isfile(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = None

        if isinstance(meta, dict):
            for asset_id, meta_val in meta.items():
                if isinstance(asset_id, str) and asset_id.lower() == "none":
                    continue

                candidate = _extract_file_candidate(meta_val)

                resolved: Optional[str] = None
                if candidate:
                    # If metadata path is relative, resolve against the category folder
                    if os.path.isabs(candidate):
                        resolved = candidate if os.path.isfile(candidate) else None
                    else:
                        # try direct join; also try if candidate has no ext
                        p = os.path.join(meta_dir, candidate)
                        if os.path.isfile(p):
                            resolved = p
                        else:
                            stem, ext = os.path.splitext(candidate)
                            if ext == "":
                                resolved = _first_existing_file(meta_dir, candidate)
                if not resolved:
                    # fallback: look for <asset_id>.<ext> in category folder
                    resolved = _first_existing_file(meta_dir, asset_id)

                if resolved and os.path.isfile(resolved):
                    index[category][asset_id] = {"path": resolved, "mime": _mime_for_file(resolved)}

        else:
            # No metadata available: fallback to <asset_id>.<ext> convention is not possible without IDs list.
            # We'll rely on /suggest options; and /assets will 404 if not found.
            pass

    return index
