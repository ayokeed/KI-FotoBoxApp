# pipeline/assets.py  (UPDATED - supports event_* folders + merged options + preview index)
from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional

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
    """Try stem + allowed extensions; return first existing path or None."""
    for ext in IMAGE_EXTS:
        p = os.path.join(base_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None

def _extract_file_candidate(meta_value: Any) -> Optional[str]:
    """Try common keys in metadata entries to find an image filename/path."""
    if isinstance(meta_value, str):
        return meta_value
    if not isinstance(meta_value, dict):
        return None
    for key in ("file", "filename", "path", "image", "img", "asset", "src"):
        v = meta_value.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _scan_image_stems(directory: str) -> list[str]:
    """Return filename stems for all image files in a directory."""
    if not directory or not os.path.isdir(directory):
        return []
    stems: list[str] = []
    for f in os.listdir(directory):
        if f.lower().endswith(IMAGE_EXTS):
            stems.append(os.path.splitext(f)[0])
    return stems

def load_all_assets(asset_dirs: Dict[str, str]) -> Dict[str, Any]:
    """
    Loads asset options for all categories.

    - backgrounds/effects:
        * scan normal folders
        * scan optional event folders (event_backgrounds/event_effects)
        * MERGE both into the public option list, so frontend can select event assets too
    - hats/glasses/masks: load keys from *_metadata.json
    Always adds "none".
    Lightweight: no ML imports.
    """
    all_assets: Dict[str, Any] = {}

    # Merge normal + event for backgrounds/effects
    bg_normal = _scan_image_stems(asset_dirs.get("backgrounds", ""))
    bg_event = _scan_image_stems(asset_dirs.get("event_backgrounds", ""))
    eff_normal = _scan_image_stems(asset_dirs.get("effects", ""))
    eff_event = _scan_image_stems(asset_dirs.get("event_effects", ""))

    backgrounds = sorted(set(bg_normal + bg_event + ["none"]))
    effects = sorted(set(eff_normal + eff_event + ["none"]))

    all_assets["backgrounds"] = backgrounds
    all_assets["effects"] = effects

    # hats + glasses + masks by metadata json keys
    for category in ["hats", "glasses", "masks"]:
        meta_dir = asset_dirs.get(category, "")
        meta_file = os.path.join(meta_dir, f"{category}_metadata.json")

        if meta_dir and os.path.isfile(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                keys = list(meta.keys()) if isinstance(meta, dict) else []
                keys = sorted(set(keys + ["none"]))
                all_assets[category] = keys
            except Exception:
                all_assets[category] = ["none"]
        else:
            all_assets[category] = ["none"]

    # Helpful debug (safe)
    try:
        print("[ASSETS] asset_dirs:")
        for k, v in asset_dirs.items():
            print(f"  - {k}: {v} (exists={os.path.isdir(v)})")
        print(f"[ASSETS] backgrounds={len(backgrounds)} effects={len(effects)}")
    except Exception:
        pass

    return all_assets

def build_asset_preview_index(asset_dirs: Dict[str, str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Builds:
      {category: {asset_id: {"path": "/abs/file.png", "mime": "image/png"}}}
    Used by GET /assets/{category}/{asset_id}.

    IMPORTANT:
    - For backgrounds/effects we index BOTH normal and event directories under the SAME category,
      so preview works even if the chosen id comes from event_* folders.
    - For hats/glasses/masks we resolve via metadata when possible.
    """
    index: Dict[str, Dict[str, Dict[str, str]]] = {
        "backgrounds": {},
        "effects": {},
        "hats": {},
        "glasses": {},
        "masks": {},
    }

    # backgrounds/effects: index normal + event dirs into the same namespace
    def _index_dir_into(category: str, directory: str) -> None:
        if not directory or not os.path.isdir(directory):
            return
        for fn in os.listdir(directory):
            if not fn.lower().endswith(IMAGE_EXTS):
                continue
            stem, _ = os.path.splitext(fn)
            full = os.path.join(directory, fn)
            # If duplicates exist (same id in event + normal), prefer event by indexing event last.
            index[category][stem] = {"path": full, "mime": _mime_for_file(full)}

    # Normal first, then event to allow event to override duplicates
    _index_dir_into("backgrounds", asset_dirs.get("backgrounds", ""))
    _index_dir_into("effects", asset_dirs.get("effects", ""))
    _index_dir_into("backgrounds", asset_dirs.get("event_backgrounds", ""))
    _index_dir_into("effects", asset_dirs.get("event_effects", ""))

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

    return index
