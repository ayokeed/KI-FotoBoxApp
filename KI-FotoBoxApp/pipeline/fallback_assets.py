from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _first_asset_stem(dir_path: str) -> Optional[str]:
    """
    Deterministic: returns first filename stem in sorted order from dir_path,
    or None if directory doesn't exist / has no images.
    """
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        return None

    files = sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS])
    if not files:
        return None
    return files[0].stem


def pick_fallback_tag(
    env_key: str,
    dir_path: str,
    default: str = "none",
) -> str:
    """
    Priority:
      1) explicit env override (e.g., FALLBACK_BG=beach)
      2) first asset from directory (e.g., assets/backgrounds/beach.png -> "beach")
      3) default ("none")
    """
    v = os.getenv(env_key)
    if v and v.strip():
        return v.strip()

    stem = _first_asset_stem(dir_path)
    return stem if stem else default
