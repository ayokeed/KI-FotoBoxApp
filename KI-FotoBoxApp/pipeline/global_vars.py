# pipeline/global_vars.py
from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

# IMPORTANT:
# Do NOT import FaceDetector / BackgroundRemover / AccessoryPlacer / OpenAI_Client here.
# This module must be safe to import during FastAPI startup (lifespan) and for /suggest.

ASSETS: Dict[str, Any] = {}
ASSET_DIRS: Dict[str, str] = {}
ASSET_OPTIONS: Dict[str, Any] = {}  # optional: if you still want the old "options" list format

_face_detector: Optional[Any] = None
_bg_remover: Optional[Any] = None
_accessory_placer: Optional[Any] = None
_openai_client: Optional[Any] = None

# Optional shared data (not required, but useful)
asset_options: Dict[str, Any] = {}
asset_dirs: Dict[str, str] = {}
if TYPE_CHECKING:
    from pipeline.face_detection import FaceDetector  # noqa: F401
    from pipeline.background_removal import BackgroundRemover  # noqa: F401
    from pipeline.accessory_application import AccessoryPlacer  # noqa: F401
    from pipeline.openai_client import OpenAI_Client  # noqa: F401



def set_assets(asset_dirs: Dict[str, str], assets: Dict[str, Any]) -> None:
    global ASSET_DIRS, ASSETS
    ASSET_DIRS = dict(asset_dirs)
    ASSETS = dict(assets)


def set_asset_options(asset_options: Dict[str, Any]) -> None:
    global ASSET_OPTIONS
    ASSET_OPTIONS = dict(asset_options)


def get_assets() -> Dict[str, Any]:
    return ASSETS


def get_asset_dirs() -> Dict[str, str]:
    return ASSET_DIRS


def get_asset_options() -> Dict[str, Any]:
    return ASSET_OPTIONS


def get_face_detector() -> Any:
    global _face_detector
    if _face_detector is None:
        from pipeline.face_detection import FaceDetector  # local import (heavy)
        _face_detector = FaceDetector()
    return _face_detector


def get_bg_remover() -> Any:
    """
    Lazily constructs BackgroundRemover only when background replacement is requested.
    """
    global _bg_remover
    if _bg_remover is None:
        from pipeline.background_removal import BackgroundRemover  # local import
        _bg_remover = BackgroundRemover()  # device auto-picks cuda/cpu, ckpt via MODNET_CKPT env or default
    return _bg_remover


def get_accessory_placer(asset_dirs: Optional[Dict[str, str]] = None) -> Any:
    """
    AccessoryPlacer typically depends on asset_dirs; pass in from ImagePipeline init.
    """
    global _accessory_placer
    if _accessory_placer is None:
        from pipeline.accessory_application import AccessoryPlacer  # local import
        if asset_dirs is None:
            asset_dirs = ASSET_DIRS
        _accessory_placer = AccessoryPlacer(asset_dirs)
    return _accessory_placer


def get_openai_client() -> Any:
    """
    Optional. If OpenAI is not configured, return None.
    """
    global _openai_client
    if _openai_client is None:
        try:
            from pipeline.openai_client import OpenAI_Client  # local import
            _openai_client = OpenAI_Client()
        except Exception:
            _openai_client = None
    return _openai_client
