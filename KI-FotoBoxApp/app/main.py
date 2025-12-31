# app/main.py
# Phase 2 (contract-compliant AI service):
# - POST /health  -> 200 OK
# - POST /suggest -> returns asset-based suggestions (no filesystem paths exposed)
# - POST /process -> returns processed image bytes (PNG) using your existing pipeline
#
# Keeps legacy endpoint:
# - POST /process_image -> returns base64 results (unchanged)

from __future__ import annotations

import os
import io
import uvicorn
import base64
from pathlib import Path
from typing import Optional, List, Any, Dict

from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import JSONResponse

# Load environment variables
load_dotenv()

# -----------------------------
# Config
# -----------------------------
AI_PHASE = int(os.getenv("AI_PHASE", "2"))  # default to Phase 2 now

# IMPORTANT:
# Your OpenAI client uses env var "USE_OPENAI" (in pipeline/openai_client.py).
# This AI_USE_OPENAI flag is kept only as a convenience; if it's "1" we force USE_OPENAI=true.
AI_USE_OPENAI = os.getenv("AI_USE_OPENAI", "0") == "1"
if AI_USE_OPENAI:
    os.environ["USE_OPENAI"] = "true"  # ensure the OpenAI_Client feature flag sees it

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = int(os.getenv("AI_MAX_IMAGE_BYTES", "15000000"))  # 15MB

# Define the asset directories
ASSET_DIRS = {
    "backgrounds": "assets/backgrounds",
    "hats": "assets/hats",
    "glasses": "assets/glasses",
    "effects": "assets/effects",
    "masks": "assets/masks",
}

# We'll store our ImagePipeline instance here.
app_pipeline = None


# ==================================================
# PREVIEW HELPERS (no filesystem paths exposed)
# ==================================================

def _guess_image_mime_from_path(p: Path) -> str:
    ext = p.suffix.lower()
    if ext == ".png":
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _file_bytes_to_data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("utf-8")


def _build_asset_index(dir_path: str) -> Dict[str, Path]:
    """
    Create { asset_id -> file_path } from an assets directory.
    asset_id is the filename without extension.
    Example: assets/backgrounds/beach.png -> "beach"
    """
    root = Path(dir_path)
    idx: Dict[str, Path] = {}
    if not root.exists() or not root.is_dir():
        return idx

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        idx[p.stem] = p
    return idx


def _preview_data_url_for_asset(asset_id: str, index: Dict[str, Path]) -> str | None:
    """
    Returns a data URL like: data:image/png;base64,...
    Never returns filesystem paths.
    """
    p = index.get(asset_id)
    if p is None:
        return None
    try:
        data = p.read_bytes()
        if not data:
            return None
        mime = _guess_image_mime_from_path(p)
        return _file_bytes_to_data_url(data, mime)
    except Exception:
        return None


# ==================================================
# EXISTING HELPERS
# ==================================================

def _normalize_override(v: str | None) -> str | None:
    """
    Treat bad frontend defaults like 'string' as no override.
    """
    if v is None:
        return None
    v = v.strip()
    if v == "" or v.lower() in {"string", "none", "null", "undefined"}:
        return None
    return v


def _choose_device() -> str:
    """
    Prefer CUDA when available.
    - If you have torch installed, use torch.cuda.is_available()
    - Otherwise, fall back to checking CUDA_VISIBLE_DEVICES
    """
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


async def _read_upload(file: UploadFile) -> bytes:
    """
    Read and validate multipart image upload.
    Contract uses field name: file
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large. Max bytes: {MAX_IMAGE_BYTES}.")
    return data


def _safe_get(d: Any, *keys: str, default=None):
    """
    Tries multiple keys on dict-like objects.
    """
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _to_id_label_list(items: Any) -> List[Dict[str, Any]]:
    """
    Convert arbitrary asset list structures into [{id,label}, ...].
    Supports:
      - ["a.jpg","b.jpg"]
      - [{"id":"x","label":"X"}, ...]
      - [{"name":"x"}, ...]
    """
    out: List[Dict[str, Any]] = []
    if not items:
        return out

    if isinstance(items, dict):
        for k, v in items.items():
            if isinstance(v, dict):
                _id = _safe_get(v, "id", default=k)
                label = _safe_get(v, "label", "name", default=str(_id))
            else:
                _id = str(k)
                label = str(k)
            out.append({"id": str(_id), "label": str(label)})
        return out

    if isinstance(items, list):
        for it in items:
            if isinstance(it, str):
                base = os.path.splitext(os.path.basename(it))[0]
                out.append({"id": base, "label": base})
            elif isinstance(it, dict):
                _id = _safe_get(it, "id", "name", "key")
                label = _safe_get(it, "label", "name", default=_id)
                if _id is not None:
                    out.append({"id": str(_id), "label": str(label)})
    return out


def _build_suggest_response(
    asset_options: Any,
    asset_indexes: Dict[str, Dict[str, Path]] | None = None,
) -> Dict[str, Any]:
    """
    Build contract-compliant JSON WITHOUT exposing filesystem paths.

    Returns categories:
      - backgrounds: [{id,label,previewUrl}]
      - effects:     [{id,label,type,previewUrl}]
      - hats:        [{id,label,previewUrl}]
      - glasses:     [{id,label,previewUrl}]
      - masks:       [{id,label,previewUrl}]
    """
    if not isinstance(asset_options, dict):
        asset_options = {}

    idxs = asset_indexes or {}

    def build_category(key: str) -> List[Dict[str, Any]]:
        raw = asset_options.get(key)
        items = _to_id_label_list(raw)
        idx = idxs.get(key, {})
        out: List[Dict[str, Any]] = []
        for it in items:
            asset_id = it["id"]
            out.append(
                {
                    "id": asset_id,
                    "label": it["label"],
                    "previewUrl": _preview_data_url_for_asset(asset_id, idx),
                }
            )
        return out

    backgrounds_out = build_category("backgrounds")

    # Effects: keep your contract-friendly shape (type="filter")
    effects_raw = asset_options.get("effects") or asset_options.get("filters") or asset_options.get("fx")
    effects_base = _to_id_label_list(effects_raw)
    eff_idx = idxs.get("effects", {})
    effects_out = [
        {
            "id": fx["id"],
            "label": fx["label"],
            "type": "filter",
            "previewUrl": _preview_data_url_for_asset(fx["id"], eff_idx),
        }
        for fx in effects_base
    ]

    hats_out = build_category("hats")
    glasses_out = build_category("glasses")
    masks_out = build_category("masks")

    return {
        "backgrounds": backgrounds_out,
        "effects": effects_out,
        "hats": hats_out,
        "glasses": glasses_out,
        "masks": masks_out,
    }


def _encode_png_bytes(img: Any) -> bytes:
    """
    Encode an image object to PNG bytes.
    Tries OpenCV (numpy array) first, then PIL.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        if isinstance(img, np.ndarray):
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            return buf.tobytes()
    except Exception:
        pass

    try:
        from PIL import Image  # type: ignore

        if isinstance(img, Image.Image):
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            return bio.getvalue()
    except Exception:
        pass

    raise HTTPException(status_code=500, detail="Unsupported image type for PNG encoding.")


@asynccontextmanager
async def load_models_and_pipeline(app: FastAPI):
    """
    Phase 2: load models once at startup.
    Also load asset options once and cache them for /suggest.
    IMPORTANT: global_vars.openai_client must NEVER be None (pipeline calls it).
    """
    device = _choose_device()

    from pipeline.face_detection import FaceDetector
    from pipeline.background_removal import BackgroundRemover
    from pipeline.accessory_application import AccessoryPlacer
    from pipeline import global_vars
    from pipeline.image_pipeline import ImagePipeline, load_all_assets
    from pipeline.openai_client import OpenAI_Client

    # Load global models and assign them in the global_vars module.
    global_vars.face_detector = FaceDetector(device=device)
    global_vars.bg_remover = BackgroundRemover(
        model_path="checkpoints/modnet_photographic_portrait_matting.ckpt",
        device=device,
    )
    global_vars.accessory_placer = AccessoryPlacer(ASSET_DIRS)

    # Load asset options once (used for /suggest + deterministic fallback)
    asset_options = load_all_assets(ASSET_DIRS)
    app.state.asset_options = asset_options

    # Build preview indexes for all asset categories (id -> file path)
    app.state.asset_indexes = {
        "backgrounds": _build_asset_index(ASSET_DIRS["backgrounds"]),
        "effects": _build_asset_index(ASSET_DIRS["effects"]),
        "hats": _build_asset_index(ASSET_DIRS["hats"]),
        "glasses": _build_asset_index(ASSET_DIRS["glasses"]),
        "masks": _build_asset_index(ASSET_DIRS["masks"]),
    }

    # IMPORTANT FIX:
    # Always create OpenAI_Client instance; it self-disables & falls back when USE_OPENAI=false or key missing.
    openai_api_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("OPEN_AI_API_KEY", "").strip()
    )
    global_vars.openai_client = OpenAI_Client(
        api_key=openai_api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        asset_options=asset_options,
    )

    # Instantiate the pipeline
    global app_pipeline
    app_pipeline = ImagePipeline(asset_dirs=ASSET_DIRS, device=device)

    print(
        f"Models and pipeline loaded successfully. Device={device}. "
        f"USE_OPENAI={os.getenv('USE_OPENAI', '(unset)')} AI_USE_OPENAI={AI_USE_OPENAI}"
    )
    yield


app = FastAPI(lifespan=load_models_and_pipeline)

# ==================================================
# CONTRACT ENDPOINTS (for backend integration)
# ==================================================

@app.post("/health")
async def health_post() -> Response:
    return Response(status_code=200)


@app.post("/suggest")
async def suggest(file: UploadFile = File(...)) -> JSONResponse:
    _ = await _read_upload(file)

    asset_options = getattr(app.state, "asset_options", None)
    if asset_options is None:
        raise HTTPException(status_code=503, detail="Assets not loaded yet.")

    asset_indexes = getattr(app.state, "asset_indexes", None)
    payload = _build_suggest_response(asset_options, asset_indexes)
    return JSONResponse(content=payload, status_code=200)


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    backgroundId: Optional[str] = Form(default=None),
    effects: Optional[List[str]] = Form(default=None),
) -> Response:
    file_bytes = await _read_upload(file)

    try:
        from pipeline.image_utils import read_imagefile
        input_image = read_imagefile(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=("Error reading the uploaded image. " + str(e)))

    if app_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized yet.")

    # Ensure background/effects overrides are applied
    backgroundId = _normalize_override(backgroundId)

    # Backend sends List[str] in Form; FastAPI can parse repeated keys.
    # Normalize and use the first effect (pipeline currently supports one effect override).
    effects = [e for e in (effects or []) if _normalize_override(e)]
    effect_override = effects[0] if effects else None

    try:
        results = app_pipeline.process_image(
            input_image,
            backgroundId,
            effect_override,
            None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")

    if not results:
        raise HTTPException(status_code=500, detail="Pipeline returned no results.")

    out_bytes = _encode_png_bytes(results[0])
    return Response(content=out_bytes, media_type="image/png", status_code=200)


# ==================================================
# LEGACY ENDPOINT (keep for your local UI/testing)
# ==================================================
@app.post("/process_image")
async def process_image_endpoint(
    image: UploadFile = File(...),
    background_override: str | None = Form(None),
    effect_override: str | None = Form(None),
    accessory_override: str | None = Form(None),
):
    try:
        file_bytes = await image.read()
        from pipeline.image_utils import read_imagefile, encode_image_to_base64

        input_image = read_imagefile(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=("Error reading the uploaded image. " + str(e)))

    if app_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized yet.")

    background_override = _normalize_override(background_override)
    effect_override = _normalize_override(effect_override)
    accessory_override = _normalize_override(accessory_override)

    try:
        results = app_pipeline.process_image(
            input_image,
            background_override,
            effect_override,
            accessory_override,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")

    encoded_results = [encode_image_to_base64(img) for img in results]
    return JSONResponse(content={"results": encoded_results})


# Optional: keep GET /health for your convenience (not part of contract)
@app.get("/health")
async def health_get():
    return {
        "status": "ok",
        "pipeline_initialized": app_pipeline is not None,
        "ai_phase": AI_PHASE,
        "use_openai": os.getenv("USE_OPENAI", "(unset)"),
    }


if __name__ == "__main__":
    # Local dev convenience
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
