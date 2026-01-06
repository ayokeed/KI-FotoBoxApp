# app/main.py
# Phase 2 (contract-compliant, Azure-safe AI service)
#
# UPDATED FOR ACCESSORIES CONTRACT:
# - /suggest returns: backgrounds, effects, accessories   (unified list)
# - GET /assets/{category}/{asset_id} serves preview bytes (no FS paths exposed)
# - /process accepts:
#     - backgroundId (string)
#     - effects (repeated form fields)  -> kept
#     - accessories (single form field JSON string) -> NEW
#
# Non-negotiables kept:
# - No FS paths exposed
# - /suggest stays fast and must not load ML
# - ML loads lazily only on /process or /process_image

from __future__ import annotations

import os
import io
import asyncio
from pathlib import Path
from typing import Optional, List, Any, Dict

import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# -----------------------------
# Config
# -----------------------------
AI_PHASE = int(os.getenv("AI_PHASE", "2"))

AI_USE_OPENAI = os.getenv("AI_USE_OPENAI", "0") == "1"
if AI_USE_OPENAI:
    os.environ["USE_OPENAI"] = "true"

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = int(os.getenv("AI_MAX_IMAGE_BYTES", "15000000"))  # 15MB

# Prefer explicit public base URL behind proxies (Azure). Example:
# PUBLIC_BASE_URL=https://photobooth-ai-service-xxxx.azurewebsites.net
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")

# CORS (only needed if browser fetches /assets/* or /suggest directly)
# Example:
# CORS_ORIGINS=https://your-frontend-host,https://your-backend-host
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

# Make ASSET_DIRS absolute and container-safe
BASE_DIR = Path(__file__).resolve().parent              # .../KI-FotoBoxApp/app
APP_ROOT = BASE_DIR.parent                              # .../KI-FotoBoxApp
ASSETS_ROOT = (APP_ROOT / "assets").resolve()

ASSET_DIRS = {
    "backgrounds":        str(ASSETS_ROOT / "backgrounds"),
    "event_backgrounds":  str(ASSETS_ROOT / "event_backgrounds"),
    "hats":               str(ASSETS_ROOT / "hats"),
    "glasses":            str(ASSETS_ROOT / "glasses"),
    "effects":            str(ASSETS_ROOT / "effects"),
    "event_effects":      str(ASSETS_ROOT / "event_effects"),
    "masks":              str(ASSETS_ROOT / "masks"),
}

ALLOWED_CATEGORIES = set(ASSET_DIRS.keys())

# MODNet checkpoint path (container-correct)
DEFAULT_MODNET_CKPT = str((APP_ROOT / "checkpoints" / "modnet_photographic_portrait_matting.ckpt").resolve())

# Global lazy-loaded pipeline
app_pipeline = None
_pipeline_lock: asyncio.Lock | None = None


# ==================================================
# GENERAL HELPERS
# ==================================================

def _normalize_override(v: str | None) -> str | None:
    if v is None:
        return None
    v = v.strip()
    if v == "" or v.lower() in {"string", "none", "null", "undefined"}:
        return None
    return v


def _choose_device() -> str:
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


async def _read_upload(file: UploadFile) -> bytes:
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
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _to_id_label_list(items: Any) -> List[Dict[str, Any]]:
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


def _base_url(request: Request) -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    return str(request.base_url).rstrip("/")


def _encode_png_bytes(img: Any) -> bytes:
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


def _parse_accessories_json(accessories_json: str | None) -> List[Dict[str, Any]]:
    """
    accessories is sent as ONE multipart field containing JSON:
      '[{"id":"round_glasses","scale":1.1,"rotation":0}]'
    """
    accessories_json = _normalize_override(accessories_json)
    if not accessories_json:
        return []

    try:
        import json
        data = json.loads(accessories_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid accessories JSON. {str(e)}")

    if data is None:
        return []

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="Invalid accessories JSON: expected array.")

    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        aid = item.get("id") or item.get("name")
        if not aid or not isinstance(aid, str):
            continue
        out.append(item)
    return out


# ==================================================
# LAZY MODEL + PIPELINE LOADING (CRITICAL FIX)
# ==================================================

async def _ensure_models_loaded(app: FastAPI) -> None:
    global app_pipeline, _pipeline_lock

    if app_pipeline is not None:
        return

    if _pipeline_lock is None:
        _pipeline_lock = asyncio.Lock()

    async with _pipeline_lock:
        if app_pipeline is not None:
            return

        device = _choose_device()

        from pipeline.face_detection import FaceDetector
        from pipeline.background_removal import BackgroundRemover
        from pipeline.accessory_application import AccessoryPlacer
        from pipeline import global_vars
        from pipeline.image_pipeline import ImagePipeline
        from pipeline.openai_client import OpenAI_Client

        global_vars.face_detector = FaceDetector(device=device)

        ckpt = os.getenv("MODNET_CKPT", "").strip() or DEFAULT_MODNET_CKPT
        global_vars.bg_remover = BackgroundRemover(model_path=ckpt, device=device)

        global_vars.accessory_placer = AccessoryPlacer(ASSET_DIRS)

        asset_options = getattr(app.state, "asset_options", {}) or {}
        openai_api_key = (os.getenv("OPENAI_API_KEY", "").strip()
                          or os.getenv("OPEN_AI_API_KEY", "").strip())

        global_vars.openai_client = OpenAI_Client(
            api_key=openai_api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            asset_options=asset_options,
        )

        app_pipeline = ImagePipeline(asset_dirs=ASSET_DIRS, device=device)

        print(
            f"[AI] Models + pipeline loaded lazily. Device={device}. ckpt={ckpt} "
            f"USE_OPENAI={os.getenv('USE_OPENAI', '(unset)')} AI_USE_OPENAI={AI_USE_OPENAI}"
        )


# ==================================================
# SUGGEST RESPONSE (Option A: short HTTP previewUrl)
# ==================================================

def _build_suggest_response_option_a(
    request: Request,
    asset_options: Any,
    asset_index: Dict[str, Dict[str, Dict[str, str]]] | None = None,
) -> Dict[str, Any]:
    """
    Contract-compliant JSON WITHOUT exposing filesystem paths.
    previewUrl is a short HTTP URL:
      {base}/assets/{category}/{asset_id}

    Returns unified:
      - backgrounds: [{id,label,previewUrl}]
      - effects:     [{id,label,type,previewUrl}]
      - accessories: [{id,label,category,previewUrl,params}]
    """
    if not isinstance(asset_options, dict):
        asset_options = {}

    base = _base_url(request)
    idx = asset_index or {}

    def has_preview(category: str, asset_id: str) -> bool:
        if asset_id.lower() == "none":
            return False
        return asset_id in (idx.get(category) or {})

    def cat_items(category: str) -> List[Dict[str, Any]]:
        raw = asset_options.get(category)
        items = _to_id_label_list(raw)
        out: List[Dict[str, Any]] = []
        for it in items:
            aid = it["id"]
            out.append(
                {
                    "id": aid,
                    "label": it["label"],
                    "previewUrl": (f"{base}/assets/{category}/{aid}" if has_preview(category, aid) else None),
                }
            )
        return out

    # Backgrounds
    backgrounds_out = cat_items("backgrounds")

    # Effects
    effects_raw = asset_options.get("effects") or asset_options.get("filters") or asset_options.get("fx")
    effects_base = _to_id_label_list(effects_raw)
    effects_out: List[Dict[str, Any]] = []
    for fx in effects_base:
        aid = fx["id"]
        effects_out.append(
            {
                "id": aid,
                "label": fx["label"],
                "type": "filter",
                "previewUrl": (f"{base}/assets/{aid}" if has_preview("effects", aid) else None),
            }
        )

    # Accessories unified (category = hats|glasses|masks)
    accessories_out: List[Dict[str, Any]] = []

    def add_accessories(category: str, default_params: Dict[str, Any]):
        for it in cat_items(category):
            accessories_out.append(
                {
                    "id": it["id"],
                    "label": it["label"],
                    "category": category,
                    "previewUrl": it["previewUrl"],
                    "params": default_params,
                }
            )

    add_accessories("glasses", {"anchor": "eyes"})
    add_accessories("hats", {"anchor": "head"})
    add_accessories("masks", {"anchor": "face"})

    return {
        "backgrounds": backgrounds_out,
        "effects": effects_out,
        "accessories": accessories_out,
    }


# ==================================================
# LIFESPAN: ASSETS ONLY (FAST)
# ==================================================

@asynccontextmanager
async def lifespan_assets_only(app: FastAPI):
    """
    Startup must be fast on Azure:
    - Load asset options + build preview index only
    - Do NOT import/instantiate ML models here
    """
    try:
        from pipeline.assets import load_all_assets, build_asset_preview_index
    except Exception as e:
        raise RuntimeError(
            "Missing pipeline.assets functions. Ensure KI-FotoBoxApp/pipeline/assets.py defines:\n"
            "- load_all_assets(asset_dirs) -> dict\n"
            "- build_asset_preview_index(asset_dirs) -> dict\n"
        ) from e

    app.state.asset_options = load_all_assets(ASSET_DIRS)
    app.state.asset_preview_index = build_asset_preview_index(ASSET_DIRS)

    idx = app.state.asset_preview_index
    print(
        "[AI] Assets loaded. "
        + " ".join([f"{k}={len(idx.get(k, {}))}" for k in sorted(ALLOWED_CATEGORIES)])
        + f" assets_root={ASSETS_ROOT} default_ckpt={DEFAULT_MODNET_CKPT}"
    )

    yield


app = FastAPI(lifespan=lifespan_assets_only)

# CORS only if needed (browser fetch)
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Type"],
        max_age=600,
    )


# ==================================================
# CONTRACT ENDPOINTS
# ==================================================

@app.post("/health")
async def health_post() -> Response:
    return Response(status_code=200)


@app.get("/health")
async def health_get():
    idx = getattr(app.state, "asset_preview_index", {}) or {}
    return {
        "status": "ok",
        "pipeline_initialized": app_pipeline is not None,
        "ai_phase": AI_PHASE,
        "use_openai": os.getenv("USE_OPENAI", "(unset)"),
        "assets_root": str(ASSETS_ROOT),
        "default_ckpt": DEFAULT_MODNET_CKPT,
        "modnet_ckpt_env": os.getenv("MODNET_CKPT", ""),
        "app_root": str(APP_ROOT),
        "assets_count": {k: len(idx.get(k, {})) for k in sorted(ALLOWED_CATEGORIES)},
        "public_base_url": PUBLIC_BASE_URL,
        "cors_origins": CORS_ORIGINS,
    }


@app.get("/assets/{category}/{asset_id}")
async def get_asset_preview(category: str, asset_id: str):
    if category not in ALLOWED_CATEGORIES:
        raise HTTPException(status_code=404, detail="Unknown category")

    if asset_id.lower() == "none":
        raise HTTPException(status_code=404, detail="No preview for 'none'")

    index = getattr(app.state, "asset_preview_index", {}) or {}
    cat = index.get(category) or {}
    meta = cat.get(asset_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Asset not found")

    path = meta.get("path")
    mime = meta.get("mime") or "application/octet-stream"
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Asset missing on disk")

    def _iterfile():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(256 * 1024)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(_iterfile(), media_type=mime)


@app.post("/suggest")
async def suggest(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    """
    Must be fast and must not trigger model loading.
    Returns short HTTP preview URLs (Option A).
    """
    _ = await _read_upload(file)

    asset_options = getattr(app.state, "asset_options", None)
    if asset_options is None:
        raise HTTPException(status_code=503, detail="Assets not loaded yet.")

    asset_index = getattr(app.state, "asset_preview_index", None)
    payload = _build_suggest_response_option_a(request, asset_options, asset_index)
    return JSONResponse(content=payload, status_code=200)


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    backgroundId: Optional[str] = Form(default=None),
    effects: Optional[List[str]] = Form(default=None),
    accessories: Optional[str] = Form(default=None),  # NEW: JSON string
) -> Response:
    file_bytes = await _read_upload(file)

    try:
        from pipeline.image_utils import read_imagefile
        input_image = read_imagefile(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=("Error reading the uploaded image. " + str(e)))

    await _ensure_models_loaded(app)

    backgroundId = _normalize_override(backgroundId)

    # effects are repeated fields in multipart (effects=confetti, effects=hearts, ...)
    effects_norm = [e for e in (effects or []) if _normalize_override(e)]
    effect_override = effects_norm[0] if effects_norm else None

    # accessories are sent as ONE json field (string)
    accessories_list = _parse_accessories_json(accessories)
    accessory_ids = [a.get("id") for a in accessories_list if isinstance(a.get("id"), str)]
    accessory_override = accessory_ids[0] if accessory_ids else None

    try:
        results = app_pipeline.process_image(
            input_image,
            backgroundId,
            effect_override,
            accessory_override,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")

    if not results:
        raise HTTPException(status_code=500, detail="Pipeline returned no results.")

    out_bytes = _encode_png_bytes(results[0])
    return Response(content=out_bytes, media_type="image/png", status_code=200)


@app.post("/process_image")
async def process_image_endpoint(
    image: UploadFile = File(...),
    background_override: str | None = Form(None),
    effect_override: str | None = Form(None),
    accessory_override: str | None = Form(None),
    accessories: str | None = Form(None),  # NEW: optional json field
):
    try:
        file_bytes = await image.read()
        from pipeline.image_utils import read_imagefile, encode_image_to_base64
        input_image = read_imagefile(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=("Error reading the uploaded image. " + str(e)))

    await _ensure_models_loaded(app)

    background_override = _normalize_override(background_override)
    effect_override = _normalize_override(effect_override)
    accessory_override = _normalize_override(accessory_override)

    accessories_list = _parse_accessories_json(accessories)
    accessory_ids = [a.get("id") for a in accessories_list if isinstance(a.get("id"), str)]
    if accessory_ids:
        accessory_override = accessory_ids[0]

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
