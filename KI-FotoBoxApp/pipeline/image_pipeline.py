# pipeline/image_pipeline.py  (UPDATED - absolute event dirs via asset_dirs + no OpenCV imread WARN spam)
from __future__ import annotations

import os
import time
import cv2
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Tuple

from pipeline.image_utils import draw_faces
from pipeline import global_vars


# ----------------------------
# Helpers
# ----------------------------

def _norm_id(x: Optional[str]) -> str:
    return (x or "").strip()

def _try_find_file_by_id(
    dir_path: str,
    asset_id: str,
    exts=(".png", ".jpg", ".jpeg"),
) -> Optional[str]:
    """
    Find an asset file by id (filename without extension) without calling cv2.imread.
    This avoids OpenCV WARN spam when files are missing.
    Returns full path or None.
    """
    if not dir_path or not asset_id:
        return None
    for ext in exts:
        p = os.path.join(dir_path, asset_id + ext)
        if os.path.isfile(p):
            return p
    return None

def _resolve_asset_file(
    asset_id: str,
    primary_dir: str,
    fallback_dir: str,
    kind: str,
    exts=(".png", ".jpg", ".jpeg"),
    strict: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve an asset file path by checking:
      1) primary_dir (e.g. /app/assets/event_backgrounds)
      2) fallback_dir (e.g. /app/assets/backgrounds)

    Returns (path, where) where 'where' is the directory used ("primary"|"fallback").
    If strict=True, raises ValueError when not found.
    """
    asset_id = _norm_id(asset_id)
    if not asset_id or asset_id.lower() == "none":
        return None, None

    p1 = _try_find_file_by_id(primary_dir, asset_id, exts=exts)
    if p1:
        return p1, "primary"

    p2 = _try_find_file_by_id(fallback_dir, asset_id, exts=exts)
    if p2:
        return p2, "fallback"

    msg = f"{kind} asset '{asset_id}' not found in '{primary_dir}' or '{fallback_dir}'."
    if strict:
        raise ValueError(msg)
    print(msg)
    return None, None

def _safe_fallback_suggestions():
    from models.image_tags import ImageTagsResponse, ImageTags

    return ImageTagsResponse(
        suggestion1=ImageTags(Background="none", Hats="none", Glasses="none", Effects="none", Masks="none"),
        suggestion2=ImageTags(Background="none", Hats="none", Glasses="none", Effects="none", Masks="none"),
        suggestion3=ImageTags(Background="none", Hats="none", Glasses="none", Effects="none", Masks="none"),
    )

def _single_suggestion_from_overrides(
    background_override: Optional[str],
    effect_override: Optional[str],
    accessory_override: Optional[str],
):
    """
    Creates a single deterministic "suggestion-like" object from overrides,
    so we can reuse the existing apply logic without calling OpenAI.
    """
    from models.image_tags import ImageTags

    bg = _norm_id(background_override) or "none"
    eff = _norm_id(effect_override) or "none"

    # Accessory override not used yet (kept for compatibility)
    if _norm_id(accessory_override):
        return ImageTags(Background=bg, Hats="none", Glasses="none", Effects=eff, Masks="none")

    return ImageTags(Background=bg, Hats="none", Glasses="none", Effects=eff, Masks="none")


# ----------------------------
# Pipeline
# ----------------------------

class ImagePipeline:
    """
    Main pipeline combining:
    - face detection
    - background removal (MODNet) (LAZY-loaded, only during /process)
    - accessory placement
    - optional OpenAI suggestions (LAZY-loaded, only when overrides NOT provided)
    """

    def __init__(self, asset_dirs: Dict[str, str], device: Optional[str] = None):
        self.device = device or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

        # Keep dirs (mutable, but avoid overwriting them for overrides)
        self.asset_dirs = dict(asset_dirs)

        # Lazy-load heavy components ONLY when needed
        self.face_detector = None
        self.bg_remover = None
        self.accessory_placer = None
        self.openai_client = None

        # Asset options prepared by FastAPI lifespan using pipeline/assets.py
        self.asset_options = global_vars.get_asset_options() or {}

        # If true and an override id is provided but the file doesn't exist -> raise
        self.strict_override_assets = str(os.environ.get("STRICT_OVERRIDE_ASSETS", "0")).lower() in ("1", "true", "yes", "y")

    def _ensure_face_detector(self):
        if self.face_detector is None:
            self.face_detector = global_vars.get_face_detector()

    def _ensure_bg_remover(self):
        if self.bg_remover is None:
            self.bg_remover = global_vars.get_bg_remover()

    def _ensure_accessory_placer(self):
        if self.accessory_placer is None:
            self.accessory_placer = global_vars.get_accessory_placer(self.asset_dirs)

    def _ensure_openai_client(self):
        if self.openai_client is None:
            self.openai_client = global_vars.get_openai_client()
            if self.openai_client is not None:
                try:
                    self.openai_client.asset_options = self.asset_options
                except Exception:
                    pass

    def generate_asset_prompt(self) -> str:
        backgrounds = ", ".join(self.asset_options.get("backgrounds", []))
        hats = ", ".join(self.asset_options.get("hats", []))
        glasses = ", ".join(self.asset_options.get("glasses", []))
        effects = ", ".join(self.asset_options.get("effects", []))
        masks = ", ".join(self.asset_options.get("masks", []))

        return (
            "Analyze the uploaded image and generate structured tags for a rule-based editing system. "
            "Return exactly three separate suggestions. Use only the following options:\n"
            f"[Background]: {backgrounds}\n"
            f"[Hats]: {hats}\n"
            f"[Glasses]: {glasses}\n"
            f"[Effects]: {effects}\n"
            f"[Masks]: {masks}\n"
            "If the image shows heart gestures, for example, use 'heart' for Effects and 'glasses_heart' for Glasses. "
            "Try to make funny combinations."
        )

    def process_image(
        self,
        image,
        background_override: Optional[str] = None,
        effect_override: Optional[str] = None,
        accessory_override: Optional[str] = None,
    ):
        """
        REQUIRED semantics:
        - If overrides provided (background/effect/accessory), DO NOT call OpenAI.
          Apply exactly ONE deterministic edit result and return [single_image].
        - If no overrides, you may generate 3 OpenAI suggestions and return 3 results (legacy behavior).
        """
        if image is None:
            raise ValueError("Input image is None.")

        overall_start = time.time()
        print("Starting image processing pipeline...")

        self._ensure_accessory_placer()
        self._ensure_face_detector()

        has_overrides = any(
            x is not None and str(x).strip()
            for x in (background_override, effect_override, accessory_override)
        )

        print(f"[DEBUG] overrides: background='{_norm_id(background_override)}' effect='{_norm_id(effect_override)}' accessory='{_norm_id(accessory_override)}'")
        print(f"[DEBUG] strict_override_assets={self.strict_override_assets}")
        print(f"[DEBUG] base asset_dirs={self.asset_dirs}")

        # ---- Face detection ----
        t_faces = time.time()
        faces = self.face_detector.detect_faces(image)
        print(f"Face detection done in {time.time() - t_faces:.3f} seconds. faces={len(faces) if faces else 0}")

        # Debug draw (optional)
        try:
            debug_img = draw_faces(image.copy(), faces)
            os.makedirs("./output", exist_ok=True)
            cv2.imwrite("./output/debug_landmarks_bbox.jpg", debug_img)
        except Exception as e:
            print(f"Debug draw failed (ignored). Error: {e}")

        # If overrides present, skip suggestions entirely and process exactly one output
        if has_overrides:
            suggestion = _single_suggestion_from_overrides(background_override, effect_override, accessory_override)
            edited = self._apply_one(image, faces, suggestion, background_override, effect_override)
            print(f"Total deterministic processing time: {time.time() - overall_start:.3f} seconds.")
            return [edited]

        # ---- Legacy suggestions path (3 results) ----
        self._ensure_openai_client()

        t_prompt = time.time()
        asset_prompt = self.generate_asset_prompt()
        print(f"Asset prompt generated in {time.time() - t_prompt:.3f} seconds.")

        temp_path = "./temp_input.jpg"
        t_save = time.time()
        cv2.imwrite(temp_path, image)
        print(f"Input image saved to temporary file in {time.time() - t_save:.3f} seconds.")

        t2 = time.time()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_suggestions = None
            if self.openai_client is not None:
                future_suggestions = executor.submit(
                    self.openai_client.describe_image_with_retry,
                    temp_path,
                    asset_prompt,
                )

            try:
                if future_suggestions is None:
                    structured_response = _safe_fallback_suggestions()
                else:
                    structured_response = future_suggestions.result()
            except Exception as e:
                print(f"Suggestion generation failed unexpectedly. Using safe fallback. Error: {e}")
                structured_response = _safe_fallback_suggestions()

        print(f"Suggestions received in {time.time() - t2:.3f} seconds.")

        if os.path.exists(temp_path):
            os.remove(temp_path)

        suggestions = [
            structured_response.suggestion1,
            structured_response.suggestion2,
            structured_response.suggestion3,
        ]

        # If no faces, return original for each suggestion
        if not faces:
            print("No faces detected.")
            return [image for _ in suggestions]

        results = []
        for idx, suggestion in enumerate(suggestions):
            edited = self._apply_one(image, faces, suggestion, None, None)
            os.makedirs("./output", exist_ok=True)
            cv2.imwrite(f"./output/result_{idx+1}.jpg", edited)
            results.append(edited)

        print(f"Total pipeline processing time: {time.time() - overall_start:.3f} seconds.")
        return results

    def _apply_one(self, image, faces, suggestion, background_override, effect_override):
        """
        Apply one suggestion deterministically.
        - Supports BOTH event_* and normal asset folders.
        - Uses asset_dirs["event_*"] if present (no hardcoded relative paths).
        - Avoids OpenCV WARN spam by testing existence with os.path.isfile.
        - Lazy loads bg remover only if background replacement is actually needed.
        """
        edited_image = image.copy()

        # Apply explicit overrides to suggestion
        bg_override = _norm_id(background_override)
        eff_override = _norm_id(effect_override)

        if bg_override:
            suggestion.Background = bg_override
            print(f"[DEBUG] override backgroundId='{bg_override}'")

        if eff_override:
            suggestion.Effects = eff_override
            print(f"[DEBUG] override effectId='{eff_override}'")

        # Directories (event first, then fallback)
        fallback_bg_dir = self.asset_dirs.get("backgrounds", "/app/assets/backgrounds")
        fallback_eff_dir = self.asset_dirs.get("effects", "/app/assets/effects")

        event_bg_dir = self.asset_dirs.get("event_backgrounds") or ""
        event_eff_dir = self.asset_dirs.get("event_effects") or ""

        # Background replacement (only if not 'none')
        bg_id = _norm_id(getattr(suggestion, "Background", None))
        if bg_id and bg_id.lower() != "none":
            bg_path, where = _resolve_asset_file(
                asset_id=bg_id,
                primary_dir=event_bg_dir,
                fallback_dir=fallback_bg_dir,
                kind="Background",
                exts=(".jpg", ".jpeg", ".png"),
                strict=self.strict_override_assets and bool(bg_override),
            )

            if bg_path:
                bg_img = cv2.imread(bg_path)  # background should be BGR 3ch
                if bg_img is None:
                    msg = f"Background '{bg_id}' resolved to '{bg_path}' but cv2.imread returned None."
                    if self.strict_override_assets and bool(bg_override):
                        raise ValueError(msg)
                    print(msg)
                else:
                    print(f"[DEBUG] background resolved ({where}) -> {bg_path}")

                    self._ensure_bg_remover()  # lazy load MODNet only now

                    bg_img = cv2.resize(bg_img, (edited_image.shape[1], edited_image.shape[0]))

                    mask = self.bg_remover.remove_background(edited_image)

                    if mask is None:
                        msg = "bg_remover.remove_background returned None."
                        if self.strict_override_assets and bool(bg_override):
                            raise ValueError(msg)
                        print(msg)
                    else:
                        # Ensure mask is single channel uint8
                        if len(mask.shape) == 3:
                            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                        if mask.dtype != "uint8":
                            mask = mask.astype("uint8")

                        mask_inv = cv2.bitwise_not(mask)

                        fg = cv2.bitwise_and(edited_image, edited_image, mask=mask)
                        new_bg = cv2.bitwise_and(bg_img, bg_img, mask=mask_inv)
                        edited_image = cv2.add(fg, new_bg)
            else:
                print(f"[DEBUG] background '{bg_id}' not applied (not found).")

        accessories = {
            "hat": getattr(suggestion, "Hats", "none"),
            "glasses": getattr(suggestion, "Glasses", "none"),
            "effect": getattr(suggestion, "Effects", "none"),
            "masks": getattr(suggestion, "Masks", "none"),
        }

        # Apply accessories to each face
        if faces:
            for face_info in faces:
                edited_image = self.accessory_placer.apply_accessories(edited_image, face_info, accessories)
        else:
            print("[DEBUG] no faces -> skipping per-face accessories")

        # Apply overall effect overlay
        effect_id = _norm_id(accessories.get("effect", None))
        if effect_id and effect_id.lower() != "none":
            eff_path, where = _resolve_asset_file(
                asset_id=effect_id,
                primary_dir=event_eff_dir,
                fallback_dir=fallback_eff_dir,
                kind="Effect",
                exts=(".png", ".jpg", ".jpeg"),
                strict=self.strict_override_assets and bool(eff_override),
            )

            if eff_path:
                print(f"[DEBUG] effect resolved ({where}) -> {eff_path}")
                # Ensure accessory_placer resolves from the correct directory
                try:
                    self.accessory_placer.asset_dirs["effects"] = event_eff_dir if where == "primary" else fallback_eff_dir
                except Exception:
                    pass
            else:
                print(f"[DEBUG] effect '{effect_id}' not found in event/fallback dirs (apply_effect may no-op).")
                try:
                    self.accessory_placer.asset_dirs["effects"] = fallback_eff_dir
                except Exception:
                    pass

            edited_image = self.accessory_placer.apply_effect(edited_image, effect_id)

        return edited_image
