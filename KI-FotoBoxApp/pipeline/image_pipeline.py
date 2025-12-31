# pipeline/image_pipeline.py  (UPDATED)
from __future__ import annotations

import os
import time
import cv2
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any

from pipeline.image_utils import draw_faces
from pipeline import global_vars


def _try_read_background(bg_dir: str, bg_name: str):
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(bg_dir, bg_name + ext)
        img = cv2.imread(p)
        if img is not None:
            return img, p
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

    bg = (background_override or "").strip() or "none"
    eff = (effect_override or "").strip() or "none"

    # If accessory_override is provided, your prior logic set hats/glasses/masks to none
    if (accessory_override or "").strip():
        return ImageTags(Background=bg, Hats="none", Glasses="none", Effects=eff, Masks="none")

    return ImageTags(Background=bg, Hats="none", Glasses="none", Effects=eff, Masks="none")


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

        # Keep dirs
        self.asset_dirs = dict(asset_dirs)

        # Lazy-load heavy components ONLY when needed
        self.face_detector = None
        self.bg_remover = None
        self.accessory_placer = None
        self.openai_client = None

        # Asset options should be prepared by FastAPI lifespan using pipeline/assets.py
        # and stored into global_vars.
        self.asset_options = global_vars.get_asset_options() or {}

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
                # provide asset options for deterministic output (if your client supports it)
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
            (background_override and background_override.strip()),
            (effect_override and effect_override.strip()),
            (accessory_override and accessory_override.strip()),
        )

        # ---- Face detection ----
        t_faces = time.time()
        faces = self.face_detector.detect_faces(image)
        print(f"Face detection done in {time.time() - t_faces:.3f} seconds.")

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
        # Uses OpenAI but MUST NOT crash the pipeline
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
        Lazily loads bg remover only if a background replacement is actually needed.
        """
        edited_image = image.copy()

        # Background override logic (your old event dirs)
        if background_override is not None and background_override.strip() != "":
            suggestion.Background = background_override
            self.asset_dirs["backgrounds"] = "assets/event_backgrounds"
            self.accessory_placer.asset_dirs["backgrounds"] = self.asset_dirs["backgrounds"]
            print(f"Overriding background with: {background_override}")

        if effect_override is not None and effect_override.strip() != "":
            suggestion.Effects = effect_override
            self.asset_dirs["effects"] = "assets/event_effects"
            self.accessory_placer.asset_dirs["effects"] = self.asset_dirs["effects"]
            print(f"Overriding effect with: {effect_override}")

        # Background replacement (only if not 'none')
        if suggestion.Background and str(suggestion.Background).lower() != "none":
            bg_img, _ = _try_read_background(
                self.accessory_placer.asset_dirs["backgrounds"],
                suggestion.Background,
            )
            if bg_img is None:
                print(f"Background '{suggestion.Background}' not found; skipping background replacement.")
            else:
                self._ensure_bg_remover()  # lazy: imports MODNet only now
                bg_img = cv2.resize(bg_img, (edited_image.shape[1], edited_image.shape[0]))
                mask = self.bg_remover.remove_background(edited_image)
                mask_inv = cv2.bitwise_not(mask)
                fg = cv2.bitwise_and(edited_image, edited_image, mask=mask)
                new_bg = cv2.bitwise_and(bg_img, bg_img, mask=mask_inv)
                edited_image = cv2.add(fg, new_bg)

        accessories = {
            "hat": getattr(suggestion, "Hats", "none"),
            "glasses": getattr(suggestion, "Glasses", "none"),
            "effect": getattr(suggestion, "Effects", "none"),
            "masks": getattr(suggestion, "Masks", "none"),
        }

        # Apply accessories to each face
        for face_info in faces:
            edited_image = self.accessory_placer.apply_accessories(edited_image, face_info, accessories)

        # Apply overall effect overlay
        if accessories.get("effect", "none") and str(accessories["effect"]).lower() != "none":
            edited_image = self.accessory_placer.apply_effect(edited_image, accessories["effect"])

        return edited_image
