# pipeline/image_pipeline.py

import os
import time
import cv2
import json
from concurrent.futures import ThreadPoolExecutor

from pipeline.image_utils import draw_faces
from pipeline import global_vars  # face_detector, bg_remover, accessory_placer, openai_client


def load_all_assets(asset_dirs):
    """
    Loads asset options for all categories.
    For categories with metadata (hats, glasses, masks), keys are loaded from metadata JSON.
    For backgrounds and effects, the file names (without extension) are scanned.
    Always adds "none" if not present.
    """
    all_assets = {}

    for category in ["backgrounds", "effects"]:
        directory = asset_dirs.get(category, "")
        if not os.path.isdir(directory):
            all_assets[category] = ["none"]
        else:
            files = [
                os.path.splitext(f)[0]
                for f in os.listdir(directory)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if "none" not in files:
                files.append("none")
            all_assets[category] = sorted(files)

    for category in ["hats", "glasses", "masks"]:
        meta_file = os.path.join(asset_dirs.get(category, ""), f"{category}_metadata.json")
        if os.path.isfile(meta_file):
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                keys = list(meta.keys())
                if "none" not in keys:
                    keys.append("none")
                all_assets[category] = sorted(keys)
            except Exception as e:
                print(f"Error loading {category} metadata: {e}")
                all_assets[category] = ["none"]
        else:
            all_assets[category] = ["none"]

    return all_assets


def _try_read_background(bg_dir: str, bg_name: str):
    """
    Background assets may be .jpg/.jpeg/.png. Try all.
    Returns (img, path) or (None, None).
    """
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(bg_dir, bg_name + ext)
        img = cv2.imread(p)
        if img is not None:
            return img, p
    return None, None


def _safe_fallback_suggestions():
    """
    Ultra-safe fallback if OpenAI client is missing or misconfigured.
    Returns 3 "none" suggestions.
    """
    from models.image_tags import ImageTagsResponse, ImageTags

    return ImageTagsResponse(
        suggestion1=ImageTags(Background="none", Hats="none", Glasses="none", Effects="none", Masks="none"),
        suggestion2=ImageTags(Background="none", Hats="none", Glasses="none", Effects="none", Masks="none"),
        suggestion3=ImageTags(Background="none", Hats="none", Glasses="none", Effects="none", Masks="none"),
    )


class ImagePipeline:
    """
    Main pipeline combining:
    - face detection
    - MODNet background removal
    - accessory placement
    - optional OpenAI suggestions (with safe offline fallback)
    """

    def __init__(self, asset_dirs, device=None):
        self.device = device or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

        self.face_detector = global_vars.face_detector
        self.bg_remover = global_vars.bg_remover
        self.accessory_placer = global_vars.accessory_placer

        # IMPORTANT: grab openai client but allow it to be missing
        self.openai_client = getattr(global_vars, "openai_client", None)

        self.asset_dirs = asset_dirs
        self.asset_options = load_all_assets(asset_dirs)

        # Keep asset options available globally for deterministic fallback logic
        global_vars.asset_options = self.asset_options

        # Ensure openai_client has asset_options for deterministic fallback
        if self.openai_client is not None and getattr(self.openai_client, "asset_options", None) is not None:
            self.openai_client.asset_options = self.asset_options

    def generate_asset_prompt(self):
        backgrounds = ", ".join(self.asset_options.get("backgrounds", []))
        hats = ", ".join(self.asset_options.get("hats", []))
        glasses = ", ".join(self.asset_options.get("glasses", []))
        effects = ", ".join(self.asset_options.get("effects", []))
        masks = ", ".join(self.asset_options.get("masks", []))

        prompt = (
            "Analyze the uploaded image and generate structured tags for a rule-based editing system. "
            "Return exactly three separate suggestions. Use only the following options:\n"
            f"[Background]: {backgrounds}\n"
            f"[Hats]: {hats}\n"
            f"[Glasses]: {glasses}\n"
            f"[Effects]: {effects}\n"
            f"[Masks]: {masks}\n"
            "If the image shows heart gestures, for example, use 'heart' for Effects and 'glasses_heart' for Glasses. "
            "Try to make funny combinations. Things like a space background and astronaut masks could be combined and triggered by waving the arms as an example."
        )
        return prompt

    def process_image(self, image, background_override, effect_override, accessory_override):
        if image is None:
            raise ValueError("Input image is None.")

        overall_start = time.time()
        print("Starting image processing pipeline...")

        # Step 1: Generate prompt
        t0 = time.time()
        asset_prompt = self.generate_asset_prompt()
        print(f"Asset prompt generated in {time.time() - t0:.3f} seconds.")

        # Step 2: Save temp input
        t1 = time.time()
        temp_path = "./temp_input.jpg"
        cv2.imwrite(temp_path, image)
        print(f"Input image saved to temporary file in {time.time() - t1:.3f} seconds.")

        # Step 3: Suggestions + face detection in parallel
        t2 = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Defensive: if openai_client is missing, do NOT try to call it
            if self.openai_client is None:
                future_suggestions = None
            else:
                future_suggestions = executor.submit(
                    self.openai_client.describe_image_with_retry,
                    temp_path,
                    asset_prompt,
                )

            future_faces = executor.submit(self.face_detector.detect_faces, image)

            # Suggestions must never crash the pipeline
            try:
                if future_suggestions is None:
                    structured_response = _safe_fallback_suggestions()
                else:
                    structured_response = future_suggestions.result()
            except Exception as e:
                print(f"Suggestion generation failed unexpectedly. Using safe fallback. Error: {e}")
                structured_response = _safe_fallback_suggestions()

            faces = future_faces.result()

        print(f"Received structured suggestions and performed face detection in {time.time() - t2:.3f} seconds.")

        # Debug draw
        try:
            debug_img = draw_faces(image.copy(), faces)
            debug_path = "./debug_landmarks_bbox.jpg"
            cv2.imwrite(debug_path, debug_img)
        except Exception as e:
            print(f"Debug draw failed (ignored). Error: {e}")

        suggestions = [
            structured_response.suggestion1,
            structured_response.suggestion2,
            structured_response.suggestion3,
        ]

        # Apply override values if provided.
        if background_override is not None and background_override.strip() != "":
            for suggestion in suggestions:
                suggestion.Background = background_override

            # NOTE: your existing "event_*" behavior kept as-is
            self.asset_dirs["backgrounds"] = "assets/event_backgrounds"
            self.accessory_placer.asset_dirs["backgrounds"] = self.asset_dirs["backgrounds"]
            print(f"Overriding background with: {background_override}")

        if effect_override is not None and effect_override.strip() != "":
            for suggestion in suggestions:
                suggestion.Effects = effect_override

            self.asset_dirs["effects"] = "assets/event_effects"
            self.accessory_placer.asset_dirs["effects"] = self.asset_dirs["effects"]
            print(f"Overriding effect with: {effect_override}")

        if accessory_override is not None and accessory_override.strip() != "":
            for suggestion in suggestions:
                suggestion.Hats = "none"
                suggestion.Glasses = "none"
                suggestion.Masks = "none"

        if not faces:
            print("No faces detected.")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return [image]

        print(f"Face detection reported {len(faces)} faces.")

        results = []

        for idx, suggestion in enumerate(suggestions):
            t_iter = time.time()
            edited_image = image.copy()

            # Step 4: Background Replacement
            if suggestion.Background and suggestion.Background.lower() != "none":
                t_bg = time.time()
                bg_img, _ = _try_read_background(
                    self.accessory_placer.asset_dirs["backgrounds"],
                    suggestion.Background,
                )
                if bg_img is None:
                    print(f"Background '{suggestion.Background}' not found; skipping background replacement.")
                else:
                    bg_img = cv2.resize(bg_img, (edited_image.shape[1], edited_image.shape[0]))
                    t_mask = time.time()
                    mask = self.bg_remover.remove_background(edited_image)
                    print(f"Background removal took {time.time() - t_mask:.3f} seconds.")
                    mask_inv = cv2.bitwise_not(mask)
                    fg = cv2.bitwise_and(edited_image, edited_image, mask=mask)
                    new_bg = cv2.bitwise_and(bg_img, bg_img, mask=mask_inv)
                    edited_image = cv2.add(fg, new_bg)
                    print(f"Background replacement completed in {time.time() - t_bg:.3f} seconds.")

            # Step 5: Assemble accessory spec
            accessories = {
                "hat": suggestion.Hats,
                "glasses": suggestion.Glasses,
                "effect": suggestion.Effects,
                "masks": suggestion.Masks,
            }

            # Step 6: Apply accessory overlays
            t_accessory = time.time()
            for face_info in faces:
                edited_image = self.accessory_placer.apply_accessories(edited_image, face_info, accessories)
            print(f"Accessory application took {time.time() - t_accessory:.3f} seconds.")

            # Step 7: Apply overall effect overlay
            if accessories.get("effect", "none") and str(accessories.get("effect", "none")).lower() != "none":
                t_effect = time.time()
                edited_image = self.accessory_placer.apply_effect(edited_image, accessories["effect"])
                print(f"Effect overlay applied in {time.time() - t_effect:.3f} seconds.")

            # Step 8: Save processed image
            os.makedirs("./output", exist_ok=True)
            output_path = f"./output/result_{idx+1}.jpg"
            cv2.imwrite(output_path, edited_image)
            print(f"Saved result {idx + 1} to {output_path} (iteration took {time.time() - t_iter:.3f} seconds).")
            results.append(edited_image)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        print(f"Total pipeline processing time: {time.time() - overall_start:.3f} seconds.")
        return results
