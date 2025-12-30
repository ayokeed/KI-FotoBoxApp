# pipeline/openai_client.py

import base64
import cv2
import logging
import os
from typing import Optional

from models.image_tags import ImageTags, ImageTagsResponse

log = logging.getLogger("ki-fotobox.openai")


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _openai_enabled(api_key: str) -> bool:
    """
    Feature flag:
      - USE_OPENAI=false => always disable
      - if USE_OPENAI not set => enable only if api_key exists
    """
    flag = os.getenv("USE_OPENAI")
    if flag is not None:
        return _truthy(flag)
    return bool(api_key and api_key.strip())


def _pick_first_non_none(options: list[str]) -> str:
    for o in options:
        if o and o.strip().lower() != "none":
            return o.strip()
    return "none"


def _fallback_from_assets(asset_options: dict) -> ImageTagsResponse:
    """
    Deterministic fallback suggestions derived from local asset options.
    - Exactly 3 suggestions
    - Not completely 'none' if possible
    - Respects: if Masks != 'none' then Hats/Glasses must be 'none' and vice versa
    """
    bgs = asset_options.get("backgrounds", ["none"])
    hats = asset_options.get("hats", ["none"])
    glasses = asset_options.get("glasses", ["none"])
    effects = asset_options.get("effects", ["none"])
    masks = asset_options.get("masks", ["none"])

    bg = _pick_first_non_none(bgs)
    hat = _pick_first_non_none(hats)
    glas = _pick_first_non_none(glasses)
    eff = _pick_first_non_none(effects)
    msk = _pick_first_non_none(masks)

    s1 = ImageTags(Background=bg, Hats="none", Glasses="none", Effects=eff, Masks="none")
    s2 = ImageTags(Background="none", Hats=hat, Glasses=glas, Effects="none", Masks="none")
    s3 = ImageTags(Background="none", Hats="none", Glasses="none", Effects="none", Masks=msk)

    def all_none(t: ImageTags) -> bool:
        return (
            t.Background.lower() == "none"
            and t.Hats.lower() == "none"
            and t.Glasses.lower() == "none"
            and t.Effects.lower() == "none"
            and t.Masks.lower() == "none"
        )

    # If you have no assets at all, ensure suggestions aren't all-none.
    # These placeholders will be safely ignored by the pipeline if assets don't exist.
    if all_none(s1):
        s1 = ImageTags(Background="fallback_bg", Hats="none", Glasses="none", Effects="none", Masks="none")
    if all_none(s2):
        s2 = ImageTags(Background="none", Hats="fallback_hat", Glasses="none", Effects="none", Masks="none")
    if all_none(s3):
        s3 = ImageTags(Background="none", Hats="none", Glasses="none", Effects="fallback_effect", Masks="none")

    return ImageTagsResponse(suggestion1=s1, suggestion2=s2, suggestion3=s3)


class OpenAI_Client:
    """
    OpenAI is OPTIONAL:
      - If USE_OPENAI=false or API key missing/invalid: return deterministic fallback tags
      - If OpenAI errors: return deterministic fallback tags
      - Never raises due to OpenAI
    """

    def __init__(self, api_key: str, model: str, asset_options: Optional[dict] = None) -> None:
        self.api_key = api_key or ""
        self.model = model
        self.asset_options = asset_options or {}

        self.enabled = _openai_enabled(self.api_key)
        self.client = None

        if self.enabled:
            try:
                # Lazy-import would be even more offline-safe,
                # but keeping your original dependency here.
                from openai import OpenAI  # type: ignore

                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                log.exception("OpenAI init failed; falling back. Error=%r", e)
                self.enabled = False
                self.client = None
        else:
            log.info("OpenAI disabled (USE_OPENAI=false or missing key). Using fallback suggestions.")

    def encode_image(self, image_path: str) -> str:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")

        max_width = 640
        if img.shape[1] > max_width:
            scale = max_width / img.shape[1]
            new_dimensions = (max_width, int(img.shape[0] * scale))
            img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA)

        ret, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if not ret:
            raise ValueError("JPEG encoding failed")

        return base64.b64encode(buf).decode("utf-8")

    def describe_image(self, image_path: str, prompt: str) -> ImageTagsResponse:
        if not self.enabled or self.client is None:
            return _fallback_from_assets(self.asset_options)

        try:
            base64_image = self.encode_image(image_path)

            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant that analyzes images for a fun photobox application. "
                            "Generate strictly structured JSON tags for a rule-based editing system. "
                            "Return exactly three separate suggestions. Each suggestion must contain the keys "
                            "Background, Hats, Glasses, Effects, Masks using only the allowed options. "
                            "You are not allowed to make a suggestion that is completely 'none'. "
                            "If you choose a Mask you can't choose Hats or Glasses for that suggestion and vice versa."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                            },
                        ],
                    },
                ],
                temperature=0.8,
                response_format=ImageTagsResponse,
            )

            parsed = response.choices[0].message.parsed
            if parsed is None:
                log.warning("OpenAI returned parsed=None; using fallback suggestions.")
                return _fallback_from_assets(self.asset_options)

            return parsed

        except Exception as e:
            # Non-negotiable: never cause 500 due to OpenAI
            log.exception("OpenAI call failed; using fallback. Error=%r", e)
            return _fallback_from_assets(self.asset_options)

    def describe_image_with_retry(self, image_path: str, prompt: str = "", retries: int = 3) -> ImageTagsResponse:
        # Offline path: no retries
        if not self.enabled or self.client is None:
            return _fallback_from_assets(self.asset_options)

        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                return self.describe_image(image_path, prompt)
            except Exception as e:
                # describe_image already catches, but keep ultra-safe
                last_error = e
                log.warning("Attempt %s/%s failed: %r", attempt + 1, retries, e)

        # Non-negotiable: do NOT raise
        log.error("Retries exhausted; returning fallback. Last error=%r", last_error)
        return _fallback_from_assets(self.asset_options)
