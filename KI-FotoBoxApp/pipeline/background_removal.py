# pipeline/background_removal.py
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)


def _resolve_ckpt_path(model_path: str | None) -> Path:
    """
    Default must work in Azure container.
    - If model_path is provided: use it.
    - Else use env MODNET_CKPT
    - Else default to checkpoints/modnet_photographic_portrait_matting.ckpt
    Resolution order for relative paths:
      1) /app/<relative> (Azure container layout)
      2) <cwd>/<relative>
    """
    ckpt_str = (model_path or "").strip()
    if not ckpt_str:
        ckpt_str = (os.getenv("MODNET_CKPT", "").strip()
                    or "checkpoints/modnet_photographic_portrait_matting.ckpt")

    p = Path(ckpt_str)
    if p.is_absolute():
        return p

    app_candidate = Path("/app") / p
    if app_candidate.is_file():
        return app_candidate

    return (Path(os.getcwd()) / p).resolve()


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """
    Support:
      - raw state_dict: { "layer.weight": tensor, ... }
      - wrapped dict: { "state_dict": { ... } }
    """
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        # if it already looks like a state_dict, return it
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj

    raise TypeError(f"Unsupported checkpoint format: {type(ckpt_obj)}")


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], bool]:
    """
    If the checkpoint was saved under torch.nn.DataParallel, keys are prefixed with 'module.'.
    Strip it so it matches a normal (non-DataParallel) model.
    """
    keys = list(state.keys())
    if not keys:
        return state, False

    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state.items()}, True

    if any(k.startswith("module.") for k in keys):
        return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}, True

    return state, False


class BackgroundRemover:
    def __init__(self, model_path: str | None = None, device: str | None = None):
        # Make MODNet's repo root importable, then import from its src/ package-style path
        try:
            modnet_root = Path(__file__).resolve().parents[1] / "MODNet"  # /app/MODNet
            if modnet_root.exists() and str(modnet_root) not in sys.path:
                sys.path.insert(0, str(modnet_root))  # allows: from src.models.modnet import MODNet
            from src.models.modnet import MODNet  # type: ignore  # noqa: N811
        except Exception as e:
            raise ModuleNotFoundError(
                "MODNet code not importable. Ensure MODNet repo exists at /app/MODNet and contains src/models/modnet.py"
            ) from e

        # Azure is CPU-only by default; keep auto-detect but allow override
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt_path = _resolve_ckpt_path(model_path)
        logger.info("MODNet checkpoint resolved to: %s", str(ckpt_path))

        if not ckpt_path.is_file():
            raise FileNotFoundError(f"MODNet checkpoint not found at '{ckpt_path}'")

        # Build model (do NOT wrap in DataParallel here; normalize checkpoint instead)
        model = MODNet(backbone_pretrained=False).to(self.device)

        # Load checkpoint on CPU first (safe on Azure), then move weights into the model
        ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")
        state = _extract_state_dict(ckpt_obj)
        state, stripped = _strip_module_prefix(state)
        logger.info("MODNet checkpoint loaded. module. prefix stripped: %s", "yes" if stripped else "no")

        # Fail fast on real mismatches after normalization
        model.load_state_dict(state, strict=True)
        model.eval()
        self.model = model

        self._transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def remove_background(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        original_w, original_h = image.shape[1], image.shape[0]
        ref_size = 512

        im_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        im_tensor = self._transform(im_pil).unsqueeze(0)

        _, _, im_h, im_w = im_tensor.shape

        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            else:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh, im_rw = im_h, im_w

        im_rw -= im_rw % 32
        im_rh -= im_rh % 32
        im_tensor = F.interpolate(im_tensor, size=(im_rh, im_rw), mode="area")

        with torch.no_grad():
            input_tensor = im_tensor.to(self.device)
            _, _, matte = self.model(input_tensor, True)

            matte = F.interpolate(matte, size=(original_h, original_w), mode="area")
            matte = matte[0, 0].detach().cpu().numpy()

        mask = (matte * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        return binary_mask
