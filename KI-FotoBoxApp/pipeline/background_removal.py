# pipeline/background_removal.py
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image



class BackgroundRemover:
    def __init__(self, model_path: str | None = None, device: str | None = None):
        # Make MODNet's repo root importable, then import from its src/ package-style path
        try:
            modnet_root = Path(__file__).resolve().parents[1] / "MODNet"  # /app/MODNet
            if modnet_root.exists():
                sys.path.insert(0, str(modnet_root))  # allows: from src.models.modnet import MODNet
            from src.models.modnet import MODNet  # type: ignore  # noqa: N811
        except Exception as e:
            raise ModuleNotFoundError(
                "MODNet code not importable. Ensure MODNet repo exists at /app/MODNet and contains src/models/modnet.py"
            ) from e

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if model_path is None or not model_path.strip():
            model_path = os.getenv("MODNET_CKPT", "").strip() or "checkpoints/modnet_photographic_portrait_matting.ckpt"

        ckpt_path = Path(model_path)
        if not ckpt_path.is_absolute():
            ckpt_path = (Path(os.getcwd()) / ckpt_path).resolve()

        if not ckpt_path.is_file():
            raise FileNotFoundError(f"MODNet checkpoint not found at '{ckpt_path}'")

        model = MODNet(backbone_pretrained=False)
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(self.device)

        state = torch.load(str(ckpt_path), map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

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
