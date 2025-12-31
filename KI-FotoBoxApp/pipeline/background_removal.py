# pipeline/background_removal.py  (UPDATED: fixes device handling + model_path + Linux case-sensitivity)
from __future__ import annotations

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# IMPORTANT:
# Your import is "MODNet..." (capital M,O,D,N). That means the folder in the repo MUST be "MODNet/"
# on Linux. If your repo has "modnet/" this will crash in Azure.
from MODNet.src.models.modnet import MODNet


class BackgroundRemover:
    """
    Uses MODNet for background segmentation.
    Returns a binary foreground mask (uint8: 0/255).
    """

    def __init__(self, model_path: str | None = None, device: str | None = None):
        # Pick device deterministically (previous code passed None -> `.to(None)` breaks)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Default model path:
        # - keep your previous default, but make it robust for container paths
        if model_path is None:
            # Prefer environment override; fallback to known paths
            model_path = os.getenv("MODNET_CKPT", "modnet/modnet.ckpt")

        # Build model
        model = MODNet(backbone_pretrained=False)

        # DataParallel only makes sense if CUDA is used and multiple GPUs exist.
        # On Azure App Service (CPU), this should NOT be wrapped.
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model = model.to(self.device)

        # Load weights safely
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"MODNet checkpoint not found at '{model_path}'. "
                f"Set MODNET_CKPT env var or include the file in the image."
            )

        state = torch.load(model_path, map_location=self.device)

        # Some checkpoints are saved as {"state_dict": ...}
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        model.load_state_dict(state, strict=True)
        model.eval()

        self.model = model

        # Prebuild transform once
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """
        Returns a binary mask (H, W) uint8 with values 0 or 255.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        original_w, original_h = image.shape[1], image.shape[0]
        ref_size = 512

        # BGR -> RGB PIL
        im_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        im_tensor = self._transform(im_pil).unsqueeze(0)  # (1,3,H,W)

        _, _, im_h, im_w = im_tensor.shape

        # Resize to ref_size while keeping aspect ratio, then to multiples of 32
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

            # back to original size
            matte = F.interpolate(matte, size=(original_h, original_w), mode="area")
            matte = matte[0, 0].detach().cpu().numpy()

        mask = (matte * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)

        # Cleanup
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        return binary_mask
