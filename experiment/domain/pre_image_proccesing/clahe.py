"""
clahe.py — CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessor.
"""
import cv2
import numpy as np
from .base import BasePreprocessor


class CLAHEPreprocessor(BasePreprocessor):
    """
    Applies CLAHE to improve local contrast.
    Works on grayscale channel of BGR image.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        """
        Args:
            clip_limit: Threshold for contrast limiting (default 2.0)
            tile_grid_size: Size of grid for histogram equalization (default 8x8)
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE. Input/output: BGR (H, W, 3) or gray (H, W)."""
        if img.ndim == 2:
            # Grayscale
            return self._clahe.apply(img)

        # BGR → LAB, apply CLAHE on L channel, convert back
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
