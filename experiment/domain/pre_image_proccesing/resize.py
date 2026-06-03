"""
resize.py — Resize preprocessor.
"""
import cv2
import numpy as np
from .base import BasePreprocessor


class ResizePreprocessor(BasePreprocessor):
    """
    Resizes image to a fixed (width, height).
    Uses INTER_LINEAR for upscale, INTER_AREA for downscale.
    """

    def __init__(self, width: int, height: int):
        """
        Args:
            width: Target width in pixels
            height: Target height in pixels
        """
        self.width = width
        self.height = height

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Resize image to (self.width, self.height)."""
        h, w = img.shape[:2]
        if h == self.height and w == self.width:
            return img
        interp = cv2.INTER_AREA if (w > self.width or h > self.height) else cv2.INTER_LINEAR
        return cv2.resize(img, (self.width, self.height), interpolation=interp)
