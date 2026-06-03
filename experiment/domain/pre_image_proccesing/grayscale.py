"""
grayscale.py — Grayscale conversion preprocessor.
"""
import cv2
import numpy as np
from .base import BasePreprocessor


class GrayscalePreprocessor(BasePreprocessor):
    """
    Converts BGR image to grayscale and back to 3-channel BGR.
    Useful for models that benefit from luminance-only input
    while keeping the channel dimension compatible.
    """

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Convert BGR → Gray → BGR (3-channel)."""
        if img.ndim == 2:
            # Already grayscale: stack to 3 channels
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
