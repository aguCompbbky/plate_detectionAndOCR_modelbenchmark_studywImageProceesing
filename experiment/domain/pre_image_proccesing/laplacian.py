"""
laplacian.py — Laplacian sharpening preprocessor.

Uses a 3x3 sharpening kernel (Laplacian-based) to enhance character edges.
bad affect on all models
"""
import cv2
import numpy as np
from .base import BasePreprocessor


class LaplacianSharpenPreprocessor(BasePreprocessor):
    """
    Sharpens the image by applying a Laplacian-based high-pass filter.
    Useful for making blurry plate characters crisper before OCR.
    """
    def __init__(self, strength: float = 1.0):
        """
        Args:
            strength: Intensity of the sharpening effect.
        """
        self.strength = strength
        # Kernel:
        #   0         -strength     0
        #  -strength  1+4*strength -strength
        #   0         -strength     0
        center = 1.0 + 4.0 * strength
        self.kernel = np.array([
            [0, -self.strength, 0],
            [-self.strength, center, -self.strength],
            [0, -self.strength, 0]
        ], dtype=np.float32)

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply generic 2D sharpening kernel filter."""
        # Ensure we don't overflow uint8 bounds by using cv2.filter2D
        # which correctly clips [0, 255] if the input is uint8.
        sharpened = cv2.filter2D(img, -1, self.kernel)
        return sharpened
