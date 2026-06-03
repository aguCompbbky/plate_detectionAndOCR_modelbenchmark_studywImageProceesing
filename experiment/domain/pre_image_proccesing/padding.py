"""
padding.py — Padding preprocessor. Intended ONLY for PaddleOCR.

PaddleOCR performs better when license plate crops have a small border
around the text to avoid clipping characters at edges.
"""
import cv2
import numpy as np
from .base import BasePreprocessor


class PaddingPreprocessor(BasePreprocessor):
    """
    Adds uniform padding around image borders.
    Intended for PaddleOCR only — other OCR models do NOT use this.
    """

    def __init__(self, pad: int = 10, color: tuple = (255, 255, 255)):
        """
        Args:
            pad: Number of pixels to add on each side
            color: Border fill color in BGR format (default: white)
        """
        self.pad = pad
        self.color = color

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Add padding around the image."""
        if img.ndim == 2:
            # Grayscale: use single-channel border value
            border_val = int(0.299 * self.color[2] + 0.587 * self.color[1] + 0.114 * self.color[0])
            return cv2.copyMakeBorder(
                img, self.pad, self.pad, self.pad, self.pad,
                cv2.BORDER_CONSTANT, value=border_val
            )
        return cv2.copyMakeBorder(
            img, self.pad, self.pad, self.pad, self.pad,
            cv2.BORDER_CONSTANT, value=self.color
        )
