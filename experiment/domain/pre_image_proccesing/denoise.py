"""
denoise.py — Denoising preprocessor using Non-local Means.
"""
import cv2
import numpy as np
from .base import BasePreprocessor


class DenoisePreprocessor(BasePreprocessor):
    """
    Applies Non-Local Means denoising.
    For BGR images uses fastNlMeansDenoisingColored.
    For grayscale uses fastNlMeansDenoising.
    """

    def __init__(self, h: float = 10.0, h_color: float = 10.0,
                 template_window: int = 7, search_window: int = 21):
        """
        Args:
            h: Filter strength for luminance (higher = more denoised but blurrier)
            h_color: Filter strength for color channels
            template_window: Template patch size (odd number)
            search_window: Window size to search for patches (odd number)
        """
        self.h = h
        self.h_color = h_color
        self.template_window = template_window
        self.search_window = search_window

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply NLM denoising."""
        if img.ndim == 2:
            return cv2.fastNlMeansDenoising(
                img, None,
                h=self.h,
                templateWindowSize=self.template_window,
                searchWindowSize=self.search_window,
            )
        return cv2.fastNlMeansDenoisingColored(
            img, None,
            h=self.h,
            hColor=self.h_color,
            templateWindowSize=self.template_window,
            searchWindowSize=self.search_window,
        )
