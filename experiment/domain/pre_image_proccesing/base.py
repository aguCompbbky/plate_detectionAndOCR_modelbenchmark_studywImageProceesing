"""
base.py — Abstract base class for all preprocessors.
"""
from abc import ABC, abstractmethod
import numpy as np


class BasePreprocessor(ABC):
    """Abstract base class for image preprocessors."""

    @abstractmethod
    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to the image.

        Args:
            img: BGR image as numpy array (H, W, C) or grayscale (H, W)

        Returns:
            Preprocessed image (same shape or compatible)
        """
        raise NotImplementedError
