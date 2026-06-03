"""
pipeline_manager.py — Orchestrates preprocessing steps into pipelines.
"""
from typing import List
import numpy as np

from .base import BasePreprocessor
from .resize import ResizePreprocessor
from .clahe import CLAHEPreprocessor
from .denoise import DenoisePreprocessor
from .laplacian import LaplacianSharpenPreprocessor
from .padding import PaddingPreprocessor


class PreprocessingPipeline:
    """
    Sequential image preprocessing pipeline.
    Applies a list of BasePreprocessor steps in order.
    """

    def __init__(self, steps: List[BasePreprocessor], name: str = "pipeline"):
        """
        Args:
            steps: Ordered list of preprocessor instances to apply
            name: Human-readable name for this pipeline (used in logging)
        """
        self.steps = steps
        self.name = name

    def run(self, img: np.ndarray) -> np.ndarray:
        """
        Apply all steps sequentially.

        Args:
            img: Input BGR image (H, W, 3) as numpy array

        Returns:
            Preprocessed image
        """
        result = img.copy()
        for step in self.steps:
            result = step.apply(result)
        return result

    def __repr__(self):
        step_names = [type(s).__name__ for s in self.steps]
        return f"PreprocessingPipeline(name={self.name!r}, steps={step_names})"


# ─── Factory functions ─────────────────────────────────────────────────────────

def get_standard_pipeline() -> PreprocessingPipeline:
    """
    Standard preprocessing for CRNN and LPRNet.
    Steps: Resize (model-specific resize happens inside each OCR adapter) +
           CLAHE + Denoise
    """
    return PreprocessingPipeline(
        steps=[
            CLAHEPreprocessor(clip_limit=2.0, tile_grid_size=(8, 8)),
            DenoisePreprocessor(h=10.0),
        ],
        name="standard"
    )


def get_paddle_pipeline() -> PreprocessingPipeline:
    """
    Extended preprocessing for PaddleOCR only.
    Adds white padding to help PaddleOCR detect edge characters.
    """
    return PreprocessingPipeline(
        steps=[
            CLAHEPreprocessor(clip_limit=2.0, tile_grid_size=(8, 8)),
            DenoisePreprocessor(h=10.0),
            PaddingPreprocessor(pad=10, color=(255, 255, 255)),
        ],
        name="paddle"
    )
