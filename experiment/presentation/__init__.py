"""__init__.py — presentation package."""
from .benchmark import run_benchmark, save_benchmark
from .visualizer import generate_all_figures

__all__ = ['run_benchmark', 'save_benchmark', 'generate_all_figures']
