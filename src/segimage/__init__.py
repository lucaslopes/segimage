"""Lightweight package init for ComfyUI custom-nodes usage.

Avoid heavy imports at module import time so ComfyUI can list nodes even if
optional dependencies are missing. CLI and processors are imported on-demand
from their respective modules.
"""

__version__ = "0.0.3"
__author__ = "Lucas Lopes Felipe"

def __getattr__(name: str):
    if name == "ImageProcessor":
        from .processor import ImageProcessor  # lazy import
        return ImageProcessor
    if name == "main":
        from .cli import main  # lazy import
        return main
    raise AttributeError(name)

__all__ = ["ImageProcessor", "main"]
