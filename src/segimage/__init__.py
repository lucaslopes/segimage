"""
segimage - Image segmentation and processing library

A Python library for processing images with support for various formats
including MATLAB .mat files and conversion to standard image formats (PNG, JPG, etc.).
"""

from .processor import ImageProcessor
from .cli import main

__version__ = "0.0.1"
__author__ = "Lucas Lopes Felipe"

__all__ = [
    "ImageProcessor",
    "main",
]
