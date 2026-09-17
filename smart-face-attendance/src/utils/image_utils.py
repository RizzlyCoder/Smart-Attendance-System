"""
image_utils.py
Small, well-tested helper functions for image I/O and preprocessing that
are shared between the recognition and emotion pipelines.
"""

import os
from typing import List, Tuple

import cv2
import numpy as np


def load_image_grayscale(path: str) -> np.ndarray:
    """Load an image from disk as grayscale. Raises FileNotFoundError/ValueError on failure."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def list_image_files(directory: str, extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> List[str]:
    """Return sorted absolute paths of image files directly inside `directory`."""
    if not os.path.isdir(directory):
        return []
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(extensions)
    ]
    return sorted(files)


def resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def equalize_histogram(img: np.ndarray) -> np.ndarray:
    """Improves robustness to lighting conditions before recognition/emotion inference."""
    return cv2.equalizeHist(img)
