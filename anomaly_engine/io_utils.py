"""I/O utilities (image reading, simple color conversions).
No functional changes.
"""
from typing import Tuple
import cv2 as cv
import numpy as np

def read_bgr(path: str) -> np.ndarray:
    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
