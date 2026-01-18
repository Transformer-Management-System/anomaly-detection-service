"""Morphological utilities.
No functional changes.
"""
import cv2 as cv

def morphology_clean(mask):
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=2)
    return mask
