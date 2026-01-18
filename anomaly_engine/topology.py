"""Topology helpers: wire skeleton, joints, coverage analysis.
No functional changes.
"""
from typing import List, Tuple, Iterable
import numpy as np
import cv2 as cv
from skimage.morphology import skeletonize


def build_wire_skeleton(img_bgr, hot_mask):
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)
    k3 = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    k5 = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    edges = cv.dilate(edges, k3, iterations=1)
    hot_dil = cv.dilate(hot_mask, k5, iterations=1)
    union = cv.bitwise_or(edges, hot_dil)
    skel_bool = skeletonize((union > 0).astype(np.uint8).astype(bool))
    skel = (skel_bool.astype(np.uint8) * 255)
    wire_band = cv.dilate(skel, k3, iterations=1)
    return skel, wire_band


def _neighbors8(y: int, x: int, h: int, w: int) -> Iterable[Tuple[int,int]]:
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dy==0 and dx==0: continue
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx


def find_skeleton_nodes(skel):
    s = (skel > 0).astype(np.uint8)
    H, W = s.shape
    endpoints, junctions = [], []
    ys, xs = np.where(s)
    for y, x in zip(ys, xs):
        deg = 0
        for ny, nx in _neighbors8(y, x, H, W):
            if s[ny, nx]: deg += 1
        if deg == 1:
            endpoints.append((x, y))
        elif deg >= 3:
            junctions.append((x, y))
    return endpoints, junctions


def is_near_joint(centroid_xy, joints, r: int = 8) -> bool:
    cx, cy = centroid_xy
    for jx, jy in joints:
        if (cx - jx)**2 + (cy - jy)**2 <= r*r:
            return True
    return False


def wire_hot_coverage(bbox, skel, hot_mask, expand: int = 10):
    H, W = skel.shape
    x,y,w,h = bbox
    x0 = max(0, x - expand); y0 = max(0, y - expand)
    x1 = min(W, x + w + expand); y1 = min(H, y + h + expand)

    skel_roi = skel[y0:y1, x0:x1] > 0
    hot_roi  = hot_mask[y0:y1, x0:x1] > 0

    k3 = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    hot_roi_d = cv.dilate((hot_roi.astype(np.uint8))*255, k3, iterations=1) > 0

    wire_len = int(skel_roi.sum())
    hot_len  = int((skel_roi & hot_roi_d).sum())
    coverage = (hot_len / wire_len) if wire_len > 0 else 0.0

    band = cv.dilate((skel_roi.astype(np.uint8))*255, k3, iterations=1) > 0
    cool_pixels = int((band & (~hot_roi)).sum())
    total_band  = int(band.sum())
    cool_frac = (cool_pixels / total_band) if total_band > 0 else 0.0
    return float(coverage), hot_len, wire_len, float(cool_frac)
