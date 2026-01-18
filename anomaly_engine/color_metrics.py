"""Color and difference metrics (LAB, HSV, deltaE, hot color mask).
No functional changes.
"""
from typing import Tuple
import cv2 as cv
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000


def lab_and_hsv(img_bgr):
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    lab = rgb2lab(img_rgb)
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    return lab, hsv


def deltaE_map(lab_base, lab_maint):
    return deltaE_ciede2000(lab_base, lab_maint).astype(np.float32)


def hot_color_mask(hsv):
    m_red1   = cv.inRange(hsv, (0,   90, 120), (10,  255, 255))
    m_red2   = cv.inRange(hsv, (170, 90, 120), (179, 255, 255))
    m_orange = cv.inRange(hsv, (11,  80, 120), (25,  255, 255))
    m_yellow = cv.inRange(hsv, (26,  60, 120), (35,  255, 255))
    return cv.bitwise_or(cv.bitwise_or(m_red1, m_red2), cv.bitwise_or(m_orange, m_yellow))
