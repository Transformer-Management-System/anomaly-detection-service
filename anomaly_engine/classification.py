"""Rule-based blob classification with topology and absolute heat promotions.
No functional changes.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
from .topology import wire_hot_coverage, is_near_joint


def classify_blob_enhanced(
    b: Dict[str,Any],
    dE_thr_fault=12.0,
    dE_thr_pot=8.0,
    skel=None,
    joints: List[Tuple[int,int]] = None,
    hot_mask=None,
    abs_hot_mask=None
):
    """Returns (label, subtype, confidence, severity)"""
    h,s,v = b['mean_hsv']
    elong = b['elongation']
    peak, mean = b['peak_deltaE'], b['mean_deltaE']

    # Color bands (OpenCV Hue 0..179)
    is_red_or_orange = (h <= 10 or h >= 170 or (11 <= h <= 25))
    is_yellowish     = (26 <= h <= 35)

    faulty = is_red_or_orange and (peak >= dE_thr_fault)
    potential = (is_yellowish and peak >= dE_thr_pot) or ((elong >= 3.0) and mean >= dE_thr_pot)

    near_joint = False
    coverage = 0.0
    cool_frac = 0.0
    if skel is not None and hot_mask is not None and joints is not None:
        near_joint = is_near_joint(b['centroid'], joints, r=8)
        coverage, hot_len, wire_len, cool_frac = wire_hot_coverage(b['bbox'], skel, hot_mask, expand=10)

    subtype = 'None'
    label = 'Normal'

    if near_joint:
        subtype = 'LooseJoint'
        if faulty: label = 'Faulty'
        elif potential: label = 'Potentially Faulty'
    else:
        FULL_COVER_THR = 0.60
        POINT_COVER_THR = 0.25
        REST_COOL_THR = 0.60
        if coverage >= FULL_COVER_THR:
            subtype = 'FullWireOverload'
            label = 'Potentially Faulty' if (faulty or potential) else 'Normal'
        elif (coverage < POINT_COVER_THR and cool_frac >= REST_COOL_THR):
            subtype = 'PointOverload'
            if faulty: label = 'Faulty'
            elif potential: label = 'Potentially Faulty'
        else:
            subtype = 'PointOverload' if (faulty or potential) else 'None'
            if faulty: label = 'Faulty'
            elif potential: label = 'Potentially Faulty'

    if label == 'Normal' and abs_hot_mask is not None:
        x,y,w,h_box = b['bbox']
        abs_roi = abs_hot_mask[y:y+h_box, x:x+w]
        abs_frac = float(abs_roi.sum()) / float(max(1, w*h_box) * 255.0)
        mean_v = b['mean_hsv'][2]
        if near_joint and (mean_v >= 200 or abs_frac >= 0.20):
            label, subtype = 'Faulty', 'LooseJoint'
        elif (coverage < 0.25 and cool_frac >= 0.60 and abs_frac >= 0.20):
            label, subtype = 'Faulty', 'PointOverload'
        elif coverage >= 0.60 and abs_frac >= 0.40:
            label, subtype = 'Potentially Faulty', 'FullWireOverload'

    color_bonus = 0.15 if is_red_or_orange else (0.05 if is_yellowish else 0.0)
    conf = 0.5 + 0.5 * np.tanh((peak - dE_thr_pot)/8.0) + color_bonus
    if subtype == 'FullWireOverload' and coverage >= 0.6: conf += 0.07
    if subtype == 'PointOverload' and coverage < 0.25 and cool_frac >= 0.6: conf += 0.07
    if subtype == 'LooseJoint' and near_joint: conf += 0.05
    conf = float(np.clip(conf, 0.0, 1.0))

    sev  = float(np.clip((0.6*peak + 0.4*mean) + 0.005*b['area'], 0, 100))
    return label, subtype, conf, sev


def summarize_image(blobs):
    if any(b.classification == 'Faulty' for b in blobs): return 'Faulty'
    if any(b.classification == 'Potentially Faulty' for b in blobs): return 'Potentially Faulty'
    return 'Normal'
