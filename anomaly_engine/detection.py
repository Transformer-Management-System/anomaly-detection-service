import cv2 as cv
from skimage.metrics import structural_similarity as ssim
import json
import numpy as np
from dataclasses import asdict

from .data_structures import BlobDet, DetectionReport
from .io_utils import read_bgr, to_gray
from .alignment import ecc_align
from .color_metrics import lab_and_hsv, deltaE_map, hot_color_mask
from .morphology import morphology_clean
from .topology import build_wire_skeleton, find_skeleton_nodes
from .blobs import blob_props
from .classification import classify_blob_enhanced, summarize_image
from .visualization import overlay_detections


def detect_anomalies(transformer_id: str, baseline_path: str, maintenance_path: str,
                     out_overlay_path: str, out_json_path: str,
                     slider_percent: float | None = None) -> DetectionReport:
    base_bgr = read_bgr(baseline_path)
    ment_bgr = read_bgr(maintenance_path)

    base_gray = to_gray(base_bgr)
    ment_gray = to_gray(ment_bgr)
    if ment_gray.shape != base_gray.shape:
        Hs, Ws = base_gray.shape
        ment_bgr = cv.resize(ment_bgr, (Ws, Hs), interpolation=cv.INTER_LINEAR)
        ment_gray = cv.resize(ment_gray, (Ws, Hs), interpolation=cv.INTER_LINEAR)

    warp, ment_aligned_gray, ok, score = ecc_align(base_gray, ment_gray)

    H, W = base_gray.shape
    if warp.shape == (3,3):
        ment_aligned_bgr = cv.warpPerspective(
            ment_bgr, warp, (W, H),
            flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP
        )
        warp_model = 'homography'
    elif warp.shape == (2,3):
        ment_aligned_bgr = cv.warpAffine(
            ment_bgr, warp, (W, H),
            flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP
        )
        warp_model = 'affine'
    else:
        raise ValueError("Unexpected warp shape")

    mean_ssim, _ = ssim(base_gray, ment_aligned_gray, full=True, data_range=255)

    base_t_pot  = 8.0  if mean_ssim >= 0.70 else 10.0
    base_t_fault = 12.0 if mean_ssim >= 0.70 else 14.0
    ratio = base_t_fault / base_t_pot

    threshold_source = "adaptive_ssim"
    scale_applied = None
    if slider_percent is not None:
        try:
            p = float(slider_percent)
        except (TypeError, ValueError):
            p = None
        if p is not None:
            p = max(0.0, min(100.0, p))
            scale_applied = 1.2 - 0.4*(p/100.0)
            t_pot = base_t_pot * scale_applied
            if mean_ssim >= 0.70:
                t_pot = float(np.clip(t_pot, 6.0, 11.0))
            else:
                t_pot = float(np.clip(t_pot, 8.0, 13.0))
            t_fault = t_pot * ratio
            threshold_source = "slider_scaled"
        else:
            t_pot = base_t_pot
            t_fault = base_t_fault
    else:
        t_pot = base_t_pot
        t_fault = base_t_fault

    hist_b = cv.calcHist([base_gray],[0],None,[64],[0,256]);  hist_b = cv.normalize(hist_b, None).flatten()
    hist_m = cv.calcHist([ment_aligned_gray],[0],None,[64],[0,256]);  hist_m = cv.normalize(hist_m, None).flatten()
    hist_corr = float(np.corrcoef(hist_b, hist_m)[0,1])
    if hist_corr < 0.60:
        t_pot   = max(6.0,  t_pot   - 2.0)
        t_fault = max(10.0, t_fault - 2.0)
        threshold_source += "+palette_soften"

    base_lab, _ = lab_and_hsv(base_bgr)
    ment_lab, ment_hsv = lab_and_hsv(ment_aligned_bgr)
    dE = deltaE_map(base_lab, ment_lab)

    mask_hot = hot_color_mask(ment_hsv)
    mask_delta = (dE >= t_pot).astype(np.uint8)*255
    mask = cv.bitwise_and(mask_hot, mask_delta)
    mask = morphology_clean(mask)

    hch, sch, vch = cv.split(ment_hsv)
    v98 = float(np.percentile(vch, 98))
    abs_hot = (
        ((hch <= 10) | (hch >= 170) | ((hch >= 11) & (hch <= 25))) &
        (sch >= 80) &
        (vch >= max(200.0, v98))
    ).astype(np.uint8) * 255

    skel, wire_band = build_wire_skeleton(ment_aligned_bgr, mask)
    endpoints, junctions = find_skeleton_nodes(skel)
    joints = endpoints + junctions

    props = blob_props(mask, dE, ment_hsv)
    blobs = []
    for p in props:
        cls, subtype, conf, sev = classify_blob_enhanced(
            p, dE_thr_fault=t_fault, dE_thr_pot=t_pot,
            skel=skel, joints=joints, hot_mask=mask, abs_hot_mask=abs_hot
        )
        blobs.append(BlobDet(label=p['label'], bbox=p['bbox'], area=p['area'],
                             centroid=p['centroid'], mean_deltaE=p['mean_deltaE'], peak_deltaE=p['peak_deltaE'],
                             mean_hsv=p['mean_hsv'], elongation=p['elongation'],
                             classification=cls, subtype=subtype, confidence=conf, severity=sev))

    image_label = summarize_image(blobs)
    overlay = overlay_detections(ment_aligned_bgr, blobs)
    cv.imwrite(out_overlay_path, overlay)

    rep = DetectionReport(
        transformer_id=transformer_id,
        baseline_path=baseline_path,
        maintenance_path=maintenance_path,
        warp_model=warp_model,
        warp_success=bool(ok),
        warp_score=float(score),
        mean_ssim=float(mean_ssim),
        image_level_label=image_label,
        blobs=blobs,
        t_pot=float(t_pot),
        t_fault=float(t_fault),
        base_t_pot=float(base_t_pot),
        base_t_fault=float(base_t_fault),
        slider_percent=float(slider_percent) if slider_percent is not None else None,
        scale_applied=float(scale_applied) if scale_applied is not None else None,
        threshold_source=threshold_source,
        ratio=float(ratio)
    )

    with open(out_json_path, "w") as f:
        json.dump({
            **{k:v for k,v in asdict(rep).items() if k!='blobs'},
            "blobs": [asdict(b) for b in blobs],
            "thresholds_used": {
                "t_pot": rep.t_pot,
                "t_fault": rep.t_fault,
                "base_t_pot": rep.base_t_pot,
                "base_t_fault": rep.base_t_fault,
                "slider_percent": rep.slider_percent,
                "scale_applied": rep.scale_applied,
                "source": rep.threshold_source,
                "ratio": rep.ratio,
                "mean_ssim": rep.mean_ssim
            }
        }, f, indent=2)

    return rep
