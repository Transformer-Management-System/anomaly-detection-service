"""Blob extraction and property computation.
No functional changes.
"""
from typing import Dict, Any, List
import numpy as np
import cv2 as cv

def blob_props(bin_mask, dE, hsv) -> List[Dict[str,Any]]:
    n, labels, stats, centroids = cv.connectedComponentsWithStats(bin_mask, connectivity=8)
    out = []
    for lab in range(1, n):
        x,y,w,h,area = stats[lab]
        if area < 25:
            continue
        roi = (labels[y:y+h, x:x+w] == lab)
        dE_roi = dE[y:y+h, x:x+w][roi]
        hsv_roi = hsv[y:y+h, x:x+w][roi]
        mean_dE = float(dE_roi.mean()) if dE_roi.size else 0.0
        peak_dE = float(dE_roi.max()) if dE_roi.size else 0.0
        mean_h = float(hsv_roi[:,0].mean()) if hsv_roi.size else 0.0
        mean_s = float(hsv_roi[:,1].mean()) if hsv_roi.size else 0.0
        mean_v = float(hsv_roi[:,2].mean()) if hsv_roi.size else 0.0

        pts = np.column_stack(np.where(roi))
        if len(pts) >= 10:
            cov = np.cov(pts.astype(np.float32).T)
            eigvals,_ = np.linalg.eig(cov)
            eigvals = np.sort(np.abs(eigvals))
            elong = float((eigvals[-1]+1e-6)/(eigvals[0]+1e-6))
        else:
            elong = 1.0

        out.append(dict(label=lab, bbox=(int(x),int(y),int(w),int(h)), area=int(area),
                        centroid=(float(centroids[lab][0]), float(centroids[lab][1])),
                        mean_deltaE=mean_dE, peak_deltaE=peak_dE,
                        mean_hsv=(mean_h, mean_s, mean_v), elongation=elong))
    return out
