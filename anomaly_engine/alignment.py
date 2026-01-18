"""Image alignment logic (ECC + feature fallback) extracted from anomaly_cv.
No functional changes.
"""
from typing import Tuple
import numpy as np
import cv2 as cv

def ecc_align(base_gray: np.ndarray, mov_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, float]:
    """
    Robust alignment:
      1) ECC on Canny edges with an inputMask (ignores legend/sky).
      2) Fallback: ORB + KNN (Lowe ratio) + RANSAC homography.
    Returns: (warp_matrix, aligned_gray, ok, score)
      - warp_matrix is 2x3 (affine) or 3x3 (homography)
      - score is ECC correlation for ECC; 0.0 for homography fallback
    """
    H, W = base_gray.shape

    # Mask: keep transformer region, drop right colorbar + top sky band
    inputMask = np.ones_like(base_gray, np.uint8) * 255
    inputMask[:, int(0.88 * W):] = 0
    inputMask[:int(0.15 * H), :] = 0

    # Edge images (photometrically robust)
    base_e = cv.Canny(base_gray, 50, 150)
    mov_e  = cv.Canny(mov_gray,  50, 150)

    warp_mode = cv.MOTION_AFFINE
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 300, 1e-6)

    try:
        cc, warp = cv.findTransformECC(
            base_e, mov_e, warp, warp_mode, criteria, inputMask=inputMask
        )
        aligned = cv.warpAffine(
            mov_gray, warp, (W, H),
            flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP
        )
        return warp, aligned, True, float(cc)
    except cv.error:
        # Feature fallback: ORB + KNN(Lowe ratio) + RANSAC Homography
        orb = cv.ORB_create(5000)
        k1, d1 = orb.detectAndCompute(base_gray, None)
        k2, d2 = orb.detectAndCompute(mov_gray,  None)
        if d1 is None or d2 is None:
            return np.eye(2,3,np.float32), mov_gray, False, 0.0

        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(d1, d2, k=2)
        good = [m for m,n in knn if n is not None and m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return np.eye(2,3,np.float32), mov_gray, False, 0.0

        pts1 = np.float32([k1[m.queryIdx].pt for m in good])
        pts2 = np.float32([k2[m.trainIdx].pt for m in good])
        Hm, mask = cv.findHomography(pts2, pts1, cv.RANSAC, 3.0)
        if Hm is None:
            return np.eye(2,3,np.float32), mov_gray, False, 0.0

        aligned = cv.warpPerspective(mov_gray, Hm, (W, H))
        return Hm.astype(np.float32), aligned, True, 0.0
