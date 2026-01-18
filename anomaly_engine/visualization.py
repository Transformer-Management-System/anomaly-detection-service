"""Overlay generation.
No functional changes.
"""
import cv2 as cv

def overlay_detections(img_bgr, blobs):
    out = img_bgr.copy()
    for b in blobs:
        x,y,w,h = b.bbox
        color = (0,0,255) if b.classification=='Faulty' else ((0,165,255) if b.classification=='Potentially Faulty' else (0,255,0))
        cv.rectangle(out, (x,y), (x+w,y+h), color, 2)
        label = f"{b.classification}:{b.subtype} pΔE={b.peak_deltaE:.1f} conf={b.confidence:.2f}"
        cv.putText(out, label, (x, max(0,y-5)), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv.LINE_AA)
    return out
