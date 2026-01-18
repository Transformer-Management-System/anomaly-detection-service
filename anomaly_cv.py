"""Legacy wrapper for anomaly detection.

This file previously contained the full implementation. It now re-exports the
modular engine located in `backend.anomaly_engine` (when the project root is on
PYTHONPATH) or `anomaly_engine` (when run directly from inside the backend
folder). Functionality and CLI behavior are preserved.
"""
from __future__ import annotations
import sys
from dataclasses import asdict  # kept for backward compatibility if callers used it

# Support both invocation styles:
# 1. python backend/anomaly_cv.py ...   (script path inside backend/)
# 2. from backend import anomaly_cv     (package import from project root)
try:  # package style
    from backend.anomaly_engine import detect_anomalies, BlobDet, DetectionReport  # type: ignore
except ImportError:  # script style (cwd likely project root or backend dir)
    from anomaly_engine import detect_anomalies, BlobDet, DetectionReport  # type: ignore

__all__ = ["detect_anomalies", "BlobDet", "DetectionReport", "asdict"]

def _cli():  # mirror previous main block
    if len(sys.argv) not in (6,7):
        print("Usage: python anomaly_cv.py <transformer_id> <baseline.jpg> <maintenance.jpg> <overlay.png> <report.json> [slider_percent]")
        sys.exit(1)
    _, txid, bpath, mpath, opath, jpath, *rest = sys.argv
    slider_arg = float(rest[0]) if rest else None
    rep = detect_anomalies(txid, bpath, mpath, opath, jpath, slider_percent=slider_arg)
    print(f"Result: {rep.image_level_label} | blobs={len(rep.blobs)} | SSIM={rep.mean_ssim:.3f} | warp={rep.warp_model} | t_pot={rep.t_pot:.2f} t_fault={rep.t_fault:.2f} (src={rep.threshold_source})")

if __name__ == "__main__":
    _cli()
