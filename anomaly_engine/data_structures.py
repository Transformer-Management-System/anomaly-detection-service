"""Data structures for anomaly detection engine (extracted from anomaly_cv).
No functional changes.
"""
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class BlobDet:
    label: int
    bbox: Tuple[int,int,int,int]   # x,y,w,h
    area: int
    centroid: Tuple[float,float]
    mean_deltaE: float
    peak_deltaE: float
    mean_hsv: Tuple[float,float,float]
    elongation: float              # major/minor axis ratio
    classification: str            # Normal/Faulty/Potentially Faulty
    subtype: str                   # LooseJoint / PointOverload / FullWireOverload / None
    confidence: float              # 0..1
    severity: float                # 0..100

@dataclass
class DetectionReport:
    transformer_id: str
    baseline_path: str
    maintenance_path: str
    warp_model: str
    warp_success: bool
    warp_score: float
    mean_ssim: float
    image_level_label: str
    blobs: List[BlobDet]
    # Threshold metadata
    t_pot: float
    t_fault: float
    base_t_pot: float
    base_t_fault: float
    slider_percent: float | None
    scale_applied: float | None
    threshold_source: str
    ratio: float
