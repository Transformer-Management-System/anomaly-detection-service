"""Anomaly detection engine package.
Expose detect_anomalies and data structures for external use.
"""
from .data_structures import BlobDet, DetectionReport
from .detection import detect_anomalies

__all__ = [
    'BlobDet', 'DetectionReport', 'detect_anomalies'
]
