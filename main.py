"""
FastAPI Microservice for Anomaly Detection using Computer Vision
"""
import os
import uuid
import logging
import tempfile
from typing import Optional, List, Any, Dict, Tuple
from datetime import datetime
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2 as cv
import httpx
import uvicorn

from anomaly_cv import detect_anomalies, DetectionReport
from anomaly_engine.io_utils import read_bgr, to_gray
from anomaly_engine.alignment import ecc_align
from anomaly_engine.visualization import overlay_detections

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Anomaly Detection Service",
    description="Computer Vision based anomaly detection microservice",
    version="1.0.0"
)

# Enable CORS for Spring Boot backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your Spring Boot backend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary directory for processing
TEMP_DIR = tempfile.gettempdir()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Anomaly Detection Service",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check for container orchestration"""
    return {"status": "healthy"}


class DetectRequest(BaseModel):
    baseline_url: str
    maintenance_url: str
    annotated_upload_url: str
    slider_percent: Optional[float] = None


class BatchDetectRequest(BaseModel):
    baseline_url: str
    maintenance_urls: List[str]
    slider_percent: Optional[float] = None


async def _download_image(client: httpx.AsyncClient, url: str, dest_path: str) -> None:
    """Download an image from a presigned URL to a local path."""
    resp = await client.get(url)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download image from URL (HTTP {resp.status_code})"
        )
    content_type = resp.headers.get("content-type", "")
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"URL did not return an image (content-type: {content_type})"
        )
    with open(dest_path, "wb") as f:
        f.write(resp.content)


def _build_detect_response(request_id: str, report: Any) -> Dict[str, Any]:
    """Convert a DetectionReport into the endpoint response payload format."""
    anomalies = []
    for i, blob in enumerate(report.blobs):
        x, y, w, h = blob.bbox
        anomalies.append({
            "id": f"anomaly_{i+1}",
            "bbox": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            },
            "confidence": float(blob.confidence),
            "severity": blob.classification,
            "severityScore": float(blob.severity),
            "classification": blob.subtype,
            "area": int(blob.area),
            "centroid": {
                "x": float(blob.centroid[0]),
                "y": float(blob.centroid[1])
            },
            "meanDeltaE": float(blob.mean_deltaE),
            "peakDeltaE": float(blob.peak_deltaE),
            "meanHsv": {
                "h": float(blob.mean_hsv[0]),
                "s": float(blob.mean_hsv[1]),
                "v": float(blob.mean_hsv[2])
            },
            "elongation": float(blob.elongation)
        })

    return {
        "requestId": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "imageLevelLabel": report.image_level_label,
        "anomalyCount": len(anomalies),
        "anomalies": anomalies,
        "metrics": {
            "meanSsim": float(report.mean_ssim),
            "warpModel": report.warp_model,
            "warpSuccess": report.warp_success,
            "warpScore": float(report.warp_score),
            "thresholdPotential": float(report.t_pot),
            "thresholdFault": float(report.t_fault),
            "basePotential": float(report.base_t_pot),
            "baseFault": float(report.base_t_fault),
            "sliderPercent": float(report.slider_percent) if report.slider_percent is not None else None,
            "scaleApplied": float(report.scale_applied) if report.scale_applied is not None else None,
            "thresholdSource": report.threshold_source,
            "ratio": report.ratio
        }
    }


def run_detection_from_paths(
    baseline_path: str,
    maintenance_path: str,
    slider_percent: Optional[float] = None,
    request_id: Optional[str] = None
) -> Tuple[Dict[str, Any], DetectionReport]:
    """Run anomaly detection from local image paths.

    Returns a tuple of (endpoint-shaped JSON payload, raw DetectionReport).
    """
    resolved_request_id = request_id or str(uuid.uuid4())
    report = detect_anomalies(
        baseline_path=baseline_path,
        maintenance_path=maintenance_path,
        slider_percent=slider_percent
    )
    return _build_detect_response(resolved_request_id, report), report


def _create_annotated_image(baseline_path: str, maintenance_path: str, blobs: List[Any]):
    """Re-align the maintenance image onto the baseline coordinate space and draw anomaly boxes.

    Because blob coordinates are emitted in aligned/baseline space by the detection
    pipeline, the overlay must be drawn on the warped maintenance image, not the
    original.
    """
    base_bgr = read_bgr(baseline_path)
    ment_bgr = read_bgr(maintenance_path)
    base_gray = to_gray(base_bgr)
    ment_gray = to_gray(ment_bgr)

    if ment_gray.shape != base_gray.shape:
        H, W = base_gray.shape
        ment_bgr = cv.resize(ment_bgr, (W, H), interpolation=cv.INTER_LINEAR)
        ment_gray = cv.resize(ment_gray, (W, H), interpolation=cv.INTER_LINEAR)

    warp, _, _, _ = ecc_align(base_gray, ment_gray)
    H, W = base_gray.shape
    if warp.shape == (3, 3):
        aligned = cv.warpPerspective(
            ment_bgr, warp, (W, H),
            flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP
        )
    else:
        aligned = cv.warpAffine(
            ment_bgr, warp, (W, H),
            flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP
        )

    return overlay_detections(aligned, blobs)


async def _upload_annotated_image(
    client: httpx.AsyncClient,
    img_bgr,
    upload_url: str
) -> Optional[str]:
    """Encode *img_bgr* as JPEG and PUT it to the presigned S3 URL.

    Returns the S3 object key on success, or None if the upload fails.
    The failure is non-fatal — callers must not raise on a None return.
    """
    ok, buf = cv.imencode(".jpg", img_bgr)
    if not ok:
        logger.warning("Failed to encode annotated image as JPEG; skipping upload")
        return None

    try:
        resp = await client.put(
            upload_url,
            content=buf.tobytes(),
            headers={"Content-Type": "image/jpeg"},
        )
        if resp.status_code == 200:
            parsed = urlparse(upload_url)
            key = parsed.path.lstrip("/")
            return key
        logger.warning(
            "Annotated image upload returned HTTP %s; annotatedImageKey will be null",
            resp.status_code,
        )
        return None
    except Exception as exc:
        logger.warning("Annotated image upload failed: %s", exc)
        return None


@app.post("/api/v1/detect")
async def detect_anomalies_endpoint(request: DetectRequest):
    """
    Detect anomalies by comparing baseline and maintenance images.
    
    Accepts presigned S3 URLs for both images. Downloads them, runs the
    detection pipeline, and returns all results and metadata in the response.
    
    Args:
        request: JSON body with baseline_url, maintenance_url, and optional slider_percent
    
    Returns:
        JSON with detection results including anomalies, metrics, and base64 overlay image
    """
    
    request_id = str(uuid.uuid4())
    baseline_path = os.path.join(TEMP_DIR, f"{request_id}_baseline.png")
    maintenance_path = os.path.join(TEMP_DIR, f"{request_id}_maintenance.png")
    
    try:
        # Download images from presigned URLs
        async with httpx.AsyncClient(timeout=60.0) as client:
            await _download_image(client, request.baseline_url, baseline_path)
            await _download_image(client, request.maintenance_url, maintenance_path)

        response_data, report = run_detection_from_paths(
            baseline_path=baseline_path,
            maintenance_path=maintenance_path,
            slider_percent=request.slider_percent,
            request_id=request_id
        )

        # Generate annotated overlay and upload to S3
        try:
            annotated_img = _create_annotated_image(
                baseline_path, maintenance_path, report.blobs
            )
        except Exception as exc:
            logger.warning("Annotated image generation failed: %s", exc)
            annotated_img = None

        if annotated_img is not None:
            async with httpx.AsyncClient(timeout=30.0) as upload_client:
                object_key = await _upload_annotated_image(
                    upload_client, annotated_img, request.annotated_upload_url
                )
            if object_key is not None:
                response_data["annotatedImageKey"] = object_key

        return JSONResponse(content=response_data)
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Anomaly detection failed: {str(e)}"
        )
    
    finally:
        # Clean up all temp files
        for path in [baseline_path, maintenance_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass


@app.post("/api/v1/detect-batch")
async def detect_anomalies_batch(request: BatchDetectRequest):
    """
    Batch detection: compare one baseline against multiple maintenance images.
    
    Accepts presigned S3 URLs. Downloads all images, runs detection on each,
    and returns all results and metadata.
    
    Args:
        request: JSON body with baseline_url, maintenance_urls list, optional slider_percent
    
    Returns:
        List of detection results for each maintenance image
    """
    
    results = []
    request_id = str(uuid.uuid4())
    baseline_path = os.path.join(TEMP_DIR, f"{request_id}_baseline.png")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Download baseline once
            await _download_image(client, request.baseline_url, baseline_path)
            
            # Process each maintenance image
            for idx, maint_url in enumerate(request.maintenance_urls):
                maintenance_path = os.path.join(TEMP_DIR, f"{request_id}_maintenance_{idx}.png")
                
                try:
                    await _download_image(client, maint_url, maintenance_path)
                    
                    report = detect_anomalies(
                        baseline_path=baseline_path,
                        maintenance_path=maintenance_path,
                        slider_percent=request.slider_percent
                    )
                    
                    anomalies = []
                    for i, blob in enumerate(report.blobs):
                        x, y, w, h = blob.bbox
                        anomalies.append({
                            "id": f"anomaly_{i+1}",
                            "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                            "confidence": float(blob.confidence),
                            "severity": blob.classification,
                            "severityScore": float(blob.severity),
                            "classification": blob.subtype,
                            "area": int(blob.area),
                            "centroid": {
                                "x": float(blob.centroid[0]),
                                "y": float(blob.centroid[1])
                            },
                            "meanDeltaE": float(blob.mean_deltaE),
                            "peakDeltaE": float(blob.peak_deltaE),
                            "elongation": float(blob.elongation)
                        })
                    
                    results.append({
                        "imageIndex": idx,
                        "imageLevelLabel": report.image_level_label,
                        "anomalyCount": len(anomalies),
                        "anomalies": anomalies,
                        "metrics": {
                            "meanSsim": float(report.mean_ssim),
                            "warpModel": report.warp_model,
                            "warpSuccess": report.warp_success,
                            "warpScore": float(report.warp_score),
                            "thresholdPotential": float(report.t_pot),
                            "thresholdFault": float(report.t_fault),
                            "thresholdSource": report.threshold_source,
                        }
                    })
                    
                finally:
                    if os.path.exists(maintenance_path):
                        try:
                            os.remove(maintenance_path)
                        except:
                            pass
        
        return JSONResponse(content={
            "requestId": request_id,
            "totalImages": len(request.maintenance_urls),
            "results": results
        })
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch detection failed: {str(e)}"
        )
    
    finally:
        if os.path.exists(baseline_path):
            try:
                os.remove(baseline_path)
            except:
                pass


if __name__ == "__main__":
    # Run with uvicorn for production use: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Disable in production
        log_level="info"
    )
