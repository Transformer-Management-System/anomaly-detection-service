"""
FastAPI Microservice for Anomaly Detection using Computer Vision
"""
import os
import uuid
import tempfile
from typing import Optional, List
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn

from anomaly_cv import detect_anomalies

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
        
        # Run anomaly detection
        report = detect_anomalies(
            baseline_path=baseline_path,
            maintenance_path=maintenance_path,
            slider_percent=request.slider_percent
        )
        
        # Build response with anomaly details
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
        
        # Prepare response — all metadata included for backend persistence
        response_data = {
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
