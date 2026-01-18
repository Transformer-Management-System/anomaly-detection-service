"""
FastAPI Microservice for Anomaly Detection using Computer Vision
"""
import os
import uuid
import tempfile
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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


@app.post("/api/v1/detect")
async def detect_anomalies_endpoint(
    baseline: UploadFile = File(..., description="Baseline image"),
    maintenance: UploadFile = File(..., description="Maintenance/inspection image"),
    transformer_id: str = Form(..., description="Transformer/asset identifier"),
    slider_percent: Optional[float] = Form(None, description="Threshold adjustment percentage")
):
    """
    Detect anomalies by comparing baseline and maintenance images.
    
    Args:
        baseline: Baseline reference image
        maintenance: Current maintenance/inspection image
        transformer_id: Unique identifier for the transformer/asset
        slider_percent: Optional threshold adjustment (-100 to 100)
    
    Returns:
        JSON with detection results including anomalies and overlay image path
    """
    
    # Validate file types
    for file in [baseline, maintenance]:
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. Only image files are accepted."
            )
    
    # Create unique temporary files
    request_id = str(uuid.uuid4())
    baseline_path = os.path.join(TEMP_DIR, f"{request_id}_baseline.png")
    maintenance_path = os.path.join(TEMP_DIR, f"{request_id}_maintenance.png")
    overlay_path = os.path.join(TEMP_DIR, f"{request_id}_overlay.png")
    report_path = os.path.join(TEMP_DIR, f"{request_id}_report.json")
    
    try:
        # Save uploaded files
        with open(baseline_path, "wb") as f:
            f.write(await baseline.read())
        
        with open(maintenance_path, "wb") as f:
            f.write(await maintenance.read())
        
        # Run anomaly detection
        report = detect_anomalies(
            transformer_id=transformer_id,
            baseline_path=baseline_path,
            maintenance_path=maintenance_path,
            out_overlay_path=overlay_path,
            out_json_path=report_path,
            slider_percent=slider_percent
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
                "classification": blob.subtype,
                "area": int(blob.area)
            })
        
        # Read overlay image as bytes
        with open(overlay_path, "rb") as f:
            overlay_bytes = f.read()
        
        # Prepare response
        response_data = {
            "requestId": request_id,
            "transformerId": transformer_id,
            "timestamp": datetime.utcnow().isoformat(),
            "imageLevelLabel": report.image_level_label,
            "anomalyCount": len(anomalies),
            "anomalies": anomalies,
            "metrics": {
                "meanSsim": float(report.mean_ssim),
                "warpModel": report.warp_model,
                "thresholdPotential": float(report.t_pot),
                "thresholdFault": float(report.t_fault),
                "basePotential": float(report.base_t_pot),
                "baseFault": float(report.base_t_fault),
                "sliderPercent": float(report.slider_percent) if report.slider_percent else None,
                "scaleApplied": float(report.scale_applied),
                "thresholdSource": report.threshold_source,
                "ratio": report.ratio
            },
            "overlayImage": {
                "filename": f"{request_id}_overlay.png",
                "size": len(overlay_bytes),
                "path": overlay_path
            }
        }
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        # Clean up on error
        for path in [baseline_path, maintenance_path, overlay_path, report_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Anomaly detection failed: {str(e)}"
        )
    
    finally:
        # Clean up input files (keep overlay for potential retrieval)
        for path in [baseline_path, maintenance_path, report_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass


@app.post("/api/v1/detect-batch")
async def detect_anomalies_batch(
    baseline: UploadFile = File(...),
    maintenances: list[UploadFile] = File(...),
    transformer_id: str = Form(...),
    slider_percent: Optional[float] = Form(None)
):
    """
    Batch detection: compare one baseline against multiple maintenance images.
    
    Args:
        baseline: Single baseline reference image
        maintenances: List of maintenance images to compare
        transformer_id: Transformer/asset identifier
        slider_percent: Optional threshold adjustment
    
    Returns:
        List of detection results for each maintenance image
    """
    
    results = []
    request_id = str(uuid.uuid4())
    baseline_path = os.path.join(TEMP_DIR, f"{request_id}_baseline.png")
    
    try:
        # Save baseline once
        with open(baseline_path, "wb") as f:
            f.write(await baseline.read())
        
        # Process each maintenance image
        for idx, maintenance in enumerate(maintenances):
            maintenance_path = os.path.join(TEMP_DIR, f"{request_id}_maintenance_{idx}.png")
            overlay_path = os.path.join(TEMP_DIR, f"{request_id}_overlay_{idx}.png")
            report_path = os.path.join(TEMP_DIR, f"{request_id}_report_{idx}.json")
            
            try:
                # Save maintenance image
                with open(maintenance_path, "wb") as f:
                    f.write(await maintenance.read())
                
                # Run detection
                report = detect_anomalies(
                    transformer_id=f"{transformer_id}_img{idx}",
                    baseline_path=baseline_path,
                    maintenance_path=maintenance_path,
                    out_overlay_path=overlay_path,
                    out_json_path=report_path,
                    slider_percent=slider_percent
                )
                
                # Build result
                anomalies = []
                for i, blob in enumerate(report.blobs):
                    x, y, w, h = blob.bbox
                    anomalies.append({
                        "id": f"anomaly_{i+1}",
                        "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                        "confidence": float(blob.confidence),
                        "severity": blob.classification,
                        "classification": blob.subtype
                    })
                
                results.append({
                    "imageIndex": idx,
                    "filename": maintenance.filename,
                    "imageLevelLabel": report.image_level_label,
                    "anomalyCount": len(anomalies),
                    "anomalies": anomalies,
                    "meanSsim": float(report.mean_ssim)
                })
                
            finally:
                # Clean up maintenance files
                for path in [maintenance_path, overlay_path, report_path]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass
        
        return JSONResponse(content={
            "requestId": request_id,
            "transformerId": transformer_id,
            "totalImages": len(maintenances),
            "results": results
        })
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch detection failed: {str(e)}"
        )
    
    finally:
        # Clean up baseline
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
