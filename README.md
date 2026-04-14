# Anomaly Detection Service

A computer vision-based microservice for detecting thermal anomalies in transformer images using image comparison and analysis techniques.

## 🎯 Overview

This service provides REST APIs for detecting anomalies in transformer inspection images by comparing baseline (reference) images with maintenance/inspection images. It uses advanced computer vision algorithms including image alignment, structural similarity analysis, color space transformations, and morphological operations to identify potential faults and overheating issues.

## ✨ Features

- **Image Alignment**: Robust image registration using ECC (Enhanced Correlation Coefficient) algorithm with ORB feature-based fallback
- **Anomaly Detection**: Identifies thermal anomalies using:
  - Structural Similarity Index (SSIM) for overall image comparison
  - Delta E color difference metrics in LAB color space
  - HSV color space analysis for hot spot detection
  - Morphological operations for noise reduction
- **Classification System**: Categorizes anomalies into:
  - **Faulty**: Critical issues requiring immediate attention (red/orange hot spots)
  - **Potentially Faulty**: Issues requiring monitoring (yellow warm spots, elongated patterns)
  - **Normal**: No significant anomalies detected
- **Subtype Detection**: Identifies specific anomaly types:
  - `LooseJoint`: Thermal issues near connection points
  - `PointOverload`: Localized overheating
  - `FullWireOverload`: Extended wire overheating
- **Flexible Thresholding**: Adjustable sensitivity via slider parameter (0 to 100)
- **Batch Processing**: Compare one baseline against multiple inspection images
- **RESTful API**: FastAPI-based endpoints with OpenAPI documentation

## 🏗️ Architecture

### Service Layer
- **FastAPI Application** (`main.py`): RESTful microservice with CORS support
- **Two Main Endpoints**:
  - `/api/v1/detect`: Single image pair comparison
  - `/api/v1/detect-batch`: Batch processing (one baseline vs multiple maintenance images)

### Detection Engine (`anomaly_engine/`)

Modular computer vision pipeline:

1. **I/O & Preprocessing** (`io_utils.py`): Image loading and color space conversions
2. **Alignment** (`alignment.py`): Image registration using ECC and feature-based methods
3. **Color Analysis** (`color_metrics.py`): LAB/HSV color space transformations and Delta E calculations
4. **Morphology** (`morphology.py`): Noise filtering and blob extraction
5. **Topology** (`topology.py`): Wire skeleton analysis and joint detection
6. **Blob Analysis** (`blobs.py`): Connected component properties extraction
7. **Classification** (`classification.py`): Rule-based anomaly categorization
8. **Visualization** (`visualization.py`): Overlay image generation with bounding boxes
9. **Detection Pipeline** (`detection.py`): Orchestrates the complete analysis workflow

### Data Structures (`data_structures.py`)

- **BlobDet**: Individual anomaly blob with bbox, area, color metrics, classification
- **DetectionReport**: Complete detection result with metadata, thresholds, and blob list

## 📋 Prerequisites

- Python 3.8+
- pip package manager

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Transformer-Management-System/anomaly-detection-service.git
   cd anomaly-detection-service
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

```
fastapi==0.109.0          # Web framework
uvicorn[standard]==0.27.0 # ASGI server
python-multipart==0.0.6   # File upload support
numpy                     # Numerical computing
opencv-python-headless    # Computer vision
scikit-image              # Image processing (SSIM)
```

## 💻 Usage

### Starting the Service

```bash
# Development mode (with auto-reload)
python main.py

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

The service will be available at `http://localhost:8000`

### API Documentation

Interactive API documentation is automatically available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### 1. Health Check
```bash
GET /
GET /health
```

**Response**:
```json
{
  "service": "Anomaly Detection Service",
  "status": "running",
  "timestamp": "2024-01-19T03:00:00.000000"
}
```

#### 2. Single Image Detection
```bash
POST /api/v1/detect
```

**Request** (multipart/form-data):
- `baseline`: Baseline reference image file
- `maintenance`: Inspection/maintenance image file
- `transformer_id`: Unique identifier (string)
- `slider_percent`: Optional threshold adjustment. Lower values are stricter, higher values are more sensitive.

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "baseline=@baseline.jpg" \
  -F "maintenance=@inspection.jpg" \
  -F "transformer_id=TF-001" \
  -F "slider_percent=0"
```

**Response**:
```json
{
  "requestId": "uuid-string",
  "transformerId": "TF-001",
  "timestamp": "2024-01-19T03:00:00.000000",
  "imageLevelLabel": "Faulty",
  "anomalyCount": 2,
  "anomalies": [
    {
      "id": "anomaly_1",
      "bbox": {"x": 840, "y": 250, "width": 179, "height": 50},
      "confidence": 0.95,
      "severity": "Faulty",
      "classification": "LooseJoint",
      "area": 4298
    }
  ],
  "metrics": {
    "meanSsim": 0.678,
    "warpModel": "affine",
    "thresholdPotential": 10.0,
    "thresholdFault": 14.0,
    "basePotential": 10.0,
    "baseFault": 14.0,
    "sliderPercent": 0.0,
    "scaleApplied": 1.0,
    "thresholdSource": "baseline",
    "ratio": 1.4
  },
  "overlayImage": {
    "filename": "uuid_overlay.png",
    "size": 123456,
    "path": "/tmp/uuid_overlay.png"
  }
}
```

#### 3. Batch Detection
```bash
POST /api/v1/detect-batch
```

**Request** (multipart/form-data):
- `baseline`: Single baseline reference image
- `maintenances`: Multiple maintenance image files
- `transformer_id`: Unique identifier
- `slider_percent`: Optional threshold adjustment. Lower values are stricter, higher values are more sensitive.

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/api/v1/detect-batch" \
  -F "baseline=@baseline.jpg" \
  -F "maintenances=@inspection1.jpg" \
  -F "maintenances=@inspection2.jpg" \
  -F "maintenances=@inspection3.jpg" \
  -F "transformer_id=TF-001"
```

**Response**:
```json
{
  "requestId": "uuid-string",
  "transformerId": "TF-001",
  "totalImages": 3,
  "results": [
    {
      "imageIndex": 0,
      "filename": "inspection1.jpg",
      "imageLevelLabel": "Faulty",
      "anomalyCount": 2,
      "anomalies": [...],
      "meanSsim": 0.678
    }
  ]
}
```

### CLI Usage

The detection engine can also be used directly from command line:

```bash
python anomaly_cv.py <transformer_id> <baseline.jpg> <maintenance.jpg> <overlay.png> <report.json> [slider_percent]
```

**Example**:
```bash
python anomaly_cv.py TF-001 baseline.jpg inspection.jpg output_overlay.png report.json 0
```

## 🔧 Configuration

### Threshold Adjustment (slider_percent)

The `slider_percent` parameter allows fine-tuning detection sensitivity:

- **Lower values** (closer to `0`): Stricter detection, with higher thresholds and fewer detected anomalies
  - Example: `0` keeps the strictest slider-adjusted thresholds in the current pipeline
- **Higher values** (closer to `100`): More sensitive detection, with lower thresholds and more detected anomalies
  - Example: `100` makes the pipeline more likely to flag smaller color differences
- **Mid-range values**: Blend between the two; for example `50` is close to the baseline threshold scale used by the code
- **Default** (not provided): Uses adaptive SSIM-based thresholds

### CORS Configuration

Update `main.py` to configure allowed origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📁 Project Structure

```
anomaly-detection-service/
├── main.py                  # FastAPI application entry point
├── anomaly_cv.py           # Legacy wrapper with CLI support
├── requirements.txt        # Python dependencies
├── __init__.py            # Package initializer
│
├── anomaly_engine/        # Core detection engine (modular)
│   ├── __init__.py       # Package exports
│   ├── detection.py      # Main detection pipeline
│   ├── data_structures.py # BlobDet, DetectionReport classes
│   ├── alignment.py      # ECC/ORB image alignment
│   ├── color_metrics.py  # LAB/HSV and Delta E calculations
│   ├── morphology.py     # Blob extraction and cleaning
│   ├── topology.py       # Wire skeleton and joint analysis
│   ├── blobs.py         # Connected component properties
│   ├── classification.py # Anomaly categorization rules
│   ├── visualization.py  # Overlay generation
│   └── io_utils.py      # Image I/O utilities
│
├── baselines/            # Reference baseline images
├── inspections/          # Inspection data and results
│   └── <inspection_id>/
│       ├── index.json   # Inspection metadata
│       └── runs/
│           └── <run_id>/
│               ├── report.json    # Detection report
│               ├── anomalies.json # Anomaly details
│               ├── baseline.png
│               ├── maintenance.png
│               └── overlay.png
│
└── .gitignore           # Git ignore patterns
```

## 🔬 Detection Algorithm

### Pipeline Overview

1. **Image Loading & Preprocessing**
   - Load baseline and maintenance images
   - Convert to grayscale for alignment
   - Resize if dimensions don't match

2. **Image Alignment**
   - Use ECC (Enhanced Correlation Coefficient) on Canny edges
   - Apply mask to ignore legend and sky regions
   - Fallback to ORB feature matching + RANSAC if ECC fails
   - Warp maintenance image to align with baseline

3. **Similarity Analysis**
   - Calculate SSIM (Structural Similarity Index)
   - Compute difference map

4. **Color Analysis**
   - Convert to LAB color space
   - Calculate Delta E (color difference)
   - Identify hot regions in HSV space

5. **Morphology & Blob Extraction**
   - Apply morphological operations to clean noise
   - Extract connected components (blobs)
   - Calculate blob properties (area, centroid, elongation)

6. **Topology Analysis**
   - Build wire skeleton
   - Identify joints and connection points
   - Analyze wire coverage by hot regions

7. **Classification**
   - Apply rule-based classification:
     - Color thresholds (red/orange = faulty, yellow = potential)
     - Elongation ratio (high = potential wire overload)
     - Proximity to joints (near joint = loose joint)
     - Wire coverage (high = full wire overload)
   - Assign severity and confidence scores

8. **Visualization**
   - Generate overlay image with bounding boxes
   - Color-code by severity (red=faulty, yellow=potential)

### Key Algorithms

- **ECC Alignment**: Maximizes correlation between edge images
- **Delta E (CIE 2000)**: Perceptually uniform color difference
- **SSIM**: Structural similarity considering luminance, contrast, and structure
- **Morphological Operations**: Opening, closing, and distance transforms
- **Watershed Segmentation**: Separates merged blobs
- **Skeleton Analysis**: Medial axis transform for wire topology

## 🛠️ Development

### Running Tests

```bash
# If tests are available
pytest
```

### Code Structure

The codebase follows a modular design:
- **Separation of Concerns**: Each module has a specific responsibility
- **Data Classes**: Type-safe data structures with dataclasses
- **Functional Style**: Pure functions where possible
- **Legacy Compatibility**: `anomaly_cv.py` maintains backward compatibility

## 🤝 Integration

This service is designed to integrate with the Transformer Management System:

- **Spring Boot Backend**: Calls this microservice for image analysis
- **Frontend Application**: Displays detection results and overlays
- **Database**: Stores inspection metadata and results in JSON format

## 📊 Output Examples

### Detection Report Structure

The service generates comprehensive reports including:
- Image-level classification (Normal/Potentially Faulty/Faulty)
- Individual anomaly details with bounding boxes
- Color metrics (Delta E, HSV values)
- Alignment quality metrics (SSIM, warp score)
- Threshold information and sensitivity settings

### Overlay Images

Generated overlay images show:
- Red bounding boxes: Faulty anomalies
- Yellow bounding boxes: Potentially faulty anomalies
- Labels with anomaly types and confidence scores

## 🐛 Troubleshooting

### Common Issues

1. **Image format not supported**
   - Ensure images are in common formats (JPEG, PNG)
   - Check that files have correct MIME types

2. **Alignment failures**
   - Images may be too different (different angles, crops)
   - Try providing images with similar framing

3. **Memory issues**
   - Large images consume significant memory
   - Consider downscaling images before upload

4. **Port already in use**
   ```bash
   # Use a different port
   uvicorn main:app --host 0.0.0.0 --port 8001
   ```

## 📝 License

This project is part of the Transformer Management System.

## 👥 Contributors

Part of the Transformer Management System project.

## 📞 Support

For issues and questions, please create an issue in the GitHub repository.