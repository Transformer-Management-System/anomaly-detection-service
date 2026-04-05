# Anomaly Detection Service — API Documentation

**Base URL:** `http://localhost:8000`  
**Version:** 1.0.0  
**Interactive docs (when running):** `http://localhost:8000/docs` (Swagger UI) · `http://localhost:8000/redoc` (ReDoc)

---

## Running the Service

```bash
# Development (auto-reload on file changes)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Endpoints

### 1. `GET /`

Root health check.

**Response `200 OK`**
```json
{
  "service": "Anomaly Detection Service",
  "status": "running",
  "timestamp": "2026-04-05T10:00:00.000000"
}
```

---

### 2. `GET /health`

Lightweight liveness probe for container orchestration (Kubernetes, ECS, etc.).

**Response `200 OK`**
```json
{
  "status": "healthy"
}
```

---

### 3. `POST /api/v1/detect`

Detect anomalies by comparing a baseline image against a maintenance/inspection image. Both images are fetched from AWS S3 presigned URLs. All detection metadata is returned in the response so the calling backend can persist it without any database writes happening in this service.

#### Request

**Content-Type:** `application/json`

| Field | Type | Required | Description |
|---|---|---|---|
| `baseline_url` | `string` | ✅ | Presigned S3 URL for the baseline reference image |
| `maintenance_url` | `string` | ✅ | Presigned S3 URL for the maintenance/inspection image |
| `slider_percent` | `float` | ❌ | Threshold sensitivity adjustment. Range `0.0–100.0`. `0` = most sensitive (lower thresholds), `100` = least sensitive (higher thresholds). Defaults to adaptive SSIM-based thresholds when omitted. |

**Example request body**
```json
{
  "baseline_url": "https://your-bucket.s3.amazonaws.com/transformers/baseline.jpg?X-Amz-Algorithm=...",
  "maintenance_url": "https://your-bucket.s3.amazonaws.com/transformers/inspection.jpg?X-Amz-Algorithm=...",
  "slider_percent": 50.0
}
```

#### Response

**`200 OK`**

| Field | Type | Description |
|---|---|---|
| `requestId` | `string` (UUID) | Unique ID for this detection request |
| `timestamp` | `string` (ISO 8601) | UTC timestamp of the detection run |
| `imageLevelLabel` | `string` | Overall image classification: `"Normal"`, `"Potentially Faulty"`, or `"Faulty"` |
| `anomalyCount` | `integer` | Total number of anomaly blobs detected |
| `anomalies` | `Anomaly[]` | Per-blob detection details (see Anomaly Object below) |
| `metrics` | `Metrics` | Pipeline diagnostics and threshold metadata (see Metrics Object below) |

##### Anomaly Object

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Sequential identifier, e.g. `"anomaly_1"` |
| `bbox` | `object` | Bounding box in image pixel coordinates |
| `bbox.x` | `integer` | Left edge (pixels) |
| `bbox.y` | `integer` | Top edge (pixels) |
| `bbox.width` | `integer` | Box width (pixels) |
| `bbox.height` | `integer` | Box height (pixels) |
| `confidence` | `float` | Detection confidence score, range `0.0–1.0` |
| `severity` | `string` | Severity class: `"Normal"`, `"Potentially Faulty"`, or `"Faulty"` |
| `severityScore` | `float` | Numeric severity score, range `0.0–100.0` |
| `classification` | `string` | Anomaly subtype: `"LooseJoint"`, `"PointOverload"`, `"FullWireOverload"`, or `"None"` |
| `area` | `integer` | Blob area in pixels |
| `centroid.x` | `float` | Centroid X coordinate (pixels) |
| `centroid.y` | `float` | Centroid Y coordinate (pixels) |
| `meanDeltaE` | `float` | Mean CIELAB colour difference (ΔE) within the blob |
| `peakDeltaE` | `float` | Maximum CIELAB colour difference (ΔE) within the blob |
| `meanHsv.h` | `float` | Mean Hue value within the blob (0–180, OpenCV scale) |
| `meanHsv.s` | `float` | Mean Saturation within the blob (0–255) |
| `meanHsv.v` | `float` | Mean Value/Brightness within the blob (0–255) |
| `elongation` | `float` | Major/minor axis ratio of the blob shape |

##### Metrics Object

| Field | Type | Description |
|---|---|---|
| `meanSsim` | `float` | Mean Structural Similarity Index (SSIM) between aligned images, range `0.0–1.0` |
| `warpModel` | `string` | Alignment model used: `"homography"` or `"affine"` |
| `warpSuccess` | `boolean` | Whether ECC image alignment converged successfully |
| `warpScore` | `float` | ECC alignment convergence score |
| `thresholdPotential` | `float` | Final ΔE threshold used for "Potentially Faulty" classification |
| `thresholdFault` | `float` | Final ΔE threshold used for "Faulty" classification |
| `basePotential` | `float` | SSIM-adaptive base threshold for "Potentially Faulty" (before slider) |
| `baseFault` | `float` | SSIM-adaptive base threshold for "Faulty" (before slider) |
| `sliderPercent` | `float \| null` | The `slider_percent` value that was passed in, or `null` if not provided |
| `scaleApplied` | `float \| null` | Computed scale factor from slider, or `null` if no slider was used |
| `thresholdSource` | `string` | Describes how thresholds were derived. Values: `"adaptive_ssim"`, `"slider_scaled"`, `"adaptive_ssim+palette_soften"`, `"slider_scaled+palette_soften"` |
| `ratio` | `float` | `thresholdFault / thresholdPotential` ratio used for consistent scaling |

**Example response**
```json
{
  "requestId": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "timestamp": "2026-04-05T10:23:45.123456",
  "imageLevelLabel": "Potentially Faulty",
  "anomalyCount": 2,
  "anomalies": [
    {
      "id": "anomaly_1",
      "bbox": {
        "x": 142,
        "y": 87,
        "width": 38,
        "height": 44
      },
      "confidence": 0.82,
      "severity": "Potentially Faulty",
      "severityScore": 54.3,
      "classification": "PointOverload",
      "area": 892,
      "centroid": { "x": 161.4, "y": 109.2 },
      "meanDeltaE": 11.3,
      "peakDeltaE": 17.8,
      "meanHsv": { "h": 14.2, "s": 187.0, "v": 231.0 },
      "elongation": 1.4
    }
  ],
  "metrics": {
    "meanSsim": 0.874,
    "warpModel": "homography",
    "warpSuccess": true,
    "warpScore": 0.021,
    "thresholdPotential": 8.0,
    "thresholdFault": 12.0,
    "basePotential": 8.0,
    "baseFault": 12.0,
    "sliderPercent": 50.0,
    "scaleApplied": 1.0,
    "thresholdSource": "slider_scaled",
    "ratio": 1.5
  }
}
```

#### Error Responses

| Status | Condition |
|---|---|
| `400` | A URL returned non-image content |
| `502` | A presigned URL download failed (S3 error, expired URL, etc.) |
| `500` | Internal detection pipeline error |

**Error body format**
```json
{
  "detail": "Human-readable error message"
}
```

---

### 4. `POST /api/v1/detect-batch`

Compare a single baseline image against multiple maintenance images in one request. Each maintenance image is processed independently against the same baseline.

#### Request

**Content-Type:** `application/json`

| Field | Type | Required | Description |
|---|---|---|---|
| `baseline_url` | `string` | ✅ | Presigned S3 URL for the shared baseline reference image |
| `maintenance_urls` | `string[]` | ✅ | Array of presigned S3 URLs, one per maintenance image |
| `slider_percent` | `float` | ❌ | Same sensitivity adjustment as `/detect`, applied to all images |

**Example request body**
```json
{
  "baseline_url": "https://your-bucket.s3.amazonaws.com/transformers/baseline.jpg?X-Amz-...",
  "maintenance_urls": [
    "https://your-bucket.s3.amazonaws.com/transformers/inspection_jan.jpg?X-Amz-...",
    "https://your-bucket.s3.amazonaws.com/transformers/inspection_feb.jpg?X-Amz-...",
    "https://your-bucket.s3.amazonaws.com/transformers/inspection_mar.jpg?X-Amz-..."
  ],
  "slider_percent": 40.0
}
```

#### Response

**`200 OK`**

| Field | Type | Description |
|---|---|---|
| `requestId` | `string` (UUID) | Unique ID for the batch request |
| `totalImages` | `integer` | Number of maintenance images processed |
| `results` | `BatchResult[]` | Per-image results (see BatchResult Object below) |

##### BatchResult Object

| Field | Type | Description |
|---|---|---|
| `imageIndex` | `integer` | Zero-based index corresponding to the position in `maintenance_urls` |
| `imageLevelLabel` | `string` | Overall label for this image: `"Normal"`, `"Potentially Faulty"`, or `"Faulty"` |
| `anomalyCount` | `integer` | Number of anomalies detected in this image |
| `anomalies` | `Anomaly[]` | Same structure as in `/detect` (excludes `meanHsv` field) |
| `metrics.meanSsim` | `float` | SSIM score for this image pair |
| `metrics.warpModel` | `string` | Alignment model used |
| `metrics.warpSuccess` | `boolean` | Whether alignment converged |
| `metrics.warpScore` | `float` | Alignment convergence score |
| `metrics.thresholdPotential` | `float` | Final ΔE threshold for "Potentially Faulty" |
| `metrics.thresholdFault` | `float` | Final ΔE threshold for "Faulty" |
| `metrics.thresholdSource` | `string` | How thresholds were derived |

**Example response**
```json
{
  "requestId": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "totalImages": 2,
  "results": [
    {
      "imageIndex": 0,
      "imageLevelLabel": "Normal",
      "anomalyCount": 0,
      "anomalies": [],
      "metrics": {
        "meanSsim": 0.951,
        "warpModel": "homography",
        "warpSuccess": true,
        "warpScore": 0.009,
        "thresholdPotential": 8.0,
        "thresholdFault": 12.0,
        "thresholdSource": "adaptive_ssim"
      }
    },
    {
      "imageIndex": 1,
      "imageLevelLabel": "Faulty",
      "anomalyCount": 1,
      "anomalies": [
        {
          "id": "anomaly_1",
          "bbox": { "x": 200, "y": 150, "width": 55, "height": 60 },
          "confidence": 0.91,
          "severity": "Faulty",
          "severityScore": 78.1,
          "classification": "FullWireOverload",
          "area": 1540,
          "centroid": { "x": 227.5, "y": 180.0 },
          "meanDeltaE": 15.6,
          "peakDeltaE": 24.1,
          "elongation": 2.1
        }
      ],
      "metrics": {
        "meanSsim": 0.831,
        "warpModel": "affine",
        "warpSuccess": true,
        "warpScore": 0.034,
        "thresholdPotential": 8.0,
        "thresholdFault": 12.0,
        "thresholdSource": "adaptive_ssim"
      }
    }
  ]
}
```

#### Error Responses

Same error format as `/detect`.

| Status | Condition |
|---|---|
| `400` | A URL returned non-image content |
| `502` | A presigned URL download failed |
| `500` | Internal detection pipeline error |

---

## Annotation Rendering Reference

The `anomalies[].bbox` and `anomalies[].severity` fields contain everything needed for the frontend to render annotations on the original maintenance image without any server-side overlay generation.

**Colour mapping**

| `severity` | Hex | RGB |
|---|---|---|
| `"Faulty"` | `#FF0000` | `rgb(255, 0, 0)` |
| `"Potentially Faulty"` | `#FFA500` | `rgb(255, 165, 0)` |
| `"Normal"` | `#00FF00` | `rgb(0, 255, 0)` |

**Label format** (matches the OpenCV overlay exactly)
```
{severity}:{classification} pΔE={peakDeltaE} conf={confidence}
// e.g. "Potentially Faulty:PointOverload pΔE=17.8 conf=0.82"
```

**Canvas rendering example (React)**
```js
const drawAnnotations = (ctx, anomalies) => {
  const colorMap = {
    "Faulty": "#FF0000",
    "Potentially Faulty": "#FFA500",
    "Normal": "#00FF00",
  };
  anomalies.forEach(a => {
    const { x, y, width, height } = a.bbox;
    const color = colorMap[a.severity] ?? "#00FF00";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);
    ctx.fillStyle = color;
    ctx.font = "11px monospace";
    const label = `${a.severity}:${a.classification} pΔE=${a.peakDeltaE.toFixed(1)} conf=${a.confidence.toFixed(2)}`;
    ctx.fillText(label, x, Math.max(0, y - 5));
  });
};
```

---

## Classification Reference

### `imageLevelLabel` / `severity`

| Value | Meaning |
|---|---|
| `"Normal"` | No significant thermal anomalies detected |
| `"Potentially Faulty"` | One or more blobs exceed the potential fault threshold; monitoring recommended |
| `"Faulty"` | One or more blobs exceed the fault threshold; immediate inspection recommended |

### `classification` (subtype)

| Value | Meaning |
|---|---|
| `"LooseJoint"` | Hotspot localized at a connection point or joint |
| `"PointOverload"` | Small isolated high-temperature region suggesting point overload |
| `"FullWireOverload"` | Elongated hotspot suggesting overload across a wire/conductor segment |
| `"None"` | Unclassified or ambiguous anomaly shape |

### `thresholdSource`

| Value | Meaning |
|---|---|
| `"adaptive_ssim"` | Thresholds set automatically based on SSIM score |
| `"slider_scaled"` | Thresholds scaled from SSIM base using `slider_percent` |
| `"adaptive_ssim+palette_soften"` | SSIM-adaptive, then further softened due to low histogram correlation between images |
| `"slider_scaled+palette_soften"` | Slider-scaled, then further softened due to low histogram correlation |
