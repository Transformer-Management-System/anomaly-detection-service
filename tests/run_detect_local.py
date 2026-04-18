"""Local runner for `/api/v1/detect` logic using dataset images.

This script executes the same detection/response-building path as the FastAPI
single-detect endpoint, but uses local image files instead of S3 URLs.

Example:
uv run python tests/run_detect_local.py --dataset-root datasets --transformer T2 --baseline-image T2_normal_001.png --maintenance-image T2_faulty_001.png --slider-percent 50
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import cv2 as cv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import run_detection_from_paths
from anomaly_engine.alignment import ecc_align
from anomaly_engine.io_utils import read_bgr, to_gray
from anomaly_engine.visualization import overlay_detections

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class _OverlayBlob:
    bbox: tuple[int, int, int, int]
    classification: str
    subtype: str
    peak_deltaE: float
    confidence: float


def _list_images(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _pick_image(folder: Path, image_name: str | None, index: int, role: str) -> Path:
    images = _list_images(folder)
    if not images:
        raise ValueError(f"No image files found in {folder} for {role}")

    if image_name:
        candidate = folder / image_name
        if not candidate.exists():
            raise ValueError(f"Requested {role} image not found: {candidate}")
        if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Requested {role} image is not a supported image type: {candidate}")
        return candidate

    if index < 0 or index >= len(images):
        raise ValueError(
            f"{role} index {index} is out of range for {folder} (available: 0..{len(images)-1})"
        )
    return images[index]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run detect endpoint logic locally from dataset images"
    )
    parser.add_argument("--dataset-root", default="datasets", help="Path to datasets root")
    parser.add_argument("--transformer", required=True, help="Transformer folder, e.g. T1")
    parser.add_argument("--baseline-subdir", default="normal", help="Baseline image subdirectory")
    parser.add_argument("--maintenance-subdir", default="faulty", help="Maintenance image subdirectory")
    parser.add_argument("--baseline-image", help="Specific baseline image filename")
    parser.add_argument("--maintenance-image", help="Specific maintenance image filename")
    parser.add_argument("--baseline-index", type=int, default=0, help="Baseline image index when --baseline-image is not provided")
    parser.add_argument("--maintenance-index", type=int, default=0, help="Maintenance image index when --maintenance-image is not provided")
    parser.add_argument("--all-maintenance", action="store_true", help="Run baseline against every maintenance image in the selected folder")
    parser.add_argument("--slider-percent", type=float, help="Optional threshold slider percent")
    parser.add_argument("--output-json", help="Optional output JSON path")
    parser.add_argument("--print-full", action="store_true", help="Print full JSON results to stdout")
    parser.add_argument("--no-show-overlay", action="store_true", help="Do not display the final overlay image")
    return parser.parse_args()


def _align_maintenance_to_baseline(baseline_path: Path, maintenance_path: Path):
    """Replicate pipeline alignment so blob bboxes map correctly on overlay."""
    base_bgr = read_bgr(str(baseline_path))
    ment_bgr = read_bgr(str(maintenance_path))

    base_gray = to_gray(base_bgr)
    ment_gray = to_gray(ment_bgr)
    if ment_gray.shape != base_gray.shape:
        hs, ws = base_gray.shape
        ment_bgr = cv.resize(ment_bgr, (ws, hs), interpolation=cv.INTER_LINEAR)
        ment_gray = cv.resize(ment_gray, (ws, hs), interpolation=cv.INTER_LINEAR)

    warp, _, _, _ = ecc_align(base_gray, ment_gray)
    h, w = base_gray.shape

    if warp.shape == (3, 3):
        return cv.warpPerspective(
            ment_bgr,
            warp,
            (w, h),
            flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
        )
    if warp.shape == (2, 3):
        return cv.warpAffine(
            ment_bgr,
            warp,
            (w, h),
            flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
        )
    return ment_bgr


def _result_to_overlay_blobs(result: dict) -> List[_OverlayBlob]:
    blobs: List[_OverlayBlob] = []
    for anomaly in result.get("anomalies", []):
        bbox = anomaly.get("bbox", {})
        blobs.append(
            _OverlayBlob(
                bbox=(
                    int(bbox.get("x", 0)),
                    int(bbox.get("y", 0)),
                    int(bbox.get("width", 0)),
                    int(bbox.get("height", 0)),
                ),
                classification=str(anomaly.get("severity", "Normal")),
                subtype=str(anomaly.get("classification", "None")),
                peak_deltaE=float(anomaly.get("peakDeltaE", 0.0)),
                confidence=float(anomaly.get("confidence", 0.0)),
            )
        )
    return blobs


def _prepare_panel(image_bgr, title: str, target_size: tuple[int, int]):
    """Resize image and add a title strip for side-by-side display."""
    w, h = target_size
    panel = cv.resize(image_bgr, (w, h), interpolation=cv.INTER_AREA)
    cv.rectangle(panel, (0, 0), (w, 34), (20, 20, 20), -1)
    cv.putText(panel, title, (12, 24), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA)
    return panel


def _display_overlay(result: dict, baseline_path: Path, maintenance_path: Path) -> None:
    """Display baseline, maintenance, and overlay side by side.

    Falls back to browser display if OpenCV GUI is unavailable.
    """
    baseline_bgr = read_bgr(str(baseline_path))
    maintenance_bgr = read_bgr(str(maintenance_path))
    aligned_maintenance = _align_maintenance_to_baseline(baseline_path, maintenance_path)
    overlay = overlay_detections(aligned_maintenance, _result_to_overlay_blobs(result))

    panel_h, panel_w = baseline_bgr.shape[:2]
    baseline_panel = _prepare_panel(baseline_bgr, "Baseline", (panel_w, panel_h))
    maintenance_panel = _prepare_panel(maintenance_bgr, "Maintenance", (panel_w, panel_h))
    overlay_panel = _prepare_panel(overlay, "Overlay", (panel_w, panel_h))
    combined = cv.hconcat([baseline_panel, maintenance_panel, overlay_panel])

    window_title = f"Detection Preview | {maintenance_path.name}"
    window_opened = False
    try:
        cv.namedWindow(window_title, cv.WINDOW_NORMAL)
        window_opened = True
        cv.imshow(window_title, combined)
        print("Preview displayed. Press any key in the image window to close.")
        cv.waitKey(0)
    except cv.error as exc:
        print(
            "OpenCV window display is unavailable in this environment; "
            "falling back to browser display.",
        )

        ok, encoded = cv.imencode(".png", combined)
        if not ok:
            print(
                "Overlay encode failed; cannot display fallback image. "
                f"Original OpenCV error: {exc}",
                file=sys.stderr,
            )
            return

        img_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        html = (
            "<html><body style='margin:0;background:#111;display:flex;justify-content:center;align-items:center;'>"
            f"<img style='max-width:98vw;max-height:98vh' src='data:image/png;base64,{img_b64}' alt='overlay'/>"
            "</body></html>"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tmp:
            tmp.write(html)
            tmp_path = Path(tmp.name)
        webbrowser.open(tmp_path.resolve().as_uri())
        print(f"Overlay opened in browser: {tmp_path}")
    finally:
        if window_opened:
            try:
                cv.destroyAllWindows()
            except cv.error:
                pass


def main() -> int:
    args = _parse_args()

    dataset_root = Path(args.dataset_root)
    transformer_dir = dataset_root / args.transformer
    baseline_dir = transformer_dir / args.baseline_subdir
    maintenance_dir = transformer_dir / args.maintenance_subdir

    if not baseline_dir.exists():
        print(f"Baseline directory does not exist: {baseline_dir}", file=sys.stderr)
        return 2
    if not maintenance_dir.exists():
        print(f"Maintenance directory does not exist: {maintenance_dir}", file=sys.stderr)
        return 2

    try:
        baseline_image = _pick_image(
            folder=baseline_dir,
            image_name=args.baseline_image,
            index=args.baseline_index,
            role="baseline",
        )

        if args.all_maintenance:
            maintenance_images = _list_images(maintenance_dir)
            if args.maintenance_image:
                maintenance_images = [
                    p for p in maintenance_images if p.name == args.maintenance_image
                ]
                if not maintenance_images:
                    raise ValueError(
                        f"Requested maintenance image not found in {maintenance_dir}: {args.maintenance_image}"
                    )
        else:
            maintenance_images = [
                _pick_image(
                    folder=maintenance_dir,
                    image_name=args.maintenance_image,
                    index=args.maintenance_index,
                    role="maintenance",
                )
            ]

    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    runs = []
    last_result = None
    last_maintenance_image = None
    for maintenance_image in maintenance_images:
        result = run_detection_from_paths(
            baseline_path=str(baseline_image),
            maintenance_path=str(maintenance_image),
            slider_percent=args.slider_percent,
        )
        runs.append({
            "maintenanceImage": maintenance_image.name,
            "result": result,
        })

        metrics = result.get("metrics", {})
        print(
            f"{maintenance_image.name}: label={result.get('imageLevelLabel')} "
            f"anomalies={result.get('anomalyCount')} "
            f"ssim={metrics.get('meanSsim')}"
        )
        last_result = result
        last_maintenance_image = maintenance_image

    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "datasetRoot": str(dataset_root.resolve()),
        "transformer": args.transformer,
        "baselineImage": baseline_image.name,
        "baselinePath": str(baseline_image),
        "maintenanceSubdir": args.maintenance_subdir,
        "sliderPercent": args.slider_percent,
        "totalRuns": len(runs),
        "runs": runs,
    }

    output_path = (
        Path(args.output_json)
        if args.output_json
        else Path("tests")
        / "output"
        / f"local_detect_{args.transformer}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved results to: {output_path}")
    if args.print_full:
        print(json.dumps(payload, indent=2))

    if not args.no_show_overlay and last_result is not None and last_maintenance_image is not None:
        _display_overlay(last_result, baseline_image, last_maintenance_image)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
