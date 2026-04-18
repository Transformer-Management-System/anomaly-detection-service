"""Local runner for `/api/v1/detect` logic using dataset images.

This script executes the same detection/response-building path as the FastAPI
single-detect endpoint, but uses local image files instead of S3 URLs.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import run_detection_from_paths

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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
    return parser.parse_args()


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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
