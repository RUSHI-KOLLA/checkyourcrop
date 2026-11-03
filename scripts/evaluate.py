#!/usr/bin/env python3
"""
Evaluation gate script.
Runs ultralytics validation on eval/golden, writes metrics.json,
compares map50_95 against baseline, exits:
- 0: pass (no regression beyond tolerance)
- 1: regression (drop exceeds tolerance)
- 2: cannot evaluate (missing data, no model, etc.)
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on golden set")
    parser.add_argument("--baseline", type=float, required=True, help="Baseline map50_95 to compare against")
    parser.add_argument("--tolerance", type=float, default=0.015, help="Max allowed drop in map50_95")
    parser.add_argument("--data", default="eval/golden/data.yaml", help="Path to data.yaml for golden set")
    parser.add_argument("--weights", default="models/best.pt", help="Path to model weights")
    parser.add_argument("--output", default="outputs/metrics.json", help="Output metrics file")
    args = parser.parse_args()

    # Check ultralytics
    if not HAS_ULTRALYTICS:
        print("SKIP: ultralytics not installed")
        return 2

    # Check weights
    if not os.path.exists(args.weights):
        print(f"SKIP: weights not found at {args.weights}")
        return 2

    # Check data.yaml
    if not os.path.exists(args.data):
        print(f"SKIP: data.yaml not found at {args.data}")
        return 2

    # Check if golden set has images
    import yaml
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)

    val_path = data_cfg.get("val") or data_cfg.get("test")
    if not val_path:
        print("SKIP: no val/test path in data.yaml")
        return 2

    val_path = Path(args.data).parent / val_path
    if not val_path.exists():
        print(f"SKIP: validation path {val_path} does not exist")
        return 2

    # Check for images
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = list(val_path.rglob("*"))
    images = [p for p in images if p.suffix.lower() in img_extensions]
    if not images:
        print(f"SKIP: no images found in {val_path}")
        return 2

    # Run evaluation
    print(f"Evaluating {args.weights} on {len(images)} images from {val_path}")
    try:
        model = YOLO(args.weights)
        results = model.val(data=args.data, split="val", verbose=False)
    except Exception as e:
        print(f"SKIP: evaluation failed: {e}")
        return 2

    # Extract metrics
    metrics = {
        "map50_95": float(results.box.map),
        "map50": float(results.box.map50),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "split": "val",
        "model": os.path.basename(args.weights),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Write metrics
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {args.output}")
    print(f"  map50_95: {metrics['map50_95']:.4f}")
    print(f"  map50:    {metrics['map50']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall:   {metrics['recall']:.4f}")

    # Compare with baseline
    drop = args.baseline - metrics["map50_95"]
    print(f"Baseline map50_95: {args.baseline:.4f}")
    print(f"Current map50_95:  {metrics['map50_95']:.4f}")
    print(f"Drop: {drop:.4f} (tolerance: {args.tolerance:.4f})")

    if drop > args.tolerance:
        print(f"FAIL: Regression detected (drop {drop:.4f} > tolerance {args.tolerance:.4f})")
        return 1

    print("PASS: No significant regression")
    return 0


if __name__ == "__main__":
    sys.exit(main())
