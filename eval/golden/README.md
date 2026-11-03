# Golden Evaluation Set

This directory contains a held-out dataset used by the evaluation gate to detect model regression.

## Structure

```
eval/golden/
├── data.yaml          # Dataset configuration (this file)
└── images/
    └── val/           # Validation images (~50 images from test split)
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

## Setup

1. Split your original dataset into train/val/test
2. Copy ~50 images from the **test** split into `eval/golden/images/val/`
3. Ensure corresponding label files (YOLO format .txt) exist alongside images
4. The gate will run `yolo val` on this set and compare map50_95 against baseline

## Baseline

After initial training, record the map50_95 from the test set and store it in `outputs/baseline_metrics.json`:

```json
{
  "map50_95": 0.7234,
  "map50": 0.8912,
  "precision": 0.78,
  "recall": 0.71,
  "split": "test",
  "model": "best.pt",
  "timestamp": "2026-01-15T10:30:00Z"
}
```

## Running the Gate Locally

```bash
python scripts/evaluate.py --baseline 0.7234 --data eval/golden/data.yaml --weights models/best.pt
```

Exit codes:
- 0: PASS (no significant regression)
- 1: FAIL (regression detected)
- 2: SKIP (cannot evaluate - missing data, weights, etc.)