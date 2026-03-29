---
title: Python API
---

# Python API

Artefex provides a clean Python API for programmatic use.

## Basic usage

```python
from artefex import analyze, restore, grade

# Diagnose degradation chain
result = analyze("photo.jpg")
for d in result.degradations:
    print(f"{d.name}: {d.confidence:.0%} confidence, severity {d.severity:.0%}")

# Quality grade
grade_result = grade("photo.jpg")
print(f"Grade: {grade_result}")

# Restore
restore("photo.jpg", output="photo_restored.png")
```

## Available functions

| Function | Description |
|---|---|
| `analyze(path)` | Run all detectors, return `AnalysisResult` |
| `restore(path, output=None)` | Restore image, optionally save to output path |
| `grade(path)` | Return A-F quality grade |
| `compare(path_a, path_b)` | Compare two images (MSE, PSNR, SSIM) |
| `find_duplicates(directory)` | Find duplicate images in a directory |
| `generate_heatmap(path)` | Generate spatial degradation heatmap |
| `detect_platform(path)` | Identify source platform (Twitter, Instagram, etc.) |

## Data models

### AnalysisResult

Returned by `analyze()`. Contains:

- `file_path` - path to the analyzed file
- `file_size` - size in bytes
- `dimensions` - (width, height) tuple
- `format` - image format string
- `degradations` - list of `Degradation` objects

### Degradation

Each detected degradation contains:

- `name` - detector name (e.g. "jpeg_artifacts")
- `confidence` - float 0.0-1.0
- `severity` - float 0.0-1.0
- `detail` - human-readable description
- `category` - category string (e.g. "compression", "noise")

## Batch processing

```python
from artefex import analyze
from pathlib import Path

results = []
for img in Path("./photos").glob("*.jpg"):
    result = analyze(str(img))
    results.append(result)

# Sort by worst quality
results.sort(key=lambda r: len(r.degradations), reverse=True)
```

See `examples/batch_processing.py` for a more complete example.
