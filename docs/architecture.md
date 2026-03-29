---
title: Architecture
---

# Architecture

## High-level flow

```
artefex analyze <image>
    |
    v
+------------------------+
| 13 Built-in Detectors  |  JPEG, noise, color, resolution, screenshot,
| + Plugin Detectors     |  watermark, EXIF, platform, AI-gen, stego,
|                        |  camera ID, copy-move forgery
+------------------------+
    |
    v
+------------------------+
| Degradation Chain      |  Sorted by severity, graded A-F
+------------------------+
    |
    v
+------------------------+
| Restoration Pipeline   |  Neural (ONNX) -> Plugin -> Classical
+------------------------+
    |
    v
  restored image + report + heatmap + grade
```

## Module organization

```
src/artefex/
  cli.py             - Typer CLI (entry point, 31 commands)
  analyze.py         - DegradationAnalyzer (core detection engine)
  restore.py         - RestorationPipeline (neural + classical + plugin)
  models.py          - Degradation and AnalysisResult dataclasses
  api.py             - Public Python API wrappers
  grade.py           - A-F quality grading

  # Specialized detectors
  detect_aigen.py    - AI-generated content detection
  detect_stego.py    - Steganography detection
  detect_camera.py   - Camera/device identification (PRNU)
  detect_forgery.py  - Copy-move forgery detection
  fingerprint.py     - Platform fingerprinting (7 platforms)

  # Neural inference
  neural.py          - ONNX inference engine
  models_registry.py - Model import and management

  # Output formats
  report.py          - Plain text reports
  report_html.py     - Rich HTML reports
  heatmap.py         - Spatial degradation heatmaps
  story.py           - Natural language narratives
  gallery.py         - HTML side-by-side galleries
  dashboard.py       - Batch analysis dashboards

  # Infrastructure
  config.py          - TOML configuration loader
  plugins.py         - Entry-point plugin system
  parallel.py        - Multi-process batch processing
  watch.py           - Directory watcher
  web.py             - FastAPI web UI
  video.py           - Video analysis and restoration
  gif_analyze.py     - GIF/APNG frame analysis
  similarity.py      - Duplicate detection (pHash, aHash, dHash)
  orientation.py     - Orientation detection and correction
  palette.py         - Color palette extraction
  quality_gate.py    - CI/CD quality gate
  accessibility.py   - Color accessibility checker
```

## Key design decisions

**Diagnosis before treatment** - The `DegradationAnalyzer` always runs first. Restoration decisions are made based on detected degradations, not assumptions.

**Hybrid restoration** - `RestorationPipeline` tries neural models (ONNX) first for quality, then plugin restorers, then classical signal processing as fallback. No GPU required.

**Plugin architecture** - Detectors and restorers can be added via Python entry points without modifying core code. See `examples/custom_plugin.py`.

**Configuration cascade** - `.artefex.toml` (walks up directory tree) -> `[tool.artefex]` in `pyproject.toml` -> `~/.artefex.toml` -> defaults.

**Graceful degradation** - Optional detectors (AI-gen, stego, camera, forgery) load dynamically and fail silently if dependencies are missing.
