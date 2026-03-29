---
title: CLI Reference
---

# CLI Reference

All commands use the `artefex` entry point.

## Analysis

| Command | Description |
|---|---|
| `artefex analyze <path>` | Diagnose degradation chain |
| `artefex analyze <path> --json` | Machine-readable JSON output |
| `artefex analyze <path> --verbose` | Detailed detection info |
| `artefex analyze <url>` | Analyze image from URL |
| `artefex analyze <dir>` | Batch mode for directories |

## Quality grading

| Command | Description |
|---|---|
| `artefex grade <path>` | A-F grade with score |
| `artefex grade <dir> --export csv` | Batch export as CSV |
| `artefex grade <dir> --export markdown` | Batch export as markdown |

## Forensic tools

| Command | Description |
|---|---|
| `artefex report <path>` | Text forensic report |
| `artefex report <path> --html` | Rich HTML report with charts |
| `artefex timeline <path>` | ASCII degradation timeline |
| `artefex story <path>` | Natural language forensic narrative |
| `artefex heatmap <path>` | Spatial degradation heatmap |
| `artefex palette <path>` | Extract dominant color palette |
| `artefex orient <path> --fix` | Detect and fix orientation |
| `artefex audit <path>` | Comprehensive audit (all tools) |

## Restoration

| Command | Description |
|---|---|
| `artefex restore <path>` | Reverse the degradation chain |
| `artefex restore <path> --format png` | Convert output format |
| `artefex restore <path> --no-neural` | Classical methods only |
| `artefex restore <dir>` | Batch restore |
| `artefex restore-preview <path>` | Save each step as separate file |

## Comparison

| Command | Description |
|---|---|
| `artefex compare <a> <b>` | MSE, PSNR, SSIM, heatmap |
| `artefex gallery <dir-a> <dir-b>` | HTML side-by-side gallery |
| `artefex duplicates <dir>` | Find duplicate images |
| `artefex duplicates <dir> --threshold 0.8` | Adjust similarity threshold |

## Video

| Command | Description |
|---|---|
| `artefex video-analyze <path>` | Sample frames for degradation |
| `artefex video-restore <path>` | Restore frame by frame |

## Web and automation

| Command | Description |
|---|---|
| `artefex web` | Launch web UI with drag-and-drop |
| `artefex watch <dir> --restore` | Auto-process new images |
| `artefex dashboard <dir>` | Generate HTML overview dashboard |
| `artefex rename-by-grade <dir> --dry-run` | Preview grade-based renaming |
| `artefex parallel-analyze <dir>` | Multi-process batch analysis |

## System

| Command | Description |
|---|---|
| `artefex version` | Show version and dependency status |
| `artefex models list` | Show available neural models |
| `artefex models import <name> <path>` | Import an ONNX model |
| `artefex plugins` | List installed plugins |
