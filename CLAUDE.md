# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Artefex is a neural forensic restoration tool for images, video, and GIFs. It diagnoses degradation chains (compression, noise, upscaling, watermarks, etc.) then intelligently reverses each step. Think `git log` for media degradation, followed by undo.

**Version**: 1.0.0 | **License**: MIT | **Python**: 3.10+

## Common Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_analyze.py -v

# Run a single test
pytest tests/test_analyze.py::test_jpeg_detection -v

# Lint
ruff check src/ tests/

# Run with coverage
pytest tests/ --cov=artefex --cov-report=term-missing
```

## Architecture

**Entry point**: `src/artefex/cli.py` - Typer CLI app (`artefex` command)

**Core flow**: CLI command -> `DegradationAnalyzer.analyze()` -> runs 13+ detectors on image -> returns `AnalysisResult` with list of `Degradation` objects -> optional `RestorationPipeline.restore()` reverses detected degradations.

**Key modules** in `src/artefex/`:

| Layer | Modules | Role |
|-------|---------|------|
| Detection | `analyze.py`, `detect_aigen.py`, `detect_stego.py`, `detect_camera.py`, `detect_forgery.py`, `fingerprint.py` | Degradation diagnosis. Main class is `DegradationAnalyzer` with `_detect_<name>()` methods. |
| Restoration | `restore.py`, `neural.py`, `models_registry.py` | `RestorationPipeline` tries neural (ONNX) first, then plugin restorers, then classical signal processing. |
| Data models | `models.py` | `Degradation` and `AnalysisResult` dataclasses. |
| Output | `report.py`, `report_html.py`, `heatmap.py`, `story.py`, `gallery.py`, `dashboard.py` | Various report formats. |
| Public API | `api.py`, `__init__.py` | Programmatic access: `analyze()`, `restore()`, `grade()`, `compare()`. |
| Infrastructure | `config.py`, `plugins.py`, `parallel.py`, `watch.py`, `web.py`, `video.py`, `gif_analyze.py` | Config loading, plugin system, batch processing, web UI, video/GIF support. |

**Plugin system**: Entry-point based (`artefex.detectors` / `artefex.restorers`). See `examples/custom_plugin.py`.

**Config precedence**: `.artefex.toml` (walks up directory tree) -> `[tool.artefex]` in `pyproject.toml` -> `~/.artefex.toml` -> defaults.

## Adding Features

**New detector**: Add `_detect_<name>()` method to `DegradationAnalyzer` in `analyze.py`, register in the `detectors` list in `analyze()`, add recommendation in `report.py`, write tests in `tests/test_analyze.py`.

**New restorer**: Add `_fix_<name>()` method to `RestorationPipeline` in `restore.py`, register in `_restorers` dict in `__init__`, write tests in `tests/test_restore.py`.

**New CLI command**: Add to `cli.py` with Typer decorators.

## Project Rules

- **No em dashes or en dashes** anywhere in code, comments, docs, or strings. Use regular hyphens (-) only.
- Keep core dependencies minimal: Pillow, numpy, typer, rich. Optional features go behind extras (`[neural]`, `[web]`, `[video]`).
- Line length: 100 characters.
- CI runs on Ubuntu/Windows/macOS with Python 3.10, 3.11, 3.12.
