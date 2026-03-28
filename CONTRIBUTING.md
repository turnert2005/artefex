# Contributing to Artefex

Thanks for your interest in contributing to Artefex.

## Development Setup

```bash
git clone https://github.com/turnert2005/artefex.git
cd artefex
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting. Run before submitting:

```bash
ruff check src/ tests/
```

Line length is 100 characters.

## Project Rules

- **No em dashes or en dashes** anywhere in code, comments, docs, or strings. Use regular hyphens (-) instead.
- Keep dependencies minimal. Core functionality should only need Pillow, numpy, typer, and rich.
- Optional features (web, video, neural) go behind optional dependency groups.

## Adding a Detector

1. Add your detection method to `src/artefex/analyze.py` in the `DegradationAnalyzer` class
2. Follow the pattern: `_detect_<name>(self, img, arr, result) -> Degradation | None`
3. Add the detector to the `detectors` list in the `analyze()` method
4. Add a corresponding recommendation in `src/artefex/report.py`
5. Write tests in `tests/test_analyze.py`

Or build it as a plugin - see `examples/custom_plugin.py`.

## Adding a Restorer

1. Add your restoration method to `src/artefex/restore.py` in the `RestorationPipeline` class
2. Follow the pattern: `_fix_<name>(self, img, degradation) -> Image`
3. Register it in the `_restorers` dict in `__init__`
4. Write tests in `tests/test_restore.py`

## Submitting Changes

1. Fork the repo
2. Create a branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Run lint (`ruff check src/ tests/`)
6. Submit a pull request

## Architecture Overview

```
src/artefex/
  analyze.py        - Degradation detection engine (8 built-in detectors)
  restore.py        - Restoration pipeline (neural + classical + plugin)
  cli.py            - Typer CLI with all commands
  models.py         - Data models (Degradation, AnalysisResult)
  models_registry.py - ONNX model management
  neural.py         - Neural inference engine (ONNX Runtime)
  plugins.py        - Entry-point plugin system
  report.py         - Plain text report generator
  report_html.py    - HTML report with embedded images and charts
  video.py          - Video analysis and restoration
  watch.py          - Directory watcher for auto-processing
  web.py            - FastAPI web UI
```
