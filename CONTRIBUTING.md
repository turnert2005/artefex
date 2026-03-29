# Contributing to Artefex

Thanks for your interest in contributing to Artefex! Whether it's a bug fix, new detector, plugin, documentation improvement, or test - every contribution matters.

## Getting started

```bash
git clone https://github.com/turnert2005/artefex.git
cd artefex
pip install -e ".[dev]"
pytest tests/ -v          # verify everything works
```

## Finding something to work on

- **[Good first issues](https://github.com/turnert2005/artefex/labels/good%20first%20issue)** - scoped tasks with mentorship available, great for your first PR
- **[Help wanted](https://github.com/turnert2005/artefex/labels/help%20wanted)** - tasks where we'd especially appreciate community help
- **[Feature requests](https://github.com/turnert2005/artefex/labels/enhancement)** - ideas from the community waiting for someone to pick them up
- **Your own idea** - open a [Discussion](https://github.com/turnert2005/artefex/discussions) or [Feature Request](https://github.com/turnert2005/artefex/issues/new?template=feature_request.md) first so we can align on scope

## Running tests and lint

```bash
pytest tests/ -v                                      # all tests
pytest tests/test_analyze.py -v                       # single file
pytest tests/test_analyze.py::test_jpeg_detection -v  # single test
ruff check src/ tests/                                # lint
```

## Code style

We use [ruff](https://docs.astral.sh/ruff/) for linting. Line length is 100 characters.

## Project rules

- **No em dashes or en dashes** anywhere in code, comments, docs, or strings. Use regular hyphens (-) instead.
- Keep dependencies minimal. Core functionality should only need Pillow, numpy, typer, and rich.
- Optional features (web, video, neural) go behind optional dependency groups.

## Adding a detector

1. Add your detection method to `src/artefex/analyze.py` in the `DegradationAnalyzer` class
2. Follow the pattern: `_detect_<name>(self, img, arr, result) -> Degradation | None`
3. Add the detector to the `detectors` list in the `analyze()` method
4. Add a corresponding recommendation in `src/artefex/report.py`
5. Write tests in `tests/test_analyze.py`

Or build it as a plugin - see `examples/custom_plugin.py`.

## Adding a restorer

1. Add your restoration method to `src/artefex/restore.py` in the `RestorationPipeline` class
2. Follow the pattern: `_fix_<name>(self, img, degradation) -> Image`
3. Register it in the `_restorers` dict in `__init__`
4. Write tests in `tests/test_restore.py`

## Submitting changes

1. Fork the repo
2. Create a branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Run lint (`ruff check src/ tests/`)
6. Submit a pull request

## Ways to contribute

Beyond code, we value contributions in:

- **Documentation** - improve README, add examples, clarify confusing sections
- **Bug reports** - well-described issues with reproduction steps save everyone time
- **Testing** - add edge cases, improve coverage, test on different platforms
- **Plugins** - build and share community plugins (see `examples/custom_plugin.py`)
- **Training data** - help create or curate datasets for the model training pipeline
- **Spreading the word** - blog posts, talks, or just starring the repo helps others find Artefex

## Architecture overview

```
src/artefex/
  analyze.py        - Degradation detection engine (13 built-in detectors)
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
