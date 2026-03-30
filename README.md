<div align="center">

# artefex

**Forensic image analysis - detect AI content, trace image history, assess quality, and clean artifacts.**

[![CI](https://github.com/turnert2005/artefex/actions/workflows/ci.yml/badge.svg)](https://github.com/turnert2005/artefex/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.0.0-purple.svg)](https://github.com/turnert2005/artefex/releases/tag/v1.0.0)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[Getting Started](#install) | [Commands](#commands) | [Contributing](CONTRIBUTING.md) | [Discussions](https://github.com/turnert2005/artefex/discussions)

</div>

---

Every image on the internet has been through hell: screenshotted, re-compressed, platform-resized, color-shifted, watermarked, and re-shared dozens of times. Existing tools blindly upscale or denoise. Artefex is different - it first **diagnoses** what happened to your media, then **reverses each step specifically**.

Think of it as `git log` for media degradation, followed by intelligent undo.

## Why Artefex?

| | Other tools | Artefex |
|---|---|---|
| **Approach** | Blindly upscale/denoise everything | Diagnose first, then reverse each degradation step |
| **Analysis** | None | 13 forensic detectors - JPEG artifacts, platform fingerprinting, AI detection, steganography, forgery |
| **AI Detection** | None | SAFE neural classifier - 98.9% accuracy on modern generators (GPT-4o, FLUX, SD-3, Midjourney) |
| **Restoration** | One-size-fits-all filter | Neural denoising (+13-21 dB), neural deblurring (+0.6-1.2 dB), classical JPEG/color correction |
| **Extensibility** | Closed | Plugin system for custom detectors and restorers |
| **Interface** | Usually GUI-only | CLI + Python API + Web UI + Docker |

## Install

```bash
pip install artefex               # core (images only)
pip install artefex[web]          # adds web UI
pip install artefex[video]        # adds video support
pip install artefex[neural]       # adds ONNX neural models
pip install artefex[all]          # everything
```

Or install from source:

```bash
git clone https://github.com/turnert2005/artefex.git
cd artefex
pip install -e ".[all]"
```

Or with Docker:

```bash
docker compose up             # web UI at http://localhost:8787
```

## Neural Models

Artefex ships with pre-trained neural models for enhanced restoration and AI detection. After installing, download the models:

```bash
# Download and install all neural models (~76 MB total)
python train/convert_pretrained.py --install
```

Or download individually from the [releases page](https://github.com/turnert2005/artefex/releases) and import:

```bash
artefex models import denoise-v1 denoise_v1.onnx
artefex models import sharpen-v1 sharpen_v1.onnx
artefex models import aigen-detect-v1 aigen_detect_v1.onnx
artefex models list                              # verify installation
```

| Model | Task | Size | Performance | License |
|-------|------|------|-------------|---------|
| DnCNN color blind | Noise removal | 2.6 MB | +13-21 dB PSNR improvement | MIT (KAIR) |
| NAFNet GoPro-w32 | Blur/detail recovery | 65.7 MB | +0.6-1.2 dB on moderate blur | MIT (megvii) |
| SAFE | AI image detection | 5.5 MB | 98.9% accuracy on modern generators | Apache 2.0 |
| DnCNN-3 | JPEG deblocking | 2.5 MB | Marginal (classical used instead) | MIT (KAIR) |

Artefex works without neural models - classical signal processing handles all restoration. Neural models enhance quality for noise and blur, and enable AI detection.

## Quick start

```bash
# Diagnose what happened to an image
artefex analyze photo.jpg

# Get a quality grade (A-F)
artefex grade photo.jpg

# Reverse the degradation chain
artefex restore photo.jpg

# Full forensic audit
artefex audit photo.jpg
```

## Commands

### Analysis

```bash
artefex analyze photo.jpg                     # diagnose degradation chain
artefex analyze photo.jpg --json              # machine-readable output
artefex analyze photo.jpg --verbose           # detailed detection info
artefex analyze https://example.com/img.jpg   # analyze from URL
artefex analyze ./photos/                     # batch mode
```

### Quality grading

```bash
artefex grade photo.jpg                       # A-F grade with score
artefex grade ./photos/ --export csv          # batch export as CSV
artefex grade ./photos/ --export markdown     # batch export as markdown
```

### Forensic tools

```bash
artefex report photo.jpg                      # text forensic report
artefex report photo.jpg --html               # rich HTML report with charts
artefex timeline photo.jpg                    # ASCII degradation timeline
artefex story photo.jpg                       # natural language forensic narrative
artefex heatmap photo.jpg                     # spatial degradation heatmap
artefex palette photo.jpg                     # extract dominant color palette
artefex orient photo.jpg --fix                # detect and fix orientation
artefex audit photo.jpg                       # comprehensive audit (all tools)
```

### Restoration

```bash
artefex restore photo.jpg                     # reverse the degradation chain
artefex restore photo.jpg --format png        # convert output format
artefex restore photo.jpg --no-neural         # classical methods only
artefex restore ./photos/                     # batch restore
artefex restore-preview photo.jpg             # save each step as separate file
```

### Comparison

```bash
artefex compare original.jpg restored.jpg     # MSE, PSNR, SSIM, heatmap
artefex gallery ./originals/ ./restored/      # HTML side-by-side gallery
artefex duplicates ./photos/                  # find duplicate images
artefex duplicates ./photos/ --threshold 0.8  # adjust similarity threshold
```

### Video

```bash
artefex video-analyze clip.mp4                # sample frames for degradation
artefex video-restore clip.mp4                # restore frame by frame
```

### Web and automation

```bash
artefex web                                   # launch web UI with drag-and-drop
artefex watch ./inbox/ --restore              # auto-process new images
artefex dashboard ./photos/                   # generate HTML overview dashboard
artefex rename-by-grade ./photos/ --dry-run   # preview grade-based renaming
artefex parallel-analyze ./photos/            # multi-process batch analysis
```

### System

```bash
artefex version                               # show version and dependency status
artefex models list                           # show available neural models
artefex models import deblock-v1 model.onnx   # import a model
artefex plugins                               # list installed plugins
```

## What it detects

| Category | Detector | Method |
|---|---|---|
| Compression | JPEG artifacts | 8x8 block boundary discontinuity analysis |
| Compression | Multiple re-compressions | Double quantization + ringing detection |
| Resolution | Upscaling/loss | High-frequency spectral analysis + autocorrelation |
| Color | Color shift | Channel imbalance + clip ratio analysis |
| Artifacts | Screenshot remnants | Border uniformity + aspect ratio + dimensions |
| Noise | Sensor/added noise | Laplacian MAD estimation |
| Overlay | Watermarks | Tile correlation + histogram peaks + alpha channel |
| Metadata | EXIF stripping | Metadata presence/completeness checks |
| Provenance | Platform fingerprint | Dimension/compression/EXIF signatures for Twitter, Instagram, WhatsApp, Facebook, Telegram, Discord, Imgur |
| Provenance | AI-generated content | SAFE neural classifier (98.9% accuracy) with heuristic fallback (frequency, histogram, noise, patch analysis) |
| Security | Steganography | LSB analysis, chi-square test, entropy, pairs analysis |
| Provenance | Camera/device ID | Sensor noise PRNU analysis (DSLR, smartphone, webcam, scanner) |
| Forgery | Copy-move detection | Patch-based feature matching for cloned regions |

## Python API

```python
from artefex import analyze, restore, grade

# Diagnose
result = analyze("photo.jpg")
for d in result.degradations:
    print(f"{d.name}: {d.confidence:.0%} confidence, severity {d.severity:.0%}")

# Grade
grade_result = grade("photo.jpg")
print(f"Grade: {grade_result}")

# Restore
restore("photo.jpg", output="photo_restored.png")
```

## Configuration

Create `.artefex.toml` in your project or `~/.artefex.toml` globally:

```toml
[analysis]
min_confidence = 0.15

[restore]
use_neural = true
output_format = "png"

[web]
port = 8787
```

Also supports `[tool.artefex]` in `pyproject.toml`.

## Training custom models

```bash
cd train/
python generate_data.py --source /path/to/clean --output ./data --type deblock
python deblock_train.py --data ./data --epochs 50
artefex models import deblock-v1 ./models/deblock_v1.onnx
```

## Plugins

Artefex supports community plugins via Python entry points:

```toml
# In your plugin's pyproject.toml
[project.entry-points."artefex.detectors"]
my_detector = "my_package:MyDetector"

[project.entry-points."artefex.restorers"]
my_restorer = "my_package:MyRestorer"
```

See `examples/custom_plugin.py` for a complete example.

## Architecture

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

## Roadmap

- [x] **v0.1** - Detection engine (8 detectors) + classical restoration pipeline
- [x] **v0.2** - Neural ONNX models, web UI, video/GIF support, training pipeline, plugin system
- [x] **v0.3** - Platform fingerprinting (7 platforms), AI-generated detection, steganography, camera ID, copy-move forgery, A-F grading, accessibility checker, color palette extraction, orientation correction, duplicate detection, quality gate for CI/CD, batch dashboard, HTML reports with histograms, forensic narrative generation
- [x] **v0.4** - Test ONNX model generation, model download infrastructure, SHA-256 verification
- [x] **v0.5** - Video temporal coherence, audio passthrough via ffmpeg, multi-codec output
- [x] **v0.8** - 244 tests (unit + integration + edge cases + E2E), API stability with TypedDicts
- [x] **v1.0** - Pre-trained neural models (DnCNN denoiser, NAFNet deblurring, SAFE AI detection), stable API **(current)**
- [ ] **v1.1** - FBCNN for JPEG deblocking (288 MB, Apache 2.0 - replaces DnCNN-3 for dramatic JPEG improvement)
- [ ] **v1.2** - Interactive web UI with WebSocket progress and batch management
- [ ] **v1.3** - Expanded neural model zoo (super-resolution, inpainting, dehazing)
- [ ] **v1.4** - Multi-class AI detection (real vs AI-generated vs AI-modified vs AI-upscaled)

## Contributing

We welcome contributions of all sizes - from typo fixes to new detectors. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

**New here?** Look for issues labeled [`good first issue`](https://github.com/turnert2005/artefex/labels/good%20first%20issue) - these are scoped tasks designed for first-time contributors.

**Have questions?** Join the [Discussions](https://github.com/turnert2005/artefex/discussions).

## License

MIT
