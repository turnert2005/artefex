# artefex

**Neural forensic restoration - diagnose and reverse media degradation chains.**

Every image on the internet has been through hell: screenshotted, re-compressed, platform-resized, color-shifted, watermarked, and re-shared dozens of times. Existing tools blindly upscale or denoise. Artefex is different - it first **diagnoses** what happened to your media, then **reverses each step specifically**.

Think of it as `git log` for media degradation, followed by intelligent undo.

## Install

```bash
pip install -e .              # core (images only)
pip install -e ".[web]"       # adds web UI
pip install -e ".[video]"     # adds video support
pip install -e ".[neural]"    # adds ONNX neural models
pip install -e ".[all]"       # everything
```

Or with Docker:

```bash
docker compose up             # web UI at http://localhost:8787
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
artefex heatmap photo.jpg                     # spatial degradation heatmap
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
| Provenance | AI-generated content | Frequency spectrum, histogram smoothness, noise uniformity, patch consistency |
| Security | Steganography | LSB analysis, chi-square test, entropy, pairs analysis |

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
| 11 Built-in Detectors  |  JPEG, noise, color, resolution, screenshot,
| + Plugin Detectors     |  watermark, EXIF, platform, AI-gen, stego
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

- [x] **v0.1** - Detection engine + classical restoration
- [x] **v0.2** - Neural models, web UI, video, training, plugins
- [x] **v0.3** - Platform fingerprinting, AI detection, steganography, grading
- [ ] **v0.4** - Pre-trained model weights + model hub
- [ ] **v0.5** - Temporal coherence for video + audio support
- [ ] **v1.0** - Stable API + community model zoo

## License

MIT
