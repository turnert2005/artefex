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

## Usage

### Analyze images
```bash
artefex analyze photo.jpg                  # single image
artefex analyze photo.jpg --verbose        # with details
artefex analyze photo.jpg --json           # machine-readable output
artefex analyze ./photos/                  # batch mode
```

### Generate forensic reports
```bash
artefex report photo.jpg
artefex report photo.jpg --output report.txt
artefex report ./photos/                   # batch mode
```

### Restore images
```bash
artefex restore photo.jpg
artefex restore photo.jpg --output clean.png
artefex restore photo.jpg --format png     # convert format
artefex restore photo.jpg --no-neural      # classical methods only
artefex restore ./photos/                  # batch mode
```

### Compare before and after
```bash
artefex compare original.jpg restored.jpg
```

Outputs MSE, PSNR, per-channel diffs, and generates a difference heatmap.

### Video analysis and restoration
```bash
artefex video-analyze clip.mp4             # sample frames and diagnose
artefex video-analyze clip.mp4 --samples 20
artefex video-restore clip.mp4             # restore frame by frame
artefex video-restore clip.mp4 --output clean.mp4
```

### Web UI
```bash
artefex web                                # launch at http://127.0.0.1:8787
artefex web --port 9000                    # custom port
```

Drag-and-drop interface for analyzing and restoring images in your browser.

### Model management
```bash
artefex models list                        # show available models
artefex models import deblock-v1 model.onnx
```

## What it detects

| Degradation | Method |
|---|---|
| JPEG compression artifacts | 8x8 block boundary discontinuity analysis |
| Multiple re-compressions | Double quantization pattern + ringing detection |
| Resolution loss / upscaling | High-frequency spectral analysis + autocorrelation |
| Color shift | Channel imbalance + clip ratio analysis |
| Screenshot artifacts | Border uniformity + aspect ratio + dimension analysis |
| Noise | Laplacian MAD estimation |
| Watermark overlays | Tile correlation + histogram peaks + alpha channel analysis |
| EXIF metadata stripping | Metadata presence/completeness checks |

## Training custom models

Artefex includes training scripts for building your own restoration models:

```bash
cd train/

# Generate synthetic training data
python generate_data.py --source /path/to/clean/images --output ./data --type deblock

# Train a deblocking model
python deblock_train.py --data ./data --epochs 50 --output ./models

# Train a denoising model
python denoise_train.py --data ./data --epochs 50 --output ./models

# Import into artefex
artefex models import deblock-v1 ./models/deblock_v1.onnx
```

## Architecture

```
artefex analyze <image>
    |
    v
+-------------------+
| Degradation       |  8 detectors run in parallel
| Analyzer          |  Each returns (name, confidence, severity, detail)
+-------------------+
    |
    v
+-------------------+
| Degradation       |  Sorted by severity, ordered as estimated chain
| Chain             |
+-------------------+
    |
    v
+-------------------+
| Restoration       |  Neural models (ONNX) when available
| Pipeline          |  Classical fallback (signal processing)
+-------------------+
    |
    v
  restored image
```

## Roadmap

- [x] **v0.1** - Detection engine + classical restoration
- [x] **v0.2** - Neural model infrastructure + web UI + video support
- [ ] **v0.3** - Pre-trained model weights + model hub
- [ ] **v0.4** - Temporal coherence for video restoration
- [ ] **v0.5** - Plugin system for community detectors
- [ ] **v1.0** - Stable API + comprehensive model zoo

## License

MIT
