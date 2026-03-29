---
title: Detectors
---

# Detectors

Artefex includes 13 built-in detectors. Each runs independently and returns a `Degradation` object with confidence and severity scores.

## Built-in detectors

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
| Provenance | Camera/device ID | Sensor noise PRNU analysis (DSLR, smartphone, webcam, scanner) |
| Forgery | Copy-move detection | Patch-based feature matching for cloned regions |

## Adding a custom detector

### As a method in the core

1. Add `_detect_<name>(self, img, arr, result) -> Degradation | None` to `DegradationAnalyzer` in `src/artefex/analyze.py`
2. Add the detector to the `detectors` list in the `analyze()` method
3. Add a recommendation in `src/artefex/report.py`
4. Write tests in `tests/test_analyze.py`

### As a plugin

Create a Python package with an entry point:

```toml
# In your plugin's pyproject.toml
[project.entry-points."artefex.detectors"]
my_detector = "my_package:MyDetector"
```

See `examples/custom_plugin.py` for a complete example.
