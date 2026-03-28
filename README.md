# artefex

**Neural forensic restoration - diagnose and reverse media degradation chains.**

Every image on the internet has been through hell: screenshotted, re-compressed, platform-resized, color-shifted, watermarked, and re-shared dozens of times. Existing tools blindly upscale or denoise. Artefex is different - it first **diagnoses** what happened to your image, then **reverses each step specifically**.

Think of it as `git log` for image degradation, followed by intelligent undo.

## Install

```bash
pip install -e .
```

## Usage

### Analyze a single image
```bash
artefex analyze photo.jpg
artefex analyze photo.jpg --verbose
```

### Batch analyze a directory
```bash
artefex analyze ./photos/
```

Output:
```
Batch analyzing 12 images in: ./photos/

         Batch Analysis Summary
+----------------+---------------+----------------+-----------+
| File           | Degradations  | Worst Severity | Top Issue |
+----------------+---------------+----------------+-----------+
| vacation.jpg   | 3             | 72%            | Multiple  |
| profile.png    | 1             | 23%            | Noise     |
| meme.jpg       | 4             | 89%            | JPEG      |
| clean.png      | 0             | 0%             | Clean     |
+----------------+---------------+----------------+-----------+
```

### Generate forensic reports
```bash
artefex report photo.jpg
artefex report photo.jpg --output report.txt
artefex report ./photos/                        # batch mode
```

### Restore images
```bash
artefex restore photo.jpg
artefex restore photo.jpg --output cleaned.png
artefex restore ./photos/                       # batch mode
```

### Compare before and after
```bash
artefex compare original.jpg restored.jpg
```

Outputs MSE, PSNR, per-channel diffs, and generates a difference heatmap.

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

## Roadmap

- [x] **v0.1** - Detection engine + basic restoration
- [ ] **v0.2** - Neural super-resolution models for detail recovery
- [ ] **v0.3** - Watermark removal via inpainting
- [ ] **v0.4** - Video support (frame-by-frame + temporal coherence)
- [ ] **v0.5** - Web UI for drag-and-drop analysis
- [ ] **v1.0** - Pluggable model system + community model registry

## How it works

1. **Analyze** - Each detector examines the image for specific degradation signatures using signal processing techniques (DCT analysis, spectral analysis, statistical methods)
2. **Diagnose** - Results are ordered into an estimated degradation chain
3. **Restore** - Targeted restoration is applied in reverse order, with each fix tuned to the specific degradation's severity

## License

MIT
