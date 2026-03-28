# neural-enhance

**Neural forensic restoration — diagnose and reverse media degradation chains.**

Every image on the internet has been through hell: screenshotted, re-compressed, platform-resized, color-shifted, watermarked, and re-shared dozens of times. Existing tools blindly upscale or denoise. neural-enhance is different — it first **diagnoses** what happened to your image, then **reverses each step specifically**.

Think of it as `git log` for image degradation, followed by intelligent undo.

## Install

```bash
pip install -e .
```

## Usage

### Analyze an image
```bash
neural-enhance analyze photo.jpg
neural-enhance analyze photo.jpg --verbose
```

Output:
```
Analyzing: photo.jpg

   Degradation Chain (estimated order)
┌───┬──────────────────────────┬────────────┬──────────┐
│ # │ Degradation              │ Confidence │ Severity │
├───┼──────────────────────────┼────────────┼──────────┤
│ 1 │ Multiple Re-compressions │        87% │      72% │
│ 2 │ JPEG Compression         │        93% │      58% │
│ 3 │ Color Shift              │        41% │      23% │
└───┴──────────────────────────┴────────────┴──────────┘
```

### Generate a forensic report
```bash
neural-enhance report photo.jpg
neural-enhance report photo.jpg --output report.txt
```

### Restore an image
```bash
neural-enhance restore photo.jpg
neural-enhance restore photo.jpg --output cleaned.png
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

## Roadmap

- [ ] **v0.1** — Detection engine + basic restoration (current)
- [ ] **v0.2** — Neural super-resolution models for detail recovery
- [ ] **v0.3** — Watermark detection and removal
- [ ] **v0.4** — Video support (frame-by-frame + temporal coherence)
- [ ] **v0.5** — Web UI for drag-and-drop analysis
- [ ] **v1.0** — Pluggable model system + community model registry

## How it works

1. **Analyze** — Each detector examines the image for specific degradation signatures using signal processing techniques (DCT analysis, spectral analysis, statistical methods)
2. **Diagnose** — Results are ordered into an estimated degradation chain
3. **Restore** — Targeted restoration is applied in reverse order, with each fix tuned to the specific degradation's severity

## License

MIT
