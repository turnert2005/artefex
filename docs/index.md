---
title: Artefex - Neural Forensic Restoration
---

# Artefex

**Neural forensic restoration - diagnose and reverse media degradation chains.**

Every image on the internet has been through hell: screenshotted, re-compressed, platform-resized, color-shifted, watermarked, and re-shared dozens of times. Existing tools blindly upscale or denoise. Artefex is different - it first **diagnoses** what happened to your media, then **reverses each step specifically**.

Think of it as `git log` for media degradation, followed by intelligent undo.

## Why Artefex?

| | Other tools | Artefex |
|---|---|---|
| **Approach** | Blindly upscale/denoise everything | Diagnose first, then reverse each degradation step |
| **Analysis** | None | 13 forensic detectors - JPEG artifacts, platform fingerprinting, AI detection, steganography, forgery |
| **Restoration** | One-size-fits-all filter | Targeted fix per degradation - neural (ONNX) with classical fallback |
| **Extensibility** | Closed | Plugin system for custom detectors and restorers |
| **Interface** | Usually GUI-only | CLI + Python API + Web UI + Docker |

## Quick start

```bash
pip install artefex[all]

# Diagnose what happened to an image
artefex analyze photo.jpg

# Get a quality grade (A-F)
artefex grade photo.jpg

# Reverse the degradation chain
artefex restore photo.jpg

# Full forensic audit
artefex audit photo.jpg
```

## Next steps

- [Getting Started](getting-started.md) - installation options and first steps
- [CLI Reference](cli-reference.md) - all 31 commands
- [Python API](api.md) - programmatic usage
- [Detectors](detectors.md) - what Artefex can detect and how
- [Architecture](architecture.md) - how the system works
- [Contributing](contributing.md) - how to get involved
