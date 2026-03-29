# Artefex v1.0 - Achieved

This document tracked the roadmap from v0.3 to v1.0. v1.0 has been released.

## Current State (v1.0.0)

Artefex is a production-ready forensic image analysis and restoration tool:

- 13 degradation/provenance detectors (JPEG, noise, upscaling, color shift, screenshots, watermarks, EXIF, platform fingerprinting, AI detection, steganography, camera ID, copy-move forgery, re-compression)
- Classical restoration pipeline with "do no harm" quality guarantee
- Neural pipeline infrastructure ready (requires GPU-trained models for enhancement - see [GPU Training Roadmap](gpu-training-roadmap.md))
- 31 CLI commands, public Python API, FastAPI web UI
- Video/GIF frame-by-frame support
- Batch processing, directory watching, quality grading
- Color accessibility, palette extraction, orientation correction
- Plugin system, config files, quality gate for CI/CD

## Milestone Plan

### v0.4 - Pre-trained Model Weights + Model Hub

**Goal:** Ship working neural models so users get neural restoration out of the box.

- [ ] Train and validate deblock-v1, denoise-v1, sharpen-v1, color-correct-v1 models
- [ ] Host model weights (GitHub Releases or dedicated CDN)
- [ ] `artefex models download` command to fetch weights on demand
- [ ] Lazy download on first use with progress bar
- [ ] Model integrity verification (SHA-256 checksums)
- [ ] Document model training reproduction steps

### v0.5 - Temporal Coherence + Audio

**Goal:** Make video restoration production-grade with frame-to-frame consistency.

- [ ] Temporal smoothing across restored video frames (prevent flicker)
- [ ] Audio track passthrough during video restoration
- [ ] Multi-format video output (MP4, WebM, MOV)
- [ ] Configurable keyframe-based analysis (skip similar frames)
- [ ] Video-specific degradation detection (encoding artifacts, interlacing, frame drops)

### v0.6 - Interactive Web UI

**Goal:** Make the web interface a first-class experience for non-CLI users.

- [ ] Real-time analysis progress with WebSocket updates
- [ ] Side-by-side comparison tools (slider, diff overlay, toggle)
- [ ] Batch upload and management queue
- [ ] Export reports from web UI
- [ ] Session persistence and history

### v0.7 - Expanded Neural Model Zoo

**Goal:** Broaden the types of restoration Artefex can perform.

- [ ] Super-resolution model (2x/4x upscaling)
- [ ] Inpainting model (watermark/object removal)
- [ ] Dehazing/defogging model
- [ ] JPEG artifact removal at various quality levels
- [ ] Community model submission process and standards

### v0.8 - Testing, Benchmarks, and API Stability

**Goal:** Harden the project for production use.

- [ ] Integration test suite (end-to-end workflows)
- [ ] Performance benchmark suite with regression tracking
- [ ] API surface review and deprecation of unstable interfaces
- [ ] Type annotations across all public APIs
- [ ] Documentation coverage for all public functions
- [ ] Cross-platform CI validation (ARM64 runners)

### v1.0 - Stable Release

**Goal:** Production-ready, stable public API, community ecosystem.

- [ ] Semantic versioning guarantee on public API
- [ ] Comprehensive documentation site with tutorials and examples
- [ ] Community model zoo with at least 10 contributed models
- [ ] PyPI stable release with long-term support commitment
- [ ] Migration guide from alpha versions
- [ ] Security audit of file handling and web UI
- [ ] Performance targets documented and met (e.g., analysis under 2s for 4K images)

## Priorities for Next Steps

The most impactful work to start now:

1. **Model training (v0.4)** - The neural pipeline is built but ships without weights. Training and hosting models unlocks the full restoration capability.
2. **Integration tests (v0.8)** - The 47 unit tests pass, but end-to-end workflow tests would catch regressions and validate the full pipeline.
3. **API stability review (v0.8)** - Lock down the public API surface before more users depend on it.
