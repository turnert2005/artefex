# Changelog

All notable changes to Artefex will be documented in this file.

## [1.0.0] - 2026-03-29

### Added
- Pre-trained neural models integrated from research community:
  - DnCNN color blind denoiser (MIT, 2.6 MB) - +13 to +21 dB on Gaussian noise
  - NAFNet GoPro-width32 deblurring (MIT, 65.7 MB) - +0.6 to +1.2 dB on moderate blur
  - SAFE AI-generated image detector (Apache 2.0, 5.5 MB) - 98.9% accuracy on modern generators (GPT-4o, FLUX, SD-3, Midjourney)
  - DnCNN-3 JPEG deblocking (MIT, 2.5 MB) - installed but classical used by default
- Model conversion script (`train/convert_pretrained.py`) for PyTorch to ONNX
- `ModelInfo.is_trained` property to distinguish real models from test stubs
- Smart severity-gated neural restoration - only applies neural when it outperforms classical
- "Do no harm" guarantee - restoration never degrades image quality by more than 0.05 dB
- Neural AI detection in `detect_aigen.py` with automatic fallback to heuristics
- Model download infrastructure with SHA-256 verification and progress callbacks
- Video temporal coherence - frame blending to prevent flicker
- Video audio passthrough via ffmpeg extraction and remuxing
- Multi-codec video output support (mp4v, avc1, XVID)
- TypedDict return types for all public API functions
- 244 tests across 10 test suites (up from 47), 83% code coverage
- Training scripts for all model types plus automation (`train/train_all.py`)
- Model validation tests that verify PSNR improvement

### Changed
- Version bumped from 0.3.0 to 1.0.0 - stable public API
- CI workflow split lint into separate job, added fail-fast: false for test matrix
- All public API functions now return typed dicts instead of bare dict
- Restoration pipeline skips low-confidence detections and non-restorable categories
- Classical JPEG deblocking now measures actual block boundary discontinuity before applying

## [0.3.0] - 2026-03-29

### Added
- Platform fingerprinting for 7 platforms (Twitter, Instagram, WhatsApp, Facebook, Telegram, Discord, Imgur)
- AI-generated image detection with 5 forensic heuristics
- LSB steganography detection with 4 statistical tests
- Camera/device identification via PRNU noise patterns (6 device profiles)
- Copy-move forgery detection using patch matching
- A-F quality grading system with CSV/markdown export
- Color accessibility checker with CVD simulation (protanopia, deuteranopia, tritanopia, achromatopsia)
- Color palette extraction via k-means clustering
- Image orientation detection and auto-correction
- Duplicate/similarity detection (pHash, aHash, dHash)
- Quality gate for CI/CD pipelines with pre-commit hook support
- Batch HTML dashboard with summary statistics
- Forensic narrative story generation
- Before/after comparison gallery with PSNR metrics

### Fixed
- Resolved 39 ruff lint errors (unused imports, f-string issues, undefined references)
- Version numbers now consistent across all project files

## [0.2.0] - 2026-03-28

### Added
- Neural ONNX inference engine with tiling support for large images
- Model registry with 4 registered models (deblock, denoise, sharpen, color-correct)
- FastAPI web UI with drag-and-drop and before/after slider
- Video support with frame-by-frame analysis and restoration
- GIF/APNG frame-by-frame analysis with temporal metrics
- Training pipeline (data generation + U-Net trainers + ONNX export)
- Plugin system via Python entry points (detectors and restorers)

## [0.1.0] - 2026-03-28

### Added
- Initial release
- 8 core degradation detectors: JPEG compression, re-compression, resolution loss, color shift, screenshot artifacts, noise, watermarks, EXIF stripping
- Classical restoration pipeline (deblocking, noise reduction, color correction, border cropping, sharpening)
- 31 CLI commands for analysis, restoration, grading, comparison, and automation
- Public Python API: `import artefex; result = artefex.analyze("photo.jpg")`
- HTML reports with embedded images and RGB histograms
- Spatial degradation heatmaps
- Parallel batch processing
- Directory watch mode with auto-restore
- URL analysis and stdin/pipe support
- Config file system (.artefex.toml)
- Docker and docker-compose support
- GitHub Actions CI/CD + PyPI publish workflow
- 47 tests across 6 test suites
