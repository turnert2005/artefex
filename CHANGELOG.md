# Changelog

All notable changes to Artefex will be documented in this file.

## [0.1.0] - 2026-03-28

### Added
- Initial release
- 13 degradation/provenance detectors: JPEG compression, re-compression, resolution loss, color shift, screenshot artifacts, noise, watermarks, EXIF stripping, platform fingerprinting (7 platforms), AI-generated content, steganography, camera/device identification, copy-move forgery
- 31 CLI commands for analysis, restoration, grading, comparison, and automation
- Hybrid restoration pipeline (neural ONNX + classical + plugins)
- Public Python API: `import artefex; result = artefex.analyze("photo.jpg")`
- Web UI with drag-and-drop and before/after slider
- Video support (frame-by-frame analysis and restoration)
- GIF/APNG frame-by-frame analysis
- Plugin system via Python entry points
- Training pipeline (data generation + U-Net trainers + ONNX export)
- HTML reports with embedded images and RGB histograms
- Batch HTML dashboard
- Spatial degradation heatmaps
- A-F quality grading with CSV/markdown export
- Quality gate for CI/CD with pre-commit hook
- Color accessibility checker with CVD simulation
- Color palette extraction
- Image orientation detection and correction
- Duplicate/similarity detection (pHash, aHash, dHash)
- Platform fingerprinting (Twitter, Instagram, WhatsApp, Facebook, Telegram, Discord, Imgur)
- Parallel batch processing
- Directory watch mode with auto-restore
- URL analysis support
- stdin/pipe support
- Config file system (.artefex.toml)
- Docker and docker-compose support
- GitHub Actions CI/CD + PyPI publish workflow
- 47 tests across 6 test suites
