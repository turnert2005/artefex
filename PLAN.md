# Artefex Development Plan

## Current State (v1.0.0)

Working forensic image analysis tool with:
- 14 forensic detectors (including physical damage detection)
- Neural models: FBCNN (+3-4 dB JPEG), DnCNN (+13-21 dB noise), NAFNet (+0.6-1.2 dB blur), SAFE (98.9% AI detection)
- LaMa inpainting model downloaded but disabled (face distortion risk)
- Guided web UI (wizard flow, no scrolling)
- Windows packaging (PyInstaller + Inno Setup)
- 244 tests passing

## Priority 1: First-Launch Model Downloader

**Goal**: Installer stays small (~30 MB), models download on first launch.

### Tasks
- [ ] Add download URLs to model registry (GitHub Releases or HuggingFace)
- [ ] Create `src/artefex/model_downloader.py` with progress callbacks
- [ ] Update `launcher.py` to check for models on startup
- [ ] Add setup screen to web UI showing download progress
- [ ] Handle offline/failed downloads gracefully (fall back to classical)
- [ ] Upload ONNX models to GitHub Releases as v1.0.0 assets
- [ ] Rebuild Windows package without bundled models
- [ ] Test full flow: fresh install -> first launch -> auto download -> analyze

## Priority 2: Web UI Progress Indicators

**Goal**: Users see real-time feedback during analysis and cleaning.

### Tasks
- [ ] Add WebSocket or SSE endpoint for progress updates
- [ ] Show spinner with step names during analysis ("Running JPEG detector...", "Running AI detector...")
- [ ] Show progress bar during cleaning with time estimate
- [ ] Show model download progress during first launch
- [ ] Add loading animation that feels responsive (not just a spinner)

## Priority 3: Safe Inpainting (v1.1)

**Goal**: Repair physical damage in old photos without distorting faces.

### Tasks
- [ ] Integrate a face detection model (or use classical Haar cascades) to create face protection mask
- [ ] Combine face mask with damage mask: never inpaint inside detected face regions
- [ ] Process at higher resolution (tile-based, not resize to 512x512)
- [ ] Add user-adjustable sensitivity slider for damage detection
- [ ] Add manual mask painting in web UI (user draws damage regions)
- [ ] Test with diverse old photographs (scratches, tears, water damage, yellowing)
- [ ] Validate inpainting quality with before/after PSNR and visual inspection
- [ ] Add "Repair Damage" as a separate button from "Clean Image"

## Model Architecture

| Model | Source | License | Size | Performance |
|-------|--------|---------|------|-------------|
| FBCNN | jiaxi-jiang/FBCNN | Apache 2.0 | 274 MB | +2.7 to +4.3 dB JPEG |
| DnCNN color blind | cszn/KAIR | MIT | 2.6 MB | +13 to +21 dB noise |
| NAFNet GoPro-w32 | megvii-research/NAFNet | MIT | 65.7 MB | +0.6 to +1.2 dB blur |
| SAFE | Ouxiang-Li/SAFE | Apache 2.0 | 5.5 MB | 98.9% AI detection |
| LaMa | opencv/inpainting_lama | Apache 2.0 | 88 MB | Inpainting (disabled) |

## Future Roadmap

- v1.1: Safe inpainting with face protection
- v1.2: Interactive web UI with WebSocket progress
- v1.3: Expanded model zoo (super-resolution, dehazing)
- v1.4: Multi-class AI detection (real/generated/modified/upscaled)
