# GPU Model Training Roadmap

This document outlines the plan for training production-quality neural models to replace the
lightweight test models shipped with Artefex v1.0.

## Current State

Artefex v1.0 ships with **test ONNX models** - small Conv-ReLU-Conv networks with random weights.
These models exercise the full neural pipeline (loading, tiling, padding, inference, output) but
do not produce meaningful restoration improvements over the classical methods.

The classical restoration pipeline (deblocking, denoising, color correction, sharpening) works
well for moderate degradation. GPU-trained models will provide superior results for heavy
degradation.

## Training Infrastructure (Already Built)

| Component | File | Status |
|-----------|------|--------|
| Data generator | `train/generate_data.py` | Ready |
| Deblock trainer | `train/deblock_train.py` | Ready (U-Net, 1ch, L1 loss) |
| Denoise trainer | `train/denoise_train.py` | Ready (U-Net, 3ch, hybrid L1+MSE) |
| Test model generator | `train/create_test_models.py` | Ready |
| Model registry | `src/artefex/models_registry.py` | Ready (4 model slots) |
| Neural engine | `src/artefex/neural.py` | Ready (tiling, padding, CUDA) |
| Download system | `src/artefex/models_registry.py` | Ready (SHA-256 verification) |

## Models to Train

### Phase 1: Core Models (v1.1)

#### 1. deblock-v1 - JPEG Artifact Removal
- **Architecture**: U-Net (64-128-256-512 channels), 1-channel (grayscale)
- **Training data**: Clean images degraded with JPEG quality 10-70
- **Loss**: L1 (already configured in `deblock_train.py`)
- **Target**: PSNR improvement of 2-4 dB over classical deblocking
- **Estimated training**: 4-8 hours on RTX 3060 or equivalent

#### 2. denoise-v1 - Adaptive Denoising
- **Architecture**: U-Net (48-96-192-384 channels), 3-channel (RGB)
- **Training data**: Clean images with Gaussian noise (sigma 10-50)
- **Loss**: 0.7 * L1 + 0.3 * MSE (already configured in `denoise_train.py`)
- **Target**: PSNR improvement of 3-5 dB over classical median filter
- **Estimated training**: 4-8 hours on RTX 3060 or equivalent

#### 3. sharpen-v1 - Detail Recovery
- **Architecture**: U-Net, 3-channel (RGB)
- **Training data**: Clean images degraded with Gaussian blur + downscale/upscale
- **Loss**: L1 + perceptual loss (VGG feature matching)
- **Needs**: New training script based on `denoise_train.py` template
- **Estimated training**: 6-10 hours

#### 4. color-correct-v1 - Color Correction
- **Architecture**: U-Net, 3-channel (RGB)
- **Training data**: Clean images with random channel shifts, white balance errors
- **Loss**: L1 + color histogram loss
- **Needs**: New training script based on `denoise_train.py` template
- **Estimated training**: 4-6 hours

### Phase 2: Extended Models (v1.3)

#### 5. Super-resolution (2x/4x upscaling)
- **Architecture**: ESRGAN or SwinIR variant
- **Training data**: DIV2K/Flickr2K dataset pairs
- **Estimated training**: 24-48 hours on A100 or equivalent

#### 6. Inpainting (watermark/object removal)
- **Architecture**: Partial convolution U-Net
- **Training data**: Clean images with synthetic masks
- **Estimated training**: 12-24 hours

#### 7. Dehazing/defogging
- **Architecture**: FFA-Net variant
- **Training data**: RESIDE dataset or synthetic haze
- **Estimated training**: 8-12 hours

## Quick Start (One Command)

On your PC (RTX 3060 12GB, Ryzen 9 5900XT, 64GB RAM), the entire process is automated:

```bash
# 1. Install PyTorch with CUDA (one-time setup)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Get training images (any folder of 500+ high-quality photos works)
#    Option A: Use your own photos
#    Option B: Download DIV2K dataset (~3.3 GB)

# 3. Run everything - generates data, trains all 4 models, validates, imports
python train/train_all.py --source /path/to/clean/images --epochs 100

# That's it. Expected time: 12-16 hours on RTX 3060.
# You can leave it running overnight.
```

## What Happens During Training

The `train_all.py` script runs 5 stages automatically:

1. **Generate training pairs** (~30 min) - creates degraded/clean image pairs
   for each model type (JPEG compression, noise, blur, color shift)
2. **Train deblock-v1** (~3-4 hrs) - JPEG artifact removal (grayscale U-Net)
3. **Train denoise-v1** (~3-4 hrs) - noise reduction (RGB U-Net)
4. **Train sharpen-v1** (~3-4 hrs) - detail recovery (RGB U-Net)
5. **Train color-correct-v1** (~2-3 hrs) - color correction (RGB U-Net)

After each model trains, it is validated (must improve PSNR by >1 dB) and
imported into the artefex registry.

## Manual Training (Step by Step)

If you prefer to train one model at a time:

### Step 1: Gather Training Data
```bash
# Generate degraded/clean training pairs for all model types
python train/generate_data.py --source /path/to/photos --output ./training_data --type deblock
python train/generate_data.py --source /path/to/photos --output ./training_data --type denoise
python train/generate_data.py --source /path/to/photos --output ./training_data --type sharpen
python train/generate_data.py --source /path/to/photos --output ./training_data --type color
```

### Step 2: Train Models
```bash
# Train each model individually
python train/deblock_train.py --data ./training_data/deblock --epochs 100 --output ./models
python train/denoise_train.py --data ./training_data/denoise --epochs 100 --output ./models
python train/sharpen_train.py --data ./training_data/sharpen --epochs 100 --output ./models
python train/color_train.py --data ./training_data/color --epochs 100 --output ./models
```

### Step 3: Import Models
```bash
artefex models import deblock-v1 ./models/deblock_v1.onnx
artefex models import denoise-v1 ./models/denoise_v1.onnx
artefex models import sharpen-v1 ./models/sharpen_v1.onnx
artefex models import color-correct-v1 ./models/color_correct_v1.onnx
```

### Step 4: Validate
```bash
# Run the automated validation tests
pytest tests/test_model_validation.py -v

# Manual check
artefex restore degraded_photo.jpg restored.png
artefex compare degraded_photo.jpg restored.png
```

## Hardware Requirements

| Option | GPU | Training Time (all 4 models) | Cost |
|--------|-----|------------------------------|------|
| **Your PC (RTX 3060 12GB)** | **RTX 3060** | **12-16 hours** | **Electricity** |
| Google Colab (free) | T4 | 24-32 hours | Free |
| Google Colab Pro | A100 | 8-12 hours | ~$10 |
| Local RTX 4090 | RTX 4090 | 4-8 hours | Electricity only |

Your RTX 3060 12GB is ideal - 12GB VRAM handles batch size 8 with room to spare.
You can increase to `--batch-size 16` for faster training.

## Acceptance Criteria

A trained model is ready for release when:

1. PSNR improvement over classical method is measurable (>1 dB)
2. No visual artifacts introduced on clean images
3. Model file size is under 50 MB
4. Inference time is under 500ms for a 256x256 patch on CPU
5. All 246 existing tests pass with the new model
6. The 10 model validation tests in `test_model_validation.py` pass
7. Manual visual inspection on 10+ diverse test images looks good

## After Training

Once models are trained and validated:

1. Run `pytest tests/test_model_validation.py -v` to confirm all 10 tests pass
2. Run `pytest tests/ -v` to confirm all 246 tests still pass
3. Commit the updated SHA-256 checksums in `models_registry.py`
4. Create a GitHub Release with the ONNX model files attached
5. Update the download URLs in `models_registry.py` for `artefex models download`
6. Users who install artefex will get neural restoration automatically
