# Artefex Training

This directory contains training scripts for Artefex neural models.

## Model Architecture

Artefex uses lightweight CNNs optimized for single-degradation restoration.
Each model targets one type of degradation and is exported to ONNX for inference.

## Available Training Scripts

### deblock_train.py
Trains the JPEG deblocking model. Creates pairs of (compressed, clean) images
and trains a U-Net style network to remove 8x8 block artifacts.

### denoise_train.py
Trains the adaptive denoiser. Uses pairs of (noisy, clean) images with
varying noise levels. The model learns to denoise while preserving edges.

## Quick Start

```bash
# Install training dependencies
pip install torch torchvision

# Generate training data (degrades clean images synthetically)
python generate_data.py --source /path/to/clean/images --output ./data

# Train a model
python deblock_train.py --data ./data/deblock --epochs 50 --output ../models

# Export to ONNX and import into artefex
artefex models import deblock-v1 ../models/deblock_v1.onnx
```

## Data Generation

The `generate_data.py` script creates synthetic training pairs by applying
controlled degradations to clean images. This ensures the model learns
to reverse specific degradation types rather than generic "enhancement."

## Training Your Own Models

1. Collect clean, high-quality source images
2. Run `generate_data.py` to create degraded/clean pairs
3. Train using the provided scripts
4. Export to ONNX format
5. Import via `artefex models import <key> <path>`

Models must accept NCHW float32 tensors normalized to [0, 1] and output
the same format. See model registry for expected input sizes.
