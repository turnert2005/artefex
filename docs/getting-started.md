---
title: Getting Started
---

# Getting Started

## Installation

### pip (recommended)

```bash
pip install artefex               # core (images only)
pip install artefex[web]          # adds web UI
pip install artefex[video]        # adds video support
pip install artefex[neural]       # adds ONNX neural models
pip install artefex[all]          # everything
```

### From source

```bash
git clone https://github.com/turnert2005/artefex.git
cd artefex
pip install -e ".[all]"
```

### Docker

```bash
docker compose up             # web UI at http://localhost:8787
```

## First analysis

```bash
artefex analyze photo.jpg
```

This runs all 13 detectors on your image and prints a degradation chain sorted by severity.

## Quality grading

```bash
artefex grade photo.jpg
```

Outputs an A-F grade based on detected degradation severity.

## Restoration

```bash
artefex restore photo.jpg
```

Reverses detected degradations in order. Uses neural models (ONNX) when available, falls back to classical signal processing.

## Configuration

Create `.artefex.toml` in your project or `~/.artefex.toml` globally:

```toml
[analysis]
min_confidence = 0.15

[restore]
use_neural = true
output_format = "png"

[web]
port = 8787
```

Also supports `[tool.artefex]` in `pyproject.toml`.

## Web UI

```bash
artefex web
```

Opens a drag-and-drop web interface at `http://localhost:8787` with before/after comparison slider.
