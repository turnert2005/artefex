"""Spatial degradation heatmap - visualize WHERE in the image degradation is worst."""

from pathlib import Path

import numpy as np
from PIL import Image


def generate_heatmap(file_path: Path, output_path: Path, patch_size: int = 32) -> dict:
    """Generate a degradation heatmap showing spatial quality variation.

    Analyzes the image in patches and produces a color-coded overlay where:
    - Green = healthy areas (low degradation)
    - Yellow = moderate degradation
    - Red = heavy degradation

    Returns dict with stats about the spatial distribution.
    """
    img = Image.open(file_path).convert("RGB")
    arr = np.array(img, dtype=np.float64)
    h, w = arr.shape[:2]

    gray = np.mean(arr[:, :, :3], axis=2)

    # Compute per-patch quality scores
    rows = (h + patch_size - 1) // patch_size
    cols = (w + patch_size - 1) // patch_size
    scores = np.zeros((rows, cols))

    for py in range(rows):
        for px in range(cols):
            y0, y1 = py * patch_size, min((py + 1) * patch_size, h)
            x0, x1 = px * patch_size, min((px + 1) * patch_size, w)
            patch = gray[y0:y1, x0:x1]

            score = _compute_patch_score(patch)
            scores[py, px] = score

    # Normalize scores to 0-1
    if scores.max() > scores.min():
        normalized = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        normalized = np.zeros_like(scores)

    # Generate heatmap overlay
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)

    for py in range(rows):
        for px in range(cols):
            y0, y1 = py * patch_size, min((py + 1) * patch_size, h)
            x0, x1 = px * patch_size, min((px + 1) * patch_size, w)

            val = normalized[py, px]
            r, g, b = _severity_color(val)
            heatmap[y0:y1, x0:x1] = [r, g, b]

    # Blend with original image
    heatmap_float = heatmap.astype(np.float64)
    original_float = arr[:, :, :3]
    blended = original_float * 0.5 + heatmap_float * 0.5
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    output_img = Image.fromarray(blended)
    output_img.save(output_path)

    # Compute stats
    worst_idx = np.unravel_index(np.argmax(normalized), normalized.shape)
    worst_y = worst_idx[0] * patch_size + patch_size // 2
    worst_x = worst_idx[1] * patch_size + patch_size // 2

    return {
        "output_path": str(output_path),
        "patch_size": patch_size,
        "grid_size": (rows, cols),
        "mean_score": float(normalized.mean()),
        "worst_region": (worst_x, worst_y),
        "healthy_pct": float(np.sum(normalized < 0.3) / normalized.size),
        "moderate_pct": float(np.sum((normalized >= 0.3) & (normalized < 0.7)) / normalized.size),
        "severe_pct": float(np.sum(normalized >= 0.7) / normalized.size),
    }


def _compute_patch_score(patch: np.ndarray) -> float:
    """Compute degradation score for a single patch. Higher = more degraded."""
    if patch.size < 16:
        return 0.0

    # Blockiness: discontinuity at 8-pixel boundaries
    blockiness = 0.0
    h, w = patch.shape
    if h >= 16 and w >= 16:
        for y in range(8, h - 1, 8):
            boundary = np.mean(np.abs(patch[y, :] - patch[y - 1, :]))
            interior = np.mean(np.abs(patch[y + 1, :] - patch[y, :]))
            if interior > 0:
                blockiness = max(blockiness, boundary / interior - 1.0)

    # Noise level
    if h >= 4 and w >= 4:
        lap = (
            patch[:-2, 1:-1] + patch[2:, 1:-1]
            + patch[1:-1, :-2] + patch[1:-1, 2:]
            - 4 * patch[1:-1, 1:-1]
        )
        noise = np.median(np.abs(lap)) * 1.4826
    else:
        noise = 0.0

    # Combine
    score = min(1.0, blockiness * 2 + noise / 30.0)
    return score


def _severity_color(val: float) -> tuple[int, int, int]:
    """Map 0-1 severity to green-yellow-red color."""
    if val < 0.5:
        # Green to yellow
        t = val * 2
        r = int(255 * t)
        g = 255
        b = 0
    else:
        # Yellow to red
        t = (val - 0.5) * 2
        r = 255
        g = int(255 * (1 - t))
        b = 0

    return (r, g, b)
