"""Restoration pipeline - applies targeted fixes for each detected degradation."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from artefex.models import AnalysisResult


class RestorationPipeline:
    """Applies a chain of restorations based on detected degradations."""

    def __init__(self):
        self._restorers = {
            "JPEG Compression": self._fix_jpeg_artifacts,
            "Multiple Re-compressions": self._fix_jpeg_artifacts,
            "Noise": self._fix_noise,
            "Color Shift": self._fix_color_shift,
            "Screenshot Artifacts": self._fix_screenshot_borders,
            "Resolution Loss / Upscaling": self._fix_resolution,
        }

    def restore(self, file_path: Path, analysis: AnalysisResult, output_path: Path) -> None:
        img = Image.open(file_path).convert("RGB")

        # Apply fixes in reverse severity order (least severe first, so heavy fixes go last)
        ordered = sorted(analysis.degradations, key=lambda d: d.severity)

        for degradation in ordered:
            restorer = self._restorers.get(degradation.name)
            if restorer:
                img = restorer(img, degradation)

        img.save(output_path, quality=95)

    def _fix_jpeg_artifacts(self, img: Image.Image, degradation) -> Image.Image:
        """Reduce JPEG block artifacts with adaptive smoothing at block boundaries."""
        arr = np.array(img, dtype=np.float64)
        h, w, c = arr.shape
        result = arr.copy()

        strength = degradation.severity * 0.6

        # Smooth specifically at 8x8 block boundaries
        for y in range(8, h - 1, 8):
            blend = strength * 0.5
            result[y, :, :] = (1 - blend) * arr[y, :, :] + blend * (
                arr[y - 1, :, :] * 0.5 + arr[y + 1, :, :] * 0.5
            )

        for x in range(8, w - 1, 8):
            blend = strength * 0.5
            result[:, x, :] = (1 - blend) * result[:, x, :] + blend * (
                result[:, x - 1, :] * 0.5 + result[:, x + 1, :] * 0.5
            )

        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def _fix_noise(self, img: Image.Image, degradation) -> Image.Image:
        """Adaptive edge-preserving denoising."""
        arr = np.array(img, dtype=np.float64)

        # Use bilateral-like filtering: smooth flat areas, preserve edges
        # Simple approximation: blend between original and median-filtered
        from PIL import ImageFilter

        radius = max(1, int(degradation.severity * 3))
        smoothed = img.filter(ImageFilter.MedianFilter(size=radius * 2 + 1))

        smoothed_arr = np.array(smoothed, dtype=np.float64)

        # Edge detection to create blend mask
        gray = np.mean(arr[:, :, :3], axis=2)
        edge_h = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        edge_v = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        edges = np.sqrt(edge_h**2 + edge_v**2)
        edge_mask = np.clip(edges / (np.percentile(edges, 85) + 1e-10), 0, 1)

        # Blend: smooth in flat areas, keep original at edges
        blend = (1 - edge_mask)[:, :, np.newaxis] * degradation.severity * 0.7
        result = arr * (1 - blend) + smoothed_arr * blend
        result = np.clip(result, 0, 255).astype(np.uint8)

        return Image.fromarray(result)

    def _fix_color_shift(self, img: Image.Image, degradation) -> Image.Image:
        """Normalize color channels toward balance."""
        arr = np.array(img, dtype=np.float64)

        means = arr.mean(axis=(0, 1))
        overall = means.mean()

        strength = degradation.severity * 0.5

        for ch in range(3):
            if means[ch] > 0:
                correction = overall / means[ch]
                arr[:, :, ch] *= 1 + (correction - 1) * strength

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _fix_screenshot_borders(self, img: Image.Image, degradation) -> Image.Image:
        """Crop solid-color borders from screenshots."""
        arr = np.array(img)
        h, w = arr.shape[:2]

        top, bottom, left, right = 0, h, 0, w

        # Find where content starts/ends
        for y in range(min(h // 4, 50)):
            if arr[y, :, :3].astype(np.float64).std() > 5:
                top = y
                break

        for y in range(h - 1, max(h - h // 4, h - 50), -1):
            if arr[y, :, :3].astype(np.float64).std() > 5:
                bottom = y + 1
                break

        for x in range(min(w // 4, 50)):
            if arr[:, x, :3].astype(np.float64).std() > 5:
                left = x
                break

        for x in range(w - 1, max(w - w // 4, w - 50), -1):
            if arr[:, x, :3].astype(np.float64).std() > 5:
                right = x + 1
                break

        if top > 0 or bottom < h or left > 0 or right < w:
            return img.crop((left, top, right, bottom))

        return img

    def _fix_resolution(self, img: Image.Image, degradation) -> Image.Image:
        """Sharpen to partially recover lost high-frequency detail."""
        # For v0.1, apply unsharp mask. Neural super-res comes in v0.2.
        strength = 0.5 + degradation.severity * 1.5
        radius = 2
        return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(strength * 100), threshold=2))
