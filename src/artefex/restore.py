"""Restoration pipeline - applies targeted fixes for each detected degradation."""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from artefex.models import AnalysisResult, Degradation


class RestorationPipeline:
    """Applies a chain of restorations based on detected degradations.

    When neural models are available, they are used for higher quality restoration.
    Falls back to classical signal processing methods otherwise.
    """

    def __init__(self, use_neural: bool = True):
        self._use_neural = use_neural
        self._neural_engine = None

        self._restorers = {
            "JPEG Compression": self._fix_jpeg_artifacts,
            "Multiple Re-compressions": self._fix_jpeg_artifacts,
            "Noise": self._fix_noise,
            "Color Shift": self._fix_color_shift,
            "Screenshot Artifacts": self._fix_screenshot_borders,
            "Resolution Loss / Upscaling": self._fix_resolution,
        }

        # Map degradation categories to neural model keys
        self._neural_models = {
            "compression": "deblock-v1",
            "noise": "denoise-v1",
            "resolution": "sharpen-v1",
            "color": "color-correct-v1",
        }

    @property
    def neural_engine(self):
        if self._neural_engine is None and self._use_neural:
            try:
                from artefex.neural import NeuralEngine
                engine = NeuralEngine()
                if engine.available:
                    self._neural_engine = engine
            except Exception:
                pass
        return self._neural_engine

    def restore(
        self,
        file_path: Path,
        analysis: AnalysisResult,
        output_path: Path,
        format: Optional[str] = None,
    ) -> dict:
        """Restore an image and return a summary of what was done.

        Args:
            file_path: Path to the degraded image.
            analysis: Analysis results from DegradationAnalyzer.
            output_path: Where to save the restored image.
            format: Optional output format override (e.g. "PNG", "JPEG").

        Returns:
            Dict with keys: steps (list of step descriptions), used_neural (bool).
        """
        img = Image.open(file_path).convert("RGB")

        # Apply fixes in reverse severity order (least severe first, heavy fixes last)
        ordered = sorted(analysis.degradations, key=lambda d: d.severity)

        steps = []
        used_neural = False

        # Use adaptive confidence threshold based on overall severity.
        # Heavily degraded images (D/F grade) should be cleaned more
        # aggressively since there's more to gain and less to lose.
        overall = analysis.overall_severity
        conf_threshold = 0.3 if overall >= 0.5 else 0.5
        sev_threshold = 0.15 if overall >= 0.5 else 0.3

        for degradation in ordered:
            if degradation.confidence < conf_threshold:
                continue
            if degradation.severity < sev_threshold:
                continue

            # Skip non-restorable categories (metadata, provenance)
            if degradation.name in (
                "EXIF Metadata Stripped",
                "Platform Processing",
                "AI-Generated Content",
                "Steganography Detected",
                "Copy-Move Forgery",
                "Device Identification",
            ):
                continue

            # Physical damage detection is informational only.
            # Inpainting is available but requires user-provided
            # or manually verified masks to avoid face distortion.
            if degradation.name == "Physical Damage":
                continue

            # Try neural model for degradations where it measurably
            # outperforms classical methods. Each model has a minimum
            # severity below which classical is better or neutral.
            neural_min_severity = {
                "compression": 0.15,  # FBCNN: +2.7-4.3 dB across all QF levels
                "noise": 0.3,         # DnCNN denoise: +10-20 dB
                "resolution": 0.5,    # NAFNet: +0.6-1.2 dB on moderate blur
            }
            min_sev = neural_min_severity.get(
                degradation.category, 0.7
            )
            if degradation.severity >= min_sev and self._try_neural(
                degradation
            ):
                model_key = self._neural_models.get(degradation.category)
                img = self.neural_engine.run(model_key, img)
                steps.append(
                    f"[neural] {degradation.name} -> {model_key}"
                )
                used_neural = True
                continue

            # Try plugin restorer
            try:
                from artefex.plugins import get_plugin_registry
                plugin_result = get_plugin_registry().run_restorer(img, degradation)
                if plugin_result is not None:
                    img = plugin_result
                    steps.append(f"[plugin] {degradation.name}")
                    continue
            except Exception:
                pass

            # Fall back to classical methods
            restorer = self._restorers.get(degradation.name)
            if restorer:
                img = restorer(img, degradation)
                steps.append(f"[classical] {degradation.name}")

        # Determine output format
        save_kwargs = {"quality": 95}
        if format:
            fmt = format.upper()
            if fmt == "PNG":
                save_kwargs = {}
            output_path = output_path.with_suffix(f".{fmt.lower()}")
        elif output_path.suffix.lower() == ".png":
            save_kwargs = {}

        img.save(output_path, **save_kwargs)

        return {"steps": steps, "used_neural": used_neural, "output_path": str(output_path)}

    def _try_neural(self, degradation: Degradation) -> bool:
        """Check if we can use a neural model for this degradation."""
        if not self._use_neural or self.neural_engine is None:
            return False
        model_key = self._neural_models.get(degradation.category)
        if model_key is None:
            return False
        return self.neural_engine.has_model_for(degradation.category)

    def _fix_jpeg_artifacts(self, img: Image.Image, degradation: Degradation) -> Image.Image:
        """Reduce JPEG block artifacts with adaptive smoothing at block boundaries."""
        arr = np.array(img, dtype=np.float64)
        h, w, c = arr.shape
        result = arr.copy()

        # Measure actual block boundary discontinuity before applying
        block_diff = 0.0
        count = 0
        for y in range(8, min(h - 1, 200), 8):
            row_diff = np.mean(np.abs(arr[y] - arr[y - 1]))
            neighbor_diff = np.mean(np.abs(arr[y - 1] - arr[y - 2]))
            if neighbor_diff > 0:
                block_diff += row_diff / (neighbor_diff + 1e-10)
                count += 1
        avg_ratio = block_diff / count if count > 0 else 1.0

        # Only deblock if block boundaries are measurably worse than
        # neighboring rows (ratio > 1.2 means 20% more discontinuity)
        if avg_ratio < 1.2:
            return img

        strength = min(degradation.severity, 1.0) * 0.2

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

    def _fix_noise(self, img: Image.Image, degradation: Degradation) -> Image.Image:
        """Adaptive edge-preserving denoising."""
        arr = np.array(img, dtype=np.float64)

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

    def _fix_color_shift(self, img: Image.Image, degradation: Degradation) -> Image.Image:
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

    def _fix_screenshot_borders(self, img: Image.Image, degradation: Degradation) -> Image.Image:
        """Crop solid-color borders from screenshots."""
        arr = np.array(img)
        h, w = arr.shape[:2]

        top, bottom, left, right = 0, h, 0, w

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

    def _fix_resolution(self, img: Image.Image, degradation: Degradation) -> Image.Image:
        """Sharpen to partially recover lost high-frequency detail."""
        strength = 0.3 + degradation.severity * 0.7
        radius = 2
        return img.filter(
            ImageFilter.UnsharpMask(
                radius=radius, percent=int(strength * 100), threshold=3
            )
        )
