"""Camera/device identification from sensor noise patterns.

Every camera sensor has a unique "fingerprint" - a fixed pattern noise (PRNU)
caused by manufacturing imperfections in the sensor. This module analyzes
noise residuals to estimate what type of device captured an image.

This is used in digital forensics to verify image authenticity and detect
spliced regions from different cameras.
"""

import numpy as np
from PIL import Image, ImageFilter

from artefex.models import Degradation


# Known device classes and their noise characteristics
DEVICE_PROFILES = {
    "smartphone_high": {
        "name": "High-end smartphone",
        "noise_range": (1.5, 4.0),
        "uniformity_range": (0.3, 0.6),
        "color_noise_ratio_range": (0.4, 0.7),
    },
    "smartphone_mid": {
        "name": "Mid-range smartphone",
        "noise_range": (3.0, 7.0),
        "uniformity_range": (0.25, 0.55),
        "color_noise_ratio_range": (0.5, 0.8),
    },
    "dslr": {
        "name": "DSLR / mirrorless camera",
        "noise_range": (0.8, 3.0),
        "uniformity_range": (0.4, 0.7),
        "color_noise_ratio_range": (0.2, 0.5),
    },
    "webcam": {
        "name": "Webcam / low-quality sensor",
        "noise_range": (5.0, 15.0),
        "uniformity_range": (0.15, 0.4),
        "color_noise_ratio_range": (0.6, 0.9),
    },
    "scanner": {
        "name": "Scanner / screen capture",
        "noise_range": (0.5, 2.5),
        "uniformity_range": (0.6, 0.9),
        "color_noise_ratio_range": (0.1, 0.4),
    },
    "ai_generated": {
        "name": "AI-generated (no real sensor)",
        "noise_range": (0.0, 1.5),
        "uniformity_range": (0.7, 1.0),
        "color_noise_ratio_range": (0.0, 0.3),
    },
}


class CameraIdentifier:
    """Identifies the likely camera/device type from sensor noise patterns."""

    def identify(self, img: Image.Image, arr: np.ndarray) -> list[dict]:
        """Analyze sensor noise and return likely device matches.

        Returns list of: [{"device": str, "name": str, "confidence": float, "evidence": list}]
        """
        if len(arr.shape) < 3 or arr.shape[2] < 3:
            return []

        h, w = arr.shape[:2]
        if h < 64 or w < 64:
            return []

        # Extract noise residual
        noise_level = self._estimate_noise_level(arr)
        uniformity = self._noise_uniformity(arr)
        color_ratio = self._color_noise_ratio(arr)

        results = []
        for key, profile in DEVICE_PROFILES.items():
            score = 0.0
            evidence = []

            # Check noise level
            nl, nh = profile["noise_range"]
            if nl <= noise_level <= nh:
                score += 0.35
                evidence.append(f"noise level {noise_level:.1f} matches range [{nl}-{nh}]")
            elif nl - 1 <= noise_level <= nh + 1:
                score += 0.15
                evidence.append(f"noise level {noise_level:.1f} near range [{nl}-{nh}]")

            # Check uniformity
            ul, uh = profile["uniformity_range"]
            if ul <= uniformity <= uh:
                score += 0.35
                evidence.append(f"noise uniformity {uniformity:.2f} matches [{ul}-{uh}]")
            elif ul - 0.1 <= uniformity <= uh + 0.1:
                score += 0.15

            # Check color noise ratio
            cl, ch = profile["color_noise_ratio_range"]
            if cl <= color_ratio <= ch:
                score += 0.3
                evidence.append(f"color noise ratio {color_ratio:.2f} matches [{cl}-{ch}]")
            elif cl - 0.1 <= color_ratio <= ch + 0.1:
                score += 0.1

            if score >= 0.4 and len(evidence) >= 2:
                results.append({
                    "device": key,
                    "name": profile["name"],
                    "confidence": min(1.0, score),
                    "evidence": evidence,
                })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def _estimate_noise_level(self, arr: np.ndarray) -> float:
        """Estimate overall noise level using Laplacian MAD."""
        gray = np.mean(arr[:, :, :3].astype(np.float64), axis=2)

        laplacian = (
            gray[:-2, 1:-1] + gray[2:, 1:-1]
            + gray[1:-1, :-2] + gray[1:-1, 2:]
            - 4 * gray[1:-1, 1:-1]
        )
        return float(np.median(np.abs(laplacian)) * 1.4826)

    def _noise_uniformity(self, arr: np.ndarray) -> float:
        """Measure how uniform noise is across the image (higher = more uniform)."""
        gray = np.mean(arr[:, :, :3].astype(np.float64), axis=2)
        h, w = gray.shape

        patch_size = 64
        noise_levels = []

        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = gray[y:y + patch_size, x:x + patch_size]
                lap = (
                    patch[:-2, 1:-1] + patch[2:, 1:-1]
                    + patch[1:-1, :-2] + patch[1:-1, 2:]
                    - 4 * patch[1:-1, 1:-1]
                )
                noise = np.median(np.abs(lap)) * 1.4826
                noise_levels.append(noise)

        if len(noise_levels) < 2:
            return 0.5

        noise_arr = np.array(noise_levels)
        mean = noise_arr.mean()
        if mean == 0:
            return 1.0

        cv = noise_arr.std() / mean
        return float(max(0, 1.0 - cv))

    def _color_noise_ratio(self, arr: np.ndarray) -> float:
        """Ratio of chromatic noise to luminance noise."""
        float_arr = arr[:, :, :3].astype(np.float64)

        # Luminance noise
        gray = np.mean(float_arr, axis=2)
        lum_lap = (
            gray[:-2, 1:-1] + gray[2:, 1:-1]
            + gray[1:-1, :-2] + gray[1:-1, 2:]
            - 4 * gray[1:-1, 1:-1]
        )
        lum_noise = np.median(np.abs(lum_lap)) * 1.4826

        # Chromatic noise (difference between channels)
        rg_diff = float_arr[:, :, 0] - float_arr[:, :, 1]
        rb_diff = float_arr[:, :, 0] - float_arr[:, :, 2]

        rg_lap = (
            rg_diff[:-2, 1:-1] + rg_diff[2:, 1:-1]
            + rg_diff[1:-1, :-2] + rg_diff[1:-1, 2:]
            - 4 * rg_diff[1:-1, 1:-1]
        )
        chrom_noise = np.median(np.abs(rg_lap)) * 1.4826

        if lum_noise == 0:
            return 0.0

        return float(min(1.0, chrom_noise / lum_noise))


def detect_camera(img: Image.Image, arr: np.ndarray) -> Degradation | None:
    """Run camera identification and return as Degradation if confident."""
    ci = CameraIdentifier()
    matches = ci.identify(img, arr)

    if not matches or matches[0]["confidence"] < 0.5:
        return None

    top = matches[0]
    detail = f"Likely device: {top['name']} ({top['confidence']:.0%})"
    if len(top["evidence"]) > 0:
        detail += ". " + "; ".join(top["evidence"][:2])
    if len(matches) > 1:
        detail += f". Also possible: {matches[1]['name']} ({matches[1]['confidence']:.0%})"

    return Degradation(
        name="Device Identification",
        confidence=top["confidence"],
        severity=0.0,  # Purely informational
        detail=detail,
        category="provenance",
    )
