"""Degradation detection engine — the core of neural-enhance."""

from pathlib import Path

import numpy as np
from PIL import Image

from neural_enhance.models import AnalysisResult, Degradation


class DegradationAnalyzer:
    """Analyzes images to detect and characterize degradation chains."""

    def analyze(self, file_path: Path) -> AnalysisResult:
        img = Image.open(file_path)
        arr = np.array(img)

        result = AnalysisResult(
            file_path=str(file_path),
            file_format=img.format or "UNKNOWN",
            dimensions=img.size,
            metadata=self._extract_metadata(img),
        )

        detectors = [
            self._detect_jpeg_compression,
            self._detect_resolution_loss,
            self._detect_color_shift,
            self._detect_screenshot_artifacts,
            self._detect_multiple_compressions,
            self._detect_noise,
        ]

        for detector in detectors:
            degradation = detector(img, arr, result)
            if degradation:
                result.degradations.append(degradation)

        # Sort by estimated occurrence order (severity as proxy for how early it happened)
        result.degradations.sort(key=lambda d: d.severity, reverse=True)

        return result

    def _extract_metadata(self, img: Image.Image) -> dict:
        meta = {
            "mode": img.mode,
            "format": img.format,
        }
        if hasattr(img, "info"):
            if "jfif_version" in img.info:
                meta["jfif_version"] = img.info["jfif_version"]
            if "dpi" in img.info:
                meta["dpi"] = img.info["dpi"]
            if "quality" in img.info:
                meta["quality"] = img.info["quality"]
        return meta

    def _detect_jpeg_compression(
        self, img: Image.Image, arr: np.ndarray, result: AnalysisResult
    ) -> Degradation | None:
        """Detect JPEG compression artifacts by analyzing 8x8 block boundaries."""
        if img.format != "JPEG" and not str(result.file_path).lower().endswith((".jpg", ".jpeg")):
            return None

        if len(arr.shape) < 3:
            gray = arr.astype(np.float64)
        else:
            gray = np.mean(arr[:, :, :3].astype(np.float64), axis=2)

        h, w = gray.shape
        if h < 16 or w < 16:
            return None

        # Measure discontinuity at 8x8 block boundaries vs interior
        boundary_diffs = []
        interior_diffs = []

        for y in range(8, h - 8, 8):
            row_boundary = np.abs(gray[y, :] - gray[y - 1, :])
            row_interior = np.abs(gray[y + 1, :] - gray[y, :])
            boundary_diffs.extend(row_boundary)
            interior_diffs.extend(row_interior)

        for x in range(8, w - 8, 8):
            col_boundary = np.abs(gray[:, x] - gray[:, x - 1])
            col_interior = np.abs(gray[:, x + 1] - gray[:, x])
            boundary_diffs.extend(col_boundary)
            interior_diffs.extend(col_interior)

        mean_boundary = np.mean(boundary_diffs)
        mean_interior = np.mean(interior_diffs)

        if mean_interior == 0:
            return None

        blockiness_ratio = mean_boundary / mean_interior

        if blockiness_ratio < 1.02:
            return None

        severity = min(1.0, (blockiness_ratio - 1.0) / 0.5)
        confidence = min(1.0, (blockiness_ratio - 1.0) / 0.3)

        return Degradation(
            name="JPEG Compression",
            confidence=confidence,
            severity=severity,
            detail=f"Block boundary discontinuity ratio: {blockiness_ratio:.3f}. "
            f"Mean boundary diff: {mean_boundary:.2f}, interior: {mean_interior:.2f}",
            category="compression",
        )

    def _detect_resolution_loss(
        self, img: Image.Image, arr: np.ndarray, result: AnalysisResult
    ) -> Degradation | None:
        """Detect if image was upscaled from a lower resolution by checking for interpolation patterns."""
        if len(arr.shape) < 3:
            gray = arr.astype(np.float64)
        else:
            gray = np.mean(arr[:, :, :3].astype(np.float64), axis=2)

        h, w = gray.shape
        if h < 32 or w < 32:
            return None

        # Check for repeating pixel patterns that indicate upscaling
        # Compare autocorrelation at small offsets
        center = gray[16 : h - 16, 16 : w - 16]

        correlations = []
        for offset in range(2, 6):
            shifted_h = gray[16 + offset : h - 16 + offset, 16 : w - 16]
            shifted_v = gray[16 : h - 16, 16 + offset : w - 16 + offset]

            min_h = min(center.shape[0], shifted_h.shape[0])
            min_w = min(center.shape[1], shifted_h.shape[1])

            if min_h < 8 or min_w < 8:
                continue

            c = center[:min_h, :min_w]
            sh = shifted_h[:min_h, :min_w]

            min_w2 = min(center.shape[1], shifted_v.shape[1])
            min_h2 = min(center.shape[0], shifted_v.shape[0])
            sv = shifted_v[:min_h2, :min_w2]
            c2 = center[:min_h2, :min_w2]

            corr_h = np.corrcoef(c.flatten(), sh.flatten())[0, 1]
            corr_v = np.corrcoef(c2.flatten(), sv.flatten())[0, 1]
            correlations.append((offset, max(corr_h, corr_v)))

        # High correlation at specific offsets suggests upscaling
        if not correlations:
            return None

        max_corr_offset, max_corr = max(correlations, key=lambda x: x[1])

        # Also check high-frequency content — upscaled images lack it
        from numpy.fft import fft2, fftshift

        spectrum = np.abs(fftshift(fft2(gray)))
        h_s, w_s = spectrum.shape
        center_region = spectrum[h_s // 4 : 3 * h_s // 4, w_s // 4 : 3 * w_s // 4]
        edge_region_sum = spectrum.sum() - center_region.sum()
        total = spectrum.sum()

        high_freq_ratio = edge_region_sum / total if total > 0 else 0

        if high_freq_ratio > 0.3:
            return None  # Plenty of high-frequency content

        severity = min(1.0, (1.0 - high_freq_ratio) * 0.8)
        confidence = min(1.0, (0.3 - high_freq_ratio) / 0.2) * 0.7

        if confidence < 0.2:
            return None

        return Degradation(
            name="Resolution Loss / Upscaling",
            confidence=confidence,
            severity=severity,
            detail=f"High-frequency content ratio: {high_freq_ratio:.3f} (low = likely upscaled). "
            f"Peak autocorrelation at offset {max_corr_offset}: {max_corr:.3f}",
            category="resolution",
        )

    def _detect_color_shift(
        self, img: Image.Image, arr: np.ndarray, result: AnalysisResult
    ) -> Degradation | None:
        """Detect color channel imbalance suggesting color space conversion artifacts."""
        if len(arr.shape) < 3 or arr.shape[2] < 3:
            return None

        r_mean = arr[:, :, 0].astype(np.float64).mean()
        g_mean = arr[:, :, 1].astype(np.float64).mean()
        b_mean = arr[:, :, 2].astype(np.float64).mean()

        overall_mean = (r_mean + g_mean + b_mean) / 3

        if overall_mean < 5 or overall_mean > 250:
            return None  # Too dark or too bright to judge

        r_dev = abs(r_mean - overall_mean) / overall_mean
        g_dev = abs(g_mean - overall_mean) / overall_mean
        b_dev = abs(b_mean - overall_mean) / overall_mean

        max_dev = max(r_dev, g_dev, b_dev)

        # Also check for clipped channels (sign of bad color space conversion)
        r_clipped = (np.sum(arr[:, :, 0] == 0) + np.sum(arr[:, :, 0] == 255)) / arr[:, :, 0].size
        g_clipped = (np.sum(arr[:, :, 1] == 0) + np.sum(arr[:, :, 1] == 255)) / arr[:, :, 1].size
        b_clipped = (np.sum(arr[:, :, 2] == 0) + np.sum(arr[:, :, 2] == 255)) / arr[:, :, 2].size

        clip_imbalance = max(r_clipped, g_clipped, b_clipped) - min(r_clipped, g_clipped, b_clipped)

        if max_dev < 0.15 and clip_imbalance < 0.05:
            return None

        severity = min(1.0, max_dev * 2 + clip_imbalance * 3)
        confidence = min(1.0, max_dev * 3 + clip_imbalance * 5)

        if confidence < 0.15:
            return None

        dominant = ["R", "G", "B"][np.argmax([r_mean, g_mean, b_mean])]

        return Degradation(
            name="Color Shift",
            confidence=confidence,
            severity=severity,
            detail=f"Channel means — R:{r_mean:.1f} G:{g_mean:.1f} B:{b_mean:.1f}. "
            f"Max deviation: {max_dev:.3f}, dominant: {dominant}. "
            f"Clip imbalance: {clip_imbalance:.3f}",
            category="color",
        )

    def _detect_screenshot_artifacts(
        self, img: Image.Image, arr: np.ndarray, result: AnalysisResult
    ) -> Degradation | None:
        """Detect screenshot indicators: solid-color borders, UI element remnants."""
        if len(arr.shape) < 3:
            return None

        h, w = arr.shape[:2]
        if h < 50 or w < 50:
            return None

        indicators = 0
        details = []

        # Check for solid-color borders (common in screenshots)
        border_size = min(5, h // 20, w // 20)

        top = arr[:border_size, :, :3]
        bottom = arr[-border_size:, :, :3]
        left = arr[:, :border_size, :3]
        right = arr[:, -border_size:, :3]

        for name, border in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
            std = border.astype(np.float64).std()
            if std < 3.0:
                indicators += 1
                details.append(f"solid {name} border (std={std:.1f})")

        # Check for common screenshot aspect ratios
        ratio = w / h
        screenshot_ratios = [16 / 9, 16 / 10, 4 / 3, 21 / 9, 2560 / 1440, 1920 / 1080]
        for sr in screenshot_ratios:
            if abs(ratio - sr) < 0.02:
                indicators += 1
                details.append(f"screen aspect ratio {ratio:.3f}")
                break

        # Check for perfectly even dimensions (screenshots are often exact pixel counts)
        if w % 2 == 0 and h % 2 == 0 and w % 10 == 0 and h % 10 == 0:
            indicators += 1
            details.append(f"round dimensions {w}x{h}")

        if indicators < 2:
            return None

        severity = min(1.0, indicators * 0.25)
        confidence = min(1.0, indicators * 0.3)

        return Degradation(
            name="Screenshot Artifacts",
            confidence=confidence,
            severity=severity,
            detail="; ".join(details),
            category="artifact",
        )

    def _detect_multiple_compressions(
        self, img: Image.Image, arr: np.ndarray, result: AnalysisResult
    ) -> Degradation | None:
        """Detect signs of multiple re-compressions (generation loss)."""
        if img.format != "JPEG" and not str(result.file_path).lower().endswith((".jpg", ".jpeg")):
            return None

        if len(arr.shape) < 3:
            gray = arr.astype(np.float64)
        else:
            gray = np.mean(arr[:, :, :3].astype(np.float64), axis=2)

        h, w = gray.shape
        if h < 24 or w < 24:
            return None

        # Multiple compressions create a characteristic "double quantization" pattern
        # Check histogram of DCT-like block differences for telltale gaps
        block_means = []
        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                block = gray[y : y + 8, x : x + 8]
                block_means.append(block.mean())

        if len(block_means) < 16:
            return None

        block_means = np.array(block_means)
        diffs = np.diff(np.sort(block_means))

        # Multiple compression creates more uniform block means (information loss)
        uniformity = 1.0 - (np.std(diffs) / (np.mean(diffs) + 1e-10))

        # Also check for "ringing" artifacts near edges
        sobel_h = np.abs(np.diff(gray, axis=0))
        sobel_v = np.abs(np.diff(gray, axis=1))

        edge_mask_h = sobel_h > np.percentile(sobel_h, 90)
        edge_mask_v = sobel_v > np.percentile(sobel_v, 90)

        # Near strong edges, check for oscillation (ringing)
        ringing_score = 0.0
        count = 0
        for y in range(2, min(h - 2, sobel_h.shape[0])):
            for x in range(2, min(w - 2, sobel_h.shape[1])):
                if edge_mask_h[min(y, edge_mask_h.shape[0] - 1), min(x, edge_mask_h.shape[1] - 1)]:
                    neighborhood = gray[y - 2 : y + 3, max(0, x - 2) : x + 3]
                    if neighborhood.size >= 4:
                        local_var = neighborhood.var()
                        ringing_score += local_var
                        count += 1
                if count > 5000:
                    break
            if count > 5000:
                break

        avg_ringing = ringing_score / max(count, 1)
        global_var = gray.var()
        ringing_ratio = avg_ringing / (global_var + 1e-10)

        if uniformity < 0.5 and ringing_ratio < 1.5:
            return None

        severity = min(1.0, uniformity * 0.5 + min(ringing_ratio / 5, 0.5))
        confidence = min(1.0, uniformity * 0.4 + min(ringing_ratio / 3, 0.6))

        if confidence < 0.2:
            return None

        return Degradation(
            name="Multiple Re-compressions",
            confidence=confidence,
            severity=severity,
            detail=f"Block uniformity: {uniformity:.3f}, ringing ratio: {ringing_ratio:.3f}. "
            f"Suggests {max(2, int(uniformity * 5))}+ compression cycles",
            category="compression",
        )

    def _detect_noise(
        self, img: Image.Image, arr: np.ndarray, result: AnalysisResult
    ) -> Degradation | None:
        """Detect added noise or sensor noise patterns."""
        if len(arr.shape) < 3:
            gray = arr.astype(np.float64)
        else:
            gray = np.mean(arr[:, :, :3].astype(np.float64), axis=2)

        h, w = gray.shape
        if h < 16 or w < 16:
            return None

        # Estimate noise using median absolute deviation of Laplacian
        laplacian = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4 * gray[1:-1, 1:-1]
        )

        noise_estimate = np.median(np.abs(laplacian)) * 1.4826  # MAD to std conversion

        if noise_estimate < 3.0:
            return None

        severity = min(1.0, noise_estimate / 30.0)
        confidence = min(1.0, noise_estimate / 15.0)

        if confidence < 0.15:
            return None

        return Degradation(
            name="Noise",
            confidence=confidence,
            severity=severity,
            detail=f"Estimated noise level (sigma): {noise_estimate:.2f}",
            category="noise",
        )
