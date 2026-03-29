"""Degradation detection engine - the core of artefex."""

from pathlib import Path

import numpy as np
from PIL import Image

from artefex.models import AnalysisResult, Degradation


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
            self._detect_watermark,
            self._detect_exif_stripping,
        ]

        for detector in detectors:
            degradation = detector(img, arr, result)
            if degradation:
                result.degradations.append(degradation)

        # Run AI-generated detection
        try:
            from artefex.detect_aigen import AIGeneratedDetector
            ai_det = AIGeneratedDetector()
            ai_deg = ai_det.detect(img, arr)
            if ai_deg:
                result.degradations.append(ai_deg)
        except Exception:
            pass

        # Run steganography detection
        try:
            from artefex.detect_stego import SteganographyDetector
            stego_det = SteganographyDetector()
            stego_deg = stego_det.detect(img, arr)
            if stego_deg:
                result.degradations.append(stego_deg)
        except Exception:
            pass

        # Run camera/device identification
        try:
            from artefex.detect_camera import detect_camera
            camera_deg = detect_camera(img, arr)
            if camera_deg:
                result.degradations.append(camera_deg)
        except Exception:
            pass

        # Run copy-move forgery detection
        try:
            from artefex.detect_forgery import CopyMoveDetector
            forgery_det = CopyMoveDetector()
            forgery_deg = forgery_det.detect(img, arr)
            if forgery_deg:
                result.degradations.append(forgery_deg)
        except Exception:
            pass

        # Run platform fingerprinting
        try:
            from artefex.fingerprint import detect_platform
            platform_deg = detect_platform(file_path)
            if platform_deg:
                result.degradations.append(platform_deg)
        except Exception:
            pass

        # Run plugin detectors
        try:
            from artefex.plugins import get_plugin_registry
            plugin_results = get_plugin_registry().run_detectors(img, arr)
            result.degradations.extend(plugin_results)
        except Exception:
            pass  # Plugins are optional

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

        # Also check high-frequency content - upscaled images lack it
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
            detail=f"Channel means - R:{r_mean:.1f} G:{g_mean:.1f} B:{b_mean:.1f}. "
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
        _ = sobel_v > np.percentile(sobel_v, 90)

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

    def _detect_watermark(
        self, img: Image.Image, arr: np.ndarray, result: AnalysisResult
    ) -> Degradation | None:
        """Detect watermark patterns by looking for semi-transparent overlays and repeating structures."""
        if len(arr.shape) < 3 or arr.shape[2] < 3:
            return None

        h, w = arr.shape[:2]
        if h < 64 or w < 64:
            return None

        gray = np.mean(arr[:, :, :3].astype(np.float64), axis=2)

        indicators = 0
        details = []

        # Check for low-contrast repeating patterns (tiled watermarks)
        # Compute autocorrelation at large offsets to detect tiling
        quarter_h, quarter_w = h // 4, w // 4
        if quarter_h > 16 and quarter_w > 16:
            center_block = gray[quarter_h : 2 * quarter_h, quarter_w : 2 * quarter_w]
            right_block = gray[quarter_h : 2 * quarter_h, 2 * quarter_w : 3 * quarter_w]
            below_block = gray[2 * quarter_h : 3 * quarter_h, quarter_w : 2 * quarter_w]

            min_h_r = min(center_block.shape[0], right_block.shape[0])
            min_w_r = min(center_block.shape[1], right_block.shape[1])
            min_h_b = min(center_block.shape[0], below_block.shape[0])
            min_w_b = min(center_block.shape[1], below_block.shape[1])

            if min_h_r > 8 and min_w_r > 8:
                cb = center_block[:min_h_r, :min_w_r]
                rb = right_block[:min_h_r, :min_w_r]
                corr_h = np.corrcoef(cb.flatten(), rb.flatten())[0, 1]
                if corr_h > 0.95:
                    indicators += 1
                    details.append(f"horizontal tile correlation {corr_h:.3f}")

            if min_h_b > 8 and min_w_b > 8:
                cb2 = center_block[:min_h_b, :min_w_b]
                bb = below_block[:min_h_b, :min_w_b]
                corr_v = np.corrcoef(cb2.flatten(), bb.flatten())[0, 1]
                if corr_v > 0.95:
                    indicators += 1
                    details.append(f"vertical tile correlation {corr_v:.3f}")

        # Check for semi-transparent overlay: look for a narrow band in the
        # luminance histogram that shouldn't be there (watermark text creates
        # a secondary peak near the bright end)
        hist, bin_edges = np.histogram(gray, bins=256, range=(0, 256))
        hist_norm = hist / hist.sum()

        # Look for suspicious secondary peaks in the upper luminance range
        upper_hist = hist_norm[200:]
        if len(upper_hist) > 5:
            upper_mean = upper_hist.mean()
            upper_peaks = np.sum(upper_hist > upper_mean * 3)
            if upper_peaks >= 2:
                indicators += 1
                details.append(f"suspicious bright peaks in histogram ({upper_peaks} peaks)")

        # Check for unnaturally uniform regions that could be watermark overlay
        # Sample the center of the image where watermarks are commonly placed
        center_region = gray[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
        local_stds = []
        block_size = 16
        for by in range(0, center_region.shape[0] - block_size, block_size):
            for bx in range(0, center_region.shape[1] - block_size, block_size):
                block = center_region[by : by + block_size, bx : bx + block_size]
                local_stds.append(block.std())

        if local_stds:
            local_stds = np.array(local_stds)
            # Watermarks create blocks with unusually low variance amidst normal content
            low_var_ratio = np.sum(local_stds < 3.0) / len(local_stds)
            high_var_ratio = np.sum(local_stds > 20.0) / len(local_stds)

            if low_var_ratio > 0.15 and high_var_ratio > 0.15:
                indicators += 1
                details.append(
                    f"mixed variance in center region "
                    f"(low={low_var_ratio:.0%}, high={high_var_ratio:.0%})"
                )

        # Check alpha channel for watermark mask
        if arr.shape[2] == 4:
            alpha = arr[:, :, 3]
            unique_alpha = np.unique(alpha)
            # Watermarks often have specific alpha values that aren't 0 or 255
            mid_alpha = unique_alpha[(unique_alpha > 10) & (unique_alpha < 245)]
            if len(mid_alpha) > 0 and len(mid_alpha) < 20:
                indicators += 2
                details.append(f"alpha channel has {len(mid_alpha)} semi-transparent levels")

        if indicators < 2:
            return None

        severity = min(1.0, indicators * 0.2)
        confidence = min(1.0, indicators * 0.25)

        return Degradation(
            name="Watermark",
            confidence=confidence,
            severity=severity,
            detail="; ".join(details),
            category="overlay",
        )

    def _detect_exif_stripping(
        self, img: Image.Image, arr: np.ndarray, result: AnalysisResult
    ) -> Degradation | None:
        """Detect if EXIF metadata has been stripped, suggesting the image was re-processed."""
        is_jpeg = img.format == "JPEG" or str(result.file_path).lower().endswith((".jpg", ".jpeg"))
        if not is_jpeg:
            return None

        indicators = 0
        details = []

        # Check for EXIF data
        exif_data = None
        try:
            exif_data = img.getexif()
        except Exception:
            pass

        has_exif = exif_data is not None and len(exif_data) > 0

        if not has_exif:
            indicators += 2
            details.append("no EXIF metadata found in JPEG")

            # A high-resolution JPEG with zero EXIF is suspicious -
            # cameras and phones always embed EXIF
            if img.size[0] >= 1000 or img.size[1] >= 1000:
                indicators += 1
                details.append(
                    f"high resolution ({img.size[0]}x{img.size[1]}) but no camera metadata"
                )
        else:
            # Check for partial EXIF (some fields stripped)
            # Common camera tags: Make(271), Model(272), DateTime(306), ExifOffset(34665)
            important_tags = {271: "Make", 272: "Model", 306: "DateTime", 34665: "ExifOffset"}
            missing = [name for tag, name in important_tags.items() if tag not in exif_data]

            if len(missing) >= 3:
                indicators += 1
                details.append(f"partial EXIF - missing: {', '.join(missing)}")

        # Check for JFIF marker without EXIF (common after re-saving)
        if "jfif_version" in result.metadata and not has_exif:
            indicators += 1
            details.append("has JFIF header but no EXIF (re-saved by software)")

        if indicators < 2:
            return None

        severity = min(1.0, indicators * 0.2)
        confidence = min(1.0, indicators * 0.3)

        return Degradation(
            name="EXIF Metadata Stripped",
            confidence=confidence,
            severity=severity,
            detail="; ".join(details),
            category="metadata",
        )
