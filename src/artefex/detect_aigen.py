"""AI-generated image detection.

Detects statistical signatures that distinguish AI-generated images from
real photographs. Uses multiple heuristics:

1. Frequency spectrum analysis - AI images have characteristic spectral patterns
2. Color distribution analysis - AI images often have unusual color histograms
3. Noise pattern analysis - real photos have sensor noise, AI images have different patterns
4. Local patch consistency - AI images sometimes have inconsistent local statistics
5. Pixel value distribution - AI generators leave subtle statistical fingerprints
"""

import numpy as np
from PIL import Image

from artefex.models import Degradation


class AIGeneratedDetector:
    """Detects whether an image was likely AI-generated.

    When a trained neural model (aigen-detect-v1) is available, uses it
    for high-accuracy classification. Falls back to heuristic analysis
    when no model is present.
    """

    def detect(self, img: Image.Image, arr: np.ndarray) -> Degradation | None:
        if len(arr.shape) < 3 or arr.shape[2] < 3:
            return None

        h, w = arr.shape[:2]
        if h < 64 or w < 64:
            return None

        # Try neural model first if available
        neural_result = self._try_neural_detection(img)
        if neural_result is not None:
            return neural_result

        indicators = 0
        total_weight = 0
        details = []

        # 1. Frequency spectrum analysis
        # AI-generated images often lack high-frequency detail in specific patterns
        freq_score = self._check_frequency_spectrum(arr)
        if freq_score > 0.5:
            indicators += freq_score
            total_weight += 1
            details.append(f"spectral anomaly score {freq_score:.2f}")

        # 2. Color histogram smoothness
        # AI images tend to have smoother color distributions than real photos
        hist_score = self._check_histogram_smoothness(arr)
        if hist_score > 0.5:
            indicators += hist_score
            total_weight += 1
            details.append(f"histogram smoothness {hist_score:.2f}")

        # 3. Noise pattern uniformity
        # Real photos have spatially-varying noise, AI images have more uniform noise
        noise_score = self._check_noise_uniformity(arr)
        if noise_score > 0.5:
            indicators += noise_score
            total_weight += 1
            details.append(f"noise uniformity {noise_score:.2f}")

        # 4. Patch-level statistical consistency
        # AI images sometimes have patches with different statistical properties
        patch_score = self._check_patch_consistency(arr)
        if patch_score > 0.5:
            indicators += patch_score
            total_weight += 1
            details.append(f"patch inconsistency {patch_score:.2f}")

        # 5. Pixel value distribution analysis
        # Check for unnatural clustering of pixel values
        pixel_score = self._check_pixel_distribution(arr)
        if pixel_score > 0.5:
            indicators += pixel_score
            total_weight += 1
            details.append(f"pixel distribution anomaly {pixel_score:.2f}")

        if total_weight < 2:
            return None

        avg_score = indicators / total_weight
        if avg_score < 0.55:
            return None

        confidence = min(1.0, avg_score * 0.8)
        severity = min(0.4, confidence * 0.4)  # Not really "damage", just informational

        return Degradation(
            name="AI-Generated Content",
            confidence=confidence,
            severity=severity,
            detail=f"AI generation likelihood: {confidence:.0%}. Signals: {'; '.join(details)}",
            category="provenance",
        )

    def _check_frequency_spectrum(self, arr: np.ndarray) -> float:
        """AI images often have a characteristic mid-frequency gap."""
        gray = np.mean(arr[:, :, :3].astype(np.float64), axis=2)

        from numpy.fft import fft2, fftshift
        spectrum = np.abs(fftshift(fft2(gray)))

        h, w = spectrum.shape
        cy, cx = h // 2, w // 2

        # Radial bins
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

        max_r = min(cy, cx)
        radial_profile = np.zeros(max_r)
        for i in range(max_r):
            mask = r == i
            if mask.any():
                radial_profile[i] = spectrum[mask].mean()

        if radial_profile.max() == 0:
            return 0.0

        radial_profile /= radial_profile.max()

        # AI images: unusual rolloff in mid-frequencies
        # Split into low/mid/high thirds
        third = max_r // 3
        low = radial_profile[:third].mean()
        mid = radial_profile[third : 2 * third].mean()
        high = radial_profile[2 * third :].mean()

        # Check for unusual mid-frequency behavior
        if low > 0:
            mid_ratio = mid / low
            high_ratio = high / low

            # AI images often have a sharper dropoff
            if mid_ratio < 0.1 and high_ratio < 0.01:
                return 0.7
            if mid_ratio < 0.15:
                return 0.5

        return 0.0

    def _check_histogram_smoothness(self, arr: np.ndarray) -> float:
        """AI images have smoother histograms than real photos."""
        smoothness_scores = []

        for ch in range(3):
            hist, _ = np.histogram(arr[:, :, ch], bins=256, range=(0, 256))
            hist = hist.astype(np.float64)

            # Compute roughness: sum of absolute differences between adjacent bins
            diffs = np.abs(np.diff(hist))
            roughness = diffs.mean() / (hist.mean() + 1e-10)

            # Very smooth histograms suggest AI generation
            smoothness_scores.append(roughness)

        avg_roughness = np.mean(smoothness_scores)

        # Lower roughness = smoother = more likely AI
        if avg_roughness < 0.3:
            return 0.8
        elif avg_roughness < 0.5:
            return 0.6
        elif avg_roughness < 0.8:
            return 0.3

        return 0.0

    def _check_noise_uniformity(self, arr: np.ndarray) -> float:
        """Real photos have spatially varying noise. AI images are more uniform."""
        gray = np.mean(arr[:, :, :3].astype(np.float64), axis=2)
        h, w = gray.shape

        # Compute local noise estimates in patches
        patch_size = 32
        noise_levels = []

        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = gray[y : y + patch_size, x : x + patch_size]
                # Laplacian-based noise estimate
                lap = (
                    patch[:-2, 1:-1] + patch[2:, 1:-1]
                    + patch[1:-1, :-2] + patch[1:-1, 2:]
                    - 4 * patch[1:-1, 1:-1]
                )
                noise = np.median(np.abs(lap)) * 1.4826
                noise_levels.append(noise)

        if len(noise_levels) < 4:
            return 0.0

        noise_levels = np.array(noise_levels)
        mean_noise = noise_levels.mean()

        if mean_noise == 0:
            return 0.0

        # Coefficient of variation of noise levels
        cv = noise_levels.std() / mean_noise

        # Very uniform noise (low cv) suggests AI generation
        if cv < 0.15:
            return 0.8
        elif cv < 0.25:
            return 0.6
        elif cv < 0.4:
            return 0.3

        return 0.0

    def _check_patch_consistency(self, arr: np.ndarray) -> float:
        """Check for suspicious local statistical patterns."""
        gray = np.mean(arr[:, :, :3].astype(np.float64), axis=2)
        h, w = gray.shape

        patch_size = 64
        means = []
        stds = []

        for y in range(0, h - patch_size, patch_size // 2):
            for x in range(0, w - patch_size, patch_size // 2):
                patch = gray[y : y + patch_size, x : x + patch_size]
                means.append(patch.mean())
                stds.append(patch.std())

        if len(means) < 4:
            return 0.0

        means = np.array(means)
        stds = np.array(stds)

        # In AI images, the relationship between local mean and std
        # is often unusually linear or uniform
        if stds.mean() > 0:
            std_cv = stds.std() / stds.mean()
            if std_cv < 0.3:
                return 0.7
            elif std_cv < 0.5:
                return 0.4

        return 0.0

    def _check_pixel_distribution(self, arr: np.ndarray) -> float:
        """Check for unnatural pixel value clustering."""
        # AI generators sometimes produce pixel values that cluster
        # at specific intervals

        scores = []
        for ch in range(min(3, arr.shape[2] if len(arr.shape) > 2 else 1)):
            channel = arr[:, :, ch].flatten().astype(np.float64)

            # Check for gaps in the distribution
            hist, _ = np.histogram(channel, bins=256, range=(0, 256))
            zero_bins = np.sum(hist == 0)
            zero_ratio = zero_bins / 256

            # Some AI generators produce images with missing values
            if zero_ratio > 0.3:
                scores.append(0.7)
            elif zero_ratio > 0.15:
                scores.append(0.4)
            else:
                scores.append(0.0)

        return max(scores) if scores else 0.0

    def _try_neural_detection(
        self, img: Image.Image
    ) -> Degradation | None:
        """Use neural AI detection model if a trained one is available."""
        try:
            from artefex.models_registry import ModelRegistry
            from artefex.neural import NeuralEngine

            registry = ModelRegistry()
            model = registry.get_model("aigen-detect-v1")
            if model is None or not model.is_available or not model.is_trained:
                return None

            engine = NeuralEngine(registry=registry)
            if not engine.available:
                return None

            # Run the detection model
            # Detection models output a probability map or classification
            # score rather than a restored image.
            import onnxruntime as ort

            session = ort.InferenceSession(
                str(model.local_path),
                providers=["CPUExecutionProvider"],
            )

            # SAFE expects a 256x256 center crop, not resize.
            # Center crop preserves pixel-level artifacts that the
            # detector looks for. Pad if image is smaller.
            input_h, input_w = model.input_size
            rgb = img.convert("RGB")
            w, h = rgb.size
            if w >= input_w and h >= input_h:
                left = (w - input_w) // 2
                top = (h - input_h) // 2
                rgb = rgb.crop(
                    (left, top, left + input_w, top + input_h)
                )
            else:
                rgb = rgb.resize(
                    (input_w, input_h), Image.LANCZOS
                )
            arr = np.array(rgb, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)[np.newaxis, :, :, :]

            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: arr})[0]

            # Interpret output as probability of AI generation
            # Most detection models output [real_prob, ai_prob] or a
            # single score where higher = more likely AI
            if output.size == 1:
                ai_prob = float(output.flat[0])
            elif output.shape[-1] >= 2:
                ai_prob = float(output.flat[1])
            else:
                ai_prob = float(np.mean(output))

            ai_prob = max(0.0, min(1.0, ai_prob))

            if ai_prob < 0.3:
                return None

            return Degradation(
                name="AI-Generated Content",
                confidence=ai_prob,
                severity=min(0.5, ai_prob * 0.5),
                detail=(
                    f"Neural AI detection confidence: {ai_prob:.0%}"
                ),
                category="provenance",
            )

        except Exception:
            return None
