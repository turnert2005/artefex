"""Steganography detection - identify hidden data embedded in images.

Detects signs of LSB (Least Significant Bit) steganography and other
common data-hiding techniques by analyzing statistical anomalies in
pixel value distributions.

Methods:
1. LSB plane analysis - checks if least significant bits are suspiciously random
2. Chi-square analysis - statistical test for LSB embedding
3. Sample pair analysis - detects sequential LSB replacement
4. Entropy analysis - unusually high entropy in LSB planes suggests hidden data
"""

import numpy as np
from PIL import Image

from artefex.models import Degradation


class SteganographyDetector:
    """Detects hidden data embedded in images via steganography."""

    def detect(self, img: Image.Image, arr: np.ndarray) -> Degradation | None:
        if len(arr.shape) < 3 or arr.shape[2] < 3:
            return None

        h, w = arr.shape[:2]
        if h < 32 or w < 32:
            return None

        indicators = 0
        total_tests = 0
        details = []

        # 1. LSB randomness test
        lsb_score = self._lsb_randomness(arr)
        total_tests += 1
        if lsb_score > 0.5:
            indicators += lsb_score
            details.append(f"LSB randomness: {lsb_score:.2f}")

        # 2. Chi-square test on LSB pairs
        chi_score = self._chi_square_test(arr)
        total_tests += 1
        if chi_score > 0.5:
            indicators += chi_score
            details.append(f"chi-square anomaly: {chi_score:.2f}")

        # 3. LSB plane entropy
        entropy_score = self._lsb_entropy(arr)
        total_tests += 1
        if entropy_score > 0.5:
            indicators += entropy_score
            details.append(f"LSB entropy: {entropy_score:.2f}")

        # 4. Pairs of values analysis
        pairs_score = self._pairs_analysis(arr)
        total_tests += 1
        if pairs_score > 0.5:
            indicators += pairs_score
            details.append(f"pair imbalance: {pairs_score:.2f}")

        if len(details) < 2:
            return None

        avg_score = indicators / total_tests
        if avg_score < 0.4:
            return None

        confidence = min(1.0, avg_score)
        severity = min(0.3, confidence * 0.3)  # Informational

        return Degradation(
            name="Steganography Detected",
            confidence=confidence,
            severity=severity,
            detail=f"Hidden data likelihood: {confidence:.0%}. Signals: {'; '.join(details)}",
            category="provenance",
        )

    def _lsb_randomness(self, arr: np.ndarray) -> float:
        """Check if LSB plane looks suspiciously random (uniform distribution)."""
        scores = []
        for ch in range(3):
            lsb = arr[:, :, ch] & 1
            ones_ratio = lsb.mean()

            # Natural images: LSB ratio varies, often not exactly 0.5
            # Steganography: drives ratio toward 0.5
            deviation = abs(ones_ratio - 0.5)

            # Very close to 0.5 is suspicious
            if deviation < 0.005:
                scores.append(0.9)
            elif deviation < 0.01:
                scores.append(0.7)
            elif deviation < 0.02:
                scores.append(0.5)
            else:
                scores.append(0.0)

        return max(scores)

    def _chi_square_test(self, arr: np.ndarray) -> float:
        """Chi-square test for pairs of values (2i, 2i+1) in pixel histogram."""
        scores = []
        for ch in range(3):
            channel = arr[:, :, ch].flatten()
            hist, _ = np.histogram(channel, bins=256, range=(0, 256))

            # Compare adjacent pairs: hist[0] vs hist[1], hist[2] vs hist[3], etc.
            chi_sq = 0.0
            pairs = 0
            for i in range(0, 255, 2):
                expected = (hist[i] + hist[i + 1]) / 2.0
                if expected > 5:  # Only count meaningful pairs
                    chi_sq += (hist[i] - expected) ** 2 / expected
                    chi_sq += (hist[i + 1] - expected) ** 2 / expected
                    pairs += 1

            if pairs == 0:
                scores.append(0.0)
                continue

            # Normalize by number of pairs
            normalized = chi_sq / pairs

            # Low chi-square = suspiciously equal pairs = possible LSB steganography
            if normalized < 0.5:
                scores.append(0.9)
            elif normalized < 1.0:
                scores.append(0.6)
            elif normalized < 2.0:
                scores.append(0.3)
            else:
                scores.append(0.0)

        return max(scores)

    def _lsb_entropy(self, arr: np.ndarray) -> float:
        """Check entropy of the LSB plane - hidden data maximizes entropy."""
        scores = []
        for ch in range(3):
            lsb = arr[:, :, ch] & 1

            # Compute spatial entropy: how random is the LSB arrangement?
            # Check 2x2 blocks
            h, w = lsb.shape
            patterns = {}
            for y in range(0, h - 1, 2):
                for x in range(0, w - 1, 2):
                    block = (
                        lsb[y, x] << 3
                        | lsb[y, x + 1] << 2
                        | lsb[y + 1, x] << 1
                        | lsb[y + 1, x + 1]
                    )
                    patterns[block] = patterns.get(block, 0) + 1

            total = sum(patterns.values())
            if total == 0:
                scores.append(0.0)
                continue

            # Compute entropy
            entropy = 0.0
            for count in patterns.values():
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)

            max_entropy = 4.0  # 16 possible patterns = 4 bits
            ratio = entropy / max_entropy

            # Very high entropy = suspicious
            if ratio > 0.98:
                scores.append(0.9)
            elif ratio > 0.95:
                scores.append(0.6)
            elif ratio > 0.90:
                scores.append(0.3)
            else:
                scores.append(0.0)

        return max(scores)

    def _pairs_analysis(self, arr: np.ndarray) -> float:
        """Sample pairs analysis for LSB replacement detection."""
        scores = []
        for ch in range(3):
            channel = arr[:, :, ch].flatten()

            # Count pairs where adjacent pixels differ only in LSB
            diffs = np.abs(channel[:-1].astype(np.int16) - channel[1:].astype(np.int16))
            lsb_diffs = np.sum(diffs == 1)
            total_pairs = len(diffs)

            if total_pairs == 0:
                scores.append(0.0)
                continue

            lsb_ratio = lsb_diffs / total_pairs

            # Natural images: lsb_ratio varies by content
            # LSB steganography: increases the number of +/-1 differences
            # Typical natural range: 0.05-0.15
            if lsb_ratio > 0.25:
                scores.append(0.8)
            elif lsb_ratio > 0.20:
                scores.append(0.5)
            else:
                scores.append(0.0)

        return max(scores)
