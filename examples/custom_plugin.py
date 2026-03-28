"""Example: creating a custom Artefex plugin.

This shows how to build a detector plugin that checks for
low dynamic range (washed out images).

To register as a plugin, add to your package's pyproject.toml:

    [project.entry-points."artefex.detectors"]
    low_dynamic_range = "my_package:LowDynamicRangeDetector"
"""

import numpy as np
from PIL import Image

from artefex.models import Degradation


class LowDynamicRangeDetector:
    """Detects images with abnormally low dynamic range (washed out)."""

    name = "Low Dynamic Range"

    def detect(self, img: Image.Image, arr: np.ndarray) -> Degradation | None:
        if len(arr.shape) < 3:
            gray = arr.astype(np.float64)
        else:
            gray = np.mean(arr[:, :, :3].astype(np.float64), axis=2)

        # Check the range of pixel values
        p5 = np.percentile(gray, 5)
        p95 = np.percentile(gray, 95)
        dynamic_range = p95 - p5

        # A healthy image typically has a range of 150+
        if dynamic_range > 120:
            return None

        severity = min(1.0, (120 - dynamic_range) / 80)
        confidence = min(1.0, (120 - dynamic_range) / 60)

        return Degradation(
            name=self.name,
            confidence=confidence,
            severity=severity,
            detail=f"Dynamic range: {dynamic_range:.1f} (5th={p5:.1f}, 95th={p95:.1f})",
            category="color",
        )


# Example restorer for low dynamic range
class DynamicRangeRestorer:
    """Stretches histogram to restore dynamic range."""

    name = "Low Dynamic Range"

    def restore(self, img: Image.Image, degradation: Degradation) -> Image.Image:
        arr = np.array(img, dtype=np.float64)

        for ch in range(min(3, arr.shape[2] if len(arr.shape) > 2 else 1)):
            channel = arr[:, :, ch] if len(arr.shape) > 2 else arr
            p2 = np.percentile(channel, 2)
            p98 = np.percentile(channel, 98)

            if p98 - p2 > 0:
                stretched = (channel - p2) / (p98 - p2) * 255
                if len(arr.shape) > 2:
                    arr[:, :, ch] = stretched
                else:
                    arr = stretched

        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    # Quick test
    import tempfile
    from pathlib import Path

    # Create a low dynamic range test image
    arr = np.random.randint(80, 180, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    detector = LowDynamicRangeDetector()
    result = detector.detect(img, np.array(img))

    if result:
        print(f"Detected: {result.name}")
        print(f"  Severity: {result.severity:.0%}")
        print(f"  Detail: {result.detail}")

        restorer = DynamicRangeRestorer()
        fixed = restorer.restore(img, result)
        print(f"  Restored image range: {np.array(fixed).min()}-{np.array(fixed).max()}")
    else:
        print("No low dynamic range detected")
