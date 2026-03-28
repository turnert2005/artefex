"""Platform fingerprinting - detect which platforms an image has passed through.

Social media platforms and messaging apps apply specific compression patterns
when images are uploaded. By analyzing JPEG quantization tables, resolution
patterns, EXIF signatures, and compression artifacts, we can identify which
platform(s) likely processed an image.

This is the feature that makes Artefex unique - forensic platform attribution.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from artefex.models import Degradation


# Known platform signatures
# Each platform has characteristic behaviors when processing images
PLATFORM_SIGNATURES = {
    "twitter": {
        "name": "Twitter / X",
        "max_dimension": 4096,
        "typical_quality_range": (75, 85),
        "strips_exif": True,
        "converts_to_jpeg": True,
        "common_dimensions": [
            (1200, 675),  # Summary card
            (1200, 1200),  # Square
            (4096, 4096),  # Max
        ],
        "jfif_signature": True,
        "quantization_signature": "moderate",
    },
    "instagram": {
        "name": "Instagram",
        "max_dimension": 1440,
        "typical_quality_range": (70, 80),
        "strips_exif": True,
        "converts_to_jpeg": True,
        "common_dimensions": [
            (1080, 1080),  # Square
            (1080, 1350),  # Portrait 4:5
            (1080, 608),   # Landscape 1.91:1
        ],
        "jfif_signature": True,
        "quantization_signature": "aggressive",
    },
    "whatsapp": {
        "name": "WhatsApp",
        "max_dimension": 1600,
        "typical_quality_range": (50, 70),
        "strips_exif": True,
        "converts_to_jpeg": True,
        "common_dimensions": [
            (1600, 1200),
            (1280, 960),
            (1024, 768),
        ],
        "jfif_signature": True,
        "quantization_signature": "heavy",
    },
    "facebook": {
        "name": "Facebook",
        "max_dimension": 2048,
        "typical_quality_range": (71, 85),
        "strips_exif": False,  # Keeps some EXIF
        "converts_to_jpeg": True,
        "common_dimensions": [
            (2048, 2048),
            (1200, 630),  # Link preview
            (1080, 1080),
        ],
        "jfif_signature": True,
        "quantization_signature": "moderate",
    },
    "telegram": {
        "name": "Telegram",
        "max_dimension": 2560,
        "typical_quality_range": (80, 90),
        "strips_exif": True,
        "converts_to_jpeg": True,
        "common_dimensions": [
            (1280, 1280),
            (2560, 2560),
        ],
        "jfif_signature": True,
        "quantization_signature": "light",
    },
    "discord": {
        "name": "Discord",
        "max_dimension": 4096,
        "typical_quality_range": (80, 90),
        "strips_exif": True,
        "converts_to_jpeg": False,  # Often keeps PNG
        "common_dimensions": [],  # Variable
        "jfif_signature": True,
        "quantization_signature": "light",
    },
    "imgur": {
        "name": "Imgur",
        "max_dimension": 5120,
        "typical_quality_range": (80, 92),
        "strips_exif": True,
        "converts_to_jpeg": False,
        "common_dimensions": [],
        "jfif_signature": True,
        "quantization_signature": "light",
    },
}


class PlatformFingerprinter:
    """Detects which platform(s) likely processed an image."""

    def fingerprint(self, file_path: Path) -> list[dict]:
        """Analyze an image and return likely platform matches with confidence.

        Returns list of dicts: [{"platform": str, "name": str, "confidence": float, "evidence": list[str]}]
        """
        img = Image.open(file_path)
        arr = np.array(img)

        results = []

        for key, sig in PLATFORM_SIGNATURES.items():
            evidence = []
            score = 0.0

            # Check dimensions
            w, h = img.size
            dim_score = self._check_dimensions(w, h, sig)
            if dim_score > 0:
                evidence.append(f"dimensions {w}x{h} match {sig['name']} patterns")
                score += dim_score

            # Check max dimension constraint
            max_dim = max(w, h)
            if max_dim <= sig["max_dimension"] and max_dim > sig["max_dimension"] * 0.7:
                evidence.append(f"fits within {sig['name']} max dimension ({sig['max_dimension']}px)")
                score += 0.1

            # Check EXIF stripping
            exif_stripped = self._check_exif_stripped(img)
            if sig["strips_exif"] and exif_stripped:
                evidence.append("EXIF metadata stripped (consistent)")
                score += 0.15
            elif not sig["strips_exif"] and not exif_stripped:
                evidence.append("EXIF preserved (consistent)")
                score += 0.1

            # Check format
            if sig["converts_to_jpeg"] and img.format == "JPEG":
                evidence.append("JPEG format (expected)")
                score += 0.1
            elif not sig["converts_to_jpeg"] and img.format == "PNG":
                evidence.append("PNG preserved (consistent)")
                score += 0.1

            # Check JFIF header
            if sig["jfif_signature"] and hasattr(img, "info"):
                if "jfif_version" in img.info:
                    evidence.append("JFIF header present")
                    score += 0.05

            # Check compression level
            if img.format == "JPEG":
                comp_score = self._check_compression_level(img, arr, sig)
                if comp_score > 0:
                    evidence.append(f"compression level matches {sig['name']} range")
                    score += comp_score

            # Only include if there's meaningful evidence
            if score >= 0.25 and len(evidence) >= 2:
                results.append({
                    "platform": key,
                    "name": sig["name"],
                    "confidence": min(1.0, score),
                    "evidence": evidence,
                })

        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def _check_dimensions(self, w: int, h: int, sig: dict) -> float:
        """Check if dimensions match known platform patterns."""
        for cw, ch in sig["common_dimensions"]:
            # Exact match
            if w == cw and h == ch:
                return 0.3
            # Close match (within 5%)
            if abs(w - cw) / cw < 0.05 and abs(h - ch) / ch < 0.05:
                return 0.2
            # Aspect ratio match
            if cw > 0 and ch > 0:
                target_ratio = cw / ch
                actual_ratio = w / h if h > 0 else 0
                if abs(actual_ratio - target_ratio) < 0.02:
                    return 0.1
        return 0.0

    def _check_exif_stripped(self, img: Image.Image) -> bool:
        """Check if EXIF data has been stripped."""
        try:
            exif = img.getexif()
            return exif is None or len(exif) == 0
        except Exception:
            return True

    def _check_compression_level(self, img: Image.Image, arr: np.ndarray, sig: dict) -> float:
        """Estimate if compression level matches platform's typical range."""
        # Estimate quality from blockiness
        if len(arr.shape) < 3:
            gray = arr.astype(np.float64)
        else:
            gray = np.mean(arr[:, :, :3].astype(np.float64), axis=2)

        h, w = gray.shape
        if h < 16 or w < 16:
            return 0.0

        boundary_diffs = []
        interior_diffs = []
        for y in range(8, min(h - 8, 200), 8):
            boundary_diffs.extend(np.abs(gray[y, :] - gray[y - 1, :]))
            interior_diffs.extend(np.abs(gray[y + 1, :] - gray[y, :]))

        if not interior_diffs:
            return 0.0

        mean_boundary = np.mean(boundary_diffs)
        mean_interior = np.mean(interior_diffs)

        if mean_interior == 0:
            return 0.0

        blockiness = mean_boundary / mean_interior

        # Map blockiness to estimated quality
        # Higher blockiness = lower quality
        # blockiness ~1.0 = high quality (90+)
        # blockiness ~1.1 = moderate (70-85)
        # blockiness ~1.3+ = heavy compression (50-70)
        est_quality = max(0, min(100, 100 - (blockiness - 1.0) * 200))

        q_low, q_high = sig["typical_quality_range"]
        if q_low <= est_quality <= q_high:
            return 0.2
        elif q_low - 10 <= est_quality <= q_high + 10:
            return 0.1

        return 0.0


def detect_platform(file_path: Path) -> Optional[Degradation]:
    """Run platform fingerprinting and return as a Degradation if detected."""
    fp = PlatformFingerprinter()
    matches = fp.fingerprint(file_path)

    if not matches:
        return None

    top = matches[0]
    if top["confidence"] < 0.3:
        return None

    detail_parts = [f"Most likely: {top['name']} ({top['confidence']:.0%})"]
    for ev in top["evidence"][:3]:
        detail_parts.append(ev)

    if len(matches) > 1:
        others = ", ".join(f"{m['name']} ({m['confidence']:.0%})" for m in matches[1:3])
        detail_parts.append(f"Also possible: {others}")

    return Degradation(
        name="Platform Processing",
        confidence=top["confidence"],
        severity=min(0.5, top["confidence"] * 0.6),
        detail="; ".join(detail_parts),
        category="provenance",
    )
