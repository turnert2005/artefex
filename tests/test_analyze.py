"""Tests for the degradation analyzer."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from artefex.analyze import DegradationAnalyzer


def _make_test_image(w=256, h=256, mode="RGB") -> Image.Image:
    """Create a test image with varied content."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # Gradient
    for y in range(h):
        for x in range(w):
            arr[y, x] = [x % 256, y % 256, (x + y) % 256]
    return Image.fromarray(arr, mode)


def _save_jpeg(img: Image.Image, quality: int) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    img.save(tmp.name, format="JPEG", quality=quality)
    return Path(tmp.name)


def _cleanup(path: Path):
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        pass  # Windows file locking


def test_analyzer_returns_result():
    img = _make_test_image()
    path = _save_jpeg(img, quality=95)
    analyzer = DegradationAnalyzer()
    result = analyzer.analyze(path)

    assert result.file_format == "JPEG"
    assert result.dimensions == (256, 256)
    _cleanup(path)


def test_heavy_jpeg_compression_detected():
    img = _make_test_image()
    path = _save_jpeg(img, quality=5)
    analyzer = DegradationAnalyzer()
    result = analyzer.analyze(path)

    names = [d.name for d in result.degradations]
    assert "JPEG Compression" in names
    _cleanup(path)


def test_clean_png_minimal_degradation():
    img = _make_test_image()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    img.save(tmp.name, format="PNG")
    path = Path(tmp.name)

    analyzer = DegradationAnalyzer()
    result = analyzer.analyze(path)

    # PNG should have no JPEG compression artifacts
    names = [d.name for d in result.degradations]
    assert "JPEG Compression" not in names
    _cleanup(path)


def test_noise_detected():
    img = _make_test_image()
    arr = np.array(img).astype(np.float64)
    noise = np.random.normal(0, 40, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy)

    path = _save_jpeg(noisy_img, quality=95)
    analyzer = DegradationAnalyzer()
    result = analyzer.analyze(path)

    names = [d.name for d in result.degradations]
    assert "Noise" in names
    _cleanup(path)


def test_multiple_recompressions():
    img = _make_test_image()

    # Compress multiple times
    for _ in range(5):
        path = _save_jpeg(img, quality=30)
        img = Image.open(path)
        img.load()  # Force read so file handle is released

    analyzer = DegradationAnalyzer()
    result = analyzer.analyze(path)

    # Should detect compression-related degradation
    categories = [d.category for d in result.degradations]
    assert "compression" in categories
    _cleanup(path)
