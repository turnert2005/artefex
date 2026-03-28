"""Tests for the restoration pipeline."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from artefex.analyze import DegradationAnalyzer
from artefex.restore import RestorationPipeline


def _make_degraded_jpeg(quality=15) -> Path:
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        for x in range(256):
            arr[y, x] = [x % 256, y % 256, (x + y) % 256]
    img = Image.fromarray(arr)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    img.save(tmp.name, format="JPEG", quality=quality)
    return Path(tmp.name)


def _cleanup(*paths):
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except PermissionError:
            pass


def test_restore_produces_output():
    path = _make_degraded_jpeg(quality=10)
    out_path = path.with_stem(path.stem + "_restored")

    analyzer = DegradationAnalyzer()
    pipeline = RestorationPipeline(use_neural=False)

    result = analyzer.analyze(path)
    info = pipeline.restore(path, result, out_path)

    assert out_path.exists()
    assert len(info["steps"]) > 0
    assert info["used_neural"] is False
    _cleanup(path, out_path)


def test_restore_returns_step_details():
    path = _make_degraded_jpeg(quality=5)
    out_path = path.with_stem(path.stem + "_restored")

    analyzer = DegradationAnalyzer()
    pipeline = RestorationPipeline(use_neural=False)

    result = analyzer.analyze(path)
    info = pipeline.restore(path, result, out_path)

    assert all(isinstance(s, str) for s in info["steps"])
    assert any("[classical]" in s for s in info["steps"])
    _cleanup(path, out_path)


def test_restore_format_conversion():
    path = _make_degraded_jpeg(quality=10)
    out_path = path.with_suffix(".png")

    analyzer = DegradationAnalyzer()
    pipeline = RestorationPipeline(use_neural=False)

    result = analyzer.analyze(path)
    pipeline.restore(path, result, out_path, format="PNG")

    # Should create a PNG
    restored = Image.open(out_path)
    assert restored.format == "PNG"
    _cleanup(path, out_path)


def test_restore_clean_image_no_changes():
    """A clean PNG with no degradation should pass through unchanged."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    for y in range(64):
        for x in range(64):
            arr[y, x] = [x * 4, y * 4, 128]
    img = Image.fromarray(arr)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    img.save(tmp.name)
    path = Path(tmp.name)

    analyzer = DegradationAnalyzer()
    pipeline = RestorationPipeline(use_neural=False)

    result = analyzer.analyze(path)

    if not result.degradations:
        # No degradation = nothing to restore, which is correct
        _cleanup(path)
        return

    out_path = path.with_stem(path.stem + "_restored")
    info = pipeline.restore(path, result, out_path)
    assert isinstance(info, dict)
    _cleanup(path, out_path)
