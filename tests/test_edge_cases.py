"""QA Round 2 - edge cases, stress testing, and hardening.

Covers neural pipeline, restore pipeline, API, config, plugin, and
boundary/stress test scenarios that were not exercised in earlier rounds.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from artefex.api import AnalyzeResult, analyze, compare, grade
from artefex.config import ArtefexConfig, load_config
from artefex.models import AnalysisResult, Degradation
from artefex.models_registry import ModelRegistry, REGISTRY
from artefex.neural import NeuralEngine
from artefex.plugins import PluginRegistry, get_plugin_registry
from artefex.restore import RestorationPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_jpeg(img: Image.Image, path: Path, quality: int = 75) -> Path:
    """Save an image as JPEG with the given quality."""
    rgb = img.convert("RGB")
    rgb.save(path, format="JPEG", quality=quality)
    return path


def _make_gradient(w: int, h: int, mode: str = "RGB") -> Image.Image:
    """Create a gradient test image of the given size and mode."""
    if mode == "L":
        arr = np.tile(
            np.linspace(0, 255, w, dtype=np.uint8), (h, 1)
        )
        return Image.fromarray(arr, mode="L")
    channels = 4 if mode == "RGBA" else 3
    arr = np.zeros((h, w, channels), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            arr[y, x, 0] = x % 256
            arr[y, x, 1] = y % 256
            arr[y, x, 2] = (x + y) % 256
            if channels == 4:
                arr[y, x, 3] = 255
    return Image.fromarray(arr, mode=mode)


@pytest.fixture()
def model_registry(tmp_path):
    """Return a ModelRegistry rooted in a temporary directory."""
    return ModelRegistry(model_dir=tmp_path / "models")


@pytest.fixture()
def test_models(model_registry):
    """Generate lightweight ONNX test models and return the registry."""
    model_registry.generate_test_models()
    return model_registry


# ===================================================================
# 1. Neural pipeline edge cases
# ===================================================================

class TestNeuralPipelineEdgeCases:
    """Neural inference engine edge cases."""

    def test_neural_restore_heavily_degraded(self, test_models, tmp_path):
        """A quality-5 JPEG should survive neural deblocking."""
        img = _make_gradient(128, 128)
        jpeg_path = tmp_path / "heavy.jpg"
        _save_jpeg(img, jpeg_path, quality=5)
        degraded = Image.open(jpeg_path)

        engine = NeuralEngine(registry=test_models)
        result = engine.run("deblock-v1", degraded)

        assert isinstance(result, Image.Image)
        assert result.size == degraded.size

    def test_neural_tiling_large_image(self, test_models):
        """Images larger than model input size trigger tiled inference."""
        img = _make_gradient(600, 400)
        engine = NeuralEngine(registry=test_models)
        result = engine.run("denoise-v1", img)

        assert isinstance(result, Image.Image)
        assert result.size == img.size

    def test_neural_padded_run_small_image(self, test_models):
        """Small images get padded up to model input size."""
        img = _make_gradient(32, 32)
        engine = NeuralEngine(registry=test_models)
        result = engine.run("sharpen-v1", img)

        assert isinstance(result, Image.Image)
        assert result.size == img.size

    def test_neural_grayscale_through_deblock(self, test_models):
        """Grayscale images through FBCNN (3ch) get converted to RGB."""
        img = _make_gradient(64, 64, mode="L")
        engine = NeuralEngine(registry=test_models)
        result = engine.run("deblock-v1", img)

        assert isinstance(result, Image.Image)
        # FBCNN is 3-channel, so grayscale input becomes RGB output
        assert result.mode == "RGB"
        assert result.size == img.size

    def test_neural_grayscale_rgb_roundtrip(self, test_models):
        """An RGB image through the 1-channel model merges color back."""
        img = _make_gradient(64, 64, mode="RGB")
        engine = NeuralEngine(registry=test_models)
        result = engine.run("deblock-v1", img)

        assert result.mode == "RGB"
        assert result.size == img.size

    def test_generate_test_models(self, model_registry):
        """generate_test_models() creates ONNX files for every registry entry."""
        models = model_registry.generate_test_models()

        assert len(models) == len(REGISTRY)
        for m in models:
            assert m.is_available
            assert m.local_path.exists()
            assert m.local_path.suffix == ".onnx"

    def test_model_registry_list_after_install(self, test_models):
        """list_models() shows availability after models are installed."""
        models = test_models.list_models()
        available = [m for m in models if m.is_available]
        assert len(available) == len(REGISTRY)
        for m in available:
            assert m.local_path is not None
            assert m.local_path.exists()


# ===================================================================
# 2. Restore pipeline edge cases
# ===================================================================

class TestRestorePipelineEdgeCases:
    """Restoration pipeline edge cases."""

    def test_restore_with_neural_enabled(self, test_models, tmp_path):
        """Test models are untrained (< 10 KB) so neural should be skipped.

        The system correctly detects test models and falls back to
        classical methods to avoid quality degradation.
        """
        img = _make_gradient(128, 128)
        jpeg_path = tmp_path / "input.jpg"
        _save_jpeg(img, jpeg_path, quality=10)
        out_path = tmp_path / "output.png"

        pipeline = RestorationPipeline(use_neural=True)
        engine = NeuralEngine(registry=test_models)
        pipeline._neural_engine = engine

        analysis = AnalysisResult(
            file_path=str(jpeg_path),
            file_format="JPEG",
            dimensions=(128, 128),
            degradations=[
                Degradation(
                    name="JPEG Compression",
                    confidence=0.9,
                    severity=0.8,
                    category="compression",
                ),
            ],
        )

        info = pipeline.restore(jpeg_path, analysis, out_path)
        # Test models are untrained so neural is correctly skipped
        assert info["used_neural"] is False
        assert out_path.exists()

    def test_restore_multiple_degradation_types(self, tmp_path):
        """Restoring an image with several degradation types at once."""
        img = _make_gradient(128, 128)
        jpeg_path = tmp_path / "multi.jpg"
        _save_jpeg(img, jpeg_path, quality=10)
        out_path = tmp_path / "multi_out.png"

        pipeline = RestorationPipeline(use_neural=False)
        analysis = AnalysisResult(
            file_path=str(jpeg_path),
            file_format="JPEG",
            dimensions=(128, 128),
            degradations=[
                Degradation(
                    name="JPEG Compression",
                    confidence=0.9,
                    severity=0.7,
                    category="compression",
                ),
                Degradation(
                    name="Noise",
                    confidence=0.6,
                    severity=0.4,
                    category="noise",
                ),
                Degradation(
                    name="Color Shift",
                    confidence=0.5,
                    severity=0.3,
                    category="color",
                ),
            ],
        )

        info = pipeline.restore(jpeg_path, analysis, out_path)
        assert len(info["steps"]) == 3
        assert out_path.exists()

    def test_restore_format_jpg_to_png(self, tmp_path):
        """Converting from JPEG to PNG during restore."""
        img = _make_gradient(64, 64)
        jpeg_path = tmp_path / "convert.jpg"
        _save_jpeg(img, jpeg_path, quality=20)
        out_path = tmp_path / "convert.jpg"  # will be changed to .png

        pipeline = RestorationPipeline(use_neural=False)
        analysis = AnalysisResult(
            file_path=str(jpeg_path),
            file_format="JPEG",
            dimensions=(64, 64),
            degradations=[
                Degradation(
                    name="JPEG Compression",
                    confidence=0.8,
                    severity=0.6,
                    category="compression",
                ),
            ],
        )

        info = pipeline.restore(jpeg_path, analysis, out_path, format="PNG")
        actual_out = Path(info["output_path"])
        assert actual_out.suffix == ".png"
        assert actual_out.exists()
        restored = Image.open(actual_out)
        assert restored.format == "PNG"

    def test_restore_format_png_to_jpg(self, tmp_path):
        """Converting from PNG to JPEG during restore."""
        img = _make_gradient(64, 64)
        png_path = tmp_path / "source.png"
        img.save(png_path, format="PNG")
        out_path = tmp_path / "source_out.png"

        pipeline = RestorationPipeline(use_neural=False)
        analysis = AnalysisResult(
            file_path=str(png_path),
            file_format="PNG",
            dimensions=(64, 64),
            degradations=[
                Degradation(
                    name="Noise",
                    confidence=0.7,
                    severity=0.5,
                    category="noise",
                ),
            ],
        )

        info = pipeline.restore(png_path, analysis, out_path, format="JPEG")
        actual_out = Path(info["output_path"])
        assert actual_out.suffix == ".jpeg"
        assert actual_out.exists()

    def test_restore_very_small_image(self, tmp_path):
        """A 16x16 image should still be restorable without errors."""
        arr = np.random.default_rng(99).integers(
            0, 256, (16, 16, 3), dtype=np.uint8
        )
        img = Image.fromarray(arr)
        jpeg_path = tmp_path / "tiny.jpg"
        _save_jpeg(img, jpeg_path, quality=10)
        out_path = tmp_path / "tiny_out.png"

        pipeline = RestorationPipeline(use_neural=False)
        analysis = AnalysisResult(
            file_path=str(jpeg_path),
            file_format="JPEG",
            dimensions=(16, 16),
            degradations=[
                Degradation(
                    name="JPEG Compression",
                    confidence=0.8,
                    severity=0.6,
                    category="compression",
                ),
            ],
        )

        pipeline.restore(jpeg_path, analysis, out_path)
        assert out_path.exists()
        restored = Image.open(out_path)
        assert restored.size == (16, 16)


# ===================================================================
# 3. API edge cases
# ===================================================================

class TestAPIEdgeCases:
    """Public API edge cases."""

    def test_analyze_result_to_dict_structure(self):
        """to_dict() must return all expected keys with correct types."""
        result = AnalyzeResult(
            file="test.jpg",
            format="JPEG",
            dimensions=(100, 100),
            degradations=[
                Degradation(
                    name="Noise",
                    confidence=0.75,
                    severity=0.5,
                    detail="gaussian noise",
                    category="noise",
                ),
            ],
            grade="C",
            score=55.0,
            grade_description="Fair quality",
            overall_severity=0.5,
        )

        d = result.to_dict()

        assert d["file"] == "test.jpg"
        assert d["format"] == "JPEG"
        assert d["dimensions"] == [100, 100]
        assert d["grade"] == "C"
        assert d["score"] == 55.0
        assert isinstance(d["overall_severity"], float)
        assert len(d["degradations"]) == 1

        deg = d["degradations"][0]
        assert deg["name"] == "Noise"
        assert deg["category"] == "noise"
        assert isinstance(deg["confidence"], float)
        assert isinstance(deg["severity"], float)
        assert deg["detail"] == "gaussian noise"

    def test_compare_different_sized_images(self, tmp_path):
        """compare() auto-resizes when images differ in dimensions."""
        img_a = _make_gradient(100, 100)
        img_b = _make_gradient(200, 150)

        path_a = tmp_path / "a.png"
        path_b = tmp_path / "b.png"
        img_a.save(path_a, format="PNG")
        img_b.save(path_b, format="PNG")

        metrics = compare(path_a, path_b)

        assert "mse" in metrics
        assert "psnr" in metrics
        assert "pixels_changed_pct" in metrics
        # MSE should be finite and non-negative
        assert metrics["mse"] >= 0
        # PSNR should be finite (images differ)
        assert metrics["psnr"] > 0

    def test_grade_pristine_vs_degraded(self, tmp_path):
        """A clean PNG should score better than a heavily degraded JPEG."""
        # Create a smooth, natural-looking image (solid with mild gradient)
        arr = np.full((128, 128, 3), 128, dtype=np.uint8)
        img_clean = Image.fromarray(arr)
        clean_path = tmp_path / "pristine.png"
        img_clean.save(clean_path, format="PNG")

        # Create a heavily degraded JPEG
        noisy = np.random.default_rng(42).integers(
            0, 256, (128, 128, 3), dtype=np.uint8
        )
        img_bad = Image.fromarray(noisy)
        bad_path = tmp_path / "bad.jpg"
        _save_jpeg(img_bad, bad_path, quality=5)

        clean_grade = grade(clean_path)
        bad_grade = grade(bad_path)

        # The clean image should score higher than the degraded one
        assert clean_grade["score"] > bad_grade["score"]

    def test_grade_heavily_degraded(self, tmp_path):
        """A quality-5 JPEG should get a poor grade (C, D, or F)."""
        img = _make_gradient(128, 128)
        path = tmp_path / "bad.jpg"
        _save_jpeg(img, path, quality=5)

        result = grade(path)

        assert result["grade"] in ("C", "D", "F")
        assert result["score"] < 80.0


# ===================================================================
# 4. Config edge cases
# ===================================================================

class TestConfigEdgeCases:
    """Configuration system edge cases."""

    def test_config_from_pyproject_toml(self, tmp_path):
        """load_config picks up [tool.artefex] from pyproject.toml."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[tool.artefex.analysis]\n'
            'min_confidence = 0.5\n'
            '\n'
            '[tool.artefex.restore]\n'
            'use_neural = false\n'
            'output_quality = 80\n',
            encoding="utf-8",
        )

        cfg = load_config(start_dir=tmp_path)

        assert cfg.min_confidence == 0.5
        assert cfg.use_neural is False
        assert cfg.output_quality == 80

    def test_config_hierarchy_project_overrides_defaults(self, tmp_path):
        """A project .artefex.toml overrides built-in defaults."""
        config_file = tmp_path / ".artefex.toml"
        config_file.write_text(
            '[analysis]\n'
            'min_confidence = 0.9\n'
            'min_severity = 0.3\n'
            '\n'
            '[web]\n'
            'port = 9999\n',
            encoding="utf-8",
        )

        cfg = load_config(start_dir=tmp_path)
        default = ArtefexConfig()

        assert cfg.min_confidence == 0.9
        assert cfg.min_severity == 0.3
        assert cfg.web_port == 9999
        # Fields not set should keep defaults
        assert cfg.use_neural == default.use_neural
        assert cfg.verbose == default.verbose


# ===================================================================
# 5. Plugin system
# ===================================================================

class TestPluginSystem:
    """Plugin registry edge cases."""

    def test_plugin_registry_loads_without_errors(self):
        """Creating and loading the registry should not raise."""
        registry = PluginRegistry()
        plugins = registry.list_plugins()

        assert isinstance(plugins, dict)
        assert "detectors" in plugins
        assert "restorers" in plugins

    def test_plugin_registry_no_external_plugins(self):
        """Without external plugins installed, lists are empty."""
        registry = PluginRegistry()
        assert isinstance(registry.detectors, dict)
        assert isinstance(registry.restorers, dict)
        # No third-party plugins in the test environment
        plugins = registry.list_plugins()
        assert plugins["detectors"] == []
        assert plugins["restorers"] == []

    def test_global_plugin_registry_singleton(self):
        """get_plugin_registry() returns the same singleton."""
        r1 = get_plugin_registry()
        r2 = get_plugin_registry()
        assert r1 is r2


# ===================================================================
# 6. Stress and boundary tests
# ===================================================================

class TestStressBoundary:
    """Boundary and stress scenarios."""

    def test_analyze_1x1_pixel_image(self, tmp_path):
        """A 1x1 image should not crash the analyzer."""
        img = Image.new("RGB", (1, 1), color=(128, 128, 128))
        path = tmp_path / "pixel.png"
        img.save(path, format="PNG")

        result = analyze(path)

        assert result.dimensions == (1, 1)
        assert isinstance(result.degradations, list)

    def test_analyze_large_rgba_image(self, tmp_path):
        """A 1000x1000 RGBA image should be handled correctly."""
        arr = np.random.default_rng(7).integers(
            0, 256, (1000, 1000, 4), dtype=np.uint8
        )
        img = Image.fromarray(arr, mode="RGBA")
        path = tmp_path / "large_rgba.png"
        img.save(path, format="PNG")

        result = analyze(path)

        assert result.dimensions == (1000, 1000)
        assert isinstance(result.degradations, list)

    @pytest.mark.slow
    def test_analyze_large_rgba_no_timeout(self, tmp_path):
        """Ensure the 1000x1000 RGBA analysis completes in time."""
        import time

        arr = np.random.default_rng(8).integers(
            0, 256, (1000, 1000, 4), dtype=np.uint8
        )
        img = Image.fromarray(arr, mode="RGBA")
        path = tmp_path / "large_timed.png"
        img.save(path, format="PNG")

        start = time.perf_counter()
        analyze(path)
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, f"Analysis took {elapsed:.1f}s, exceeding 5s limit"

    def test_analyze_png_with_transparency(self, tmp_path):
        """RGBA PNG with varying alpha should not crash analysis."""
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        for y in range(64):
            for x in range(64):
                arr[y, x] = [x * 4, y * 4, 128, (x + y) * 2]
        img = Image.fromarray(arr, mode="RGBA")
        path = tmp_path / "alpha.png"
        img.save(path, format="PNG")

        result = analyze(path)

        assert result.format == "PNG"
        assert isinstance(result.degradations, list)

    def test_restore_1x1_pixel_image(self, tmp_path):
        """Restoring a 1x1 image should not crash."""
        img = Image.new("RGB", (1, 1), color=(100, 100, 100))
        path = tmp_path / "pixel.jpg"
        _save_jpeg(img, path, quality=10)
        out_path = tmp_path / "pixel_out.png"

        pipeline = RestorationPipeline(use_neural=False)
        analysis = AnalysisResult(
            file_path=str(path),
            file_format="JPEG",
            dimensions=(1, 1),
            degradations=[
                Degradation(
                    name="JPEG Compression",
                    confidence=0.7,
                    severity=0.5,
                    category="compression",
                ),
            ],
        )

        pipeline.restore(path, analysis, out_path)
        assert out_path.exists()
        restored = Image.open(out_path)
        assert restored.size == (1, 1)

    def test_neural_engine_unknown_model_key(self, test_models):
        """Requesting an unknown model key should raise ValueError."""
        engine = NeuralEngine(registry=test_models)
        with pytest.raises(ValueError, match="Unknown model"):
            engine.run("nonexistent-model", _make_gradient(32, 32))

    def test_analyze_result_is_clean_property(self):
        """is_clean should be True only when no degradations exist."""
        clean = AnalyzeResult(
            file="a.png",
            format="PNG",
            dimensions=(10, 10),
            degradations=[],
            grade="A",
            score=100.0,
            grade_description="Pristine",
            overall_severity=0.0,
        )
        degraded = AnalyzeResult(
            file="b.jpg",
            format="JPEG",
            dimensions=(10, 10),
            degradations=[
                Degradation("Noise", 0.5, 0.5, category="noise"),
            ],
            grade="C",
            score=50.0,
            grade_description="Fair",
            overall_severity=0.5,
        )

        assert clean.is_clean is True
        assert bool(clean) is False
        assert degraded.is_clean is False
        assert bool(degraded) is True
        assert degraded.top_issue == "Noise"
        assert clean.top_issue is None
