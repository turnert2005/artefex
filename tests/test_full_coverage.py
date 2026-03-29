"""Comprehensive tests for undertested and 0%-coverage modules.

Covers: gif_analyze, video, watch, web, orientation, restore, models_registry, plugins.
"""

import hashlib
import io
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image, ExifTags

from artefex.models import AnalysisResult, Degradation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(width=64, height=64, color=(128, 100, 80)):
    """Create a simple solid-color RGB PIL image."""
    return Image.new("RGB", (width, height), color)


def _save_test_image(path: Path, width=64, height=64, color=(128, 100, 80)):
    """Save a simple test image to *path* and return the Path."""
    img = _make_rgb_image(width, height, color)
    img.save(path)
    return path


def _make_gradient_image(width=64, height=64):
    """Create an image with a horizontal gradient (useful for edge tests)."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        arr[:, x, :] = int(255 * x / max(width - 1, 1))
    return Image.fromarray(arr)


def _make_multiframe_gif(path: Path, n_frames=4, size=(32, 32)):
    """Create a multi-frame GIF programmatically."""
    frames = []
    for i in range(n_frames):
        r = int(255 * i / max(n_frames - 1, 1))
        frames.append(Image.new("RGB", size, (r, 100, 50)))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    return path


# ---------------------------------------------------------------------------
# 1. GIF Analysis (gif_analyze.py)
# ---------------------------------------------------------------------------

class TestGifAnalyzer:
    """Tests for GifAnalyzer and GifAnalysisResult."""

    def test_single_frame_image(self, tmp_path):
        """GifAnalyzer should handle a static single-frame PNG."""
        from artefex.gif_analyze import GifAnalyzer

        img_path = _save_test_image(tmp_path / "single.png")
        ga = GifAnalyzer(max_frames=5)
        result = ga.analyze(img_path)

        assert result.frame_count == 1
        assert result.is_animated is False
        assert len(result.frame_results) == 1

    def test_multiframe_gif(self, tmp_path):
        """GifAnalyzer should detect and analyze multiple frames."""
        from artefex.gif_analyze import GifAnalyzer

        gif_path = _make_multiframe_gif(tmp_path / "multi.gif", n_frames=4)
        ga = GifAnalyzer(max_frames=10)
        result = ga.analyze(gif_path)

        assert result.is_animated is True
        assert result.frame_count == 4
        assert len(result.frame_results) > 0
        assert result.total_duration_ms > 0

    def test_gif_analysis_result_properties(self, tmp_path):
        """GifAnalysisResult should expose degradation_summary correctly."""
        from artefex.gif_analyze import GifAnalyzer

        gif_path = _make_multiframe_gif(tmp_path / "props.gif", n_frames=3)
        ga = GifAnalyzer(max_frames=5)
        result = ga.analyze(gif_path)

        # frame_results should be a list
        assert isinstance(result.frame_results, list)
        # degradation_summary should be a dict (possibly empty for clean images)
        assert isinstance(result.degradation_summary, dict)
        assert result.avg_frame_duration_ms > 0

    def test_frame_similarity_calculation(self, tmp_path):
        """Frame similarity list should have len == n_sampled_frames - 1."""
        from artefex.gif_analyze import GifAnalyzer

        gif_path = _make_multiframe_gif(tmp_path / "sim.gif", n_frames=5)
        ga = GifAnalyzer(max_frames=10)
        result = ga.analyze(gif_path)

        # Similarity is computed between consecutive sampled frames
        assert isinstance(result.frame_similarity, list)
        assert len(result.frame_similarity) == len(result.frame_results) - 1
        # Each value should be between 0 and 1
        for s in result.frame_similarity:
            assert 0.0 <= s <= 1.0

    def test_progress_callback(self, tmp_path):
        """on_progress callback should be invoked for animated GIFs."""
        from artefex.gif_analyze import GifAnalyzer

        gif_path = _make_multiframe_gif(tmp_path / "cb.gif", n_frames=3)
        ga = GifAnalyzer(max_frames=10)
        calls = []
        ga.analyze(gif_path, on_progress=lambda done, total: calls.append((done, total)))
        assert len(calls) > 0


# ---------------------------------------------------------------------------
# 2. Video (video.py) - opencv IS available
# ---------------------------------------------------------------------------

class TestVideoAnalyzer:
    """Tests for VideoAnalyzer and VideoRestorer."""

    @staticmethod
    def _make_test_video(path: Path, n_frames=10, size=(64, 64)):
        """Create a tiny synthetic AVI video using cv2.VideoWriter."""
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
        for i in range(n_frames):
            r = int(255 * i / max(n_frames - 1, 1))
            frame = np.full((size[1], size[0], 3), (r, 100, 50), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path

    def test_video_analyze(self, tmp_path):
        """VideoAnalyzer.analyze() should return populated result."""
        from artefex.video import VideoAnalyzer

        vid = self._make_test_video(tmp_path / "test.avi")
        va = VideoAnalyzer(sample_count=3)
        result = va.analyze(vid)

        assert result.frame_count >= 1
        assert result.fps > 0
        assert result.resolution == (64, 64)
        assert len(result.frame_results) > 0

    def test_video_analysis_result_overall_severity(self, tmp_path):
        """VideoAnalysisResult.overall_severity property should work."""
        from artefex.video import VideoAnalysisResult

        # Empty summary - severity is 0
        vr = VideoAnalysisResult()
        assert vr.overall_severity == 0.0

        # With a summary entry
        vr.degradation_summary = {"Noise": {"avg_severity": 0.6}}
        assert vr.overall_severity == 0.6

    def test_video_restorer_produces_output(self, tmp_path):
        """VideoRestorer.restore() should create an output video file."""
        from artefex.video import VideoAnalyzer, VideoRestorer

        vid = self._make_test_video(tmp_path / "src.avi", n_frames=6)
        out = tmp_path / "restored.avi"

        # Analyze first to build a summary with at least one degradation
        va = VideoAnalyzer(sample_count=3)
        analysis = va.analyze(vid)

        # If no degradation detected, inject one so the restorer does work
        if not analysis.degradation_summary:
            analysis.degradation_summary["Noise"] = {
                "count": 3,
                "frequency": 1.0,
                "avg_severity": 0.5,
                "avg_confidence": 0.8,
                "category": "noise",
            }

        vr = VideoRestorer(use_neural=False, temporal_strength=0.1)
        info = vr.restore(vid, out, analysis=analysis)

        assert info["frames_processed"] >= 1
        assert out.exists()

    def test_video_restorer_temporal_strength(self):
        """VideoRestorer should accept and clamp temporal_strength."""
        from artefex.video import VideoRestorer

        vr = VideoRestorer(use_neural=False, temporal_strength=0.5)
        assert vr.temporal_strength == 0.5

        vr2 = VideoRestorer(use_neural=False, temporal_strength=2.0)
        assert vr2.temporal_strength == 1.0

        vr3 = VideoRestorer(use_neural=False, temporal_strength=-1.0)
        assert vr3.temporal_strength == 0.0

    def test_video_restorer_unsupported_codec(self, tmp_path):
        """VideoRestorer.restore() should reject unknown codecs."""
        from artefex.video import VideoRestorer

        vid = self._make_test_video(tmp_path / "codec.avi", n_frames=3)
        out = tmp_path / "out.avi"
        vr = VideoRestorer(use_neural=False)
        analysis = MagicMock()
        analysis.degradation_summary = {}

        with pytest.raises(ValueError, match="Unsupported codec"):
            vr.restore(vid, out, analysis=analysis, codec="ZZZZ")


# ---------------------------------------------------------------------------
# 3. Watch (watch.py)
# ---------------------------------------------------------------------------

class TestDirectoryWatcher:
    """Tests for the DirectoryWatcher."""

    def test_watcher_init(self, tmp_path):
        """Watcher can be initialized with a directory."""
        from artefex.watch import DirectoryWatcher

        watcher = DirectoryWatcher(tmp_path)
        assert watcher.watch_dir == tmp_path
        assert watcher.auto_restore is False

    def test_watcher_detects_new_files(self, tmp_path):
        """Watcher should detect newly added image files."""
        from artefex.watch import DirectoryWatcher

        watcher = DirectoryWatcher(tmp_path, auto_restore=False)
        # Mark existing files as seen (simulate initial scan)
        for f in tmp_path.iterdir():
            if f.suffix.lower() in {".jpg", ".png"}:
                watcher._seen.add(str(f))

        # No new files yet
        assert watcher._scan() == []

        # Add a new image
        _save_test_image(tmp_path / "new_test.png")
        new_files = watcher._scan()
        assert len(new_files) == 1
        assert new_files[0].name == "new_test.png"

    def test_watcher_process_analyzes_file(self, tmp_path):
        """_process should analyze the file and return info dict."""
        from artefex.watch import DirectoryWatcher

        img_path = _save_test_image(tmp_path / "proc.png")
        watcher = DirectoryWatcher(tmp_path, auto_restore=False)
        info = watcher._process(img_path)

        assert info["file"] == "proc.png"
        assert "degradations" in info
        assert "overall_severity" in info
        assert info["restored"] is False

    def test_watcher_auto_restore(self, tmp_path):
        """With auto_restore=True, _process should create a restored file."""
        from artefex.watch import DirectoryWatcher

        img_path = _save_test_image(tmp_path / "auto.png")
        out_dir = tmp_path / "artefex_output"
        watcher = DirectoryWatcher(
            tmp_path, output_dir=out_dir, auto_restore=True, use_neural=False,
        )
        info = watcher._process(img_path)

        # Even if no degradation found, the process should complete
        assert info["file"] == "auto.png"
        # If degradations were detected, output should exist
        if info["degradations"] > 0:
            assert info["restored"] is True
            assert out_dir.exists()

    def test_watcher_callback(self, tmp_path):
        """on_result callback should be called by _process."""
        from artefex.watch import DirectoryWatcher

        img_path = _save_test_image(tmp_path / "cb.png")
        watcher = DirectoryWatcher(tmp_path)
        results = []
        watcher._process(img_path, on_result=lambda info: results.append(info))
        assert len(results) == 1
        assert results[0]["file"] == "cb.png"


# ---------------------------------------------------------------------------
# 4. Web (web.py) - fastapi IS available
# ---------------------------------------------------------------------------

class TestWebApp:
    """Tests for FastAPI web endpoints."""

    @pytest.fixture()
    def client(self):
        from starlette.testclient import TestClient
        from artefex.web import app
        return TestClient(app)

    def test_index_returns_html(self, client):
        """GET / should return the HTML UI."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "artefex" in resp.text.lower()

    def test_api_analyze(self, client, tmp_path):
        """POST /api/analyze should return degradation JSON."""
        img_path = _save_test_image(tmp_path / "web_test.png")
        with open(img_path, "rb") as f:
            resp = client.post(
                "/api/analyze",
                files={"file": ("test.png", f, "image/png")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "degradations" in data
        assert "overall_severity" in data
        assert "dimensions" in data

    def test_api_restore(self, client, tmp_path):
        """POST /api/restore should return a PNG image."""
        img_path = _save_test_image(tmp_path / "web_restore.png")
        with open(img_path, "rb") as f:
            resp = client.post(
                "/api/restore",
                files={"file": ("test.png", f, "image/png")},
            )
        assert resp.status_code == 200
        assert "image/png" in resp.headers["content-type"]
        # Should be a valid image
        img = Image.open(io.BytesIO(resp.content))
        assert img.size[0] > 0

    def test_api_report(self, client, tmp_path):
        """POST /api/report should return report text."""
        img_path = _save_test_image(tmp_path / "web_report.png")
        with open(img_path, "rb") as f:
            resp = client.post(
                "/api/report",
                files={"file": ("test.png", f, "image/png")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "report" in data
        assert isinstance(data["report"], str)


# ---------------------------------------------------------------------------
# 5. Orientation (orientation.py)
# ---------------------------------------------------------------------------

class TestOrientation:
    """Tests for orientation detection and auto-correction."""

    def test_auto_orient_with_exif(self):
        """auto_orient should correct an image with EXIF orientation=6 (90 CW)."""
        from artefex.orientation import auto_orient

        img = _make_rgb_image(80, 40)
        # Inject EXIF orientation tag
        exif = img.getexif()
        # Find the orientation tag id
        orient_tag = None
        for tag, name in ExifTags.TAGS.items():
            if name == "Orientation":
                orient_tag = tag
                break
        assert orient_tag is not None
        exif[orient_tag] = 6  # Rotated 90 CW
        img.info["exif"] = exif.tobytes()
        # Re-open to make sure exif is embedded
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif.tobytes())
        buf.seek(0)
        img_with_exif = Image.open(buf)

        corrected, info = auto_orient(img_with_exif)
        assert info["exif_orientation"] == 6
        assert info["needs_correction"] is True
        # Rotation of 90-degree image swaps width/height
        assert corrected.size[0] == 40
        assert corrected.size[1] == 80

    def test_horizon_tilt_estimation(self):
        """_estimate_horizon_tilt should return a float angle."""
        from artefex.orientation import _estimate_horizon_tilt

        # Create an image with a strong horizontal edge in the middle
        arr = np.zeros((100, 100), dtype=np.float64)
        arr[50:, :] = 200.0  # sharp edge at row 50
        tilt = _estimate_horizon_tilt(arr)
        assert isinstance(tilt, float)
        assert -15 <= tilt <= 15

    def test_auto_orient_no_correction_needed(self):
        """auto_orient with orientation=1 should return image unchanged."""
        from artefex.orientation import auto_orient

        img = _make_rgb_image(60, 60)
        corrected, info = auto_orient(img)
        assert info["needs_correction"] is False or info["exif_orientation"] is None
        assert corrected.size == img.size

    def test_detect_orientation_returns_dict(self):
        """detect_orientation should return a well-formed dict."""
        from artefex.orientation import detect_orientation

        img = _make_rgb_image(64, 64)
        result = detect_orientation(img)
        assert "exif_orientation" in result
        assert "horizon_tilt" in result
        assert "needs_correction" in result
        assert "suggested_rotation" in result

    def test_auto_orient_rotated_180(self):
        """auto_orient should handle EXIF orientation=3 (180 rotation)."""
        from artefex.orientation import auto_orient

        img = _make_rgb_image(80, 40)
        exif = img.getexif()
        orient_tag = None
        for tag, name in ExifTags.TAGS.items():
            if name == "Orientation":
                orient_tag = tag
                break
        exif[orient_tag] = 3  # Rotated 180
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif.tobytes())
        buf.seek(0)
        img_with_exif = Image.open(buf)

        corrected, info = auto_orient(img_with_exif)
        assert info["exif_orientation"] == 3
        # 180 rotation keeps same dimensions
        assert corrected.size == (80, 40)


# ---------------------------------------------------------------------------
# 6. Restore classical methods (restore.py)
# ---------------------------------------------------------------------------

class TestRestorationPipeline:
    """Tests for classical restoration methods in RestorationPipeline."""

    def _pipeline(self):
        from artefex.restore import RestorationPipeline
        return RestorationPipeline(use_neural=False)

    def test_fix_jpeg_artifacts(self):
        """_fix_jpeg_artifacts should return a modified image."""
        pipeline = self._pipeline()
        img = _make_gradient_image(64, 64)
        deg = Degradation(
            name="JPEG Compression", confidence=0.9,
            severity=0.7, category="compression",
        )
        result = pipeline._fix_jpeg_artifacts(img, deg)
        assert isinstance(result, Image.Image)
        assert result.size == img.size

    def test_fix_noise(self):
        """_fix_noise should denoise without changing image size."""
        pipeline = self._pipeline()
        # Create a noisy image
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        deg = Degradation(
            name="Noise", confidence=0.8, severity=0.5, category="noise",
        )
        result = pipeline._fix_noise(img, deg)
        assert isinstance(result, Image.Image)
        assert result.size == img.size

    def test_fix_color_shift(self):
        """_fix_color_shift should normalize color channels."""
        pipeline = self._pipeline()
        # Image with heavy red tint
        img = _make_rgb_image(64, 64, color=(220, 80, 80))
        deg = Degradation(
            name="Color Shift", confidence=0.7,
            severity=0.6, category="color",
        )
        result = pipeline._fix_color_shift(img, deg)
        assert isinstance(result, Image.Image)
        # After correction the red channel mean should be closer to overall mean
        orig = np.array(img, dtype=np.float64)
        fixed = np.array(result, dtype=np.float64)
        orig_spread = orig.mean(axis=(0, 1)).std()
        fixed_spread = fixed.mean(axis=(0, 1)).std()
        assert fixed_spread <= orig_spread + 1.0

    def test_fix_screenshot_borders(self):
        """_fix_screenshot_borders should crop solid black borders."""
        pipeline = self._pipeline()
        # 100x100 image with 10px black border around a colored center
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[10:90, 10:90, :] = 128
        img = Image.fromarray(arr)
        deg = Degradation(
            name="Screenshot Artifacts", confidence=0.8,
            severity=0.3, category="artifact",
        )
        result = pipeline._fix_screenshot_borders(img, deg)
        assert isinstance(result, Image.Image)
        # Should be smaller than original due to border cropping
        assert result.size[0] <= img.size[0]
        assert result.size[1] <= img.size[1]

    def test_fix_resolution(self):
        """_fix_resolution should apply sharpening."""
        pipeline = self._pipeline()
        img = _make_gradient_image(64, 64)
        deg = Degradation(
            name="Resolution Loss / Upscaling", confidence=0.7,
            severity=0.5, category="resolution",
        )
        result = pipeline._fix_resolution(img, deg)
        assert isinstance(result, Image.Image)
        assert result.size == img.size

    def test_full_restore_pipeline(self, tmp_path):
        """restore() should produce an output file and return step info."""
        pipeline = self._pipeline()
        img_path = _save_test_image(tmp_path / "input.png")
        out_path = tmp_path / "output.png"
        analysis = AnalysisResult(
            degradations=[
                Degradation(
                    name="Noise", confidence=0.8,
                    severity=0.5, category="noise",
                ),
            ],
        )
        info = pipeline.restore(img_path, analysis, out_path)
        assert out_path.exists()
        assert len(info["steps"]) >= 1
        assert info["used_neural"] is False

    def test_restore_empty_degradation_list(self, tmp_path):
        """restore() with no degradations should still produce output."""
        pipeline = self._pipeline()
        img_path = _save_test_image(tmp_path / "clean.png")
        out_path = tmp_path / "clean_out.png"
        analysis = AnalysisResult(degradations=[])
        info = pipeline.restore(img_path, analysis, out_path)
        assert out_path.exists()
        assert info["steps"] == []


def _try_import_onnx():
    """Helper to check if onnx is importable."""
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# 7. Models registry (models_registry.py)
# ---------------------------------------------------------------------------

class TestModelRegistry:
    """Tests for ModelRegistry methods."""

    def test_list_models(self, tmp_path):
        """list_models should return all registered models."""
        from artefex.models_registry import ModelRegistry, REGISTRY

        reg = ModelRegistry(model_dir=tmp_path / "models")
        models = reg.list_models()
        assert len(models) == len(REGISTRY)
        for m in models:
            assert m.key in REGISTRY

    def test_import_model(self, tmp_path):
        """import_model should copy a file into the model directory."""
        from artefex.models_registry import ModelRegistry

        model_dir = tmp_path / "models"
        reg = ModelRegistry(model_dir=model_dir)

        # Create a fake model file
        src = tmp_path / "fake_model.onnx"
        src.write_bytes(b"fake onnx data")

        info = reg.import_model(src, "deblock-v1")
        assert info.is_available
        assert info.local_path.exists()
        assert info.local_path.read_bytes() == b"fake onnx data"

    def test_import_model_unknown_key(self, tmp_path):
        """import_model should raise ValueError for unknown keys."""
        from artefex.models_registry import ModelRegistry

        reg = ModelRegistry(model_dir=tmp_path / "models")
        src = tmp_path / "fake.onnx"
        src.write_bytes(b"data")

        with pytest.raises(ValueError, match="Unknown model key"):
            reg.import_model(src, "nonexistent-model")

    def test_model_path_available(self, tmp_path):
        """model_path should return path when model file exists."""
        from artefex.models_registry import ModelRegistry, REGISTRY

        model_dir = tmp_path / "models"
        reg = ModelRegistry(model_dir=model_dir)

        # Place a fake model file
        info = REGISTRY["denoise-v1"]
        dest = model_dir / info["filename"]
        dest.write_bytes(b"model bytes")

        path = reg.model_path("denoise-v1")
        assert path is not None
        assert path.exists()

    def test_model_path_unavailable(self, tmp_path):
        """model_path should return None when model is not downloaded."""
        from artefex.models_registry import ModelRegistry

        reg = ModelRegistry(model_dir=tmp_path / "models")
        assert reg.model_path("deblock-v1") is None

    def test_verify_sha256(self, tmp_path):
        """_verify_sha256 should validate a file hash correctly."""
        from artefex.models_registry import ModelRegistry

        test_file = tmp_path / "hashtest.bin"
        content = b"hello artefex"
        test_file.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        assert ModelRegistry._verify_sha256(test_file, expected) is True
        assert ModelRegistry._verify_sha256(test_file, "wrong_hash") is False

    def test_get_model_for_category(self, tmp_path):
        """get_model_for_category should find models by category."""
        from artefex.models_registry import ModelRegistry

        reg = ModelRegistry(model_dir=tmp_path / "models")
        model = reg.get_model_for_category("compression")
        assert model is not None
        assert model.category == "compression"

        missing = reg.get_model_for_category("nonexistent_category")
        assert missing is None

    @pytest.mark.skipif(
        not _try_import_onnx(),
        reason="onnx package not available",
    )
    def test_generate_test_models(self, tmp_path):
        """generate_test_models should create ONNX files for all entries."""
        from artefex.models_registry import ModelRegistry, REGISTRY

        reg = ModelRegistry(model_dir=tmp_path / "models")
        results = reg.generate_test_models()
        assert len(results) == len(REGISTRY)
        for m in results:
            assert m.is_available
            assert m.local_path.exists()
            assert m.local_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# 8. Plugins (plugins.py)
# ---------------------------------------------------------------------------

class TestPluginRegistry:
    """Tests for the plugin registry."""

    def test_registry_loads_without_error(self):
        """PluginRegistry should initialize and load without crashing."""
        from artefex.plugins import PluginRegistry

        reg = PluginRegistry()
        # Force load
        plugins = reg.list_plugins()
        assert "detectors" in plugins
        assert "restorers" in plugins

    def test_run_detectors_empty(self):
        """run_detectors should return empty list when no plugins installed."""
        from artefex.plugins import PluginRegistry

        reg = PluginRegistry()
        img = _make_rgb_image(32, 32)
        arr = np.array(img)
        results = reg.run_detectors(img, arr)
        # May be empty (no external plugins), but should not crash
        assert isinstance(results, list)

    def test_run_restorer_no_match(self):
        """run_restorer should return None when no plugin handles it."""
        from artefex.plugins import PluginRegistry

        reg = PluginRegistry()
        img = _make_rgb_image(32, 32)
        deg = Degradation(
            name="SomeUnknownDegradation",
            confidence=0.9, severity=0.5,
        )
        result = reg.run_restorer(img, deg)
        assert result is None

    def test_run_detectors_with_mock_plugin(self):
        """Simulate a detector plugin by injecting into _detectors."""
        from artefex.plugins import PluginRegistry

        reg = PluginRegistry()
        reg._loaded = True  # skip entry point loading

        mock_detector = MagicMock()
        mock_detector.name = "mock_detector"
        mock_detector.detect.return_value = Degradation(
            name="MockIssue", confidence=0.95,
            severity=0.8, category="test",
        )
        reg._detectors["mock_detector"] = mock_detector

        img = _make_rgb_image(32, 32)
        arr = np.array(img)
        results = reg.run_detectors(img, arr)

        assert len(results) == 1
        assert results[0].name == "MockIssue"
        mock_detector.detect.assert_called_once()

    def test_run_restorer_with_mock_plugin(self):
        """Simulate a restorer plugin by injecting into _restorers."""
        from artefex.plugins import PluginRegistry

        reg = PluginRegistry()
        reg._loaded = True

        restored_img = _make_rgb_image(32, 32, color=(0, 255, 0))
        mock_restorer = MagicMock()
        mock_restorer.name = "MockIssue"
        mock_restorer.restore.return_value = restored_img
        reg._restorers["MockIssue"] = mock_restorer

        img = _make_rgb_image(32, 32)
        deg = Degradation(name="MockIssue", confidence=0.9, severity=0.5)
        result = reg.run_restorer(img, deg)

        assert result is not None
        assert result.size == (32, 32)
        mock_restorer.restore.assert_called_once()

    def test_get_plugin_registry_singleton(self):
        """get_plugin_registry should return the same instance."""
        from artefex.plugins import get_plugin_registry

        r1 = get_plugin_registry()
        r2 = get_plugin_registry()
        assert r1 is r2
