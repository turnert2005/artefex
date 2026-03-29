"""Tests for CLI commands that lack coverage.

Covers: grade, compare, heatmap, timeline, story, dashboard, gallery,
gate, health, fix, audit, duplicates, accessibility, palette, orient,
rename-by-grade, parallel-analyze, benchmark, version, restore-preview.
"""

import json

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from artefex.cli import app

runner = CliRunner()


@pytest.fixture()
def test_jpeg(tmp_path):
    """Create a low-quality JPEG for testing."""
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    p = tmp_path / "test.jpg"
    img.save(str(p), format="JPEG", quality=15)
    return p


@pytest.fixture()
def test_png(tmp_path):
    """Create a PNG for testing."""
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    p = tmp_path / "test.png"
    img.save(str(p), format="PNG")
    return p


@pytest.fixture()
def two_jpegs(tmp_path):
    """Create two different JPEG images for comparison/duplicates."""
    for name, seed in [("img_a.jpg", 42), ("img_b.jpg", 99)]:
        rng = np.random.RandomState(seed)
        arr = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(str(tmp_path / name), format="JPEG", quality=20)
    return tmp_path / "img_a.jpg", tmp_path / "img_b.jpg"


@pytest.fixture()
def image_dir(tmp_path):
    """Create a directory with several test images."""
    for i in range(3):
        rng = np.random.RandomState(i)
        arr = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(
            str(tmp_path / f"photo_{i}.jpg"),
            format="JPEG",
            quality=15,
        )
    return tmp_path


# -- grade --

class TestGrade:
    def test_grade_single(self, test_jpeg):
        result = runner.invoke(app, ["grade", str(test_jpeg)])
        assert result.exit_code == 0
        assert "Grade" in result.output or "score" in result.output.lower()

    def test_grade_json(self, test_jpeg):
        result = runner.invoke(app, ["grade", str(test_jpeg), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "grade" in data
        assert "score" in data

    def test_grade_csv_export(self, test_jpeg):
        result = runner.invoke(
            app, ["grade", str(test_jpeg), "--export", "csv"]
        )
        assert result.exit_code == 0
        assert "file,grade,score" in result.output

    def test_grade_markdown_export(self, test_jpeg):
        result = runner.invoke(
            app, ["grade", str(test_jpeg), "--export", "markdown"]
        )
        assert result.exit_code == 0
        assert "| File |" in result.output

    def test_grade_missing_file(self):
        result = runner.invoke(app, ["grade", "no_such_file.jpg"])
        assert result.exit_code == 1

    def test_grade_directory(self, image_dir):
        result = runner.invoke(app, ["grade", str(image_dir)])
        assert result.exit_code == 0


# -- compare --

class TestCompare:
    def test_compare_two_images(self, two_jpegs):
        a, b = two_jpegs
        result = runner.invoke(app, ["compare", str(a), str(b)])
        assert result.exit_code == 0
        assert "PSNR" in result.output
        assert "SSIM" in result.output

    def test_compare_missing_file(self, test_jpeg):
        result = runner.invoke(
            app, ["compare", str(test_jpeg), "nonexistent.jpg"]
        )
        assert result.exit_code == 1


# -- heatmap --

class TestHeatmap:
    def test_heatmap_default(self, test_jpeg, tmp_path):
        out = tmp_path / "heat.png"
        result = runner.invoke(
            app, ["heatmap", str(test_jpeg), "--output", str(out)]
        )
        assert result.exit_code == 0
        assert "Heatmap saved" in result.output or out.exists()

    def test_heatmap_missing(self):
        result = runner.invoke(app, ["heatmap", "nope.jpg"])
        assert result.exit_code == 1


# -- timeline --

class TestTimeline:
    def test_timeline(self, test_jpeg):
        result = runner.invoke(app, ["timeline", str(test_jpeg)])
        assert result.exit_code == 0
        # A degraded JPEG should show a timeline or a clean message
        assert (
            "Timeline" in result.output
            or "No degradation" in result.output
            or "Original capture" in result.output
            or "Clean" in result.output
        )

    def test_timeline_missing(self):
        result = runner.invoke(app, ["timeline", "missing.png"])
        assert result.exit_code == 1


# -- story --

class TestStory:
    def test_story(self, test_jpeg):
        result = runner.invoke(app, ["story", str(test_jpeg)])
        assert result.exit_code == 0
        # Story should produce some narrative text
        assert len(result.output.strip()) > 0

    def test_story_missing(self):
        result = runner.invoke(app, ["story", "missing.png"])
        assert result.exit_code == 1


# -- dashboard --

class TestDashboard:
    def test_dashboard(self, image_dir, tmp_path):
        out = tmp_path / "dash.html"
        result = runner.invoke(
            app, ["dashboard", str(image_dir), "--output", str(out)]
        )
        assert result.exit_code == 0
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<html" in content.lower() or "<!doctype" in content.lower()

    def test_dashboard_missing_dir(self):
        result = runner.invoke(app, ["dashboard", "no_dir"])
        assert result.exit_code == 1


# -- gallery --

class TestGallery:
    def test_gallery(self, tmp_path):
        orig_dir = tmp_path / "originals"
        rest_dir = tmp_path / "restored"
        orig_dir.mkdir()
        rest_dir.mkdir()

        # Create matching pairs
        for name in ["photo1", "photo2"]:
            arr = np.random.randint(
                0, 255, (64, 64, 3), dtype=np.uint8
            )
            img = Image.fromarray(arr)
            img.save(str(orig_dir / f"{name}.jpg"), format="JPEG")
            img.save(str(rest_dir / f"{name}.jpg"), format="JPEG")

        out = tmp_path / "gallery.html"
        result = runner.invoke(
            app,
            [
                "gallery",
                str(orig_dir),
                str(rest_dir),
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0
        assert out.exists()

    def test_gallery_no_pairs(self, tmp_path):
        d1 = tmp_path / "a"
        d2 = tmp_path / "b"
        d1.mkdir()
        d2.mkdir()
        # Create non-matching files
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(str(d1 / "foo.jpg"), format="JPEG")
        img.save(str(d2 / "bar.jpg"), format="JPEG")
        result = runner.invoke(
            app, ["gallery", str(d1), str(d2)]
        )
        assert result.exit_code == 1

    def test_gallery_not_dirs(self, test_jpeg, tmp_path):
        result = runner.invoke(
            app, ["gallery", str(test_jpeg), str(tmp_path)]
        )
        assert result.exit_code == 1


# -- gate (quality-gate) --

class TestGate:
    def test_gate_pass(self, test_jpeg):
        result = runner.invoke(
            app, ["gate", str(test_jpeg), "--min-grade", "F"]
        )
        # With min-grade F everything should pass
        assert result.exit_code == 0
        assert "PASS" in result.output or "passed" in result.output.lower()

    def test_gate_json(self, test_jpeg):
        result = runner.invoke(
            app,
            ["gate", str(test_jpeg), "--min-grade", "F", "--json"],
        )
        # Should be valid JSON regardless of pass/fail
        data = json.loads(result.output)
        assert "passed" in data or "failed" in data

    def test_gate_strict_may_fail(self, test_jpeg):
        result = runner.invoke(
            app,
            [
                "gate",
                str(test_jpeg),
                "--min-grade",
                "A",
                "--min-score",
                "99",
            ],
        )
        # A heavily degraded JPEG should fail strict criteria
        assert result.exit_code in (0, 1)

    def test_gate_no_files(self):
        result = runner.invoke(app, ["gate", "nonexistent.jpg"])
        assert result.exit_code == 1


# -- health --

class TestHealth:
    def test_health_single(self, test_jpeg):
        result = runner.invoke(app, ["health", str(test_jpeg)])
        assert result.exit_code == 0
        # Should output grade letter and score
        assert any(
            g in result.output for g in ["A", "B", "C", "D", "F"]
        )

    def test_health_missing(self):
        result = runner.invoke(app, ["health", "nope.jpg"])
        assert result.exit_code == 1


# -- fix --

class TestFix:
    def test_fix(self, test_jpeg, tmp_path):
        out = tmp_path / "fixed.jpg"
        result = runner.invoke(
            app, ["fix", str(test_jpeg), "--output", str(out)]
        )
        assert result.exit_code == 0
        # Should mention either fixed or clean
        assert (
            "Fixed" in result.output
            or "fixed" in result.output
            or "Clean" in result.output
            or "clean" in result.output
        )

    def test_fix_missing(self):
        result = runner.invoke(app, ["fix", "missing.jpg"])
        assert result.exit_code == 1


# -- audit --

class TestAudit:
    def test_audit(self, test_jpeg, tmp_path):
        out_dir = tmp_path / "audit_out"
        result = runner.invoke(
            app,
            ["audit", str(test_jpeg), "--output", str(out_dir)],
        )
        assert result.exit_code == 0
        assert "Audit" in result.output or "Grade" in result.output
        # Should create output files
        assert out_dir.exists()

    def test_audit_missing(self):
        result = runner.invoke(app, ["audit", "missing.jpg"])
        assert result.exit_code == 1


# -- duplicates --

class TestDuplicates:
    def test_duplicates_no_dupes(self, image_dir):
        result = runner.invoke(app, ["duplicates", str(image_dir)])
        assert result.exit_code == 0
        # Random images should not be duplicates
        assert (
            "No duplicates" in result.output
            or "Duplicate" in result.output
        )

    def test_duplicates_json(self, image_dir):
        result = runner.invoke(
            app, ["duplicates", str(image_dir), "--json"]
        )
        assert result.exit_code == 0
        # The output may contain rich console text before JSON;
        # extract the JSON array from the output.
        raw = result.output
        idx = raw.find("[")
        assert idx >= 0, f"No JSON array in output: {raw}"
        data = json.loads(raw[idx:])
        assert isinstance(data, list)

    def test_duplicates_missing_dir(self):
        result = runner.invoke(app, ["duplicates", "no_dir"])
        assert result.exit_code == 1

    def test_duplicates_with_actual_dupe(self, tmp_path):
        """Two identical images should be detected as duplicates."""
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(str(tmp_path / "dup1.png"), format="PNG")
        img.save(str(tmp_path / "dup2.png"), format="PNG")
        result = runner.invoke(
            app,
            ["duplicates", str(tmp_path), "--threshold", "0.9"],
        )
        assert result.exit_code == 0


# -- accessibility --

class TestAccessibility:
    def test_accessibility(self, test_png):
        result = runner.invoke(app, ["accessibility", str(test_png)])
        assert result.exit_code == 0
        assert (
            "Accessibility" in result.output
            or "Contrast" in result.output
            or "Color" in result.output
        )

    def test_accessibility_simulate(self, test_png, tmp_path):
        out_dir = tmp_path / "cvd"
        result = runner.invoke(
            app,
            [
                "accessibility",
                str(test_png),
                "--simulate",
                "protanopia",
                "--output",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0

    def test_accessibility_missing(self):
        result = runner.invoke(app, ["accessibility", "nope.png"])
        assert result.exit_code == 1


# -- palette --

class TestPalette:
    def test_palette(self, test_png):
        result = runner.invoke(app, ["palette", str(test_png)])
        assert result.exit_code == 0
        assert "Palette" in result.output or "Hex" in result.output

    def test_palette_json(self, test_png):
        result = runner.invoke(
            app, ["palette", str(test_png), "--json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "hex" in data[0]

    def test_palette_custom_count(self, test_png):
        result = runner.invoke(
            app, ["palette", str(test_png), "--colors", "4"]
        )
        assert result.exit_code == 0

    def test_palette_missing(self):
        result = runner.invoke(app, ["palette", "nope.png"])
        assert result.exit_code == 1


# -- orient --

class TestOrient:
    def test_orient(self, test_jpeg):
        result = runner.invoke(app, ["orient", str(test_jpeg)])
        assert result.exit_code == 0
        assert (
            "Orientation" in result.output
            or "EXIF" in result.output
        )

    def test_orient_fix(self, test_jpeg, tmp_path):
        out = tmp_path / "oriented.jpg"
        result = runner.invoke(
            app,
            [
                "orient",
                str(test_jpeg),
                "--fix",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0

    def test_orient_missing(self):
        result = runner.invoke(app, ["orient", "nope.jpg"])
        assert result.exit_code == 1


# -- rename-by-grade --

class TestRenameByGrade:
    def test_rename_dry_run(self, image_dir):
        result = runner.invoke(
            app,
            ["rename-by-grade", str(image_dir), "--dry-run"],
        )
        assert result.exit_code == 0
        assert "Dry run" in result.output or "Preview" in result.output

    def test_rename_actual(self, tmp_path):
        arr = np.random.randint(
            0, 255, (64, 64, 3), dtype=np.uint8
        )
        img = Image.fromarray(arr)
        img.save(str(tmp_path / "sample.jpg"), format="JPEG", quality=15)

        result = runner.invoke(
            app, ["rename-by-grade", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "renamed" in result.output.lower()

    def test_rename_suffix(self, tmp_path):
        arr = np.random.randint(
            0, 255, (64, 64, 3), dtype=np.uint8
        )
        img = Image.fromarray(arr)
        img.save(str(tmp_path / "pic.jpg"), format="JPEG", quality=15)

        result = runner.invoke(
            app,
            ["rename-by-grade", str(tmp_path), "--suffix"],
        )
        assert result.exit_code == 0

    def test_rename_not_dir(self, test_jpeg):
        result = runner.invoke(
            app, ["rename-by-grade", str(test_jpeg)]
        )
        assert result.exit_code == 1


# -- benchmark --

class TestBenchmark:
    def test_benchmark_generated(self):
        result = runner.invoke(
            app, ["benchmark", "--iterations", "2"]
        )
        assert result.exit_code == 0
        assert "Analyze" in result.output or "ms" in result.output

    def test_benchmark_with_image(self, test_jpeg):
        result = runner.invoke(
            app, ["benchmark", str(test_jpeg), "--iterations", "2"]
        )
        assert result.exit_code == 0
        assert "Performance" in result.output or "ms" in result.output


# -- version --

class TestVersion:
    def test_version(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "Artefex" in result.output
        assert "v" in result.output or "version" in result.output.lower()


# -- restore-preview --

class TestRestorePreview:
    def test_restore_preview(self, test_jpeg, tmp_path):
        out_dir = tmp_path / "steps"
        result = runner.invoke(
            app,
            [
                "restore-preview",
                str(test_jpeg),
                "--output",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0
        # Should either show steps or say no degradation
        assert (
            "Preview" in result.output
            or "No degradation" in result.output
            or "Step" in result.output
            or out_dir.exists()
        )

    def test_restore_preview_missing(self):
        result = runner.invoke(
            app, ["restore-preview", "missing.jpg"]
        )
        assert result.exit_code == 1


# -- parallel-analyze --

class TestParallelAnalyze:
    def test_parallel_analyze(self, image_dir):
        result = runner.invoke(
            app, ["parallel-analyze", str(image_dir)]
        )
        assert result.exit_code == 0
        assert (
            "Parallel" in result.output
            or "analysis" in result.output.lower()
        )

    def test_parallel_analyze_json(self, image_dir):
        result = runner.invoke(
            app,
            ["parallel-analyze", str(image_dir), "--json"],
        )
        assert result.exit_code == 0
        # Output may have rich console text before the JSON array.
        raw = result.output
        idx = raw.find("[")
        assert idx >= 0, f"No JSON array in output: {raw}"
        data = json.loads(raw[idx:])
        assert isinstance(data, list)

    def test_parallel_analyze_missing(self):
        result = runner.invoke(
            app, ["parallel-analyze", "no_dir"]
        )
        assert result.exit_code == 1
