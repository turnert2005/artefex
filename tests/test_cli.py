"""Tests for the CLI interface."""

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from artefex.cli import app

runner = CliRunner()


def _make_test_jpeg(quality=20) -> Path:
    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
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


def test_cli_analyze_single():
    path = _make_test_jpeg()
    result = runner.invoke(app, ["analyze", str(path)])
    assert result.exit_code == 0
    assert "Analyzing" in result.output
    _cleanup(path)


def test_cli_analyze_json():
    path = _make_test_jpeg()
    result = runner.invoke(app, ["analyze", str(path), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "degradations" in data
    assert "dimensions" in data
    _cleanup(path)


def test_cli_analyze_missing_file():
    result = runner.invoke(app, ["analyze", "nonexistent.jpg"])
    assert result.exit_code == 1


def test_cli_report():
    path = _make_test_jpeg()
    result = runner.invoke(app, ["report", str(path)])
    assert result.exit_code == 0
    assert "ARTEFEX FORENSIC REPORT" in result.output
    _cleanup(path)


def test_cli_restore():
    path = _make_test_jpeg(quality=10)
    out_path = path.with_stem(path.stem + "_restored")
    result = runner.invoke(app, ["restore", str(path)])
    assert result.exit_code == 0
    assert "Restored" in result.output or "No degradation" in result.output
    _cleanup(path, out_path)


def test_cli_models_list():
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "Artefex Neural Models" in result.output


def test_cli_plugins():
    result = runner.invoke(app, ["plugins"])
    assert result.exit_code == 0
    assert "plugin" in result.output.lower()


def test_cli_no_args_shows_help():
    result = runner.invoke(app, [])
    assert "Usage" in result.output or "Commands" in result.output or "artefex" in result.output
