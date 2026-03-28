"""Tests for advanced features: steganography, similarity, heatmap, audit."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from artefex.detect_stego import SteganographyDetector
from artefex.detect_aigen import AIGeneratedDetector
from artefex.detect_camera import CameraIdentifier
from artefex.similarity import phash, ahash, dhash, hamming_distance, similarity_score, find_duplicates
from artefex.heatmap import generate_heatmap
from artefex.accessibility import check_accessibility, simulate_cvd


def _make_image(w=128, h=128) -> Image.Image:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _save_temp(img, suffix=".png") -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    img.save(tmp.name)
    return Path(tmp.name)


def _cleanup(*paths):
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except PermissionError:
            pass


# Steganography detection tests

def test_stego_detector_returns_none_or_degradation():
    img = _make_image()
    arr = np.array(img)
    det = SteganographyDetector()
    result = det.detect(img, arr)
    # Should return None or a Degradation
    assert result is None or result.name == "Steganography Detected"


def test_stego_detector_small_image():
    img = _make_image(16, 16)
    arr = np.array(img)
    det = SteganographyDetector()
    result = det.detect(img, arr)
    assert result is None


# AI detection tests

def test_ai_detector_returns_none_or_degradation():
    img = _make_image()
    arr = np.array(img)
    det = AIGeneratedDetector()
    result = det.detect(img, arr)
    assert result is None or result.name == "AI-Generated Content"


def test_ai_detector_small_image():
    img = _make_image(32, 32)
    arr = np.array(img)
    det = AIGeneratedDetector()
    result = det.detect(img, arr)
    assert result is None


# Similarity tests

def test_phash_identical_images():
    img = _make_image()
    h1 = phash(img)
    h2 = phash(img)
    assert h1 == h2


def test_ahash_identical_images():
    img = _make_image()
    h1 = ahash(img)
    h2 = ahash(img)
    assert h1 == h2


def test_dhash_identical_images():
    img = _make_image()
    h1 = dhash(img)
    h2 = dhash(img)
    assert h1 == h2


def test_similarity_identical():
    img = _make_image()
    h = phash(img)
    assert similarity_score(h, h) == 1.0


def test_hamming_distance_identical():
    assert hamming_distance(0xFF, 0xFF) == 0


def test_hamming_distance_different():
    assert hamming_distance(0x00, 0xFF) == 8


def test_find_duplicates_with_copies():
    img = _make_image()
    p1 = _save_temp(img)
    p2 = _save_temp(img)
    p3 = _save_temp(_make_image(200, 200))  # Different image

    groups = find_duplicates([p1, p2, p3], threshold=0.9)

    # p1 and p2 should be grouped together
    assert len(groups) >= 1
    found = False
    for g in groups:
        if len(g["files"]) >= 2:
            found = True
    assert found

    _cleanup(p1, p2, p3)


# Heatmap tests

def test_heatmap_generates_output():
    img = _make_image(256, 256)
    path = _save_temp(img, ".jpg")
    out_path = path.with_stem(path.stem + "_heatmap").with_suffix(".png")

    stats = generate_heatmap(path, out_path)

    assert out_path.exists()
    assert "healthy_pct" in stats
    assert "severe_pct" in stats
    assert stats["patch_size"] == 32

    _cleanup(path, out_path)


# Camera identification tests

def test_camera_identifier_returns_list():
    img = _make_image(256, 256)
    arr = np.array(img)
    ci = CameraIdentifier()
    results = ci.identify(img, arr)
    assert isinstance(results, list)
    for r in results:
        assert "device" in r
        assert "confidence" in r


def test_camera_identifier_small_image():
    img = _make_image(32, 32)
    arr = np.array(img)
    ci = CameraIdentifier()
    results = ci.identify(img, arr)
    assert results == []


# Accessibility tests

def test_accessibility_check():
    img = _make_image(128, 128)
    result = check_accessibility(img)
    assert "information_loss" in result
    assert "contrast_ratio" in result
    assert "wcag_aa_pass" in result
    assert "protanopia" in result["information_loss"]


def test_cvd_simulation_output_size():
    img = _make_image(64, 64)
    simulated = simulate_cvd(img, "protanopia")
    assert simulated.size == img.size


def test_cvd_simulation_all_types():
    img = _make_image(64, 64)
    for cvd_type in ["protanopia", "deuteranopia", "tritanopia", "achromatopsia"]:
        simulated = simulate_cvd(img, cvd_type)
        assert simulated.size == img.size
