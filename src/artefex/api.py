"""Public Python API for Artefex.

This module provides a clean, simple interface for using Artefex programmatically.

Usage:
    import artefex

    # Analyze an image
    result = artefex.analyze("photo.jpg")
    print(result.grade)          # "C"
    print(result.score)          # 45.2
    print(result.degradations)   # [Degradation(...), ...]

    # Restore an image
    info = artefex.restore("photo.jpg", "photo_clean.png")

    # Grade an image
    grade = artefex.grade("photo.jpg")
    print(grade["grade"])        # "C"

    # Compare two images
    metrics = artefex.compare("original.jpg", "restored.jpg")
    print(metrics["psnr"])       # 28.5

    # Find duplicates
    groups = artefex.find_duplicates("./photos/")

    # Generate heatmap
    stats = artefex.generate_heatmap("photo.jpg", "heatmap.png")

    # Detect platform
    platforms = artefex.detect_platform("photo.jpg")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypedDict, Union

import numpy as np

from artefex.models import AnalysisResult, Degradation


class _RestoreResultRequired(TypedDict):
    """Required fields for RestoreResult."""

    steps: list[str]
    used_neural: bool
    output_path: str


class RestoreResult(_RestoreResultRequired, total=False):
    """Result of a restore operation."""

    message: str  # present when no degradation detected


class GradeResult(TypedDict):
    """Result of a grade operation."""

    grade: str
    score: float
    description: str
    breakdown: list[dict]


class CompareResult(TypedDict):
    """Result of a compare operation."""

    mse: float
    psnr: float
    mean_diff_r: float
    mean_diff_g: float
    mean_diff_b: float
    pixels_changed_pct: float


class DuplicateGroup(TypedDict):
    """A group of duplicate images."""

    files: list[str]
    similarity: float


class HeatmapResult(TypedDict):
    """Result of a heatmap generation."""

    healthy_pct: float
    moderate_pct: float
    severe_pct: float
    mean_score: float
    worst_region: tuple


class PlatformMatch(TypedDict):
    """A platform detection match."""

    platform: str
    name: str
    confidence: float
    evidence: list[str]


@dataclass
class AnalyzeResult:
    """Friendly wrapper around analysis results."""

    file: str
    format: str
    dimensions: tuple[int, int]
    degradations: list[Degradation]
    grade: str
    score: float
    grade_description: str
    overall_severity: float
    _raw: AnalysisResult = field(repr=False, default_factory=AnalysisResult)

    def __len__(self) -> int:
        return len(self.degradations)

    def __bool__(self) -> bool:
        return len(self.degradations) > 0

    @property
    def is_clean(self) -> bool:
        return len(self.degradations) == 0

    @property
    def top_issue(self) -> Optional[str]:
        if self.degradations:
            return self.degradations[0].name
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "format": self.format,
            "dimensions": list(self.dimensions),
            "grade": self.grade,
            "score": self.score,
            "overall_severity": round(self.overall_severity, 3),
            "degradations": [
                {
                    "name": d.name,
                    "category": d.category,
                    "confidence": round(d.confidence, 3),
                    "severity": round(d.severity, 3),
                    "detail": d.detail,
                }
                for d in self.degradations
            ],
        }


def analyze(path: Union[str, Path]) -> AnalyzeResult:
    """Analyze an image for degradation.

    Args:
        path: Path to an image file.

    Returns:
        AnalyzeResult with degradations, grade, and score.
    """
    from artefex.analyze import DegradationAnalyzer
    from artefex.grade import compute_grade as _compute_grade

    file_path = Path(path)
    analyzer = DegradationAnalyzer()
    result = analyzer.analyze(file_path)
    grade_info = _compute_grade(result)

    return AnalyzeResult(
        file=str(file_path),
        format=result.file_format,
        dimensions=result.dimensions,
        degradations=result.degradations,
        grade=grade_info["grade"],
        score=grade_info["score"],
        grade_description=grade_info["description"],
        overall_severity=result.overall_severity,
        _raw=result,
    )


def restore(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    format: Optional[str] = None,
    use_neural: bool = True,
) -> RestoreResult:
    """Analyze and restore an image.

    Args:
        input_path: Path to degraded image.
        output_path: Where to save the restored image.
        format: Optional output format ("png", "jpg", "webp").
        use_neural: Whether to use neural models if available.

    Returns:
        Dict with steps taken and output path.
    """
    from artefex.analyze import DegradationAnalyzer
    from artefex.restore import RestorationPipeline

    in_path = Path(input_path)
    out_path = Path(output_path)

    analyzer = DegradationAnalyzer()
    pipeline = RestorationPipeline(use_neural=use_neural)

    result = analyzer.analyze(in_path)

    if not result.degradations:
        return {
            "steps": [],
            "used_neural": False,
            "output_path": str(out_path),
            "message": "No degradation detected",
        }

    return pipeline.restore(in_path, result, out_path, format=format)


def grade(path: Union[str, Path]) -> GradeResult:
    """Grade an image's quality on an A-F scale.

    Args:
        path: Path to image file.

    Returns:
        Dict with grade, score, description, and breakdown.
    """
    from artefex.analyze import DegradationAnalyzer
    from artefex.grade import compute_grade as _compute_grade

    analyzer = DegradationAnalyzer()
    result = analyzer.analyze(Path(path))
    return _compute_grade(result)


def compare(
    path1: Union[str, Path],
    path2: Union[str, Path],
) -> CompareResult:
    """Compare two images and return quality metrics.

    Args:
        path1: Path to first image.
        path2: Path to second image.

    Returns:
        Dict with mse, psnr, ssim, per-channel diffs, and change percentage.
    """
    from PIL import Image

    img1 = Image.open(Path(path1)).convert("RGB")
    img2 = Image.open(Path(path2)).convert("RGB")

    arr1 = np.array(img1, dtype=np.float64)
    arr2 = np.array(img2, dtype=np.float64)

    if arr1.shape != arr2.shape:
        img2 = img2.resize(img1.size, Image.LANCZOS)
        arr2 = np.array(img2, dtype=np.float64)

    mse = float(np.mean((arr1 - arr2) ** 2))
    psnr = float(10 * np.log10(255.0**2 / mse)) if mse > 0 else float("inf")

    diff = np.abs(arr1 - arr2)
    change_mask = np.any(diff > 10, axis=2)

    return {
        "mse": round(mse, 2),
        "psnr": round(psnr, 2),
        "mean_diff_r": round(float(diff[:, :, 0].mean()), 2),
        "mean_diff_g": round(float(diff[:, :, 1].mean()), 2),
        "mean_diff_b": round(float(diff[:, :, 2].mean()), 2),
        "pixels_changed_pct": round(float(change_mask.sum() / change_mask.size), 4),
    }


def find_duplicates(
    directory: Union[str, Path],
    threshold: float = 0.9,
    method: str = "phash",
) -> list[DuplicateGroup]:
    """Find duplicate images in a directory.

    Args:
        directory: Directory to scan.
        threshold: Similarity threshold (0-1).
        method: Hash method - "phash", "ahash", or "dhash".

    Returns:
        List of groups: [{"files": [...], "similarity": float}]
    """
    from artefex.similarity import find_duplicates as _find_dupes

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
    dir_path = Path(directory)
    files = sorted(f for f in dir_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)

    return _find_dupes(files, threshold=threshold, hash_fn=method)


def generate_heatmap(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    patch_size: int = 32,
) -> HeatmapResult:
    """Generate a spatial degradation heatmap.

    Args:
        input_path: Image to analyze.
        output_path: Where to save the heatmap overlay.
        patch_size: Size of analysis patches in pixels.

    Returns:
        Dict with spatial stats (healthy_pct, moderate_pct, severe_pct, etc).
    """
    from artefex.heatmap import generate_heatmap as _gen_heatmap

    return _gen_heatmap(Path(input_path), Path(output_path), patch_size=patch_size)


def detect_platform(path: Union[str, Path]) -> list[PlatformMatch]:
    """Detect which social media platform(s) likely processed an image.

    Args:
        path: Path to image file.

    Returns:
        List of matches: [{"platform": str, "name": str, "confidence": float, "evidence": [...]}]
    """
    from artefex.fingerprint import PlatformFingerprinter

    fp = PlatformFingerprinter()
    return fp.fingerprint(Path(path))
