"""Quality gate - enforce image quality standards in CI/CD pipelines.

Use as a pre-commit hook or CI step to reject images that don't meet
quality thresholds. Exits with non-zero status if any image fails.

Usage as pre-commit hook (.pre-commit-config.yaml):

    repos:
      - repo: https://github.com/turnert2005/artefex
        rev: v0.1.0
        hooks:
          - id: artefex-quality-gate
            args: ['--min-grade', 'C', '--max-severity', '0.7']

Usage in CI:
    artefex gate ./assets/ --min-grade C --min-score 40
"""

import sys
from pathlib import Path

from artefex.analyze import DegradationAnalyzer
from artefex.grade import compute_grade

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

GRADE_ORDER = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}


def run_quality_gate(
    paths: list[Path],
    min_grade: str = "D",
    min_score: float = 0.0,
    max_severity: float = 1.0,
    block_ai: bool = False,
    block_stego: bool = False,
) -> list[dict]:
    """Run quality gate on a list of image files.

    Returns list of failures. Empty list = all passed.
    """
    analyzer = DegradationAnalyzer()
    failures = []

    min_grade_val = GRADE_ORDER.get(min_grade.upper(), 0)

    for path in paths:
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        result = analyzer.analyze(path)
        grade_info = compute_grade(result)
        grade_val = GRADE_ORDER.get(grade_info["grade"], 0)

        reasons = []

        if grade_val < min_grade_val:
            reasons.append(f"grade {grade_info['grade']} below minimum {min_grade}")

        if grade_info["score"] < min_score:
            reasons.append(f"score {grade_info['score']} below minimum {min_score}")

        if result.overall_severity > max_severity:
            reasons.append(f"severity {result.overall_severity:.0%} exceeds maximum {max_severity:.0%}")

        if block_ai:
            for d in result.degradations:
                if d.name == "AI-Generated Content" and d.confidence > 0.5:
                    reasons.append(f"AI-generated content detected ({d.confidence:.0%} confidence)")

        if block_stego:
            for d in result.degradations:
                if d.name == "Steganography Detected" and d.confidence > 0.5:
                    reasons.append(f"steganography detected ({d.confidence:.0%} confidence)")

        if reasons:
            failures.append({
                "file": str(path),
                "grade": grade_info["grade"],
                "score": grade_info["score"],
                "reasons": reasons,
            })

    return failures
