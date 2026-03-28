"""Image quality grading system - assigns A-F grades based on analysis."""

from artefex.models import AnalysisResult


GRADE_THRESHOLDS = {
    "A": (0.0, 0.05),   # Pristine - no meaningful degradation
    "B": (0.05, 0.2),   # Good - minor artifacts
    "C": (0.2, 0.4),    # Fair - noticeable degradation
    "D": (0.4, 0.7),    # Poor - significant degradation
    "F": (0.7, 1.0),    # Failed - heavy degradation
}

GRADE_DESCRIPTIONS = {
    "A": "Pristine quality - no meaningful degradation detected",
    "B": "Good quality - minor artifacts present but image is largely intact",
    "C": "Fair quality - noticeable degradation that may affect clarity",
    "D": "Poor quality - significant degradation across multiple dimensions",
    "F": "Severely degraded - heavy artifacts, multiple compression cycles likely",
}

GRADE_COLORS = {
    "A": "green",
    "B": "green",
    "C": "yellow",
    "D": "red",
    "F": "red",
}


def compute_grade(result: AnalysisResult) -> dict:
    """Compute a quality grade from analysis results.

    Returns dict with:
        grade: str (A-F)
        score: float (0-100, higher is better)
        description: str
        breakdown: list[dict] (per-degradation scoring)
    """
    if not result.degradations:
        return {
            "grade": "A",
            "score": 100.0,
            "description": GRADE_DESCRIPTIONS["A"],
            "color": GRADE_COLORS["A"],
            "breakdown": [],
        }

    # Compute weighted severity score
    # More degradations = worse, higher severity = worse, higher confidence = more weight
    total_penalty = 0.0
    breakdown = []

    for d in result.degradations:
        # Weight by confidence - uncertain detections penalize less
        penalty = d.severity * d.confidence
        total_penalty += penalty

        breakdown.append({
            "name": d.name,
            "penalty": round(penalty * 100, 1),
            "severity": round(d.severity * 100, 1),
            "confidence": round(d.confidence * 100, 1),
        })

    # Normalize: cap at 1.0 with diminishing returns for stacking
    # Use sqrt to compress multiple degradations
    import math
    normalized = min(1.0, math.sqrt(total_penalty / max(len(result.degradations), 1)))

    # Also factor in count - more degradations is worse
    count_factor = min(1.0, len(result.degradations) / 5)
    final_severity = normalized * 0.7 + count_factor * 0.3
    final_severity = min(1.0, final_severity)

    score = max(0, (1.0 - final_severity) * 100)

    # Determine grade
    grade = "F"
    for g, (low, high) in GRADE_THRESHOLDS.items():
        if low <= final_severity < high:
            grade = g
            break

    return {
        "grade": grade,
        "score": round(score, 1),
        "description": GRADE_DESCRIPTIONS[grade],
        "color": GRADE_COLORS[grade],
        "breakdown": breakdown,
    }
