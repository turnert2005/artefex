"""artefex: Neural forensic restoration for degraded media."""

__version__ = "1.0.0"

# Public API - import the most useful things at the top level
from artefex.api import (
    analyze,
    restore,
    grade,
    compare,
    find_duplicates,
    generate_heatmap,
    detect_platform,
)

__all__ = [
    "__version__",
    "analyze",
    "restore",
    "grade",
    "compare",
    "find_duplicates",
    "generate_heatmap",
    "detect_platform",
]
