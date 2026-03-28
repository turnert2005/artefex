"""Data models for degradation analysis results."""

from dataclasses import dataclass, field


@dataclass
class Degradation:
    """A single detected degradation in an image."""

    name: str
    confidence: float  # 0.0 - 1.0
    severity: float  # 0.0 - 1.0
    detail: str = ""
    category: str = ""  # e.g. "compression", "resolution", "color", "artifact"


@dataclass
class AnalysisResult:
    """Complete analysis of an image's degradation chain."""

    file_path: str = ""
    file_format: str = ""
    dimensions: tuple[int, int] = (0, 0)
    degradations: list[Degradation] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def overall_severity(self) -> float:
        if not self.degradations:
            return 0.0
        return max(d.severity for d in self.degradations)

    @property
    def degradation_count(self) -> int:
        return len(self.degradations)
