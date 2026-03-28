"""Animated GIF/APNG analysis - frame-by-frame degradation detection."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from artefex.analyze import DegradationAnalyzer
from artefex.models import AnalysisResult, Degradation


@dataclass
class GifAnalysisResult:
    """Analysis of an animated image."""

    file_path: str = ""
    frame_count: int = 0
    dimensions: tuple[int, int] = (0, 0)
    is_animated: bool = False
    loop_count: int = 0
    total_duration_ms: int = 0
    avg_frame_duration_ms: float = 0.0
    color_palette_size: int = 0
    # Per-frame analysis
    frame_results: list[AnalysisResult] = field(default_factory=list)
    # Temporal analysis
    frame_similarity: list[float] = field(default_factory=list)
    degradation_summary: dict[str, dict] = field(default_factory=dict)


class GifAnalyzer:
    """Analyzes animated GIF/APNG images frame by frame."""

    def __init__(self, max_frames: int = 50):
        self.max_frames = max_frames
        self.analyzer = DegradationAnalyzer()

    def analyze(self, file_path: Path, on_progress=None) -> GifAnalysisResult:
        img = Image.open(file_path)

        is_animated = hasattr(img, "n_frames") and img.n_frames > 1
        n_frames = getattr(img, "n_frames", 1)

        result = GifAnalysisResult(
            file_path=str(file_path),
            frame_count=n_frames,
            dimensions=img.size,
            is_animated=is_animated,
            loop_count=img.info.get("loop", 0),
        )

        if not is_animated:
            # Single frame - just analyze normally
            frame_result = self.analyzer.analyze(file_path)
            result.frame_results = [frame_result]
            self._aggregate(result)
            return result

        # Analyze sampled frames
        sample_indices = np.linspace(0, n_frames - 1, min(self.max_frames, n_frames), dtype=int)

        total_duration = 0
        prev_arr = None
        frame_similarities = []

        import tempfile

        for idx_pos, frame_idx in enumerate(sample_indices):
            img.seek(frame_idx)
            frame = img.convert("RGB")

            # Get frame duration
            duration = img.info.get("duration", 100)
            total_duration += duration

            # Save frame to temp for analysis
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                frame.save(tmp_path)

            try:
                frame_analysis = self.analyzer.analyze(tmp_path)
                frame_analysis.file_path = f"frame_{frame_idx}"
                result.frame_results.append(frame_analysis)
            finally:
                tmp_path.unlink(missing_ok=True)

            # Compute frame-to-frame similarity
            curr_arr = np.array(frame, dtype=np.float64)
            if prev_arr is not None and prev_arr.shape == curr_arr.shape:
                mse = np.mean((curr_arr - prev_arr) ** 2)
                similarity = max(0, 1.0 - mse / (255.0 ** 2))
                frame_similarities.append(similarity)
            prev_arr = curr_arr

            if on_progress:
                on_progress(idx_pos + 1, len(sample_indices))

        result.total_duration_ms = total_duration
        result.avg_frame_duration_ms = total_duration / n_frames if n_frames > 0 else 0
        result.frame_similarity = frame_similarities

        # Check palette
        try:
            img.seek(0)
            if img.mode == "P":
                palette = img.getpalette()
                if palette:
                    result.color_palette_size = len(palette) // 3
        except Exception:
            pass

        self._aggregate(result)
        return result

    def _aggregate(self, result: GifAnalysisResult):
        """Aggregate per-frame results into summary."""
        degradation_counts: dict[str, list[Degradation]] = {}

        for fr in result.frame_results:
            for d in fr.degradations:
                if d.name not in degradation_counts:
                    degradation_counts[d.name] = []
                degradation_counts[d.name].append(d)

        for name, dlist in degradation_counts.items():
            result.degradation_summary[name] = {
                "count": len(dlist),
                "frequency": len(dlist) / len(result.frame_results) if result.frame_results else 0,
                "avg_severity": float(np.mean([d.severity for d in dlist])),
                "avg_confidence": float(np.mean([d.confidence for d in dlist])),
                "category": dlist[0].category,
            }
