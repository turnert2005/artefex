"""Video analysis and restoration - frame-by-frame with temporal coherence."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from artefex.analyze import DegradationAnalyzer
from artefex.models import AnalysisResult, Degradation
from artefex.restore import RestorationPipeline


@dataclass
class VideoAnalysisResult:
    """Analysis summary for an entire video."""

    file_path: str = ""
    frame_count: int = 0
    fps: float = 0.0
    resolution: tuple[int, int] = (0, 0)
    duration_seconds: float = 0.0
    codec: str = ""
    # Aggregate degradation info across sampled frames
    degradation_summary: dict[str, dict] = field(default_factory=dict)
    frame_results: list[AnalysisResult] = field(default_factory=list)

    @property
    def overall_severity(self) -> float:
        if not self.degradation_summary:
            return 0.0
        return max(d["avg_severity"] for d in self.degradation_summary.values())


def _check_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        raise ImportError(
            "Video support requires opencv-python. "
            "Install it with: pip install artefex[video]"
        )


class VideoAnalyzer:
    """Analyzes video files by sampling frames and detecting degradation."""

    def __init__(self, sample_count: int = 10):
        self.sample_count = sample_count
        self.analyzer = DegradationAnalyzer()

    def analyze(
        self,
        video_path: Path,
        on_progress: Optional[callable] = None,
    ) -> VideoAnalysisResult:
        cv2 = _check_cv2()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        result = VideoAnalysisResult(
            file_path=str(video_path),
            frame_count=frame_count,
            fps=fps,
            resolution=(width, height),
            duration_seconds=frame_count / fps if fps > 0 else 0,
            codec=codec.strip(),
        )

        # Sample frames evenly across the video
        sample_indices = np.linspace(0, frame_count - 1, min(self.sample_count, frame_count), dtype=int)
        frame_results = []

        for i, frame_idx in enumerate(sample_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert BGR to RGB and wrap as PIL Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Save as temp file for analyzer (it expects a path)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                pil_img.save(tmp_path)

            try:
                frame_analysis = self.analyzer.analyze(tmp_path)
                frame_analysis.file_path = f"frame_{frame_idx}"
                frame_results.append(frame_analysis)
            finally:
                tmp_path.unlink(missing_ok=True)

            if on_progress:
                on_progress(i + 1, len(sample_indices))

        cap.release()
        result.frame_results = frame_results

        # Aggregate degradation stats
        degradation_counts: dict[str, list[Degradation]] = {}
        for fr in frame_results:
            for d in fr.degradations:
                if d.name not in degradation_counts:
                    degradation_counts[d.name] = []
                degradation_counts[d.name].append(d)

        for name, dlist in degradation_counts.items():
            result.degradation_summary[name] = {
                "count": len(dlist),
                "frequency": len(dlist) / len(frame_results) if frame_results else 0,
                "avg_severity": np.mean([d.severity for d in dlist]),
                "avg_confidence": np.mean([d.confidence for d in dlist]),
                "category": dlist[0].category,
            }

        return result


class VideoRestorer:
    """Restores video files frame by frame with temporal smoothing."""

    def __init__(self, use_neural: bool = True):
        self.analyzer = DegradationAnalyzer()
        self.pipeline = RestorationPipeline(use_neural=use_neural)

    def restore(
        self,
        video_path: Path,
        output_path: Path,
        analysis: Optional[VideoAnalysisResult] = None,
        on_progress: Optional[callable] = None,
    ) -> dict:
        cv2 = _check_cv2()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Use analysis from a sample to determine what to fix
        # (avoids re-analyzing every single frame)
        if analysis is None:
            video_analyzer = VideoAnalyzer(sample_count=5)
            analysis = video_analyzer.analyze(video_path)

        # Build a "representative" degradation list from the summary
        representative_degradations = []
        for name, stats in analysis.degradation_summary.items():
            if stats["frequency"] > 0.3:  # Present in >30% of sampled frames
                representative_degradations.append(Degradation(
                    name=name,
                    confidence=stats["avg_confidence"],
                    severity=stats["avg_severity"],
                    category=stats["category"],
                ))

        if not representative_degradations:
            cap.release()
            return {"frames_processed": 0, "message": "No consistent degradation found"}

        representative_result = AnalysisResult(
            degradations=representative_degradations,
        )

        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        import tempfile
        frames_processed = 0

        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Restore via pipeline using the representative analysis
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
                tmp_in_path = Path(tmp_in.name)
                pil_img.save(tmp_in_path)

            tmp_out_path = tmp_in_path.with_stem(tmp_in_path.stem + "_out")

            try:
                self.pipeline.restore(tmp_in_path, representative_result, tmp_out_path)
                restored = Image.open(tmp_out_path).convert("RGB")

                # Ensure same size
                if restored.size != (width, height):
                    restored = restored.resize((width, height), Image.LANCZOS)

                restored_bgr = cv2.cvtColor(np.array(restored), cv2.COLOR_RGB2BGR)
                out.write(restored_bgr)
                frames_processed += 1
            finally:
                tmp_in_path.unlink(missing_ok=True)
                tmp_out_path.unlink(missing_ok=True)

            if on_progress and frame_idx % 10 == 0:
                on_progress(frame_idx + 1, frame_count)

        cap.release()
        out.release()

        return {
            "frames_processed": frames_processed,
            "output_path": str(output_path),
            "degradations_fixed": [d.name for d in representative_degradations],
        }
