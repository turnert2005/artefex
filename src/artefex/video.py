"""Video analysis and restoration - frame-by-frame with temporal coherence."""

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from artefex.analyze import DegradationAnalyzer
from artefex.models import AnalysisResult, Degradation
from artefex.restore import RestorationPipeline

logger = logging.getLogger(__name__)

SUPPORTED_CODECS = ("mp4v", "avc1", "XVID")


def _has_ffmpeg() -> bool:
    """Check whether ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


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
        sample_count = min(self.sample_count, frame_count)
        sample_indices = np.linspace(
            0, frame_count - 1, sample_count, dtype=int
        )
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

    def __init__(
        self,
        use_neural: bool = True,
        temporal_strength: float = 0.15,
    ):
        self.analyzer = DegradationAnalyzer()
        self.pipeline = RestorationPipeline(use_neural=use_neural)
        self.temporal_strength = max(0.0, min(1.0, temporal_strength))

    def restore(
        self,
        video_path: Path,
        output_path: Path,
        analysis: Optional[VideoAnalysisResult] = None,
        on_progress: Optional[callable] = None,
        codec: str = "mp4v",
        quality: int = 95,
    ) -> dict:
        cv2 = _check_cv2()

        if codec not in SUPPORTED_CODECS:
            raise ValueError(
                f"Unsupported codec '{codec}'. "
                f"Choose from: {', '.join(SUPPORTED_CODECS)}"
            )
        quality = max(0, min(100, quality))

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
            # Present in >30% of sampled frames
            if stats["frequency"] > 0.3:
                representative_degradations.append(Degradation(
                    name=name,
                    confidence=stats["avg_confidence"],
                    severity=stats["avg_severity"],
                    category=stats["category"],
                ))

        if not representative_degradations:
            cap.release()
            return {
                "frames_processed": 0,
                "message": "No consistent degradation found",
            }

        representative_result = AnalysisResult(
            degradations=representative_degradations,
        )

        # Extract audio from the original video (if ffmpeg is available)
        audio_tmp_path = None
        has_audio = False
        if _has_ffmpeg():
            audio_tmp_path = Path(
                tempfile.mktemp(suffix=".aac")
            )
            has_audio = _extract_audio(
                video_path, audio_tmp_path
            )
        else:
            logger.info(
                "ffmpeg not found on PATH - skipping audio passthrough"
            )

        # Write restored frames to a temp file when we need to mux audio.
        # Otherwise write directly to the final output path.
        video_write_path = output_path
        if has_audio:
            video_write_path = Path(
                tempfile.mktemp(suffix=output_path.suffix)
            )

        # Set up video writer with the requested codec
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(
            str(video_write_path), fourcc, fps, (width, height)
        )

        frames_processed = 0
        prev_frame: Optional[np.ndarray] = None

        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Restore via pipeline using the representative analysis
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as tmp_in:
                tmp_in_path = Path(tmp_in.name)
                pil_img.save(tmp_in_path)

            tmp_out_path = tmp_in_path.with_stem(
                tmp_in_path.stem + "_out"
            )

            try:
                self.pipeline.restore(
                    tmp_in_path,
                    representative_result,
                    tmp_out_path,
                )
                restored = Image.open(tmp_out_path).convert("RGB")

                # Ensure same size
                if restored.size != (width, height):
                    restored = restored.resize(
                        (width, height), Image.LANCZOS
                    )

                restored_arr = np.array(restored, dtype=np.float32)

                # Temporal coherence - blend with previous restored
                # frame to reduce flickering between frames
                if (
                    prev_frame is not None
                    and self.temporal_strength > 0
                ):
                    blended = (
                        (1.0 - self.temporal_strength) * restored_arr
                        + self.temporal_strength * prev_frame
                    )
                    restored_arr = blended

                prev_frame = restored_arr.copy()

                restored_bgr = cv2.cvtColor(
                    restored_arr.astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                )
                out.write(restored_bgr)
                frames_processed += 1
            finally:
                tmp_in_path.unlink(missing_ok=True)
                tmp_out_path.unlink(missing_ok=True)

            if on_progress and frame_idx % 10 == 0:
                on_progress(frame_idx + 1, frame_count)

        cap.release()
        out.release()

        # Mux audio back into the restored video
        if has_audio and audio_tmp_path is not None:
            _mux_audio(video_write_path, audio_tmp_path, output_path)
            video_write_path.unlink(missing_ok=True)

        # Clean up audio temp file
        if audio_tmp_path is not None:
            audio_tmp_path.unlink(missing_ok=True)

        return {
            "frames_processed": frames_processed,
            "output_path": str(output_path),
            "degradations_fixed": [
                d.name for d in representative_degradations
            ],
            "codec": codec,
            "quality": quality,
            "audio_preserved": has_audio,
        }


def _extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extract audio from a video file using ffmpeg.

    Returns True if audio was successfully extracted, False otherwise.
    """
    try:
        # Check if the video actually contains an audio stream
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if not probe.stdout.strip():
            logger.info("No audio stream found in source video")
            return False

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vn",
                "-acodec", "copy",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.warning(
                "ffmpeg audio extraction failed: %s",
                result.stderr[:200],
            )
            return False
        return audio_path.exists() and audio_path.stat().st_size > 0
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("Audio extraction skipped: %s", exc)
        return False


def _mux_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> bool:
    """Combine a video file (no audio) with an audio file using ffmpeg.

    Returns True on success, False otherwise.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.warning(
                "ffmpeg audio muxing failed: %s",
                result.stderr[:200],
            )
            # Fall back - just copy the video without audio
            if video_path != output_path:
                shutil.copy2(str(video_path), str(output_path))
            return False
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("Audio muxing skipped: %s", exc)
        if video_path != output_path:
            shutil.copy2(str(video_path), str(output_path))
        return False
