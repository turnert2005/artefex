"""Parallel batch processing for faster analysis of large image collections."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional



def _analyze_single(file_path_str: str) -> dict:
    """Analyze a single file (picklable for multiprocessing)."""
    from artefex.analyze import DegradationAnalyzer
    from artefex.grade import compute_grade

    path = Path(file_path_str)
    analyzer = DegradationAnalyzer()
    result = analyzer.analyze(path)
    grade = compute_grade(result)

    return {
        "file": path.name,
        "path": str(path),
        "format": result.file_format,
        "dimensions": list(result.dimensions),
        "grade": grade["grade"],
        "score": grade["score"],
        "degradation_count": len(result.degradations),
        "overall_severity": round(result.overall_severity, 3),
        "top_issue": result.degradations[0].name if result.degradations else None,
        "degradations": [
            {
                "name": d.name,
                "severity": round(d.severity, 3),
                "confidence": round(d.confidence, 3),
                "category": d.category,
            }
            for d in result.degradations
        ],
    }


def parallel_analyze(
    files: list[Path],
    max_workers: Optional[int] = None,
    on_progress: Optional[callable] = None,
) -> list[dict]:
    """Analyze multiple images in parallel using multiprocessing.

    Args:
        files: List of image paths to analyze.
        max_workers: Number of worker processes (None = auto).
        on_progress: Callback(completed, total) for progress updates.

    Returns:
        List of result dicts, one per file.
    """
    file_strs = [str(f) for f in files]
    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(_analyze_single, f): f for f in file_strs
        }

        for future in as_completed(future_to_file):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                file_str = future_to_file[future]
                results.append({
                    "file": Path(file_str).name,
                    "path": file_str,
                    "error": str(e),
                })

            completed += 1
            if on_progress:
                on_progress(completed, len(files))

    # Sort by original file order
    file_order = {str(f): i for i, f in enumerate(files)}
    results.sort(key=lambda r: file_order.get(r.get("path", ""), 0))

    return results
