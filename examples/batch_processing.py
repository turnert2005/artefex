"""Batch processing example using the Artefex Python API."""

import json
from pathlib import Path

from artefex.analyze import DegradationAnalyzer
from artefex.restore import RestorationPipeline


def batch_analyze(directory: str) -> list[dict]:
    """Analyze all images in a directory and return structured results."""
    analyzer = DegradationAnalyzer()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    results = []
    dir_path = Path(directory)

    for file in sorted(dir_path.iterdir()):
        if file.suffix.lower() not in image_exts:
            continue

        result = analyzer.analyze(file)
        results.append({
            "file": file.name,
            "degradations": len(result.degradations),
            "overall_severity": round(result.overall_severity, 3),
            "issues": [
                {"name": d.name, "severity": round(d.severity, 3)}
                for d in result.degradations
            ],
        })

    return results


def batch_restore(directory: str, output_dir: str):
    """Restore all degraded images in a directory."""
    analyzer = DegradationAnalyzer()
    pipeline = RestorationPipeline()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for file in sorted(Path(directory).iterdir()):
        if file.suffix.lower() not in image_exts:
            continue

        result = analyzer.analyze(file)
        if not result.degradations:
            print(f"  {file.name}: clean, skipping")
            continue

        out_path = out / f"{file.stem}_restored{file.suffix}"
        info = pipeline.restore(file, result, out_path)
        print(f"  {file.name}: {len(info['steps'])} fixes -> {out_path.name}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python batch_processing.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    print("=== Batch Analysis ===\n")
    results = batch_analyze(directory)
    print(json.dumps(results, indent=2))

    print("\n=== Batch Restore ===\n")
    batch_restore(directory, str(Path(directory) / "restored"))
