"""Basic usage of the Artefex Python API."""

from pathlib import Path

from artefex.analyze import DegradationAnalyzer
from artefex.restore import RestorationPipeline
from artefex.report import render_report


def analyze_image(image_path: str):
    """Analyze a single image and print results."""
    analyzer = DegradationAnalyzer()
    result = analyzer.analyze(Path(image_path))

    print(f"File: {result.file_path}")
    print(f"Format: {result.file_format}")
    print(f"Dimensions: {result.dimensions}")
    print(f"Overall severity: {result.overall_severity:.0%}")
    print()

    for d in result.degradations:
        print(f"  {d.name}: confidence={d.confidence:.0%}, severity={d.severity:.0%}")
        print(f"    {d.detail}")
    print()


def restore_image(image_path: str, output_path: str):
    """Analyze and restore an image."""
    analyzer = DegradationAnalyzer()
    pipeline = RestorationPipeline()

    result = analyzer.analyze(Path(image_path))

    if not result.degradations:
        print("No degradation detected.")
        return

    info = pipeline.restore(Path(image_path), result, Path(output_path))

    print("Restoration steps:")
    for step in info["steps"]:
        print(f"  {step}")
    print(f"\nSaved to: {info['output_path']}")


def generate_report(image_path: str):
    """Generate a forensic report."""
    analyzer = DegradationAnalyzer()
    result = analyzer.analyze(Path(image_path))
    report = render_report(Path(image_path), result)
    print(report)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python basic_usage.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    print("=== Analysis ===\n")
    analyze_image(path)

    print("=== Restoration ===\n")
    out = Path(path).with_stem(Path(path).stem + "_restored")
    restore_image(path, str(out))
