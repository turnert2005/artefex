"""Forensic report generation."""

from pathlib import Path

from artefex.models import AnalysisResult


def render_report(file_path: Path, result: AnalysisResult) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  ARTEFEX FORENSIC REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  File:       {file_path.name}")
    lines.append(f"  Path:       {file_path}")
    lines.append(f"  Format:     {result.file_format}")
    lines.append(f"  Dimensions: {result.dimensions[0]}x{result.dimensions[1]}")
    lines.append(f"  Mode:       {result.metadata.get('mode', 'unknown')}")
    lines.append("")

    if result.metadata.get("dpi"):
        lines.append(f"  DPI:        {result.metadata['dpi']}")
    if result.metadata.get("jfif_version"):
        lines.append(f"  JFIF:       {result.metadata['jfif_version']}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("  DEGRADATION ANALYSIS")
    lines.append("-" * 60)
    lines.append("")

    if not result.degradations:
        lines.append("  No degradation detected. Image appears clean.")
    else:
        lines.append(f"  Overall severity: {result.overall_severity:.0%}")
        lines.append(f"  Degradations found: {result.degradation_count}")
        lines.append("")

        for i, d in enumerate(result.degradations, 1):
            lines.append(f"  [{i}] {d.name}")
            lines.append(f"      Category:   {d.category}")
            lines.append(f"      Confidence: {d.confidence:.0%}")
            lines.append(f"      Severity:   {d.severity:.0%}")
            lines.append(f"      Detail:     {d.detail}")
            lines.append("")

    lines.append("-" * 60)
    lines.append("  RESTORATION RECOMMENDATION")
    lines.append("-" * 60)
    lines.append("")

    if not result.degradations:
        lines.append("  No restoration needed.")
    else:
        lines.append("  Recommended restoration pipeline:")
        lines.append("")
        for i, d in enumerate(result.degradations, 1):
            action = _recommend_action(d.name)
            lines.append(f"  Step {i}: {action}")
        lines.append("")
        lines.append("  Run `artefex restore <file>` to apply.")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def _recommend_action(degradation_name: str) -> str:
    actions = {
        "JPEG Compression": "Apply deblocking filter to reduce 8x8 block artifacts",
        "Resolution Loss / Upscaling": "Apply neural super-resolution to recover detail",
        "Color Shift": "Normalize color channels and correct white balance",
        "Screenshot Artifacts": "Crop solid borders and remove UI remnants",
        "Multiple Re-compressions": "Apply heavy deblocking + detail reconstruction",
        "Noise": "Apply adaptive denoising while preserving edges",
        "Watermark": "Attempt watermark removal via inpainting",
        "EXIF Metadata Stripped": "Flag as re-processed (metadata cannot be recovered)",
        "Platform Processing": "Image was processed by a social media platform (informational)",
        "AI-Generated Content": "Image shows signs of AI generation (informational)",
        "Steganography Detected": "Image may contain hidden embedded data (informational)",
        "Device Identification": "Device type identified from sensor noise (informational)",
        "Copy-Move Forgery": "Cloned regions detected - image may have been manipulated",
    }
    return actions.get(degradation_name, f"Address {degradation_name}")
