"""CLI entry point for artefex."""

import json as json_mod
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from artefex.analyze import DegradationAnalyzer
from artefex.report import render_report

app = typer.Typer(
    name="artefex",
    help="Neural forensic restoration - diagnose and reverse media degradation chains.",
    no_args_is_help=True,
)
console = Console()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def _collect_images(path: Path) -> list[Path]:
    """Collect image files from a path (file or directory)."""
    if path.is_file():
        return [path]
    if path.is_dir():
        files = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(path.glob(f"*{ext}"))
            files.extend(path.glob(f"*{ext.upper()}"))
        return sorted(set(files))
    return []


def _print_analysis_table(results, verbose: bool = False):
    """Print a degradation analysis table for a single result."""
    if not results.degradations:
        console.print("[green]No degradation detected.[/green] Image looks clean.")
        return

    table = Table(title="Degradation Chain (estimated order)")
    table.add_column("#", style="dim", width=3)
    table.add_column("Degradation", style="bold")
    table.add_column("Confidence", justify="right")
    table.add_column("Severity", justify="right")

    for i, d in enumerate(results.degradations, 1):
        color = "red" if d.severity > 0.7 else "yellow" if d.severity > 0.4 else "green"
        table.add_row(
            str(i),
            d.name,
            f"{d.confidence:.0%}",
            f"[{color}]{d.severity:.0%}[/{color}]",
        )

    console.print(table)

    if verbose:
        console.print("\n[bold]Details:[/bold]")
        for d in results.degradations:
            console.print(f"  [dim]*[/dim] {d.name}: {d.detail}")


@app.command()
def _result_to_dict(result) -> dict:
    """Convert an AnalysisResult to a JSON-serializable dict."""
    return {
        "file": result.file_path,
        "format": result.file_format,
        "dimensions": list(result.dimensions),
        "overall_severity": round(result.overall_severity, 3),
        "degradations": [
            {
                "name": d.name,
                "category": d.category,
                "confidence": round(d.confidence, 3),
                "severity": round(d.severity, 3),
                "detail": d.detail,
            }
            for d in result.degradations
        ],
        "metadata": {k: str(v) for k, v in result.metadata.items()},
    }


@app.command()
def analyze(
    path: Path = typer.Argument(..., help="Image file or directory to analyze"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed detection info"),
    json: bool = typer.Option(False, "--json", "-j", help="Output results as JSON"),
):
    """Diagnose the degradation chain of an image or batch of images."""
    if not path.exists():
        console.print(f"[red]Error:[/red] Path not found: {path}")
        raise typer.Exit(1)

    files = _collect_images(path)
    if not files:
        console.print(f"[red]Error:[/red] No image files found in: {path}")
        raise typer.Exit(1)

    analyzer = DegradationAnalyzer()

    if json:
        results_list = []
        for file in files:
            result = analyzer.analyze(file)
            results_list.append(_result_to_dict(result))
        print(json_mod.dumps(results_list if len(files) > 1 else results_list[0], indent=2))
        return

    if len(files) == 1:
        console.print(f"\n[bold]Analyzing:[/bold] {files[0].name}\n")
        results = analyzer.analyze(files[0])
        _print_analysis_table(results, verbose)
        console.print()
        return

    # Batch mode
    console.print(f"\n[bold]Batch analyzing {len(files)} images in:[/bold] {path}\n")

    summary = Table(title="Batch Analysis Summary")
    summary.add_column("File", style="bold")
    summary.add_column("Degradations", justify="center")
    summary.add_column("Worst Severity", justify="right")
    summary.add_column("Top Issue")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing...", total=len(files))

        for file in files:
            progress.update(task, description=f"Analyzing {file.name}...")
            results = analyzer.analyze(file)

            if results.degradations:
                worst = max(results.degradations, key=lambda d: d.severity)
                color = "red" if worst.severity > 0.7 else "yellow" if worst.severity > 0.4 else "green"
                summary.add_row(
                    file.name,
                    str(len(results.degradations)),
                    f"[{color}]{worst.severity:.0%}[/{color}]",
                    worst.name,
                )
            else:
                summary.add_row(file.name, "0", "[green]0%[/green]", "[green]Clean[/green]")

            progress.advance(task)

    console.print()
    console.print(summary)
    console.print()


@app.command()
def report(
    path: Path = typer.Argument(..., help="Image file or directory"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file or directory"),
):
    """Generate detailed forensic reports for one or more images."""
    if not path.exists():
        console.print(f"[red]Error:[/red] Path not found: {path}")
        raise typer.Exit(1)

    files = _collect_images(path)
    if not files:
        console.print(f"[red]Error:[/red] No image files found in: {path}")
        raise typer.Exit(1)

    analyzer = DegradationAnalyzer()

    if len(files) == 1:
        results = analyzer.analyze(files[0])
        report_text = render_report(files[0], results)
        if output:
            output.write_text(report_text)
            console.print(f"[green]Report saved to:[/green] {output}")
        else:
            console.print(report_text)
        return

    # Batch mode
    out_dir = output or path
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating reports...", total=len(files))

        for file in files:
            progress.update(task, description=f"Reporting {file.name}...")
            results = analyzer.analyze(file)
            report_text = render_report(file, results)
            report_path = out_dir / f"{file.stem}_report.txt"
            report_path.write_text(report_text)
            progress.advance(task)

    console.print(f"\n[green]{len(files)} reports saved to:[/green] {out_dir}\n")


@app.command()
def restore(
    path: Path = typer.Argument(..., help="Image file or directory to restore"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file or directory"),
    format: Optional[str] = typer.Option(None, "--format", "-f", help="Output format (png, jpg, webp)"),
    no_neural: bool = typer.Option(False, "--no-neural", help="Disable neural models, use classical only"),
):
    """Reverse the degradation chain and restore one or more images."""
    if not path.exists():
        console.print(f"[red]Error:[/red] Path not found: {path}")
        raise typer.Exit(1)

    files = _collect_images(path)
    if not files:
        console.print(f"[red]Error:[/red] No image files found in: {path}")
        raise typer.Exit(1)

    from artefex.restore import RestorationPipeline

    analyzer = DegradationAnalyzer()
    pipeline = RestorationPipeline(use_neural=not no_neural)

    if pipeline.neural_engine:
        console.print("[dim]Neural models: enabled[/dim]")
    else:
        console.print("[dim]Neural models: not available (using classical methods)[/dim]")

    if len(files) == 1:
        file = files[0]
        console.print(f"\n[bold]Restoring:[/bold] {file.name}\n")
        results = analyzer.analyze(file)

        if not results.degradations:
            console.print("[green]No degradation detected.[/green] Nothing to restore.")
            return

        console.print(f"[dim]Found {len(results.degradations)} degradation(s). Reversing chain...[/dim]\n")
        out_path = output or file.with_stem(f"{file.stem}_restored")
        info = pipeline.restore(file, results, out_path, format=format)

        for step in info["steps"]:
            console.print(f"  [dim]*[/dim] {step}")

        console.print(f"\n[green]Restored image saved to:[/green] {info['output_path']}\n")
        return

    # Batch mode
    out_dir = output or path / "restored"
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)

    restored_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Restoring...", total=len(files))

        for file in files:
            progress.update(task, description=f"Restoring {file.name}...")
            results = analyzer.analyze(file)

            if results.degradations:
                out_path = out_dir / f"{file.stem}_restored{file.suffix}"
                pipeline.restore(file, results, out_path, format=format)
                restored_count += 1

            progress.advance(task)

    console.print(f"\n[green]{restored_count}/{len(files)} images restored to:[/green] {out_dir}\n")


@app.command()
def compare(
    original: Path = typer.Argument(..., help="Original (degraded) image"),
    restored: Path = typer.Argument(..., help="Restored image"),
):
    """Compare original and restored images side by side."""
    for f in [original, restored]:
        if not f.exists():
            console.print(f"[red]Error:[/red] File not found: {f}")
            raise typer.Exit(1)

    from PIL import Image
    import numpy as np

    img_orig = Image.open(original).convert("RGB")
    img_rest = Image.open(restored).convert("RGB")

    arr_orig = np.array(img_orig, dtype=np.float64)
    arr_rest = np.array(img_rest, dtype=np.float64)

    # Resize if needed
    if arr_orig.shape != arr_rest.shape:
        img_rest = img_rest.resize(img_orig.size, Image.LANCZOS)
        arr_rest = np.array(img_rest, dtype=np.float64)

    # Compute metrics
    mse = np.mean((arr_orig - arr_rest) ** 2)
    psnr = 10 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")

    # Per-channel difference
    diff = np.abs(arr_orig - arr_rest)
    r_diff = diff[:, :, 0].mean()
    g_diff = diff[:, :, 1].mean()
    b_diff = diff[:, :, 2].mean()

    # Structural change percentage
    change_mask = np.any(diff > 10, axis=2)
    change_pct = change_mask.sum() / change_mask.size

    console.print(f"\n[bold]Comparison: {original.name} vs {restored.name}[/bold]\n")

    table = Table()
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Original size", f"{img_orig.size[0]}x{img_orig.size[1]}")
    table.add_row("Restored size", f"{img_rest.size[0]}x{img_rest.size[1]}")
    table.add_row("MSE", f"{mse:.2f}")
    table.add_row("PSNR", f"{psnr:.2f} dB")
    table.add_row("Mean diff (R)", f"{r_diff:.2f}")
    table.add_row("Mean diff (G)", f"{g_diff:.2f}")
    table.add_row("Mean diff (B)", f"{b_diff:.2f}")
    table.add_row("Pixels changed (>10)", f"{change_pct:.1%}")

    console.print(table)

    # Generate difference heatmap
    diff_gray = np.mean(diff, axis=2)
    diff_normalized = np.clip(diff_gray / diff_gray.max() * 255, 0, 255).astype(np.uint8) if diff_gray.max() > 0 else diff_gray.astype(np.uint8)
    diff_img = Image.fromarray(diff_normalized, mode="L")
    diff_path = original.parent / f"{original.stem}_vs_{restored.stem}_diff.png"
    diff_img.save(diff_path)

    console.print(f"\n[green]Difference heatmap saved to:[/green] {diff_path}\n")


@app.command()
def models(
    action: str = typer.Argument(
        "list", help="Action: list, import"
    ),
    key: Optional[str] = typer.Argument(None, help="Model key (for import)"),
    path: Optional[Path] = typer.Argument(None, help="Path to model file (for import)"),
):
    """Manage neural models for restoration."""
    from artefex.models_registry import ModelRegistry

    registry = ModelRegistry()

    if action == "list":
        all_models = registry.list_models()

        table = Table(title="Artefex Neural Models")
        table.add_column("Key", style="bold")
        table.add_column("Name")
        table.add_column("Category")
        table.add_column("Version")
        table.add_column("Status", justify="center")

        for m in all_models:
            status = "[green]installed[/green]" if m.is_available else "[dim]not installed[/dim]"
            table.add_row(m.key, m.name, m.category, m.version, status)

        console.print(table)

        installed = sum(1 for m in all_models if m.is_available)
        console.print(f"\n{installed}/{len(all_models)} models installed")

        if installed == 0:
            console.print(
                "\n[dim]To use neural models, install onnxruntime and import model files:[/dim]"
                "\n[dim]  pip install onnxruntime[/dim]"
                "\n[dim]  artefex models import deblock-v1 path/to/model.onnx[/dim]"
            )

        console.print()

    elif action == "import":
        if not key or not path:
            console.print("[red]Error:[/red] Usage: artefex models import <key> <path>")
            raise typer.Exit(1)

        if not path.exists():
            console.print(f"[red]Error:[/red] File not found: {path}")
            raise typer.Exit(1)

        try:
            model = registry.import_model(path, key)
            console.print(f"[green]Imported:[/green] {model.name} -> {model.local_path}")
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    else:
        console.print(f"[red]Error:[/red] Unknown action: {action}. Use 'list' or 'import'.")
        raise typer.Exit(1)


@app.command(name="video-analyze")
def video_analyze(
    path: Path = typer.Argument(..., help="Video file to analyze"),
    samples: int = typer.Option(10, "--samples", "-s", help="Number of frames to sample"),
    json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Analyze a video file by sampling frames for degradation."""
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(1)

    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        console.print(f"[red]Error:[/red] Not a supported video format: {path.suffix}")
        raise typer.Exit(1)

    try:
        from artefex.video import VideoAnalyzer
    except ImportError:
        console.print("[red]Error:[/red] Video support requires: pip install artefex\\[video]")
        raise typer.Exit(1)

    va = VideoAnalyzer(sample_count=samples)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Sampling frames...", total=samples)

        def on_progress(current, total):
            progress.update(task, completed=current, total=total)

        result = va.analyze(path, on_progress=on_progress)

    if json:
        data = {
            "file": result.file_path,
            "frame_count": result.frame_count,
            "fps": result.fps,
            "resolution": list(result.resolution),
            "duration_seconds": round(result.duration_seconds, 2),
            "codec": result.codec,
            "overall_severity": round(result.overall_severity, 3),
            "degradation_summary": {
                name: {k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()}
                for name, stats in result.degradation_summary.items()
            },
        }
        print(json_mod.dumps(data, indent=2))
        return

    console.print(f"\n[bold]Video Analysis:[/bold] {path.name}\n")

    info_table = Table(title="Video Info")
    info_table.add_column("Property", style="bold")
    info_table.add_column("Value")
    info_table.add_row("Resolution", f"{result.resolution[0]}x{result.resolution[1]}")
    info_table.add_row("Frames", str(result.frame_count))
    info_table.add_row("FPS", f"{result.fps:.2f}")
    info_table.add_row("Duration", f"{result.duration_seconds:.1f}s")
    info_table.add_row("Codec", result.codec)
    info_table.add_row("Frames sampled", str(len(result.frame_results)))
    console.print(info_table)

    if not result.degradation_summary:
        console.print("\n[green]No consistent degradation detected across frames.[/green]\n")
        return

    deg_table = Table(title="Degradation Summary (across sampled frames)")
    deg_table.add_column("Degradation", style="bold")
    deg_table.add_column("Frequency", justify="right")
    deg_table.add_column("Avg Severity", justify="right")
    deg_table.add_column("Avg Confidence", justify="right")

    for name, stats in sorted(result.degradation_summary.items(), key=lambda x: x[1]["avg_severity"], reverse=True):
        color = "red" if stats["avg_severity"] > 0.7 else "yellow" if stats["avg_severity"] > 0.4 else "green"
        deg_table.add_row(
            name,
            f"{stats['frequency']:.0%}",
            f"[{color}]{stats['avg_severity']:.0%}[/{color}]",
            f"{stats['avg_confidence']:.0%}",
        )

    console.print(deg_table)
    console.print()


@app.command(name="video-restore")
def video_restore(
    path: Path = typer.Argument(..., help="Video file to restore"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output video path"),
    no_neural: bool = typer.Option(False, "--no-neural", help="Disable neural models"),
):
    """Restore a video file frame by frame."""
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(1)

    try:
        from artefex.video import VideoRestorer
    except ImportError:
        console.print("[red]Error:[/red] Video support requires: pip install artefex\\[video]")
        raise typer.Exit(1)

    out_path = output or path.with_stem(f"{path.stem}_restored")
    if not out_path.suffix:
        out_path = out_path.with_suffix(".mp4")

    restorer = VideoRestorer(use_neural=not no_neural)

    console.print(f"\n[bold]Restoring video:[/bold] {path.name}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing frames...", total=100)

        def on_progress(current, total):
            progress.update(task, completed=current, total=total, description=f"Frame {current}/{total}...")

        info = restorer.restore(path, out_path, on_progress=on_progress)

    console.print(f"\n[green]Restored {info['frames_processed']} frames[/green]")
    if info.get("degradations_fixed"):
        console.print(f"[dim]Fixed: {', '.join(info['degradations_fixed'])}[/dim]")
    console.print(f"[green]Saved to:[/green] {info.get('output_path', out_path)}\n")


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8787, "--port", "-p", help="Port to listen on"),
):
    """Launch the Artefex web UI."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Error:[/red] Web UI requires: pip install artefex\\[web]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Artefex Web UI[/bold]")
    console.print(f"Open [link=http://{host}:{port}]http://{host}:{port}[/link] in your browser\n")
    uvicorn.run("artefex.web:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
