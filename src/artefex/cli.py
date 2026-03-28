"""CLI entry point for artefex."""

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
def analyze(
    path: Path = typer.Argument(..., help="Image file or directory to analyze"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed detection info"),
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
    pipeline = RestorationPipeline()

    if len(files) == 1:
        file = files[0]
        console.print(f"\n[bold]Restoring:[/bold] {file.name}\n")
        results = analyzer.analyze(file)

        if not results.degradations:
            console.print("[green]No degradation detected.[/green] Nothing to restore.")
            return

        console.print(f"[dim]Found {len(results.degradations)} degradation(s). Reversing chain...[/dim]\n")
        out_path = output or file.with_stem(f"{file.stem}_restored")
        pipeline.restore(file, results, out_path)
        console.print(f"[green]Restored image saved to:[/green] {out_path}\n")
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
                pipeline.restore(file, results, out_path)
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


if __name__ == "__main__":
    app()
