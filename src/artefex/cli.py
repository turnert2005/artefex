"""CLI entry point for artefex."""

import json as json_mod
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
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
log = logging.getLogger("artefex")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def _setup_logging(debug: bool = False, quiet: bool = False):
    """Configure logging level."""
    if quiet:
        log.setLevel(logging.WARNING)
    elif debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(message)s",
            handlers=[RichHandler(console=console, show_time=False)],
        )
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)


def _read_stdin_image() -> Optional[Path]:
    """Read image data from stdin and save to temp file."""
    if sys.stdin.isatty():
        return None

    try:
        data = sys.stdin.buffer.read()
        if not data or len(data) < 100:
            return None

        # Detect format from magic bytes
        ext = ".jpg"
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            ext = ".png"
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            ext = ".webp"
        elif data[:2] == b"BM":
            ext = ".bmp"
        elif data[:4] == b"GIF8":
            ext = ".gif"

        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.write(data)
        tmp.close()
        return Path(tmp.name)
    except Exception:
        return None


def _compute_ssim(img1: "np.ndarray", img2: "np.ndarray") -> float:
    """Compute mean SSIM between two images (numpy float64 arrays, HxWxC)."""
    import numpy as np

    # Convert to grayscale for SSIM
    if len(img1.shape) == 3:
        gray1 = np.mean(img1, axis=2)
        gray2 = np.mean(img2, axis=2)
    else:
        gray1, gray2 = img1, img2

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = _uniform_filter(gray1, 11)
    mu2 = _uniform_filter(gray2, 11)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _uniform_filter(gray1 ** 2, 11) - mu1_sq
    sigma2_sq = _uniform_filter(gray2 ** 2, 11) - mu2_sq
    sigma12 = _uniform_filter(gray1 * gray2, 11) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = num / den
    return float(np.mean(ssim_map))


def _uniform_filter(arr: "np.ndarray", size: int) -> "np.ndarray":
    """Simple uniform (box) filter via cumulative sums."""
    import numpy as np

    # Pad and compute running average using cumsum (fast, no PIL dependency)
    pad = size // 2
    padded = np.pad(arr, pad, mode="reflect")

    # Row-wise cumsum
    cs = np.cumsum(padded, axis=1)
    row_avg = (cs[:, size:] - cs[:, :-size]) / size

    # Column-wise cumsum
    cs2 = np.cumsum(row_avg, axis=0)
    result = (cs2[size:, :] - cs2[:-size, :]) / size

    # Trim to original size
    return result[: arr.shape[0], : arr.shape[1]]


def _download_url(url: str) -> Optional[Path]:
    """Download an image from a URL to a temp file. Returns path or None."""
    import tempfile
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Artefex/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "")
            ext = ".jpg"
            if "png" in content_type:
                ext = ".png"
            elif "webp" in content_type:
                ext = ".webp"
            elif "gif" in content_type:
                ext = ".gif"

            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp.write(resp.read())
            tmp.close()
            return Path(tmp.name)
    except (urllib.error.URLError, OSError) as e:
        console.print(f"[red]Error downloading:[/red] {e}")
        return None


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
    path: str = typer.Argument("-", help="Image file, directory, URL, or - for stdin"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed detection info"),
    json: bool = typer.Option(False, "--json", "-j", help="Output results as JSON"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-essential output"),
):
    """Diagnose the degradation chain of an image or batch of images."""
    _setup_logging(debug=debug, quiet=quiet)
    tmp_downloaded = None

    # Check for stdin
    if path == "-":
        tmp_downloaded = _read_stdin_image()
        if tmp_downloaded is None:
            console.print("[red]Error:[/red] No image data on stdin")
            raise typer.Exit(1)
        files = [tmp_downloaded]
    # Check if it's a URL
    elif path.startswith("http://") or path.startswith("https://"):
        console.print(f"[dim]Downloading image from URL...[/dim]")
        tmp_downloaded = _download_url(path)
        if tmp_downloaded is None:
            raise typer.Exit(1)
        files = [tmp_downloaded]
    else:
        p = Path(path)
        if not p.exists():
            console.print(f"[red]Error:[/red] Path not found: {path}")
            raise typer.Exit(1)
        files = _collect_images(p)
        if not files:
            console.print(f"[red]Error:[/red] No image files found in: {path}")
            raise typer.Exit(1)

    analyzer = DegradationAnalyzer()

    if json:
        results_list = []
        for file in files:
            result = analyzer.analyze(file)
            results_list.append(_result_to_dict(result))
        if tmp_downloaded:
            tmp_downloaded.unlink(missing_ok=True)
        print(json_mod.dumps(results_list if len(files) > 1 else results_list[0], indent=2))
        return

    if len(files) == 1:
        console.print(f"\n[bold]Analyzing:[/bold] {files[0].name}\n")
        results = analyzer.analyze(files[0])
        if tmp_downloaded:
            tmp_downloaded.unlink(missing_ok=True)
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
    html: bool = typer.Option(False, "--html", help="Generate HTML report with embedded images"),
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
        if html:
            from artefex.report_html import render_html_report
            report_text = render_html_report(files[0], results)
            out = output or files[0].with_suffix(".html")
            out.write_text(report_text, encoding="utf-8")
            console.print(f"[green]HTML report saved to:[/green] {out}")
        else:
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
            if html:
                from artefex.report_html import render_html_report
                report_text = render_html_report(file, results)
                report_path = out_dir / f"{file.stem}_report.html"
                report_path.write_text(report_text, encoding="utf-8")
            else:
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


@app.command(name="restore-preview")
def restore_preview(
    path: Path = typer.Argument(..., help="Image file to preview restoration steps"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for step images"),
):
    """Show what each restoration step changes by saving intermediate images."""
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(1)

    from artefex.restore import RestorationPipeline
    from PIL import Image as PILImage
    import numpy as np

    analyzer = DegradationAnalyzer()
    result = analyzer.analyze(path)

    if not result.degradations:
        console.print("[green]No degradation detected.[/green] Nothing to preview.")
        return

    out_dir = output or path.parent / f"{path.stem}_steps"
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]Restoration Preview:[/bold] {path.name}")
    console.print(f"[dim]{len(result.degradations)} degradation(s) detected[/dim]\n")

    # Save original
    orig = PILImage.open(path).convert("RGB")
    orig.save(out_dir / "00_original.png")

    # Apply each fix one at a time
    pipeline = RestorationPipeline(use_neural=False)
    img = orig.copy()
    ordered = sorted(result.degradations, key=lambda d: d.severity)

    table = Table(title="Step-by-Step Preview")
    table.add_column("Step", style="dim", width=4)
    table.add_column("Fix Applied", style="bold")
    table.add_column("File")
    table.add_column("Pixels Changed", justify="right")

    prev_arr = np.array(img, dtype=np.float64)

    for i, degradation in enumerate(ordered, 1):
        restorer = pipeline._restorers.get(degradation.name)
        if restorer:
            img = restorer(img, degradation)

            step_name = f"{i:02d}_{degradation.name.lower().replace(' ', '_').replace('/', '_')}"
            step_path = out_dir / f"{step_name}.png"
            img.save(step_path)

            # Compute what changed
            curr_arr = np.array(img, dtype=np.float64)
            diff = np.abs(curr_arr - prev_arr)
            changed = np.any(diff > 2, axis=2).sum()
            total = diff.shape[0] * diff.shape[1]
            pct = changed / total

            table.add_row(str(i), degradation.name, step_path.name, f"{pct:.1%}")
            prev_arr = curr_arr

    # Save final
    img.save(out_dir / "final_restored.png")

    console.print(table)
    console.print(f"\n[green]{len(ordered)} steps saved to:[/green] {out_dir}\n")


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

    # SSIM (structural similarity)
    ssim = _compute_ssim(arr_orig, arr_rest)

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
    table.add_row("SSIM", f"{ssim:.4f}")
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
def dashboard(
    path: Path = typer.Argument(..., help="Directory of images to analyze"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output HTML file"),
):
    """Generate an HTML dashboard summarizing all images in a directory."""
    if not path.exists() or not path.is_dir():
        console.print(f"[red]Error:[/red] Directory not found: {path}")
        raise typer.Exit(1)

    files = _collect_images(path)
    if not files:
        console.print(f"[red]Error:[/red] No image files found in: {path}")
        raise typer.Exit(1)

    from artefex.dashboard import generate_dashboard

    out_path = output or path / "artefex_dashboard.html"

    console.print(f"\n[bold]Generating dashboard for {len(files)} images...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing...", total=len(files))

        def on_progress(current, total):
            progress.update(task, completed=current, total=total)

        generate_dashboard(files, out_path, on_progress=on_progress)

    console.print(f"\n[green]Dashboard saved to:[/green] {out_path}\n")


@app.command()
def heatmap(
    path: Path = typer.Argument(..., help="Image file to generate heatmap for"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path for heatmap image"),
    patch_size: int = typer.Option(32, "--patch-size", "-p", help="Patch size for analysis grid"),
):
    """Generate a spatial degradation heatmap showing where damage is worst."""
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(1)

    from artefex.heatmap import generate_heatmap

    out_path = output or path.with_stem(f"{path.stem}_heatmap")
    if not out_path.suffix:
        out_path = out_path.with_suffix(".png")

    console.print(f"\n[bold]Generating heatmap:[/bold] {path.name}\n")

    stats = generate_heatmap(path, out_path, patch_size=patch_size)

    table = Table()
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Grid size", f"{stats['grid_size'][1]}x{stats['grid_size'][0]} patches")
    table.add_row("Patch size", f"{stats['patch_size']}px")
    table.add_row("[green]Healthy regions[/green]", f"{stats['healthy_pct']:.0%}")
    table.add_row("[yellow]Moderate degradation[/yellow]", f"{stats['moderate_pct']:.0%}")
    table.add_row("[red]Severe degradation[/red]", f"{stats['severe_pct']:.0%}")
    table.add_row("Worst region", f"({stats['worst_region'][0]}, {stats['worst_region'][1]})")

    console.print(table)
    console.print(f"\n[green]Heatmap saved to:[/green] {out_path}\n")


@app.command()
def grade(
    path: str = typer.Argument(..., help="Image file, directory, or URL"),
    json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Export format: csv, markdown"),
):
    """Grade image quality on an A-F scale."""
    from artefex.grade import compute_grade

    # Handle URL
    tmp_downloaded = None
    if path.startswith("http://") or path.startswith("https://"):
        tmp_downloaded = _download_url(path)
        if tmp_downloaded is None:
            raise typer.Exit(1)
        files = [tmp_downloaded]
    else:
        p = Path(path)
        if not p.exists():
            console.print(f"[red]Error:[/red] Path not found: {path}")
            raise typer.Exit(1)
        files = _collect_images(p)

    if not files:
        console.print(f"[red]Error:[/red] No image files found")
        raise typer.Exit(1)

    analyzer = DegradationAnalyzer()
    grades = []

    for file in files:
        result = analyzer.analyze(file)
        g = compute_grade(result)
        g["file"] = file.name
        grades.append(g)

    if tmp_downloaded:
        tmp_downloaded.unlink(missing_ok=True)

    if json:
        print(json_mod.dumps(grades if len(grades) > 1 else grades[0], indent=2))
        return

    if export == "csv":
        print("file,grade,score,degradations")
        for g in grades:
            n_degs = len(g["breakdown"])
            print(f"{g['file']},{g['grade']},{g['score']},{n_degs}")
        return

    if export == "markdown":
        print("| File | Grade | Score | Description |")
        print("|---|---|---|---|")
        for g in grades:
            print(f"| {g['file']} | {g['grade']} | {g['score']}/100 | {g['description']} |")
        return

    if len(grades) == 1:
        g = grades[0]
        console.print(f"\n[bold]{g['file']}[/bold]\n")
        console.print(f"  [{g['color']}]Grade: {g['grade']}[/{g['color']}]  Score: {g['score']}/100")
        console.print(f"  [dim]{g['description']}[/dim]\n")

        if g["breakdown"]:
            table = Table(title="Score Breakdown")
            table.add_column("Issue", style="bold")
            table.add_column("Penalty", justify="right")
            table.add_column("Severity", justify="right")
            table.add_column("Confidence", justify="right")

            for b in g["breakdown"]:
                table.add_row(b["name"], f"-{b['penalty']}", f"{b['severity']}%", f"{b['confidence']}%")

            console.print(table)
            console.print()
    else:
        table = Table(title="Quality Grades")
        table.add_column("File", style="bold")
        table.add_column("Grade", justify="center")
        table.add_column("Score", justify="right")
        table.add_column("Issues", justify="center")
        table.add_column("Description")

        for g in grades:
            table.add_row(
                g["file"],
                f"[{g['color']}]{g['grade']}[/{g['color']}]",
                f"{g['score']}",
                str(len(g["breakdown"])),
                g["description"][:50],
            )

        console.print(table)

        # Summary stats
        avg_score = sum(g["score"] for g in grades) / len(grades)
        grade_counts = {}
        for g in grades:
            grade_counts[g["grade"]] = grade_counts.get(g["grade"], 0) + 1
        dist = " ".join(f"{g}:{c}" for g, c in sorted(grade_counts.items()))
        console.print(f"\n[dim]Average score: {avg_score:.1f}/100 | Distribution: {dist}[/dim]\n")


@app.command()
def timeline(
    path: Path = typer.Argument(..., help="Image file to visualize degradation timeline"),
):
    """Visualize the estimated degradation history as a timeline."""
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(1)

    analyzer = DegradationAnalyzer()
    result = analyzer.analyze(path)

    if not result.degradations:
        console.print(f"\n[bold]{path.name}[/bold] - [green]No degradation detected. Clean image.[/green]\n")
        return

    console.print(f"\n[bold]Degradation Timeline:[/bold] {path.name}\n")
    console.print(f"  [dim]Original capture[/dim]")
    console.print(f"  [green]|[/green]")

    # Order by severity (highest = earliest estimated degradation)
    ordered = sorted(result.degradations, key=lambda d: d.severity, reverse=True)

    for i, d in enumerate(ordered):
        sev_pct = int(d.severity * 100)
        conf_pct = int(d.confidence * 100)

        if d.severity > 0.7:
            color = "red"
            bar_char = "#"
        elif d.severity > 0.4:
            color = "yellow"
            bar_char = "="
        else:
            color = "green"
            bar_char = "-"

        bar = bar_char * max(1, sev_pct // 5)
        console.print(f"  [{color}]|[/{color}]")
        console.print(f"  [{color}]+-- {d.name}[/{color}]")
        console.print(f"  [{color}]|[/{color}]   [{color}]{bar}[/{color}] {sev_pct}% severity, {conf_pct}% confidence")
        console.print(f"  [{color}]|[/{color}]   [dim]{d.detail[:80]}[/dim]")

    console.print(f"  [red]|[/red]")
    console.print(f"  [red]v[/red]")
    console.print(f"  [dim]Current state (overall severity: {result.overall_severity:.0%})[/dim]")
    console.print()
    console.print(f"  [dim]Run `artefex restore {path.name}` to reverse this chain.[/dim]\n")


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
def plugins():
    """List installed Artefex plugins."""
    from artefex.plugins import get_plugin_registry

    registry = get_plugin_registry()
    info = registry.list_plugins()

    table = Table(title="Installed Plugins")
    table.add_column("Type", style="bold")
    table.add_column("Name")

    if not info["detectors"] and not info["restorers"]:
        console.print("\n[dim]No plugins installed.[/dim]")
        console.print(
            "\n[dim]Plugins are Python packages that register via entry points.[/dim]"
            "\n[dim]See: https://github.com/turnert2005/artefex#plugins[/dim]\n"
        )
        return

    for name in info["detectors"]:
        table.add_row("detector", name)
    for name in info["restorers"]:
        table.add_row("restorer", name)

    console.print(table)
    console.print(
        f"\n{len(info['detectors'])} detector(s), {len(info['restorers'])} restorer(s)\n"
    )


@app.command()
def watch(
    path: Path = typer.Argument(..., help="Directory to watch for new images"),
    auto_restore: bool = typer.Option(False, "--restore", "-r", help="Auto-restore new images"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for restored images"),
    interval: float = typer.Option(2.0, "--interval", "-i", help="Seconds between scans"),
):
    """Watch a directory and auto-analyze new images as they appear."""
    if not path.exists() or not path.is_dir():
        console.print(f"[red]Error:[/red] Directory not found: {path}")
        raise typer.Exit(1)

    from artefex.watch import DirectoryWatcher

    watcher = DirectoryWatcher(
        watch_dir=path,
        output_dir=output,
        auto_restore=auto_restore,
    )

    mode = "analyze + restore" if auto_restore else "analyze only"
    console.print(f"\n[bold]Watching:[/bold] {path}")
    console.print(f"[dim]Mode: {mode} | Interval: {interval}s | Ctrl+C to stop[/dim]\n")

    def on_result(info):
        severity = info["overall_severity"]
        color = "red" if severity > 0.7 else "yellow" if severity > 0.4 else "green"
        status = f"[{color}]{severity:.0%}[/{color}]"

        if info["degradations"] == 0:
            console.print(f"  [green]+[/green] {info['file']} - clean")
        else:
            top = info["top_issue"]
            restored_msg = " -> restored" if info.get("restored") else ""
            console.print(f"  [yellow]+[/yellow] {info['file']} - {info['degradations']} issues, severity {status}, top: {top}{restored_msg}")

    try:
        watcher.run(interval=interval, on_result=on_result)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching.[/dim]\n")


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


@app.command()
def fix(
    path: str = typer.Argument(..., help="Image file or URL to auto-fix"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
):
    """Smart one-shot: analyze, grade, restore, and report in one command."""
    from artefex.grade import compute_grade

    # Handle URL
    tmp_downloaded = None
    if path.startswith("http://") or path.startswith("https://"):
        console.print(f"[dim]Downloading...[/dim]")
        tmp_downloaded = _download_url(path)
        if tmp_downloaded is None:
            raise typer.Exit(1)
        file = tmp_downloaded
    else:
        file = Path(path)
        if not file.exists():
            console.print(f"[red]Error:[/red] File not found: {path}")
            raise typer.Exit(1)

    analyzer = DegradationAnalyzer()
    result = analyzer.analyze(file)
    grade_info = compute_grade(result)

    console.print(f"\n  [bold]{file.name}[/bold]  [{grade_info['color']}]{grade_info['grade']}[/{grade_info['color']}] ({grade_info['score']}/100)")

    if not result.degradations:
        console.print(f"  [green]Clean image - no restoration needed.[/green]\n")
        if tmp_downloaded:
            tmp_downloaded.unlink(missing_ok=True)
        return

    for d in result.degradations:
        color = "red" if d.severity > 0.7 else "yellow" if d.severity > 0.4 else "green"
        console.print(f"  [{color}]*[/{color}] {d.name} ({d.severity:.0%})")

    # Auto-restore
    from artefex.restore import RestorationPipeline
    pipeline = RestorationPipeline()

    out_path = output or Path(file.stem + "_fixed" + file.suffix)
    info = pipeline.restore(file, result, out_path)

    console.print(f"\n  [green]Fixed -> {out_path}[/green]")

    # Quick before/after
    from PIL import Image
    import numpy as np
    orig = np.array(Image.open(file).convert("RGB"), dtype=np.float64)
    fixed = np.array(Image.open(out_path).convert("RGB"), dtype=np.float64)
    if orig.shape == fixed.shape:
        mse = np.mean((orig - fixed) ** 2)
        psnr = 10 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")
        console.print(f"  [dim]PSNR: {psnr:.1f} dB | {len(info['steps'])} fixes applied[/dim]\n")
    else:
        console.print(f"  [dim]{len(info['steps'])} fixes applied[/dim]\n")

    if tmp_downloaded:
        tmp_downloaded.unlink(missing_ok=True)


@app.command()
def audit(
    path: Path = typer.Argument(..., help="Image file for comprehensive audit"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for audit files"),
):
    """Run a comprehensive audit combining all analysis tools on a single image."""
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(1)

    from artefex.grade import compute_grade
    from artefex.fingerprint import PlatformFingerprinter
    from artefex.heatmap import generate_heatmap
    from artefex.report_html import render_html_report

    out_dir = output or path.parent / f"{path.stem}_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]Comprehensive Audit:[/bold] {path.name}\n")

    # 1. Full analysis
    console.print("  [dim]*[/dim] Running degradation analysis...")
    analyzer = DegradationAnalyzer()
    result = analyzer.analyze(path)

    # 2. Grade
    console.print("  [dim]*[/dim] Computing quality grade...")
    grade_info = compute_grade(result)

    # 3. Platform fingerprint
    console.print("  [dim]*[/dim] Running platform fingerprinting...")
    fp = PlatformFingerprinter()
    platforms = fp.fingerprint(path)

    # 4. Heatmap
    console.print("  [dim]*[/dim] Generating degradation heatmap...")
    heatmap_path = out_dir / f"{path.stem}_heatmap.png"
    heatmap_stats = generate_heatmap(path, heatmap_path)

    # 5. HTML report
    console.print("  [dim]*[/dim] Generating HTML report...")
    html_path = out_dir / f"{path.stem}_report.html"
    html = render_html_report(path, result)
    html_path.write_text(html, encoding="utf-8")

    # 6. Restore
    console.print("  [dim]*[/dim] Running restoration...")
    from artefex.restore import RestorationPipeline
    pipeline = RestorationPipeline(use_neural=False)
    restored_path = out_dir / f"{path.stem}_restored.png"
    if result.degradations:
        pipeline.restore(path, result, restored_path)

    # Print summary
    console.print()
    console.print(f"  [{grade_info['color']}]Grade: {grade_info['grade']}[/{grade_info['color']}]  Score: {grade_info['score']}/100")
    console.print(f"  [dim]{grade_info['description']}[/dim]")
    console.print()

    if result.degradations:
        table = Table(title="Degradation Chain")
        table.add_column("#", style="dim", width=3)
        table.add_column("Issue", style="bold")
        table.add_column("Severity", justify="right")

        for i, d in enumerate(result.degradations, 1):
            color = "red" if d.severity > 0.7 else "yellow" if d.severity > 0.4 else "green"
            table.add_row(str(i), d.name, f"[{color}]{d.severity:.0%}[/{color}]")

        console.print(table)

    if platforms:
        console.print(f"\n  [bold]Platform attribution:[/bold]")
        for p in platforms[:3]:
            console.print(f"    {p['name']}: {p['confidence']:.0%} confidence")

    console.print(f"\n  [bold]Spatial analysis:[/bold]")
    console.print(f"    Healthy: {heatmap_stats['healthy_pct']:.0%}  Moderate: {heatmap_stats['moderate_pct']:.0%}  Severe: {heatmap_stats['severe_pct']:.0%}")

    console.print(f"\n  [bold]Output files:[/bold]")
    console.print(f"    Heatmap:  {heatmap_path}")
    console.print(f"    Report:   {html_path}")
    if result.degradations:
        console.print(f"    Restored: {restored_path}")
    console.print()


@app.command()
def duplicates(
    path: Path = typer.Argument(..., help="Directory to scan for duplicates"),
    threshold: float = typer.Option(0.9, "--threshold", "-t", help="Similarity threshold (0-1)"),
    method: str = typer.Option("phash", "--method", "-m", help="Hash method: phash, ahash, dhash"),
    json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Find duplicate or near-duplicate images in a directory."""
    if not path.exists() or not path.is_dir():
        console.print(f"[red]Error:[/red] Directory not found: {path}")
        raise typer.Exit(1)

    files = _collect_images(path)
    if not files:
        console.print(f"[red]Error:[/red] No images found in: {path}")
        raise typer.Exit(1)

    from artefex.similarity import find_duplicates

    console.print(f"\n[bold]Scanning {len(files)} images for duplicates...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Hashing...", total=len(files))

        def on_progress(current, total):
            progress.update(task, completed=current, total=total)

        groups = find_duplicates(files, threshold=threshold, hash_fn=method, on_progress=on_progress)

    if json:
        print(json_mod.dumps(groups, indent=2))
        return

    if not groups:
        console.print("[green]No duplicates found.[/green]\n")
        return

    table = Table(title=f"Duplicate Groups (threshold: {threshold:.0%})")
    table.add_column("Group", style="dim", width=5)
    table.add_column("Files", style="bold")
    table.add_column("Similarity", justify="right")

    for i, group in enumerate(groups, 1):
        files_str = "\n".join(Path(f).name for f in group["files"])
        table.add_row(str(i), files_str, f"{group['similarity']:.0%}")

    console.print(table)
    console.print(f"\n[yellow]{len(groups)} group(s)[/yellow] of duplicates found across {sum(len(g['files']) for g in groups)} files\n")


@app.command(name="accessibility")
def accessibility_cmd(
    path: Path = typer.Argument(..., help="Image file to check"),
    simulate: Optional[str] = typer.Option(None, "--simulate", "-s", help="Generate CVD simulation: protanopia, deuteranopia, tritanopia, achromatopsia, all"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for simulations"),
):
    """Check color accessibility and simulate color blindness."""
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(1)

    from artefex.accessibility import check_accessibility, generate_cvd_comparison, simulate_cvd, SIMULATION_TYPES
    from PIL import Image as PILImage

    img = PILImage.open(path).convert("RGB")

    # Run accessibility check
    result = check_accessibility(img)

    console.print(f"\n[bold]Accessibility Report:[/bold] {path.name}\n")

    table = Table(title="Color Vision Deficiency Impact")
    table.add_column("Condition", style="bold")
    table.add_column("Info Loss", justify="right")
    table.add_column("Color Diff", justify="right")

    for cvd_type, data in result["information_loss"].items():
        color = "red" if data["loss_pct"] > 15 else "yellow" if data["loss_pct"] > 8 else "green"
        table.add_row(
            data["name"],
            f"[{color}]{data['loss_pct']}%[/{color}]",
            f"{data['mean_color_diff']:.1f}",
        )

    console.print(table)

    wcag_color = "green" if result["wcag_aa_pass"] else "red"
    console.print(f"\n  Contrast ratio: [{wcag_color}]{result['contrast_ratio']}:1[/{wcag_color}]  WCAG AA: {'Pass' if result['wcag_aa_pass'] else 'Fail'}")

    if result["recommendations"]:
        console.print(f"\n  [bold]Recommendations:[/bold]")
        for rec in result["recommendations"]:
            console.print(f"  [yellow]*[/yellow] {rec}")

    # Generate simulations if requested
    if simulate:
        out_dir = output or path.parent / f"{path.stem}_accessibility"

        if simulate == "all":
            outputs = generate_cvd_comparison(path, out_dir)
            console.print(f"\n  [green]Simulations saved to:[/green] {out_dir}")
            for cvd_type, out_path in outputs.items():
                console.print(f"    {cvd_type}: {Path(out_path).name}")
        elif simulate in SIMULATION_TYPES:
            out_dir.mkdir(parents=True, exist_ok=True)
            simulated = simulate_cvd(img, simulate)
            out_path = out_dir / f"{path.stem}_{simulate}.png"
            simulated.save(out_path)
            console.print(f"\n  [green]Simulation saved to:[/green] {out_path}")
        else:
            console.print(f"\n  [red]Unknown type:[/red] {simulate}. Use: {', '.join(SIMULATION_TYPES.keys())}, or all")

    console.print()


@app.command()
def benchmark(
    path: Optional[Path] = typer.Argument(None, help="Image file to benchmark (or uses built-in test)"),
    iterations: int = typer.Option(5, "--iterations", "-n", help="Number of iterations"),
):
    """Benchmark Artefex performance on an image."""
    import time
    import tempfile

    from PIL import Image as PILImage

    # Create test image if none provided
    if path is None or not path.exists():
        arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = PILImage.fromarray(arr)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        img.save(tmp.name, quality=30)
        path = Path(tmp.name)
        console.print("[dim]Using generated 512x512 test image[/dim]")

    import numpy as np

    console.print(f"\n[bold]Benchmarking:[/bold] {path.name} ({iterations} iterations)\n")

    analyzer = DegradationAnalyzer()

    # Warm up
    analyzer.analyze(path)

    # Benchmark analyze
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = analyzer.analyze(path)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    analyze_avg = np.mean(times)
    analyze_std = np.std(times)

    # Benchmark restore
    from artefex.restore import RestorationPipeline
    pipeline = RestorationPipeline(use_neural=False)
    out_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    out_tmp.close()

    restore_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        pipeline.restore(path, result, Path(out_tmp.name))
        elapsed = time.perf_counter() - start
        restore_times.append(elapsed)

    restore_avg = np.mean(restore_times)
    restore_std = np.std(restore_times)

    # Benchmark grade
    from artefex.grade import compute_grade
    grade_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        compute_grade(result)
        elapsed = time.perf_counter() - start
        grade_times.append(elapsed)

    grade_avg = np.mean(grade_times)

    Path(out_tmp.name).unlink(missing_ok=True)

    table = Table(title="Performance Results")
    table.add_column("Operation", style="bold")
    table.add_column("Avg Time", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_column("Throughput", justify="right")

    table.add_row("Analyze", f"{analyze_avg*1000:.1f} ms", f"{analyze_std*1000:.1f} ms", f"{1/analyze_avg:.1f} img/s")
    table.add_row("Restore", f"{restore_avg*1000:.1f} ms", f"{restore_std*1000:.1f} ms", f"{1/restore_avg:.1f} img/s")
    table.add_row("Grade", f"{grade_avg*1000:.3f} ms", "-", f"{1/grade_avg:.0f} img/s")

    console.print(table)

    total = analyze_avg + restore_avg
    console.print(f"\n  [dim]Full pipeline (analyze + restore): {total*1000:.1f} ms ({1/total:.1f} img/s)[/dim]")
    console.print(f"  [dim]Detectors: {len(result.degradations)} found in {analyze_avg*1000:.1f} ms[/dim]\n")


@app.command()
def version():
    """Show Artefex version and system info."""
    from artefex import __version__

    banner = """
 [bold magenta]    _         _        __
   / \\   _ __| |_ ___ / _| _____  __
  / _ \\ | '__| __/ _ \\ |_ / _ \\ \\/ /
 / ___ \\| |  | ||  __/  _|  __/>  <
/_/   \\_\\_|   \\__\\___|_|  \\___/_/\\_\\[/bold magenta]
"""
    console.print(banner)
    console.print(f"  [bold]Artefex[/bold] v{__version__}")
    console.print(f"  Neural forensic restoration\n")

    # Check optional deps
    deps = []
    try:
        import onnxruntime
        deps.append(f"  [green]+[/green] onnxruntime {onnxruntime.__version__}")
    except ImportError:
        deps.append("  [dim]-[/dim] onnxruntime (not installed)")

    try:
        import fastapi
        deps.append(f"  [green]+[/green] fastapi {fastapi.__version__}")
    except ImportError:
        deps.append("  [dim]-[/dim] fastapi (not installed)")

    try:
        import cv2
        deps.append(f"  [green]+[/green] opencv {cv2.__version__}")
    except ImportError:
        deps.append("  [dim]-[/dim] opencv (not installed)")

    console.print("  [bold]Dependencies:[/bold]")
    for d in deps:
        console.print(d)

    # Check models
    from artefex.models_registry import ModelRegistry
    registry = ModelRegistry()
    models = registry.list_models()
    installed = sum(1 for m in models if m.is_available)
    console.print(f"\n  [bold]Models:[/bold] {installed}/{len(models)} installed")

    # Check plugins
    from artefex.plugins import get_plugin_registry
    plugins = get_plugin_registry().list_plugins()
    total_plugins = len(plugins["detectors"]) + len(plugins["restorers"])
    console.print(f"  [bold]Plugins:[/bold] {total_plugins} loaded\n")


if __name__ == "__main__":
    app()
