"""CLI entry point for artefex."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from artefex.analyze import DegradationAnalyzer
from artefex.report import render_report

app = typer.Typer(
    name="artefex",
    help="Neural forensic restoration - diagnose and reverse media degradation chains.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def analyze(
    file: Path = typer.Argument(..., help="Path to image file to analyze"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed detection info"),
):
    """Diagnose the degradation chain of an image."""
    if not file.exists():
        console.print(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(1)

    console.print(f"\n[bold]Analyzing:[/bold] {file.name}\n")

    analyzer = DegradationAnalyzer()
    results = analyzer.analyze(file)

    if not results.degradations:
        console.print("[green]No degradation detected.[/green] This image looks clean.")
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
            console.print(f"  [dim]•[/dim] {d.name}: {d.detail}")

    console.print()


@app.command()
def report(
    file: Path = typer.Argument(..., help="Path to image file"),
    output: Path = typer.Option(None, "--output", "-o", help="Save report to file"),
):
    """Generate a detailed forensic report for an image."""
    if not file.exists():
        console.print(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(1)

    analyzer = DegradationAnalyzer()
    results = analyzer.analyze(file)
    report_text = render_report(file, results)

    if output:
        output.write_text(report_text)
        console.print(f"[green]Report saved to:[/green] {output}")
    else:
        console.print(report_text)


@app.command()
def restore(
    file: Path = typer.Argument(..., help="Path to image file to restore"),
    output: Path = typer.Option(None, "--output", "-o", help="Output path for restored image"),
):
    """Reverse the degradation chain and restore an image."""
    if not file.exists():
        console.print(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(1)

    console.print(f"\n[bold]Restoring:[/bold] {file.name}\n")

    analyzer = DegradationAnalyzer()
    results = analyzer.analyze(file)

    if not results.degradations:
        console.print("[green]No degradation detected.[/green] Nothing to restore.")
        return

    console.print(f"[dim]Found {len(results.degradations)} degradation(s). Reversing chain...[/dim]\n")

    from artefex.restore import RestorationPipeline

    pipeline = RestorationPipeline()
    out_path = output or file.with_stem(f"{file.stem}_restored")
    pipeline.restore(file, results, out_path)

    console.print(f"[green]Restored image saved to:[/green] {out_path}\n")


if __name__ == "__main__":
    app()
