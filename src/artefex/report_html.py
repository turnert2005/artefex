"""HTML forensic report generator with embedded images and charts."""

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from artefex.models import AnalysisResult


def render_html_report(file_path: Path, result: AnalysisResult) -> str:
    """Generate a self-contained HTML forensic report."""

    # Encode original image as base64 thumbnail
    try:
        img = Image.open(file_path).convert("RGB")
        thumb = img.copy()
        thumb.thumbnail((400, 400))
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        img_b64 = ""

    # Build severity chart data
    chart_bars = ""
    for d in result.degradations:
        color = "#f87171" if d.severity > 0.7 else "#fbbf24" if d.severity > 0.4 else "#4ade80"
        width = max(2, int(d.severity * 100))
        chart_bars += f"""
        <div class="bar-row">
            <span class="bar-label">{d.name}</span>
            <div class="bar-track">
                <div class="bar-fill" style="width:{width}%;background:{color}"></div>
            </div>
            <span class="bar-value">{d.severity:.0%}</span>
        </div>"""

    # Build degradation rows
    deg_rows = ""
    for i, d in enumerate(result.degradations, 1):
        sev_class = "high" if d.severity > 0.7 else "mid" if d.severity > 0.4 else "low"
        deg_rows += f"""
        <tr>
            <td>{i}</td>
            <td><strong>{d.name}</strong></td>
            <td>{d.category}</td>
            <td>{d.confidence:.0%}</td>
            <td class="sev-{sev_class}">{d.severity:.0%}</td>
        </tr>
        <tr class="detail-row">
            <td></td>
            <td colspan="4">{d.detail}</td>
        </tr>"""

    # Build histogram visualization
    histogram_svg = _generate_histogram_svg(file_path)

    no_deg_msg = ""
    if not result.degradations:
        no_deg_msg = '<p class="clean">No degradation detected. Image appears clean.</p>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Artefex Report - {file_path.name}</title>
<style>
  :root {{
    --bg: #0c0c14; --surface: #13131d; --border: #1e1e30;
    --text: #d8d8e8; --dim: #6b6b88; --accent: #7c6ff7;
    --green: #4ade80; --yellow: #fbbf24; --red: #f87171;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: 'SF Mono','Cascadia Code','Fira Code',monospace;
    background: var(--bg); color: var(--text);
    padding: 2rem; line-height: 1.6;
  }}
  .container {{ max-width: 800px; margin: 0 auto; }}
  h1 {{
    font-size: 1.5rem;
    background: linear-gradient(135deg, var(--accent), #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
  }}
  h2 {{
    font-size: 1rem; color: var(--accent); margin: 1.5rem 0 0.75rem;
    border-bottom: 1px solid var(--border); padding-bottom: 0.3rem;
  }}
  .meta {{ color: var(--dim); font-size: 0.8rem; margin-bottom: 1.5rem; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem; margin-bottom: 1rem;
  }}
  .thumb {{ text-align: center; margin-bottom: 1rem; }}
  .thumb img {{ max-width: 100%; border-radius: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{
    text-align: left; padding: 0.4rem 0.6rem; color: var(--dim);
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em;
    border-bottom: 1px solid var(--border);
  }}
  td {{ padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border); }}
  .detail-row td {{ color: var(--dim); font-size: 0.75rem; padding-top: 0; border-bottom: 1px solid var(--border); }}
  .sev-high {{ color: var(--red); font-weight: 600; }}
  .sev-mid {{ color: var(--yellow); }}
  .sev-low {{ color: var(--green); }}
  .clean {{ color: var(--green); text-align: center; padding: 1rem; }}
  .info-grid {{ display: grid; grid-template-columns: auto 1fr; gap: 0.2rem 1rem; font-size: 0.82rem; }}
  .info-grid .label {{ color: var(--dim); }}
  .bar-row {{ display: flex; align-items: center; gap: 0.5rem; margin: 0.3rem 0; font-size: 0.8rem; }}
  .bar-label {{ width: 160px; text-align: right; color: var(--dim); }}
  .bar-track {{ flex: 1; height: 16px; background: var(--border); border-radius: 3px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
  .bar-value {{ width: 40px; font-weight: 600; }}
  .hist {{ text-align: center; margin: 0.5rem 0; }}
  .footer {{ text-align: center; color: var(--dim); font-size: 0.7rem; margin-top: 2rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>artefex forensic report</h1>
  <p class="meta">{file_path.name} - {result.file_format} - {result.dimensions[0]}x{result.dimensions[1]}</p>

  {"<div class='card thumb'><img src='data:image/png;base64," + img_b64 + "' alt='Image preview'></div>" if img_b64 else ""}

  <h2>File Info</h2>
  <div class="card">
    <div class="info-grid">
      <span class="label">Format</span><span>{result.file_format}</span>
      <span class="label">Dimensions</span><span>{result.dimensions[0]}x{result.dimensions[1]}</span>
      <span class="label">Mode</span><span>{result.metadata.get('mode', 'unknown')}</span>
      <span class="label">Overall Severity</span><span class="{'sev-high' if result.overall_severity > 0.7 else 'sev-mid' if result.overall_severity > 0.4 else 'sev-low'}">{result.overall_severity:.0%}</span>
    </div>
  </div>

  <h2>Degradation Chain</h2>
  <div class="card">
    {no_deg_msg}
    {f'<table><thead><tr><th>#</th><th>Degradation</th><th>Category</th><th>Confidence</th><th>Severity</th></tr></thead><tbody>{deg_rows}</tbody></table>' if result.degradations else ''}
  </div>

  {"<h2>Severity Overview</h2><div class='card'>" + chart_bars + "</div>" if result.degradations else ""}

  <h2>Color Histogram</h2>
  <div class="card hist">
    {histogram_svg}
  </div>

  <div class="footer">Generated by Artefex - Neural Forensic Restoration</div>
</div>
</body>
</html>"""

    return html


def _generate_histogram_svg(file_path: Path) -> str:
    """Generate an inline SVG histogram of the image's RGB channels."""
    try:
        img = Image.open(file_path).convert("RGB")
        arr = np.array(img)
    except Exception:
        return "<p style='color:var(--dim)'>Could not generate histogram</p>"

    width = 512
    height = 120
    max_val = 0

    channels = {
        "R": (arr[:, :, 0], "#f87171"),
        "G": (arr[:, :, 1], "#4ade80"),
        "B": (arr[:, :, 2], "#60a5fa"),
    }

    histograms = {}
    for name, (channel, color) in channels.items():
        hist, _ = np.histogram(channel, bins=256, range=(0, 256))
        histograms[name] = (hist, color)
        max_val = max(max_val, hist.max())

    if max_val == 0:
        return "<p style='color:var(--dim)'>Empty histogram</p>"

    svg_paths = ""
    for name, (hist, color) in histograms.items():
        points = []
        for i, val in enumerate(hist):
            x = i / 255 * width
            y = height - (val / max_val * height)
            points.append(f"{x:.1f},{y:.1f}")

        points_str = " ".join(points)
        svg_paths += f'<polyline points="{points_str}" fill="none" stroke="{color}" stroke-width="1.2" opacity="0.7"/>\n'

    svg = f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="max-width:100%">
    <rect width="{width}" height="{height}" fill="#0a0a12" rx="4"/>
    {svg_paths}
</svg>"""

    return svg
