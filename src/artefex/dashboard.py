"""Batch HTML dashboard - overview of all images in a directory."""

import base64
import io
from pathlib import Path

from PIL import Image

from artefex.analyze import DegradationAnalyzer
from artefex.grade import compute_grade


def generate_dashboard(
    files: list[Path],
    output_path: Path,
    on_progress=None,
) -> str:
    """Generate a single HTML dashboard summarizing all images.

    Returns the output path.
    """
    analyzer = DegradationAnalyzer()
    entries = []

    for i, file in enumerate(files):
        result = analyzer.analyze(file)
        grade_info = compute_grade(result)

        # Thumbnail
        try:
            img = Image.open(file).convert("RGB")
            thumb = img.copy()
            thumb.thumbnail((120, 120))
            buf = io.BytesIO()
            thumb.save(buf, format="PNG")
            thumb_b64 = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            thumb_b64 = ""

        entries.append({
            "file": file.name,
            "thumb": thumb_b64,
            "grade": grade_info["grade"],
            "score": grade_info["score"],
            "color": grade_info["color"],
            "degradations": len(result.degradations),
            "severity": result.overall_severity,
            "top_issue": result.degradations[0].name if result.degradations else "Clean",
            "issues": [
                {"name": d.name, "severity": d.severity, "confidence": d.confidence}
                for d in result.degradations
            ],
        })

        if on_progress:
            on_progress(i + 1, len(files))

    # Compute summary stats
    avg_score = sum(e["score"] for e in entries) / len(entries) if entries else 0
    grade_dist = {}
    for e in entries:
        grade_dist[e["grade"]] = grade_dist.get(e["grade"], 0) + 1

    # Build rows
    rows = ""
    for e in entries:
        thumb_html = f"<img src='data:image/png;base64,{e['thumb']}' alt='{e['file']}'>" if e["thumb"] else ""
        issue_tags = ""
        for iss in e["issues"][:3]:
            sev_cls = "high" if iss["severity"] > 0.7 else "mid" if iss["severity"] > 0.4 else "low"
            issue_tags += f"<span class='tag {sev_cls}'>{iss['name']}</span> "

        rows += f"""
        <tr>
            <td class="thumb-cell">{thumb_html}</td>
            <td><strong>{e['file']}</strong></td>
            <td class="grade-{e['color']}">{e['grade']}</td>
            <td>{e['score']}</td>
            <td>{e['degradations']}</td>
            <td>{issue_tags or '<span class="tag low">Clean</span>'}</td>
        </tr>"""

    # Grade distribution chart
    dist_bars = ""
    for g in ["A", "B", "C", "D", "F"]:
        count = grade_dist.get(g, 0)
        pct = (count / len(entries) * 100) if entries else 0
        color_map = {"A": "#4ade80", "B": "#86efac", "C": "#fbbf24", "D": "#fb923c", "F": "#f87171"}
        dist_bars += f"""
        <div class="dist-row">
            <span class="dist-label">{g}</span>
            <div class="dist-track"><div class="dist-fill" style="width:{pct}%;background:{color_map[g]}"></div></div>
            <span class="dist-count">{count}</span>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Artefex Dashboard - {len(entries)} Images</title>
<style>
  :root {{ --bg:#0a0a0f; --surface:#12121a; --border:#1e1e2e; --text:#e0e0e8; --dim:#6b6b80; --accent:#7c6ff7; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'SF Mono','Cascadia Code',monospace; background:var(--bg); color:var(--text); padding:2rem; }}
  .container {{ max-width:1100px; margin:0 auto; }}
  h1 {{ font-size:1.5rem; background:linear-gradient(135deg,var(--accent),#a78bfa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  .meta {{ color:var(--dim); font-size:0.8rem; margin:0.5rem 0 1.5rem; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:1rem; margin-bottom:1.5rem; }}
  .stat {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:1rem; text-align:center; }}
  .stat .value {{ font-size:1.8rem; font-weight:700; }}
  .stat .label {{ font-size:0.7rem; color:var(--dim); text-transform:uppercase; margin-top:0.25rem; }}
  .card {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:1rem; margin-bottom:1.5rem; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
  th {{ text-align:left; padding:0.5rem; color:var(--dim); font-size:0.7rem; text-transform:uppercase; border-bottom:1px solid var(--border); }}
  td {{ padding:0.5rem; border-bottom:1px solid var(--border); vertical-align:middle; }}
  .thumb-cell img {{ width:48px; height:48px; object-fit:cover; border-radius:4px; }}
  .grade-green {{ color:#4ade80; font-size:1.2rem; font-weight:700; }}
  .grade-yellow {{ color:#fbbf24; font-size:1.2rem; font-weight:700; }}
  .grade-red {{ color:#f87171; font-size:1.2rem; font-weight:700; }}
  .tag {{ display:inline-block; padding:0.15rem 0.4rem; border-radius:3px; font-size:0.7rem; margin:0.1rem; }}
  .tag.high {{ background:rgba(248,113,113,0.15); color:#f87171; }}
  .tag.mid {{ background:rgba(251,191,36,0.15); color:#fbbf24; }}
  .tag.low {{ background:rgba(74,222,128,0.15); color:#4ade80; }}
  .dist-row {{ display:flex; align-items:center; gap:0.5rem; margin:0.3rem 0; }}
  .dist-label {{ width:20px; text-align:center; font-weight:600; }}
  .dist-track {{ flex:1; height:20px; background:var(--border); border-radius:3px; overflow:hidden; }}
  .dist-fill {{ height:100%; border-radius:3px; }}
  .dist-count {{ width:30px; text-align:right; color:var(--dim); }}
  .footer {{ text-align:center; color:var(--dim); font-size:0.7rem; margin-top:2rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>artefex dashboard</h1>
  <p class="meta">{len(entries)} images analyzed</p>

  <div class="stats">
    <div class="stat"><div class="value">{len(entries)}</div><div class="label">Images</div></div>
    <div class="stat"><div class="value">{avg_score:.0f}</div><div class="label">Avg Score</div></div>
    <div class="stat"><div class="value">{grade_dist.get('A', 0) + grade_dist.get('B', 0)}</div><div class="label">Good (A/B)</div></div>
    <div class="stat"><div class="value">{grade_dist.get('D', 0) + grade_dist.get('F', 0)}</div><div class="label">Poor (D/F)</div></div>
  </div>

  <h2 style="font-size:0.9rem;color:var(--accent);margin-bottom:0.75rem">Grade Distribution</h2>
  <div class="card">
    {dist_bars}
  </div>

  <h2 style="font-size:0.9rem;color:var(--accent);margin-bottom:0.75rem">All Images</h2>
  <div class="card" style="overflow-x:auto">
    <table>
      <thead><tr><th></th><th>File</th><th>Grade</th><th>Score</th><th>Issues</th><th>Detected</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>

  <div class="footer">Generated by Artefex - Neural Forensic Restoration</div>
</div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return str(output_path)
