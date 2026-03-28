"""Artefex web UI - drag-and-drop forensic analysis and restoration."""

import io
import base64
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from artefex.analyze import DegradationAnalyzer
from artefex.restore import RestorationPipeline
from artefex.report import render_report

app = FastAPI(title="Artefex", description="Neural forensic restoration")
analyzer = DegradationAnalyzer()
pipeline = RestorationPipeline()

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Artefex - Neural Forensic Restoration</title>
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --border: #1e1e2e;
    --text: #e0e0e8;
    --dim: #6b6b80;
    --accent: #7c6ff7;
    --accent-dim: #5a4fd4;
    --green: #4ade80;
    --yellow: #fbbf24;
    --red: #f87171;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }
  .container { max-width: 900px; margin: 0 auto; padding: 2rem 1rem; }
  h1 {
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    background: linear-gradient(135deg, var(--accent), #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .subtitle { color: var(--dim); font-size: 0.85rem; margin-bottom: 2rem; }

  .dropzone {
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    background: var(--surface);
    margin-bottom: 1.5rem;
  }
  .dropzone:hover, .dropzone.dragover {
    border-color: var(--accent);
    background: rgba(124, 111, 247, 0.05);
  }
  .dropzone p { color: var(--dim); font-size: 0.9rem; }
  .dropzone .icon { font-size: 2.5rem; margin-bottom: 0.5rem; }

  .preview-area {
    display: none;
    gap: 1rem;
    margin-bottom: 1.5rem;
  }
  .preview-area.visible { display: grid; grid-template-columns: 1fr 1fr; }
  .preview-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  .preview-box .label {
    padding: 0.5rem 0.75rem;
    font-size: 0.75rem;
    color: var(--dim);
    border-bottom: 1px solid var(--border);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .preview-box img {
    width: 100%;
    display: block;
    image-rendering: auto;
  }

  .results {
    display: none;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 1.5rem;
  }
  .results.visible { display: block; }
  .results-header {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
    font-weight: 600;
    font-size: 0.9rem;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }
  th {
    text-align: left;
    padding: 0.5rem 1rem;
    color: var(--dim);
    font-weight: 500;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
  }
  td {
    padding: 0.5rem 1rem;
    border-bottom: 1px solid var(--border);
  }
  tr:last-child td { border-bottom: none; }

  .severity-high { color: var(--red); }
  .severity-mid { color: var(--yellow); }
  .severity-low { color: var(--green); }

  .actions {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
  }
  .btn {
    padding: 0.6rem 1.2rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--surface);
    color: var(--text);
    font-family: inherit;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn:hover { border-color: var(--accent); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }
  .btn-primary:hover { background: var(--accent-dim); }

  .status {
    padding: 0.75rem 1rem;
    border-radius: 6px;
    font-size: 0.85rem;
    margin-bottom: 1rem;
    display: none;
  }
  .status.visible { display: block; }
  .status.info { background: rgba(124, 111, 247, 0.1); border: 1px solid var(--accent); }
  .status.success { background: rgba(74, 222, 128, 0.1); border: 1px solid var(--green); color: var(--green); }

  .spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
    margin-right: 0.5rem;
    vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .report-box {
    display: none;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    font-size: 0.8rem;
    white-space: pre-wrap;
    max-height: 400px;
    overflow-y: auto;
    color: var(--dim);
  }
  .report-box.visible { display: block; }

  @media (max-width: 600px) {
    .preview-area.visible { grid-template-columns: 1fr; }
    .actions { flex-direction: column; }
  }
</style>
</head>
<body>
<div class="container">
  <h1>artefex</h1>
  <p class="subtitle">Neural forensic restoration - diagnose and reverse media degradation chains</p>

  <div class="dropzone" id="dropzone">
    <div class="icon">+</div>
    <p>Drop an image here or click to select</p>
    <input type="file" id="fileInput" accept="image/*" style="display:none">
  </div>

  <div class="status" id="status"></div>

  <div class="preview-area" id="previewArea">
    <div class="preview-box">
      <div class="label">Original</div>
      <img id="originalImg" alt="Original">
    </div>
    <div class="preview-box">
      <div class="label">Restored</div>
      <img id="restoredImg" alt="Restored">
    </div>
  </div>

  <div class="results" id="results">
    <div class="results-header">Degradation Chain</div>
    <table>
      <thead>
        <tr><th>#</th><th>Degradation</th><th>Category</th><th>Confidence</th><th>Severity</th></tr>
      </thead>
      <tbody id="resultsBody"></tbody>
    </table>
  </div>

  <div class="actions" id="actions" style="display:none">
    <button class="btn btn-primary" id="restoreBtn" onclick="restoreImage()">Restore Image</button>
    <button class="btn" id="reportBtn" onclick="getReport()">Forensic Report</button>
    <button class="btn" id="downloadBtn" style="display:none" onclick="downloadRestored()">Download Restored</button>
  </div>

  <div class="report-box" id="reportBox"></div>
</div>

<script>
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
let currentFile = null;
let restoredBlob = null;

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

function setStatus(msg, type='info') {
  const el = document.getElementById('status');
  el.className = 'status visible ' + type;
  el.innerHTML = msg;
}
function clearStatus() {
  document.getElementById('status').className = 'status';
}

async function handleFile(file) {
  currentFile = file;
  restoredBlob = null;

  // Show original preview
  const url = URL.createObjectURL(file);
  document.getElementById('originalImg').src = url;
  document.getElementById('restoredImg').src = '';
  document.getElementById('previewArea').className = 'preview-area visible';
  document.getElementById('downloadBtn').style.display = 'none';
  document.getElementById('reportBox').className = 'report-box';

  // Analyze
  setStatus('<span class="spinner"></span> Analyzing image...');
  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: form });
    const data = await res.json();

    const tbody = document.getElementById('resultsBody');
    tbody.innerHTML = '';

    if (data.degradations.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="color:var(--green)">No degradation detected. Image looks clean.</td></tr>';
    } else {
      data.degradations.forEach((d, i) => {
        const sevClass = d.severity > 0.7 ? 'severity-high' : d.severity > 0.4 ? 'severity-mid' : 'severity-low';
        tbody.innerHTML += '<tr>' +
          '<td>' + (i+1) + '</td>' +
          '<td>' + d.name + '</td>' +
          '<td>' + d.category + '</td>' +
          '<td>' + Math.round(d.confidence * 100) + '%</td>' +
          '<td class="' + sevClass + '">' + Math.round(d.severity * 100) + '%</td>' +
          '</tr>';
      });
    }

    document.getElementById('results').className = 'results visible';
    document.getElementById('actions').style.display = 'flex';
    clearStatus();
  } catch (err) {
    setStatus('Error analyzing image: ' + err.message, 'info');
  }
}

async function restoreImage() {
  if (!currentFile) return;
  const btn = document.getElementById('restoreBtn');
  btn.disabled = true;
  setStatus('<span class="spinner"></span> Restoring image...');

  const form = new FormData();
  form.append('file', currentFile);

  try {
    const res = await fetch('/api/restore', { method: 'POST', body: form });
    restoredBlob = await res.blob();
    const url = URL.createObjectURL(restoredBlob);
    document.getElementById('restoredImg').src = url;
    document.getElementById('downloadBtn').style.display = 'inline-block';
    setStatus('Restoration complete.', 'success');
  } catch (err) {
    setStatus('Error restoring: ' + err.message, 'info');
  }
  btn.disabled = false;
}

async function getReport() {
  if (!currentFile) return;
  setStatus('<span class="spinner"></span> Generating report...');

  const form = new FormData();
  form.append('file', currentFile);

  try {
    const res = await fetch('/api/report', { method: 'POST', body: form });
    const data = await res.json();
    document.getElementById('reportBox').textContent = data.report;
    document.getElementById('reportBox').className = 'report-box visible';
    clearStatus();
  } catch (err) {
    setStatus('Error generating report: ' + err.message, 'info');
  }
}

function downloadRestored() {
  if (!restoredBlob) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(restoredBlob);
  const name = currentFile.name.replace(/\\.[^.]+$/, '') + '_restored.png';
  a.download = name;
  a.click();
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=_get_suffix(file.filename), delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        result = analyzer.analyze(tmp_path)
        return JSONResponse({
            "file": file.filename,
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
        })
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/restore")
async def api_restore(file: UploadFile = File(...)):
    contents = await file.read()
    suffix = _get_suffix(file.filename)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    out_path = tmp_path.with_stem(tmp_path.stem + "_restored").with_suffix(".png")

    try:
        result = analyzer.analyze(tmp_path)
        pipeline.restore(tmp_path, result, out_path)

        img_bytes = out_path.read_bytes()
        return StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={Path(file.filename).stem}_restored.png"},
        )
    finally:
        tmp_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


@app.post("/api/report")
async def api_report(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=_get_suffix(file.filename), delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        result = analyzer.analyze(tmp_path)
        report_text = render_report(tmp_path, result)
        return JSONResponse({"report": report_text})
    finally:
        tmp_path.unlink(missing_ok=True)


def _get_suffix(filename: str) -> str:
    if filename:
        return Path(filename).suffix or ".jpg"
    return ".jpg"
