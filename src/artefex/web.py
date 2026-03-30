"""Artefex web UI - drag-and-drop forensic analysis and restoration."""

import io
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from artefex.analyze import DegradationAnalyzer
from artefex.restore import RestorationPipeline
from artefex.report import render_report
from artefex.grade import compute_grade

app = FastAPI(title="Artefex", description="Neural forensic restoration")
analyzer = DegradationAnalyzer()
pipeline = RestorationPipeline()

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Artefex - Image Forensic Analysis</title>
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
    height: 100vh;
    overflow: hidden;
  }

  /* View system */
  .view { display: none; height: 100vh; }
  .view.active { display: flex; flex-direction: column; }

  /* View 1 - Upload */
  #view-upload {
    justify-content: center;
    align-items: center;
  }
  #view-upload .upload-inner {
    text-align: center;
    width: 100%;
    max-width: 560px;
    padding: 1rem;
  }
  #view-upload h1 {
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, var(--accent), #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  #view-upload .tagline {
    color: var(--dim);
    font-size: 0.85rem;
    margin-bottom: 2rem;
  }

  .dropzone {
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 2.5rem 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    background: var(--surface);
  }
  .dropzone:hover, .dropzone.dragover {
    border-color: var(--accent);
    background: rgba(124, 111, 247, 0.05);
  }
  .dropzone p { color: var(--dim); font-size: 0.9rem; }
  .dropzone .icon { font-size: 2.5rem; margin-bottom: 0.5rem; color: var(--accent); }

  /* Top bar (views 2/3) */
  .top-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 1rem;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    cursor: pointer;
  }
  .top-bar:hover { background: rgba(124, 111, 247, 0.05); }
  .top-bar .logo {
    font-size: 1rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .top-bar .filename {
    color: var(--dim);
    font-size: 0.8rem;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .top-bar .new-label {
    color: var(--accent);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* Status bar */
  .status {
    padding: 0.5rem 1rem;
    font-size: 0.8rem;
    display: none;
    flex-shrink: 0;
  }
  .status.visible { display: block; }
  .status.info {
    background: rgba(124, 111, 247, 0.1);
    border-bottom: 1px solid var(--accent);
  }
  .status.success {
    background: rgba(74, 222, 128, 0.1);
    border-bottom: 1px solid var(--green);
    color: var(--green);
  }
  .spinner {
    display: inline-block;
    width: 12px;
    height: 12px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
    margin-right: 0.4rem;
    vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* View 2 - Results split layout */
  .results-layout {
    display: flex;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }
  .results-left {
    width: 40%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
    background: var(--bg);
    border-right: 1px solid var(--border);
    overflow: hidden;
  }
  .results-left img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    border-radius: 6px;
  }
  .results-right {
    width: 60%;
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
  }

  /* Badges */
  .badges {
    display: none;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    flex-shrink: 0;
    flex-wrap: wrap;
    border-bottom: 1px solid var(--border);
  }
  .badges.visible { display: flex; }
  .badge {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.8rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    flex: 1;
    min-width: 160px;
  }
  .badge .badge-icon { font-size: 1.2rem; }
  .badge .badge-label {
    font-size: 0.65rem;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.8;
  }
  .badge-green {
    background: rgba(74, 222, 128, 0.12);
    border: 1px solid var(--green);
    color: var(--green);
  }
  .badge-yellow {
    background: rgba(251, 191, 36, 0.12);
    border: 1px solid var(--yellow);
    color: var(--yellow);
  }
  .badge-orange {
    background: rgba(251, 146, 60, 0.12);
    border: 1px solid #fb923c;
    color: #fb923c;
  }
  .badge-red {
    background: rgba(248, 113, 113, 0.12);
    border: 1px solid var(--red);
    color: var(--red);
  }

  /* Degradation table */
  .results {
    display: none;
    flex: 1;
    min-height: 0;
    overflow: hidden;
    flex-direction: column;
  }
  .results.visible { display: flex; }
  .results-header {
    padding: 0.5rem 1rem;
    border-bottom: 1px solid var(--border);
    font-weight: 600;
    font-size: 0.8rem;
    flex-shrink: 0;
    background: var(--surface);
  }
  .results-scroll {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
  }
  th {
    text-align: left;
    padding: 0.4rem 0.75rem;
    color: var(--dim);
    font-weight: 500;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    background: var(--surface);
  }
  td {
    padding: 0.4rem 0.75rem;
    border-bottom: 1px solid var(--border);
  }
  tr:last-child td { border-bottom: none; }
  .severity-high { color: var(--red); }
  .severity-mid { color: var(--yellow); }
  .severity-low { color: var(--green); }

  /* Actions bar */
  .actions {
    display: none;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    flex-shrink: 0;
    flex-wrap: wrap;
    border-top: 1px solid var(--border);
    background: var(--surface);
  }
  .actions.visible { display: flex; }
  .btn {
    padding: 0.5rem 1rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--surface);
    color: var(--text);
    font-family: inherit;
    font-size: 0.8rem;
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

  /* View 3a - Slider */
  .slider-view {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
  }
  .slider-container {
    display: none;
    position: relative;
    flex: 1;
    width: 100%;
    min-height: 0;
    overflow: hidden;
    cursor: col-resize;
    background: var(--bg);
  }
  .slider-container.visible { display: block; }
  .slider-container img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
  .slider-overlay {
    position: absolute;
    top: 0; left: 0; bottom: 0;
    overflow: hidden;
  }
  .slider-overlay img {
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
  }
  .slider-line {
    position: absolute;
    top: 0; bottom: 0;
    width: 2px;
    background: var(--accent);
    z-index: 10;
  }
  .slider-handle {
    position: absolute;
    top: 50%; transform: translateY(-50%);
    width: 28px; height: 28px;
    background: var(--accent);
    border: 2px solid #fff;
    border-radius: 50%;
    margin-left: -14px;
    z-index: 11;
  }
  .slider-labels {
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0.75rem;
    font-size: 0.7rem;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: var(--surface);
    border-top: 1px solid var(--border);
    flex-shrink: 0;
  }
  .slider-actions {
    display: flex;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    flex-shrink: 0;
    flex-wrap: wrap;
    background: var(--surface);
    border-top: 1px solid var(--border);
  }

  /* View 3b - Report */
  .report-view {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
  }
  .report-box {
    display: none;
    flex: 1;
    min-height: 0;
    background: var(--surface);
    padding: 1rem;
    font-size: 0.8rem;
    white-space: pre-wrap;
    overflow-y: auto;
    color: var(--dim);
    margin: 0.75rem 1rem;
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  .report-box.visible { display: block; }
  .report-actions {
    display: flex;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    flex-shrink: 0;
    flex-wrap: wrap;
    background: var(--surface);
    border-top: 1px solid var(--border);
  }

  /* Hidden helpers */
  .hidden { display: none !important; }
  #previewArea { display: none; }
  #restoredBox { display: none; }

  @media (max-width: 768px) {
    .results-layout {
      flex-direction: column;
    }
    .results-left {
      width: 100%;
      height: 35%;
      border-right: none;
      border-bottom: 1px solid var(--border);
    }
    .results-right {
      width: 100%;
      flex: 1;
    }
    .badges { flex-direction: column; }
    .actions.visible { flex-direction: column; }
    .slider-actions { flex-direction: column; }
    .report-actions { flex-direction: column; }
  }
</style>
</head>
<body>

<!-- View 1: Upload -->
<div class="view active" id="view-upload">
  <div class="upload-inner">
    <h1>artefex</h1>
    <p class="tagline">Forensic image analysis - detect AI content, trace image history, assess quality</p>
    <div class="dropzone" id="dropzone">
      <div class="icon">+</div>
      <p>Drop an image here to analyze it</p>
      <p style="margin-top:0.4rem;font-size:0.75rem;color:var(--dim)">
        Detects AI-generated content, traces platform history, identifies forgery</p>
      <input type="file" id="fileInput" accept="image/*"
        style="display:none">
    </div>
  </div>
</div>

<!-- View 2: Results -->
<div class="view" id="view-results">
  <div class="top-bar" onclick="newImage()">
    <span class="logo">artefex</span>
    <span class="filename" id="currentFilename"></span>
    <span class="new-label">Analyze another</span>
  </div>
  <div class="status" id="status"></div>
  <div class="results-layout">
    <div class="results-left">
      <img id="originalImg" alt="Original">
    </div>
    <div class="results-right">
      <div class="badges" id="badges">
        <div class="badge" id="aiBadge"></div>
        <div class="badge" id="gradeBadge"></div>
      </div>
      <div class="results" id="results">
        <div class="results-header">Findings</div>
        <div class="results-scroll">
          <table>
            <thead>
              <tr>
                <th>#</th><th>Degradation</th><th>Category</th>
                <th>Confidence</th><th>Severity</th>
              </tr>
            </thead>
            <tbody id="resultsBody"></tbody>
          </table>
        </div>
      </div>
      <div class="actions" id="actions">
        <button class="btn btn-primary" id="restoreBtn"
          onclick="restoreImage()">Clean Image</button>
        <button class="btn btn-primary" id="reportBtn"
          onclick="getReport()">Forensic Report</button>
      </div>
    </div>
  </div>
</div>

<!-- View 3a: Restore / Slider -->
<div class="view" id="view-restore">
  <div class="top-bar" onclick="newImage()">
    <span class="logo">artefex</span>
    <span class="filename" id="restoreFilename"></span>
    <span class="new-label">Analyze another</span>
  </div>
  <div class="status" id="restoreStatus"></div>
  <div class="slider-view">
    <div class="slider-container" id="sliderContainer">
      <img id="sliderBg" alt="Restored">
      <div class="slider-overlay" id="sliderOverlay">
        <img id="sliderFg" alt="Original">
      </div>
      <div class="slider-line" id="sliderLine"></div>
      <div class="slider-handle" id="sliderHandle"></div>
    </div>
    <div class="slider-labels">
      <span>Original</span>
      <span>Restored</span>
    </div>
    <div class="slider-actions">
      <button class="btn btn-primary" id="downloadBtn"
        onclick="downloadRestored()">Download Restored</button>
      <button class="btn" onclick="backToResults()">Back to Results</button>
      <button class="btn" onclick="getReport()">Forensic Report</button>
    </div>
  </div>
</div>

<!-- View 3b: Report -->
<div class="view" id="view-report">
  <div class="top-bar" onclick="newImage()">
    <span class="logo">artefex</span>
    <span class="filename" id="reportFilename"></span>
    <span class="new-label">Analyze another</span>
  </div>
  <div class="report-view">
    <div class="report-box visible" id="reportBox"></div>
    <div class="report-actions">
      <button class="btn btn-primary" id="copyReportBtn"
        onclick="copyReport()">Copy Report</button>
      <button class="btn" id="downloadReportBtn"
        onclick="downloadReport()">Download Report</button>
      <button class="btn" onclick="backToResults()">Back to Results</button>
    </div>
  </div>
</div>

<!-- Hidden elements for ID preservation -->
<div id="previewArea" style="display:none">
  <div id="restoredBox" style="display:none">
    <img id="restoredImg" alt="Restored">
  </div>
</div>

<script>
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
let currentFile = null;
let restoredBlob = null;

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', e => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function showView(id) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

function newImage() {
  showView('view-upload');
  fileInput.value = '';
  currentFile = null;
  restoredBlob = null;
  clearStatus();
}

function backToResults() {
  showView('view-results');
}

function setStatus(msg, type) {
  type = type || 'info';
  const el = document.getElementById('status');
  el.className = 'status visible ' + type;
  el.innerHTML = msg;
  // Also update restore status if on that view
  const rs = document.getElementById('restoreStatus');
  if (rs) {
    rs.className = 'status visible ' + type;
    rs.innerHTML = msg;
  }
}

function clearStatus() {
  document.getElementById('status').className = 'status';
  const rs = document.getElementById('restoreStatus');
  if (rs) rs.className = 'status';
}

async function handleFile(file) {
  currentFile = file;
  restoredBlob = null;

  // Set filenames in top bars
  var fname = file.name;
  document.getElementById('currentFilename').textContent = fname;
  document.getElementById('restoreFilename').textContent = fname;
  document.getElementById('reportFilename').textContent = fname;

  // Switch to results view
  showView('view-results');

  // Reset state
  document.getElementById('results').className = 'results';
  document.getElementById('actions').className = 'actions';
  document.getElementById('badges').className = 'badges';
  document.getElementById('reportBox').textContent = '';
  document.getElementById('sliderContainer').className = 'slider-container';

  // Analyze
  setStatus('<span class="spinner"></span> Analyzing image...');
  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: form });
    if (!res.ok) {
      const text = await res.text();
      try {
        const err = JSON.parse(text);
        setStatus(err.error || 'Server error', 'info');
      } catch(e) { setStatus('Server error: ' + text.slice(0, 100), 'info'); }
      return;
    }

    // Show preview
    const url = URL.createObjectURL(file);
    document.getElementById('originalImg').src = url;
    document.getElementById('restoredImg').src = '';

    const data = await res.json();

    const tbody = document.getElementById('resultsBody');
    tbody.innerHTML = '';

    if (data.degradations.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="color:var(--green)">' +
        'No degradation detected. Image looks clean.</td></tr>';
    } else {
      data.degradations.forEach((d, i) => {
        const sevClass = d.severity > 0.7 ? 'severity-high' :
          d.severity > 0.4 ? 'severity-mid' : 'severity-low';
        tbody.innerHTML += '<tr>' +
          '<td>' + (i+1) + '</td>' +
          '<td>' + d.name + '</td>' +
          '<td>' + d.category + '</td>' +
          '<td>' + Math.round(d.confidence * 100) + '%</td>' +
          '<td class="' + sevClass + '">' +
          Math.round(d.severity * 100) + '%</td></tr>';
      });
    }

    // AI detection badge
    const aiDeg = data.degradations.find(
      d => (d.name === 'AI-Generated Content' && d.confidence > 0.3)
        || (d.name === 'Device Identification'
            && d.detail && d.detail.includes('AI-generated'))
    );
    const aiBadge = document.getElementById('aiBadge');
    if (aiDeg) {
      const pct = Math.round(aiDeg.confidence * 100);
      aiBadge.className = aiDeg.confidence > 0.7
        ? 'badge badge-red' : 'badge badge-orange';
      aiBadge.innerHTML =
        '<span class="badge-icon">!</span>' +
        '<div><div>Likely AI-Generated (' + pct +
        '%)</div><div class="badge-label">' +
        'AI detection</div></div>';
    } else {
      aiBadge.className = 'badge badge-green';
      aiBadge.innerHTML =
        '<span class="badge-icon">&#10003;</span>' +
        '<div><div>Likely Real Photo</div>' +
        '<div class="badge-label">' +
        'AI detection</div></div>';
    }

    // Quality grade badge
    const gradeBadge = document.getElementById('gradeBadge');
    if (data.grade) {
      const gc = {'A':'badge-green','B':'badge-green',
        'C':'badge-yellow','D':'badge-orange',
        'F':'badge-red'}[data.grade] || 'badge-yellow';
      const gradeDesc = {'A':'Pristine - minimal processing detected',
        'B':'Good - light processing detected',
        'C':'Fair - moderate processing history',
        'D':'Poor - significant processing history',
        'F':'Heavy - extensive processing/degradation'}[data.grade] || '';
      gradeBadge.className = 'badge ' + gc;
      gradeBadge.innerHTML =
        '<span class="badge-icon">' + data.grade +
        '</span><div><div>Quality Score: ' +
        Math.round(data.score) + '/100</div>' +
        '<div class="badge-label">' +
        gradeDesc + '</div></div>';
      gradeBadge.style.display = '';
    } else {
      gradeBadge.style.display = 'none';
    }

    document.getElementById('badges').className = 'badges visible';

    // Show detector count: X findings from 13 checks
    const findingsHeader = document.querySelector('.results-header');
    if (findingsHeader) {
      const count = data.degradations.length;
      findingsHeader.textContent =
        count + ' Finding' + (count !== 1 ? 's' : '') +
        ' from 13 forensic checks';
    }

    document.getElementById('results').className = 'results visible';

    // Disable Clean button if nothing is cleanable
    const cleanable = data.degradations.filter(
      d => ['compression','noise','resolution','color','artifact']
        .includes(d.category) && d.confidence >= 0.3
    );
    const restoreBtn = document.getElementById('restoreBtn');
    if (cleanable.length === 0) {
      restoreBtn.disabled = true;
      restoreBtn.title =
        'No cleanable artifacts detected. ' +
        'Findings are informational (history/provenance).';
      restoreBtn.textContent = 'No Artifacts to Clean';
    } else {
      restoreBtn.disabled = false;
      restoreBtn.title = '';
      restoreBtn.textContent = 'Clean Image';
    }

    document.getElementById('actions').className = 'actions visible';
    clearStatus();
  } catch (err) {
    setStatus('Error analyzing image: ' + err.message, 'info');
  }
}

async function restoreImage() {
  if (!currentFile) return;
  const btn = document.getElementById('restoreBtn');
  btn.disabled = true;

  showView('view-restore');
  setStatus('<span class="spinner"></span> Restoring image...');

  const form = new FormData();
  form.append('file', currentFile);

  try {
    const res = await fetch('/api/restore', { method: 'POST', body: form });
    restoredBlob = await res.blob();
    const url = URL.createObjectURL(restoredBlob);
    const stepCount = parseInt(res.headers.get('X-Restore-Steps') || '0');
    const summary = res.headers.get('X-Restore-Summary') || '';
    const neural = res.headers.get('X-Restore-Neural') === 'True';

    document.getElementById('restoredImg').src = url;

    if (stepCount === 0) {
      setStatus(
        'Image is already clean - no artifacts need removal. ' +
        'The grade reflects image history, not current quality issues we can fix.',
        'success'
      );
      backToResults();
    } else {
      initSlider(document.getElementById('originalImg').src, url);
      const method = neural ? 'neural + classical' : 'classical';
      setStatus(
        'Cleaned ' + stepCount +
        ' artifact' + (stepCount > 1 ? 's' : '') +
        ' using ' + method + '. Drag the slider to compare original vs cleaned.',
        'success'
      );
    }
  } catch (err) {
    setStatus('Error restoring: ' + err.message, 'info');
  }
  btn.disabled = false;
}

async function getReport() {
  if (!currentFile) return;

  showView('view-report');
  document.getElementById('reportBox').textContent = '';
  setStatus('<span class="spinner"></span> Generating report...');

  const form = new FormData();
  form.append('file', currentFile);

  try {
    const res = await fetch('/api/report', { method: 'POST', body: form });
    const data = await res.json();
    let report = data.report;
    if (restoredBlob) {
      report = report.replace(
        /Run `artefex restore.*$/m,
        'Restoration has been applied.'
      );
    }
    document.getElementById('reportBox').textContent = report;
    clearStatus();
  } catch (err) {
    setStatus('Error generating report: ' + err.message, 'info');
  }
}

function copyReport() {
  const text = document.getElementById('reportBox').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copyReportBtn');
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy Report'; }, 2000);
  });
}

function downloadReport() {
  const text = document.getElementById('reportBox').textContent;
  const blob = new Blob([text], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  const name = currentFile.name.replace(/\\.[^.]+$/, '') + '_report.txt';
  a.download = name;
  a.click();
}

function downloadRestored() {
  if (!restoredBlob) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(restoredBlob);
  const name = currentFile.name.replace(/\\.[^.]+$/, '') + '_restored.png';
  a.download = name;
  a.click();
}

function initSlider(origUrl, restoredUrl) {
  const container = document.getElementById('sliderContainer');
  const overlay = document.getElementById('sliderOverlay');
  const line = document.getElementById('sliderLine');
  const handle = document.getElementById('sliderHandle');

  document.getElementById('sliderBg').src = restoredUrl;
  document.getElementById('sliderFg').src = origUrl;
  container.classList.add('visible');

  const setPos = (pct) => {
    overlay.style.width = pct + '%';
    line.style.left = pct + '%';
    handle.style.left = pct + '%';
  };
  setPos(50);

  let dragging = false;
  const onMove = (e) => {
    if (!dragging) return;
    const rect = container.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setPos(pct);
  };
  container.addEventListener('mousedown', () => { dragging = true; });
  container.addEventListener('touchstart', () => { dragging = true; });
  document.addEventListener('mouseup', () => { dragging = false; });
  document.addEventListener('touchend', () => { dragging = false; });
  container.addEventListener('mousemove', onMove);
  container.addEventListener('touchmove', onMove);
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...)):
    suffix = _get_suffix(file.filename)
    allowed = {
        ".jpg", ".jpeg", ".png", ".bmp",
        ".tiff", ".tif", ".webp", ".gif",
    }
    if suffix.lower() not in allowed:
        return JSONResponse(
            {"error": f"Unsupported file type: {suffix}. "
             "Please upload an image (JPG, PNG, etc)."},
            status_code=400,
        )

    contents = await file.read()
    with tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False
    ) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        result = analyzer.analyze(tmp_path)
        grade_info = compute_grade(result)
        return JSONResponse({
            "file": file.filename,
            "format": result.file_format,
            "dimensions": list(result.dimensions),
            "overall_severity": round(
                result.overall_severity, 3
            ),
            "grade": grade_info["grade"],
            "score": round(grade_info["score"], 1),
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
        info = pipeline.restore(tmp_path, result, out_path)

        steps = info.get("steps", [])
        active = [
            s for s in steps
            if "[classical]" in s or "[neural]" in s
        ]
        used_neural = info.get("used_neural", False)

        img_bytes = out_path.read_bytes()
        # Encode step info in response headers for JS to read
        step_summary = "; ".join(active) if active else "none"
        headers = {
            "Content-Disposition": (
                "attachment; filename="
                f"{Path(file.filename).stem}_restored.png"
            ),
            "X-Restore-Steps": str(len(active)),
            "X-Restore-Summary": step_summary[:200],
            "X-Restore-Neural": str(used_neural),
        }
        return StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/png",
            headers=headers,
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
