import logging
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

log = logging.getLogger("app.api.v1.visualize")
router = APIRouter(tags=["Visual UI"])

_DEBUG_POST_URL = "/api/v1/pipeline/match_stats/debug"


_HTML_RAW = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LazarTrack · Pipeline Visualizer</title>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #111827;
    --panel-2: #0f172a;
    --border: #1f2937;
    --text: #e5e7eb;
    --muted: #94a3b8;
    --accent: #6366f1;
    --accent-2: #22d3ee;
    --good: #22c55e;
    --warn: #f59e0b;
    --bad: #ef4444;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: var(--sans); }
  header {
    padding: 20px 28px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(99,102,241,0.08), transparent);
  }
  header h1 { margin: 0 0 4px 0; font-size: 20px; letter-spacing: 0.3px; }
  header p { margin: 0; color: var(--muted); font-size: 13px; }
  main { max-width: 1400px; margin: 0 auto; padding: 24px 28px 80px; }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 20px;
  }
  .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }
  label.file-label {
    display: inline-flex; align-items: center; gap: 10px;
    padding: 10px 14px; border-radius: 10px; border: 1px dashed #374151;
    color: var(--muted); cursor: pointer; font-size: 14px;
    transition: border-color .15s, color .15s, background .15s;
  }
  label.file-label:hover { border-color: var(--accent); color: var(--text); background: rgba(99,102,241,0.05); }
  input[type=file] { display: none; }
  .file-name { color: var(--muted); font-size: 13px; max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  button.primary {
    background: var(--accent); color: white; border: none;
    padding: 10px 18px; border-radius: 10px; font-size: 14px; font-weight: 600;
    cursor: pointer; transition: transform .1s, background .15s, box-shadow .15s;
    box-shadow: 0 6px 16px rgba(99,102,241,0.25);
  }
  button.primary:hover:not(:disabled) { background: #4f46e5; }
  button.primary:active:not(:disabled) { transform: translateY(1px); }
  button.primary:disabled { opacity: 0.5; cursor: not-allowed; }
  .spinner {
    width: 14px; height: 14px; display: inline-block; border: 2px solid rgba(255,255,255,0.35);
    border-top-color: white; border-radius: 50%; animation: spin 0.8s linear infinite;
    margin-right: 8px; vertical-align: -2px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: 12px; }
  .stat-box {
    background: var(--panel-2); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px;
  }
  .stat-box .k { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
  .stat-box .v { font-size: 20px; font-weight: 700; margin-top: 4px; }
  .stat-box .v.ok { color: var(--good); }
  .stat-box .v.warn { color: var(--warn); }
  h2.section { margin: 8px 0 14px; font-size: 15px; letter-spacing: 0.3px; color: var(--muted); text-transform: uppercase; }
  h3.step-title {
    margin: 0 0 10px; font-size: 15px; display: flex; align-items: center; gap: 10px;
  }
  .step-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 8px;
    background: rgba(99,102,241,0.18); color: #a5b4fc; font-weight: 700; font-size: 13px;
  }
  .step-meta { margin-left: auto; color: var(--muted); font-family: var(--mono); font-size: 12px; }
  .step-images { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
  .img-frame {
    background: #000; border: 1px solid var(--border); border-radius: 10px; padding: 8px;
    display: flex; flex-direction: column;
  }
  .img-frame img {
    width: 100%; height: auto; display: block; border-radius: 6px;
    image-rendering: -webkit-optimize-contrast;
  }
  .img-frame .caption { margin-top: 6px; font-size: 11px; color: var(--muted); font-family: var(--mono); word-break: break-all; }
  .step-grid { display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); }
  table.players { width: 100%; border-collapse: collapse; font-size: 13px; }
  table.players th, table.players td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; }
  table.players th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
  table.players tr:last-child td { border-bottom: none; }
  table.players td.num, table.players th.num { text-align: right; font-family: var(--mono); }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; font-family: var(--mono); }
  .pill.good { background: rgba(34,197,94,0.12); color: #86efac; }
  .pill.warn { background: rgba(245,158,11,0.12); color: #fcd34d; }
  .pill.bad  { background: rgba(239,68,68,0.12); color: #fca5a5; }
  .pill.info { background: rgba(34,211,238,0.12); color: #67e8f9; }
  details.cell {
    background: var(--panel-2); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 12px; margin: 6px 0;
  }
  details.cell > summary {
    cursor: pointer; font-family: var(--mono); font-size: 13px;
    display: flex; align-items: center; gap: 10px; list-style: none;
  }
  details.cell > summary::-webkit-details-marker { display: none; }
  .cell-key { font-weight: 700; color: var(--accent-2); min-width: 160px; }
  .cell-final { margin-left: auto; font-family: var(--mono); font-size: 13px; }
  .cell-body { margin-top: 10px; display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); }
  .cell-body img { width: 100%; height: auto; border: 1px solid var(--border); border-radius: 6px; display: block; }
  .cell-body .meta { font-size: 12px; color: var(--muted); font-family: var(--mono); }
  .cell-body .ocr-texts { font-size: 12px; color: var(--text); font-family: var(--mono); }
  pre.raw-json {
    max-height: 480px; overflow: auto; background: #05070c; border: 1px solid var(--border);
    border-radius: 10px; padding: 14px; font-family: var(--mono); font-size: 12px; line-height: 1.55;
    white-space: pre-wrap; word-break: break-word;
  }
  .kv { display: grid; grid-template-columns: 140px 1fr; gap: 6px 14px; font-size: 13px; }
  .kv .k { color: var(--muted); }
  .kv .v { font-family: var(--mono); }
  .bar { height: 6px; background: var(--border); border-radius: 999px; overflow: hidden; margin-top: 10px; }
  .bar > span { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
  .hidden { display: none !important; }
  .error {
    background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.35); color: #fecaca;
    padding: 12px 14px; border-radius: 10px; font-family: var(--mono); font-size: 13px;
  }
  .player-card {
    background: var(--panel-2); border: 1px solid var(--border); border-radius: 12px; padding: 14px;
  }
  .player-head { display: flex; gap: 12px; align-items: center; margin-bottom: 10px; }
  .avatar {
    width: 40px; height: 40px; border-radius: 10px; display: inline-flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 15px; background: linear-gradient(135deg, #6366f1, #22d3ee); color: #0b0f17;
  }
  .player-head .name { font-weight: 700; font-size: 15px; }
  .player-head .subtitle { font-size: 12px; color: var(--muted); font-family: var(--mono); }
  .tag-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 2px; }
</style>
</head>
<body>
<header>
  <h1>🪖 LazarTrack · Pipeline Visualizer</h1>
  <p>Upload a Free Fire match scoreboard screenshot → see every step of the 5-step OCR pipeline with images, crops, bounding boxes, and extracted stats.</p>
</header>
<main>
  <section class="card">
    <div class="row">
      <label class="file-label" for="file">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        <span>Choose screenshot (PNG / JPEG / WEBP)</span>
      </label>
      <input id="file" type="file" accept="image/png,image/jpeg,image/webp" />
      <div class="file-name" id="fileName">No file selected</div>
      <div style="flex:1"></div>
      <button id="runBtn" class="primary" disabled>Run 5-Step Pipeline</button>
    </div>
  </section>

  <section id="statusCard" class="card hidden">
    <div class="row">
      <span id="statusText" class="step-meta" style="font-size:14px;">Initializing…</span>
      <div class="bar" style="flex:1; margin: 0 0 0 16px;"><span id="progressBar" style="width:2%"></span></div>
    </div>
  </section>

  <section id="errorSection" class="hidden"></section>

  <section id="resultSection" class="hidden">
    <div class="card">
      <h2 class="section">Result Summary</h2>
      <div class="summary-grid" id="summaryGrid"></div>
    </div>

    <div class="card">
      <h2 class="section">Extracted Players</h2>
      <table class="players" id="playersTable">
        <thead>
          <tr>
            <th>#</th><th>In-Game Name</th>
            <th class="num">K</th><th class="num">D</th><th class="num">A</th>
            <th class="num">Damage</th><th class="num">Act.Dmg</th>
            <th class="num">Knocked</th><th class="num">Heal</th>
            <th class="num">Help Up</th><th class="num">Revived</th>
            <th class="num">HSR</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="card">
      <h2 class="section">Pipeline Steps · Image Crops &amp; Bounding Boxes</h2>
      <div id="stepsContainer"></div>
    </div>

    <div class="card">
      <h2 class="section">Per-Player Debug (Cells, Scaled Crops, Raw OCR)</h2>
      <div id="playerDebugContainer"></div>
    </div>

    <div class="card">
      <h2 class="section">Metadata</h2>
      <div class="kv" id="metaKV"></div>
    </div>

    <div class="card">
      <h2 class="section">Raw JSON Response</h2>
      <pre class="raw-json" id="rawJson"></pre>
    </div>
  </section>
</main>

<script>
const fileInput = document.getElementById('file');
const fileName = document.getElementById('fileName');
const runBtn = document.getElementById('runBtn');
const statusCard = document.getElementById('statusCard');
const statusText = document.getElementById('statusText');
const progressBar = document.getElementById('progressBar');
const errorSection = document.getElementById('errorSection');
const resultSection = document.getElementById('resultSection');

let currentFile = null;

fileInput.addEventListener('change', (e) => {
  const f = e.target.files && e.target.files[0];
  currentFile = f || null;
  fileName.textContent = f ? `${f.name} · ${(f.size/1024).toFixed(1)} KB` : 'No file selected';
  runBtn.disabled = !f;
});

runBtn.addEventListener('click', () => { if (currentFile) run(currentFile); });

function setStatus(text, pct) {
  statusCard.classList.remove('hidden');
  statusText.textContent = text;
  if (pct != null) progressBar.style.width = pct + '%';
}
function showError(msg) {
  errorSection.innerHTML = `<div class="card"><div class="error">${escapeHtml(msg)}</div></div>`;
  errorSection.classList.remove('hidden');
  resultSection.classList.add('hidden');
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function run(file) {
  errorSection.classList.add('hidden');
  resultSection.classList.add('hidden');
  runBtn.disabled = true;
  try {
    setStatus('Uploading image…', 5);
    const fd = new FormData();
    fd.append('file', file);
    setStatus('Running 5-step OCR pipeline (Step 1–5… may take 10–40s)', 12);
    const t0 = Date.now();
    const resp = await fetch(__DEBUG_POST_URL__, { method: 'POST', body: fd });
    const raw = await resp.text();
    let data;
    try { data = JSON.parse(raw); } catch { throw new Error(`Non-JSON response (HTTP ${resp.status}): ${raw.slice(0,500)}`); }
    if (resp.status !== 200) throw new Error(data.detail || ('HTTP ' + resp.status));
    setStatus('Rendering results…', 90);
    render(data);
    setStatus(`Done in ${((Date.now()-t0)/1000).toFixed(1)}s`, 100);
  } catch (e) {
    console.error(e);
    showError(String(e.stack || e));
  } finally {
    runBtn.disabled = !currentFile;
  }
}

function render(data) {
  resultSection.classList.remove('hidden');
  const result = data.result || data.data || {};
  const players = result.players || [];
  const metadata = result.metadata || {};
  const stepDurations = metadata.step_durations || {};
  const totalDur = Object.values(stepDurations).reduce((a,b)=>a+b,0) || 0;

  document.getElementById('summaryGrid').innerHTML = `
    <div class="stat-box"><div class="k">Pipeline</div><div class="v" style="font-size:16px">${escapeHtml(data.pipeline_name||'')}</div></div>
    <div class="stat-box"><div class="k">Map</div><div class="v info pill" style="display:inline-block;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(result.map||'Unknown')}</div></div>
    <div class="stat-box"><div class="k">Rank</div><div class="v ${(result.rank||0)<=3?'ok':'warn'}">#${result.rank||'?'}</div></div>
    <div class="stat-box"><div class="k">Teams</div><div class="v">${result.total_teams||'?'}</div></div>
    <div class="stat-box"><div class="k">Players</div><div class="v">${players.length}</div></div>
    <div class="stat-box"><div class="k">Confidence</div><div class="v">${((data.confidence||0)*100).toFixed(0)}%</div></div>
    <div class="stat-box"><div class="k">Pipeline Time</div><div class="v" style="font-size:16px">${totalDur.toFixed(2)}s</div></div>
  `;

  const tbody = document.querySelector('#playersTable tbody');
  tbody.innerHTML = players.map((p,i) => `
    <tr>
      <td>${i+1}</td>
      <td>${escapeHtml(p.inGameName||'')}</td>
      <td class="num">${p.kills||0}</td>
      <td class="num">${p.deaths||0}</td>
      <td class="num">${p.assists||0}</td>
      <td class="num">${p.damage||0}</td>
      <td class="num">${p.actualDamage||0}</td>
      <td class="num">${p.knockedDown||0}</td>
      <td class="num">${p.heal||0}</td>
      <td class="num">${p.helpUp||0}</td>
      <td class="num">${p.revival||0}</td>
      <td class="num">${(p.headShotRate||0)}%</td>
    </tr>
  `).join('');

  document.getElementById('stepsContainer').innerHTML =
    (data.steps || []).map(renderStep).join('');

  document.getElementById('playerDebugContainer').innerHTML =
    renderPlayerDebug(data.steps || []);

  const flatMeta = [];
  const pushKV = (k,v)=> flatMeta.push(`<div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(v)}</div>`);
  pushKV('detected_rank', metadata.detected_rank || '');
  pushKV('detected_tagline', metadata.detected_tagline || '');
  pushKV('pipeline_id', data.pipeline_id || '');
  pushKV('map_ocr_texts', (metadata.map_ocr_texts||[]).join(' | '));
  if (metadata.roi) {
    const r = metadata.roi;
    pushKV('roi', `x1=${r.x1} y1=${r.y1} x2=${r.x2} y2=${r.y2} anchors: top=${r.top_anchor_found} bottom=${r.bottom_anchor_found}`);
  }
  if (metadata.columns) {
    const c = metadata.columns;
    pushKV('table_size', `${c.width} × ${c.height} · ${c.num_columns} cols`);
  }
  pushKV('step_durations', Object.entries(stepDurations).map(([k,v])=>`${k}:${v}s`).join('  '));
  document.getElementById('metaKV').innerHTML = flatMeta.join('');

  document.getElementById('rawJson').textContent = JSON.stringify(data, null, 2);
}

function renderStep(step) {
  const n = step.step; const name = step.name; const dur = step.duration_seconds;
  const images = [];
  if (step.image) images.push({label:`Step ${n} output`, src: step.image});
  if (step.images && typeof step.images === 'object') {
    for (const k of Object.keys(step.images)) images.push({label:k, src:step.images[k]});
  }
  if (step.column_images && typeof step.column_images === 'object') {
    for (const k of Object.keys(step.column_images)) images.push({label:`col · ${k}`, src:step.column_images[k]});
  }
  const metas = [];
  if (step.width) metas.push(`w=${step.width}`);
  if (step.height) metas.push(`h=${step.height}`);
  if (step.target_width) metas.push(`target_w=${step.target_width}`);
  if (step.num_columns) metas.push(`cols=${step.num_columns}`);
  if (step.num_players != null) metas.push(`players=${step.num_players}`);
  return `
    <div class="card" style="background:var(--panel-2)">
      <h3 class="step-title">
        <span class="step-badge">${n}</span>
        <span>${escapeHtml(name)}</span>
        <span class="step-meta">${dur}s${metas.length?'  ·  '+metas.join('  ·  '):''}</span>
      </h3>
      ${step.meta?`<details style="margin-bottom:8px"><summary style="color:var(--muted);cursor:pointer;font-size:12px;font-family:var(--mono)">Step metadata JSON</summary><pre class="raw-json" style="margin-top:8px">${escapeHtml(JSON.stringify(step.meta,null,2))}</pre></details>`:''}
      <div class="step-images">${images.map(im => `
        <div class="img-frame">
          <img loading="lazy" src="${im.src}" alt="${escapeHtml(im.label)}" />
          <div class="caption">${escapeHtml(im.label)}</div>
        </div>
      `).join('') || '<div style="color:var(--muted);font-size:13px;padding:10px 4px">No step-level images. See per-player debug for cell-level crops & OCR.</div>'}</div>
    </div>
  `;
}

function renderPlayerDebug(steps) {
  const s5 = (steps || []).find(s => s.step === 5);
  if (!s5 || !s5.players || !s5.players.length) return '<div style="color:var(--muted)">No per-player debug rows in Step 5 response.</div>';
  return s5.players.map((p, idx) => {
    const label = p.label || `Player ${idx+1}`;
    const empty = !!p.empty;
    const nameCell = (p.cells||{})['col1_name'] || {};
    const kdaCell  = (p.cells||{})['col1_kda']  || {};
    const statsCells = Object.entries(p.cells||{}).filter(([k])=>k.startsWith('col') && k !== 'col1_name' && k !== 'col1_kda');
    const finalData = p.final || {};
    const initials = (finalData.inGameName||label).replace(/[^A-Za-z0-9]/g,'').slice(0,2).toUpperCase();
    return `
      <div class="player-card" style="margin-bottom:12px">
        <div class="player-head">
          <div class="avatar">${escapeHtml(initials)}</div>
          <div>
            <div class="name">${escapeHtml(label)} ${empty?'<span class="pill info">empty slot</span>':''}</div>
            <div class="subtitle">y=[${p.y_start}..${p.y_end}] · preflight_std=${(p.preflight_std||0).toFixed(1)} · ocr_items=${p.preflight_ocr_items||0}</div>
            ${!empty && finalData.inGameName ? `<div class="tag-chips">
              <span class="pill info">${escapeHtml(finalData.inGameName)}</span>
              <span class="pill good">K=${finalData.kills||0} D=${finalData.deaths||0} A=${finalData.assists||0}</span>
              <span class="pill warn">Dmg=${finalData.damage||0} / Act=${finalData.actualDamage||0}</span>
              <span class="pill info">Heal=${finalData.heal||0} Help=${finalData.helpUp||0} Rev=${finalData.revival||0}</span>
              <span class="pill info">Knock=${finalData.knockedDown||0} HSR=${finalData.headShotRate||0}%</span>
            </div>`:''}
          </div>
        </div>
        ${empty?'':renderCell('col1_name (Player Name)', nameCell) + renderCell('col1_kda (K/D/A Ratio)', kdaCell) +
          statsCells.map(([k,c])=>renderCell(k,c)).join('')}
      </div>
    `;
  }).join('');
}

function renderCell(title, cell) {
  if (!cell) return '';
  const imgs = [];
  if (cell.raw_crop) imgs.push({label:'raw crop', src:cell.raw_crop});
  if (cell.variant_crops) {
    Object.entries(cell.variant_crops).forEach(([vname, vsrc]) => {
      imgs.push({label: `variant: ${vname}`, src: vsrc});
    });
  }
  if (cell.scaled_4x) imgs.push({label:'scaled 4x (name)', src:cell.scaled_4x});
  if (cell.scaled_5x) imgs.push({label:'scaled 5x (HSR)', src:cell.scaled_5x});
  if (cell.scaled_6x) imgs.push({label:'scaled 6x (numeric)', src:cell.scaled_6x});
  return `
    <details class="cell" open>
      <summary>
        <span class="cell-key">${escapeHtml(title)}</span>
        ${cell.left_48_only?'<span class="pill info" style="margin-left:4px">left 48% only</span>':''}
        <span class="cell-final">→ ${escapeHtml(cell.final == null ? '' : String(cell.final))}</span>
      </summary>
      <div class="cell-body">
        ${imgs.map(im => `
          <div>
            <img loading="lazy" src="${im.src}" alt="${escapeHtml(im.label)}" />
            <div class="meta">${escapeHtml(im.label)}</div>
          </div>
        `).join('')}
        <div>
          ${cell.candidates ? `<div class="meta">Ensemble Candidates ([variant|interp|scale]):</div><div class="ocr-texts">${escapeHtml(JSON.stringify(cell.candidates, null, 2))}</div>` : ''}
          <div class="meta" style="margin-top:8px">Raw OCR texts:</div>
          <div class="ocr-texts">${escapeHtml(JSON.stringify(cell.ocr_raw || [], null, 2))}</div>
          ${cell.parsed_kda?`<div class="meta" style="margin-top:8px">Parsed K/D/A:</div><div class="ocr-texts">${escapeHtml(JSON.stringify(cell.parsed_kda))}</div>`:''}
          ${cell.error?`<div class="meta" style="margin-top:8px;color:#ef4444">Error</div><div class="ocr-texts" style="color:#fca5a5">${escapeHtml(cell.error)}</div>`:''}
        </div>
      </div>
    </details>
  `;
}
</script>
</body>
</html>
"""

_HTML = _HTML_RAW.replace("__DEBUG_POST_URL__", repr(_DEBUG_POST_URL))


@router.get("/visualize", response_class=HTMLResponse, include_in_schema=False)
async def visualize_ui():
    """Browser UI for the 5-step pipeline: upload a screenshot, see step images + all extracted data."""
    return HTMLResponse(content=_HTML)
