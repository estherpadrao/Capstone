/* =========================================================
   STATE
   ========================================================= */
let _hasPCI = false, _hasBCI = false;
// Set only after the viz payload has been applied — guards compare auto-run
let _pciVizDone = false, _bciVizDone = false;
let _activeTab = 'about';

/* On page load, restore session flags but do NOT auto-run compare —
   compare only fires after both viz panels have been explicitly populated
   in this browser session. */
(async function restoreSessionFlags() {
  try {
    const r = await get('/api/session/status');
    if (r.has_pci) _hasPCI = true;
    if (r.has_bci) _hasBCI = true;
  } catch (e) { /* server not ready yet */ }
  showTab('about');
})();

/* =========================================================
   BROWSER NOTIFICATIONS
   ========================================================= */
if ('Notification' in window && Notification.permission === 'default') {
  Notification.requestPermission();
}

function notify(title, body) {
  if (!document.hidden) return;           // user is already on this tab
  if (!('Notification' in window)) return;
  if (Notification.permission !== 'granted') return;
  new Notification(title, { body });
}

/* =========================================================
   WEIGHT NORMALIZATION (Option C)
   Renormalize on blur; always normalize before sending.
   Tolerance: sums within [0.99, 1.01] are left untouched
   so the user doesn't need to hit exact decimals.
   ========================================================= */
const WEIGHT_GROUPS = {
  pci: ['w-health', 'w-edu', 'w-parks', 'w-community', 'w-food', 'w-transit'],
  bci: ['bci-w-market', 'bci-w-labour', 'bci-w-supplier'],
};
const WEIGHT_TOL = 0.01; // allow ±1 % around 1.0 before normalizing

function _normWeights(ids) {
  const vals  = ids.map(id => Math.max(0, parseFloat(document.getElementById(id)?.value) || 0));
  const total = vals.reduce((a, b) => a + b, 0);
  if (total === 0) return vals;
  return vals.map(v => v / total);
}

function normalizeWeightGroup(ids) {
  const vals  = ids.map(id => Math.max(0, parseFloat(document.getElementById(id)?.value) || 0));
  const total = vals.reduce((a, b) => a + b, 0);
  if (total === 0 || Math.abs(total - 1.0) <= WEIGHT_TOL) return; // close enough
  const normed = vals.map(v => Math.round((v / total) * 1000) / 1000); // 3 dp
  ids.forEach((id, i) => { const el = document.getElementById(id); if (el) el.value = normed[i]; });
}

/* =========================================================
   UTILS
   ========================================================= */
function syncVal(inputId, labelId) {
  document.getElementById(labelId).textContent = document.getElementById(inputId).value;
}

function markDirty() {
  // reserved for future use
}

function setStatus(msg, state) {
  document.getElementById('status-text').textContent = msg;
  const dot = document.getElementById('status-dot');
  dot.className = 'dot' + (state ? ' ' + state : '');
}

function showTab(name) {
  _activeTab = name;
  ['pci','bci','compare','diagnostics','sensitivity','about','scenario'].forEach(t => {
    document.getElementById('tab-' + t).classList.toggle('hidden', t !== name);
  });
  document.querySelectorAll('.tab').forEach(btn => {
    const tabName = btn.getAttribute('data-tab') ||
                    btn.onclick.toString().match(/showTab\('(\w+)'/)?.[1];
    btn.classList.toggle('active', tabName === name);
  });
  if (name === 'about') loadAbout();
}

function getUserParams() {
  // Amenity tag toggles
  const amenityTags = {};
  document.querySelectorAll('#tag-toggles .toggle-item[data-tag]').forEach(el => {
    amenityTags[el.dataset.tag] = el.querySelector('input').checked;
  });
  // Supplier tag toggles
  const supplierTags = {};
  document.querySelectorAll('#supplier-tag-toggles .toggle-item[data-tag]').forEach(el => {
    supplierTags[el.dataset.tag] = el.querySelector('input').checked;
  });

  return {
    hansen_beta:           parseFloat(document.getElementById('p-beta').value),
    active_street_lambda:  parseFloat(document.getElementById('p-lambda').value),
    amenity_weights: (() => {
      const [h, e, p, c, f, t] = _normWeights(WEIGHT_GROUPS.pci);
      return { health: h, education: e, parks: p, community: c, food_retail: f, transit: t };
    })(),
    enabled_amenity_tags:   amenityTags,
    enabled_supplier_tags:  supplierTags,
    beta_market:   parseFloat(document.getElementById('b-beta-m').value),
    beta_labour:   parseFloat(document.getElementById('b-beta-l').value),
    beta_supplier: parseFloat(document.getElementById('b-beta-s').value),
    interface_lambda: parseFloat(document.getElementById('b-lambda').value),
    bci_method:    document.getElementById('bci-method').value,
    ...(() => {
      const [m, l, s] = _normWeights(WEIGHT_GROUPS.bci);
      return { market_weight: m, labour_weight: l, supplier_weight: s };
    })(),
    use_urban_interface: true,
    mask_parks: false,
  };
}

async function post(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  return r.json();
}

async function get(url) {
  return (await fetch(url)).json();
}

function setMapSrc(frameId, html) {
  const frame = document.getElementById(frameId);
  if (!html) return;
  frame.srcdoc = html;
}

function setImg(id, b64) {
  if (!b64) return;
  document.getElementById(id).src = 'data:image/png;base64,' + b64;
}

function renderStats(containerId, stats, keyMap) {
  const el = document.getElementById(containerId);
  el.innerHTML = '';
  Object.entries(keyMap).forEach(([key, label]) => {
    const val = stats[key];
    if (val === undefined || val === null) return;
    const box = document.createElement('div');
    box.className = 'stat-box';
    const displayVal = typeof val === 'number' ? val.toLocaleString() : val;
    box.innerHTML = `<div class="val">${displayVal}</div><div class="lbl">${label}</div>`;
    el.appendChild(box);
  });
}

const PCI_STAT_MAP = {
  city_pci: 'City PCI', mean: 'Mean', median: 'Median',
  std: 'Std Dev', gini: 'Gini', n_hexagons: 'Hexagons',
  cv_pct: 'CV %', p25: '25th %ile', p75: '75th %ile'
};
const BCI_STAT_MAP = {
  mean: 'Mean BCI', median: 'Median', std: 'Std Dev',
  n_hexagons: 'Hexagons', n_hotspots: 'Hotspots',
  n_underserved: 'Underserved', cv_pct: 'CV %',
  corr_market_bci: 'r(Market)', corr_labour_bci: 'r(Labour)',
  corr_supplier_bci: 'r(Supplier)'
};
const CMP_STAT_MAP = {
  pearson_r: 'Pearson r', pearson_r2: 'R²',
  spearman_r: 'Spearman ρ', kendall_t: 'Kendall τ',
  quad_high_high: '↑↑ Both High', quad_low_low: '↓↓ Both Low',
  quad_high_low: '↑ PCI / ↓ BCI', quad_low_high: '↓ PCI / ↑ BCI'
};

/* =========================================================
   TOGGLE HANDLERS
   ========================================================= */
document.querySelectorAll('.toggle-item').forEach(el => {
  el.addEventListener('click', () => {
    const cb = el.querySelector('input');
    cb.checked = !cb.checked;
    el.classList.toggle('active', cb.checked);
    markDirty();
  });
});

document.getElementById('bci-method').addEventListener('change', function() {
  document.getElementById('bci-weight-inputs').classList.toggle('hidden', this.value !== 'weighted');
});

// Blur → renormalize each weight group so displayed values stay honest
Object.values(WEIGHT_GROUPS).forEach(ids => {
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('blur', () => normalizeWeightGroup(ids));
  });
});

/* =========================================================
   PCI PIPELINE
   ========================================================= */
async function runPCI() {
  const city = document.getElementById('city-select').value;
  const up   = getUserParams();
  const btn  = document.getElementById('btn-pci');
  btn.disabled = true;

  try {
    setStatus('Step 1/4 · Fetching boundary & amenities…', 'running');
    let r = await post('/api/pci/init', {city_name: city, user_params: up});
    if (r.status !== 'ok') throw new Error(r.message);

    setStatus('Step 2/4 · Building network & Census fetch…', 'running');
    r = await post('/api/pci/build_network', {});
    if (r.status !== 'ok') throw new Error(r.message);

    setStatus('Step 3/4 · Computing travel times & PCI…', 'running');
    r = await post('/api/pci/compute', {user_params: up});
    if (r.status !== 'ok') throw new Error(r.message);

    setStatus('Step 4/4 · Rendering visualisations…', 'running');
    r = await get('/api/pci/visualize');
    if (r.status !== 'ok') throw new Error(r.message);

    applyPCIViz(r);
    _hasPCI = true;
    setStatus(`✓ PCI complete — ${r.stats.n_hexagons} hexagons, city score: ${r.stats.city_pci}`, 'ok');
    notify('PCI Complete', `City score: ${r.stats.city_pci} · ${r.stats.n_hexagons} hexagons`);
    checkCompare();

  } catch(e) {
    console.error(e);
    setStatus('PCI error: ' + e.message, 'err');
  }
  btn.disabled = false;
  showTab('pci');
}

function applyPCIViz(r) {
  _pciVizDone = true;
  document.getElementById('pci-placeholder').classList.add('hidden');
  document.getElementById('pci-content').classList.remove('hidden');
  renderStats('pci-stats', r.stats, PCI_STAT_MAP);
  setImg('img-topo-layers',    r.topography_layers);
  setImg('img-topo-3d',        r.topography_3d);
  setImg('img-pci-components', r.pci_components);
  setImg('img-pci-dist',       r.pci_distribution);
  setMapSrc('map-pci',         r.pci_map);
  if (r.neighborhoods && r.neighborhoods.length) {
    renderNeighborhoodTable('pci-nb-wrap', 'pci-nb-table', r.neighborhoods, 'PCI');
  }
}

/* =========================================================
   BCI PIPELINE
   ========================================================= */
async function runBCI() {
  const city = document.getElementById('city-select').value;
  const up   = getUserParams();
  const btn  = document.getElementById('btn-bci');
  btn.disabled = true;

  try {
    setStatus('Step 1/4 · Fetching suppliers & masses…', 'running');
    let r = await post('/api/bci/init', {city_name: city, user_params: up});
    if (r.status !== 'ok') throw new Error(r.message);

    setStatus('Step 2/4 · Building component networks…', 'running');
    r = await post('/api/bci/build_network', {});
    if (r.status !== 'ok') throw new Error(r.message);

    setStatus('Step 3/4 · Computing accessibility & BCI…', 'running');
    r = await post('/api/bci/compute', {user_params: up});
    if (r.status !== 'ok') throw new Error(r.message);

    setStatus('Step 4/4 · Rendering BCI visualisations…', 'running');
    r = await get('/api/bci/visualize');
    if (r.status !== 'ok') throw new Error(r.message);

    applyBCIViz(r);
    _hasBCI = true;
    setStatus(`✓ BCI complete — ${r.stats.n_hexagons} hexagons`, 'ok');
    notify('BCI Complete', `${r.stats.n_hexagons} hexagons computed`);
    checkCompare();

  } catch(e) {
    console.error(e);
    setStatus('BCI error: ' + e.message, 'err');
  }
  btn.disabled = false;
  showTab('bci');
}

function applyBCIViz(r) {
  _bciVizDone = true;
  document.getElementById('bci-placeholder').classList.add('hidden');
  document.getElementById('bci-content').classList.remove('hidden');
  renderStats('bci-stats', r.stats, BCI_STAT_MAP);
  setImg('img-bci-masses',       r.bci_masses);
  setImg('img-bci-topography',   r.bci_topography);
  setImg('img-bci-components',   r.bci_components);
  setImg('img-bci-dist',         r.bci_distribution);
  setMapSrc('map-bci',           r.bci_map);
  if (r.neighborhoods && r.neighborhoods.length) {
    renderNeighborhoodTable('bci-nb-wrap', 'bci-nb-table', r.neighborhoods, 'BCI');
  }
}

/* =========================================================
   NEIGHBOURHOOD TABLE
   ========================================================= */
function renderNeighborhoodTable(wrapperId, tableId, data, scoreLabel) {
  const wrap = document.getElementById(wrapperId);
  const el   = document.getElementById(tableId);
  if (!data || !data.length) { wrap.classList.add('hidden'); return; }

  const sorted = [...data].sort((a, b) => b.avg_score - a.avg_score);
  const PAGE   = 10;
  let shown    = 0;

  el.innerHTML = '<table style="margin-top:4px;width:100%">'
    + '<thead><tr>'
    + '<th style="width:36px">Color</th>'
    + '<th>Neighbourhood</th>'
    + `<th>Avg ${scoreLabel}</th>`
    + '<th>Hex Count</th>'
    + `</tr></thead><tbody id="${tableId}-body"></tbody></table>`
    + `<button id="${tableId}-more" style="margin-top:8px;font-size:.78rem;`
    + `padding:4px 12px;cursor:pointer;border:1px solid var(--border);`
    + `background:var(--panel);color:var(--text);border-radius:4px"></button>`;

  const tbody = document.getElementById(tableId + '-body');
  const btn   = document.getElementById(tableId + '-more');

  function showMore() {
    const chunk = sorted.slice(shown, shown + PAGE);
    let html = '';
    chunk.forEach(row => {
      const color = row.color || '#aaaaaa';
      const score = typeof row.avg_score === 'number' ? row.avg_score.toFixed(1) : '—';
      html += `<tr>
        <td><span style="display:inline-block;width:16px;height:16px;border-radius:3px;
                         background:${color};border:1px solid rgba(255,255,255,0.2);
                         vertical-align:middle"></span></td>
        <td style="font-size:.82rem">${row.name}</td>
        <td style="font-size:.82rem;font-weight:600;color:var(--accent)">${score}</td>
        <td style="font-size:.82rem;color:var(--muted)">${row.hex_count}</td>
      </tr>`;
    });
    shown += chunk.length;
    tbody.insertAdjacentHTML('beforeend', html);
    const remaining = sorted.length - shown;
    if (remaining <= 0) btn.style.display = 'none';
    else btn.textContent = `Show ${Math.min(PAGE, remaining)} more`;
  }

  btn.addEventListener('click', showMore);
  showMore(); // render first 10
  wrap.classList.remove('hidden');
}

/* =========================================================
   RUN BOTH
   ========================================================= */
async function runBoth() {
  await runPCI();
  await runBCI();
  await loadCompare();
}

/* =========================================================
   COMPARE
   ========================================================= */
function checkCompare() {
  // Auto-run only after both viz panels have been populated in this session
  if (_pciVizDone && _bciVizDone) loadCompare();
}

async function runCompare() {
  const errEl = document.getElementById('compare-error-msg');
  if (!_hasPCI && !_hasBCI) {
    errEl.textContent = 'Error: Both PCI and BCI must be computed or restored before running comparison.';
    return;
  }
  if (!_hasPCI) {
    errEl.textContent = 'Error: PCI has not been computed or restored yet.';
    return;
  }
  if (!_hasBCI) {
    errEl.textContent = 'Error: BCI has not been computed or restored yet.';
    return;
  }
  errEl.textContent = '';
  await loadCompare();
}

async function loadCompare() {
  try {
    setStatus('Loading comparative analysis…', 'running');
    const r = await get('/api/compare/visualize');
    if (r.status !== 'ok') throw new Error(r.message);

    document.getElementById('compare-placeholder').classList.add('hidden');
    document.getElementById('compare-content').classList.remove('hidden');

    renderStats('compare-stats', r.stats, CMP_STAT_MAP);
    setImg('img-scatter',      r.scatter);
    setImg('img-dist-compare', r.distribution);
    setImg('img-spatial',      r.spatial);
    setMapSrc('map-compare',   r.comparison_map);
    setStatus('✓ Comparison loaded', 'ok');
  } catch(e) {
    console.error(e);
    setStatus('Compare error: ' + e.message, 'err');
  }
}

/* =========================================================
   DIAGNOSTICS
   ========================================================= */
async function runNetworkDiag() {
  setStatus('Running network diagnostics…', 'running');
  try {
    const r = await get('/api/diagnostics/network');
    if (r.status !== 'ok') throw new Error(r.message);
    const d = r.diagnostics;
    const unified = d.unified || {};
    const modeStats = d.mode_stats || {};
    const hc = d.hex_coverage || {};

    // ── Per-mode table ──
    let html = '<p style="color:var(--muted);font-size:.78rem;font-weight:600;margin-bottom:6px">PER-MODE NETWORKS</p>';
    html += '<table style="margin-bottom:16px"><thead><tr>'
          + '<th>Mode</th><th>Nodes</th><th>Edges</th><th>Connected</th><th>Components</th><th>Largest component</th>'
          + '</tr></thead><tbody>';
    Object.entries(modeStats).forEach(([mode, s]) => {
      const connIcon = s.connected ? '✅' : '⚠';
      html += `<tr>
        <td><b>${mode}</b></td>
        <td>${s.nodes.toLocaleString()}</td>
        <td>${s.edges.toLocaleString()}</td>
        <td>${connIcon}</td>
        <td>${s.n_components}</td>
        <td>${s.largest_component.toLocaleString()}</td>
      </tr>`;
    });
    html += '</tbody></table>';

    // ── Unified graph table ──
    html += '<p style="color:var(--muted);font-size:.78rem;font-weight:600;margin-bottom:6px">UNIFIED GRAPH</p>';
    html += '<table style="margin-bottom:16px"><thead><tr>'
          + '<th>Metric</th><th>Value</th></tr></thead><tbody>';
    const unifiedRows = [
      ['Nodes',            (unified.nodes||0).toLocaleString()],
      ['Edges',            (unified.edges||0).toLocaleString()],
      ['Connected',        unified.connected ? '✅ Yes' : '⚠ No'],
      ['Components',       unified.n_components],
      ['Largest component',(unified.largest_component||0).toLocaleString()],
      ['Travel time mean', (unified.time_min_mean||0).toFixed(2) + ' min'],
      ['Travel time max',  (unified.time_min_max||0).toFixed(2) + ' min'],
    ];
    unifiedRows.forEach(([k,v]) => html += `<tr><td>${k}</td><td>${v}</td></tr>`);

    // Edges by mode sub-rows
    Object.entries(unified.edges_by_mode || {}).forEach(([m, cnt]) => {
      html += `<tr><td style="padding-left:20px;color:var(--muted)">↳ ${m} edges</td><td>${cnt.toLocaleString()}</td></tr>`;
    });
    html += '</tbody></table>';

    // ── Hex coverage table ──
    html += '<p style="color:var(--muted);font-size:.78rem;font-weight:600;margin-bottom:6px">HEX → NODE COVERAGE</p>';
    html += '<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>'
          + `<tr><td>Hexes</td><td>${hc.n_hexes ?? '—'}</td></tr>`
          + `<tr><td>Unique nodes mapped</td><td>${hc.n_unique_nodes ?? '—'}</td></tr>`
          + `<tr><td>Coverage ratio</td><td>${hc.coverage_ratio ?? '—'}</td></tr>`
          + '</tbody></table>';

    document.getElementById('diag-output').innerHTML = html;
    setStatus('✓ Network diagnostics complete', 'ok');
  } catch(e) {
    setStatus('Diag error: ' + e.message, 'err');
    document.getElementById('diag-output').textContent = 'Error: ' + e.message;
  }
}

async function runTopoDiag() {
  setStatus('Loading topography summary…', 'running');
  try {
    const r = await get('/api/diagnostics/topography');
    if (r.status !== 'ok') throw new Error(r.message);
    const rows = r.summary || [];
    if (!rows.length) {
      document.getElementById('topo-diag-output').textContent = 'No topography data available.';
      return;
    }
    const headers = Object.keys(rows[0]);
    let html = '<table><thead><tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr></thead><tbody>';
    rows.forEach(row => {
      html += '<tr>' + headers.map(h => {
        const v = row[h];
        return `<td>${typeof v === 'number' ? v.toFixed(3) : v}</td>`;
      }).join('') + '</tr>';
    });
    html += '</tbody></table>';
    document.getElementById('topo-diag-output').innerHTML = html;
    setStatus('✓ Topography summary loaded', 'ok');
  } catch(e) {
    setStatus('Topo diag error: ' + e.message, 'err');
  }
}

async function runIsochrones() {
  const maxOrigins = Math.min(parseInt(document.getElementById('iso-max-origins').value) || 5, 10);
  setStatus('Running isochrone analysis…', 'running');
  document.getElementById('iso-output').textContent = 'Computing isochrones…';
  try {
    let r = await post('/api/isochrones/run', {max_origins: maxOrigins});
    if (r.status !== 'ok') throw new Error(r.message);

    // Show tables
    let html = '';
    const sec = (title) => `<p style="color:var(--muted);font-size:.8rem;margin:12px 0 4px"><b>${title}</b></p>`;

    if (r.pci_per_origin && r.pci_per_origin.length) {
      html += sec('PCI — Amenity Counts per Origin (transit, 15 min)');
      html += buildTable(r.pci_per_origin);
    }
    if (r.pci_summary && r.pci_summary.length) {
      html += sec('PCI — Mean Amenities Reachable: Top vs Bottom Origins');
      html += buildTable(r.pci_summary);
    }
    if (r.bci_per_origin && r.bci_per_origin.length) {
      html += sec('BCI — Demand Counts per Origin (transit, 15 min)');
      html += buildTable(r.bci_per_origin);
    }
    if (r.bci_pop_summary && r.bci_pop_summary.length) {
      html += sec('BCI — Mean Population/Demand Reachable: Top vs Bottom Origins');
      html += buildTable(r.bci_pop_summary);
    }
    if (r.bci_biz_summary && r.bci_biz_summary.length) {
      html += sec('BCI — Mean Business Density Reachable: Top vs Bottom Origins');
      html += buildTable(r.bci_biz_summary);
    }
    document.getElementById('iso-output').innerHTML = html || 'No data returned.';

    // Load maps
    r = await get('/api/isochrones/maps');
    if (r.pci_iso_map || r.bci_iso_map) {
      document.getElementById('iso-maps').classList.remove('hidden');
      if (r.pci_iso_map) setMapSrc('map-pci-iso', r.pci_iso_map);
      if (r.bci_iso_map) setMapSrc('map-bci-iso', r.bci_iso_map);
    }

    setStatus('✓ Isochrone analysis complete', 'ok');
  } catch(e) {
    console.error(e);
    setStatus('Isochrone error: ' + e.message, 'err');
    document.getElementById('iso-output').textContent = 'Error: ' + e.message;
  }
}

function buildTable(rows) {
  if (!rows || !rows.length) return '';
  const cols = Object.keys(rows[0]);
  let h = '<table style="margin-bottom:8px"><thead><tr>';
  cols.forEach(c => h += `<th>${c}</th>`);
  h += '</tr></thead><tbody>';
  rows.forEach(row => {
    h += '<tr>';
    cols.forEach(c => {
      const v = row[c];
      h += `<td>${typeof v === 'number' ? v.toFixed(1) : (v ?? '—')}</td>`;
    });
    h += '</tr>';
  });
  return h + '</tbody></table>';
}

/* =========================================================
   ABOUT TAB
   ========================================================= */
let _aboutLoaded = false;
async function loadAbout() {
  if (_aboutLoaded) return;
  try {
    const r = await get('/api/about');
    if (r.status !== 'ok') throw new Error(r.message);
    document.getElementById('about-content').innerHTML =
      (typeof marked !== 'undefined')
        ? marked.parse(r.markdown)
        : `<pre style="white-space:pre-wrap">${r.markdown}</pre>`;
    _aboutLoaded = true;
  } catch(e) {
    document.getElementById('about-content').textContent = 'Could not load about.md: ' + e.message;
  }
}

/* =========================================================
   RESTORE SAVED RESULTS
   ========================================================= */
async function restoreResults() {
  const city = document.getElementById('city-select').value;
  if (!city) { setStatus('Select a city first', 'err'); return; }
  setStatus('Restoring saved results…', 'running');
  try {
    const r = await post('/api/restore', {city_name: city});
    if (r.status !== 'ok') throw new Error(r.message);

    if (r.has_pci) {
      _hasPCI = true;
      // Reload visualisations from state
      const viz = await get('/api/pci/visualize');
      if (viz.status === 'ok') applyPCIViz(viz);
    }
    if (r.has_bci) {
      _hasBCI = true;
      const viz = await get('/api/bci/visualize');
      if (viz.status === 'ok') applyBCIViz(viz);
    }
    if (_hasPCI && _hasBCI) await loadCompare();

    const parts = [];
    if (r.has_pci) parts.push('PCI');
    if (r.has_bci) parts.push('BCI');
    setStatus(`✓ Restored ${parts.join(' + ')} results for ${r.city_name}`, 'ok');
  } catch(e) {
    setStatus('Restore failed: ' + e.message, 'err');
  }
}

/* =========================================================
   TOOLTIP (body-level, never clipped by overflow)
   ========================================================= */
(function() {
  const tt = document.getElementById('js-tooltip');
  let hideTimer;

  document.addEventListener('mouseover', function(e) {
    const el = e.target.closest('.tip');
    if (!el) return;
    clearTimeout(hideTimer);
    tt.textContent = el.getAttribute('data-tip') || '';
    tt.style.opacity = '1';
    position(e);
  });

  document.addEventListener('mousemove', function(e) {
    if (!e.target.closest('.tip')) return;
    position(e);
  });

  document.addEventListener('mouseout', function(e) {
    if (!e.target.closest('.tip')) return;
    hideTimer = setTimeout(() => { tt.style.opacity = '0'; }, 80);
  });

  function position(e) {
    const pad = 12;
    const tw = tt.offsetWidth, th = tt.offsetHeight;
    let x = e.clientX + pad;
    let y = e.clientY - th - pad;
    // Keep within viewport
    if (x + tw > window.innerWidth  - 4) x = e.clientX - tw - pad;
    if (y < 4) y = e.clientY + pad;
    tt.style.left = x + 'px';
    tt.style.top  = y + 'px';
  }
})();

// ── Sensitivity Analysis ──

async function runSensitivity(model) {
  const statusEl = document.getElementById('sens-status');
  statusEl.textContent = `Running ${model.toUpperCase()} sensitivity analysis…`;

  // Hide legacy single-result containers (only shown if block system not yet active)
  ['sens-tornado', 'sens-table'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.add('hidden');
  });

  try {
    const r = await post(`/api/sensitivity/${model}`, {});
    if (r.error) { statusEl.textContent = 'Error: ' + r.error; return; }

    // Ensure the stacking container exists.
    // Anchor it before #sens-tornado (the legacy single-result container)
    // so it appears in the right position in the sensitivity tab.
    let stack = document.getElementById('sens-results-stack');
    if (!stack) {
      stack = document.createElement('div');
      stack.id = 'sens-results-stack';
      const legacyAnchor = document.getElementById('sens-tornado');
      if (legacyAnchor) {
        legacyAnchor.parentNode.insertBefore(stack, legacyAnchor);
      } else {
        // Fallback: append after the status element
        statusEl.insertAdjacentElement('afterend', stack);
      }
    }

    const blockId = `sens-block-${model}`;
    let block = document.getElementById(blockId);
    if (!block) {
      // First time this model is run — prepend so newest is always on top
      block = document.createElement('div');
      block.id = blockId;
      stack.insertBefore(block, stack.firstChild);
    }
    // Populate (or update) this model's result block
    block.innerHTML =
      `<div class="card" style="margin-bottom:16px">` +
        `<div class="card-title">${model.toUpperCase()} Sensitivity Analysis</div>` +
        `<img src="data:image/png;base64,${r.tornado_png}" class="viz-img" />` +
        `<div>${r.table_html}</div>` +
      `</div>`;

    statusEl.textContent = `${model.toUpperCase()} sensitivity complete.`;
    notify(`${model.toUpperCase()} Sensitivity Done`, 'Tornado chart ready.');
  } catch(e) {
    const msg = (e.message.includes('Unexpected token') || e.message.includes('not valid JSON'))
      ? `Please run ${model.toUpperCase()} fully first (init → build network → compute), then try again.`
      : e.message;
    statusEl.textContent = 'Error: ' + msg;
  }
}

/* =========================================================
   SCENARIO TESTING
   ========================================================= */

// Internal state
let _scHexes     = [];   // H3 hex IDs selected by hex click or text input
let _scEdgeNodes = [];   // Edge node pairs selected by edge click: [{u,v,mode,time_min}]

// ── Functions exposed to the map iframe (same-origin direct call) ──────────

/* Called directly by the folium iframe on hex click */
function scAddHex(hex_id) {
  if (!hex_id || _scHexes.includes(hex_id)) return;
  _scHexes.push(hex_id);
  _scRenderChips();
}

/* Called directly by the folium iframe on edge click */
function scReceiveEdge(data) {
  // Update edge info panel
  const panel = document.getElementById('sc-edge-info');
  if (panel) {
    document.getElementById('sc-edge-mode').textContent = data.mode || '—';
    document.getElementById('sc-edge-time').textContent =
      data.time_min != null ? Number(data.time_min).toFixed(2) + ' min' : '—';
    document.getElementById('sc-edge-u').textContent = data.u || '—';
    document.getElementById('sc-edge-v').textContent = data.v || '—';
    panel.classList.remove('hidden');
  }
  // Add as edge chip (dedup by u|v key)
  const key = `${data.u}|${data.v}`;
  if (!_scEdgeNodes.find(e => `${e.u}|${e.v}` === key)) {
    _scEdgeNodes.push({ u: data.u, v: data.v, mode: data.mode || '', time_min: data.time_min });
    _scRenderChips();
  }
}

// ── postMessage fallback (in case direct call is unavailable) ────────────────
window.addEventListener('message', function (evt) {
  const d = evt.data;
  if (!d || typeof d !== 'object') return;
  if (d.type === 'hex-selected')  scAddHex(d.hex_id);
  if (d.type === 'edge-selected') scReceiveEdge(d);
});

// ── Hex / edge chip management ─────────────────────────────────────────────

function scAddHexFromInput() {
  const el  = document.getElementById('sc-hex-input');
  const raw = el.value.trim();
  raw.split(',').forEach(h => { const t = h.trim(); if (t) scAddHex(t); });
  el.value = '';
}

function scRemoveHex(hex_id) {
  _scHexes = _scHexes.filter(h => h !== hex_id);
  _scRenderChips();
}

function scRemoveEdge(idx) {
  _scEdgeNodes.splice(idx, 1);
  _scRenderChips();
}

function scClearHexes() {
  _scHexes     = [];
  _scEdgeNodes = [];
  document.getElementById('sc-hex-input').value = '';
  _scRenderChips();
}

function _scRenderChips() {
  const container = document.getElementById('sc-chips');
  const hexChips = _scHexes.map(h =>
    `<span style="display:inline-flex;align-items:center;gap:4px;
                  padding:3px 8px;border-radius:20px;font-size:.73rem;
                  background:var(--accent);color:#fff;font-family:monospace">
       ${h}
       <span style="cursor:pointer;opacity:.8;font-size:.85rem"
             onclick="scRemoveHex('${h}')">✕</span>
     </span>`
  );
  const edgeChips = _scEdgeNodes.map((e, i) =>
    `<span style="display:inline-flex;align-items:center;gap:4px;
                  padding:3px 8px;border-radius:20px;font-size:.73rem;
                  background:#c07000;color:#fff;font-family:monospace"
           title="Edge node pair — usable in edge penalty / removal scenarios">
       ⊸&nbsp;${e.u}→${e.v}
       <span style="cursor:pointer;opacity:.8;font-size:.85rem"
             onclick="scRemoveEdge(${i})">✕</span>
     </span>`
  );
  container.innerHTML = [...hexChips, ...edgeChips].join('');
}

// ── Show / hide parameter rows based on scenario type ─────────────────────

function onScTypeChange() {
  const type            = document.getElementById('sc-type').value;
  const factorRow       = document.getElementById('sc-factor-row');
  const amenityAddRow   = document.getElementById('sc-amenity-add-row');
  const supplierAddRow  = document.getElementById('sc-supplier-add-row');
  const warn            = document.getElementById('sc-build-warn');

  factorRow.classList.add('hidden');
  amenityAddRow.classList.add('hidden');
  supplierAddRow.classList.add('hidden');
  warn.classList.add('hidden');

  if (type === 'edge_penalty') {
    factorRow.classList.remove('hidden');
    warn.textContent =
      '⚠ Edge penalty requires full travel-time recomputation — expect 2–10 minutes.';
    warn.classList.remove('hidden');
  }
  if (type === 'edge_remove') {
    warn.textContent =
      '⚠ Edge removal requires full travel-time recomputation — expect 2–10 minutes.';
    warn.classList.remove('hidden');
  }
  if (type === 'amenity_add') {
    amenityAddRow.classList.remove('hidden');
    loadAmenityTypes();
  }
  if (type === 'supplier_add') {
    supplierAddRow.classList.remove('hidden');
  }
}

async function loadAmenityTypes() {
  const sel = document.getElementById('sc-amenity-type');
  if (!sel) return;
  sel.innerHTML = '<option value="">Loading…</option>';
  try {
    const r = await get('/api/scenario/amenity_types');
    if (r.status !== 'ok') throw new Error(r.message);
    sel.innerHTML = r.types.map(t =>
      `<option value="${t.name}"
               data-unit="${t.unit}"
               data-raw-range="${t.raw_range}"
               data-weight="${t.weight}">${t.label} — ${t.unit}</option>`
    ).join('');
    _updateAmenityCountLabel();
  } catch (e) {
    sel.innerHTML = '<option value="">Run PCI first</option>';
  }
}

function _updateAmenityCountLabel() {
  const sel = document.getElementById('sc-amenity-type');
  if (!sel) return;
  const opt  = sel.selectedOptions[0];
  const unit = opt ? opt.getAttribute('data-unit') : 'units';
  const lbl  = document.getElementById('sc-amenity-count-lbl');
  if (lbl) lbl.textContent = `Count (${unit})`;
}

// ── Load network map ───────────────────────────────────────────────────────

function loadNetworkMap() {
  const btn   = document.getElementById('btn-load-netmap');
  const frame = document.getElementById('sc-net-map');
  const hint  = document.getElementById('netmap-hint');

  if (!frame) {
    console.error('[loadNetworkMap] iframe #sc-net-map not found in DOM');
    setStatus('Network map error: iframe element missing', 'err');
    if (btn) btn.disabled = false;
    return;
  }
  if (!btn) {
    console.error('[loadNetworkMap] button #btn-load-netmap not found in DOM');
  }

  if (btn) btn.disabled = true;
  if (hint) hint.textContent = 'Loading network map…';
  setStatus('Loading network map…', 'running');

  frame.onload = function () {
    if (hint) hint.textContent = 'Click any hex or drive edge to add it to the target selection.';
    setStatus('✓ Network map loaded', 'ok');
    if (btn) btn.disabled = false;
  };
  frame.onerror = function () {
    if (hint) hint.textContent = 'Error loading network map — check that PCI or BCI has been run.';
    setStatus('Network map error', 'err');
    if (btn) btn.disabled = false;
  };

  /* Load as a proper same-origin page so window.parent.scAddHex() works
     without any cross-origin restriction.
     Append a timestamp to bust the browser's iframe src cache so that
     clicking the button a second time always triggers a fresh onload. */
  frame.src = '/api/scenario/network_map_view?t=' + Date.now();
  frame.classList.remove('hidden');
}

// ── Run scenario ───────────────────────────────────────────────────────────

async function runScenario(indexType) {
  const type   = document.getElementById('sc-type').value;
  const radius = parseInt(document.getElementById('sc-radius').value)  || 0;
  const factor = parseFloat(document.getElementById('sc-factor').value) || 2.0;

  // Validate index compatibility
  // amenity_* → PCI only  |  supplier_* → BCI only  |  edge_* → both allowed
  const isPciOnly = type.startsWith('amenity_');
  const isBciOnly = type.startsWith('supplier_');
  if (isPciOnly && indexType === 'bci') {
    setStatus('Amenity scenarios only apply to PCI.', 'err'); return;
  }
  if (isBciOnly && indexType === 'pci') {
    setStatus('Supplier scenarios only apply to BCI.', 'err'); return;
  }

  if (_scHexes.length === 0 && _scEdgeNodes.length === 0) {
    setStatus('Select at least one hex or edge on the map first.', 'err'); return;
  }

  const amenityType  = document.getElementById('sc-amenity-type')?.value  || 'education';
  const amenityCount = parseFloat(document.getElementById('sc-amenity-count')?.value) || 1;
  const strength     = parseFloat(document.getElementById('sc-supplier-strength')?.value) || 1.0;

  const body = {
    scenario_type: type,
    hex_ids:       _scHexes,
    edge_nodes:    _scEdgeNodes.map(e => ({ u: e.u, v: e.v })),
    radius, factor, strength,
    amenity_type:  amenityType,
    amenity_count: amenityCount,
  };
  const url     = indexType === 'pci' ? '/api/scenario/run_pci' : '/api/scenario/run_bci';
  const btnId   = indexType === 'pci' ? 'btn-sc-pci' : 'btn-sc-bci';
  const btn     = document.getElementById(btnId);
  btn.disabled  = true;

  const label = type.replace(/_/g, ' ');
  setStatus(`Running scenario: ${label}…`, 'running');

  try {
    const r = await post(url, body);
    if (r.status !== 'ok') throw new Error(r.message);
    _scRenderResults(r, indexType, label);
    setStatus(`✓ Scenario complete — ${r.n_affected} hex(es) targeted`, 'ok');
    notify('Scenario Complete', `${label} · ${r.n_affected} hex(es) affected`);
  } catch (e) {
    console.error(e);
    setStatus('Scenario error: ' + e.message, 'err');
  }
  btn.disabled = false;
}

// ── Render results ─────────────────────────────────────────────────────────

function _scRenderResults(r, indexType, label) {
  const resultsEl = document.getElementById('sc-results');
  resultsEl.classList.remove('hidden');

  // Title
  document.getElementById('sc-result-title').textContent =
    `Impact Summary — ${indexType.toUpperCase()} · ${label}`;

  // Server-side warning (e.g. connectivity)
  const warnEl = document.getElementById('sc-server-warn');
  if (r.warning) {
    warnEl.textContent = r.warning;
    warnEl.classList.remove('hidden');
  } else {
    warnEl.classList.add('hidden');
  }

  // Stats — split into baseline / modified / delta boxes
  const s = r.stats || {};
  renderStats('sc-stats-base', s, {
    baseline_mean: 'Mean Score',
  });
  renderStats('sc-stats-mod', s, {
    modified_mean: 'Mean Score',
  });
  renderStats('sc-stats-delta', s, {
    mean_delta:   'Mean Δ',
    median_delta: 'Median Δ',
    max_gain:     'Max Gain',
    max_loss:     'Max Loss',
    n_improved:   '# Improved',
    n_degraded:   '# Degraded',
    n_unchanged:  '# Unchanged',
    p10_delta:    'P10 Δ',
    p25_delta:    'P25 Δ',
    p75_delta:    'P75 Δ',
    p90_delta:    'P90 Δ',
  });

  // Delta map
  if (r.delta_map_html) {
    document.getElementById('sc-delta-map').srcdoc = r.delta_map_html;
  }

  // Top 10 table
  const tbody = document.getElementById('sc-top-tbody');
  tbody.innerHTML = '';
  (r.top_hexes || []).forEach((row, i) => {
    const delta     = row.delta;
    const colour    = delta > 0 ? 'var(--green)' : (delta < 0 ? 'var(--red)' : 'var(--muted)');
    const sign      = delta > 0 ? '+' : '';
    const nb        = row.neighborhood || '—';
    tbody.insertAdjacentHTML('beforeend',
      `<tr>
         <td style="color:var(--muted)">${i + 1}</td>
         <td style="font-family:monospace;font-size:.78rem">${row.hex_id}</td>
         <td style="color:var(--muted)">${nb}</td>
         <td style="text-align:right;font-weight:700;color:${colour}">${sign}${delta}</td>
       </tr>`
    );
  });

  // Scroll results into view
  resultsEl.scrollIntoView({behavior: 'smooth', block: 'start'});
}

// ── Reset ──────────────────────────────────────────────────────────────────

function scReset() {
  scClearHexes();
  document.getElementById('sc-results').classList.add('hidden');
  document.getElementById('sc-server-warn').classList.add('hidden');
  document.getElementById('sc-build-warn').classList.add('hidden');
  document.getElementById('sc-type').value = 'amenity_remove';
  onScTypeChange();
  setStatus('Scenario reset.', 'ok');
}

/* =========================================================
   SIDEBAR INFO BOXES
   Injected on page load to explain each parameter section.
   Targets known element IDs; walks up to the nearest .section
   ancestor and inserts after its .section-title.
   ========================================================= */

function _makeInfoBox(title, bodyHtml) {
  const box     = document.createElement('div');
  box.className = 'info-box';

  const header = document.createElement('div');
  header.className = 'info-box-header';
  header.innerHTML =
    `<i class="fas fa-circle-info" style="font-size:.68rem"></i>${title}` +
    `<i class="fas fa-chevron-down info-chevron"></i>`;

  const body     = document.createElement('div');
  body.className = 'info-box-body';
  body.innerHTML = bodyHtml;

  header.addEventListener('click', () => {
    body.classList.toggle('open');
    header.querySelector('.info-chevron').style.transform =
      body.classList.contains('open') ? 'rotate(180deg)' : '';
  });

  box.appendChild(header);
  box.appendChild(body);
  return box;
}

function _insertInfoBoxInSection(anchorId, box) {
  const anchor = document.getElementById(anchorId);
  if (!anchor) return;
  const section = anchor.closest('.section');
  if (!section) { anchor.parentNode.insertBefore(box, anchor); return; }
  const title = section.querySelector('.section-title');
  if (title) title.insertAdjacentElement('afterend', box);
  else section.insertBefore(box, section.firstChild);
}

(function _injectInfoBoxes() {
  _insertInfoBoxInSection('p-beta', _makeInfoBox('PCI Accessibility Parameters',
    '<b>Hansen Beta (β)</b> — distance-decay exponent. Controls how fast accessibility' +
    ' drops with travel time. Higher β = only very close amenities count;' +
    ' lower β = longer trips are nearly as valued.' +
    ' Typical range: 0.1 (flat/transit-rich) – 0.5 (hilly/car-dependent).<br><br>' +
    '<b>Active Street Lambda (λ)</b> — multiplier that rewards hexagons with more' +
    ' walkable or cyclable streets. Raises PCI for pedestrian-friendly areas.'
  ));

  _insertInfoBoxInSection('w-health', _makeInfoBox('Amenity Weights',
    'Each weight sets how much a category contributes to the composite <b>amenity mass</b>.' +
    ' Weights are automatically normalised to sum to 1.' +
    ' Raise a category to give it more influence on the final PCI score.' +
    ' Example: raising <em>Health</em> emphasises hospital and clinic proximity.'
  ));

  _insertInfoBoxInSection('tag-toggles', _makeInfoBox('Amenity Categories',
    'Toggle which OSM feature types are included in the analysis.' +
    ' Disabled categories contribute 0 to amenity mass regardless of their weight.' +
    ' Use this to focus on only the services relevant to your research question.'
  ));

  _insertInfoBoxInSection('b-beta-m', _makeInfoBox('BCI Accessibility Parameters',
    'Distance-decay exponents for each BCI component:<br>' +
    '<b>Market β</b> — decay for consumer/retail access.<br>' +
    '<b>Labour β</b> — decay for employment-zone access.<br>' +
    '<b>Supplier β</b> — decay for wholesale/industrial supply access.<br><br>' +
    '<b>Interface Lambda (λ)</b> adds a bonus for hexagons near city boundaries' +
    ' or airports, capturing export and logistics potential.'
  ));

  _insertInfoBoxInSection('bci-method', _makeInfoBox('BCI Aggregation Method',
    '<b>Geometric Mean</b> — penalises imbalanced access; a hexagon must score well' +
    ' on all three components to get a high BCI.<br>' +
    '<b>Weighted Average</b> — lets you assign explicit importance to each component.' +
    ' Weights are normalised automatically.<br>' +
    '<b>Min</b> — BCI equals the weakest component; highlights bottlenecks.'
  ));

  _insertInfoBoxInSection('supplier-tag-toggles', _makeInfoBox('Supplier Categories',
    'Toggle which OSM business/land-use types count as suppliers.' +
    ' These feed the <em>supplier mass</em> component of BCI.' +
    ' Disable categories that are not relevant to your city\'s economic profile.'
  ));
})();
