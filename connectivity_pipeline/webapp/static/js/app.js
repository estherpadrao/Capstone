/* =========================================================
   STATE
   ========================================================= */
let _hasPCI = false, _hasBCI = false;
let _activeTab = 'about';

/* On page load, ask the server which results are already in session.
   This restores _hasPCI / _hasBCI after a browser refresh without
   requiring the user to click Restore. */
(async function restoreSessionFlags() {
  try {
    const r = await get('/api/session/status');
    if (r.has_pci) _hasPCI = true;
    if (r.has_bci) _hasBCI = true;
    // Do NOT auto-load compare here — PCI/BCI tabs are not yet populated.
    // Compare only loads when the user actually runs or restores results.
  } catch (e) { /* server not ready yet */ }
})();

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
  ['pci','bci','compare','diagnostics','sensitivity','about','scenario','hidden_trends'].forEach(t => {
    document.getElementById('tab-' + t).classList.toggle('hidden', t !== name);
  });
  document.querySelectorAll('.tab').forEach(btn => {
    const tabName = btn.getAttribute('data-tab') ||
                    btn.onclick.toString().match(/showTab\('(\w+)'/)?.[1];
    btn.classList.toggle('active', tabName === name);
  });
  if (name === 'about') loadAbout();
  if (name === 'hidden_trends') htSyncScenarioOptions();
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
  const el = document.getElementById(id);
  if (!el) return;
  el.src = 'data:image/png;base64,' + b64;
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
  n_hexagons: 'Hexagons', cv_pct: 'CV %',
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
    checkCompare();

  } catch(e) {
    console.error(e);
    setStatus('PCI error: ' + e.message, 'err');
  }
  btn.disabled = false;
  showTab('pci');
}

function applyPCIViz(r) {
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
    checkCompare();

  } catch(e) {
    console.error(e);
    setStatus('BCI error: ' + e.message, 'err');
  }
  btn.disabled = false;
  showTab('bci');
}

function applyBCIViz(r) {
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
  // Auto-run the first time both indices become available
  if (_hasPCI && _hasBCI) loadCompare();
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
  document.getElementById('compare-error-msg').textContent = '';
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
          + '<th>Mode</th><th>Nodes</th><th>Edges</th><th>Connected</th><th>Components</th><th>Largest connected component</th>'
          + '</tr></thead><tbody>';
    Object.entries(modeStats).forEach(([mode, s]) => {
      const connIcon = s.connected ? '✅' : '⚠';
      html += `<tr>
        <td><b>${mode}</b></td>
        <td>${s.nodes.toLocaleString()}</td>
        <td>${s.edges.toLocaleString()}</td>
        <td>${connIcon}</td>
        <td>${s.n_components}</td>
        <td>${s.largest_component.toLocaleString()} nodes</td>
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
      ['Largest connected component', (unified.largest_component||0).toLocaleString() + ' nodes'],
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

const _sensData  = {};   // { 'pci': {tornado_png, table_html}, 'bci': {...} }
const _sensOrder = [];   // ['pci','bci'] — index 0 = top of stack

function _renderSensStack() {
  const stack = document.getElementById('sens-results-stack');
  stack.innerHTML = '';
  _sensOrder.forEach(m => {
    const d = _sensData[m];
    const label = m.toUpperCase();
    const colorClass = m === 'pci' ? 'sens-block-pci' : 'sens-block-bci';
    const block = document.createElement('div');
    block.id = `sens-block-${m}`;
    block.className = `card ${colorClass}`;
    block.style.marginBottom = '0';
    block.innerHTML = `
      <div class="card-title" style="font-size:.9rem;text-transform:uppercase;letter-spacing:.06em">
        ${label} Sensitivity
      </div>
      <div class="card-title" style="font-size:.85rem;font-weight:400;margin-top:8px">Tornado Chart</div>
      <img src="data:image/png;base64,${d.tornado_png}"
           style="max-width:100%;border-radius:8px;margin-bottom:20px" />
      <div class="card-title" style="font-size:.85rem;font-weight:400;margin-bottom:6px">Score Changes by Parameter</div>
      <p style="color:var(--muted);font-size:.78rem;margin-bottom:10px">
        Cells show change from baseline. Green = score improves, red = score drops, dash = negligible change.
      </p>
      <div style="overflow-x:auto">${d.table_html}</div>`;
    stack.appendChild(block);
  });
}

async function runSensitivity(model) {
  const statusEl = document.getElementById('sens-status');
  statusEl.textContent = `Running ${model.toUpperCase()} sensitivity analysis…`;
  try {
    const r = await post(`/api/sensitivity/${model}`, {});
    if (r.error) { statusEl.textContent = 'Error: ' + r.error; return; }

    _sensData[model] = { tornado_png: r.tornado_png, table_html: r.table_html };

    // Move to top if not already there; add to top if new
    const idx = _sensOrder.indexOf(model);
    if (idx > 0) {
      _sensOrder.splice(idx, 1);
      _sensOrder.unshift(model);
    } else if (idx === -1) {
      _sensOrder.unshift(model);
    }
    // idx === 0 → already at top, data updated above, re-render in place

    _renderSensStack();
    statusEl.textContent = `${model.toUpperCase()} sensitivity complete.`;
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
let _scHexes       = [];   // H3 hex IDs selected by hex click or text input
let _scEdgeNodes   = [];   // Edge node pairs added to target: [{u,v,mode,time_min}]
let _scActiveIndex = null; // 'pci' | 'bci' — set when a map is loaded
let _scHexScores   = [];   // [{hex_id, score}] for the picker table
let _scLastEdge    = null; // Last edge clicked on map or in table (not yet committed)

function scAddHex(hex_id) {
  if (!hex_id || _scHexes.includes(hex_id)) return;
  _scHexes.push(hex_id);
  _scRenderChips();
}

/* Update the "Last Clicked Edge" info panel and store for later commit.
   Does NOT add to target chips — call scCommitLastEdge() for that. */
function scReceiveEdge(data) {
  _scLastEdge = data;
  const panel = document.getElementById('sc-edge-info');
  if (panel) {
    document.getElementById('sc-edge-mode').textContent = data.mode || '—';
    document.getElementById('sc-edge-time').textContent =
      data.time_min != null ? Number(data.time_min).toFixed(2) + ' min' : '—';
    document.getElementById('sc-edge-u').textContent = data.u || '—';
    document.getElementById('sc-edge-v').textContent = data.v || '—';
    panel.classList.remove('hidden');
  }
}

/* Add the last clicked edge to the target chips (dedup). */
function scCommitLastEdge() {
  scCommitEdge(_scLastEdge);
}

/* Add an edge object directly to target chips (dedup). Used by table rows. */
function scCommitEdge(data) {
  if (!data || !data.u || !data.v) return;
  const key = `${data.u}|${data.v}`;
  if (_scEdgeNodes.find(e => `${e.u}|${e.v}` === key)) return;
  _scEdgeNodes.push({ u: data.u, v: data.v, mode: data.mode || 'drive', time_min: data.time_min });
  _scRenderChips();
}

// ── localStorage channel (primary: fires reliably from same-origin iframes) ──
window.addEventListener('storage', function (evt) {
  if (evt.key === '_sc_hex_click' && evt.newValue) {
    try { scAddHex(JSON.parse(evt.newValue).hex_id); } catch (e) {}
  }
  if (evt.key === '_sc_edge_click' && evt.newValue) {
    try { scReceiveEdge(JSON.parse(evt.newValue)); } catch (e) {}
  }
});

// ── postMessage fallback ─────────────────────────────────────────────────────
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
  // Refresh picker highlights
  if (_scHexScores.length) {
    const q = document.getElementById('sc-picker-filter')?.value || '';
    if (q.trim()) scFilterHexPicker(); else _scRenderPickerRows(_scHexScores);
  }
  if (_scEdgeRows.length) {
    const q = document.getElementById('sc-edge-filter')?.value || '';
    if (q.trim()) scFilterEdgePicker(); else _scRenderEdgeRows(_scEdgeRows);
  }
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
  }
  if (type === 'supplier_add') {
    supplierAddRow.classList.remove('hidden');
  }
}

function _updateAmenityCountLabel() {
  const UNITS = {
    health: 'clinics / hospitals', education: 'schools / facilities',
    parks: 'm² of park area', community: 'community centres',
    food_retail: 'food / retail outlets', transit: 'transit stops',
  };
  const sel  = document.getElementById('sc-amenity-type');
  const lbl  = document.getElementById('sc-amenity-count-lbl');
  if (!sel || !lbl) return;
  const unit = UNITS[sel.value] || 'units';
  lbl.textContent = `Count (${unit})`;
}

// ── Native Leaflet network map (no iframe) ─────────────────────────────────

let _scLeafletMap = null;  // current Leaflet map instance

async function loadNetworkMap(indexType) {
  const btnPci = document.getElementById('btn-load-pci-map');
  const btnBci = document.getElementById('btn-load-bci-map');
  const mapDiv = document.getElementById('sc-net-map');
  const hint   = document.getElementById('netmap-hint');

  btnPci.disabled = true;
  btnBci.disabled = true;
  hint.textContent = `Loading ${indexType.toUpperCase()} map…`;
  setStatus(`Loading ${indexType.toUpperCase()} network map…`, 'running');

  // Destroy previous map instance so Leaflet can re-initialise on the same div
  if (_scLeafletMap) { _scLeafletMap.remove(); _scLeafletMap = null; }
  mapDiv.classList.remove('hidden');

  try {
    const [hexResp, edgeResp] = await Promise.all([
      fetch(`/api/scenario/hex_geojson?index=${indexType}`),
      fetch('/api/scenario/edge_geojson'),
    ]);
    const hexGJ  = await hexResp.json();
    const edgeGJ = await edgeResp.json();

    // Centre map on hex layer bounds
    const tmpLayer = L.geoJson(hexGJ);
    const bounds   = tmpLayer.getBounds();
    const center   = bounds.isValid() ? bounds.getCenter() : [0, 0];

    _scLeafletMap = L.map(mapDiv, { zoomControl: true }).setView(center, 12);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '© CartoDB', maxZoom: 19,
    }).addTo(_scLeafletMap);

    // ── Hex layer ──────────────────────────────────────────────────────────
    const scores  = hexGJ.features.map(f => f.properties._score).filter(v => v != null && !isNaN(v));
    const scoreLo = scores.length ? Math.min(...scores) : 0;
    const scoreHi = scores.length ? Math.max(...scores) : 1;

    L.geoJson(hexGJ, {
      style: function(feature) {
        const v = feature.properties._score;
        if (v == null || isNaN(v)) return {fillColor:'#555', fillOpacity:.25, weight:.5, color:'#444'};
        return {fillColor: _scScoreColor(v, scoreLo, scoreHi),
                fillOpacity: .55, weight: .8, color: '#333'};
      },
      onEachFeature: function(feature, layer) {
        const hid = feature.properties.hex_id;
        const sc  = feature.properties._score;
        layer.bindTooltip(
          `<b>${hid}</b><br>${indexType.toUpperCase()}: ${sc != null ? Number(sc).toFixed(3) : '—'}`,
          {sticky: true});
        layer.on('click', function() {
          // Find matching row in picker table and trigger its click
          const row = document.querySelector(
            `#sc-picker-tbody tr[data-hex="${hid}"]`);
          if (row) { row.click(); row.scrollIntoView({block:'nearest'}); }
          else scAddHex(hid);
        });
      },
    }).addTo(_scLeafletMap);

    // ── Edge layer — two passes: thin visual + wide invisible hit-target ───────
    // Visual layer: thin line, non-interactive so it never blocks the hit layer
    L.geoJson(edgeGJ, {
      style: {color:'#BF360C', weight:2, opacity:.7},
      interactive: false,
    }).addTo(_scLeafletMap);

    // Hit-target layer: 10 px wide, fully transparent, captures all clicks
    L.geoJson(edgeGJ, {
      style: {color:'#000', weight:10, opacity:0},
      onEachFeature: function(feature, layer) {
        const p = feature.properties;
        layer.bindTooltip(
          `From: ${p.u}<br>To: ${p.v}<br>Time: ${p.time_min} min`,
          {sticky: true});
        layer.on('click', function(e) {
          if (e.originalEvent) e.originalEvent.stopPropagation();
          // Always call scReceiveEdge directly — no row.click() routing
          scReceiveEdge({u: p.u, v: p.v, time_min: p.time_min, mode: 'drive'});
          // Scroll picker row into view as a visual hint if it exists
          const row = document.querySelector(
            `#sc-edge-picker-tbody tr[data-u="${p.u}"][data-v="${p.v}"]`);
          if (row) row.scrollIntoView({block:'nearest'});
        });
      },
    }).addTo(_scLeafletMap);

    _scActiveIndex = indexType;
    hint.textContent =
      `${indexType.toUpperCase()} map loaded — click a hex or edge to select it.`;
    hint.style.color = '';
    setStatus(`✓ ${indexType.toUpperCase()} network map loaded`, 'ok');
    _loadHexPicker(indexType);
  } catch (e) {
    hint.textContent = `Error loading map — make sure ${indexType.toUpperCase()} has been run first.`;
    hint.style.color = 'var(--text)';
    mapDiv.classList.add('hidden');
    setStatus('Network map error: ' + e.message, 'err');
  }
  btnPci.disabled = false;
  btnBci.disabled = false;
}

function _scScoreColor(v, lo, hi) {
  const stops = [
    [0.00, [49,  54,  149]],
    [0.25, [116, 173, 209]],
    [0.50, [255, 255, 191]],
    [0.75, [244, 109,  67]],
    [1.00, [165,   0,  38]],
  ];
  const t = hi > lo ? Math.max(0, Math.min(1, (v - lo) / (hi - lo))) : 0.5;
  let i = 0;
  while (i < stops.length - 2 && t > stops[i + 1][0]) i++;
  const t0 = stops[i][0],     c0 = stops[i][1];
  const t1 = stops[i + 1][0], c1 = stops[i + 1][1];
  const f  = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
  const r  = Math.round(c0[0] + f * (c1[0] - c0[0]));
  const g  = Math.round(c0[1] + f * (c1[1] - c0[1]));
  const b  = Math.round(c0[2] + f * (c1[2] - c0[2]));
  return `rgb(${r},${g},${b})`;
}

// ── Hex picker ─────────────────────────────────────────────────────────────

async function _loadHexPicker(indexType) {
  try {
    const r = await get('/api/scenario/hex_list');
    if (r.status !== 'ok') return;
    _scHexScores = r.hexes;
    _scRenderPickerRows(_scHexScores);
    document.getElementById('sc-picker-panel').classList.remove('hidden');
    document.getElementById('sc-picker-count').textContent =
      `${_scHexScores.length} hex${_scHexScores.length !== 1 ? 'es' : ''}`;
  } catch (e) { /* picker is optional — silently skip */ }
  _loadEdgePicker();
}

// ── Edge picker ─────────────────────────────────────────────────────────────

let _scEdgeRows = [];  // [{u, v, time_min}] for the edge picker table

async function _loadEdgePicker() {
  try {
    const r = await get('/api/scenario/edge_list');
    if (r.status !== 'ok') return;
    _scEdgeRows = r.edges;
    _scRenderEdgeRows(_scEdgeRows);
    document.getElementById('sc-edge-picker-panel').classList.remove('hidden');
    document.getElementById('sc-edge-picker-count').textContent =
      `${_scEdgeRows.length} edge${_scEdgeRows.length !== 1 ? 's' : ''}`;
  } catch (e) { /* silently skip */ }
}

function _scRenderEdgeRows(rows) {
  const tbody = document.getElementById('sc-edge-picker-tbody');
  if (!tbody) return;
  tbody.innerHTML = rows.map(e => {
    const key      = `${e.u}|${e.v}`;
    const sel      = !!_scEdgeNodes.find(x => `${x.u}|${x.v}` === key);
    const rowStyle = sel ? 'background:#c07000;color:#fff;' : 'cursor:pointer;';
    return `<tr style="${rowStyle}" data-u="${e.u}" data-v="${e.v}"
                onclick="scCommitEdge({u:'${e.u}',v:'${e.v}',time_min:${e.time_min},mode:'drive'})"
                title="${sel ? 'Already selected' : 'Click to select'}">
      <td style="padding:3px 8px;font-family:monospace;font-size:.71rem">${e.u}</td>
      <td style="padding:3px 8px;font-family:monospace;font-size:.71rem">${e.v}</td>
      <td style="padding:3px 8px;text-align:right">${e.time_min.toFixed(3)}</td>
      <td style="padding:3px 6px;text-align:center">
        ${sel ? '<span style="font-size:.8rem">✓</span>' : ''}
      </td>
    </tr>`;
  }).join('');
}

function scFilterEdgePicker() {
  const q = (document.getElementById('sc-edge-filter')?.value || '').trim().toLowerCase();
  if (!q) {
    _scRenderEdgeRows(_scEdgeRows);
    document.getElementById('sc-edge-picker-count').textContent = `${_scEdgeRows.length} edges`;
    return;
  }
  const numMatch = q.match(/^([><]=?|=)(\d+\.?\d*)$/);
  let filtered;
  if (numMatch) {
    const op = numMatch[1], val = parseFloat(numMatch[2]);
    filtered = _scEdgeRows.filter(e => {
      if (op === '>')  return e.time_min >  val;
      if (op === '>=') return e.time_min >= val;
      if (op === '<')  return e.time_min <  val;
      if (op === '<=') return e.time_min <= val;
      if (op === '=')  return Math.abs(e.time_min - val) < 0.0005;
      return false;
    });
  } else {
    filtered = _scEdgeRows.filter(e =>
      e.u.toLowerCase().includes(q) || e.v.toLowerCase().includes(q));
  }
  _scRenderEdgeRows(filtered);
  document.getElementById('sc-edge-picker-count').textContent = `${filtered.length} edges`;
}

function _scRenderPickerRows(rows) {
  const tbody = document.getElementById('sc-picker-tbody');
  if (!tbody) return;
  tbody.innerHTML = rows.map(h => {
    const sel      = _scHexes.includes(h.hex_id);
    const score    = h.score != null ? Number(h.score).toFixed(3) : '—';
    const rowStyle = sel ? 'background:var(--accent);color:#fff;' : 'cursor:pointer;';
    return `<tr style="${rowStyle}" data-hex="${h.hex_id}"
                onclick="scAddHex('${h.hex_id}')"
                title="${sel ? 'Already selected' : 'Click to select'}">
      <td style="padding:3px 8px;font-family:monospace;font-size:.71rem">${h.hex_id}</td>
      <td style="padding:3px 8px;text-align:right">${score}</td>
      <td style="padding:3px 6px;text-align:center">
        ${sel ? '<span style="font-size:.8rem">✓</span>' : ''}
      </td>
    </tr>`;
  }).join('');
}

function scFilterHexPicker() {
  const q = (document.getElementById('sc-picker-filter')?.value || '').trim().toLowerCase();
  if (!q) {
    _scRenderPickerRows(_scHexScores);
    document.getElementById('sc-picker-count').textContent =
      `${_scHexScores.length} hexes`;
    return;
  }
  // Supports: ID substring, ">50", "<30", "=0.45"
  let filtered;
  const numMatch = q.match(/^([><]=?|=)(\d+\.?\d*)$/);
  if (numMatch) {
    const op  = numMatch[1];
    const val = parseFloat(numMatch[2]);
    filtered  = _scHexScores.filter(h => {
      if (h.score == null) return false;
      if (op === '>')  return h.score >  val;
      if (op === '>=') return h.score >= val;
      if (op === '<')  return h.score <  val;
      if (op === '<=') return h.score <= val;
      if (op === '=')  return Math.abs(h.score - val) < 0.0005;
      return false;
    });
  } else {
    filtered = _scHexScores.filter(h => h.hex_id.toLowerCase().includes(q));
  }
  _scRenderPickerRows(filtered);
  document.getElementById('sc-picker-count').textContent = `${filtered.length} hexes`;
}

// ── Run scenario ───────────────────────────────────────────────────────────

async function runScenario() {
  if (!_scActiveIndex) {
    setStatus('Load a PCI or BCI map first.', 'err'); return;
  }
  const indexType = _scActiveIndex;
  const type      = document.getElementById('sc-type').value;
  const radius    = parseInt(document.getElementById('sc-radius').value)  || 0;
  const factor    = parseFloat(document.getElementById('sc-factor').value) || 2.0;

  // Validate index compatibility
  // amenity_* → PCI only  |  supplier_* → BCI only  |  edge_* → both allowed
  const isPciOnly = type.startsWith('amenity_');
  const isBciOnly = type.startsWith('supplier_');
  if (isPciOnly && indexType === 'bci') {
    setStatus('Amenity scenarios only apply to PCI — load the PCI map first.', 'err'); return;
  }
  if (isBciOnly && indexType === 'pci') {
    setStatus('Supplier scenarios only apply to BCI — load the BCI map first.', 'err'); return;
  }

  if (_scHexes.length === 0 && _scEdgeNodes.length === 0) {
    setStatus('Select at least one hex or edge first.', 'err'); return;
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
  const url = indexType === 'pci' ? '/api/scenario/run_pci' : '/api/scenario/run_bci';
  const btn = document.getElementById('btn-sc-run');
  btn.disabled = true;

  const label = type.replace(/_/g, ' ');
  setStatus(`Running ${indexType.toUpperCase()} scenario: ${label}…`, 'running');

  try {
    const r = await post(url, body);
    if (r.status !== 'ok') throw new Error(r.message);
    _scRenderResults(r, indexType, label);
    setStatus(`✓ Scenario complete — ${r.n_affected} hex(es) targeted`, 'ok');
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
  const pct   = s.change_threshold_pct ?? 1;
  const tNote = `|Δ| > ${pct}% of hex baseline`;
  renderStats('sc-stats-delta', s, {
    mean_delta:   'Mean Δ',
    median_delta: 'Median Δ',
    max_gain:     'Max Gain',
    max_loss:     'Max Loss',
    n_improved:   `# Improved (${tNote})`,
    n_degraded:   `# Degraded (${tNote})`,
    n_unchanged:  `# Unchanged (${tNote})`,
    p10_delta:    'P10 Δ',
    p25_delta:    'P25 Δ',
    p75_delta:    'P75 Δ',
    p90_delta:    'P90 Δ',
  });

  // Delta map
  if (r.insight_plot) setImg('sc-insight-img', r.insight_plot);

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

// Load About content on page open (tab is visible by default)
loadAbout();

/* =========================================================
   HIDDEN TRENDS — batch scenario analysis
   ========================================================= */

function htSyncScenarioOptions() {
  const index    = document.getElementById('ht-index').value;
  const selType  = document.getElementById('ht-scenario-type');
  const scType   = selType.value;
  const pciGrp   = document.getElementById('ht-pci-opts');
  const bciGrp   = document.getElementById('ht-bci-opts');
  const amenRow  = document.getElementById('ht-amenity-type-row');
  const factRow  = document.getElementById('ht-factor-row');
  const slowWarn = document.getElementById('ht-slow-warn');

  // Show/hide index-specific fast-path groups.
  // display:none on <optgroup> is ignored by Chrome/Safari, so we also
  // disable the individual <option> elements inside the hidden group.
  const pciOpts = Array.from(pciGrp.querySelectorAll('option'));
  const bciOpts = Array.from(bciGrp.querySelectorAll('option'));
  if (index === 'pci') {
    pciGrp.style.display = '';
    pciOpts.forEach(o => { o.disabled = false; });
    bciGrp.style.display = 'none';
    bciOpts.forEach(o => { o.disabled = true; });
    if (['supplier_remove','supplier_add'].includes(scType))
      selType.value = 'amenity_remove';
  } else {
    pciGrp.style.display = 'none';
    pciOpts.forEach(o => { o.disabled = true; });
    bciGrp.style.display = '';
    bciOpts.forEach(o => { o.disabled = false; });
    if (['amenity_remove','amenity_add'].includes(scType))
      selType.value = 'supplier_remove';
  }

  const currentType = selType.value;
  amenRow.style.display  = currentType === 'amenity_add'  ? '' : 'none';
  factRow.style.display  = currentType === 'edge_penalty' ? '' : 'none';
  const isNetwork = ['edge_penalty','edge_remove'].includes(currentType);
  slowWarn.classList.toggle('hidden', !isNetwork);

  // For network scenarios, runs-per-band and radius are fixed (1 run, radius 0
  // only) — disable and grey out those controls so they can't be changed.
  const nBandEl   = document.getElementById('ht-n-per-band');
  const radiusRow = document.getElementById('ht-radius-row');
  const r0cb      = document.getElementById('ht-radius-0');
  const r1cb      = document.getElementById('ht-radius-1');
  const muted     = 'var(--muted)';
  const text      = '';

  if (isNetwork) {
    if (nBandEl)   { nBandEl.value = 1; nBandEl.disabled = true;
                     nBandEl.style.opacity = '0.4'; nBandEl.style.cursor = 'not-allowed'; }
    if (radiusRow) { radiusRow.style.opacity = '0.4'; radiusRow.style.pointerEvents = 'none'; }
    if (r0cb)      { r0cb.checked = true;  r0cb.disabled = true; }
    if (r1cb)      { r1cb.checked = false; r1cb.disabled = true; }
  } else {
    if (nBandEl)   { nBandEl.disabled = false;
                     nBandEl.style.opacity = ''; nBandEl.style.cursor = ''; }
    if (radiusRow) { radiusRow.style.opacity = ''; radiusRow.style.pointerEvents = ''; }
    if (r0cb)      { r0cb.disabled = false; }
    if (r1cb)      { r1cb.disabled = false; }
  }

  htUpdateTotalRuns();
}

function htUpdateTotalRuns() {
  const nBand   = parseInt(document.getElementById('ht-n-per-band')?.value, 10) || 2;
  const r0      = document.getElementById('ht-radius-0')?.checked ? 1 : 0;
  const r1      = document.getElementById('ht-radius-1')?.checked ? 1 : 0;
  const nRadii  = Math.max(r0 + r1, 1);
  const scType  = document.getElementById('ht-scenario-type')?.value || '';
  const isNet   = ['edge_penalty','edge_remove'].includes(scType);
  const total   = isNet ? 3 : 3 * nRadii * nBand;
  const el = document.getElementById('ht-total-runs');
  if (el) el.textContent = total;
}

function htDownloadPlot() {
  const img = document.getElementById('ht-batch-plot');
  if (!img || !img.src || img.src === window.location.href) return;
  const a = document.createElement('a');
  a.href = img.src;
  a.download = 'hidden_trends_batch.png';
  a.click();
}

async function runHiddenTrends() {
  const btn        = document.getElementById('btn-ht-run');
  const statusEl   = document.getElementById('ht-status');
  const resultsEl  = document.getElementById('ht-results');
  const index       = document.getElementById('ht-index').value;
  const scType      = document.getElementById('ht-scenario-type').value;
  const nPerBand    = parseInt(document.getElementById('ht-n-per-band').value, 10) || 2;
  const seed        = parseInt(document.getElementById('ht-seed').value, 10);
  const amenityType = document.getElementById('ht-amenity-type').value;
  const factor      = parseFloat(document.getElementById('ht-factor').value) || 2.0;
  const radii       = [
    ...(document.getElementById('ht-radius-0')?.checked ? [0] : []),
    ...(document.getElementById('ht-radius-1')?.checked ? [1] : []),
  ];
  if (radii.length === 0) {
    statusEl.textContent = 'Select at least one radius.'; return;
  }

  btn.disabled   = true;
  resultsEl.classList.add('hidden');
  const totalRuns = parseInt(document.getElementById('ht-total-runs')?.textContent, 10) || '?';
  statusEl.textContent = `Running ${totalRuns} scenarios…`;
  setStatus('Running Hidden Trends batch analysis…', 'running');

  try {
    const r = await post('/api/hidden_trends/run', {
      index, scenario_type: scType, n_per_band: nPerBand, seed,
      amenity_type: amenityType, factor, radii,
    });
    if (r.status !== 'ok') { statusEl.textContent = 'Error: ' + r.message; return; }

    // Aggregated plot
    if (r.batch_plot) setImg('ht-batch-plot', r.batch_plot);

    resultsEl.classList.remove('hidden');
    const n = (r.runs || []).filter(r => !r.error).length;
    statusEl.textContent = `✓ ${n} runs complete.`;
    setStatus('✓ Hidden Trends batch complete', 'ok');
  } catch(e) {
    statusEl.textContent = 'Error: ' + e.message;
    setStatus('Hidden Trends error: ' + e.message, 'err');
  } finally {
    btn.disabled = false;
  }
}
