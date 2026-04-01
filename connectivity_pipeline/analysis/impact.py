"""
File: impact.py — Scenario Testing & Impact Analysis
=====================================================
Computes the effect of hypothetical modifications on PCI / BCI scores.

Fast paths   reuse cached travel times (amenity / supplier changes).
Slow paths   recompute travel times from scratch (edge penalty / removal).

IMPORTANT: All functions are non-destructive — shared objects (ham,
bci_hansen, the unified graph, component graphs) are temporarily mutated
inside try/finally blocks and are FULLY RESTORED after each call.
Session state and saved results are NEVER altered.
"""

import copy
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import folium
import random
from branca.colormap import LinearColormap
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple

from core.h3_helper import HexGrid
from pci.pci_calculator import HansenAccessibilityModel, TopographicPCICalculator
from bci.bci_calculator import BCIHansenAccessibility, BCICalculator


# ---------------------------------------------------------------------------
# H3 hex expansion
# ---------------------------------------------------------------------------

def _is_node_id(s: str) -> bool:
    """Return True if the string looks like an OSM node ID (all digits)."""
    return str(s).isdigit()


def expand_hexes(hex_ids: List[str], radius: int = 0) -> List[str]:
    """Expand H3 hex IDs with k-ring radius.

    Strings that look like direct node IDs (all-digit, e.g. '123456789')
    are passed through unchanged — they are handled by _find_nodes_near_hexes.
    """
    h3_ids   = [h for h in hex_ids if not _is_node_id(h)]
    node_ids = [h for h in hex_ids if _is_node_id(h)]

    if radius <= 0:
        return list(set(hex_ids))
    try:
        import h3
        result: set = set(node_ids)
        for h in h3_ids:
            result.update(h3.k_ring(h, radius))
        return list(result)
    except Exception:
        return list(set(hex_ids))


# ---------------------------------------------------------------------------
# Minimal mass proxy for HAM fast-path scenarios
# ---------------------------------------------------------------------------

class _MassProxy:
    """
    Duck-types only the part of MassCalculator that
    HansenAccessibilityModel.compute_accessibility() actually uses.
    """
    def __init__(self, composite: pd.Series):
        self._composite = composite


# ---------------------------------------------------------------------------
# Network visualisation map
# ---------------------------------------------------------------------------

def make_network_map(
    grid,
    net,
    pci:       Optional[pd.Series] = None,
    bci:       Optional[pd.Series] = None,
    city_name: str = "",
) -> folium.Map:
    """
    Folium map: hex grid (coloured by PCI / BCI if available) + walk /
    transit / drive edges.

    Clicking a hex fires:
        window.parent.postMessage({type:'hex-selected', hex_id:'...'}, '*')

    Clicking an edge fires:
        window.parent.postMessage({type:'edge-selected', u, v, time_min, mode}, '*')
    """
    from analysis.shared import _folium_center

    gdf    = grid.gdf.copy()
    center = _folium_center(gdf)
    m      = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    # ── Hex layer ──────────────────────────────────────────────────────────
    if pci is not None:
        gdf["_score"] = gdf["hex_id"].map(pci)
        caption = "PCI"
    elif bci is not None:
        gdf["_score"] = gdf["hex_id"].map(bci)
        caption = "BCI"
    else:
        gdf["_score"] = 0.0
        caption = "Hexes"

    valid = gdf["_score"].dropna()
    lo = float(valid.min()) if len(valid) else 0.0
    hi = float(valid.max()) if len(valid) else 1.0
    if lo == hi:
        hi = lo + 1.0

    score_cmap = LinearColormap(
        ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"],
        vmin=lo, vmax=hi, caption=caption,
    )

    def _hex_style(feat):
        v = feat["properties"].get("_score")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return {"fillColor": "#555", "fillOpacity": 0.25,
                    "weight": 0.5, "color": "#444"}
        return {"fillColor": score_cmap(v), "fillOpacity": 0.55,
                "weight": 0.6, "color": "#333"}

    folium.GeoJson(
        gdf[["hex_id", "_score", "geometry"]].to_json(),
        name="Hex Grid",
        style_function=_hex_style,
        tooltip=folium.GeoJsonTooltip(
            fields=["hex_id", "_score"],
            aliases=["Hex ID:", f"{caption}:"],
            sticky=True,
        ),
    ).add_to(m)
    score_cmap.add_to(m)

    # ── Network edges ───────────────────────────────────────────────────────
    if net is not None:
        _add_network_edges(m, net)

    # ── Click handler: hex → parent function call, edge → parent function call ─
    # Uses a recursive layer walk.  Primary mechanism: direct same-origin call
    # to window.parent.scAddHex / window.parent.scReceiveEdge (reliable for
    # srcdoc iframes).  Falls back to postMessage for resilience.
    # Retries every 300 ms until the Leaflet map object is found (up to 20 s).
    click_script = folium.Element("""
    <script>
    (function () {
      function walkLayers(layer, fn) {
        if (layer.feature && layer.feature.properties) fn(layer);
        if (typeof layer.eachLayer === 'function') {
          layer.eachLayer(function (sub) { walkLayers(sub, fn); });
        }
      }

      function findMap() {
        var keys = Object.keys(window);
        for (var i = 0; i < keys.length; i++) {
          if (keys[i].indexOf('map_') !== 0) continue;
          try {
            var v = window[keys[i]];
            if (v && typeof v.eachLayer === 'function' &&
                typeof v.getZoom === 'function') return v;
          } catch (e) {}
        }
        for (var i = 0; i < keys.length; i++) {
          try {
            var v = window[keys[i]];
            if (v && typeof v.eachLayer === 'function' &&
                typeof v.getZoom === 'function' &&
                typeof v.setView === 'function') return v;
          } catch (e) {}
        }
        return null;
      }

      function sendHex(hex_id) {
        /* Prefer direct same-origin parent call; fall back to postMessage */
        try {
          if (typeof window.parent.scAddHex === 'function') {
            window.parent.scAddHex(hex_id); return;
          }
        } catch (e) {}
        window.parent.postMessage({ type: 'hex-selected', hex_id: hex_id }, '*');
      }

      function sendEdge(u, v, time_min, mode) {
        var msg = { u: u, v: v, time_min: time_min, mode: mode };
        try {
          if (typeof window.parent.scReceiveEdge === 'function') {
            window.parent.scReceiveEdge(msg); return;
          }
        } catch (e) {}
        window.parent.postMessage(
          { type: 'edge-selected', u: u, v: v, time_min: time_min, mode: mode }, '*');
      }

      var _attached = false;
      function attachHandlers() {
        var lmap = findMap();
        if (!lmap) return false;

        walkLayers(lmap, function (sub) {
          var props = sub.feature.properties;
          if (props.hex_id) {
            sub.off('click').on('click', function () {
              sendHex(props.hex_id);
            });
          } else if (props.u !== undefined && props.v !== undefined) {
            sub.off('click').on('click', function (e) {
              sendEdge(String(props.u), String(props.v), props.time_min, props.mode);
              if (window.L && L.DomEvent) L.DomEvent.stopPropagation(e);
            });
          }
        });
        return true;
      }

      /* Keep retrying every 300 ms until we find the map (up to ~20 s) */
      var _tries = 0;
      function tryAttach() {
        if (_attached) return;
        if (attachHandlers()) { _attached = true; return; }
        if (++_tries < 65) setTimeout(tryAttach, 300);
      }
      window.addEventListener('load', tryAttach);
      tryAttach();
    })();
    </script>
    """)
    m.get_root().html.add_child(click_script)

    folium.LayerControl().add_to(m)
    return m


def _add_network_edges(m: folium.Map, net):
    """Add walk / transit / drive edges as GeoJson polylines.

    Transit and drive are shown by default (connected backbone networks).
    Walk is togglable via the legend but hidden initially to reduce clutter.
    Edge caps are per-mode to reflect typical network sizes.
    """
    MODE_STYLES = {
        #  mode       colour      shown   max_edges
        "walk":    ("#455A64", False, 30_000),
        "transit": ("#1976D2", True,  20_000),
        "drive":   ("#BF360C", True,  15_000),
    }

    for mode, (color, show, max_edges) in MODE_STYLES.items():
        G = net.networks.get(mode)
        if G is None:
            continue

        edges = list(G.edges(data=True))
        if len(edges) > max_edges:
            rng   = random.Random(42)
            edges = rng.sample(edges, max_edges)

        features = []
        nodes = G.nodes  # single dict lookup, reused across iterations
        for u, v, data in edges:
            try:
                nu = nodes[u]
                nv = nodes[v]
                ux, uy = nu.get("x", 0), nu.get("y", 0)
                vx, vy = nv.get("x", 0), nv.get("y", 0)
                if not (ux or uy) or not (vx or vy):
                    continue
                t = round(data.get("time_min", 0.0), 2)
                features.append({
                    "type": "Feature",
                    # Direct dict avoids Shapely object construction overhead
                    "geometry": {"type": "LineString",
                                 "coordinates": [[ux, uy], [vx, vy]]},
                    "properties": {
                        "mode": mode, "time_min": t,
                        "u": str(u), "v": str(v),
                    },
                })
            except Exception:
                continue

        if not features:
            continue

        fc    = {"type": "FeatureCollection", "features": features}
        layer = folium.FeatureGroup(name=f"{mode.title()} Network", show=show)
        # No GeoJsonTooltip: attaching hover handlers to 15k features is the
        # primary cause of slow browser-side map parsing.  Edge u/v/time_min
        # are still in properties and available to click handlers.
        folium.GeoJson(
            fc,
            style_function=lambda feat, c=color: {
                "color": c, "weight": 2.2, "opacity": 0.75,
            },
        ).add_to(layer)
        layer.add_to(m)


# ---------------------------------------------------------------------------
# PCI Scenario — Fast path: amenity removal
# ---------------------------------------------------------------------------

def run_pci_amenity_removal(
    s:       dict,
    hex_ids: List[str],
    radius:  int = 0,
) -> dict:
    """
    Zero out composite amenity mass for target hexes, recompute PCI.
    Reuses cached travel times — no Dijkstra rerun.
    ham.mass and ham._accessibility are fully restored on exit.
    """
    ham       = s["ham"]
    mass_calc = s["mass_calc"]
    grid      = s["grid"]
    up        = s["user_params"]
    city_cfg  = s["city_cfg"]
    avg_cost  = s.get("avg_mode_cost", 3.94)
    baseline  = s["pci"]

    target_hexes = expand_hexes(hex_ids, radius)

    comp_mod = mass_calc._composite.copy()
    for h in target_hexes:
        if h in comp_mod.index:
            comp_mod[h] = 0.0

    orig_mass   = ham.mass
    orig_access = ham._accessibility
    ham.mass    = _MassProxy(comp_mod)
    try:
        ham.compute_accessibility(
            beta=up["hansen_beta"],
            income_data=s.get("income_by_hex"),
            mode_cost=avg_cost,
        )
        pci_calc_mod = TopographicPCICalculator(grid, ham, mass_calc)
        pci_mod = pci_calc_mod.compute_pci(
            active_lambda=up["active_street_lambda"],
            mask_parks=up.get("mask_parks", False),
            park_threshold=city_cfg.get("park_threshold", 0.90),
        )
    finally:
        ham.mass          = orig_mass
        ham._accessibility = orig_access

    return _build_result(grid, baseline, pci_mod, target_hexes, s.get("city_name", ""))


# ---------------------------------------------------------------------------
# PCI Scenario — Fast path: amenity addition
# ---------------------------------------------------------------------------

def run_pci_amenity_addition(
    s:            dict,
    hex_id:       str,
    amenity_type: str   = "education",
    count:        float = 1.0,
) -> dict:
    """
    Add `count` units of `amenity_type` to a single hex, recompute PCI.
    Reuses cached travel times (fast path).

    Mass delta is approximated as:
        w_T × count / (raw_range_T × total_weight)
    which represents the composite mass added when raw count increases by
    `count` units, holding the normalization range fixed.
    """
    ham       = s["ham"]
    mass_calc = s["mass_calc"]
    grid      = s["grid"]
    up        = s["user_params"]
    city_cfg  = s["city_cfg"]
    avg_cost  = s.get("avg_mode_cost", 3.94)
    baseline  = s["pci"]

    comp_mod = mass_calc._composite.copy()

    # ── Compute per-unit composite mass delta ───────────────────────────────
    layer = mass_calc.layers.get(amenity_type)
    if layer is not None and layer.weight > 0:
        total_weight = sum(
            l.weight for l in mass_calc.layers.values() if l.weight > 0
        ) or 1.0
        raw_range  = float(layer.raw_values.max() - layer.raw_values.min())
        raw_range  = max(raw_range, 1.0)          # avoid ÷0
        mass_delta = layer.weight * count / (raw_range * total_weight)
    else:
        # Fallback: fraction of city-mean composite
        city_mean  = float(comp_mod[comp_mod > 0].mean()) if (comp_mod > 0).any() else 1.0
        mass_delta = count * city_mean * 0.05

    if hex_id in comp_mod.index:
        comp_mod[hex_id] = comp_mod[hex_id] + mass_delta

    orig_mass   = ham.mass
    orig_access = ham._accessibility
    ham.mass    = _MassProxy(comp_mod)
    try:
        ham.compute_accessibility(
            beta=up["hansen_beta"],
            income_data=s.get("income_by_hex"),
            mode_cost=avg_cost,
        )
        pci_calc_mod = TopographicPCICalculator(grid, ham, mass_calc)
        pci_mod = pci_calc_mod.compute_pci(
            active_lambda=up["active_street_lambda"],
            mask_parks=up.get("mask_parks", False),
            park_threshold=city_cfg.get("park_threshold", 0.90),
        )
    finally:
        ham.mass          = orig_mass
        ham._accessibility = orig_access

    return _build_result(grid, baseline, pci_mod, [hex_id], s.get("city_name", ""))


# ---------------------------------------------------------------------------
# PCI Scenario — Slow path: edge travel-time penalty
# ---------------------------------------------------------------------------

def run_pci_edge_penalty(
    s:       dict,
    hex_ids: List[str],
    factor:  float = 2.0,
    radius:  int   = 0,
) -> dict:
    """
    Multiply time_min on edges near target hexes by factor.
    Slow path — travel times are recomputed (~2–10 min).
    All modified edge weights are restored in the finally block.
    """
    net       = s.get("network")
    grid      = s["grid"]
    mass_calc = s["mass_calc"]
    up        = s["user_params"]
    city_cfg  = s["city_cfg"]
    avg_cost  = s.get("avg_mode_cost", 3.94)
    baseline  = s["pci"]

    if net is None:
        raise RuntimeError("Network not in session — build network first.")

    target_hexes   = expand_hexes(hex_ids, radius)
    affected_nodes = _find_nodes_near_hexes(net.unified_graph, grid.gdf, target_hexes)

    G = net.unified_graph
    modified_edges: Dict[Tuple, float] = {}
    for u, v, k, data in G.edges(data=True, keys=True):
        if u in affected_nodes or v in affected_nodes:
            orig_t = data.get("time_min", 1.0)
            modified_edges[(u, v, k)] = orig_t
            G[u][v][k]["time_min"]    = orig_t * factor

    ham_mod = HansenAccessibilityModel(grid, net, mass_calc)
    try:
        ham_mod.compute_travel_times(max_time=city_cfg["max_travel_time"])
        ham_mod.compute_accessibility(
            beta=up["hansen_beta"],
            income_data=s.get("income_by_hex"),
            mode_cost=avg_cost,
        )
        pci_calc_mod = TopographicPCICalculator(grid, ham_mod, mass_calc)
        pci_mod = pci_calc_mod.compute_pci(
            active_lambda=up["active_street_lambda"],
            mask_parks=up.get("mask_parks", False),
            park_threshold=city_cfg.get("park_threshold", 0.90),
        )
    finally:
        for (u, v, k), orig_t in modified_edges.items():
            try:
                G[u][v][k]["time_min"] = orig_t
            except Exception:
                pass

    return _build_result(grid, baseline, pci_mod, target_hexes, s.get("city_name", ""))


# ---------------------------------------------------------------------------
# PCI Scenario — Slow path: edge removal with connectivity guard
# ---------------------------------------------------------------------------

def run_pci_edge_removal(
    s:       dict,
    hex_ids: List[str],
    radius:  int = 0,
) -> dict:
    """
    Remove all edges whose BOTH endpoints are inside the target region.
    Slow path — travel times recomputed.
    Warns if the graph becomes disconnected.
    All removed edges are restored in the finally block.
    """
    net       = s.get("network")
    grid      = s["grid"]
    mass_calc = s["mass_calc"]
    up        = s["user_params"]
    city_cfg  = s["city_cfg"]
    avg_cost  = s.get("avg_mode_cost", 3.94)
    baseline  = s["pci"]

    if net is None:
        raise RuntimeError("Network not in session — build network first.")

    target_hexes   = expand_hexes(hex_ids, radius)
    affected_nodes = _find_nodes_near_hexes(net.unified_graph, grid.gdf, target_hexes)

    G = net.unified_graph
    edges_to_remove = [
        (u, v, k) for u, v, k in G.edges(keys=True)
        if u in affected_nodes and v in affected_nodes
    ]
    if not edges_to_remove:
        raise RuntimeError(
            "No edges found with both endpoints inside the selected region. "
            "Try a larger radius or more hexes."
        )

    # Connectivity guard
    G_test = nx.Graph(G)
    G_test.remove_edges_from([(u, v) for u, v, k in edges_to_remove])
    disconnected = not nx.is_connected(G_test)
    warning = (
        "⚠ Removing these edges disconnects the network — "
        "some hexes may become unreachable."
    ) if disconnected else None

    removed: List[Tuple] = []
    for u, v, k in edges_to_remove:
        data = dict(G[u][v][k])
        removed.append((u, v, k, data))
    for u, v, k, _ in removed:
        G.remove_edge(u, v, key=k)

    ham_mod = HansenAccessibilityModel(grid, net, mass_calc)
    try:
        ham_mod.compute_travel_times(max_time=city_cfg["max_travel_time"])
        ham_mod.compute_accessibility(
            beta=up["hansen_beta"],
            income_data=s.get("income_by_hex"),
            mode_cost=avg_cost,
        )
        pci_calc_mod = TopographicPCICalculator(grid, ham_mod, mass_calc)
        pci_mod = pci_calc_mod.compute_pci(
            active_lambda=up["active_street_lambda"],
            mask_parks=up.get("mask_parks", False),
            park_threshold=city_cfg.get("park_threshold", 0.90),
        )
    finally:
        for u, v, k, data in removed:
            try:
                G.add_edge(u, v, key=k, **data)
            except Exception:
                pass

    result = _build_result(grid, baseline, pci_mod, target_hexes, s.get("city_name", ""))
    if warning:
        result["warning"] = warning
    return result


# ---------------------------------------------------------------------------
# BCI Scenario — Fast path: supplier removal / addition
# ---------------------------------------------------------------------------

def run_bci_supplier_change(
    s:        dict,
    hex_ids:  List[str],
    mode:     str   = "remove",   # "remove" | "add"
    strength: float = 1.0,
    radius:   int   = 0,
) -> dict:
    """
    Modify supplier mass for target hexes, recompute BCI.
    Reuses cached BCI travel times — no Dijkstra rerun.
    bci_hansen.accessibility is fully restored on exit.
    """
    bci_hansen = s["bci_hansen"]
    mass_calc  = s["mass_calc_bci"]
    grid       = s["grid"]
    up         = s["user_params"]
    baseline   = s["bci"]

    target_hexes = expand_hexes(hex_ids, radius)

    supplier_mod = mass_calc.supplier_mass.copy()
    city_mean    = (
        float(supplier_mod[supplier_mod > 0].mean())
        if (supplier_mod > 0).any() else 1.0
    )
    for h in target_hexes:
        if h in supplier_mod.index:
            if mode == "remove":
                supplier_mod[h] = 0.0
            else:
                supplier_mod[h] = supplier_mod[h] + strength * city_mean

    orig_access = {k: v.copy() for k, v in bci_hansen.accessibility.items()}

    bci_hansen.compute_all_accessibility(
        market_mass=mass_calc.market_mass,
        labour_mass=mass_calc.labour_mass,
        supplier_mass=supplier_mod,
    )
    bci_calc_mod = BCICalculator(grid, bci_hansen, mass_calc)
    try:
        bci_mod = bci_calc_mod.compute_bci(
            method=up["bci_method"],
            market_weight=up["market_weight"],
            labour_weight=up["labour_weight"],
            supplier_weight=up["supplier_weight"],
            use_interface=up.get("use_urban_interface", True),
            interface_lambda=up.get("interface_lambda", 0.15),
        )
    finally:
        bci_hansen.accessibility = orig_access

    return _build_result(grid, baseline, bci_mod, target_hexes, s.get("city_name", ""))


# ---------------------------------------------------------------------------
# BCI Scenario — Slow path: edge travel-time penalty
# ---------------------------------------------------------------------------

def run_bci_edge_penalty(
    s:       dict,
    hex_ids: List[str],
    factor:  float = 2.0,
    radius:  int   = 0,
) -> dict:
    """
    Multiply time_min on edges in each BCI component graph near target hexes.
    Slow path — BCI travel times are recomputed.
    Component graph edges and bci_hansen state are fully restored on exit.
    """
    bci_hansen = s["bci_hansen"]
    mass_calc  = s["mass_calc_bci"]
    grid       = s["grid"]
    up         = s["user_params"]
    city_cfg   = s["city_cfg"]
    baseline   = s["bci"]

    if not bci_hansen._component_graphs:
        raise RuntimeError("BCI component graphs not built — run BCI build_network first.")

    target_hexes = expand_hexes(hex_ids, radius)

    # Save original bci_hansen state (travel times + accessibility)
    orig_travel_times = copy.deepcopy(bci_hansen._travel_times)
    orig_access       = {k: v.copy() for k, v in bci_hansen.accessibility.items()}

    # Modify edges in each component graph, track changes for restoration
    comp_modified: Dict[str, Dict[Tuple, float]] = {}
    for comp, G in bci_hansen._component_graphs.items():
        affected  = _find_nodes_near_hexes(G, grid.gdf, target_hexes)
        mod_edges: Dict[Tuple, float] = {}
        for u, v, k, data in G.edges(data=True, keys=True):
            if u in affected or v in affected:
                orig_t = data.get("time_min", 1.0)
                mod_edges[(u, v, k)] = orig_t
                G[u][v][k]["time_min"] = orig_t * factor
        comp_modified[comp] = mod_edges

    bci_calc_mod = BCICalculator(grid, bci_hansen, mass_calc)
    try:
        bci_hansen.compute_all_travel_times(max_time=city_cfg["max_travel_time"])
        bci_hansen.compute_all_accessibility(
            market_mass=mass_calc.market_mass,
            labour_mass=mass_calc.labour_mass,
            supplier_mass=mass_calc.supplier_mass,
        )
        bci_mod = bci_calc_mod.compute_bci(
            method=up["bci_method"],
            market_weight=up["market_weight"],
            labour_weight=up["labour_weight"],
            supplier_weight=up["supplier_weight"],
            use_interface=up.get("use_urban_interface", True),
            interface_lambda=up.get("interface_lambda", 0.15),
        )
    finally:
        # Restore component graph edge weights
        for comp, mod_edges in comp_modified.items():
            G = bci_hansen._component_graphs[comp]
            for (u, v, k), orig_t in mod_edges.items():
                try:
                    G[u][v][k]["time_min"] = orig_t
                except Exception:
                    pass
        # Restore bci_hansen internal state
        bci_hansen._travel_times = orig_travel_times
        bci_hansen.accessibility = orig_access

    return _build_result(grid, baseline, bci_mod, target_hexes, s.get("city_name", ""))


# ---------------------------------------------------------------------------
# BCI Scenario — Slow path: edge removal with connectivity guard
# ---------------------------------------------------------------------------

def run_bci_edge_removal(
    s:       dict,
    hex_ids: List[str],
    radius:  int = 0,
) -> dict:
    """
    Remove edges (both endpoints in region) from every BCI component graph.
    Slow path — BCI travel times recomputed.
    Warns if any component graph becomes disconnected.
    All state is fully restored on exit.
    """
    bci_hansen = s["bci_hansen"]
    mass_calc  = s["mass_calc_bci"]
    grid       = s["grid"]
    up         = s["user_params"]
    city_cfg   = s["city_cfg"]
    baseline   = s["bci"]

    if not bci_hansen._component_graphs:
        raise RuntimeError("BCI component graphs not built — run BCI build_network first.")

    target_hexes = expand_hexes(hex_ids, radius)

    orig_travel_times = copy.deepcopy(bci_hansen._travel_times)
    orig_access       = {k: v.copy() for k, v in bci_hansen.accessibility.items()}

    # Remove edges from each component graph
    comp_removed: Dict[str, List[Tuple]] = {}
    warnings = []
    for comp, G in bci_hansen._component_graphs.items():
        affected = _find_nodes_near_hexes(G, grid.gdf, target_hexes)
        to_remove = [
            (u, v, k) for u, v, k in G.edges(keys=True)
            if u in affected and v in affected
        ]
        if not to_remove:
            comp_removed[comp] = []
            continue

        # Connectivity guard per component
        G_test = nx.Graph(G)
        G_test.remove_edges_from([(u, v) for u, v, k in to_remove])
        if not nx.is_connected(G_test):
            warnings.append(f"{comp}")

        removed = []
        for u, v, k in to_remove:
            data = dict(G[u][v][k])
            removed.append((u, v, k, data))
        for u, v, k, _ in removed:
            G.remove_edge(u, v, key=k)
        comp_removed[comp] = removed

    total_removed = sum(len(v) for v in comp_removed.values())
    if total_removed == 0:
        raise RuntimeError(
            "No edges found with both endpoints inside the selected region. "
            "Try a larger radius or more hexes."
        )

    warning = (
        "⚠ Edge removal disconnects component graph(s): " + ", ".join(warnings) + "."
    ) if warnings else None

    bci_calc_mod = BCICalculator(grid, bci_hansen, mass_calc)
    try:
        bci_hansen.compute_all_travel_times(max_time=city_cfg["max_travel_time"])
        bci_hansen.compute_all_accessibility(
            market_mass=mass_calc.market_mass,
            labour_mass=mass_calc.labour_mass,
            supplier_mass=mass_calc.supplier_mass,
        )
        bci_mod = bci_calc_mod.compute_bci(
            method=up["bci_method"],
            market_weight=up["market_weight"],
            labour_weight=up["labour_weight"],
            supplier_weight=up["supplier_weight"],
            use_interface=up.get("use_urban_interface", True),
            interface_lambda=up.get("interface_lambda", 0.15),
        )
    finally:
        for comp, removed in comp_removed.items():
            G = bci_hansen._component_graphs[comp]
            for u, v, k, data in removed:
                try:
                    G.add_edge(u, v, key=k, **data)
                except Exception:
                    pass
        bci_hansen._travel_times = orig_travel_times
        bci_hansen.accessibility = orig_access

    result = _build_result(grid, baseline, bci_mod, target_hexes, s.get("city_name", ""))
    if warning:
        result["warning"] = warning
    return result


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _find_nodes_near_hexes(
    G,
    grid_gdf: gpd.GeoDataFrame,
    hex_ids:  List[str],
) -> set:
    """Return graph nodes within ~500 m of the centroid of each target hex.

    Strings in hex_ids that look like direct node IDs (all-digit) are added
    to the result set immediately, bypassing the spatial lookup.  This allows
    edges clicked in the network map to be targeted without converting to hex.
    """
    # ── Direct node IDs (from edge-click selection) ─────────────────────────
    h3_ids   = [h for h in hex_ids if not _is_node_id(h)]
    node_strs = [h for h in hex_ids if _is_node_id(h)]

    affected: set = set()
    for n_str in node_strs:
        try:
            n = int(n_str)
            if G.has_node(n):
                affected.add(n)
        except (ValueError, TypeError):
            pass

    # ── Hex-based spatial lookup ─────────────────────────────────────────────
    rows = grid_gdf[grid_gdf["hex_id"].isin(h3_ids)]
    if rows.empty:
        return affected

    node_list   = list(G.nodes(data=True))
    node_ids    = [n[0] for n in node_list]
    node_coords = np.array([
        (d.get("y", 0), d.get("x", 0)) for _, d in node_list
    ])
    if len(node_coords) == 0:
        return affected

    centroids = np.array([
        (row.geometry.centroid.y, row.geometry.centroid.x)
        for _, row in rows.iterrows()
    ])
    kdt = cKDTree(node_coords)
    for c in centroids:
        for i in kdt.query_ball_point(c, r=0.005):   # ≈ 500 m
            affected.add(node_ids[i])
    return affected


def _build_result(
    grid,
    baseline:       pd.Series,
    modified:       pd.Series,
    affected_hexes: List[str],
    city_name:      str,
) -> dict:
    delta = (modified - baseline).reindex(baseline.index)
    delta_map = make_delta_map(grid.gdf, delta, baseline, modified, city_name)
    return {
        "stats":          compute_impact_stats(baseline, modified),
        # srcdoc expects a complete HTML document; _repr_html_ returns a
        # notebook iframe wrapper which can fail to render in nested iframes.
        "delta_map_html": delta_map.get_root().render(),
        "top_hexes":      top_affected_hexes(grid.gdf, delta),
        "n_affected":     len(affected_hexes),
    }


def compute_impact_stats(baseline: pd.Series, modified: pd.Series) -> dict:
    """Summary statistics comparing baseline vs modified index."""
    delta = (modified - baseline).dropna()
    b, m  = baseline.dropna(), modified.dropna()
    if len(delta) == 0:
        return {}
    return {
        "baseline_mean": round(float(b.mean()),          2),
        "modified_mean": round(float(m.mean()),          2),
        "mean_delta":    round(float(delta.mean()),      2),
        "median_delta":  round(float(delta.median()),    2),
        "max_gain":      round(float(delta.max()),       2),
        "max_loss":      round(float(delta.min()),       2),
        "n_improved":    int((delta >  0.5).sum()),
        "n_degraded":    int((delta < -0.5).sum()),
        "n_unchanged":   int((delta.abs() <= 0.5).sum()),
        "p10_delta":     round(float(delta.quantile(0.10)), 2),
        "p25_delta":     round(float(delta.quantile(0.25)), 2),
        "p75_delta":     round(float(delta.quantile(0.75)), 2),
        "p90_delta":     round(float(delta.quantile(0.90)), 2),
    }


def top_affected_hexes(
    grid_gdf: gpd.GeoDataFrame,
    delta:    pd.Series,
    n:        int = 10,
) -> list:
    """Top n hexes by absolute score change."""
    cols = ["hex_id"] + (["neighborhood"] if "neighborhood" in grid_gdf.columns else [])
    gdf  = grid_gdf[cols].copy()
    gdf["delta"]     = gdf["hex_id"].map(delta)
    gdf              = gdf.dropna(subset=["delta"])
    gdf["abs_delta"] = gdf["delta"].abs()
    top  = gdf.nlargest(n, "abs_delta").drop(columns=["abs_delta"])
    top["delta"] = top["delta"].round(2)
    return top.to_dict(orient="records")


def make_delta_map(
    grid_gdf: gpd.GeoDataFrame,
    delta:    pd.Series,
    baseline: pd.Series,
    modified: pd.Series,
    city_name: str = "",
) -> folium.Map:
    """Diverging choropleth: red = score drops, green = score gains."""
    from analysis.shared import _folium_center

    gdf = grid_gdf[["hex_id", "geometry"]].copy()
    gdf["delta"]    = gdf["hex_id"].map(delta).round(2)
    gdf["baseline"] = gdf["hex_id"].map(baseline).round(2)
    gdf["modified"] = gdf["hex_id"].map(modified).round(2)

    center = _folium_center(gdf)
    m      = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    valid = delta.dropna()
    if len(valid) == 0:
        return m

    abs_max = max(abs(float(valid.min())), abs(float(valid.max())), 0.01)
    cmap = LinearColormap(
        ["#d73027", "#fc8d59", "#ffffbf", "#91cf60", "#1a9850"],
        vmin=-abs_max, vmax=abs_max,
        caption="Score Δ (modified − baseline)",
    )

    def _style(feat):
        v = feat["properties"].get("delta")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return {"fillColor": "#555", "fillOpacity": 0.20,
                    "weight": 0.2, "color": "#444"}
        return {"fillColor": cmap(v), "fillOpacity": 0.75,
                "weight": 0.4,  "color": "#333"}

    folium.GeoJson(
        gdf.to_json(),
        style_function=_style,
        tooltip=folium.GeoJsonTooltip(
            fields=["hex_id", "baseline", "modified", "delta"],
            aliases=["Hex:", "Baseline:", "Modified:", "Δ:"],
        ),
        name="Score Δ",
    ).add_to(m)
    cmap.add_to(m)
    folium.LayerControl().add_to(m)
    return m
