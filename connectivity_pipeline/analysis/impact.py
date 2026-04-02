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
import io
import base64
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import folium
import matplotlib
import matplotlib.pyplot as plt
import random
from branca.colormap import LinearColormap
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
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

    _hex_click = folium.JsCode("""
    function(feature, layer) {
        layer.on('click', function(e) {
            var hex_id = feature.properties.hex_id;
            if (!hex_id) return;
            /* localStorage fires a storage event in the parent window (same origin,
               different browsing context) — most reliable iframe→parent channel. */
            try {
                localStorage.setItem('_sc_hex_click',
                    JSON.stringify({hex_id: hex_id, t: Date.now()}));
            } catch(ex) {}
            /* fallbacks */
            try {
                if (typeof window.parent.scAddHex === 'function') {
                    window.parent.scAddHex(hex_id); return;
                }
            } catch(ex2) {}
            try {
                window.parent.postMessage({type:'hex-selected', hex_id: hex_id}, '*');
            } catch(ex3) {}
        });
    }
    """)

    folium.GeoJson(
        gdf[["hex_id", "_score", "geometry"]].to_json(),
        name="Hex Grid",
        style_function=_hex_style,
        tooltip=folium.GeoJsonTooltip(
            fields=["hex_id", "_score"],
            aliases=["Hex ID:", f"{caption}:"],
            sticky=True,
        ),
        on_each_feature=_hex_click,
    ).add_to(m)
    score_cmap.add_to(m)

    # ── Network edges ───────────────────────────────────────────────────────
    if net is not None:
        _add_network_edges(m, net)

    folium.LayerControl().add_to(m)
    return m


def _add_network_edges(m: folium.Map, net):
    """Add walk / transit / drive edges as GeoJson polylines.

    Transit and drive are shown by default (connected backbone networks).
    Walk is togglable via the legend but hidden initially to reduce clutter.
    Edge caps are per-mode to reflect typical network sizes.
    """
    from shapely.geometry import mapping, LineString

    MODE_STYLES = {
        #  mode       colour      shown   max_edges
        "drive": ("#BF360C", True, 15_000),   # deep red — connected road network
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
        for u, v, data in edges:
            try:
                uy, ux = G.nodes[u].get("y", 0), G.nodes[u].get("x", 0)
                vy, vx = G.nodes[v].get("y", 0), G.nodes[v].get("x", 0)
                if not (ux or uy) or not (vx or vy):
                    continue
                t = round(data.get("time_min", 0.0), 2)
                features.append({
                    "type": "Feature",
                    "geometry": mapping(LineString([(ux, uy), (vx, vy)])),
                    "properties": {
                        "mode": mode, "time_min": t,
                        "u": str(u), "v": str(v),
                    },
                })
            except Exception:
                continue

        if not features:
            continue

        _edge_click = folium.JsCode("""
        function(feature, layer) {
            layer.on('click', function(e) {
                var p = feature.properties;
                if (p.u === undefined) return;
                var msg = {u: String(p.u), v: String(p.v),
                           time_min: p.time_min, mode: p.mode};
                try {
                    localStorage.setItem('_sc_edge_click',
                        JSON.stringify({u: msg.u, v: msg.v,
                                        time_min: msg.time_min, mode: msg.mode,
                                        t: Date.now()}));
                } catch(ex) {}
                try {
                    if (typeof window.parent.scReceiveEdge === 'function') {
                        window.parent.scReceiveEdge(msg); return;
                    }
                } catch(ex2) {}
                try {
                    window.parent.postMessage(
                        {type:'edge-selected', u: msg.u, v: msg.v,
                         time_min: msg.time_min, mode: msg.mode}, '*');
                } catch(ex3) {}
            });
        }
        """)

        fc    = {"type": "FeatureCollection", "features": features}
        layer = folium.FeatureGroup(name=f"{mode.title()} Network", show=show)
        folium.GeoJson(
            fc,
            style_function=lambda feat, c=color: {
                "color": c, "weight": 2.2, "opacity": 0.75,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["mode", "u", "v", "time_min"],
                aliases=["Mode:", "From:", "To:", "Time (min):"],
            ),
            on_each_feature=_edge_click,
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
        # Fallback for zero-weight layers (food_retail, transit): those categories
        # don't enter PCI composite via the weighted formula, so we approximate:
        # each unit of count adds 5 % of city-mean composite mass.
        # Callers can pass a large count to scale up (e.g. the batch runner uses
        # count = 20 so that mass_delta = city_mean, matching the main path).
        city_mean  = float(comp_mod[comp_mod > 0].mean()) if (comp_mod > 0).any() else 0.1
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
    return {
        "stats":          compute_impact_stats(baseline, modified),
        "delta_map_html": make_delta_map(
            grid.gdf, delta, baseline, modified, city_name
        )._repr_html_(),
        "insight_plot":   make_scenario_insight_plot(
            grid.gdf, baseline, modified, delta,
        ),
        "top_hexes":  top_affected_hexes(grid.gdf, delta),
        "n_affected": len(affected_hexes),
    }



# A hex is counted as improved / degraded only if the absolute score change
# exceeds this fraction of its own baseline score.  1 % means a hex that
# scores 40 must move by at least 0.4 points to register as changed; a hex
# that scores 80 must move by at least 0.8 points.  This adapts to the
# magnitude of the scenario rather than using a city-wide fixed cutoff.
RELATIVE_CHANGE_THRESHOLD = 0.01   # 1 % of each hex's baseline score


def compute_impact_stats(baseline: pd.Series, modified: pd.Series) -> dict:
    """Summary statistics comparing baseline vs modified index.

    Improved / degraded / unchanged are defined relative to each hex's own
    baseline score (threshold = RELATIVE_CHANGE_THRESHOLD × baseline_score),
    so the classification adapts to the scale of the scenario.
    """
    delta = (modified - baseline).dropna()
    b, m  = baseline.dropna(), modified.dropna()
    if len(delta) == 0:
        return {}

    # Per-hex threshold: 1 % of that hex's baseline score
    per_hex_thresh = baseline.reindex(delta.index).abs() * RELATIVE_CHANGE_THRESHOLD

    return {
        "baseline_mean":        round(float(b.mean()),          2),
        "modified_mean":        round(float(m.mean()),          2),
        "mean_delta":           round(float(delta.mean()),      2),
        "median_delta":         round(float(delta.median()),    2),
        "max_gain":             round(float(delta.max()),       2),
        "max_loss":             round(float(delta.min()),       2),
        # Threshold-gated counts (used in single-scenario tab)
        "n_improved":           int((delta >  per_hex_thresh).sum()),
        "n_degraded":           int((delta < -per_hex_thresh).sum()),
        "n_unchanged":          int((delta.abs() <= per_hex_thresh).sum()),
        "change_threshold_pct": int(RELATIVE_CHANGE_THRESHOLD * 100),
        # Raw counts — any nonzero delta, no threshold (used in batch plot)
        "n_any_improved":       int((delta > 0).sum()),
        "n_any_degraded":       int((delta < 0).sum()),
        "p10_delta":            round(float(delta.quantile(0.10)), 2),
        "p25_delta":            round(float(delta.quantile(0.25)), 2),
        "p75_delta":            round(float(delta.quantile(0.75)), 2),
        "p90_delta":            round(float(delta.quantile(0.90)), 2),
    }


def make_scenario_insight_plot(
    grid_gdf:  gpd.GeoDataFrame,
    baseline:  pd.Series,
    modified:  pd.Series,
    delta:     pd.Series,
    index_label: str = "Score",
) -> str:
    """Three-panel insight figure returned as a base64 PNG string.

    Panel 1 — Distribution shift (KDE before/after)
        Shows whether the scenario shifts the whole score distribution or
        only the tails.  Green fill = the modified curve exceeds baseline
        (more hexes in that range gained); red fill = fewer hexes there.

    Panel 2 — Baseline score vs Δ score scatter
        Each point is one hex.  The trend line answers the equity question:
          • Negative slope → progressive: low-scoring (underserved) hexes
            benefit more than high-scoring ones.
          • Positive slope → regressive: already well-connected hexes gain
            the most (or the underserved ones lose the most).
          • Flat → effect is spatially uniform.

    Panel 3 — Mean Δ per neighbourhood (top/bottom 15 by absolute impact)
        Gives the political/planning read: which named areas win or lose.
        Omitted if no neighbourhood column is present in the grid.
    """
    matplotlib.use("Agg")
    from analysis.shared import _fig_to_base64

    # ── Webapp dark-theme colours ──────────────────────────────────────────
    BG      = "#12141f"
    SURFACE = "#1a1d2e"
    TEXT    = "#e8eaf0"
    MUTED   = "#7b7f9e"
    ACCENT  = "#7c6ee0"
    GREEN   = "#4caf50"
    RED     = "#ef5350"
    GRIDL   = "#2a2d3e"

    b = baseline.dropna()
    m = modified.dropna()
    d = delta.dropna()

    has_nb = "neighborhood" in grid_gdf.columns
    ncols  = 3 if has_nb else 2
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 6.5))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRIDL)

    # ── Panel 1: distribution shift ────────────────────────────────────────
    ax1 = axes[0]
    x_lo = min(b.min(), m.min()) - 2
    x_hi = max(b.max(), m.max()) + 2
    xs   = np.linspace(x_lo, x_hi, 300)
    if len(b) > 1 and len(m) > 1:
        kde_b = gaussian_kde(b)(xs)
        kde_m = gaussian_kde(m)(xs)
        ax1.plot(xs, kde_b, color=MUTED,   lw=2, label="Baseline")
        ax1.plot(xs, kde_m, color=ACCENT,  lw=2, label="Modified")
        ax1.fill_between(xs, kde_b, kde_m, where=(kde_m >= kde_b),
                         alpha=0.25, color=GREEN, label="More hexes here")
        ax1.fill_between(xs, kde_b, kde_m, where=(kde_m <  kde_b),
                         alpha=0.25, color=RED,   label="Fewer hexes here")
    ax1.set_xlabel(index_label, fontsize=9)
    ax1.set_ylabel("Density", fontsize=9)
    ax1.set_title("Score Distribution Before vs After", fontsize=10, pad=10)
    ax1.legend(fontsize=8, labelcolor=TEXT, facecolor=SURFACE, edgecolor=GRIDL,
               framealpha=0.9)
    ax1.yaxis.grid(True, color=GRIDL, linewidth=0.5)
    ax1.margins(x=0.05, y=0.12)

    # ── Panel 2: baseline vs delta scatter ─────────────────────────────────
    ax2 = axes[1]
    common = b.index.intersection(d.index)
    bv = b.reindex(common).values
    dv = d.reindex(common).values
    pt_colors = np.where(dv > 0, GREEN, np.where(dv < 0, RED, MUTED))
    ax2.scatter(bv, dv, c=pt_colors, alpha=0.45, s=22, linewidths=0,
                zorder=3)
    ax2.axhline(0, color=MUTED, lw=1.2, linestyle="--",
                label="No change", zorder=2)

    if len(bv) > 2:
        slope, intercept = np.polyfit(bv, dv, 1)
        xs2 = np.linspace(bv.min(), bv.max(), 100)
        trend_col = GREEN if slope < 0 else RED
        ax2.plot(xs2, slope * xs2 + intercept, color=trend_col,
                 lw=2, linestyle="--",
                 label=f"Trend  (slope {slope:+.3f})")
        ax2.legend(fontsize=8, labelcolor=TEXT, facecolor=SURFACE,
                   edgecolor=GRIDL, framealpha=0.9)

        if slope < -0.01:
            note, nc = "Progressive — underserved areas benefit more", GREEN
        elif slope > 0.01:
            note, nc = "Regressive — well-served areas benefit more", RED
        else:
            note, nc = "Uniform — effect evenly distributed", MUTED
        ax2.text(0.04, 0.97, note, transform=ax2.transAxes,
                 fontsize=8.5, color=nc, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE,
                           edgecolor=GRIDL, alpha=0.85))

    # Colour guide for scatter dots
    ax2.scatter([], [], color=GREEN, s=40, label="Score improved (Δ > 0)")
    ax2.scatter([], [], color=RED,   s=40, label="Score worsened (Δ < 0)")

    ax2.set_xlabel(f"Baseline {index_label} (before scenario)", fontsize=9)
    ax2.set_ylabel("Score Change (Δ after − before)", fontsize=9)
    ax2.set_title("Who Benefits?\nDo Low-Scoring or High-Scoring Hexes Gain More?",
                  fontsize=10, pad=10)
    ax2.yaxis.grid(True, color=GRIDL, linewidth=0.5)
    ax2.margins(x=0.08, y=0.15)

    # ── Panel 3: neighbourhood bar chart ───────────────────────────────────
    if has_nb:
        ax3 = axes[2]
        gdf = grid_gdf[["hex_id", "neighborhood"]].copy()
        gdf["delta"] = gdf["hex_id"].map(delta)
        nb_delta = (gdf.dropna(subset=["delta"])
                       .groupby("neighborhood")["delta"]
                       .mean()
                       .sort_values())

        # Keep at most 15 bottom + 15 top to avoid an unreadable wall of bars
        if len(nb_delta) > 30:
            nb_delta = pd.concat([nb_delta.head(15), nb_delta.tail(15)])

        bar_colors = [GREEN if v >= 0 else RED for v in nb_delta.values]
        bars = ax3.barh(nb_delta.index, nb_delta.values,
                        color=bar_colors, alpha=0.82)
        ax3.axvline(0, color=MUTED, lw=1.2)
        # Value labels on bars
        for bar, val in zip(bars, nb_delta.values):
            ha   = "left" if val >= 0 else "right"
            xpos = val + (nb_delta.abs().max() * 0.015) * (1 if val >= 0 else -1)
            ax3.text(xpos, bar.get_y() + bar.get_height() / 2,
                     f"{val:+.3f}", va="center", ha=ha,
                     fontsize=6.5, color=TEXT)
        ax3.set_xlabel("Mean Score Change (Δ)", fontsize=9)
        ax3.set_title("Which Neighbourhoods Win or Lose?\n(top/bottom 15 by impact)",
                      fontsize=10, pad=10)
        ax3.tick_params(axis="y", labelsize=7)
        ax3.xaxis.grid(True, color=GRIDL, linewidth=0.5)
        for lbl in ax3.get_yticklabels():
            lbl.set_color(TEXT)
        # Legend
        ax3.barh([], [], color=GREEN, alpha=0.82, label="Net improvement")
        ax3.barh([], [], color=RED,   alpha=0.82, label="Net degradation")
        ax3.legend(fontsize=8, labelcolor=TEXT, facecolor=SURFACE,
                   edgecolor=GRIDL, framealpha=0.9, loc="lower right")
        # Padding so value labels don't clip
        x_lo, x_hi = ax3.get_xlim()
        pad = nb_delta.abs().max() * 0.15
        ax3.set_xlim(x_lo - pad, x_hi + pad)

    plt.tight_layout(pad=2.5)
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# Hidden Trends — batch scenario runner
# ---------------------------------------------------------------------------

_NETWORK_SCENARIO_TYPES = {"edge_penalty", "edge_remove"}


def run_batch_scenarios(
    s:             dict,
    index:         str        = "pci",
    scenario_type: str        = "amenity_remove",
    n_per_band:    int        = 2,
    seed:          int        = 42,
    amenity_type:  str        = "education",
    factor:        float      = 2.0,
    radii:         list       = None,
) -> dict:
    """Run a stratified batch of random scenarios for pattern discovery.

    Strategy
    --------
    Hexes are divided into three equal-size score bands (Low / Mid / High).
    ``n_per_band`` random starting hexes are drawn from each band per radius.
    The ``seed`` pins which hexes are selected — the same seed across different
    scenario types selects the exact same starting locations, enabling fair
    comparison.

    Fast-path scenarios (no Dijkstra recompute):
        PCI  → amenity_remove | amenity_add
        BCI  → supplier_remove | supplier_add

    Network scenarios (full Dijkstra recompute — slow):
        PCI/BCI → edge_penalty | edge_remove
        Forced to 1 run per stratum, radius 0 only — 3 total runs.
        Expect 2–10 min per run.

    Parameters
    ----------
    n_per_band : int
        Runs per (score band × radius) cell (max 20).
    radii : list of int, optional
        Radius values to iterate over (0 = single hex, 1 = hex+neighbours).
        Defaults to [0, 1] for remove scenarios and [0] for add scenarios.
        Network scenarios are always forced to [0].
    amenity_type : str
        Amenity category used for amenity_add runs (e.g. "education").
    factor : float
        Travel-time multiplier for edge_penalty runs.

    Returns
    -------
    dict with keys:
        runs        — list of per-run dicts
        batch_plot  — aggregated insight figure as base64 PNG
    """
    rng = np.random.default_rng(seed)

    scores = s.get(index, pd.Series(dtype=float)).dropna()
    if len(scores) == 0:
        return {"runs": [], "batch_plot": ""}

    q33 = scores.quantile(1 / 3)
    q67 = scores.quantile(2 / 3)
    strata = {
        "Low":  scores[scores <= q33].index.tolist(),
        "Mid":  scores[(scores > q33) & (scores <= q67)].index.tolist(),
        "High": scores[scores > q67].index.tolist(),
    }

    is_network = scenario_type in _NETWORK_SCENARIO_TYPES
    if is_network:
        # Network scenarios are slow — 1 run per stratum, radius 0 only
        radii      = [0]
        n_per_band = 1
    elif radii is not None and len(radii) > 0:
        radii = sorted(set(int(r) for r in radii if int(r) >= 0))
    elif "remove" in scenario_type:
        radii = [0, 1]
    else:
        radii = [0]

    # ── Pre-compute a meaningful "add" count for amenity_add ──────────────────
    # Amenity layers differ wildly in unit: parks are in m² (range can be
    # 0–500 000 m²) while education/health are discrete counts (range 0–20).
    # Using count=1.0 for parks adds ≈1 m² — invisible to the model.
    #
    # Fix: pass count = raw_range × 0.1 (10 % of the layer's full scale).
    # This makes mass_delta = weight × 0.1 / total_weight — consistent across
    # ALL amenity types regardless of unit, because raw_range cancels out:
    #   mass_delta = weight × (raw_range × 0.1) / (raw_range × total_weight)
    #             = weight × 0.1 / total_weight
    _amenity_add_count = 1.0
    if scenario_type == "amenity_add":
        mc    = s.get("mass_calc")
        layer = mc.layers.get(amenity_type) if mc else None
        if layer is not None and mc._composite is not None:
            city_mean = float(mc._composite[mc._composite > 0].mean()) if (mc._composite > 0).any() else 0.1
            if layer.weight > 0:
                # Principled sizing: target mass_delta = city_mean (same scale as what
                # amenity_remove subtracts from a median hex).
                # Solving  weight × count / (raw_range × total_weight) = city_mean:
                #   count = city_mean × raw_range × total_weight / weight
                # raw_range cancels in the formula, so result is unit-consistent across
                # amenity types (parks m² vs discrete counts like education/health).
                total_weight = sum(l.weight for l in mc.layers.values() if l.weight > 0) or 1.0
                raw_range    = float(layer.raw_values.max() - layer.raw_values.min())
                raw_range    = max(raw_range, 1.0)
                _amenity_add_count = city_mean * raw_range * total_weight / layer.weight
            else:
                # weight=0 layer (food_retail, transit): fallback uses count × city_mean × 0.05,
                # so count = 20 gives mass_delta = city_mean (same target as above).
                _amenity_add_count = 20.0

    runs = []
    for stratum, hex_pool in strata.items():
        if not hex_pool:
            continue
        for radius in radii:
            for _ in range(n_per_band):
                hex_id        = str(rng.choice(hex_pool))
                expanded      = expand_hexes([hex_id], radius)
                valid_targets = [h for h in expanded if h in scores.index]
                target_mean   = (float(scores.reindex(valid_targets).mean())
                                 if valid_targets else None)
                label = f"{stratum} · radius {radius} · {hex_id[:8]}…"
                try:
                    if scenario_type == "amenity_remove":
                        result = run_pci_amenity_removal(s, [hex_id], radius)
                    elif scenario_type == "amenity_add":
                        result = run_pci_amenity_addition(
                            s, hex_id, amenity_type, _amenity_add_count)
                    elif scenario_type == "supplier_remove":
                        result = run_bci_supplier_change(
                            s, [hex_id], "remove", 1.0, radius)
                    elif scenario_type == "supplier_add":
                        result = run_bci_supplier_change(
                            s, [hex_id], "add", 1.0, radius)
                    elif scenario_type == "edge_penalty" and index == "pci":
                        result = run_pci_edge_penalty(s, [hex_id], factor, radius)
                    elif scenario_type == "edge_penalty" and index == "bci":
                        result = run_bci_edge_penalty(s, [hex_id], factor, radius)
                    elif scenario_type == "edge_remove" and index == "pci":
                        result = run_pci_edge_removal(s, [hex_id], radius)
                    elif scenario_type == "edge_remove" and index == "bci":
                        result = run_bci_edge_removal(s, [hex_id], radius)
                    else:
                        continue

                    runs.append({
                        "label":        label,
                        "score_bucket": stratum,
                        "radius":       radius,
                        "hex_id":       hex_id,
                        "target_mean":  (round(target_mean, 2)
                                         if target_mean is not None else None),
                        "n_target":     len(expanded),
                        "stats":        result.get("stats", {}),
                        "error":        None,
                    })
                except Exception as exc:
                    runs.append({
                        "label":        label,
                        "score_bucket": stratum,
                        "radius":       radius,
                        "hex_id":       hex_id,
                        "target_mean":  None,
                        "n_target":     0,
                        "stats":        {},
                        "error":        str(exc),
                    })

    batch_plot = make_batch_insight_plot(runs) if runs else ""
    return {"runs": runs, "batch_plot": batch_plot}


def make_batch_insight_plot(runs: list, index_label: str = "Score") -> str:
    """Aggregated three-panel insight figure for the Hidden Trends tab.

    Panel 1 — Target score vs city-wide Δ (scatter)
        X-axis = mean baseline score of the modified hexes — i.e. where in
        the score distribution the intervention landed.
        Y-axis = city-wide mean Δ after the scenario.
        A trend line across all runs reveals whether intervening in
        already well-connected (high-score) areas produces larger or smaller
        city-wide shifts than intervening in underserved (low-score) areas.

    Panel 2 — Mean city-wide Δ by score bucket and radius (grouped bars)
        Directly compares: does location (stratum) or scale (radius) matter
        more?  Green bars = net city improvement, red = net loss.

    Panel 3 — # Hexes improved vs # hexes degraded per run (scatter)
        Points above the diagonal = the run produced a net gain across the
        city (more hexes improved than worsened).  Color = score bucket.
        Reveals whether the effect is concentrated or diffuse, and whether
        certain buckets consistently produce lop-sided outcomes.
    """
    matplotlib.use("Agg")
    from analysis.shared import _fig_to_base64

    BG, SURFACE = "#12141f", "#1a1d2e"
    TEXT, MUTED, GRIDL = "#e8eaf0", "#7b7f9e", "#2a2d3e"
    GREEN, RED   = "#4caf50", "#ef5350"
    # Each radius gets a distinct hue so legend colour = radius, not sign
    RADIUS_COLORS = ["#7c6ee0", "#ffa726", "#26c6da", "#ab47bc"]
    BUCKET_COLORS = {"Low": "#ef5350", "Mid": "#ffa726", "High": "#42a5f5"}
    BUCKET_LABELS = {
        "Low":  "Low\n(underserved)",
        "Mid":  "Mid",
        "High": "High\n(well-served)",
    }

    valid = [r for r in runs if not r.get("error") and r.get("stats")]
    if not valid:
        return ""

    fig, axes = plt.subplots(3, 1, figsize=(16, 24))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRIDL)

    # ── Panel 1: target score vs city-wide delta ────────────────────────────
    ax1 = axes[0]
    for bucket, color in BUCKET_COLORS.items():
        pts = [r for r in valid
               if r["score_bucket"] == bucket
               and r["target_mean"] is not None]
        if pts:
            ax1.scatter([p["target_mean"] for p in pts],
                        [p["stats"].get("mean_delta", 0) for p in pts],
                        color=color, s=90, alpha=0.85,
                        label=f"{BUCKET_LABELS[bucket]} — interventions in this score range",
                        zorder=3, clip_on=False)
    ax1.axhline(0, color=MUTED, lw=1.2, linestyle="--",
                label="Zero change (no city-wide effect)", zorder=2)
    all_x = [r["target_mean"] for r in valid if r["target_mean"] is not None]
    all_y = [r["stats"].get("mean_delta", 0) for r in valid
             if r["target_mean"] is not None]
    if len(all_x) > 2:
        slope, intercept = np.polyfit(all_x, all_y, 1)
        xs_line = np.linspace(min(all_x), max(all_x), 100)
        trend_desc = ("downward → underserved areas are higher-leverage"
                      if slope < -0.01 else
                      "upward → well-served areas are higher-leverage"
                      if slope > 0.01 else "flat → leverage is similar across bands")
        ax1.plot(xs_line, slope * xs_line + intercept,
                 color=TEXT, lw=1.6, linestyle="--", alpha=0.6,
                 label=f"Trend line ({trend_desc})")
    ax1.set_xlabel(f"Baseline {index_label} of Targeted Hexes", fontsize=9)
    ax1.set_ylabel("City-wide Mean Score Change (Δ)", fontsize=9)
    ax1.set_title("Does Targeting Underserved Areas Move the City More?",
                  fontsize=10, pad=10)
    ax1.legend(fontsize=8, labelcolor=TEXT, facecolor=SURFACE,
               edgecolor=GRIDL, framealpha=0.9, loc="best")
    ax1.yaxis.grid(True, color=GRIDL, linewidth=0.5)
    ax1.margins(x=0.15, y=0.22)

    # ── Panel 2: grouped bar — delta by (bucket × radius) ──────────────────
    ax2 = axes[1]
    buckets       = ["Low", "Mid", "High"]
    radii_present = sorted(set(r["radius"] for r in valid))
    n_radii = len(radii_present)
    x       = np.arange(len(buckets))
    bar_w   = 0.6 / max(n_radii, 1)

    for i, radius in enumerate(radii_present):
        deltas = []
        for bucket in buckets:
            pts = [r["stats"].get("mean_delta", 0)
                   for r in valid
                   if r["score_bucket"] == bucket and r["radius"] == radius]
            deltas.append(float(np.mean(pts)) if pts else 0.0)
        offset    = (i - (n_radii - 1) / 2) * bar_w
        base_col  = RADIUS_COLORS[i % len(RADIUS_COLORS)]
        bars = ax2.bar(x + offset, deltas, bar_w * 0.88,
                       color=base_col, alpha=0.82,
                       edgecolor="none")
        # Value labels on each bar
        for bar, d in zip(bars, deltas):
            if d == 0:
                continue
            va  = "bottom" if d >= 0 else "top"
            ypos = bar.get_height() if d >= 0 else bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     ypos + (0.0005 if d >= 0 else -0.0005),
                     f"{d:+.4f}", ha="center", va=va,
                     fontsize=7, color=TEXT)

    ax2.set_xticks(x)
    ax2.set_xticklabels([BUCKET_LABELS[b] for b in buckets], fontsize=9)
    ax2.axhline(0, color=MUTED, lw=1.2, zorder=2)
    # Annotate the zero line
    y_range = ax2.get_ylim()
    ax2.text(x[-1] + 0.55, 0, " break-even", va="center",
             fontsize=7, color=MUTED, style="italic", clip_on=True)
    ax2.set_xlabel("Score Band of Intervention Location", fontsize=9)
    ax2.set_ylabel("Mean City-wide Score Change (Δ)", fontsize=9)
    ax2.set_title("Does Location or Scale Drive the Effect?",
                  fontsize=10, pad=10)
    # Build clean legend handles — solid fill only, no border artifacts
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=RADIUS_COLORS[i % len(RADIUS_COLORS)], edgecolor="none",
              label=f"Radius {radius}  "
                    f"({'single hex' if radius == 0 else f'hex + {6*radius} neighbours'})")
        for i, radius in enumerate(radii_present)
    ]
    ax2.legend(handles=legend_handles, fontsize=8, labelcolor=TEXT,
               facecolor=SURFACE, edgecolor=GRIDL, framealpha=0.9, loc="best")
    ax2.yaxis.grid(True, color=GRIDL, linewidth=0.5, zorder=0)
    # Pad y-axis so value labels don't clip
    y_lo, y_hi = ax2.get_ylim()
    pad = max(abs(y_hi - y_lo) * 0.18, 1e-6)
    ax2.set_ylim(y_lo - pad, y_hi + pad)

    # ── Panel 3: hexes improved vs degraded per run (raw, no threshold) ────────
    # Use n_any_improved / n_any_degraded (any nonzero delta) rather than the
    # threshold-gated counts.  The 1 % relative threshold is correct for the
    # single-scenario tab but too strict for "add" batch scenarios where each
    # run adds one facility — real but small effects would all show as 0 / 0.
    ax3 = axes[2]
    from collections import Counter
    coord_count: Counter = Counter()
    coord_colors: dict   = {}
    all_ni, all_nd = [], []
    for r in valid:
        st    = r["stats"]
        ni    = st.get("n_any_improved", st.get("n_improved", 0))
        nd    = st.get("n_any_degraded", st.get("n_degraded", 0))
        key   = (nd, ni)
        coord_count[key] += 1
        # Store the color of the last-seen bucket at each coordinate
        # (spread will reveal per-bucket info; overlap is shown via size)
        coord_colors[key] = BUCKET_COLORS.get(r["score_bucket"], MUTED)
        all_ni.append(ni)
        all_nd.append(nd)

    for (nd, ni), cnt in coord_count.items():
        # Dot area scales with overlap count; edgecolor shows whether stacked
        size = 80 + 40 * (cnt - 1)
        edge = TEXT if cnt > 1 else "none"
        ax3.scatter(nd, ni, color=coord_colors[(nd, ni)],
                    s=size, alpha=0.85, zorder=4, clip_on=False,
                    edgecolors=edge, linewidths=0.8)
        if cnt > 1:
            ax3.text(nd, ni, f"×{cnt}", ha="center", va="bottom",
                     fontsize=7, color=TEXT, zorder=5)

    # Compute axis limits AFTER plotting so all points are accounted for
    max_val = max(max(all_ni, default=1), max(all_nd, default=1))
    # If all points are at origin, flag it and use a small fixed range
    all_zero = max_val == 0
    lim = max(max_val * 1.30, 1)   # at least 1 so axis is visible
    diag = np.linspace(0, lim, 50)
    ax3.plot(diag, diag, color=MUTED, lw=1.4, linestyle="--",
             alpha=0.7, zorder=2,
             label="Break-even line — equal hexes improved & degraded")
    # Shade net-gain / net-loss regions
    ax3.fill_between(diag, diag, lim, alpha=0.07, color=GREEN, zorder=1)
    ax3.fill_between(diag, 0,    diag, alpha=0.07, color=RED,   zorder=1)
    ax3.text(lim * 0.05, lim * 0.90,
             "Net city GAIN zone\n(more hexes improved than worsened)",
             fontsize=8, color=GREEN, va="top")
    ax3.text(lim * 0.45, lim * 0.10,
             "Net city LOSS zone\n(more hexes worsened than improved)",
             fontsize=8, color=RED, va="bottom")
    if all_zero:
        ax3.text(0.5, 0.5,
                 "All runs produced negligible change\n"
                 "(scenario effect below 1% threshold)",
                 transform=ax3.transAxes, ha="center", va="center",
                 fontsize=9, color=MUTED, style="italic",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE,
                           edgecolor=GRIDL, alpha=0.9))
    ax3.set_xlim(0, lim)
    ax3.set_ylim(0, lim)
    ax3.set_xlabel("# Hexes whose score decreased  (number of hexes in the city)", fontsize=9)
    ax3.set_ylabel("# Hexes whose score increased  (number of hexes in the city)", fontsize=9)
    ax3.set_title("Does Each Run Help or Hurt More of the City?\n"
                  "(any nonzero Δ counts · dot size = # overlapping runs)",
                  fontsize=10, pad=10)
    ax3.yaxis.grid(True, color=GRIDL, linewidth=0.5)
    ax3.xaxis.grid(True, color=GRIDL, linewidth=0.5)
    for bucket, color in BUCKET_COLORS.items():
        ax3.scatter([], [], color=color, s=60,
                    label=f"{BUCKET_LABELS[bucket]} — intervention started here")
    ax3.legend(fontsize=8, labelcolor=TEXT, facecolor=SURFACE,
               edgecolor=GRIDL, framealpha=0.9, loc="upper left")

    plt.tight_layout(pad=2.5)
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


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
