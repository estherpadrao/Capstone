"""
File 15a: Network Diagnostics
==============================
Network validation and mass topography helpers extracted from isochrones.py.
Called by the /diagnostics route in the webapp.

Functions
---------
validate_network       -- comprehensive connectivity / reachability check
print_mass_topography  -- text summary of the composite mass surface
"""

import numpy as np
import networkx as nx
from typing import Dict

from core.h3_helper import HexGrid
from core.network_builder import MultiModalNetworkBuilder
from core.mass_calculator import MassCalculator


def validate_network(
    network_builder: MultiModalNetworkBuilder,
    grid: HexGrid,
    verbose: bool = True,
) -> Dict:
    """
    Comprehensive network diagnostics:
      - node/edge counts per mode
      - weak connectivity + component sizes
      - edge weight distribution
      - hex-to-node reachability check
    Returns a dict of stats (also prints if verbose=True).
    """
    results = {}
    G_unified = network_builder.unified_graph

    if G_unified is None:
        return {"error": "Networks not built yet."}

    # --- Per-mode stats ---
    mode_stats = {}
    for mode, G in network_builder.networks.items():
        comps   = list(nx.weakly_connected_components(G))
        n_comps = len(comps)
        largest = max(len(c) for c in comps) if comps else 0
        mode_stats[mode] = {
            "nodes":     G.number_of_nodes(),
            "edges":     G.number_of_edges(),
            "connected": nx.is_weakly_connected(G),
            "n_components": n_comps,
            "largest_component": largest,
        }

    # --- Unified graph ---
    u_comps  = list(nx.weakly_connected_components(G_unified))
    time_vals = [d["time_min"] for _, _, d in G_unified.edges(data=True) if "time_min" in d]
    mode_counts: Dict[str, int] = {}
    for _, _, d in G_unified.edges(data=True):
        m = d.get("mode", "unknown")
        mode_counts[m] = mode_counts.get(m, 0) + 1

    unified_stats = {
        "nodes":       G_unified.number_of_nodes(),
        "edges":       G_unified.number_of_edges(),
        "connected":   nx.is_weakly_connected(G_unified),
        "n_components": len(u_comps),
        "largest_component": max(len(c) for c in u_comps) if u_comps else 0,
        "time_min_mean": float(np.mean(time_vals)) if time_vals else 0,
        "time_min_max":  float(np.max(time_vals))  if time_vals else 0,
        "edges_by_mode": mode_counts,
    }

    # --- Hex-to-node coverage ---
    centroids = grid.centroids
    hex_ids   = centroids["hex_id"].tolist()
    coords    = [(r.geometry.y, r.geometry.x) for _, r in centroids.iterrows()]
    nearest   = network_builder.get_nearest_nodes_batch(coords, mode="unified")
    unique_near = len(set(nearest))
    hex_coverage = {
        "n_hexes":          len(hex_ids),
        "n_unique_nodes":   unique_near,
        "coverage_ratio":   round(unique_near / max(len(hex_ids), 1), 3),
    }

    results = {
        "mode_stats":    mode_stats,
        "unified":       unified_stats,
        "hex_coverage":  hex_coverage,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("🔍 NETWORK VALIDATION")
        print("=" * 60)
        for mode, s in mode_stats.items():
            conn = "✅" if s["connected"] else f"⚠ {s['n_components']} components"
            print(f"\n  {mode.upper()}: {s['nodes']:,} nodes | {s['edges']:,} edges | {conn}")
            if not s["connected"]:
                print(f"     Largest component: {s['largest_component']:,} nodes")

        u = unified_stats
        print(f"\n  UNIFIED: {u['nodes']:,} nodes | {u['edges']:,} edges")
        print(f"   Connected: {'✅' if u['connected'] else '⚠'}")
        print(f"   time_min: mean={u['time_min_mean']:.2f}, max={u['time_min_max']:.2f}")
        print(f"   Edges by mode: {u['edges_by_mode']}")
        hc = hex_coverage
        print(f"\n  HEX→NODE: {hc['n_hexes']} hexes mapped to "
              f"{hc['n_unique_nodes']} unique nodes "
              f"(ratio {hc['coverage_ratio']:.2f})")
        print("=" * 60)

    return results


def print_mass_topography(
    grid: HexGrid,
    mass_calc: MassCalculator,
    city_name: str = "",
):
    """Print a text summary of the topographic mass surface."""
    print("\n" + "=" * 60)
    print(f"🏔  MASS TOPOGRAPHY — {city_name}")
    print("=" * 60)

    df = mass_calc.summary()
    print(df.to_string(index=False))

    if mass_calc._composite is not None:
        c = mass_calc._composite
        print(f"\n  Composite mass: min={c.min():.4f}, max={c.max():.4f}, "
              f"mean={c.mean():.4f}, std={c.std():.4f}")
        peaks   = mass_calc.identify_peaks(90)
        valleys = mass_calc.identify_valleys(10)
        print(f"  Peaks (top 10%):   {len(peaks)} hexes")
        print(f"  Valleys (bot 10%): {len(valleys)} hexes")
    print("=" * 60)
