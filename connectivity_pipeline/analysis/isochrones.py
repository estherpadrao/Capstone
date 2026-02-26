"""
File 15: Isochrones & Network Diagnostics
==========================================
- Network validation (connectivity, edge weights, node reachability)
- Mass topography print-out
- Isochrone generation for top/bottom 10% hexes (PCI & BCI)
- Amenity / population / business counts within isochrones
- Summary comparison tables
"""

import math
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import folium
try:
    from folium.plugins import LayerControl
except ImportError:
    from folium import LayerControl
from shapely.geometry import Point, MultiPoint, Polygon
from typing import Dict, List, Optional, Tuple

from core.h3_helper import HexGrid
from core.network_builder import MultiModalNetworkBuilder
from core.mass_calculator import MassCalculator

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Isochrone configuration
# ---------------------------------------------------------------------------

ISO_THRESHOLDS  = [10, 20, 30]   # minutes
ISO_MODE_CONFIG = {
    "walk":    {"speed_kmh": 4.8,  "color": "#2ecc71", "osm_type": "walk"},
    "bike":    {"speed_kmh": 15.0, "color": "#3498db", "osm_type": "bike"},
    "drive":   {"speed_kmh": 24.0, "color": "#e74c3c", "osm_type": "drive"},
    "transit": {"speed_kmh": 13.0, "color": "#9b59b6", "osm_type": "walk"},
}
ISO_OPACITY = {10: 0.50, 20: 0.35, 30: 0.20}


# ===========================================================================
# PART 1: Network diagnostics
# ===========================================================================

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


# ===========================================================================
# PART 2: Isochrone generator
# ===========================================================================

class IsochroneBuilder:
    """
    Builds isochrone polygons (convex hull of reachable network nodes)
    for each (origin, mode, threshold) combination.
    Networks are cached across calls.
    """

    def __init__(self, city_name: str):
        self.city_name   = city_name
        self._networks: Dict[str, object] = {}

    def _get_network(self, mode: str):
        if mode not in self._networks:
            print(f"   📡 Downloading {mode} network for {self.city_name}...")
            osm_type = ISO_MODE_CONFIG[mode]["osm_type"]
            G = ox.graph_from_place(self.city_name, network_type=osm_type, simplify=True)
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
            self._networks[mode] = G
        return self._networks[mode]

    def _override_speed(self, G, speed_kmh: float):
        speed_ms = speed_kmh * 1000 / 3600
        for u, v, k, d in G.edges(data=True, keys=True):
            G[u][v][k]["travel_time"] = d.get("length", 100) / speed_ms
        return G

    def build(
        self, lat: float, lng: float, mode: str, threshold_min: int
    ) -> Optional[Polygon]:
        speed_kmh = ISO_MODE_CONFIG[mode]["speed_kmh"]
        cutoff_s  = threshold_min * 60
        try:
            G = self._get_network(mode)
            if mode == "transit":
                G = self._override_speed(G.copy(), speed_kmh)

            orig = ox.nearest_nodes(G, lng, lat)
            sub  = nx.ego_graph(G, orig, radius=cutoff_s, distance="travel_time")

            if len(sub.nodes) < 3:
                return self._buffer(lat, lng, speed_kmh, threshold_min)

            pts  = [Point(d["x"], d["y"]) for _, d in sub.nodes(data=True)]
            hull = MultiPoint(pts).convex_hull.buffer(0.001)
            return hull
        except Exception as e:
            print(f"      ⚠  {mode} {threshold_min}min: {e}")
            return self._buffer(lat, lng, speed_kmh, threshold_min)

    @staticmethod
    def _buffer(lat, lng, speed_kmh, threshold_min) -> Polygon:
        dist_km  = speed_kmh * (threshold_min / 60)
        dist_deg = dist_km / 111.0
        return Point(lng, lat).buffer(dist_deg)


# ===========================================================================
# PART 3: Amenity / demand counter inside isochrones
# ===========================================================================

class IsochroneCounter:
    """Count what falls inside an isochrone polygon."""

    def __init__(self, amenities: dict, grid_gdf: gpd.GeoDataFrame):
        self.amenities = amenities   # {name: GeoDataFrame}
        self.grid      = grid_gdf.to_crs("EPSG:4326")

    def count_amenities(self, poly: Polygon) -> dict:
        poly_gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
        counts   = {}
        for name, gdf in self.amenities.items():
            if gdf is None or len(gdf) == 0:
                counts[name] = 0
                continue
            try:
                pts = gdf.to_crs("EPSG:4326").copy()
                mask = pts.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
                if mask.any():
                    pts.loc[mask, "geometry"] = pts.loc[mask].geometry.centroid
                joined = gpd.sjoin(pts[["geometry"]], poly_gdf, how="inner", predicate="within")
                counts[name] = len(joined)
            except Exception:
                counts[name] = 0
        counts["total_amenities"] = sum(v for k, v in counts.items() if k != "total_amenities")
        return counts

    def count_demand(self, poly: Polygon) -> dict:
        poly_gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
        cols = [c for c in ["hex_id", "geometry", "population", "supplier_mass", "median_income"]
                if c in self.grid.columns]
        try:
            hexes = gpd.sjoin(self.grid[cols].dropna(subset=["geometry"]),
                              poly_gdf, how="inner", predicate="intersects")
        except Exception:
            return {"population": 0, "businesses": 0, "market_demand_B": 0}

        pop = float(hexes["population"].sum())      if "population"    in hexes.columns else 0
        biz = float(hexes["supplier_mass"].sum())   if "supplier_mass" in hexes.columns else 0
        if "population" in hexes.columns and "median_income" in hexes.columns:
            mkt = float((hexes["population"] * hexes["median_income"].fillna(0)).sum() / 1e9)
        else:
            mkt = 0.0
        return {"population": round(pop), "businesses": round(biz, 2), "market_demand_B": round(mkt, 3)}


# ===========================================================================
# PART 4: Run full isochrone analysis
# ===========================================================================

def run_isochrone_analysis(
    grid: HexGrid,
    city_name: str,
    amenities: Optional[dict] = None,
    top_pct: float = 0.90,
    bottom_pct: float = 0.10,
    max_origins: int = 5,
) -> Dict:
    """
    Full isochrone workflow for PCI and BCI.
    Returns a dict with:
      - iso_gdf         : GeoDataFrame of all isochrone polygons
      - pci_counts_df   : amenity counts per isochrone
      - bci_counts_df   : demand counts per isochrone
      - pci_summary     : pivot table
      - bci_pop_summary : pivot table
      - bci_biz_summary : pivot table
    """
    builder = IsochroneBuilder(city_name)
    counter = IsochroneCounter(amenities or {}, grid.gdf)

    def _get_origins(col: str):
        valid = grid.gdf[grid.gdf[col].notna()].copy()
        top_t = valid[col].quantile(top_pct)
        bot_t = valid[col].quantile(bottom_pct)
        top_h = valid[valid[col] >= top_t].sort_values(col, ascending=False)
        bot_h = valid[valid[col] <= bot_t].sort_values(col, ascending=True)
        if max_origins:
            step_t = max(1, len(top_h) // max_origins)
            step_b = max(1, len(bot_h) // max_origins)
            top_h = top_h.iloc[::step_t][:max_origins]
            bot_h = bot_h.iloc[::step_b][:max_origins]
        print(f"   {col}: {len(top_h)} top / {len(bot_h)} bottom origins")
        return {"top": top_h, "bottom": bot_h}

    def _build_for_group(origins_gdf, group, index_col):
        rows = []
        o4326 = origins_gdf.to_crs("EPSG:4326")
        for _, row in o4326.iterrows():
            c   = row.geometry.centroid
            lat, lng = c.y, c.x
            for mode in ISO_MODE_CONFIG:
                for thr in ISO_THRESHOLDS:
                    print(f"      {index_col} {group} hex {row['hex_id']} | {mode} {thr}min...", end=" ")
                    poly = builder.build(lat, lng, mode, thr)
                    print("✓")
                    rows.append({
                        "hex_id": row["hex_id"], "group": group, "index": index_col,
                        "mode": mode, "threshold": thr,
                        "score": row[index_col], "lat": lat, "lng": lng,
                        "geometry": poly,
                    })
        return rows

    all_rows = []
    pci_counts_list = []
    bci_counts_list = []

    for index_col in [c for c in ["PCI", "BCI"] if c in grid.gdf.columns]:
        origins = _get_origins(index_col)
        for group, gdf in origins.items():
            rows = _build_for_group(gdf, group, index_col)
            all_rows.extend(rows)
            for r in rows:
                base = {k: r[k] for k in ["hex_id", "group", "mode", "threshold", "score"]}
                if index_col == "PCI":
                    pci_counts_list.append({**base, **counter.count_amenities(r["geometry"])})
                else:
                    bci_counts_list.append({**base, **counter.count_demand(r["geometry"])})

    iso_gdf = gpd.GeoDataFrame(all_rows, crs="EPSG:4326") if all_rows else gpd.GeoDataFrame()
    pci_df  = pd.DataFrame(pci_counts_list)
    bci_df  = pd.DataFrame(bci_counts_list)

    # Summary pivots
    def _pivot(df, val_cols):
        if df.empty:
            return pd.DataFrame()
        available = [c for c in val_cols if c in df.columns]
        if not available:
            return pd.DataFrame()
        pvt = df.groupby(["group", "mode", "threshold"])[available].mean().round(1)
        for col in available:
            top    = pvt.xs("top",    level="group", drop_level=False)[col]
            bottom = pvt.xs("bottom", level="group", drop_level=False)[col]
            # Attach ratio
        return pvt

    pci_summary = _pivot(pci_df, ["health", "education", "parks", "community",
                                   "food_retail", "transit", "total_amenities"])
    bci_pop_summary = _pivot(bci_df, ["population", "market_demand_B"])
    bci_biz_summary = _pivot(bci_df, ["businesses"])

    return {
        "iso_gdf":         iso_gdf,
        "pci_counts_df":   pci_df,
        "bci_counts_df":   bci_df,
        "pci_summary":     pci_summary,
        "bci_pop_summary": bci_pop_summary,
        "bci_biz_summary": bci_biz_summary,
    }


# ===========================================================================
# PART 5: Isochrone maps
# ===========================================================================

def make_pci_isochrone_map(
    grid: HexGrid,
    iso_gdf: gpd.GeoDataFrame,
    pci_counts_df: pd.DataFrame,
    pci_origins: dict,
    pci: pd.Series,
) -> folium.Map:
    """Interactive folium map for PCI isochrones."""
    from branca.colormap import LinearColormap

    center = tuple(grid.gdf.total_bounds[[1, 3]].mean()), tuple(grid.gdf.total_bounds[[0, 2]].mean())
    center = ((grid.gdf.total_bounds[1] + grid.gdf.total_bounds[3]) / 2,
              (grid.gdf.total_bounds[0] + grid.gdf.total_bounds[2]) / 2)
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    # PCI base layer
    valid = pci.dropna()
    cmap  = LinearColormap(
        ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"],
        vmin=0, vmax=100, caption="PCI"
    )
    gdf = grid.gdf.copy()
    gdf["PCI"] = gdf["hex_id"].map(pci)
    base_fg = folium.FeatureGroup(name="🗺 PCI Grid", show=True)
    for _, row in gdf.iterrows():
        val = row.get("PCI")
        color = cmap(val) if pd.notna(val) else "#ccc"
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda f, c=color: {"fillColor": c, "fillOpacity": 0.5, "weight": 0.2},
        ).add_to(base_fg)
    base_fg.add_to(m)
    cmap.add_to(m)

    # Isochrones
    pci_iso = iso_gdf[iso_gdf.get("index", iso_gdf.columns[0]) == "PCI"] if len(iso_gdf) else iso_gdf
    _add_isochrone_layers(m, pci_iso, pci_counts_df, "PCI")

    # Origin markers
    for group_label, color in [("top", "#f39c12"), ("bottom", "#c0392b")]:
        if group_label not in pci_origins:
            continue
        fg = folium.FeatureGroup(name=f"📍 {'Top' if group_label=='top' else 'Bottom'} 10% Origins (PCI)")
        for _, row in pci_origins[group_label].to_crs("EPSG:4326").iterrows():
            c = row.geometry.centroid
            folium.CircleMarker([c.y, c.x], radius=8, color="#2c3e50",
                                fill=True, fill_color=color, fill_opacity=1.0,
                                tooltip=f"PCI={row.get('PCI', ''):.1f} ({group_label})").add_to(fg)
        fg.add_to(m)

    _add_isochrone_legend(m, "PCI")
    LayerControl(collapsed=False).add_to(m)
    return m


def make_bci_isochrone_map(
    grid: HexGrid,
    iso_gdf: gpd.GeoDataFrame,
    bci_counts_df: pd.DataFrame,
    bci_origins: dict,
    bci: pd.Series,
) -> folium.Map:
    """Interactive folium map for BCI isochrones."""
    from branca.colormap import LinearColormap

    center = ((grid.gdf.total_bounds[1] + grid.gdf.total_bounds[3]) / 2,
              (grid.gdf.total_bounds[0] + grid.gdf.total_bounds[2]) / 2)
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    valid = bci.dropna()
    cmap  = LinearColormap(
        ["#fff5eb", "#fdd0a2", "#fdae6b", "#fd8d3c", "#e6550d", "#a63603"],
        vmin=0, vmax=100, caption="BCI"
    )
    gdf = grid.gdf.copy()
    gdf["BCI"] = gdf["hex_id"].map(bci)
    base_fg = folium.FeatureGroup(name="🗺 BCI Grid", show=True)
    for _, row in gdf.iterrows():
        val   = row.get("BCI")
        color = cmap(val) if pd.notna(val) else "#ccc"
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda f, c=color: {"fillColor": c, "fillOpacity": 0.5, "weight": 0.2},
        ).add_to(base_fg)
    base_fg.add_to(m)
    cmap.add_to(m)

    bci_iso = iso_gdf[iso_gdf.get("index", iso_gdf.columns[0]) == "BCI"] if len(iso_gdf) else iso_gdf
    _add_isochrone_layers(m, bci_iso, bci_counts_df, "BCI")

    for group_label, color in [("top", "#27ae60"), ("bottom", "#8e44ad")]:
        if group_label not in bci_origins:
            continue
        fg = folium.FeatureGroup(name=f"📍 {'Top' if group_label=='top' else 'Bottom'} 10% Origins (BCI)")
        for _, row in bci_origins[group_label].to_crs("EPSG:4326").iterrows():
            c = row.geometry.centroid
            folium.CircleMarker([c.y, c.x], radius=8, color="#2c3e50",
                                fill=True, fill_color=color, fill_opacity=1.0,
                                tooltip=f"BCI={row.get('BCI', ''):.2f} ({group_label})").add_to(fg)
        fg.add_to(m)

    _add_isochrone_legend(m, "BCI")
    LayerControl(collapsed=False).add_to(m)
    return m


def _add_isochrone_layers(m, iso_gdf, counts_df, index_col):
    if iso_gdf is None or len(iso_gdf) == 0:
        return
    counts_idx = counts_df.set_index(["hex_id", "threshold"]) if not counts_df.empty else pd.DataFrame()

    for group in ["top", "bottom"]:
        for mode, mconf in ISO_MODE_CONFIG.items():
            name = f"{'🟢 Top' if group=='top' else '🔴 Bottom'} 10% | {mode.title()}"
            show = (group == "top" and mode == "walk")
            fg   = folium.FeatureGroup(name=name, show=show)

            sub = iso_gdf[(iso_gdf["group"] == group) & (iso_gdf["mode"] == mode)]
            for thr in sorted(ISO_THRESHOLDS, reverse=True):
                t_sub = sub[sub["threshold"] == thr]
                for _, row in t_sub.iterrows():
                    if row["geometry"] is None:
                        continue
                    try:
                        c = counts_idx.loc[(row["hex_id"], thr)]
                        if index_col == "PCI":
                            tip = (f"<b>{group.upper()} {index_col} | {mode} {thr}min</b><br>"
                                   f"Score: {row['score']:.1f}<br>"
                                   f"🏥 Health: {c.get('health', 0):.0f}<br>"
                                   f"🎓 Education: {c.get('education', 0):.0f}<br>"
                                   f"🌳 Parks: {c.get('parks', 0):.0f}<br>"
                                   f"🤝 Community: {c.get('community', 0):.0f}<br>"
                                   f"🍽 Food: {c.get('food_retail', 0):.0f}<br>"
                                   f"🚌 Transit: {c.get('transit', 0):.0f}<br>"
                                   f"<b>Total: {c.get('total_amenities', 0):.0f}</b>")
                        else:
                            tip = (f"<b>{group.upper()} {index_col} | {mode} {thr}min</b><br>"
                                   f"Score: {row['score']:.2f}<br>"
                                   f"👥 Pop: {c.get('population', 0):,.0f}<br>"
                                   f"🏢 Businesses: {c.get('businesses', 0):.1f}<br>"
                                   f"💰 Demand: ${c.get('market_demand_B', 0):.2f}B")
                    except Exception:
                        tip = f"{group} | {mode} | {thr}min"

                    folium.GeoJson(
                        row["geometry"].__geo_interface__,
                        style_function=lambda f, c=mconf["color"], op=ISO_OPACITY[thr]: {
                            "fillColor": c, "fillOpacity": op,
                            "color": c, "weight": 1.2 if op > 0.4 else 0.6,
                        },
                        tooltip=folium.Tooltip(tip, sticky=True),
                    ).add_to(fg)
            fg.add_to(m)


def _add_isochrone_legend(m, index_col):
    html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
                padding:12px 16px;border-radius:8px;box-shadow:2px 2px 6px rgba(0,0,0,.3);
                font-size:13px;line-height:1.8;">
    <b>{index_col} Isochrones</b><br>
    <span style="color:#2ecc71">■</span> Walk &nbsp;
    <span style="color:#3498db">■</span> Bike &nbsp;
    <span style="color:#e74c3c">■</span> Drive &nbsp;
    <span style="color:#9b59b6">■</span> Transit<br>
    <b>Opacity:</b> dark=10 min · mid=20 min · light=30 min
    </div>"""
    m.get_root().html.add_child(folium.Element(html))
