"""
analysis/shared.py
==================
Shared utility functions used by both pci_analysis.py and bci_analysis.py.
Centralising here avoids duplication and ensures both modules use identical
implementations.
"""

import io
import base64
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Tuple, Optional, Dict, List
import folium


# ---------------------------------------------------------------------------
# Figure → base64
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to base64 PNG string for embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Folium map centre
# ---------------------------------------------------------------------------

def _folium_center(gdf: gpd.GeoDataFrame) -> Tuple[float, float]:
    """Return (lat, lng) centre of a GeoDataFrame's bounding box."""
    bounds = gdf.total_bounds
    return (bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2


# ---------------------------------------------------------------------------
# Score distribution plot (shared by PCI and BCI)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Neighborhood helpers (OSM fetch, colors, stats, folium overlay)
# ---------------------------------------------------------------------------

def fetch_osm_neighborhoods(
    boundary_polygon,
    min_area_m2: float = 50_000,
) -> gpd.GeoDataFrame:
    """
    Fetch named neighbourhood polygons from OpenStreetMap within *boundary_polygon*.

    Queries OSM tags ``place ∈ {neighbourhood, suburb, quarter}``.
    Only polygon geometries with a non-empty ``name`` field are kept.
    Tiny slivers (< *min_area_m2* m²) are filtered out.

    Returns a GeoDataFrame with columns ``name`` and ``geometry`` (EPSG:4326).
    Returns an empty GeoDataFrame on failure or when OSM has no data.
    """
    import osmnx as ox

    tags = {"place": ["neighbourhood", "suburb", "quarter"]}
    empty = gpd.GeoDataFrame(columns=["name", "geometry"], crs="EPSG:4326")

    try:
        # osmnx ≥ 1.2 uses features_from_polygon; older versions use geometries_from_polygon
        try:
            gdf = ox.features_from_polygon(boundary_polygon, tags)
        except AttributeError:
            gdf = ox.geometries_from_polygon(boundary_polygon, tags)

        if gdf is None or gdf.empty:
            print("   ⚠  No OSM neighbourhood features found — falling back to census tracts")
            return empty

        # Keep only polygon/multipolygon features
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        if gdf.empty or "name" not in gdf.columns:
            return empty

        gdf = gdf[["name", "geometry"]].dropna(subset=["name", "geometry"])
        gdf = gdf.to_crs("EPSG:4326")

        # Drop slivers
        area_series = gdf.to_crs(epsg=3857).geometry.area
        gdf = gdf[area_series.values >= min_area_m2].reset_index(drop=True)

        print(f"   🏘  {len(gdf)} OSM neighbourhood polygons fetched")
        return gdf

    except Exception as exc:
        print(f"   ⚠  OSM neighbourhood fetch failed ({exc}) — falling back to census tracts")
        return empty


def build_neighborhood_colors(names) -> Dict[str, str]:
    """
    Assign a visually distinct hex colour to each neighbourhood name.

    Uses matplotlib's tab20 / tab20b / tab20c cycle for up to 60 colours.
    ``"Unknown"`` always maps to ``#aaaaaa``.
    """
    sorted_names = sorted(
        {str(n) for n in names if n and str(n).strip() not in ("", "Unknown")}
    )
    cmap_names = ["tab20", "tab20b", "tab20c"]
    color_map: Dict[str, str] = {}
    for i, name in enumerate(sorted_names):
        cmap = plt.get_cmap(cmap_names[(i // 20) % 3])
        color_map[name] = mcolors.to_hex(cmap((i % 20) / 20.0))
    color_map["Unknown"] = "#aaaaaa"
    return color_map


def compute_neighborhood_stats(
    grid_gdf: gpd.GeoDataFrame,
    score_col: str,
    neighborhoods_gdf: gpd.GeoDataFrame,
    color_map: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """
    Compute average *score_col* per neighbourhood via spatial join.

    A hex contributes to **every** neighbourhood it touches, even partially,
    so averages reflect all hexagons with any overlap.

    Returns a list of dicts sorted by avg_score descending:
        ``{name, avg_score, hex_count, color}``
    """
    if neighborhoods_gdf is None or len(neighborhoods_gdf) == 0:
        return []
    if score_col not in grid_gdf.columns:
        return []

    hex_gdf = grid_gdf[["hex_id", score_col, "geometry"]].dropna(subset=[score_col]).copy()
    nb_gdf  = neighborhoods_gdf[["name", "geometry"]].copy()

    try:
        joined = gpd.sjoin(
            hex_gdf.to_crs(epsg=3857),
            nb_gdf.to_crs(epsg=3857),
            how="left",
            predicate="intersects",
        )
    except Exception as exc:
        print(f"   ⚠  Neighbourhood stats join failed: {exc}")
        return []

    joined = joined.dropna(subset=["name"])
    if joined.empty:
        return []

    stats = (
        joined.groupby("name")[score_col]
        .agg(avg_score="mean", hex_count="count")
        .round({"avg_score": 2})
        .reset_index()
        .sort_values("avg_score", ascending=False)
    )

    color_map = color_map or {}
    records = []
    for _, row in stats.iterrows():
        records.append({
            "name":      row["name"],
            "avg_score": round(float(row["avg_score"]), 2),
            "hex_count": int(row["hex_count"]),
            "color":     color_map.get(row["name"], "#aaaaaa"),
        })
    return records


def add_neighborhood_overlay(
    m: folium.Map,
    neighborhoods_gdf: gpd.GeoDataFrame,
    color_map: Dict[str, str],
    layer_name: str = "Neighborhoods",
) -> None:
    """
    Add a togglable neighbourhood boundary layer to *m*.

    Each neighbourhood polygon is filled with its assigned colour at low
    opacity, with a solid border.  A tooltip shows the neighbourhood name.
    The layer is registered with :class:`folium.LayerControl`.
    """
    if neighborhoods_gdf is None or neighborhoods_gdf.empty:
        return

    features = []
    for _, row in neighborhoods_gdf.iterrows():
        name  = str(row["name"])
        color = color_map.get(name, "#888888")
        try:
            features.append({
                "type": "Feature",
                "geometry":   row["geometry"].__geo_interface__,
                "properties": {"name": name, "color": color},
            })
        except Exception:
            continue

    if not features:
        return

    feature_collection = {"type": "FeatureCollection", "features": features}

    fg = folium.FeatureGroup(name=layer_name, show=True)
    folium.GeoJson(
        feature_collection,
        style_function=lambda f: {
            "fillColor":   f["properties"]["color"],
            "fillOpacity": 0.18,
            "color":       f["properties"]["color"],
            "weight":      2.0,
            "opacity":     0.85,
        },
        tooltip=folium.GeoJsonTooltip(["name"], aliases=["Neighborhood:"]),
    ).add_to(fg)
    fg.add_to(m)


def _plot_score_distribution(
    series: pd.Series,
    city_name: str,
    title_prefix: str,
    color: str,
) -> str:
    """
    Histogram + CDF for any 0–100 score series.
    Returns base64 PNG.

    Parameters
    ----------
    series       : score values
    city_name    : shown in suptitle
    title_prefix : e.g. "PCI" or "BCI"
    color        : matplotlib colour for bars / line
    """
    valid = series.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(valid, bins=30, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(valid.mean(),   color="red",    ls="--", lw=2,
               label=f"Mean: {valid.mean():.1f}")
    ax.axvline(valid.median(), color="orange", ls="--", lw=2,
               label=f"Median: {valid.median():.1f}")
    ax.set_xlabel(f"{title_prefix} Score")
    ax.set_ylabel("Hexagons")
    ax.set_title(f"{title_prefix} Distribution")
    ax.legend()

    ax = axes[1]
    sorted_v = np.sort(valid)
    cum = np.arange(1, len(sorted_v) + 1) / len(sorted_v) * 100
    ax.plot(sorted_v, cum, color=color, lw=2)
    ax.fill_between(sorted_v, cum, alpha=0.25, color=color)
    ax.axhline(50, color="gray", ls=":", alpha=0.5)
    ax.axvline(valid.median(), color="orange", ls="--", alpha=0.7)
    ax.set_xlabel(f"{title_prefix} Score")
    ax.set_ylabel("Cumulative %")
    ax.set_title("Cumulative Distribution")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    plt.suptitle(f"{city_name} — {title_prefix} Distribution", fontsize=13, y=1.02)
    plt.tight_layout()
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result
