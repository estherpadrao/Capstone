"""
File 8: PCI Analysis & Visualisations
=======================================
Generates all maps, distribution charts, and diagnostic stats for PCI.
All functions return figures / folium maps so the webapp can embed them.
Re-renders only what changed based on which parameter was updated.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import folium
from folium import GeoJsonTooltip
from branca.colormap import LinearColormap
import branca.colormap as cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Optional, Tuple
import io, base64

from core.h3_helper import HexGrid
from core.mass_calculator import MassCalculator
from core.network_builder import MultiModalNetworkBuilder
from pci.pci_calculator import HansenAccessibilityModel, TopographicPCICalculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to base64 PNG for embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _folium_center(gdf: gpd.GeoDataFrame) -> Tuple[float, float]:
    bounds = gdf.total_bounds
    return (bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2


# ---------------------------------------------------------------------------
# 1. Topography visualisation
# ---------------------------------------------------------------------------

def plot_topography_layers(
    grid: HexGrid,
    mass_calc: MassCalculator,
    city_name: str = "",
) -> str:
    """
    Matplotlib grid: one panel per amenity layer + composite.
    Returns base64 PNG.
    """
    layer_names = list(mass_calc.layers.keys()) + ["composite"]
    n = len(layer_names)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4.5))
    axes = np.array(axes).flatten()

    for ax, name in zip(axes, layer_names):
        col = name + "_norm" if name != "composite" else "mass"
        if col not in grid.gdf.columns:
            ax.set_visible(False)
            continue
        grid.gdf.plot(
            column=col, ax=ax, cmap="YlOrRd",
            edgecolor="gray", linewidth=0.1,
            legend=True, legend_kwds={"shrink": 0.6},
        )
        ax.set_title(name.replace("_", " ").title(), fontsize=11)
        ax.set_axis_off()

    for ax in axes[len(layer_names):]:
        ax.set_visible(False)

    plt.suptitle(f"{city_name} — Attractiveness Topography", fontsize=14, y=1.01)
    plt.tight_layout()
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


def plot_topography_3d(
    grid: HexGrid,
    mass_calc: MassCalculator,
    city_name: str = "",
) -> str:
    """3-D surface plot of composite mass. Returns base64 PNG."""
    arr, _ = mass_calc.get_topography_array()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    X = np.arange(arr.shape[1])
    Y = np.arange(arr.shape[0])
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, arr, cmap="terrain", edgecolor="none", alpha=0.9)
    ax.set_title(f"{city_name} — Amenity Topography (3D)", fontsize=13)
    ax.set_xlabel("East →")
    ax.set_ylabel("North →")
    ax.set_zlabel("Mass")
    plt.tight_layout()
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# 2. Network + mass interactive map
# ---------------------------------------------------------------------------

def make_mass_network_map(
    grid: HexGrid,
    network_builder: MultiModalNetworkBuilder,
    city_name: str = "",
) -> folium.Map:
    """
    Interactive folium map showing composite mass choropleth
    with per-mode network edges as toggleable layers.
    """
    center = _folium_center(grid.gdf)
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    # Mass choropleth
    if "mass" in grid.gdf.columns:
        valid = grid.gdf["mass"].dropna()
        cmap  = cm.linear.YlOrRd_09.scale(valid.min(), valid.max())
        cmap.caption = "Composite Mass"

        gdf_json = grid.gdf[["hex_id", "mass", "geometry"]].copy()

        def _style(feature):
            val = feature["properties"].get("mass")
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return {"fillColor": "#ccc", "fillOpacity": 0.3, "weight": 0.3}
            return {"fillColor": cmap(val), "fillOpacity": 0.65, "weight": 0.2, "color": "gray"}

        folium.GeoJson(
            gdf_json.to_json(),
            style_function=_style,
            tooltip=GeoJsonTooltip(["hex_id", "mass"], aliases=["Hex:", "Mass:"]),
            name="Composite Mass",
        ).add_to(m)
        cmap.add_to(m)

    # Network edges per mode
    mode_colors = {"walk": "blue", "bike": "green", "drive": "orange", "transit": "purple"}
    for mode, G in network_builder.networks.items():
        fg = folium.FeatureGroup(name=f"{mode.title()} Network", show=False)
        edges = list(G.edges(data=True))[:8000]  # cap for performance
        for u, v, _ in edges:
            u_d = G.nodes.get(u, {})
            v_d = G.nodes.get(v, {})
            if "x" in u_d and "x" in v_d:
                folium.PolyLine(
                    [(u_d["y"], u_d["x"]), (v_d["y"], v_d["x"])],
                    color=mode_colors.get(mode, "gray"),
                    weight=0.8, opacity=0.5,
                ).add_to(fg)
        fg.add_to(m)

    # Transit stops
    if network_builder.transit_stops_gdf is not None:
        fg = folium.FeatureGroup(name="Transit Stops", show=False)
        for _, stop in network_builder.transit_stops_gdf.head(500).iterrows():
            folium.CircleMarker(
                [stop.geometry.y, stop.geometry.x],
                radius=3, color="purple", fill=True, fill_opacity=0.7,
            ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ---------------------------------------------------------------------------
# 3. Final PCI interactive map
# ---------------------------------------------------------------------------

def make_pci_map(
    grid: HexGrid,
    pci: pd.Series,
    network_builder: Optional[MultiModalNetworkBuilder] = None,
    city_name: str = "",
) -> folium.Map:
    """Interactive folium choropleth of PCI scores."""
    center = _folium_center(grid.gdf)
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    cmap = LinearColormap(
        ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"],
        vmin=0, vmax=100,
        caption="PCI (0–100)",
    )

    gdf = grid.gdf.copy()
    gdf["PCI"] = gdf["hex_id"].map(pci)

    tooltip_fields = ["hex_id", "PCI"]
    tooltip_aliases = ["Hex:", "PCI:"]
    if "median_income" in gdf.columns:
        tooltip_fields.append("median_income")
        tooltip_aliases.append("Income ($):")

    def _style(feature):
        val = feature["properties"].get("PCI")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return {"fillColor": "#cccccc", "fillOpacity": 0.3, "weight": 0.3}
        return {"fillColor": cmap(val), "fillOpacity": 0.7, "weight": 0.3, "color": "gray"}

    folium.GeoJson(
        gdf.to_json(),
        style_function=_style,
        tooltip=GeoJsonTooltip(tooltip_fields, aliases=tooltip_aliases),
        name="PCI",
    ).add_to(m)
    cmap.add_to(m)

    if network_builder and network_builder.transit_stops_gdf is not None:
        fg = folium.FeatureGroup(name="Transit Stops", show=False)
        for _, stop in network_builder.transit_stops_gdf.head(500).iterrows():
            folium.CircleMarker(
                [stop.geometry.y, stop.geometry.x],
                radius=2, color="blue", fill=True,
            ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl().add_to(m)
    return m


# ---------------------------------------------------------------------------
# 4. Distribution charts
# ---------------------------------------------------------------------------

def plot_pci_distribution(
    pci: pd.Series,
    city_name: str = "",
) -> str:
    """Histogram + CDF of PCI scores. Returns base64 PNG."""
    valid = pci.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(valid, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(valid.mean(),   color="red",    ls="--", lw=2, label=f"Mean: {valid.mean():.1f}")
    ax.axvline(valid.median(), color="orange", ls="--", lw=2, label=f"Median: {valid.median():.1f}")
    ax.set_xlabel("PCI Score");  ax.set_ylabel("Hexagons")
    ax.set_title("PCI Distribution")
    ax.legend()

    # CDF
    ax = axes[1]
    sorted_v = np.sort(valid)
    cum = np.arange(1, len(sorted_v) + 1) / len(sorted_v) * 100
    ax.plot(sorted_v, cum, color="steelblue", lw=2)
    ax.fill_between(sorted_v, cum, alpha=0.25)
    ax.axhline(50, color="gray", ls=":", alpha=0.5)
    ax.axvline(valid.median(), color="orange", ls="--", alpha=0.7)
    ax.set_xlabel("PCI Score");  ax.set_ylabel("Cumulative %")
    ax.set_title("Cumulative Distribution")
    ax.set_xlim(0, 100);  ax.set_ylim(0, 100)

    plt.suptitle(f"{city_name} — PCI Distribution", fontsize=13, y=1.02)
    plt.tight_layout()
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


def plot_pci_components(
    grid: HexGrid,
    city_name: str = "",
) -> str:
    """Side-by-side maps: Topographic Mass | Hansen Accessibility | Final PCI."""
    cols_needed = ["mass", "accessibility", "PCI"]
    available   = [c for c in cols_needed if c in grid.gdf.columns]
    n = len(available)
    if n == 0:
        return ""

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    cmaps    = {"mass": "terrain", "accessibility": "YlGnBu", "PCI": "RdYlGn"}
    titles   = {
        "mass":          "Topographic Mass\n(Weighted Amenity Surface)",
        "accessibility": "Hansen Accessibility\n(Network-Weighted Reach)",
        "PCI":           "People Connectivity Index\n(Final Score 0–100)",
    }

    for ax, col in zip(axes, available):
        grid.gdf.plot(
            column=col, ax=ax, cmap=cmaps[col],
            edgecolor="gray", linewidth=0.1, legend=True,
            missing_kwds={"color": "lightgray"},
            legend_kwds={"shrink": 0.6},
        )
        ax.set_title(titles[col], fontsize=11)
        ax.set_axis_off()

    plt.suptitle(f"{city_name} — Topographic Hansen Model", fontsize=13, y=1.02)
    plt.tight_layout()
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# 5. Descriptive statistics dict
# ---------------------------------------------------------------------------

def compute_pci_stats(pci: pd.Series, grid: HexGrid) -> dict:
    """Return all descriptive statistics for the webapp summary panel."""
    valid = pci.dropna()
    if len(valid) == 0:
        return {}

    # Gini
    arr = np.sort(valid.values)
    n   = len(arr)
    idx = np.arange(1, n + 1)
    gini = float((2 * np.dot(idx, arr) / (n * arr.sum())) - (n + 1) / n) if arr.sum() > 0 else 0.0

    # Area-weighted city PCI
    gdf_v = grid.gdf[grid.gdf["PCI"].notna()] if "PCI" in grid.gdf.columns else grid.gdf
    total_area = gdf_v["area_m2"].sum() if "area_m2" in gdf_v.columns else 1
    city_pci = float(
        (gdf_v["PCI"] * gdf_v["area_m2"]).sum() / total_area
    ) if "PCI" in gdf_v.columns and total_area > 0 else float(valid.mean())

    return {
        "city_pci":        round(city_pci, 2),
        "mean":            round(float(valid.mean()), 2),
        "median":          round(float(valid.median()), 2),
        "std":             round(float(valid.std()), 2),
        "min":             round(float(valid.min()), 2),
        "max":             round(float(valid.max()), 2),
        "p25":             round(float(valid.quantile(0.25)), 2),
        "p75":             round(float(valid.quantile(0.75)), 2),
        "iqr":             round(float(valid.quantile(0.75) - valid.quantile(0.25)), 2),
        "skewness":        round(float(valid.skew()), 3),
        "kurtosis":        round(float(valid.kurtosis()), 3),
        "gini":            round(gini, 3),
        "n_hexagons":      int(len(valid)),
        "cv_pct":          round(float(valid.std() / valid.mean() * 100), 2),
    }
