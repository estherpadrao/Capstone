"""
File 13: BCI Analysis & Visualisations
=======================================
Maps and charts for BCI. Mirrors the structure of pci_analysis.py
so the webapp can call both with a uniform interface.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from folium import GeoJsonTooltip
import branca.colormap as cm
from branca.colormap import LinearColormap
from typing import Dict, Optional, Tuple
import io, base64

from core.h3_helper import HexGrid
from core.network_builder import MultiModalNetworkBuilder
from bci.bci_calculator import BCICalculator, BCIHansenAccessibility


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _center(gdf: gpd.GeoDataFrame) -> Tuple[float, float]:
    b = gdf.total_bounds
    return (b[1] + b[3]) / 2, (b[0] + b[2]) / 2


# ---------------------------------------------------------------------------
# 1. Individual mass maps (matplotlib)
# ---------------------------------------------------------------------------

def plot_bci_masses(
    grid: HexGrid,
    city_name: str = "",
) -> str:
    """
    3-panel matplotlib: Market Mass | Labour Mass | Supplier Mass.
    Returns base64 PNG.
    """
    panels = [
        ("market_mass",   "Market Mass (P×Y)",          "YlOrRd"),
        ("labour_mass",   "Labour Mass (L)",             "YlGnBu"),
        ("supplier_mass", "Supplier Mass (S)",           "PuRd"),
    ]
    available = [(col, title, cmap) for col, title, cmap in panels if col in grid.gdf.columns]
    n = len(available)
    if n == 0:
        return ""

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (col, title, cmap) in zip(axes, available):
        grid.gdf.plot(
            column=col, ax=ax, cmap=cmap,
            edgecolor="gray", linewidth=0.15, legend=True,
            legend_kwds={"shrink": 0.55},
        )
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()

    plt.suptitle(f"{city_name} — BCI Mass Components", fontsize=13, y=1.02)
    plt.tight_layout()
    result = _fig_to_b64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# 2. Accessibility components (matplotlib, 6-panel)
# ---------------------------------------------------------------------------

def plot_bci_components(
    grid: HexGrid,
    city_name: str = "",
) -> str:
    """
    2×3 panel: accessibility (row 1) + masses + final BCI (row 2).
    Returns base64 PNG.
    """
    panels = [
        ("A_market",      "Market Accessibility\n(Access to Customers)",        "YlOrRd"),
        ("A_labour",      "Labour Accessibility\n(Access to Workers)",           "YlGnBu"),
        ("A_supplier",    "Supplier Accessibility\n(Access to Business Svcs)",   "PuRd"),
        ("market_mass",   "Market Mass (P×Y)",                                   "Oranges"),
        ("supplier_mass", "Supplier Mass (S)",                                   "Purples"),
        ("BCI",           "Final BCI (0–100)",                                   "RdYlGn"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for ax, (col, title, cmap) in zip(axes, panels):
        if col not in grid.gdf.columns:
            ax.set_visible(False)
            continue
        grid.gdf.plot(
            column=col, ax=ax, cmap=cmap,
            edgecolor="gray", linewidth=0.15, legend=True,
            missing_kwds={"color": "lightgray"},
            legend_kwds={"shrink": 0.5},
        )
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()

    plt.suptitle(f"{city_name} — BCI Full Component Analysis", fontsize=13, y=1.01)
    plt.tight_layout()
    result = _fig_to_b64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# 3. Interactive maps per mass component
# ---------------------------------------------------------------------------

def make_market_map(grid: HexGrid, city_name: str = "") -> folium.Map:
    """Interactive folium map of Market Mass."""
    return _make_component_map(grid, "market_mass", "Market Mass (P×Y)", "YlOrRd", city_name)

def make_labour_map(grid: HexGrid, city_name: str = "") -> folium.Map:
    return _make_component_map(grid, "labour_mass", "Labour Mass (L)", "YlGnBu", city_name)

def make_supplier_map(grid: HexGrid, city_name: str = "") -> folium.Map:
    return _make_component_map(grid, "supplier_mass", "Supplier Mass (S)", "PuRd", city_name)


def _make_component_map(
    grid: HexGrid,
    column: str,
    caption: str,
    colormap_name: str,
    city_name: str,
) -> folium.Map:
    center = _center(grid.gdf)
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    if column not in grid.gdf.columns:
        return m

    valid = grid.gdf[column].dropna()
    cmap  = getattr(cm.linear, f"{colormap_name}_09").scale(valid.min(), valid.max())
    cmap.caption = caption

    gdf = grid.gdf[["hex_id", column, "geometry"]].copy()

    def _style(feature, _cmap=cmap):
        val = feature["properties"].get(column)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return {"fillColor": "#ccc", "fillOpacity": 0.3, "weight": 0.2}
        return {"fillColor": _cmap(val), "fillOpacity": 0.7, "weight": 0.2, "color": "gray"}

    folium.GeoJson(
        gdf.to_json(),
        style_function=_style,
        tooltip=GeoJsonTooltip(["hex_id", column], aliases=["Hex:", caption + ":"]),
        name=caption,
    ).add_to(m)
    cmap.add_to(m)
    folium.LayerControl().add_to(m)
    return m


# ---------------------------------------------------------------------------
# 4. Final BCI interactive map
# ---------------------------------------------------------------------------

def make_bci_map(
    grid: HexGrid,
    bci: pd.Series,
    city_name: str = "",
) -> folium.Map:
    """Interactive folium choropleth of final BCI scores."""
    center = _center(grid.gdf)
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    cmap = LinearColormap(
        ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"],
        vmin=0, vmax=100, caption="BCI (0–100)",
    )

    gdf = grid.gdf.copy()
    gdf["BCI"] = gdf["hex_id"].map(bci)

    tooltip_fields  = ["hex_id", "BCI"]
    tooltip_aliases = ["Hex:", "BCI:"]
    for extra in ["market_mass", "labour_mass", "supplier_mass"]:
        if extra in gdf.columns:
            tooltip_fields.append(extra)
            tooltip_aliases.append(extra.replace("_", " ").title() + ":")

    def _style(feature):
        val = feature["properties"].get("BCI")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return {"fillColor": "#cccccc", "fillOpacity": 0.3, "weight": 0.3}
        return {"fillColor": cmap(val), "fillOpacity": 0.7, "weight": 0.3, "color": "gray"}

    folium.GeoJson(
        gdf.to_json(),
        style_function=_style,
        tooltip=GeoJsonTooltip(tooltip_fields, aliases=tooltip_aliases),
        name="BCI",
    ).add_to(m)
    cmap.add_to(m)
    folium.LayerControl().add_to(m)
    return m


# ---------------------------------------------------------------------------
# 5. Distribution charts
# ---------------------------------------------------------------------------

def plot_bci_distribution(bci: pd.Series, city_name: str = "") -> str:
    """Histogram + component correlation bar chart. Returns base64 PNG."""
    valid = bci.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(valid, bins=30, color="#e74c3c", edgecolor="white", alpha=0.85)
    axes[0].axvline(valid.mean(),   color="navy",  ls="--", lw=2, label=f"Mean: {valid.mean():.1f}")
    axes[0].axvline(valid.median(), color="gold",  ls="--", lw=2, label=f"Median: {valid.median():.1f}")
    axes[0].set_xlabel("BCI Score")
    axes[0].set_ylabel("Hexagons")
    axes[0].set_title("BCI Distribution")
    axes[0].legend()

    sorted_v = np.sort(valid)
    cum = np.arange(1, len(sorted_v) + 1) / len(sorted_v) * 100
    axes[1].plot(sorted_v, cum, color="#e74c3c", lw=2)
    axes[1].fill_between(sorted_v, cum, alpha=0.25, color="#e74c3c")
    axes[1].set_xlabel("BCI Score")
    axes[1].set_ylabel("Cumulative %")
    axes[1].set_title("Cumulative Distribution")

    plt.suptitle(f"{city_name} — BCI Distribution", fontsize=13, y=1.02)
    plt.tight_layout()
    result = _fig_to_b64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# 6. Descriptive statistics
# ---------------------------------------------------------------------------

def compute_bci_stats(bci: pd.Series, grid: HexGrid, bci_calc: BCICalculator) -> dict:
    """Return stats dict for the webapp BCI summary panel."""
    valid = bci.dropna()
    if len(valid) == 0:
        return {}

    summary = bci_calc.summary()

    return {
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
        "n_hexagons":      int(len(valid)),
        "cv_pct":          round(float(valid.std() / valid.mean() * 100), 2) if valid.mean() else 0.0,
        **{k: v for k, v in summary.items() if k not in {"mean", "median", "std", "min", "max"}},
    }
