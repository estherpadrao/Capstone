"""
File 14: Comparative Analysis (PCI vs BCI)
===========================================
Shown only when both indices have been computed.
Produces scatter plots, distribution comparisons, spatial maps,
quadrant analysis, and correlation statistics.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import folium
from branca.colormap import LinearColormap
from typing import Dict, Optional, Tuple
import io, base64

from core.h3_helper import HexGrid


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
# Guard
# ---------------------------------------------------------------------------

def both_available(grid: HexGrid) -> bool:
    return "PCI" in grid.gdf.columns and "BCI" in grid.gdf.columns


# ---------------------------------------------------------------------------
# 1. Descriptive statistics comparison
# ---------------------------------------------------------------------------

def compute_comparative_stats(grid: HexGrid) -> dict:
    """Return a dict with all comparative statistics."""
    if not both_available(grid):
        return {"error": "Both PCI and BCI must be computed first."}

    pci = grid.gdf["PCI"].dropna()
    bci = grid.gdf["BCI"].dropna()

    # Align on same hexes
    valid = grid.gdf[["PCI", "BCI"]].dropna()

    pearson_r,  pearson_p  = stats.pearsonr(valid["PCI"],  valid["BCI"])
    spearman_r, spearman_p = stats.spearmanr(valid["PCI"], valid["BCI"])
    kendall_t,  kendall_p  = stats.kendalltau(valid["PCI"], valid["BCI"])

    pci_med = pci.median()
    bci_med = bci.median()

    quadrants = _compute_quadrants(grid)
    quad_counts = quadrants["quadrant"].value_counts().to_dict()

    cv_pci = float(pci.std() / pci.mean() * 100) if pci.mean() else 0.0
    cv_bci = float(bci.std() / bci.mean() * 100) if bci.mean() else 0.0

    return {
        "n_hexagons": int(len(valid)),
        "pci_mean":   round(float(pci.mean()), 2),
        "pci_median": round(float(pci.median()), 2),
        "pci_std":    round(float(pci.std()), 2),
        "pci_min":    round(float(pci.min()), 2),
        "pci_max":    round(float(pci.max()), 2),
        "pci_p25":    round(float(pci.quantile(0.25)), 2),
        "pci_p75":    round(float(pci.quantile(0.75)), 2),
        "pci_skew":   round(float(pci.skew()), 3),
        "pci_kurt":   round(float(pci.kurtosis()), 3),
        "pci_cv":     round(cv_pci, 2),
        "bci_mean":   round(float(bci.mean()), 2),
        "bci_median": round(float(bci.median()), 2),
        "bci_std":    round(float(bci.std()), 2),
        "bci_min":    round(float(bci.min()), 2),
        "bci_max":    round(float(bci.max()), 2),
        "bci_p25":    round(float(bci.quantile(0.25)), 2),
        "bci_p75":    round(float(bci.quantile(0.75)), 2),
        "bci_skew":   round(float(bci.skew()), 3),
        "bci_kurt":   round(float(bci.kurtosis()), 3),
        "bci_cv":     round(cv_bci, 2),
        "pearson_r":  round(float(pearson_r), 4),
        "pearson_p":  float(pearson_p),
        "pearson_r2": round(float(pearson_r ** 2), 4),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": float(spearman_p),
        "kendall_t":  round(float(kendall_t), 4),
        "kendall_p":  float(kendall_p),
        "pci_median_ref": round(float(pci_med), 2),
        "bci_median_ref": round(float(bci_med), 2),
        "quad_high_high": int(quad_counts.get("High PCI & High BCI", 0)),
        "quad_high_low":  int(quad_counts.get("High PCI, Low BCI", 0)),
        "quad_low_high":  int(quad_counts.get("Low PCI, High BCI", 0)),
        "quad_low_low":   int(quad_counts.get("Low PCI & Low BCI", 0)),
    }


def _compute_quadrants(grid: HexGrid) -> gpd.GeoDataFrame:
    gdf = grid.gdf.copy()
    pci_med = gdf["PCI"].median()
    bci_med = gdf["BCI"].median()
    conditions = [
        (gdf["PCI"] >= pci_med) & (gdf["BCI"] >= bci_med),
        (gdf["PCI"] >= pci_med) & (gdf["BCI"] <  bci_med),
        (gdf["PCI"] <  pci_med) & (gdf["BCI"] >= bci_med),
        (gdf["PCI"] <  pci_med) & (gdf["BCI"] <  bci_med),
    ]
    labels = [
        "High PCI & High BCI",
        "High PCI, Low BCI",
        "Low PCI, High BCI",
        "Low PCI & Low BCI",
    ]
    gdf["quadrant"] = "Other"
    for cond, label in zip(conditions, labels):
        gdf.loc[cond, "quadrant"] = label
    return gdf


# ---------------------------------------------------------------------------
# 2. Scatter + hexbin plot
# ---------------------------------------------------------------------------

def plot_scatter(grid: HexGrid, city_name: str = "") -> str:
    """PCI vs BCI scatter + density hexbin. Returns base64 PNG."""
    valid = grid.gdf[["PCI", "BCI"]].dropna()
    if len(valid) < 3:
        return ""

    pearson_r, _ = stats.pearsonr(valid["PCI"], valid["BCI"])
    pci_med = valid["PCI"].median()
    bci_med = valid["BCI"].median()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- Scatter ---
    ax = axes[0]
    ax.scatter(valid["PCI"], valid["BCI"], alpha=0.4, s=40,
               c=range(len(valid)), cmap="viridis", edgecolors="none")
    z = np.polyfit(valid["PCI"], valid["BCI"], 1)
    xline = np.linspace(valid["PCI"].min(), valid["PCI"].max(), 100)
    ax.plot(xline, np.poly1d(z)(xline), "r-", lw=2, label=f"Fit (r={pearson_r:.3f})")
    ax.plot([0, 100], [0, 100], "k--", lw=1, alpha=0.3, label="PCI = BCI")
    ax.axvline(pci_med, color="blue",  ls=":", alpha=0.5, label=f"PCI median ({pci_med:.1f})")
    ax.axhline(bci_med, color="green", ls=":", alpha=0.5, label=f"BCI median ({bci_med:.1f})")
    ax.set_xlabel("PCI", fontsize=12)
    ax.set_ylabel("BCI", fontsize=12)
    ax.set_title(f"PCI vs BCI Scatter  (r = {pearson_r:.3f})", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Hexbin density ---
    ax = axes[1]
    hb = ax.hexbin(valid["PCI"], valid["BCI"], gridsize=30, cmap="YlOrRd", mincnt=1)
    ax.plot(xline, np.poly1d(z)(xline), "b-", lw=2)
    ax.plot([0, 100], [0, 100], "k--", lw=1, alpha=0.3)
    ax.set_xlabel("PCI", fontsize=12)
    ax.set_ylabel("BCI", fontsize=12)
    ax.set_title("Density (Hexbin)", fontsize=13)
    plt.colorbar(hb, ax=ax, label="Point density")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"{city_name} — PCI vs BCI", fontsize=14, y=1.02)
    plt.tight_layout()
    result = _fig_to_b64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# 3. Distribution comparison
# ---------------------------------------------------------------------------

def plot_distribution_comparison(grid: HexGrid, city_name: str = "") -> str:
    """Histogram overlay, box plots, violin plots, QQ plot. Returns base64 PNG."""
    pci = grid.gdf["PCI"].dropna()
    bci = grid.gdf["BCI"].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Histogram overlay
    ax = axes[0, 0]
    ax.hist(pci, bins=50, alpha=0.6, label="PCI", color="steelblue", edgecolor="black", lw=0.3)
    ax.hist(bci, bins=50, alpha=0.6, label="BCI", color="tomato",    edgecolor="black", lw=0.3)
    ax.set_xlabel("Value"); ax.set_ylabel("Frequency")
    ax.set_title("Distribution Comparison: Histograms")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Box plots
    ax = axes[0, 1]
    ax.boxplot([pci, bci], labels=["PCI", "BCI"], patch_artist=True,
               boxprops=dict(facecolor="lightblue"),
               medianprops=dict(color="red", lw=2))
    ax.set_ylabel("Value")
    ax.set_title("Box Plots"); ax.grid(True, alpha=0.3, axis="y")

    # Violin plots
    ax = axes[1, 0]
    ax.violinplot([pci, bci], positions=[1, 2], showmeans=True, showmedians=True)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["PCI", "BCI"])
    ax.set_ylabel("Value")
    ax.set_title("Violin Plots"); ax.grid(True, alpha=0.3, axis="y")

    # QQ plot
    ax = axes[1, 1]
    n = min(len(pci), len(bci))
    pci_q = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(pci)), np.sort(pci))
    bci_q = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(bci)), np.sort(bci))
    ax.scatter(pci_q, bci_q, alpha=0.4, s=20)
    ax.plot([0, 100], [0, 100], "r--", lw=2, label="Perfect agreement")
    ax.set_xlabel("PCI quantiles"); ax.set_ylabel("BCI quantiles")
    ax.set_title("Q–Q Plot"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle(f"{city_name} — PCI vs BCI Distribution", fontsize=14, y=1.01)
    plt.tight_layout()
    result = _fig_to_b64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# 4. Spatial comparison maps (matplotlib)
# ---------------------------------------------------------------------------

def plot_spatial_comparison(grid: HexGrid, city_name: str = "") -> str:
    """4-panel: PCI, BCI, Difference, Quadrant. Returns base64 PNG."""
    quad_gdf = _compute_quadrants(grid)
    grid.gdf["pci_bci_diff"] = grid.gdf["PCI"] - grid.gdf["BCI"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    grid.gdf.plot(column="PCI", cmap="Blues",  ax=axes[0, 0], legend=True,
                  edgecolor="gray", linewidth=0.1,
                  legend_kwds={"label": "PCI", "shrink": 0.8})
    axes[0, 0].set_title("People Connectivity Index (PCI)", fontweight="bold")
    axes[0, 0].axis("off")

    grid.gdf.plot(column="BCI", cmap="Reds",   ax=axes[0, 1], legend=True,
                  edgecolor="gray", linewidth=0.1,
                  legend_kwds={"label": "BCI", "shrink": 0.8})
    axes[0, 1].set_title("Business Connectivity Index (BCI)", fontweight="bold")
    axes[0, 1].axis("off")

    grid.gdf.plot(column="pci_bci_diff", cmap="RdBu_r", ax=axes[1, 0], legend=True,
                  edgecolor="gray", linewidth=0.1,
                  legend_kwds={"label": "PCI − BCI", "shrink": 0.8})
    axes[1, 0].set_title("Difference Map (PCI − BCI)\nBlue = Residential | Red = Business",
                          fontweight="bold")
    axes[1, 0].axis("off")

    quad_colors = {
        "High PCI & High BCI": "green",
        "High PCI, Low BCI":   "steelblue",
        "Low PCI, High BCI":   "tomato",
        "Low PCI & Low BCI":   "lightgray",
    }
    for quad, color in quad_colors.items():
        sub = quad_gdf[quad_gdf["quadrant"] == quad]
        if len(sub):
            sub.plot(ax=axes[1, 1], color=color, edgecolor="black",
                     linewidth=0.1, label=quad, alpha=0.8)
    axes[1, 1].set_title("Quadrant Analysis", fontweight="bold")
    axes[1, 1].legend(loc="upper left", fontsize=8)
    axes[1, 1].axis("off")

    plt.suptitle(f"{city_name} — Spatial PCI vs BCI", fontsize=14, y=1.01)
    plt.tight_layout()
    result = _fig_to_b64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# 5. Interactive comparison map (folium)
# ---------------------------------------------------------------------------

def make_comparison_map(grid: HexGrid) -> folium.Map:
    """Interactive folium map with PCI, BCI, and Difference as toggleable layers."""
    center = _center(grid.gdf)
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    gdf = grid.gdf.copy()
    if "pci_bci_diff" not in gdf.columns:
        gdf["pci_bci_diff"] = gdf["PCI"] - gdf["BCI"]

    def _add_choropleth(col, colors, name, show=True):
        folium.Choropleth(
            geo_data=gdf.to_json(),
            data=gdf,
            columns=["hex_id", col],
            key_on="feature.properties.hex_id",
            fill_color=colors,
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=name,
            name=name,
            show=show,
        ).add_to(m)

    _add_choropleth("PCI",           "Blues",  "People Connectivity Index (PCI)", show=True)
    _add_choropleth("BCI",           "Reds",   "Business Connectivity Index (BCI)", show=False)
    _add_choropleth("pci_bci_diff",  "RdBu",   "Difference (PCI − BCI)", show=False)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
