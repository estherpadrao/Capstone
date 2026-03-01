"""
File 8a: PCI Matplotlib Plots
==============================
Static chart functions for PCI (topography layers, 3-D surface,
component panels, distribution). All return base64 PNG strings.

Extracted from pci_analysis.py §1, §2, §3, §6.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.h3_helper import HexGrid
from core.mass_calculator import MassCalculator
from analysis.shared import _fig_to_base64


# ---------------------------------------------------------------------------
# 1. Individual mass layer plots
# ---------------------------------------------------------------------------

def plot_topography_layers(
    grid: HexGrid,
    mass_calc: MassCalculator,
    city_name: str = "",
) -> str:
    """
    Multi-panel matplotlib figure — one panel per PCI mass layer
    (health_norm, education_norm, parks_norm, community_norm, etc.).
    Only renders layers whose amenity weight is non-zero.
    Returns base64 PNG.
    """
    layer_meta = [
        ("health_norm",      "Health Facilities",    "YlOrRd"),
        ("education_norm",   "Education Facilities", "YlGnBu"),
        ("parks_norm",       "Parks & Green Space",  "Greens"),
        ("community_norm",   "Community Amenities",  "PuBu"),
        ("food_retail_norm", "Food & Retail",        "OrRd"),
        ("transit_norm",     "Transit Access",       "Blues"),
    ]

    available = [
        (col, title, cmap)
        for col, title, cmap in layer_meta
        if col in grid.gdf.columns
        and (mass_calc.amenity_weights.get(col.replace("_norm", ""), 0.0) if mass_calc else 0.0) != 0.0
    ]

    if not available:
        return ""

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (col, title, cmap) in zip(axes, available):
        layer_name = col.replace("_norm", "")
        weight = mass_calc.amenity_weights.get(layer_name, 0.0) if mass_calc else 0.0
        subtitle = f"{title}\n(w={weight:.3f})" if weight else title

        grid.gdf.plot(
            column=col, ax=ax, cmap=cmap,
            edgecolor="gray", linewidth=0.15, legend=True,
            missing_kwds={"color": "lightgray"},
            legend_kwds={"shrink": 0.55},
        )
        ax.set_title(subtitle, fontsize=11)
        ax.set_axis_off()

    plt.suptitle(f"{city_name} — PCI Mass Layers", fontsize=13, y=1.02)
    plt.tight_layout()
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# 2. 3-D topography surface
# ---------------------------------------------------------------------------

def plot_topography_3d(
    grid: HexGrid,
    mass_calc: MassCalculator,
    city_name: str = "",
) -> str:
    """
    3-D surface plot of the composite mass topography.
    Returns base64 PNG.
    """
    if mass_calc._composite is None:
        mass_calc.compute_composite_mass()

    try:
        arr, _ = mass_calc.get_topography_array()
    except Exception:
        return ""

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection="3d")

    x = np.linspace(0, 1, arr.shape[1])
    y = np.linspace(0, 1, arr.shape[0])
    X, Y = np.meshgrid(x, y)

    surf = ax.plot_surface(
        X, Y, arr,
        cmap="terrain", edgecolor="none", alpha=0.9,
    )
    fig.colorbar(surf, shrink=0.5, aspect=10, label="Mass")
    ax.set_title(
        f"Attractiveness Topography: {city_name}\n"
        "(Hills = high opportunity, Valleys = low opportunity)",
        fontsize=14,
    )
    ax.set_xlabel("West ← → East", fontsize=12)
    ax.set_ylabel("South ← → North", fontsize=12)
    ax.set_zlabel("Attractiveness", fontsize=12)
    ax.set_zlim(0, arr.max() * 1.1)

    plt.tight_layout()
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# 3. PCI component multi-panel
# ---------------------------------------------------------------------------

def plot_pci_components(
    grid: HexGrid,
    city_name: str = "",
) -> str:
    """
    3-panel matplotlib: Topographic Mass | Hansen Accessibility | Final PCI.
    Returns base64 PNG.
    """
    panels = [
        ("mass",          "Topographic Mass\n(Weighted Amenity Surface)",    "terrain"),
        ("accessibility", "Hansen Accessibility\n(Network-Weighted Reach)",  "YlGnBu"),
        ("PCI",           "People Connectivity Index\n(Final Score 0–100)", "RdYlGn"),
    ]

    available = [(col, title, cmap) for col, title, cmap in panels if col in grid.gdf.columns]
    if not available:
        return ""

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (col, title, cmap) in zip(axes, available):
        grid.gdf.plot(
            column=col, ax=ax, cmap=cmap,
            edgecolor="gray", linewidth=0.1, legend=True,
            missing_kwds={"color": "lightgray", "label": "Parks"},
            legend_kwds={"shrink": 0.6},
        )
        ax.set_title(title, fontsize=12)
        ax.set_axis_off()

    plt.suptitle(f"{city_name} — Topographic Hansen Accessibility Model", fontsize=13, y=1.02)
    plt.tight_layout()
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# 6. Distribution chart
# ---------------------------------------------------------------------------

def plot_pci_distribution(pci: pd.Series, city_name: str = "") -> str:
    """Histogram + CDF of PCI scores. Returns base64 PNG."""
    valid = pci.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(valid, bins=30, color="#3498db", edgecolor="white", alpha=0.85)
    ax.axvline(valid.mean(),   color="red",    ls="--", lw=2,
               label=f"Mean: {valid.mean():.1f}")
    ax.axvline(valid.median(), color="orange", ls="--", lw=2,
               label=f"Median: {valid.median():.1f}")
    ax.set_xlabel("PCI Score")
    ax.set_ylabel("Hexagons")
    ax.set_title("PCI Distribution")
    ax.legend()

    ax = axes[1]
    sorted_v = np.sort(valid)
    cum = np.arange(1, len(sorted_v) + 1) / len(sorted_v) * 100
    ax.plot(sorted_v, cum, color="#3498db", lw=2)
    ax.fill_between(sorted_v, cum, alpha=0.25, color="#3498db")
    ax.axhline(50, color="gray", ls=":", alpha=0.5)
    ax.axvline(valid.median(), color="orange", ls="--", alpha=0.7)
    ax.set_xlabel("PCI Score")
    ax.set_ylabel("Cumulative %")
    ax.set_title("Cumulative Distribution")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    plt.suptitle(f"{city_name} — PCI Distribution", fontsize=13, y=1.02)
    plt.tight_layout()
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result
