"""
File 13a: BCI Matplotlib Plots
================================
Static chart functions for BCI (mass panels, component panels,
3-D topography surfaces, distribution). All return base64 PNG strings.

Extracted from bci_analysis.py §1, §2, §5, §6.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.h3_helper import HexGrid
from analysis.shared import _fig_to_base64, _plot_score_distribution


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
        ("market_mass",   "Market Mass (P×Y)",  "YlOrRd"),
        ("labour_mass",   "Labour Mass (L)",     "YlGnBu"),
        ("supplier_mass", "Supplier Mass (S)",   "PuRd"),
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
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# 2. Accessibility components (matplotlib, 3-panel)
# ---------------------------------------------------------------------------

def plot_bci_components(
    grid: HexGrid,
    city_name: str = "",
) -> str:
    """
    3-panel matplotlib: Market Accessibility | Labour Accessibility | Supplier Accessibility.
    Each component is normalised to [0, 100] for display (minmax).
    Returns base64 PNG.
    """
    panels = [
        ("A_market",   "Market Accessibility\n(Access to Customers)",       "YlOrRd"),
        ("A_labour",   "Labour Accessibility\n(Access to Workers)",          "YlGnBu"),
        ("A_supplier", "Supplier Accessibility\n(Access to Business Svcs)", "PuRd"),
    ]

    available = [(col, title, cmap) for col, title, cmap in panels if col in grid.gdf.columns]
    if not available:
        return ""

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    viz_cols = []
    for ax, (col, title, cmap) in zip(axes, available):
        # Normalise to [0, 100] for display
        raw = grid.gdf[col].copy()
        mn, mx = raw.min(), raw.max()
        viz_col = f"_viz_{col}"
        grid.gdf[viz_col] = (raw - mn) / (mx - mn) * 100 if mx > mn else raw
        viz_cols.append(viz_col)

        grid.gdf.plot(
            column=viz_col, ax=ax, cmap=cmap,
            edgecolor="gray", linewidth=0.2, legend=True,
            legend_kwds={"shrink": 0.6, "label": "Score (0–100)"},
        )
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()

    # Drop temporary display columns without touching original accessibility values
    grid.gdf.drop(columns=viz_cols, errors="ignore", inplace=True)

    plt.suptitle(f"{city_name} — Hansen Accessibility: BCI Components", fontsize=13, y=1.02)
    plt.tight_layout()
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# 5. BCI Mass Topographies (3-D surface per mass component)
# ---------------------------------------------------------------------------

def plot_bci_topography(
    mass_calc,
    city_name: str = "",
) -> str:
    """
    3-D surface plots for Market Mass, Labour Mass, and Supplier Mass.
    Returns base64 PNG of a 1×3 figure (three side-by-side 3-D panels).
    """
    panels = [
        ("market",   "Market Mass\n(Customer Demand P×Y)",  "YlOrRd"),
        ("labour",   "Labour Mass\n(Worker Availability L)", "YlGnBu"),
        ("supplier", "Supplier Mass\n(Business Density S)",  "PuRd"),
    ]

    arrays = []
    labels = []
    cmaps  = []
    for mass_type, title, cmap in panels:
        try:
            arr, _ = mass_calc.get_topography_array(mass_type)
            arrays.append(np.clip(arr, 0, None))
            labels.append(title)
            cmaps.append(cmap)
        except Exception:
            continue

    if not arrays:
        return ""

    n   = len(arrays)
    fig = plt.figure(figsize=(12 * n, 8))

    for i, (arr, title, cmap) in enumerate(zip(arrays, labels, cmaps)):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")

        x = np.arange(arr.shape[1])
        y = np.arange(arr.shape[0])
        X, Y = np.meshgrid(x, y)

        surf = ax.plot_surface(
            X, Y, arr,
            cmap=cmap, alpha=0.8, linewidth=0, antialiased=True,
        )
        fig.colorbar(surf, ax=ax, shrink=0.5, label="Mass")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Mass")
        ax.set_zlim(0, arr.max() * 1.1 if arr.max() > 0 else 1)

    plt.suptitle(f"{city_name} — BCI Mass Topographies", fontsize=14, y=1.01)
    plt.tight_layout()
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# 6. Distribution charts
# ---------------------------------------------------------------------------

def plot_bci_distribution(bci: pd.Series, city_name: str = "") -> str:
    """Histogram + CDF of BCI scores. Returns base64 PNG."""
    return _plot_score_distribution(bci, city_name, "BCI", "#e74c3c")
