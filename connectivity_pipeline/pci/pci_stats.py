"""
File 8c: PCI Statistics
========================
Descriptive statistics for PCI scores.

Extracted from pci_analysis.py §7.
"""

import numpy as np
import pandas as pd

from core.h3_helper import HexGrid


# ---------------------------------------------------------------------------
# 7. Descriptive statistics
# ---------------------------------------------------------------------------

def compute_pci_stats(pci: pd.Series, grid: HexGrid) -> dict:
    """Return stats dict for the webapp PCI summary panel."""
    valid = pci.dropna()
    if len(valid) == 0:
        return {}

    def _gini(x: pd.Series) -> float:
        arr = np.sort(x.dropna().values)
        n = len(arr)
        if n == 0 or arr.sum() == 0:
            return 0.0
        idx = np.arange(1, n + 1)
        return float((2 * np.dot(idx, arr) / (n * arr.sum())) - (n + 1) / n)

    stats = {
        "mean":          round(float(valid.mean()), 2),
        "median":        round(float(valid.median()), 2),
        "std":           round(float(valid.std()), 2),
        "min":           round(float(valid.min()), 2),
        "max":           round(float(valid.max()), 2),
        "p25":           round(float(valid.quantile(0.25)), 2),
        "p75":           round(float(valid.quantile(0.75)), 2),
        "iqr":           round(float(valid.quantile(0.75) - valid.quantile(0.25)), 2),
        "skewness":      round(float(valid.skew()), 3),
        "kurtosis":      round(float(valid.kurtosis()), 3),
        "gini":          round(_gini(valid), 3),
        "n_hexagons":    int(len(valid)),
        "cv_pct":        round(float(valid.std() / valid.mean() * 100), 2) if valid.mean() else 0.0,
        "n_hotspots":    int((valid >= valid.quantile(0.90)).sum()),
        "n_underserved": int((valid <= valid.quantile(0.10)).sum()),
    }

    # Income–PCI correlation (shows equity dimension)
    if "median_income" in grid.gdf.columns:
        inc = grid.gdf.set_index("hex_id")["median_income"].reindex(valid.index).dropna()
        if len(inc) > 2:
            stats["corr_income_pci"] = round(float(valid.reindex(inc.index).corr(inc)), 3)

    if "area_m2" in grid.gdf.columns:
        gdf_v = grid.gdf[grid.gdf["hex_id"].isin(valid.index)].copy()
        gdf_v["_pci"] = gdf_v["hex_id"].map(valid)
        total_area = gdf_v["area_m2"].sum()
        if total_area > 0:
            stats["city_pci"] = round(
                float((gdf_v["_pci"] * gdf_v["area_m2"]).sum() / total_area), 2
            )
            stats["area_km2"] = round(float(total_area / 1e6), 2)

    return stats
