"""
File 13c: BCI Statistics
=========================
Descriptive statistics for BCI scores.

Extracted from bci_analysis.py §7.
"""

import numpy as np
import pandas as pd

from core.h3_helper import HexGrid


# ---------------------------------------------------------------------------
# 7. Descriptive statistics
# ---------------------------------------------------------------------------

def compute_bci_stats(bci: pd.Series, grid: HexGrid, bci_calc=None) -> dict:
    """Return stats dict for the webapp BCI summary panel.

    bci_calc is optional: when results are restored from disk the calculator
    object is not available (it holds network references), so extended stats
    are computed directly from the grid columns that were attached during the
    original compute run.
    """
    valid = bci.dropna()
    if len(valid) == 0:
        return {}

    base = {
        "mean":       round(float(valid.mean()), 2),
        "median":     round(float(valid.median()), 2),
        "std":        round(float(valid.std()), 2),
        "min":        round(float(valid.min()), 2),
        "max":        round(float(valid.max()), 2),
        "p25":        round(float(valid.quantile(0.25)), 2),
        "p75":        round(float(valid.quantile(0.75)), 2),
        "iqr":        round(float(valid.quantile(0.75) - valid.quantile(0.25)), 2),
        "skewness":   round(float(valid.skew()), 3),
        "kurtosis":   round(float(valid.kurtosis()), 3),
        "n_hexagons": int(len(valid)),
        "cv_pct":     round(float(valid.std() / valid.mean() * 100), 2) if valid.mean() else 0.0,
    }

    if bci_calc is not None:
        # Full run: delegate to the calculator's own summary
        summary = bci_calc.summary()
    else:
        # Restore path: reconstruct extended stats from grid columns
        def _grid_col(name):
            if name in grid.gdf.columns:
                return grid.gdf.set_index("hex_id")[name].reindex(valid.index)
            return pd.Series(dtype=float)

        am  = _grid_col("A_market_norm")
        al  = _grid_col("A_labour_norm")
        as_ = _grid_col("A_supplier_norm")

        def _safe_corr(s):
            d = s.dropna()
            return round(float(valid.reindex(d.index).corr(d)), 3) if len(d) > 2 else None

        summary = {
            "n_hotspots":        int((valid >= valid.quantile(0.90)).sum()),
            "n_underserved":     int((valid <= valid.quantile(0.10)).sum()),
            "corr_market_bci":   _safe_corr(am),
            "corr_labour_bci":   _safe_corr(al),
            "corr_supplier_bci": _safe_corr(as_),
        }

    base.update({k: v for k, v in summary.items()
                 if k not in {"mean", "median", "std", "min", "max", "n_hexagons"}})
    return base
