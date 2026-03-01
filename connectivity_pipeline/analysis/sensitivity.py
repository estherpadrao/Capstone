"""
analysis/sensitivity.py
=======================
One-at-a-time (OAT) sensitivity analysis for PCI and BCI parameters.

Baseline values are read from the session (whatever was used to compute scores).
Travel times are reused from cache — only mass/accessibility/score reruns.

Outputs: tornado chart (PNG b64) + stats table (HTML string)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, base64
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from core.mass_calculator import MassCalculator
from pci.pci_calculator import TopographicPCICalculator
from bci.bci_calculator import BCICalculator, DEFAULT_BETA_PARAMS
from bci.bci_masses import BCIMassCalculator


# ---------------------------------------------------------------------------
# Parameter ranges — (low, high). Baseline read from session at runtime.
# ---------------------------------------------------------------------------

PCI_PARAM_RANGES = {
    "beta":        (0.04, 0.14),
    "lambda":      (0.00, 0.60),
    "w_health":    (0.00, 0.50),
    "w_education": (0.00, 0.50),
    "w_parks":     (0.00, 0.50),
    "w_community": (0.00, 0.50),
    "w_food":      (0.00, 0.50),
    "w_transit":   (0.00, 0.50),
}

BCI_PARAM_RANGES = {
    "beta_market":   (0.06, 0.20),
    "beta_labour":   (0.02, 0.10),
    "beta_supplier": (0.05, 0.18),
    "urban_lambda":  (0.00, 0.50),
}

PARAM_LABELS = {
    "beta":           "beta (decay)",
    "lambda":         "lambda (active street)",
    "w_health":       "weight: health",
    "w_education":    "weight: education",
    "w_parks":        "weight: parks",
    "w_community":    "weight: community",
    "w_food":         "weight: food",
    "w_transit":      "weight: transit",
    "beta_market":    "beta market",
    "beta_labour":    "beta labour",
    "beta_supplier":  "beta supplier",
    "urban_lambda":   "lambda urban interface",
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SensResult:
    param:        str
    base_val:     float
    low_val:      float
    high_val:     float
    mean_base:    float;  mean_low:    float;  mean_high:    float
    median_base:  float;  median_low:  float;  median_high:  float
    p10_base:     float;  p10_low:     float;  p10_high:     float
    p90_base:     float;  p90_low:     float;  p90_high:     float

    @property
    def swing(self) -> float:
        return self.mean_high - self.mean_low


def _stats(s: pd.Series) -> dict:
    v = s.dropna()
    return dict(mean=float(v.mean()), median=float(v.median()),
                p10=float(v.quantile(0.10)), p90=float(v.quantile(0.90)))


# ---------------------------------------------------------------------------
# PCI helpers
# ---------------------------------------------------------------------------

def _run_pci(s: dict, beta: float, lam: float, weights: dict) -> pd.Series:
    grid = s["grid"]
    ham  = s["ham"]
    up   = s["user_params"]

    mass_calc = MassCalculator(
        grid,
        amenity_weights=weights,
        decay_coefficients=up.get("decay_coefficients", {}),
    )
    for name, gdf in s["amenities"].items():
        mass_calc.add_amenity_layer(name, gdf, use_area=(name == "parks"))
    mass_calc.compute_composite_mass()
    ham.mass = mass_calc

    ham.compute_accessibility(
        beta=beta,
        income_data=s.get("income_by_hex"),
        mode_cost=s.get("avg_mode_cost", 3.94),
    )

    pci_calc = TopographicPCICalculator(grid, ham, mass_calc)
    return pci_calc.compute_pci(
        active_lambda=lam,
        mask_parks=up.get("mask_parks", False),
    )


def _pci_variant(param, val, base_beta, base_lam, base_w):
    beta    = val if param == "beta"   else base_beta
    lam     = val if param == "lambda" else base_lam
    weights = dict(base_w)
    if param.startswith("w_"):
        key = param[2:]
        weights[key] = val
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
    return beta, lam, weights


def run_pci_sensitivity(s: dict) -> Tuple[List[SensResult], str, str]:
    print("\n  PCI SENSITIVITY ANALYSIS")
    up       = s["user_params"]
    base_beta = up["hansen_beta"]
    base_lam  = up["active_street_lambda"]
    base_w    = dict(up["amenity_weights"])

    st_base = _stats(s["pci"])
    results: List[SensResult] = []

    for param, (lo, hi) in PCI_PARAM_RANGES.items():
        if param == "beta":        bv = base_beta
        elif param == "lambda":    bv = base_lam
        else:                      bv = base_w.get(param[2:], 0.0)
        print(f"  {PARAM_LABELS[param]}: {lo} / {bv:.3f} / {hi}")

        st_lo = _stats(_run_pci(s, *_pci_variant(param, lo, base_beta, base_lam, base_w)))
        st_hi = _stats(_run_pci(s, *_pci_variant(param, hi, base_beta, base_lam, base_w)))

        results.append(SensResult(
            param=param, base_val=bv, low_val=lo, high_val=hi,
            mean_base=st_base["mean"],     mean_low=st_lo["mean"],     mean_high=st_hi["mean"],
            median_base=st_base["median"], median_low=st_lo["median"], median_high=st_hi["median"],
            p10_base=st_base["p10"],       p10_low=st_lo["p10"],       p10_high=st_hi["p10"],
            p90_base=st_base["p90"],       p90_low=st_lo["p90"],       p90_high=st_hi["p90"],
        ))

    tornado = _plot_tornado(results, "PCI")
    table   = _stats_table_html(results)
    print("  Done.")
    return results, tornado, table


# ---------------------------------------------------------------------------
# BCI helpers
# ---------------------------------------------------------------------------

def _run_bci(s: dict, bm: float, bl: float, bs: float, ul: float) -> pd.Series:
    grid       = s["grid"]
    bci_hansen = s["bci_hansen"]
    mass_calc  = s["mass_calc_bci"]
    up         = s.get("user_params", {})

    bci_hansen.beta["market"]   = bm
    bci_hansen.beta["labour"]   = bl
    bci_hansen.beta["supplier"] = bs

    bci_hansen.compute_all_accessibility(
        market_mass=mass_calc.market_mass,
        labour_mass=mass_calc.labour_mass,
        supplier_mass=mass_calc.supplier_mass,
    )

    bci_calc = BCICalculator(grid, bci_hansen, mass_calc)
    return bci_calc.compute_bci(
        method=up.get("bci_method", "weight_free"),
        market_weight=up.get("market_weight", 0.40),
        labour_weight=up.get("labour_weight", 0.35),
        supplier_weight=up.get("supplier_weight", 0.25),
        use_interface=up.get("use_urban_interface", False),
        interface_lambda=ul,
    )


def run_bci_sensitivity(s: dict) -> Tuple[List[SensResult], str, str]:
    print("\n  BCI SENSITIVITY ANALYSIS")
    bci_hansen = s["bci_hansen"]
    base_bm = bci_hansen.beta.get("market",   DEFAULT_BETA_PARAMS["market"])
    base_bl = bci_hansen.beta.get("labour",   DEFAULT_BETA_PARAMS["labour"])
    base_bs = bci_hansen.beta.get("supplier", DEFAULT_BETA_PARAMS["supplier"])
    base_ul = s.get("urban_lambda", 0.25)

    st_base = _stats(s["bci"])
    bv_map  = {"beta_market": base_bm, "beta_labour": base_bl,
               "beta_supplier": base_bs, "urban_lambda": base_ul}

    results: List[SensResult] = []

    for param, (lo, hi) in BCI_PARAM_RANGES.items():
        bv = bv_map[param]
        print(f"  {PARAM_LABELS[param]}: {lo} / {bv:.3f} / {hi}")

        def _kw(val):
            return dict(
                bm=val if param == "beta_market"   else base_bm,
                bl=val if param == "beta_labour"   else base_bl,
                bs=val if param == "beta_supplier" else base_bs,
                ul=val if param == "urban_lambda"  else base_ul,
            )

        st_lo = _stats(_run_bci(s, **_kw(lo)))
        st_hi = _stats(_run_bci(s, **_kw(hi)))

        results.append(SensResult(
            param=param, base_val=bv, low_val=lo, high_val=hi,
            mean_base=st_base["mean"],     mean_low=st_lo["mean"],     mean_high=st_hi["mean"],
            median_base=st_base["median"], median_low=st_lo["median"], median_high=st_hi["median"],
            p10_base=st_base["p10"],       p10_low=st_lo["p10"],       p10_high=st_hi["p10"],
            p90_base=st_base["p90"],       p90_low=st_lo["p90"],       p90_high=st_hi["p90"],
        ))

    tornado = _plot_tornado(results, "BCI")
    table   = _stats_table_html(results)
    print("  Done.")
    return results, tornado, table


# ---------------------------------------------------------------------------
# Tornado chart
# ---------------------------------------------------------------------------

def _plot_tornado(results: List[SensResult], title: str) -> str:
    rs   = sorted(results, key=lambda r: abs(r.swing), reverse=True)
    n    = len(rs)
    base = rs[0].mean_base

    fig, ax = plt.subplots(figsize=(11, max(5, n * 1.1 + 1.5)))

    # Determine axis range so all bars are visible even if tiny
    all_deltas = [abs(r.mean_low - base) for r in rs] + [abs(r.mean_high - base) for r in rs]
    max_delta  = max(all_deltas) if all_deltas else 1.0
    min_bar    = max_delta * 0.04   # minimum visible bar width (4% of range)

    for i, r in enumerate(rs):
        lo_d = r.mean_low  - base
        hi_d = r.mean_high - base
        lo_c = "#1a9850" if lo_d >= 0 else "#d73027"
        hi_c = "#1a9850" if hi_d >= 0 else "#d73027"

        # Ensure minimum visible width, preserving sign
        lo_draw = lo_d if abs(lo_d) >= min_bar else (min_bar if lo_d >= 0 else -min_bar)
        hi_draw = hi_d if abs(hi_d) >= min_bar else (min_bar if hi_d >= 0 else -min_bar)

        ax.barh(i, lo_draw, height=0.5, color=lo_c, alpha=0.85)
        ax.barh(i, hi_draw, height=0.5, color=hi_c, alpha=0.85)

        # Annotate: put label outside the bar, nudged away from zero line
        # Use axis fraction offset so labels never overlap regardless of scale
        off = max_delta * 0.05
        lo_tip = lo_draw - off if lo_draw <= 0 else lo_draw + off
        hi_tip = hi_draw - off if hi_draw <= 0 else hi_draw + off
        ax.text(lo_tip, i + 0.18, f"{lo_d:+.1f}  ({r.low_val})",
                va="center", ha="right" if lo_draw <= 0 else "left",
                fontsize=7.5, color=lo_c)
        ax.text(hi_tip, i - 0.18, f"{hi_d:+.1f}  ({r.high_val})",
                va="center", ha="right" if hi_draw <= 0 else "left",
                fontsize=7.5, color=hi_c)

    ax.set_yticks(range(n))
    ax.set_yticklabels([PARAM_LABELS.get(r.param, r.param) for r in rs], fontsize=10)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel(f"Change in mean score from baseline ({base:.1f})", fontsize=10)
    ax.set_title(f"{title} Sensitivity — Tornado Chart\nBaseline mean: {base:.2f}",
                 fontsize=12, fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(color="#1a9850", label="Score increases"),
        mpatches.Patch(color="#d73027", label="Score decreases"),
    ], loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Stats table (HTML) — shows delta from baseline, colour-coded
# ---------------------------------------------------------------------------

def _stats_table_html(results: List[SensResult]) -> str:
    rs = sorted(results, key=lambda r: abs(r.swing), reverse=True)

    def delta_cell(val, base):
        d = val - base
        if abs(d) < 0.05:
            return f'<td style="padding:5px 10px;text-align:right;color:var(--muted)">—</td>'
        col  = "#1a9850" if d > 0 else "#d73027"
        sign = "+" if d > 0 else ""
        return f'<td style="padding:5px 10px;text-align:right;color:{col};font-weight:500">{sign}{d:.1f}</td>'

    th = 'style="padding:6px 10px;text-align:right;white-space:nowrap;border-bottom:2px solid var(--border)"'
    th_l = 'style="padding:6px 10px;text-align:left;border-bottom:2px solid var(--border)"'

    html = f'<table style="width:100%;border-collapse:collapse;font-size:.82rem">'
    html += f'<thead><tr>'
    html += f'<th {th_l}>Parameter</th>'
    html += f'<th {th}>Baseline</th>'
    html += f'<th {th} colspan="4">At low value → score change</th>'
    html += f'<th {th} colspan="4">At high value → score change</th>'
    html += f'<th {th}>Swing</th>'
    html += f'</tr>'

    # Sub-header
    sub = 'style="padding:2px 10px;text-align:right;font-size:.72rem;color:var(--muted);border-bottom:1px solid var(--border)"'
    sub_l = 'style="padding:2px 10px;text-align:left;font-size:.72rem;color:var(--muted);border-bottom:1px solid var(--border)"'
    html += f'<tr><td {sub_l}></td><td {sub}></td>'
    for label in ["Mean", "Median", "P10", "P90", "Mean", "Median", "P10", "P90"]:
        html += f'<td {sub}>{label}</td>'
    html += f'<td {sub}></td></tr>'
    html += '</thead><tbody>'

    for r in rs:
        sw  = r.swing
        sw_col = "#1a9850" if sw > 0.5 else ("#d73027" if sw < -0.5 else "var(--muted)")
        sign   = "+" if sw > 0 else ""

        html += '<tr style="border-bottom:1px solid var(--border)">'
        html += f'<td style="padding:5px 10px;font-weight:500">{PARAM_LABELS.get(r.param, r.param)}</td>'
        html += f'<td style="padding:5px 10px;text-align:right;color:var(--muted)">{r.base_val} → <small>lo:{r.low_val} hi:{r.high_val}</small></td>'

        # Low deltas
        for val, base in [(r.mean_low, r.mean_base), (r.median_low, r.median_base),
                          (r.p10_low, r.p10_base),   (r.p90_low,  r.p90_base)]:
            html += delta_cell(val, base)

        # High deltas
        for val, base in [(r.mean_high, r.mean_base), (r.median_high, r.median_base),
                          (r.p10_high, r.p10_base),   (r.p90_high,  r.p90_base)]:
            html += delta_cell(val, base)

        html += f'<td style="padding:5px 10px;text-align:right;font-weight:700;color:{sw_col}">{sign}{sw:.1f}</td>'
        html += '</tr>'

    html += '</tbody></table>'
    return html


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")