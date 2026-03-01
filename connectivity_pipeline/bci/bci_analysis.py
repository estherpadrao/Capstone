"""
File 13: BCI Analysis — Compatibility Shim
===========================================
This module previously contained all BCI visualisation and stats code.
It has been split into three focused modules:

  bci/bci_plots.py  — matplotlib charts (§1, §2, §5, §6)
  bci/bci_maps.py   — folium interactive maps (§3, §4)
  bci/bci_stats.py  — descriptive statistics (§7)

All public names are re-exported here so existing imports continue to work
without modification.
"""

from bci.bci_plots import (
    plot_bci_masses,
    plot_bci_components,
    plot_bci_topography,
    plot_bci_distribution,
)

from bci.bci_maps import (
    make_market_map,
    make_labour_map,
    make_supplier_map,
    make_bci_map,
)

from bci.bci_stats import compute_bci_stats

__all__ = [
    "plot_bci_masses",
    "plot_bci_components",
    "plot_bci_topography",
    "plot_bci_distribution",
    "make_market_map",
    "make_labour_map",
    "make_supplier_map",
    "make_bci_map",
    "compute_bci_stats",
]
