"""
File 8: PCI Analysis — Compatibility Shim
==========================================
This module previously contained all PCI visualisation and stats code.
It has been split into three focused modules:

  pci/pci_plots.py  — matplotlib charts (§1, §2, §3, §6)
  pci/pci_maps.py   — folium interactive maps (§4, §5)
  pci/pci_stats.py  — descriptive statistics (§7)

All public names are re-exported here so existing imports continue to work
without modification.
"""

from pci.pci_plots import (
    plot_topography_layers,
    plot_topography_3d,
    plot_pci_components,
    plot_pci_distribution,
)

from pci.pci_maps import (
    make_mass_network_map,
    make_pci_map,
)

from pci.pci_stats import compute_pci_stats

__all__ = [
    "plot_topography_layers",
    "plot_topography_3d",
    "plot_pci_components",
    "plot_pci_distribution",
    "make_mass_network_map",
    "make_pci_map",
    "compute_pci_stats",
]
