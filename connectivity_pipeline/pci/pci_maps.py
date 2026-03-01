"""
File 8b: PCI Folium Maps
=========================
Interactive folium map functions for PCI.

Extracted from pci_analysis.py §4, §5.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium import GeoJsonTooltip
import branca.colormap as cm
from branca.colormap import LinearColormap

from core.h3_helper import HexGrid
from analysis.shared import (
    _folium_center,
    build_neighborhood_colors,
    add_neighborhood_overlay,
)


# ---------------------------------------------------------------------------
# 4. Interactive mass + network map
# ---------------------------------------------------------------------------

def make_mass_network_map(
    grid: HexGrid,
    net,
    city_name: str = "",
) -> folium.Map:
    """
    Interactive folium map: composite mass choropleth (from the 'mass'
    column attached to the grid) with optional walk-network edge overlay.
    """
    center = _folium_center(grid.gdf)
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    if "mass" not in grid.gdf.columns:
        return m

    valid = grid.gdf["mass"].dropna()
    if len(valid) == 0:
        return m

    cmap = cm.linear.YlOrRd_09.scale(valid.min(), valid.max())
    cmap.caption = "Composite Mass"

    gdf = grid.gdf[["hex_id", "mass", "geometry"]].copy()

    def _style(feature, _cmap=cmap):
        val = feature["properties"].get("mass")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return {"fillColor": "#ccc", "fillOpacity": 0.3, "weight": 0.2}
        return {"fillColor": _cmap(val), "fillOpacity": 0.7, "weight": 0.2, "color": "gray"}

    folium.GeoJson(
        gdf.to_json(),
        style_function=_style,
        tooltip=GeoJsonTooltip(["hex_id", "mass"], aliases=["Hex:", "Mass:"]),
        name="Composite Mass",
    ).add_to(m)
    cmap.add_to(m)

    # Overlay walk network edges if available
    if net is not None:
        try:
            import osmnx as ox
            walk_g = net.networks.get("walk")
            if walk_g is not None:
                edges = ox.graph_to_gdfs(walk_g, nodes=False)
                edges = edges.to_crs("EPSG:4326")
                folium.GeoJson(
                    edges[["geometry"]].to_json(),
                    style_function=lambda _: {
                        "color": "#555", "weight": 0.5, "opacity": 0.25
                    },
                    name="Walk Network",
                ).add_to(m)
        except Exception:
            pass

    folium.LayerControl().add_to(m)
    return m


# ---------------------------------------------------------------------------
# 5. Final PCI interactive map
# ---------------------------------------------------------------------------

def make_pci_map(
    grid: HexGrid,
    pci: pd.Series,
    net,
    city_name: str = "",
    neighborhoods_gdf=None,
) -> folium.Map:
    """Interactive folium choropleth of final PCI scores with optional neighbourhood overlay."""
    center = _folium_center(grid.gdf)
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    cmap = LinearColormap(
        ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"],
        vmin=0, vmax=100, caption="PCI (0–100)",
    )

    gdf = grid.gdf.copy()
    gdf["PCI"] = gdf["hex_id"].map(pci)

    tooltip_fields  = ["hex_id", "PCI"]
    tooltip_aliases = ["Hex:", "PCI:"]
    if "neighborhood" in gdf.columns:
        tooltip_fields.append("neighborhood")
        tooltip_aliases.append("Neighborhood:")
    for extra in ["median_income", "active_street_score", "mass"]:
        if extra in gdf.columns:
            tooltip_fields.append(extra)
            tooltip_aliases.append(extra.replace("_", " ").title() + ":")

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

    # Neighbourhood boundary overlay (toggleable)
    if neighborhoods_gdf is not None and not neighborhoods_gdf.empty:
        nb_colors = build_neighborhood_colors(neighborhoods_gdf["name"].tolist())
        add_neighborhood_overlay(m, neighborhoods_gdf, nb_colors, layer_name="Neighborhoods")

    folium.LayerControl(collapsed=False).add_to(m)
    return m
