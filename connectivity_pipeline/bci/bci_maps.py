"""
File 13b: BCI Folium Maps
==========================
Interactive folium map functions for BCI (component maps and final BCI map).

Extracted from bci_analysis.py §3, §4.
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
# 3. Interactive maps per mass component
# ---------------------------------------------------------------------------

def make_market_map(grid: HexGrid, city_name: str = "") -> folium.Map:
    """Interactive folium map of Market Mass."""
    return _make_component_map(grid, "market_mass", "Market Mass (P×Y)", "YlOrRd", city_name)


def make_labour_map(grid: HexGrid, city_name: str = "") -> folium.Map:
    """Interactive folium map of Labour Mass."""
    return _make_component_map(grid, "labour_mass", "Labour Mass (L)", "YlGnBu", city_name)


def make_supplier_map(grid: HexGrid, city_name: str = "") -> folium.Map:
    """Interactive folium map of Supplier Mass."""
    return _make_component_map(grid, "supplier_mass", "Supplier Mass (S)", "PuRd", city_name)


def _make_component_map(
    grid: HexGrid,
    column: str,
    caption: str,
    colormap_name: str,
    city_name: str,
) -> folium.Map:
    center = _folium_center(grid.gdf)
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
    neighborhoods_gdf=None,
) -> folium.Map:
    """Interactive folium choropleth of final BCI scores with optional neighbourhood overlay."""
    center = _folium_center(grid.gdf)
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    cmap = LinearColormap(
        ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"],
        vmin=0, vmax=100, caption="BCI (0–100)",
    )

    gdf = grid.gdf.copy()
    gdf["BCI"] = gdf["hex_id"].map(bci)

    tooltip_fields  = ["hex_id", "BCI"]
    tooltip_aliases = ["Hex:", "BCI:"]
    if "neighborhood" in gdf.columns:
        tooltip_fields.append("neighborhood")
        tooltip_aliases.append("Neighborhood:")
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

    # Neighbourhood boundary overlay (toggleable)
    if neighborhoods_gdf is not None and not neighborhoods_gdf.empty:
        nb_colors = build_neighborhood_colors(neighborhoods_gdf["name"].tolist())
        add_neighborhood_overlay(m, neighborhoods_gdf, nb_colors, layer_name="Neighborhoods")

    folium.LayerControl(collapsed=False).add_to(m)
    return m
