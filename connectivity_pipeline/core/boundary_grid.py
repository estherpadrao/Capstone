"""
File 3: Boundary & Grid Clipper
================================
Fetches the city boundary from OSM or a local file,
and exposes the clipped HexGrid ready for both PCI and BCI.
"""

import os
import osmnx as ox
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import shape
import json
from typing import Optional

from core.h3_helper import HexGrid


class BoundaryFetcher:
    """
    Retrieves a city boundary polygon (EPSG:4326) and constructs
    the H3 hex grid.  Used by both PCI and BCI so they share the
    same spatial footprint.
    """

    def __init__(self, city_name: str):
        self.city_name = city_name
        self._boundary_gdf: Optional[gpd.GeoDataFrame] = None
        self._boundary_polygon = None

    # ------------------------------------------------------------------
    # Boundary
    # ------------------------------------------------------------------

    def fetch_from_osm(self) -> gpd.GeoDataFrame:
        """Download boundary from Nominatim / OpenStreetMap."""
        print(f"📍 Fetching boundary: {self.city_name}")
        gdf = ox.geocode_to_gdf(self.city_name)
        gdf = gdf.to_crs("EPSG:4326")
        self._boundary_gdf = gdf
        self._boundary_polygon = unary_union(gdf.geometry)
        area_km2 = gdf.to_crs(epsg=3857).geometry.area.sum() / 1e6
        print(f"   ✓ Boundary fetched: {area_km2:.1f} km²")
        return gdf

    def fetch_from_file(self, path: str) -> gpd.GeoDataFrame:
        """Load boundary from a local GeoJSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Boundary file not found: {path}")
        print(f"📍 Loading boundary from file: {path}")
        gdf = gpd.read_file(path).to_crs("EPSG:4326")
        self._boundary_gdf = gdf
        self._boundary_polygon = unary_union(gdf.geometry)
        area_km2 = gdf.to_crs(epsg=3857).geometry.area.sum() / 1e6
        print(f"   ✓ Boundary loaded: {area_km2:.1f} km²")
        return gdf

    def get_boundary(
        self,
        use_local: bool = False,
        local_path: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Convenience method: use local file if available, else OSM.

        Parameters
        ----------
        use_local : prefer local GeoJSON
        local_path : path to local GeoJSON (required if use_local=True)
        """
        if use_local and local_path and os.path.exists(local_path):
            return self.fetch_from_file(local_path)
        return self.fetch_from_osm()

    @property
    def boundary_polygon(self):
        """Shapely (Multi)Polygon of city boundary (EPSG:4326)."""
        if self._boundary_polygon is None:
            raise RuntimeError("Call get_boundary() first.")
        return self._boundary_polygon

    @property
    def boundary_gdf(self) -> gpd.GeoDataFrame:
        if self._boundary_gdf is None:
            raise RuntimeError("Call get_boundary() first.")
        return self._boundary_gdf

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    def build_grid(self, resolution: int = 8) -> HexGrid:
        """Create and return a HexGrid tessellating the boundary."""
        grid = HexGrid(resolution=resolution)
        grid.create_from_boundary(self.boundary_gdf, clip_to_boundary=True)
        print(f"   ✓ Grid: {len(grid)} hexagons at H3 resolution {resolution}")
        return grid
