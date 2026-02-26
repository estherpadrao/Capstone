"""
File 1: H3 Helper
=================
H3 hexagonal grid utilities — version-safe wrappers for H3 v3.x and v4.x,
plus the HexGrid class used by both PCI and BCI.
"""

import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from typing import Set, Dict, Optional, List


# ---------------------------------------------------------------------------
# Version-safe H3 wrappers
# ---------------------------------------------------------------------------

def h3_polyfill(polygon, resolution: int) -> Set[str]:
    """Fill a Shapely polygon with H3 hexagons (v3 + v4 safe)."""
    try:
        return set(h3.geo_to_cells(polygon, resolution))        # v4
    except AttributeError:
        geojson = polygon.__geo_interface__
        return set(h3.polyfill(geojson, resolution, geo_json_conformant=True))  # v3


def h3_to_boundary(hex_id: str) -> List[tuple]:
    """Return (lng, lat) boundary vertices of a hex (Shapely order)."""
    try:
        coords = h3.cell_to_boundary(hex_id)     # v4 → [(lat, lng), ...]
    except AttributeError:
        coords = h3.h3_to_geo_boundary(hex_id)   # v3

    return [(lng, lat) for lat, lng in coords]


def h3_to_center(hex_id: str) -> tuple:
    """Return (lat, lng) centroid of a hex."""
    try:
        return h3.cell_to_latlng(hex_id)          # v4
    except AttributeError:
        return h3.h3_to_geo(hex_id)               # v3


def h3_get_neighbors(hex_id: str, ring_size: int = 1) -> Set[str]:
    """Return the k-ring of neighboring hex IDs."""
    try:
        return h3.grid_disk(hex_id, ring_size)    # v4
    except AttributeError:
        return h3.k_ring(hex_id, ring_size)       # v3


# ---------------------------------------------------------------------------
# HexGrid class
# ---------------------------------------------------------------------------

class HexGrid:
    """
    Manages the H3 hexagonal tessellation over a city boundary.

    Shared by PCI and BCI — both receive the same grid object so that
    results can be attached to the same GeoDataFrame and compared directly.
    """

    def __init__(self, resolution: int = 8):
        self.resolution = resolution
        self.boundary: Optional[object] = None   # Shapely geometry
        self.hex_ids: Set[str] = set()
        self.gdf: Optional[gpd.GeoDataFrame] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def create_from_boundary(
        self,
        boundary,
        clip_to_boundary: bool = True
    ) -> gpd.GeoDataFrame:
        """
        Tessellate a city boundary with H3 hexagons.

        Parameters
        ----------
        boundary : GeoDataFrame | Shapely geometry
        clip_to_boundary : trim hexes that extend outside the boundary
        """
        if isinstance(boundary, gpd.GeoDataFrame):
            if boundary.crs and boundary.crs.to_epsg() != 4326:
                boundary = boundary.to_crs(4326)
            self.boundary = unary_union(boundary.geometry)
        else:
            self.boundary = boundary

        self.hex_ids = h3_polyfill(self.boundary, self.resolution)
        if not self.hex_ids:
            raise ValueError(
                "No hexagons generated — check that the boundary is valid and in EPSG:4326."
            )

        print(f"   Generated {len(self.hex_ids)} raw hexagons at H3 resolution {self.resolution}")
        self.gdf = self._build_geodataframe(clip_to_boundary)
        return self.gdf

    def _build_geodataframe(self, clip: bool) -> gpd.GeoDataFrame:
        records = []
        for hex_id in self.hex_ids:
            coords = h3_to_boundary(hex_id)
            records.append({"hex_id": hex_id, "geometry": Polygon(coords)})

        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

        if clip and self.boundary is not None:
            boundary_gdf = gpd.GeoDataFrame(geometry=[self.boundary], crs="EPSG:4326")
            gdf = gpd.overlay(gdf, boundary_gdf, how="intersection")
            self.hex_ids = set(gdf["hex_id"].tolist())

        # Area in m²
        gdf_m = gdf.to_crs(epsg=3857)
        gdf["area_m2"] = gdf_m.geometry.area

        return gdf

    # ------------------------------------------------------------------
    # Data attachment
    # ------------------------------------------------------------------

    def attach_data(self, data, column_name: str) -> gpd.GeoDataFrame:
        """
        Attach a dict or Series (keyed on hex_id) as a column in gdf.
        Missing hexes are filled with NaN.
        """
        if isinstance(data, dict):
            self.gdf[column_name] = self.gdf["hex_id"].map(data)
        elif isinstance(data, pd.Series):
            self.gdf[column_name] = self.gdf["hex_id"].map(data)
        else:
            raise TypeError(f"attach_data expects dict or pd.Series, got {type(data)}")
        return self.gdf

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @property
    def centroids(self) -> gpd.GeoDataFrame:
        """GeoDataFrame of hex centroids (EPSG:4326)."""
        points = []
        for hex_id in self.gdf["hex_id"]:
            lat, lng = h3_to_center(hex_id)
            points.append(Point(lng, lat))
        return gpd.GeoDataFrame(
            {"hex_id": self.gdf["hex_id"].values},
            geometry=points,
            crs="EPSG:4326"
        )

    def get_neighbors(self, hex_id: str, ring_size: int = 1) -> List[str]:
        """Return neighbors of a hex that are also in this grid."""
        return [
            h for h in h3_get_neighbors(hex_id, ring_size)
            if h != hex_id and h in self.hex_ids
        ]

    def __len__(self) -> int:
        return len(self.hex_ids)

    def __repr__(self) -> str:
        return (
            f"HexGrid(resolution={self.resolution}, "
            f"hexagons={len(self.hex_ids)}, "
            f"crs='EPSG:4326')"
        )
