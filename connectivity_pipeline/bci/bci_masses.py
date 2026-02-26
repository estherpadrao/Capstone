"""
Files 9 / 10 / 11: BCI Mass Calculators
=========================================
Separate modules for each of the three BCI demand-side masses.
Each is a thin class that can be computed independently.

File 9  → MarketMassCalculator   (Population × Income)
File 10 → LabourMassCalculator   (Employed population)
File 11 → SupplierMassCalculator (Business / agglomeration density)

All three share the same HexGrid and Census data; they are combined
in bci_calculator.py (File 12).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from dataclasses import dataclass, field
from typing import Optional

from core.h3_helper import HexGrid


# ---------------------------------------------------------------------------
# Shared normalisation helper
# ---------------------------------------------------------------------------

def _minmax_normalise(series: pd.Series) -> pd.Series:
    vals = series.fillna(0)
    lo, hi = vals.min(), vals.max()
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (vals - lo) / (hi - lo)


# ===========================================================================
# FILE 9: Market Mass  (M = Population × Income)
# ===========================================================================

class MarketMassCalculator:
    """
    Market Mass represents purchasing-power / consumer demand at each hex.

        M_market_i = P_i × (Y_i / Ȳ)

    where P_i is population and Y_i is median household income
    (normalised to the city median Ȳ to prevent extreme compression).

    Parameters
    ----------
    hex_ids : list of hex IDs (from grid.gdf["hex_id"])
    """

    def __init__(self, hex_ids: list):
        self.hex_ids = hex_ids
        self._raw:  Optional[pd.Series] = None
        self._norm: Optional[pd.Series] = None

    def compute(
        self,
        population: pd.Series,
        income: pd.Series,
        normalise_income: bool = True,
    ) -> pd.Series:
        """
        Parameters
        ----------
        population      : Series indexed by hex_id
        income          : Series indexed by hex_id (annual median HH income USD)
        normalise_income: divide income by city median before multiplying
        """
        P = population.reindex(self.hex_ids).fillna(0).astype(float)
        Y = income.reindex(self.hex_ids).astype(float)

        median_inc = Y.median()
        Y = Y.fillna(median_inc if pd.notna(median_inc) else 1.0)

        if normalise_income and median_inc and median_inc > 0:
            Y = Y / median_inc   # → relative income (1.0 = city median)

        raw = P * Y
        raw.index = self.hex_ids
        self._raw  = raw
        self._norm = _minmax_normalise(raw)

        print(f"   Market Mass — range: [{raw.min():.2f}, {raw.max():.2f}]  "
              f"non-zero hexes: {int((raw > 0).sum())}")
        return self._norm   # normalised [0, 1]

    @property
    def raw(self) -> pd.Series:
        if self._raw is None:
            raise RuntimeError("Call compute() first.")
        return self._raw

    @property
    def normalised(self) -> pd.Series:
        if self._norm is None:
            raise RuntimeError("Call compute() first.")
        return self._norm


# ===========================================================================
# FILE 10: Labour Mass  (M = Employed population)
# ===========================================================================

class LabourMassCalculator:
    """
    Labour Mass represents the pool of available workers at each location.

        M_labour_i = L_i  (employed population 16+)

    Simple count, normalised to [0, 1].
    """

    def __init__(self, hex_ids: list):
        self.hex_ids = hex_ids
        self._raw:  Optional[pd.Series] = None
        self._norm: Optional[pd.Series] = None

    def compute(self, labour: pd.Series) -> pd.Series:
        """
        Parameters
        ----------
        labour : Series indexed by hex_id (employed population count)
        """
        raw = labour.reindex(self.hex_ids).fillna(0).astype(float)
        self._raw  = raw
        self._norm = _minmax_normalise(raw)

        print(f"   Labour Mass — total workers: {raw.sum():,.0f}  "
              f"max per hex: {raw.max():,.0f}")
        return self._norm

    @property
    def raw(self) -> pd.Series:
        if self._raw is None:
            raise RuntimeError("Call compute() first.")
        return self._raw

    @property
    def normalised(self) -> pd.Series:
        if self._norm is None:
            raise RuntimeError("Call compute() first.")
        return self._norm


# ===========================================================================
# FILE 11: Supplier / Customer Mass  (M = Business density)
# ===========================================================================

class SupplierMassCalculator:
    """
    Supplier Mass represents agglomeration economies and access to
    business services / inputs.

        M_supplier_i = BusinessDensity_i

    where BusinessDensity is the count of supplier-type OSM features
    per hexagon, normalised to [0, 1].

    Optionally computes an urban-interface bonus (airport proximity
    and edge-of-urban-area weighting).
    """

    def __init__(self, hex_ids: list):
        self.hex_ids = hex_ids
        self._raw:  Optional[pd.Series] = None
        self._norm: Optional[pd.Series] = None
        self._urban_interface: Optional[pd.Series] = None

    def compute(self, supplier_counts: pd.Series) -> pd.Series:
        """
        Parameters
        ----------
        supplier_counts : Series indexed by hex_id (count of OSM business features)
        """
        raw = supplier_counts.reindex(self.hex_ids).fillna(0).astype(float)
        self._raw  = raw
        self._norm = _minmax_normalise(raw)

        print(f"   Supplier Mass — total features: {raw.sum():,.0f}  "
              f"hexes with suppliers: {int((raw > 0).sum())}")
        return self._norm

    def compute_urban_interface(
        self,
        grid: HexGrid,
        boundary_polygon=None,
        airport_locations: Optional[list] = None,
    ) -> pd.Series:
        """
        Urban interface bonus: captures edge-of-centre agglomeration and
        proximity to airports / transport hubs.

        Score = 0.5 × EdgeScore_i + 0.5 × AirportProximity_i
        (each normalised to [0, 1])

        Parameters
        ----------
        grid              : HexGrid
        boundary_polygon  : city boundary (for edge proximity)
        airport_locations : list of (lat, lng) tuples for major airports
        """
        from shapely.geometry import Point
        import math

        gdf = grid.gdf.copy().to_crs("EPSG:4326")
        centroids = gdf.geometry.centroid

        # --- Edge / urban-fringe score ---
        edge_scores = pd.Series(0.0, index=gdf["hex_id"])
        if boundary_polygon is not None:
            from shapely.ops import unary_union
            # Handle both Polygon and MultiPolygon (e.g. SF has islands)
            if boundary_polygon.geom_type == "Polygon":
                boundary_ext = boundary_polygon.exterior
            else:
                rings = [geom.exterior for geom in boundary_polygon.geoms]
                boundary_ext = unary_union(rings)
            for hx, pt in zip(gdf["hex_id"], centroids):
                dist_m = pt.distance(boundary_ext) * 111000  # approx degrees to m
                # Hexes close to city boundary score higher
                edge_scores[hx] = 1.0 / (1.0 + dist_m / 2000)
            edge_scores = _minmax_normalise(edge_scores)

        # --- Airport proximity score ---
        airport_scores = pd.Series(0.0, index=gdf["hex_id"])
        if airport_locations:
            for hx, pt in zip(gdf["hex_id"], centroids):
                dists = [
                    math.sqrt((pt.y - alat) ** 2 + (pt.x - alng) ** 2) * 111
                    for alat, alng in airport_locations
                ]
                min_dist_km = min(dists)
                # Gaussian decay: σ = 5 km
                airport_scores[hx] = math.exp(-(min_dist_km ** 2) / (2 * 5 ** 2))
            airport_scores = _minmax_normalise(airport_scores)

        interface = 0.5 * edge_scores + 0.5 * airport_scores
        self._urban_interface = _minmax_normalise(interface)
        return self._urban_interface

    @property
    def raw(self) -> pd.Series:
        if self._raw is None:
            raise RuntimeError("Call compute() first.")
        return self._raw

    @property
    def normalised(self) -> pd.Series:
        if self._norm is None:
            raise RuntimeError("Call compute() first.")
        return self._norm

    @property
    def urban_interface(self) -> Optional[pd.Series]:
        return self._urban_interface
