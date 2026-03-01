"""
BCI Mass Calculator
===================
Single unified class matching the notebook's BCIMassCalculator exactly.

Three mass components:
  Market Mass   = P × (Y / Y_median)   — purchasing power / customer demand
  Labour Mass   = L                    — employed population (worker availability)
  Supplier Mass = S                    — business density (OSM features per hex)

All three store the RAW mass (before any normalisation) so that the Hansen
accessibility model can use them directly, matching notebook behaviour.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from dataclasses import dataclass
from typing import Optional, Dict

from core.h3_helper import HexGrid


# ---------------------------------------------------------------------------
# Internal layer wrapper (mirrors notebook BCIMassLayer)
# ---------------------------------------------------------------------------

@dataclass
class BCIMassLayer:
    """Wraps one mass component and its min-max normalised counterpart."""
    name: str
    raw_values: pd.Series
    normalized_values: pd.Series = None

    def __post_init__(self):
        if self.normalized_values is None:
            self.normalized_values = self._normalize(self.raw_values)

    @staticmethod
    def _normalize(values: pd.Series) -> pd.Series:
        """Min-max normalize to [0, 1]."""
        vals = values.fillna(0)
        lo, hi = vals.min(), vals.max()
        if hi == lo:
            return pd.Series(0.0, index=values.index)
        return (vals - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Unified BCIMassCalculator  (matches notebook class of the same name)
# ---------------------------------------------------------------------------

class BCIMassCalculator:
    """
    Computes all three BCI mass components for a hex grid.

    Mirrors notebook BCIMassCalculator exactly:
      - compute_market_mass(normalize_income=True)  →  P × (Y / Y_median)
      - compute_labour_mass()                        →  L
      - compute_supplier_mass(supplier_counts)       →  S (business count)

    All public mass attributes (market_mass, labour_mass, supplier_mass) are
    RAW values so they can be passed directly to BCIHansenAccessibility.
    Normalised [0,1] versions live in self.layers[key].normalized_values.
    """

    def __init__(self, hex_grid: HexGrid):
        self.hex_grid = hex_grid
        self.hex_ids  = hex_grid.gdf["hex_id"].tolist()
        self.layers:  Dict[str, BCIMassLayer] = {}

        # Raw census inputs (set by load_census_data)
        self.population:   Optional[pd.Series] = None
        self.income:       Optional[pd.Series] = None
        self.labour:       Optional[pd.Series] = None
        self.income_filled: Optional[pd.Series] = None

        # Computed raw masses
        self.market_mass:   Optional[pd.Series] = None
        self.labour_mass:   Optional[pd.Series] = None
        self.supplier_mass: Optional[pd.Series] = None

        # Optional urban-interface score (SupplierMassCalculator compat)
        self._urban_interface: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Step 1 — load census data
    # ------------------------------------------------------------------

    def load_census_data(
        self,
        population: pd.Series,
        income: pd.Series,
        labour: pd.Series,
    ):
        """Load hex-indexed census data. Matches notebook load_census_data()."""
        self.population = population.reindex(self.hex_ids).fillna(0).astype(float)
        self.income      = income.reindex(self.hex_ids).astype(float)      # keep NaN
        self.labour      = labour.reindex(self.hex_ids).fillna(0).astype(float)

        median_income      = self.income.median()
        self.income_filled = self.income.fillna(
            median_income if pd.notna(median_income) else 1.0
        )

        print(f"   📊 Census data loaded:")
        print(f"      Population : sum={self.population.sum():,.0f}, max={self.population.max():,.0f}")
        print(f"      Income     : median=${median_income:,.0f}, "
              f"valid={self.income.notna().sum()}/{len(self.income)}")
        print(f"      Labour     : sum={self.labour.sum():,.0f}, max={self.labour.max():,.0f}")

    # ------------------------------------------------------------------
    # Step 2 — compute masses
    # ------------------------------------------------------------------

    def compute_market_mass(self, normalize_income: bool = True) -> pd.Series:
        """
        Market Mass = P × Y

        Y is optionally normalised to the city median (normalize_income=True)
        so extreme income values don't dominate.  Matches notebook exactly.
        """
        print("   📊 Computing Market Mass (P × Y)...")

        P = self.population.copy()
        Y = self.income_filled.copy()

        if normalize_income:
            Y_median = Y.median()
            if Y_median and Y_median > 0:
                Y = Y / Y_median   # relative income: 1.0 = city median

        self.market_mass = (P * Y).fillna(0).clip(lower=0)
        self.market_mass.name = "market_mass"
        self.layers["market"] = BCIMassLayer("market", self.market_mass)

        print(f"      ✓ Market mass: sum={self.market_mass.sum():,.0f}, "
              f"max={self.market_mass.max():,.1f}")
        return self.market_mass   # RAW — not normalised

    def compute_labour_mass(self) -> pd.Series:
        """Labour Mass = L (employed population). Matches notebook exactly."""
        print("   📊 Computing Labour Mass (L)...")

        self.labour_mass = self.labour.copy().clip(lower=0)
        self.labour_mass.name = "labour_mass"
        self.layers["labour"] = BCIMassLayer("labour", self.labour_mass)

        print(f"      ✓ Labour mass: sum={self.labour_mass.sum():,.0f}, "
              f"max={self.labour_mass.max():,.1f}")
        return self.labour_mass   # RAW

    def compute_supplier_mass(self, supplier_counts: pd.Series) -> pd.Series:
        """Supplier Mass = S (OSM business density). Matches notebook exactly."""
        print("   📊 Computing Supplier Mass (S)...")

        self.supplier_mass = (
            supplier_counts.reindex(self.hex_ids).fillna(0).clip(lower=0)
        )
        self.supplier_mass.name = "supplier_mass"
        self.layers["supplier"] = BCIMassLayer("supplier", self.supplier_mass)

        print(f"      ✓ Supplier mass: sum={self.supplier_mass.sum():,.0f}, "
              f"max={self.supplier_mass.max():,.1f}")
        return self.supplier_mass  # RAW

    # ------------------------------------------------------------------
    # Optional urban interface
    # ------------------------------------------------------------------

    def compute_urban_interface(
        self,
        boundary_polygon=None,
        airport_locations: Optional[list] = None,
    ) -> pd.Series:
        """
        Urban interface score (normalised [0, 1]):
          0.5 × EdgeScore  +  0.5 × AirportProximity

        Stored in self.layers["urban_interface"].raw_values and
        self._urban_interface for backward compatibility.
        """
        from shapely.geometry import Point
        import math

        gdf       = self.hex_grid.gdf.copy().to_crs("EPSG:4326")
        centroids = gdf.geometry.centroid

        # --- Edge / urban-fringe score ---
        edge_scores = pd.Series(0.0, index=gdf["hex_id"])
        if boundary_polygon is not None:
            from shapely.ops import unary_union
            rings = ([boundary_polygon.exterior]
                     if boundary_polygon.geom_type == "Polygon"
                     else [g.exterior for g in boundary_polygon.geoms])
            boundary_ext = unary_union(rings)
            for hx, pt in zip(gdf["hex_id"], centroids):
                dist_m = pt.distance(boundary_ext) * 111_000
                edge_scores[hx] = 1.0 / (1.0 + dist_m / 2000)
            mn, mx = edge_scores.min(), edge_scores.max()
            edge_scores = (edge_scores - mn) / (mx - mn) if mx > mn else edge_scores

        # --- Airport proximity score ---
        airport_scores = pd.Series(0.0, index=gdf["hex_id"])
        if airport_locations:
            for hx, pt in zip(gdf["hex_id"], centroids):
                dists = [
                    math.sqrt((pt.y - alat) ** 2 + (pt.x - alng) ** 2) * 111
                    for alat, alng in airport_locations
                ]
                airport_scores[hx] = math.exp(-(min(dists) ** 2) / (2 * 5 ** 2))
            mn, mx = airport_scores.min(), airport_scores.max()
            airport_scores = (airport_scores - mn) / (mx - mn) if mx > mn else airport_scores

        interface_raw = 0.5 * edge_scores + 0.5 * airport_scores
        mn, mx = interface_raw.min(), interface_raw.max()
        interface = (
            (interface_raw - mn) / (mx - mn) if mx > mn
            else pd.Series(0.5, index=interface_raw.index)
        )
        interface.name = "urban_interface"
        self._urban_interface = interface
        self.layers["urban_interface"] = BCIMassLayer("urban_interface", interface)
        return interface

    # ------------------------------------------------------------------
    # Topography helper (for 3-D plots)
    # ------------------------------------------------------------------

    def get_topography_array(self, mass_type: str = "market"):
        """
        Returns (2-D numpy array, metadata dict) for a surface plot.
        Matches notebook get_topography_array() exactly.
        """
        from scipy.ndimage import gaussian_filter

        mass_map = {
            "market":   self.market_mass,
            "labour":   self.labour_mass,
            "supplier": self.supplier_mass,
        }
        if mass_type not in mass_map or mass_map[mass_type] is None:
            raise ValueError(f"Unknown or uncomputed mass_type: '{mass_type}'")

        mass     = mass_map[mass_type].fillna(0).clip(lower=0)
        centroids = self.hex_grid.centroids.merge(
            pd.DataFrame({"hex_id": self.hex_ids, "mass": mass.values}),
            on="hex_id",
        )

        bounds     = self.hex_grid.gdf.total_bounds
        resolution = max(int(np.sqrt(len(self.hex_ids)) * 2), 20)

        x_edges = np.linspace(bounds[0], bounds[2], resolution + 1)
        y_edges = np.linspace(bounds[1], bounds[3], resolution + 1)

        x_coords  = centroids.geometry.x.values
        y_coords  = centroids.geometry.y.values
        mass_vals = centroids["mass"].values

        x_bins = np.clip(np.digitize(x_coords, x_edges) - 1, 0, resolution - 1)
        y_bins = np.clip(np.digitize(y_coords, y_edges) - 1, 0, resolution - 1)

        grid   = np.zeros((resolution, resolution))
        counts = np.zeros((resolution, resolution))
        for i in range(len(centroids)):
            grid[y_bins[i], x_bins[i]]   += mass_vals[i]
            counts[y_bins[i], x_bins[i]] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            grid = np.where(counts > 0, grid / counts, 0)

        grid = gaussian_filter(np.clip(grid, 0, None), sigma=1.5)
        return grid, {"bounds": bounds, "resolution": resolution}

    # ------------------------------------------------------------------
    # Backward-compatibility shims for code that used the old 3-class API
    # ------------------------------------------------------------------

    @property
    def raw(self) -> pd.Series:
        """Alias for market_mass.raw — keeps old MarketMassCalculator API."""
        if self.market_mass is None:
            raise RuntimeError("Call compute_market_mass() first.")
        return self.market_mass

    @property
    def urban_interface(self) -> Optional[pd.Series]:
        return self._urban_interface

    def summary(self) -> pd.DataFrame:
        """Summary statistics for all computed masses."""
        records = []
        for name, mass in [
            ("market",   self.market_mass),
            ("labour",   self.labour_mass),
            ("supplier", self.supplier_mass),
        ]:
            if mass is not None:
                records.append({
                    "mass_type":    name,
                    "sum":          mass.sum(),
                    "mean":         mass.mean(),
                    "max":          mass.max(),
                    "nonzero_hexes": int((mass > 0).sum()),
                })
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Backward-compatibility aliases
# (old code that imported the three separate classes still works)
# ---------------------------------------------------------------------------

class MarketMassCalculator:
    """Thin shim — delegates to BCIMassCalculator for backward compatibility."""

    def __init__(self, hex_ids: list):
        self._hex_ids = hex_ids
        self._calc: Optional[BCIMassCalculator] = None

    def _ensure_calc(self, grid_or_none=None):
        if self._calc is None:
            # Create a minimal HexGrid-like proxy if a full grid is not provided
            class _GridProxy:
                def __init__(self, hex_ids):
                    import geopandas as gpd
                    import pandas as pd
                    self.gdf = gpd.GeoDataFrame({"hex_id": hex_ids})
                    self.centroids = gpd.GeoDataFrame({"hex_id": hex_ids})
            self._calc = BCIMassCalculator(_GridProxy(self._hex_ids))

    def compute(
        self,
        population: pd.Series,
        income: pd.Series,
        normalise_income: bool = True,
    ) -> pd.Series:
        self._ensure_calc()
        self._calc.load_census_data(population, income,
                                     pd.Series(0.0, index=self._hex_ids))
        return self._calc.compute_market_mass(normalize_income=normalise_income)

    @property
    def raw(self) -> pd.Series:
        return self._calc.market_mass

    @property
    def normalised(self) -> pd.Series:
        return self._calc.layers["market"].normalized_values


class LabourMassCalculator:
    """Thin shim — delegates to BCIMassCalculator for backward compatibility."""

    def __init__(self, hex_ids: list):
        self._hex_ids = hex_ids
        self._raw: Optional[pd.Series] = None
        self._norm: Optional[pd.Series] = None

    def compute(self, labour: pd.Series) -> pd.Series:
        raw = labour.reindex(self._hex_ids).fillna(0).astype(float).clip(lower=0)
        self._raw  = raw
        lo, hi = raw.min(), raw.max()
        self._norm = (raw - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=raw.index)
        print(f"   Labour Mass — total workers: {raw.sum():,.0f}  max per hex: {raw.max():,.0f}")
        return self._norm

    @property
    def raw(self) -> pd.Series:
        return self._raw

    @property
    def normalised(self) -> pd.Series:
        return self._norm


class SupplierMassCalculator:
    """Thin shim — delegates to BCIMassCalculator for backward compatibility."""

    def __init__(self, hex_ids: list):
        self._hex_ids = hex_ids
        self._raw: Optional[pd.Series] = None
        self._norm: Optional[pd.Series] = None
        self._urban_interface: Optional[pd.Series] = None

    def compute(self, supplier_counts: pd.Series) -> pd.Series:
        raw = supplier_counts.reindex(self._hex_ids).fillna(0).astype(float)
        self._raw  = raw
        lo, hi = raw.min(), raw.max()
        self._norm = (raw - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=raw.index)
        print(f"   Supplier Mass — total: {raw.sum():,.0f}  hexes: {int((raw > 0).sum())}")
        return self._norm

    def compute_urban_interface(
        self,
        grid: HexGrid,
        boundary_polygon=None,
        airport_locations: Optional[list] = None,
    ) -> pd.Series:
        """Delegates to BCIMassCalculator.compute_urban_interface."""
        _calc = BCIMassCalculator(grid)
        result = _calc.compute_urban_interface(boundary_polygon, airport_locations)
        self._urban_interface = result
        return result

    @property
    def raw(self) -> pd.Series:
        return self._raw

    @property
    def normalised(self) -> pd.Series:
        return self._norm

    @property
    def urban_interface(self) -> Optional[pd.Series]:
        return self._urban_interface
