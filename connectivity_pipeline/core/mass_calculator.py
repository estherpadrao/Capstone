"""
File 5: Mass & Topography Calculator (PCI)
==========================================
Computes the weighted amenity mass surface (the "topography") used by PCI.
Weights and decay coefficients are fully user-configurable.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import ndimage
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from shapely.ops import unary_union

from core.h3_helper import HexGrid


# ---------------------------------------------------------------------------
# Defaults (from Zheng, Oeser & van Wee, 2021)
# ---------------------------------------------------------------------------

DEFAULT_AMENITY_WEIGHTS: Dict[str, float] = {
    "health":      0.319,
    "education":   0.276,
    "parks":       0.255,
    "community":   0.148,
    "food_retail": 0.0,
    "transit":     0.0,
}

DEFAULT_DECAY_COEFFICIENTS: Dict[str, float] = {
    "health":     0.08,
    "education":  0.08,
    "parks":      0.08,
    "community":  0.08,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MassLayer:
    """Holds raw and normalised values for one amenity category."""
    name: str
    raw_values: pd.Series
    weight: float = 1.0
    normalized_values: pd.Series = field(default=None)

    def __post_init__(self):
        if self.normalized_values is None:
            self.normalized_values = self._normalize(self.raw_values)

    @staticmethod
    def _normalize(values: pd.Series) -> pd.Series:
        vals = values.fillna(0)
        rng = vals.max() - vals.min()
        if rng == 0:
            return pd.Series(0.0, index=values.index)
        return (vals - vals.min()) / rng


# ---------------------------------------------------------------------------
# Mass Calculator
# ---------------------------------------------------------------------------

class MassCalculator:
    """
    Builds the attractiveness 'topography' for the PCI model.

    For each hex i:
        mass_i = Σ_k  w_k × count_k_i_normalized

    where k ∈ {health, education, parks, community}.

    food_retail and transit are fetched but not included in the
    topographic mass (they feed the network only).
    """

    def __init__(
        self,
        grid: HexGrid,
        amenity_weights: Optional[Dict[str, float]] = None,
        decay_coefficients: Optional[Dict[str, float]] = None,
    ):
        self.grid = grid
        self.hex_ids = grid.gdf["hex_id"].tolist()

        # User-overridable parameters
        self.amenity_weights: Dict[str, float] = {
            **DEFAULT_AMENITY_WEIGHTS,
            **(amenity_weights or {})
        }
        self.decay_coefficients: Dict[str, float] = {
            **DEFAULT_DECAY_COEFFICIENTS,
            **(decay_coefficients or {})
        }

        self.layers: Dict[str, MassLayer] = {}
        self._composite: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Building the surface
    # ------------------------------------------------------------------

    def add_amenity_layer(
        self,
        name: str,
        gdf: Optional[gpd.GeoDataFrame],
        use_area: bool = False,
    ):
        """
        Count (or measure area of) amenity features per hexagon and
        register as a MassLayer.

        Parameters
        ----------
        name     : amenity category name
        gdf      : GeoDataFrame of features; None → zeros
        use_area : use feature area instead of count (for parks)
        """
        if gdf is None or len(gdf) == 0:
            raw = pd.Series(0.0, index=self.hex_ids, name=name)
            self.layers[name] = MassLayer(
                name=name,
                raw_values=raw,
                weight=self.amenity_weights.get(name, 1.0),
            )
            return

        hex_gdf = self.grid.gdf[["hex_id", "geometry"]].copy()
        features = gdf.to_crs("EPSG:4326").copy()

        if use_area:
            # Compute intersection area between each park polygon and each hexagon.
            # This correctly splits a park that spans multiple hexagons — each hex
            # gets only the area that actually falls within it.
            features_m = features.to_crs(epsg=3857)
            features_m = features_m[features_m.geometry.notna() & ~features_m.geometry.is_empty].copy()
            features_m["geometry"] = features_m.geometry.make_valid()

            def _to_polygonal(geom):
                if geom is None or geom.is_empty:
                    return None
                if geom.geom_type in {"Polygon", "MultiPolygon"}:
                    return geom
                if geom.geom_type == "GeometryCollection":
                    polys = [
                        g for g in geom.geoms
                        if g.geom_type in {"Polygon", "MultiPolygon"} and not g.is_empty
                    ]
                    if not polys:
                        return None
                    return unary_union(polys)
                return None

            features_m["geometry"] = features_m.geometry.apply(_to_polygonal)
            features_m = features_m[features_m.geometry.notna() & ~features_m.geometry.is_empty].copy()
            hex_m      = hex_gdf.to_crs(epsg=3857)
            try:
                clipped = gpd.overlay(
                    hex_m[["hex_id", "geometry"]],
                    features_m[["geometry"]],
                    how="intersection",
                    keep_geom_type=False,
                )
                clipped["_area"] = clipped.geometry.area
                raw = clipped.groupby("hex_id")["_area"].sum()
            except Exception:
                # Fallback to centroid method if overlay fails (e.g. invalid geometries)
                features_m["_area"] = features_m.geometry.area
                features["_area"]   = features_m["_area"].values
                features["geometry"] = features.geometry.centroid
                joined = gpd.sjoin(
                    hex_gdf,
                    features[["geometry", "_area"]],
                    how="left",
                    predicate="intersects",
                )
                joined["_area"] = joined["_area"].fillna(0)
                raw = joined.groupby("hex_id")["_area"].sum()
        else:
            # Use centroid for polygon features
            pts = features.copy()
            mask = pts.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
            if mask.any():
                pts.loc[mask, "geometry"] = pts.loc[mask].geometry.centroid
            joined = gpd.sjoin(
                pts[["geometry"]], hex_gdf, how="left", predicate="within"
            )
            # Fallback: assign unmatched features to nearest hex (matches notebook)
            unassigned = joined["hex_id"].isna()
            if unassigned.any():
                nearest = gpd.sjoin_nearest(
                    pts.loc[unassigned, ["geometry"]], hex_gdf, how="left"
                )
                joined.loc[unassigned, "hex_id"] = nearest["hex_id"].values
            raw = joined.groupby("hex_id").size()

        raw = raw.reindex(self.hex_ids).fillna(0).astype(float)
        raw.name = name

        self.layers[name] = MassLayer(
            name=name,
            raw_values=raw,
            weight=self.amenity_weights.get(name, 1.0),
        )

    def compute_composite_mass(self) -> pd.Series:
        """
        Compute the weighted composite mass surface.
        Only categories that have a weight entry are combined.
        """
        if not self.layers:
            raise RuntimeError("No amenity layers added. Call add_amenity_layer() first.")

        total_weight = 0.0
        composite = pd.Series(0.0, index=self.hex_ids)

        for name, layer in self.layers.items():
            w = self.amenity_weights.get(name, 0.0)
            if w == 0.0:
                continue
            composite += w * layer.normalized_values.reindex(self.hex_ids).fillna(0)
            total_weight += w

        if total_weight > 0:
            composite /= total_weight

        self._composite = composite
        return composite

    # ------------------------------------------------------------------
    # Spatial helper: 2-D array for scipy operations
    # ------------------------------------------------------------------

    def get_topography_array(
        self,
        layer_name: Optional[str] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Return a 2-D numpy array of mass values for 3-D plotting.

        Parameters
        ----------
        layer_name : if None, uses composite; else a named layer
        """
        values = (
            self._composite if layer_name is None
            else self.layers[layer_name].normalized_values
        )
        arr = self._to_2d_array(values)
        meta = {
            "vmin": float(arr.min()),
            "vmax": float(arr.max()),
            "shape": arr.shape,
        }
        return arr, meta

    def _to_2d_array(self, series: pd.Series) -> np.ndarray:
        """Reshape a Series (indexed by hex_id) into a 2-D grid for scipy."""
        gdf = self.grid.gdf.copy()
        gdf["_val"] = gdf["hex_id"].map(series).fillna(0)
        gdf = gdf.to_crs(epsg=3857)
        cx = gdf.geometry.centroid.x.values
        cy = gdf.geometry.centroid.y.values

        # Create a regular grid
        n = max(int(np.sqrt(len(gdf))) + 1, 10)
        xi = np.linspace(cx.min(), cx.max(), n)
        yi = np.linspace(cy.min(), cy.max(), n)

        from scipy.interpolate import griddata
        zi = griddata(
            (cx, cy),
            gdf["_val"].values,
            (xi[None, :], yi[:, None]),
            method="linear",
            fill_value=0.0,
        )
        return zi

    def _from_2d_array(self, arr: np.ndarray, template: pd.Series) -> pd.Series:
        """Map smoothed 2-D array values back onto the hex Series."""
        gdf = self.grid.gdf.copy().to_crs(epsg=3857)
        cx = gdf.geometry.centroid.x.values
        cy = gdf.geometry.centroid.y.values

        n = arr.shape[0]
        xi = np.linspace(cx.min(), cx.max(), n)
        yi = np.linspace(cy.min(), cy.max(), n)

        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (yi, xi), arr, method="linear", bounds_error=False, fill_value=0.0
        )
        vals = interp(np.column_stack([cy, cx]))
        result = pd.Series(vals, index=gdf["hex_id"].values)
        return result.reindex(template.index).fillna(0)

    # ------------------------------------------------------------------
    # Peak / valley identification
    # ------------------------------------------------------------------

    def identify_peaks(self, threshold_percentile: float = 90) -> gpd.GeoDataFrame:
        """Return hexes above the given percentile of composite mass."""
        if self._composite is None:
            self.compute_composite_mass()
        thresh = self._composite.quantile(threshold_percentile / 100)
        mask = self.grid.gdf["hex_id"].map(self._composite) >= thresh
        return self.grid.gdf[mask].copy()

    def identify_valleys(self, threshold_percentile: float = 10) -> gpd.GeoDataFrame:
        """Return hexes below the given percentile of composite mass."""
        if self._composite is None:
            self.compute_composite_mass()
        thresh = self._composite.quantile(threshold_percentile / 100)
        mask = self.grid.gdf["hex_id"].map(self._composite) <= thresh
        return self.grid.gdf[mask].copy()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, layer in self.layers.items():
            rows.append({
                "Layer":     name,
                "Weight":    self.amenity_weights.get(name, 0.0),
                "Non-zero hexes": int((layer.raw_values > 0).sum()),
                "Max raw":   float(layer.raw_values.max()),
                "Mean norm": float(layer.normalized_values.mean()),
            })
        return pd.DataFrame(rows)
