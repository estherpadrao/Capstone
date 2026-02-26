"""
File 7: PCI Calculator
======================
Hansen accessibility model + final PCI score computation.
Active street bonus (lambda) and park masking are user-configurable.
Beta can be set globally or per mode.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy.spatial import cKDTree
from typing import Dict, Optional

from core.h3_helper import HexGrid
from core.mass_calculator import MassCalculator
from core.network_builder import MultiModalNetworkBuilder


# ---------------------------------------------------------------------------
# Hansen Accessibility Model
# ---------------------------------------------------------------------------

class HansenAccessibilityModel:
    """
    Computes Hansen gravity-type accessibility:

        A_i = Σ_j  M_j × exp(-β × t_ij) × CostAdjustment_ij

    where:
        M_j   = attractiveness (topographic mass) of destination j
        t_ij  = travel time (minutes) from i to j
        β     = distance decay coefficient (user-configurable, per mode)
        CostAdjustment = income-weighted travel cost penalty
    """

    def __init__(
        self,
        grid: HexGrid,
        network_builder: MultiModalNetworkBuilder,
        mass_calculator: MassCalculator,
    ):
        self.grid   = grid
        self.net    = network_builder
        self.mass   = mass_calculator
        self.hex_ids = grid.gdf["hex_id"].tolist()

        self._hex_to_node: Dict[str, int] = {}
        self._travel_times: Optional[Dict[str, Dict[str, float]]] = None  # {origin_hex: {dest_hex: min}}
        self._accessibility: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Step 1: compute travel times
    # ------------------------------------------------------------------

    def compute_travel_times(self, max_time: float = 90.0):
        """
        Compute shortest-path travel times between all hex pairs
        using the unified multi-modal graph.

        Results cached in self._travel_times.
        """
        print(f"⏱  Computing travel times (max {max_time} min)...")
        G = self.net.unified_graph
        if G is None:
            raise RuntimeError("Network not built. Call network_builder.build_all_networks() first.")

        # Map each hex centroid to its nearest network node
        centroids = self.grid.centroids
        coords    = [(r.geometry.y, r.geometry.x) for _, r in centroids.iterrows()]
        hex_id_list = centroids["hex_id"].tolist()
        nearest = self.net.get_nearest_nodes_batch(coords, mode="unified")
        self._hex_to_node = dict(zip(hex_id_list, nearest))

        unique_nodes = list(set(self._hex_to_node.values()))
        # node → list of hex_ids
        node_to_hexes: Dict[int, list] = {}
        for hx, nd in self._hex_to_node.items():
            node_to_hexes.setdefault(nd, []).append(hx)

        travel_times: Dict[str, Dict[str, float]] = {hx: {} for hx in self.hex_ids}
        total = len(unique_nodes)

        for i, source_node in enumerate(unique_nodes):
            if i % 100 == 0:
                print(f"   {i + 1}/{total}...", end="\r")
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    G, source_node, cutoff=max_time, weight="time_min"
                )
            except Exception:
                continue

            # Map node distances back to hex distances
            for dest_node, dist_min in lengths.items():
                for dest_hex in node_to_hexes.get(dest_node, []):
                    for src_hex in node_to_hexes.get(source_node, []):
                        if dist_min < travel_times[src_hex].get(dest_hex, float("inf")):
                            travel_times[src_hex][dest_hex] = dist_min

        self._travel_times = travel_times
        n_pairs = sum(len(v) for v in travel_times.values())
        print(f"\n   ✓ {n_pairs:,} hex-pairs within {max_time} min")

    # ------------------------------------------------------------------
    # Step 2: compute accessibility
    # ------------------------------------------------------------------

    def compute_accessibility(
        self,
        beta: float = 0.08,
        income_data: Optional[pd.Series] = None,
        mode_cost: float = 3.94,
        per_mode_beta: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Compute Hansen accessibility score for each hex.

        Parameters
        ----------
        beta         : global decay coefficient (used unless per_mode_beta set)
        income_data  : median household income by hex_id (for cost-of-time)
        mode_cost    : average trip cost in USD (for cost adjustment)
        per_mode_beta: optional dict {mode: beta} to override per mode
        """
        if self._travel_times is None:
            raise RuntimeError("Call compute_travel_times() first.")

        if self.mass._composite is None:
            self.mass.compute_composite_mass()

        mass_values = self.mass._composite.to_dict()

        # Build income-based effective time penalty per hex.
        # Lower income -> travel cost is a larger fraction of hourly wage
        # -> adds effective minutes to each trip (cost-of-time approach).
        # t_effective = t_travel + (mode_cost / hourly_wage) * 60
        # This means poorer hexes face genuinely higher effective impedance
        # to every destination, not just a flat output scaling.
        time_penalty: Dict[str, float] = {}
        if income_data is not None:
            median_wage = self.net.median_hourly_wage
            for hx in self.hex_ids:
                inc = income_data.get(hx) if isinstance(income_data, dict) \
                      else income_data.get(hx, np.nan)
                if pd.isna(inc) or inc <= 0:
                    hourly = median_wage
                else:
                    hourly = max(inc / 2080, 7.25)  # annual -> hourly, min wage floor
                # Extra minutes of effective travel time due to cost burden
                # Capped at 30 min so very low incomes don't zero out all access
                time_penalty[hx] = min(mode_cost / hourly * 60, 30.0)
        else:
            time_penalty = {hx: 0.0 for hx in self.hex_ids}

        scores: Dict[str, float] = {}
        for src_hex, dest_dict in self._travel_times.items():
            acc = 0.0
            extra_t = time_penalty.get(src_hex, 0.0)
            for dest_hex, t_min in dest_dict.items():
                m = mass_values.get(dest_hex, 0.0)
                if m <= 0 or t_min <= 0:
                    continue
                # Effective travel time includes cost-of-time penalty
                t_eff = t_min + extra_t
                acc += m * np.exp(-beta * t_eff)
            scores[src_hex] = acc

        result = pd.Series(scores).reindex(self.hex_ids).fillna(0)
        self._accessibility = result
        return result


# ---------------------------------------------------------------------------
# PCI Calculator
# ---------------------------------------------------------------------------

class TopographicPCICalculator:
    """
    Converts Hansen accessibility into a normalised 0-100 PCI score
    with an optional active-streets bonus.

    PCI_i = Normalise( A_i × (1 + λ × ActiveStreet_i) )
    """

    def __init__(
        self,
        grid: HexGrid,
        hansen_model: HansenAccessibilityModel,
        mass_calculator: MassCalculator,
    ):
        self.grid  = grid
        self.hansen = hansen_model
        self.mass   = mass_calculator
        self.hex_ids = grid.gdf["hex_id"].tolist()
        self._pci: Optional[pd.Series] = None

    def compute_pci(
        self,
        active_lambda: float = 0.30,
        mask_parks: bool = False,
        park_threshold: float = 0.90,
    ) -> pd.Series:
        """
        Compute final PCI scores — matches notebook implementation exactly.

        Steps (same order as notebook):
        1. Normalise accessibility to [0, 100]
        2. Compute active street score (food_retail percentile rank)
        3. Apply multiplier: PCI_raw = acc_norm * (1 + lambda * active)
        4. Re-normalise to [0, 100]
        5. Optionally mask park-dominated hexes
        """
        if self.hansen._accessibility is None:
            raise RuntimeError("Call hansen_model.compute_accessibility() first.")

        # Step 1: normalise accessibility to [0, 100] first (notebook order)
        acc = self.hansen._accessibility.copy()
        if acc.max() > acc.min():
            acc_norm = (acc - acc.min()) / (acc.max() - acc.min()) * 100
        else:
            acc_norm = pd.Series(50.0, index=self.hex_ids)

        # Step 2 & 3: active street bonus
        if active_lambda > 0:
            if "active_street_score" in self.grid.gdf.columns:
                active = self.grid.gdf.set_index("hex_id")["active_street_score"]
                active = active.reindex(self.hex_ids).fillna(0)
            else:
                active = self._compute_active_street_score()
                self.grid.attach_data(active, "active_street_score")
            pci_raw = acc_norm * (1 + active_lambda * active)
        else:
            pci_raw = acc_norm

        # Step 4: re-normalise to [0, 100]
        if pci_raw.max() > pci_raw.min():
            pci = (pci_raw - pci_raw.min()) / (pci_raw.max() - pci_raw.min()) * 100
        else:
            pci = pd.Series(50.0, index=self.hex_ids)

        # Step 5: park masking
        if mask_parks and "parks" in self.mass.layers:
            park_area = self.mass.layers["parks"].raw_values
            hex_area = self.grid.gdf.set_index("hex_id")["area_m2"]
            coverage = (park_area / hex_area.reindex(park_area.index)).fillna(0)
            park_mask = coverage > park_threshold
            pci = pci.where(~park_mask, np.nan)

        self._pci = pci
        return pci

    def _compute_active_street_score(self) -> pd.Series:
        """
        Active street score from food_retail + community amenity density,
        expressed as percentile rank [0, 1] — matches notebook implementation.
        Food retail fetched fresh from OSM (not from mass layers) so it
        captures street-level activity without polluting the PCI mass surface.
        """
        import osmnx as ox
        from shapely.ops import unary_union

        active = pd.Series(0.0, index=self.hex_ids)
        hex_gdf = self.grid.gdf[["hex_id", "geometry"]].copy()

        food_tags = {
            "amenity": ["restaurant", "cafe", "fast_food", "bar", "pub", "food_court"],
            "shop": ["bakery", "butcher", "greengrocer", "supermarket", "convenience"],
        }

        try:
            boundary = unary_union(self.grid.gdf.geometry)
            food_gdf = ox.features_from_polygon(boundary, tags=food_tags)
            if len(food_gdf) > 0:
                food_gdf = food_gdf.to_crs("EPSG:4326")
                food_gdf["geometry"] = food_gdf.geometry.centroid
                joined = gpd.sjoin(food_gdf[["geometry"]], hex_gdf, how="left", predicate="within")
                unassigned = joined["hex_id"].isna()
                if unassigned.any():
                    nearest = gpd.sjoin_nearest(food_gdf.loc[unassigned, ["geometry"]], hex_gdf, how="left")
                    joined.loc[unassigned, "hex_id"] = nearest["hex_id"].values
                food_count = joined.groupby("hex_id").size()
                food_count = food_count.reindex(self.hex_ids, fill_value=0).astype(float)
                active += food_count
        except Exception as e:
            # Fall back to network intersection degree if OSM fetch fails
            try:
                G = self.hansen.net.networks.get("walk")
                if G is not None:
                    centroids = self.grid.gdf.copy()
                    centroids["cx"] = centroids.geometry.centroid.x
                    centroids["cy"] = centroids.geometry.centroid.y
                    for _, row in centroids.iterrows():
                        try:
                            node = ox.nearest_nodes(G, row["cx"], row["cy"])
                            active[row["hex_id"]] = float(G.degree(node))
                        except Exception:
                            pass
            except Exception:
                pass

        # Percentile rank [0, 1] — matches notebook's rank(pct=True)
        return active.rank(pct=True)

    def compute_city_summary(self) -> dict:
        """Return a dict of city-wide statistics."""
        if self._pci is None:
            raise RuntimeError("Call compute_pci() first.")
        valid = self._pci.dropna()
        grid_valid = self.grid.gdf[self.grid.gdf["PCI"].notna()] \
            if "PCI" in self.grid.gdf.columns else self.grid.gdf
        total_area = grid_valid["area_m2"].sum() if "area_m2" in grid_valid.columns else np.nan
        weighted_pci = (
            (grid_valid["PCI"] * grid_valid["area_m2"]).sum() / total_area
            if not np.isnan(total_area) else valid.mean()
        )
        return {
            "city_pci":    round(float(weighted_pci), 2),
            "mean":        round(float(valid.mean()), 2),
            "median":      round(float(valid.median()), 2),
            "std":         round(float(valid.std()), 2),
            "p25":         round(float(valid.quantile(0.25)), 2),
            "p75":         round(float(valid.quantile(0.75)), 2),
            "gini":        round(float(self._gini(valid)), 3),
            "n_hexagons":  int(len(valid)),
            "area_km2":    round(float(total_area / 1e6), 2) if not np.isnan(total_area) else None,
        }

    @staticmethod
    def _gini(x: pd.Series) -> float:
        arr = np.sort(x.dropna().values)
        n = len(arr)
        if n == 0 or arr.sum() == 0:
            return 0.0
        idx = np.arange(1, n + 1)
        return float((2 * np.dot(idx, arr) / (n * arr.sum())) - (n + 1) / n)