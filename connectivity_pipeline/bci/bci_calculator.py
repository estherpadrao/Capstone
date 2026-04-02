"""
File 12: BCI Calculator
========================
Computes Hansen accessibility for all three BCI components
(market, labour, supplier) using component-specific networks,
then combines them into the final BCI score.

BCI_i = f(A_market_i, A_labour_i, A_supplier_i)

Supports:
  - weight-free combination  (BCI = sum of normalised A's)
  - weighted combination     (BCI = Σ w_k × A_k)
  - urban interface bonus    (BCI × (1 + λ × UrbanInterface_i))
  - per-component beta       (separate distance decay per mass type)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple

from core.h3_helper import HexGrid
from core.network_builder import MultiModalNetworkBuilder
from bci.bci_masses import BCIMassCalculator, MarketMassCalculator, LabourMassCalculator, SupplierMassCalculator


# ---------------------------------------------------------------------------
# Component-specific network configuration
# ---------------------------------------------------------------------------

# Which modal networks each BCI component uses
DEFAULT_NETWORK_CONFIG: Dict[str, List[str]] = {
    "market":   ["walk", "transit"],    # customers arrive on foot / transit
    "labour":   ["drive", "transit"],   # workers commute by car / transit
    "supplier": ["drive"],              # deliveries / logistics by car
}

DEFAULT_BETA_PARAMS: Dict[str, float] = {
    "market":   0.12,   # customers more distance-sensitive
    "labour":   0.05,   # workers tolerate longer commutes
    "supplier": 0.10,   # intermediate sensitivity
}


# ---------------------------------------------------------------------------
# Hansen accessibility for BCI
# ---------------------------------------------------------------------------

class BCIHansenAccessibility:
    """
    Computes Hansen gravity accessibility for each BCI component
    on its own modal sub-graph.

    A_k_i = Σ_j  M_k_j × exp(-β_k × t_ij^k)

    where t_ij^k is the travel time on the network for component k.
    """

    def __init__(
        self,
        grid: HexGrid,
        network_builder: MultiModalNetworkBuilder,
        beta_params: Optional[Dict[str, float]] = None,
        network_config: Optional[Dict[str, List[str]]] = None,
    ):
        self.grid    = grid
        self.net     = network_builder
        self.hex_ids = grid.gdf["hex_id"].tolist()
        self.beta    = {**DEFAULT_BETA_PARAMS, **(beta_params or {})}
        self.net_config = {**DEFAULT_NETWORK_CONFIG, **(network_config or {})}

        # Computed attributes
        self._component_graphs: Dict[str, nx.MultiDiGraph] = {}
        # Per-component spatial indices (built in build_component_graphs)
        self._kdtrees:   Dict[str, cKDTree] = {}
        self._node_ids:  Dict[str, list]    = {}
        self._hex_to_node: Dict[str, Dict[str, int]] = {}   # component → {hex_id: node}
        self._travel_times: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.accessibility: Dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # Step 1 — Build component sub-graphs + per-component KD-trees
    # ------------------------------------------------------------------

    def build_component_graphs(self):
        """
        Merge the required modal networks for each component into a single
        graph, then build a component-specific spatial KD-tree so that hexes
        are mapped to the nearest node **within that component's graph**
        (matches notebook BCINetworkBuilder._build_component_networks exactly).
        """
        for component, modes in self.net_config.items():
            G = nx.MultiDiGraph()
            for mode in modes:
                src = self.net.networks.get(mode)
                if src is None:
                    print(f"   ⚠  {mode} network missing for {component}")
                    continue
                for u, v, k, data in src.edges(data=True, keys=True):
                    G.add_edge(u, v, **{**data, "mode": mode})
                for node, ndata in src.nodes(data=True):
                    if node not in G.nodes:
                        G.add_node(node, **ndata)
                    else:
                        G.nodes[node].update(ndata)
            self._component_graphs[component] = G

            # Build a KD-tree over THIS component graph's nodes
            nodes    = list(G.nodes(data=True))
            node_ids = [n[0] for n in nodes]
            coords   = np.array([(d.get("y", 0), d.get("x", 0)) for _, d in nodes])
            self._node_ids[component]  = node_ids
            self._kdtrees[component]   = cKDTree(coords) if len(coords) else None

            print(f"   ✓ {component} graph: {G.number_of_nodes():,} nodes "
                  f"({', '.join(modes)})")

    # ------------------------------------------------------------------
    # Step 2 — Map hexes to nodes using component-specific KD-tree
    # ------------------------------------------------------------------

    def _map_hexes_to_nodes(self, component: str):
        """
        Map each hex centroid to its nearest node in the component graph.
        Uses a component-specific spatial index, matching notebook
        BCIHansenAccessibility._compute_travel_times_for_component().
        """
        if component in self._hex_to_node:
            return

        kdt      = self._kdtrees.get(component)
        node_ids = self._node_ids.get(component)
        if kdt is None or not node_ids:
            self._hex_to_node[component] = {}
            return

        centroids  = self.grid.centroids
        hex_ids    = centroids["hex_id"].tolist()
        hex_coords = np.array(
            [(r.geometry.y, r.geometry.x) for _, r in centroids.iterrows()]
        )
        _, idxs  = kdt.query(hex_coords)
        nearest  = [node_ids[i] for i in idxs]
        self._hex_to_node[component] = dict(zip(hex_ids, nearest))
        print(f"      ✓ Mapped {len(hex_ids)} hexes to {component} network nodes")

    # ------------------------------------------------------------------
    # Step 3 — Compute travel times for one component
    # ------------------------------------------------------------------

    def compute_travel_times_for(self, component: str, max_time: float = 120.0):
        """
        Compute shortest-path travel times for one BCI component.

        Matches notebook BCIHansenAccessibility._compute_travel_times_for_component:
        - Iterates each unique source node
        - Runs single-source Dijkstra with cutoff
        - Maps node distances directly to hex distances via hex_to_node mapping
        """
        print(f"   ⏱  {component} travel times (max {max_time} min)...")
        self._map_hexes_to_nodes(component)
        hex_to_node = self._hex_to_node.get(component, {})

        G = self._component_graphs.get(component)
        if G is None or not hex_to_node:
            self._travel_times[component] = {}
            return

        unique_nodes = list(set(hex_to_node.values()))
        travel_times: Dict[str, Dict[str, float]] = {}

        for i, source_node in enumerate(unique_nodes):
            if i % 100 == 0:
                print(f"      {i + 1}/{len(unique_nodes)}...", end="\r")
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    G, source_node, cutoff=max_time, weight="time_min"
                )
            except Exception:
                continue

            # All hexes mapped to this source node share the same origin distances
            source_hexes = [h for h, n in hex_to_node.items() if n == source_node]
            for src_hex in source_hexes:
                travel_times[src_hex] = {}
                for dest_hex, dest_node in hex_to_node.items():
                    if dest_node in lengths:
                        travel_times[src_hex][dest_hex] = lengths[dest_node]

        self._travel_times[component] = travel_times
        n = sum(len(v) for v in travel_times.values())
        print(f"\n      ✓ {n:,} hex-pairs reachable")

    def compute_all_travel_times(self, max_time: float = 90.0):
        for comp in self.net_config:
            self.compute_travel_times_for(comp, max_time)

    # ------------------------------------------------------------------
    # Step 4 — Hansen accessibility per component
    # ------------------------------------------------------------------

    def compute_accessibility_for(
        self,
        component: str,
        mass_values: pd.Series,
    ) -> pd.Series:
        """
        A_k_i = Σ_j M_k_j × exp(-β_k × t_ij)

        Parameters
        ----------
        component   : "market" | "labour" | "supplier"
        mass_values : normalised mass Series indexed by hex_id
        """
        beta  = self.beta.get(component, 0.08)
        tt    = self._travel_times.get(component, {})
        mass  = mass_values.to_dict()
        scores = {}

        for src_hex in self.hex_ids:
            acc = 0.0
            for dest_hex, t in tt.get(src_hex, {}).items():
                m = mass.get(dest_hex, 0.0)
                if m > 0 and t >= 0:
                    acc += m * np.exp(-beta * t)
            scores[src_hex] = acc

        result = pd.Series(scores).reindex(self.hex_ids).fillna(0)
        self.accessibility[component] = result
        return result

    def compute_all_accessibility(
        self,
        market_mass: pd.Series,
        labour_mass: pd.Series,
        supplier_mass: pd.Series,
    ) -> Dict[str, pd.Series]:
        self.compute_accessibility_for("market",   market_mass)
        self.compute_accessibility_for("labour",   labour_mass)
        self.compute_accessibility_for("supplier", supplier_mass)
        return self.accessibility

    # ------------------------------------------------------------------
    # Normalisation utility
    # ------------------------------------------------------------------

    @staticmethod
    def normalise(series: pd.Series) -> pd.Series:
        """Divide by max so scores are in [0, 1]."""
        mx = series.max()
        if mx == 0:
            return series.copy()
        return series / mx


# ---------------------------------------------------------------------------
# Final BCI Calculator
# ---------------------------------------------------------------------------

class BCICalculator:
    """
    Combines the three accessibility components into the final BCI score.

    Modes:
    -------
    weight_free : BCI = (A_market/max + A_labour/max + A_supplier/max)
    weighted    : BCI = w_m×A_m + w_l×A_l + w_s×A_s  (weights sum to 1)

    Urban interface bonus (optional):
        BCI_final = BCI × (1 + λ × UrbanInterface_i)

    Score is normalised to [0, 100] at the end.
    """

    def __init__(
        self,
        grid: HexGrid,
        hansen_model: BCIHansenAccessibility,
        market_calc,          # BCIMassCalculator | MarketMassCalculator
        labour_calc=None,     # LabourMassCalculator (optional when unified calc used)
        supplier_calc=None,   # SupplierMassCalculator (optional when unified calc used)
    ):
        self.grid   = grid
        self.hansen = hansen_model
        # Accept either the unified BCIMassCalculator or the three separate shims
        if isinstance(market_calc, BCIMassCalculator):
            self._mass_calc = market_calc
            self.market   = market_calc
            self.labour   = market_calc
            self.supplier = market_calc
        else:
            self._mass_calc = None
            self.market   = market_calc
            self.labour   = labour_calc
            self.supplier = supplier_calc
        self.hex_ids  = grid.gdf["hex_id"].tolist()
        self._bci: Optional[pd.Series] = None
        self.components: Dict[str, pd.Series] = {}

    def compute_bci(
        self,
        method: str = "weight_free",
        market_weight:   float = 0.40,
        labour_weight:   float = 0.35,
        supplier_weight: float = 0.25,
        use_interface:   bool  = False,
        interface_lambda: float = 0.15,
    ) -> pd.Series:
        """
        Parameters
        ----------
        method          : "weight_free" or "weighted"
        market_weight   : used only when method="weighted"
        labour_weight   : used only when method="weighted"
        supplier_weight : used only when method="weighted"
        use_interface   : apply urban interface bonus
        interface_lambda: weight of the bonus (λ)
        """
        A = self.hansen.accessibility
        if not A:
            raise RuntimeError("Call hansen_model.compute_all_accessibility() first.")

        Am = BCIHansenAccessibility.normalise(A["market"])
        Al = BCIHansenAccessibility.normalise(A["labour"])
        As = BCIHansenAccessibility.normalise(A["supplier"])

        self.components = {"A_market_norm": Am, "A_labour_norm": Al, "A_supplier_norm": As}

        if method == "weight_free":
            bci_raw = Am + Al + As
        elif method == "weighted":
            tw = market_weight + labour_weight + supplier_weight
            bci_raw = (
                (market_weight / tw)   * Am
                + (labour_weight / tw) * Al
                + (supplier_weight / tw) * As
            )
        else:
            raise ValueError(f"Unknown BCI method: {method}. Use 'weight_free' or 'weighted'.")

        # Urban interface bonus — works with both unified and shim calculators
        _ui = (
            self._mass_calc._urban_interface
            if self._mass_calc is not None
            else getattr(self.supplier, "urban_interface", None)
        )
        if use_interface and _ui is not None:
            ui = _ui.reindex(self.hex_ids).fillna(0)
            bci_raw = bci_raw * (1 + interface_lambda * ui)
            self.components["urban_interface"] = ui

        # Normalise to [0, 100]
        lo, hi = bci_raw.min(), bci_raw.max()
        if hi > lo:
            bci = (bci_raw - lo) / (hi - lo) * 100
        else:
            bci = bci_raw * 0

        self._bci = bci.reindex(self.hex_ids)
        return self._bci

    # ------------------------------------------------------------------
    # Peak / valley identification
    # ------------------------------------------------------------------

    def identify_peaks(self, percentile: float = 90) -> gpd.GeoDataFrame:
        if self._bci is None:
            raise RuntimeError("Call compute_bci() first.")
        thresh = self._bci.quantile(percentile / 100)
        mask   = self.grid.gdf["hex_id"].map(self._bci) >= thresh
        return self.grid.gdf[mask].copy()

    def identify_valleys(self, percentile: float = 10) -> gpd.GeoDataFrame:
        if self._bci is None:
            raise RuntimeError("Call compute_bci() first.")
        thresh = self._bci.quantile(percentile / 100)
        mask   = self.grid.gdf["hex_id"].map(self._bci) <= thresh
        return self.grid.gdf[mask].copy()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if self._bci is None:
            return {}
        valid = self._bci.dropna()

        am = self.components.get("A_market_norm", pd.Series(dtype=float))
        al = self.components.get("A_labour_norm", pd.Series(dtype=float))
        as_ = self.components.get("A_supplier_norm", pd.Series(dtype=float))

        return {
            "mean":              round(float(valid.mean()), 2),
            "median":            round(float(valid.median()), 2),
            "std":               round(float(valid.std()), 2),
            "min":               round(float(valid.min()), 2),
            "max":               round(float(valid.max()), 2),
            "n_hexagons":        int(len(valid)),
            "corr_market_bci":   round(float(valid.corr(am.reindex(valid.index))), 3),
            "corr_labour_bci":   round(float(valid.corr(al.reindex(valid.index))), 3),
            "corr_supplier_bci": round(float(valid.corr(as_.reindex(valid.index))), 3),
        }