"""
File 6: Multi-Modal Network Builder
=====================================
Builds walk / bike / drive / transit networks (with GTFS).
Networks are built once, cached, and reused by both PCI and BCI.
Income data is embedded in the graph for cost-of-time adjustments.
"""

import os
import math
import hashlib
import pickle
import zipfile
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
ox.settings.use_cache = True
ox.settings.log_console = False


# ---------------------------------------------------------------------------
# GTFS loader
# ---------------------------------------------------------------------------

class GTFSTransitLoader:
    """Parse a GTFS zip and expose stops + route segments as a graph."""

    def __init__(self, gtfs_path: str):
        self.gtfs_path = gtfs_path
        self.stops = None
        self.routes = None
        self.stop_times = None
        self.trips = None
        self.shapes = None

    def load(self):
        print("📂 Loading GTFS data...")
        with zipfile.ZipFile(self.gtfs_path, "r") as z:
            with z.open("stops.txt") as f:
                self.stops = pd.read_csv(f)
            with z.open("routes.txt") as f:
                self.routes = pd.read_csv(f)
            with z.open("trips.txt") as f:
                self.trips = pd.read_csv(f)
            with z.open("stop_times.txt") as f:
                self.stop_times = pd.read_csv(
                    f, usecols=["trip_id", "stop_id", "stop_sequence", "arrival_time"]
                )
            try:
                with z.open("shapes.txt") as f:
                    self.shapes = pd.read_csv(f)
            except Exception:
                self.shapes = None
        print(f"   ✓ {len(self.stops)} stops, "
              f"{len(self.routes)} routes, "
              f"{len(self.stop_times)} stop-times")

    def get_stops_gdf(self) -> gpd.GeoDataFrame:
        from shapely.geometry import Point
        gdf = gpd.GeoDataFrame(
            self.stops,
            geometry=[Point(r.stop_lon, r.stop_lat) for _, r in self.stops.iterrows()],
            crs="EPSG:4326",
        )
        return gdf

    def get_route_segments(self) -> pd.DataFrame:
        """Return consecutive stop pairs with travel time estimates."""
        merged = self.stop_times.merge(
            self.stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left"
        ).sort_values(["trip_id", "stop_sequence"])

        segments = []
        for trip_id, grp in merged.groupby("trip_id"):
            rows = grp.reset_index(drop=True)
            for i in range(len(rows) - 1):
                r0, r1 = rows.iloc[i], rows.iloc[i + 1]
                dist_km = self._haversine(
                    r0.stop_lat, r0.stop_lon, r1.stop_lat, r1.stop_lon
                )
                segments.append({
                    "from_stop": r0.stop_id,
                    "to_stop":   r1.stop_id,
                    "from_lat":  r0.stop_lat,
                    "from_lon":  r0.stop_lon,
                    "to_lat":    r1.stop_lat,
                    "to_lon":    r1.stop_lon,
                    "dist_km":   dist_km,
                })
        return pd.DataFrame(segments).drop_duplicates(subset=["from_stop", "to_stop"])

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> float:
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1))
             * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Multi-modal network builder
# ---------------------------------------------------------------------------

class MultiModalNetworkBuilder:
    """
    Builds and caches walk / bike / drive / transit networks.

    Networks are built once; travel_times are stamped on every edge
    using city-specific speeds and time penalties.

    Parameters
    ----------
    boundary_polygon : Shapely polygon (EPSG:4326)
    gtfs_path        : path to GTFS zip (optional)
    travel_speeds    : dict {mode: km/h}
    travel_costs     : dict {mode: USD per trip}
    time_penalties   : dict with transit_wait, parking_search, bike_unlock (min)
    median_hourly_wage : float (USD/hr) for cost-of-time conversion
    """

    def __init__(
        self,
        boundary_polygon,
        gtfs_path: Optional[str] = None,
        travel_speeds: Optional[Dict[str, float]] = None,
        travel_costs: Optional[Dict[str, float]] = None,
        time_penalties: Optional[Dict[str, float]] = None,
        median_hourly_wage: float = 46.07,
    ):
        self.boundary = boundary_polygon
        self.gtfs_path = gtfs_path
        self.median_hourly_wage = median_hourly_wage

        self.travel_speeds = travel_speeds or {
            "walk": 4.8, "bike": 15.0, "drive": 24.0, "transit": 13.0
        }
        self.travel_costs = travel_costs or {
            "walk": 0.0, "bike": 3.99, "drive": 9.00, "transit": 2.75
        }
        self.time_penalties = time_penalties or {
            "transit_wait":    9.0,
            "transit_board":   0.5,
            "parking_search":  13.0,
            "bike_unlock":     1.0,
        }

        self.networks: Dict[str, nx.MultiDiGraph] = {}
        self.unified_graph: Optional[nx.MultiDiGraph] = None
        self.transit_stops_gdf: Optional[gpd.GeoDataFrame] = None
        self._gtfs_loader: Optional[GTFSTransitLoader] = None

        # KD-trees for fast nearest-node lookup, keyed by mode
        self._kdtrees: Dict[str, cKDTree] = {}
        self._node_arrays: Dict[str, np.ndarray] = {}
        self._node_id_arrays: Dict[str, list] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_all_networks(self, cache_dir: Optional[str] = None) -> Dict[str, nx.MultiDiGraph]:
        """Build all modal networks and the unified graph.

        If *cache_dir* is given, a pre-built network pickle is loaded when
        available (skipping all OSM fetches).  On a cache miss the networks are
        built normally and then saved for future runs.
        """
        if cache_dir and self._try_load_cache(cache_dir):
            return self.networks

        self._build_walk()
        self._build_bike()
        self._build_drive()
        self._build_transit()
        self._build_unified()
        self._build_kdtrees()

        if cache_dir:
            self._save_cache(cache_dir)

        return self.networks

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _boundary_hash(self) -> str:
        """Short deterministic hash of the boundary polygon WKT."""
        return hashlib.md5(self.boundary.wkt.encode()).hexdigest()[:12]

    def _cache_path(self, cache_dir: str) -> str:
        return os.path.join(cache_dir, f"network_{self._boundary_hash()}.pkl")

    def _save_cache(self, cache_dir: str):
        os.makedirs(cache_dir, exist_ok=True)
        payload = {
            "networks":         self.networks,
            "unified_graph":    self.unified_graph,
            "_kdtrees":         self._kdtrees,
            "_node_arrays":     self._node_arrays,
            "_node_id_arrays":  self._node_id_arrays,
        }
        path = self._cache_path(cache_dir)
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"   💾 Network cached → {os.path.basename(path)}")

    def _try_load_cache(self, cache_dir: str) -> bool:
        path = self._cache_path(cache_dir)
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            self.networks        = payload["networks"]
            self.unified_graph   = payload["unified_graph"]
            self._kdtrees        = payload["_kdtrees"]
            self._node_arrays    = payload["_node_arrays"]
            self._node_id_arrays = payload["_node_id_arrays"]
            total_nodes = self.unified_graph.number_of_nodes() if self.unified_graph else 0
            print(f"   📦 Network loaded from cache ({total_nodes:,} nodes)")
            return True
        except Exception as exc:
            print(f"   ⚠  Network cache load failed ({exc}) — rebuilding")
            return False

    def get_nearest_node(self, lat: float, lng: float, mode: str = "walk") -> int:
        """Return nearest network node for a given (lat, lng) point."""
        tree = self._kdtrees.get(mode) or self._kdtrees.get("walk")
        node_ids = self._node_id_arrays.get(mode) or self._node_id_arrays.get("walk")
        if tree is None:
            raise RuntimeError("Networks not built yet. Call build_all_networks() first.")
        _, idx = tree.query([lat, lng])
        return node_ids[idx]

    def get_nearest_nodes_batch(
        self,
        coords: List[Tuple[float, float]],
        mode: str = "walk",
    ) -> List[int]:
        """Batch version of get_nearest_node."""
        tree = self._kdtrees.get(mode) or self._kdtrees.get("walk")
        node_ids = self._node_id_arrays.get(mode) or self._node_id_arrays.get("walk")
        _, idxs = tree.query(coords)
        return [node_ids[i] for i in idxs]

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_walk(self):
        print("🚶 Building walk network...")
        G = ox.graph_from_polygon(self.boundary, network_type="walk", simplify=True)
        self._stamp_travel_time(G, "walk")
        self.networks["walk"] = G
        print(f"   ✓ {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    def _build_bike(self):
        print("🚲 Building bike network...")
        try:
            G = ox.graph_from_polygon(self.boundary, network_type="bike", simplify=True)
        except Exception:
            G = self.networks.get("walk", ox.graph_from_polygon(
                self.boundary, network_type="walk", simplify=True
            ))
        self._stamp_travel_time(G, "bike", extra_min=self.time_penalties["bike_unlock"])
        self.networks["bike"] = G
        print(f"   ✓ {G.number_of_nodes():,} nodes")

    def _build_drive(self):
        print("🚗 Building drive network...")
        G = ox.graph_from_polygon(self.boundary, network_type="drive", simplify=True)
        self._stamp_travel_time(
            G, "drive", extra_min=self.time_penalties["parking_search"]
        )
        self.networks["drive"] = G
        print(f"   ✓ {G.number_of_nodes():,} nodes")

    def _build_transit(self):
        if self.gtfs_path and os.path.exists(self.gtfs_path):
            self._build_transit_gtfs()
        else:
            print("🚌 No GTFS file found — using walk network for transit")
            if "walk" in self.networks:
                G = self.networks["walk"].copy()
                self._stamp_travel_time(
                    G, "transit",
                    extra_min=self.time_penalties["transit_wait"]
                )
                self.networks["transit"] = G

    def _build_transit_gtfs(self):
        print("🚌 Building transit network from GTFS...")
        self._gtfs_loader = GTFSTransitLoader(self.gtfs_path)
        self._gtfs_loader.load()

        stops_gdf = self._gtfs_loader.get_stops_gdf()
        boundary_gdf = gpd.GeoDataFrame(geometry=[self.boundary], crs="EPSG:4326")
        self.transit_stops_gdf = gpd.sjoin(
            stops_gdf, boundary_gdf, how="inner", predicate="within"
        )
        print(f"   ✓ {len(self.transit_stops_gdf)} stops within boundary")

        # Start from walk graph, add transit edges
        G = self.networks.get("walk", nx.MultiDiGraph()).copy()
        segments = self._gtfs_loader.get_route_segments()

        walk_nodes = list(G.nodes(data=True))
        if walk_nodes:
            node_coords = np.array(
                [(d.get("y", 0), d.get("x", 0)) for _, d in walk_nodes]
            )
            node_ids_list = [n for n, _ in walk_nodes]
            tree = cKDTree(node_coords)

            transit_speed_ms = self.travel_speeds["transit"] * 1000 / 3600
            wait_min = self.time_penalties["transit_wait"]

            for _, seg in segments.iterrows():
                if seg.dist_km <= 0:
                    continue
                travel_min = (seg.dist_km / self.travel_speeds["transit"]) * 60
                _, fi = tree.query([seg.from_lat, seg.from_lon])
                _, ti = tree.query([seg.to_lat, seg.to_lon])
                u, v = node_ids_list[fi], node_ids_list[ti]
                if u != v:
                    G.add_edge(
                        u, v,
                        length=seg.dist_km * 1000,
                        time_min=travel_min + wait_min,
                        mode="transit",
                    )

        self.networks["transit"] = G
        print(f"   ✓ Transit graph: {G.number_of_nodes():,} nodes")

    def _build_unified(self):
        """Merge all modal networks into one graph."""
        print("🔗 Building unified multi-modal graph...")
        G_unified = nx.MultiDiGraph()
        for mode, G in self.networks.items():
            for u, v, k, data in G.edges(data=True, keys=True):
                G_unified.add_edge(u, v, **{**data, "mode": mode})
        # Copy node attributes
        for G in self.networks.values():
            for node, data in G.nodes(data=True):
                if node not in G_unified.nodes:
                    G_unified.add_node(node, **data)
                else:
                    G_unified.nodes[node].update(data)
        self.unified_graph = G_unified
        print(f"   ✓ Unified graph: {G_unified.number_of_nodes():,} nodes, "
              f"{G_unified.number_of_edges():,} edges")

    def _build_kdtrees(self):
        """Pre-build KD-trees for fast nearest-node lookup per mode."""
        for mode, G in self.networks.items():
            nodes = [(n, d) for n, d in G.nodes(data=True) if "x" in d and "y" in d]
            if not nodes:
                continue
            ids  = [n for n, _ in nodes]
            coords = np.array([(d["y"], d["x"]) for _, d in nodes])
            self._kdtrees[mode]       = cKDTree(coords)
            self._node_arrays[mode]   = coords
            self._node_id_arrays[mode] = ids

        # Also for unified
        G = self.unified_graph
        if G:
            nodes = [(n, d) for n, d in G.nodes(data=True) if "x" in d and "y" in d]
            if nodes:
                ids    = [n for n, _ in nodes]
                coords = np.array([(d["y"], d["x"]) for _, d in nodes])
                self._kdtrees["unified"]       = cKDTree(coords)
                self._node_id_arrays["unified"] = ids

    # ------------------------------------------------------------------
    # Edge weight stamping
    # ------------------------------------------------------------------

    def _stamp_travel_time(
        self,
        G: nx.MultiDiGraph,
        mode: str,
        extra_min: float = 0.0,
    ):
        """
        Stamp `time_min` on every edge.
        Extra minutes are added once per trip (origin penalty).
        We add extra_min / number_of_edges as a per-edge approximation.
        """
        speed_ms = self.travel_speeds[mode] * 1000 / 3600
        n_edges  = max(G.number_of_edges(), 1)
        per_edge_penalty = extra_min / n_edges  # distribute origin penalty

        for u, v, k, data in G.edges(data=True, keys=True):
            length = data.get("length", 100)
            travel_min = (length / speed_ms) / 60
            G[u][v][k]["time_min"] = travel_min + per_edge_penalty
            G[u][v][k]["mode"] = mode

    # ------------------------------------------------------------------
    # Network diagnostics
    # ------------------------------------------------------------------

    def validate(self) -> Dict:
        """Return a dict of diagnostic statistics for the unified graph."""
        G = self.unified_graph
        if G is None:
            return {"error": "Unified graph not built yet."}

        components = list(nx.weakly_connected_components(G))
        time_vals  = [
            d["time_min"]
            for _, _, d in G.edges(data=True)
            if "time_min" in d
        ]
        mode_counts: Dict[str, int] = {}
        for _, _, d in G.edges(data=True):
            m = d.get("mode", "unknown")
            mode_counts[m] = mode_counts.get(m, 0) + 1

        return {
            "nodes":            G.number_of_nodes(),
            "edges":            G.number_of_edges(),
            "is_weakly_connected": nx.is_weakly_connected(G),
            "n_components":     len(components),
            "largest_component": max(len(c) for c in components),
            "time_min_mean":    float(np.mean(time_vals)) if time_vals else 0.0,
            "time_min_max":     float(np.max(time_vals)) if time_vals else 0.0,
            "edges_by_mode":    mode_counts,
        }
