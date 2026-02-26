"""
File 2: OSM Data Fetcher
========================
Fetches amenity and supplier data from OpenStreetMap.
Each tag category can be toggled on/off at runtime.
Used by both PCI (amenities) and BCI (suppliers).
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from typing import Dict, List, Optional, Set
import warnings

warnings.filterwarnings("ignore")
ox.settings.use_cache = True
ox.settings.log_console = False

# ---------------------------------------------------------------------------
# PCI OSM tags — each top-level key can be toggled on/off
# ---------------------------------------------------------------------------

PCI_OSM_TAGS: Dict[str, dict] = {
    "health": {
        "amenity": ["hospital", "clinic", "doctors", "dentist", "pharmacy"],
    },
    "education": {
        "amenity": ["school", "university", "college", "library",
                    "kindergarten", "childcare", "prep_school"],
    },
    "parks": {
        "leisure": ["park", "garden", "playground", "nature_reserve"],
        "landuse": ["recreation_ground", "grass"],
    },
    "transit": {
        "highway": ["bus_stop"],
        "railway": ["station", "halt", "tram_stop", "subway_entrance"],
        "amenity": ["bus_station", "ferry_terminal"],
    },
    "food_retail": {
        "amenity": ["restaurant", "cafe", "bar", "pub", "fast_food"],
        "shop": ["supermarket", "convenience", "grocery"],
    },
    "community": {
        "amenity": ["community_centre", "social_facility", "place_of_worship",
                    "theatre", "cinema", "arts_centre"],
    },
}

# BCI supplier tags
BCI_SUPPLIER_TAGS: Dict[str, dict] = {
    "offices": {"office": True},
    "industrial_commercial": {"landuse": ["industrial", "commercial"]},
    "commercial_buildings": {
        "building": ["commercial", "industrial", "warehouse", "office"]
    },
    "wholesale": {"shop": ["wholesale", "trade", "hardware", "electronics"]},
    "finance": {"amenity": ["bank", "bureau_de_change", "post_office"]},
}


# ---------------------------------------------------------------------------
# PCI amenity fetcher
# ---------------------------------------------------------------------------

class OSMDataFetcher:
    """
    Fetches PCI amenity layers from OpenStreetMap.

    Tag categories can be toggled via `enabled_tags`:
        fetcher = OSMDataFetcher(boundary)
        fetcher.set_enabled_tags({"health": True, "parks": False, ...})
    """

    def __init__(self, boundary):
        """
        Parameters
        ----------
        boundary : Shapely polygon (EPSG:4326) or GeoDataFrame
        """
        if isinstance(boundary, gpd.GeoDataFrame):
            from shapely.ops import unary_union
            boundary = unary_union(boundary.to_crs(4326).geometry)
        self.boundary = boundary
        self._cache: Dict[str, Optional[gpd.GeoDataFrame]] = {}
        # All tags enabled by default
        self.enabled_tags: Dict[str, bool] = {k: True for k in PCI_OSM_TAGS}

    def set_enabled_tags(self, tags: Dict[str, bool]):
        """
        Enable or disable individual tag categories.

        Example:
            fetcher.set_enabled_tags({"parks": False, "food_retail": False})
        """
        for key, val in tags.items():
            if key in self.enabled_tags:
                self.enabled_tags[key] = bool(val)
            else:
                raise KeyError(f"Unknown tag category '{key}'. "
                               f"Valid: {list(self.enabled_tags.keys())}")

    def fetch_all(self) -> Dict[str, Optional[gpd.GeoDataFrame]]:
        """Fetch all enabled amenity categories (cached per category)."""
        amenities = {}
        for name in PCI_OSM_TAGS:
            if not self.enabled_tags.get(name, True):
                print(f"   ⏭  {name}: disabled")
                amenities[name] = None
                continue
            amenities[name] = self.fetch_category(name)
        return amenities

    def fetch_category(self, name: str) -> Optional[gpd.GeoDataFrame]:
        """Fetch a single amenity category (result cached in memory)."""
        if name in self._cache:
            return self._cache[name]

        tags = PCI_OSM_TAGS.get(name)
        if tags is None:
            raise KeyError(f"Unknown amenity category: {name}")

        try:
            gdf = ox.features_from_polygon(self.boundary, tags=tags)
            if gdf is None or len(gdf) == 0:
                print(f"   ⚠  {name}: no features found")
                self._cache[name] = None
                return None

            gdf = gdf.to_crs("EPSG:4326")
            print(f"   ✓  {name}: {len(gdf)} features")
            self._cache[name] = gdf
            return gdf

        except Exception as exc:
            print(f"   ⚠  {name}: failed ({exc})")
            self._cache[name] = None
            return None

    def clear_cache(self, name: Optional[str] = None):
        """Clear cached results for one or all categories."""
        if name:
            self._cache.pop(name, None)
        else:
            self._cache.clear()

    @property
    def available_tags(self) -> List[str]:
        return list(PCI_OSM_TAGS.keys())


# ---------------------------------------------------------------------------
# BCI supplier fetcher
# ---------------------------------------------------------------------------

class SupplierDataFetcher:
    """
    Fetches business/commercial data from OSM for BCI supplier mass.
    Supports the same toggle interface as OSMDataFetcher.
    """

    def __init__(self, boundary):
        if isinstance(boundary, gpd.GeoDataFrame):
            from shapely.ops import unary_union
            boundary = unary_union(boundary.to_crs(4326).geometry)
        self.boundary = boundary
        self._cache: Dict[str, Optional[gpd.GeoDataFrame]] = {}
        self.enabled_tags: Dict[str, bool] = {k: True for k in BCI_SUPPLIER_TAGS}

    def set_enabled_tags(self, tags: Dict[str, bool]):
        for key, val in tags.items():
            if key in self.enabled_tags:
                self.enabled_tags[key] = bool(val)
            else:
                raise KeyError(f"Unknown supplier tag '{key}'. "
                               f"Valid: {list(self.enabled_tags.keys())}")

    def fetch_suppliers(self) -> gpd.GeoDataFrame:
        """Fetch all enabled supplier categories and combine into one GDF."""
        all_gdfs = []
        for name, tags in BCI_SUPPLIER_TAGS.items():
            if not self.enabled_tags.get(name, True):
                continue
            gdf = self._fetch_one(name, tags)
            if gdf is not None and len(gdf) > 0:
                gdf["supplier_type"] = name
                all_gdfs.append(gdf)

        if not all_gdfs:
            print("   ⚠  No supplier data fetched")
            return gpd.GeoDataFrame(columns=["geometry", "supplier_type"], crs="EPSG:4326")

        combined = pd.concat(all_gdfs, ignore_index=True)
        return gpd.GeoDataFrame(combined, crs="EPSG:4326")

    def _fetch_one(self, name: str, tags: dict) -> Optional[gpd.GeoDataFrame]:
        if name in self._cache:
            return self._cache[name]
        try:
            gdf = ox.features_from_polygon(self.boundary, tags=tags)
            if gdf is not None and len(gdf) > 0:
                gdf = gdf.to_crs("EPSG:4326")
                print(f"   ✓  {name}: {len(gdf)} features")
            else:
                gdf = None
            self._cache[name] = gdf
            return gdf
        except Exception as exc:
            print(f"   ⚠  {name}: failed ({exc})")
            self._cache[name] = None
            return None

    def compute_supplier_density(
        self,
        hex_gdf: gpd.GeoDataFrame,
        suppliers_gdf: gpd.GeoDataFrame
    ) -> pd.Series:
        """Count supplier features per hexagon (using centroid join)."""
        if suppliers_gdf is None or len(suppliers_gdf) == 0:
            return pd.Series(0.0, index=hex_gdf["hex_id"])

        pts = suppliers_gdf.copy()
        pts["geometry"] = pts.geometry.centroid

        joined = gpd.sjoin(pts, hex_gdf[["hex_id", "geometry"]], how="left", predicate="within")
        counts = joined.groupby("hex_id").size()
        return counts.reindex(hex_gdf["hex_id"]).fillna(0)
