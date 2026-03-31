"""
File 4: Census Data Fetcher
============================
Fetches ACS 5-year data and TIGER tract geometries from the US Census Bureau.
Used by both PCI (income/cost adjustment) and BCI (population, labour, income).
"""

import os
import hashlib
import pickle
import requests
import tempfile
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Optional, Dict
from core.h3_helper import HexGrid


# Census codes that represent suppressed / missing data
CENSUS_MISSING_VALUES = [-666666666, -999999999, -888888888, -222222222]

# Standard ACS variable identifiers
CENSUS_VARS = {
    "median_income":  "B19013_001E",   # Median household income
    "population":     "B01003_001E",   # Total population
    "labour":         "B23025_004E",   # Employed population 16+
}


class CensusDataFetcher:
    """
    Downloads Census ACS data + TIGER tract geometries and spatially
    joins them onto the H3 hex grid.

    Parameters
    ----------
    year        : ACS 5-year survey year (e.g. 2022)
    state_fips  : 2-digit state FIPS code   (e.g. "06" = California)
    county_fips : 3-digit county FIPS code  (e.g. "075" = San Francisco)
    api_key     : optional Census API key (improves rate limits)
    """

    def __init__(
        self,
        year: int,
        state_fips: str,
        county_fips: str,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.year = year
        self.state_fips = str(state_fips).zfill(2)
        self.county_fips = str(county_fips).zfill(3)
        self.api_key = api_key
        self._cache_dir = cache_dir
        self._cache_key = f"{year}_{self.state_fips}_{self.county_fips}"
        self._tracts_gdf: Optional[gpd.GeoDataFrame] = None
        self._acs_df: Optional[pd.DataFrame] = None
        self._merged_gdf: Optional[gpd.GeoDataFrame] = None

    # ------------------------------------------------------------------
    # ACS data
    # ------------------------------------------------------------------

    def fetch_acs_data(self, variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch ACS 5-year estimates for the specified variables.

        Parameters
        ----------
        variables : list of ACS variable codes; defaults to all CENSUS_VARS
        """
        if variables is None:
            variables = list(CENSUS_VARS.values())

        # Disk cache check
        if self._cache_dir:
            path = os.path.join(self._cache_dir, f"acs_{self._cache_key}.pkl")
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        df = pickle.load(f)
                    print(f"   📦 ACS data loaded from cache ({len(df)} tracts)")
                    self._acs_df = df
                    return df
                except Exception as exc:
                    print(f"   ⚠  ACS cache load failed ({exc}) — re-fetching")

        base_url = f"https://api.census.gov/data/{self.year}/acs/acs5"
        params = {
            "get": "NAME," + ",".join(variables),
            "for": "tract:*",
            "in": f"state:{self.state_fips} county:{self.county_fips}",
        }
        if self.api_key:
            params["key"] = self.api_key

        print(f"📊 Fetching ACS {self.year} data "
              f"(state={self.state_fips}, county={self.county_fips})...")

        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()

        data = resp.json()
        df = pd.DataFrame(data[1:], columns=data[0])

        # Build 11-digit GEOID
        df["state"]  = df["state"].astype(str).str.zfill(2)
        df["county"] = df["county"].astype(str).str.zfill(3)
        df["tract"]  = df["tract"].astype(str).str.zfill(6)
        df["GEOID"]  = df["state"] + df["county"] + df["tract"]

        # Convert to numeric, replace Census missing codes
        for var in variables:
            if var in df.columns:
                df[var] = pd.to_numeric(df[var], errors="coerce")
                df[var] = df[var].replace(CENSUS_MISSING_VALUES, np.nan)
                df.loc[df[var] < 0, var] = np.nan
                valid = df[var].notna().sum()
                print(f"   {var}: {valid}/{len(df)} valid tracts")

        self._acs_df = df

        if self._cache_dir:
            try:
                os.makedirs(self._cache_dir, exist_ok=True)
                path = os.path.join(self._cache_dir, f"acs_{self._cache_key}.pkl")
                with open(path, "wb") as f:
                    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:
                print(f"   ⚠  Could not save ACS data to disk cache: {exc}")

        return df

    # ------------------------------------------------------------------
    # TIGER geometries
    # ------------------------------------------------------------------

    def fetch_tiger_tracts(self) -> gpd.GeoDataFrame:
        """Download TIGER tract shapefile and return as GeoDataFrame."""
        if self._tracts_gdf is not None:
            return self._tracts_gdf

        # Disk cache check
        if self._cache_dir:
            path = os.path.join(self._cache_dir, f"tiger_{self._cache_key}.pkl")
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        gdf = pickle.load(f)
                    print(f"   📦 TIGER tracts loaded from cache ({len(gdf)} tracts)")
                    self._tracts_gdf = gdf
                    return gdf
                except Exception as exc:
                    print(f"   ⚠  TIGER cache load failed ({exc}) — re-downloading")

        print("🗺  Downloading TIGER tract geometries...")

        urls = [
            f"https://www2.census.gov/geo/tiger/TIGER{self.year}/TRACT/"
            f"tl_{self.year}_{self.state_fips}_tract.zip",
            f"https://www2.census.gov/geo/tiger/TIGER{self.year - 1}/TRACT/"
            f"tl_{self.year - 1}_{self.state_fips}_tract.zip",
        ]

        for url in urls:
            try:
                resp = requests.get(url, timeout=120, stream=True)
                resp.raise_for_status()
                with tempfile.TemporaryDirectory() as tmp:
                    zip_path = os.path.join(tmp, "tracts.zip")
                    with open(zip_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    with zipfile.ZipFile(zip_path, "r") as z:
                        z.extractall(tmp)
                    shp_files = [
                        os.path.join(tmp, fn)
                        for fn in os.listdir(tmp)
                        if fn.endswith(".shp")
                    ]
                    if not shp_files:
                        continue
                    gdf = gpd.read_file(shp_files[0]).to_crs("EPSG:4326")
                    # Filter to county
                    gdf = gdf[gdf["COUNTYFP"] == self.county_fips].copy()
                    self._tracts_gdf = gdf
                    print(f"   ✓ {len(gdf)} tracts loaded")
                    if self._cache_dir:
                        try:
                            os.makedirs(self._cache_dir, exist_ok=True)
                            cache_path = os.path.join(
                                self._cache_dir, f"tiger_{self._cache_key}.pkl"
                            )
                            with open(cache_path, "wb") as cf:
                                pickle.dump(gdf, cf, protocol=pickle.HIGHEST_PROTOCOL)
                        except Exception as exc:
                            print(f"   ⚠  Could not save TIGER data to disk cache: {exc}")
                    return gdf
            except Exception as exc:
                print(f"   ⚠  {url}: {exc}")

        raise RuntimeError("Could not download TIGER tract geometries.")

    # ------------------------------------------------------------------
    # Merge & assign to hexes
    # ------------------------------------------------------------------

    def get_tract_data(self) -> gpd.GeoDataFrame:
        """Return ACS data merged with TIGER geometries."""
        if self._merged_gdf is not None:
            return self._merged_gdf

        acs = self.fetch_acs_data()
        tiger = self.fetch_tiger_tracts()

        merged = tiger.merge(acs, on="GEOID", how="left")
        self._merged_gdf = merged
        return merged

    def assign_to_hexes(
        self,
        grid: HexGrid,
        variable: str,
        agg: str = "mean",
    ) -> pd.Series:
        """
        Spatially join a Census variable onto the hex grid.

        For income (median values): centroid point-in-polygon join — each hex
        gets the income of the tract containing its centroid, with
        nearest-neighbor fallback for unmatched hexes.
        For count variables (population, labour): uses area-weighted sum.

        Parameters
        ----------
        grid     : HexGrid
        variable : ACS variable code (e.g. "B19013_001E")
        agg      : aggregation method: "mean" (income) or "sum" (counts)
        """
        tracts = self.get_tract_data()

        if variable not in tracts.columns:
            raise KeyError(f"Variable '{variable}' not in tract data. "
                           f"Call fetch_acs_data(['{variable}']) first.")

        tracts = tracts[["GEOID", variable, "geometry"]].dropna(subset=["geometry"])
        tracts = tracts.copy()
        tracts[variable] = pd.to_numeric(tracts[variable], errors="coerce")
        county_median = tracts[variable].median()

        if agg == "mean":
            # Income: centroid-based spatial join — each hex gets the income
            # of the tract containing its centroid, with nearest-neighbor fallback.
            centroids = grid.centroids.copy()
            tract_geoms = tracts[["geometry", variable]].copy()

            if centroids.crs != tract_geoms.crs:
                tract_geoms = tract_geoms.to_crs(centroids.crs)

            joined = gpd.sjoin(centroids, tract_geoms, how="left", predicate="within")
            matched = joined[variable].notna().sum()
            total = len(joined)
            print(f"   Matched {matched}/{total} hexes to tracts")

            # Nearest-neighbor fallback for unmatched hexes
            unmatched_mask = joined[variable].isna()
            if unmatched_mask.any():
                unmatched_centroids = centroids.loc[unmatched_mask]
                if len(unmatched_centroids) > 0:
                    try:
                        nearest = gpd.sjoin_nearest(unmatched_centroids, tract_geoms, how="left")
                        for idx in nearest.index:
                            if idx in joined.index:
                                joined.loc[idx, variable] = nearest.loc[idx, variable]
                        print(f"   Fixed {unmatched_mask.sum()} unmatched via nearest neighbor")
                    except Exception as e:
                        print(f"   ⚠️ Nearest neighbor failed: {e}")

            result = joined.set_index("hex_id")[variable]
            result = result.fillna(county_median)

        else:
            # Count variables: area-weighted sum (population, labour)
            # Weight = intersection_area / total_intersection_area_for_hex
            # This ensures weights sum to 1 per hex — matches notebook exactly
            hex_gdf = grid.gdf[["hex_id", "geometry"]].copy()
            intersected = gpd.overlay(
                hex_gdf.to_crs(epsg=3857),
                tracts.to_crs(epsg=3857),
                how="intersection",
            )
            intersected["area"] = intersected.geometry.area
            intersected[variable] = pd.to_numeric(intersected[variable], errors="coerce")
            intersected = intersected.dropna(subset=[variable])

            # Area-weighted average per hex (weights sum to 1 within each hex)
            result = intersected.groupby("hex_id").apply(
                lambda d: (d[variable] * d["area"]).sum() / d["area"].sum()
                if d["area"].sum() > 0 else np.nan
            )
            # Count variables: missing hexes get 0, not median
            return result.reindex(grid.gdf["hex_id"]).fillna(0)

        # Income: missing hexes get county median
        return result.reindex(grid.gdf["hex_id"]).fillna(county_median)

    # ------------------------------------------------------------------
    # Convenience: assign all standard variables at once
    # ------------------------------------------------------------------

    def assign_all_to_hexes(self, grid: HexGrid) -> Dict[str, pd.Series]:
        """
        Returns a dict with keys:
            "median_income", "population", "labour"
        each as a pd.Series indexed by hex_id.
        """
        # Fetch all variables in one API call
        self.fetch_acs_data(list(CENSUS_VARS.values()))
        self.fetch_tiger_tracts()

        return {
            "median_income": self.assign_to_hexes(
                grid, CENSUS_VARS["median_income"], agg="mean"
            ),
            "population": self.assign_to_hexes(
                grid, CENSUS_VARS["population"], agg="sum"
            ),
            "labour": self.assign_to_hexes(
                grid, CENSUS_VARS["labour"], agg="sum"
            ),
        }

    def assign_neighborhoods_to_hexes(
        self,
        grid: HexGrid,
        osm_neighborhoods_gdf=None,
    ) -> pd.Series:
        """
        Assign neighbourhood names to hex cells via area-weighted polygon overlap.

        Priority
        --------
        1. *osm_neighborhoods_gdf* — GeoDataFrame with ``name`` + ``geometry``
           from OpenStreetMap (actual names like "Mission District").
           Used when provided and non-empty.
        2. Census TIGER tract names ("Census Tract 101.02") as fallback.

        Split hexes
        -----------
        Every source polygon covering ≥ 10 % of a hex area is listed with its
        share, e.g. "Mission District (65%) / Noe Valley (35%)".

        Returns
        -------
        pd.Series indexed by hex_id with neighbourhood name strings.
        """
        use_osm = (
            osm_neighborhoods_gdf is not None
            and not osm_neighborhoods_gdf.empty
            and "name" in osm_neighborhoods_gdf.columns
        )

        if use_osm:
            source = (
                osm_neighborhoods_gdf[["name", "geometry"]]
                .copy()
                .dropna(subset=["geometry", "name"])
                .rename(columns={"name": "_name"})
            )
            label = "OSM neighbourhood"
        else:
            tracts   = self.fetch_tiger_tracts()
            name_col = (
                "NAMELSAD" if "NAMELSAD" in tracts.columns else
                "NAME"     if "NAME"     in tracts.columns else
                "GEOID"
            )
            source = (
                tracts[[name_col, "geometry"]]
                .copy()
                .dropna(subset=["geometry"])
                .rename(columns={name_col: "_name"})
            )
            # Drop water-only tracts (e.g. Census Tract 9902 = San Francisco Bay)
            source = source[~source["_name"].astype(str).str.contains("9901","9902", na=False)]
            label = "census tract"

        hex_gdf = grid.gdf[["hex_id", "geometry"]].copy()

        try:
            intersected = gpd.overlay(
                hex_gdf.to_crs(epsg=3857),
                source.to_crs(epsg=3857),
                how="intersection",
            )
        except Exception as exc:
            print(f"   ⚠  Neighbourhood overlay failed: {exc}")
            return pd.Series("Unknown", index=grid.gdf["hex_id"]).rename("neighborhood")

        intersected["_area"] = intersected.geometry.area

        def _fmt(group):
            total = group["_area"].sum()
            if total == 0:
                return "Unknown"
            grp = group.copy()
            grp["_pct"] = grp["_area"] / total
            grp = grp.sort_values("_pct", ascending=False)
            sig = grp[grp["_pct"] >= 0.10]
            if len(sig) == 0:
                return str(grp.iloc[0]["_name"])
            if len(sig) == 1:
                return str(sig.iloc[0]["_name"])
            parts = [f"{r['_name']} ({r['_pct']:.0%})" for _, r in sig.iterrows()]
            return " / ".join(parts)

        result = (
            intersected.groupby("hex_id")
            .apply(_fmt)
            .rename("neighborhood")
        )
        print(f"   📍 {len(result)} hexes assigned to {label} neighbourhoods")
        return result.reindex(grid.gdf["hex_id"]).fillna("Unknown")