"""
File 4: Census Data Fetcher
============================
Fetches ACS 5-year data and TIGER tract geometries from the US Census Bureau.
Used by both PCI (income/cost adjustment) and BCI (population, labour, income).
"""

import os
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
    ):
        self.year = year
        self.state_fips = str(state_fips).zfill(2)
        self.county_fips = str(county_fips).zfill(3)
        self.api_key = api_key
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
        return df

    # ------------------------------------------------------------------
    # TIGER geometries
    # ------------------------------------------------------------------

    def fetch_tiger_tracts(self) -> gpd.GeoDataFrame:
        """Download TIGER tract shapefile and return as GeoDataFrame."""
        if self._tracts_gdf is not None:
            return self._tracts_gdf

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

        For income (median values): uses centroid point-in-polygon join,
        matching the notebook implementation exactly.
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
            # Income: centroid point-in-polygon join (matches notebook)
            centroids = grid.gdf.copy().to_crs("EPSG:4326")
            centroids["geometry"] = centroids.geometry.centroid
            centroids = centroids[["hex_id", "geometry"]]

            joined = gpd.sjoin(
                centroids,
                tracts.to_crs("EPSG:4326")[["geometry", variable]],
                how="left",
                predicate="within",
            )

            # Nearest-neighbour fallback for unmatched centroids
            unmatched = joined[variable].isna()
            if unmatched.any():
                nearest = gpd.sjoin_nearest(
                    centroids.loc[unmatched],
                    tracts.to_crs("EPSG:4326")[["geometry", variable]],
                    how="left",
                )
                joined.loc[unmatched, variable] = nearest[variable].values

            result = joined.set_index("hex_id")[variable]
            result = result.fillna(county_median)

        else:
            # Count variables: area-weighted sum (population, labour)
            hex_gdf = grid.gdf[["hex_id", "geometry"]].copy()
            intersected = gpd.overlay(
                hex_gdf.to_crs("EPSG:4326"),
                tracts.to_crs("EPSG:4326"),
                how="intersection",
            )
            intersected = intersected.to_crs(epsg=3857)
            intersected["isect_area"] = intersected.geometry.area

            tract_areas = tracts.to_crs(epsg=3857).copy()
            tract_areas["tract_area"] = tract_areas.geometry.area
            intersected = intersected.merge(
                tract_areas[["GEOID", "tract_area"]], on="GEOID", how="left"
            )
            intersected["weight"] = (
                intersected["isect_area"] / intersected["tract_area"].replace(0, np.nan)
            )
            intersected["weighted_val"] = intersected[variable] * intersected["weight"]
            result = intersected.groupby("hex_id")["weighted_val"].sum()

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