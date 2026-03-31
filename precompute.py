"""
precompute.py — Warm all data caches for configured cities
===========================================================
Run this script once locally before deployment.  It populates
connectivity_pipeline/data/cache/ with pre-fetched network graphs,
OSM amenity/supplier GeoDataFrames, and Census data so that the
webapp never needs to hit external APIs at runtime.

Usage
-----
    # All cities
    python precompute.py

    # One specific city
    python precompute.py --city "San Francisco, California, USA"

    # Skip network build (amenities + census only)
    python precompute.py --skip-network

After running, upload the generated data/cache/ directory to your
server alongside the rest of the project.
"""

import argparse
import os
import sys
import traceback

# Make sure project modules are importable from the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "connectivity_pipeline"))

from core.city_config import CITY_CONFIGS, get_default_user_params
from core.boundary_grid import BoundaryFetcher
from core.osm_fetcher import OSMDataFetcher, SupplierDataFetcher
from core.census_fetcher import CensusDataFetcher
from core.network_builder import MultiModalNetworkBuilder

CACHE_DIR = os.path.join(PROJECT_ROOT, "connectivity_pipeline", "data", "cache")


def precompute_city(city_name: str, city_cfg: dict, skip_network: bool = False):
    print(f"\n{'=' * 60}")
    print(f"  Precomputing: {city_name}")
    print(f"{'=' * 60}")

    user_params = get_default_user_params()
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Boundary + grid
    # ------------------------------------------------------------------
    print("\n[1/4] Boundary & grid...")
    fetcher = BoundaryFetcher(city_name)
    fetcher.get_boundary(
        use_local=city_cfg.get("use_local_polygon", False),
        local_path=city_cfg.get("local_polygon_path"),
    )
    grid = fetcher.build_grid(resolution=city_cfg["h3_resolution"])
    print(f"   ✓ Grid: {len(grid)} hexagons")

    # ------------------------------------------------------------------
    # 2. OSM amenities (PCI)
    # ------------------------------------------------------------------
    print("\n[2/4] OSM amenities (PCI)...")
    osm = OSMDataFetcher(fetcher.boundary_polygon, cache_dir=CACHE_DIR)
    osm.set_enabled_tags(user_params["enabled_amenity_tags"])
    amenities = osm.fetch_all()
    fetched = sum(1 for v in amenities.values() if v is not None)
    print(f"   ✓ {fetched}/{len(amenities)} amenity categories cached")

    # ------------------------------------------------------------------
    # 3. OSM suppliers (BCI)
    # ------------------------------------------------------------------
    print("\n[3/4] OSM suppliers (BCI)...")
    supplier_fetcher = SupplierDataFetcher(fetcher.boundary_polygon, cache_dir=CACHE_DIR)
    supplier_fetcher.set_enabled_tags(user_params["enabled_supplier_tags"])
    supplier_fetcher.fetch_suppliers()
    print("   ✓ Supplier categories cached")

    # ------------------------------------------------------------------
    # 4. Census ACS + TIGER
    # ------------------------------------------------------------------
    print("\n[4/4] Census ACS + TIGER tracts...")
    census = CensusDataFetcher(
        year=city_cfg["census_year"],
        state_fips=city_cfg["state_fips"],
        county_fips=city_cfg["county_fips"],
        cache_dir=CACHE_DIR,
    )
    census.assign_all_to_hexes(grid)
    print("   ✓ Census data cached")

    # ------------------------------------------------------------------
    # 5. Multi-modal network (optional — largest file, slowest step)
    # ------------------------------------------------------------------
    if skip_network:
        print("\n[skipped] Network build (--skip-network flag set)")
    else:
        print("\n[5/5] Multi-modal network (walk / bike / drive / transit)...")
        print("   ⏳ This is the slowest step (~5-15 min per city)...")
        net = MultiModalNetworkBuilder(
            fetcher.boundary_polygon,
            gtfs_path=city_cfg.get("gtfs_path"),
            travel_speeds=city_cfg["travel_speeds"],
            travel_costs=city_cfg["travel_costs"],
            time_penalties=city_cfg["time_penalties"],
            median_hourly_wage=city_cfg["median_hourly_wage"],
        )
        net.build_all_networks(cache_dir=CACHE_DIR)
        print("   ✓ Network cached")

    print(f"\n  Done: {city_name}")


def main():
    parser = argparse.ArgumentParser(description="Pre-warm data caches for all cities.")
    parser.add_argument(
        "--city",
        default=None,
        help="Name of a single city to precompute (must match CITY_CONFIGS key). "
             "Omit to precompute all cities.",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip the multi-modal network build (amenities + census only).",
    )
    args = parser.parse_args()

    if args.city:
        if args.city not in CITY_CONFIGS:
            print(f"Error: '{args.city}' not found in CITY_CONFIGS.")
            print(f"Available: {list(CITY_CONFIGS.keys())}")
            sys.exit(1)
        cities = {args.city: CITY_CONFIGS[args.city]}
    else:
        cities = CITY_CONFIGS

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Cities to precompute: {list(cities.keys())}")

    failed = []
    for city_name, city_cfg in cities.items():
        try:
            precompute_city(city_name, city_cfg, skip_network=args.skip_network)
        except Exception:
            print(f"\n  ✗ Failed: {city_name}")
            traceback.print_exc()
            failed.append(city_name)

    print(f"\n{'=' * 60}")
    print(f"  Precompute complete.")
    print(f"  Cache directory: {CACHE_DIR}")
    if failed:
        print(f"  Failed cities: {failed}")
    else:
        print("  All cities cached successfully.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
