#!/usr/bin/env python3
"""
Web Application
===============
Flask app that exposes PCI and BCI as interactive tools.

Run:
    python webapp/app.py

Then open http://localhost:5000 in your browser.

Architecture
------------
State is kept in a server-side session dict (`STATE`) keyed by session ID.
Each run stores: grid, network_builder, mass_calc, pci series, bci series, etc.
Recomputation is modular: changing beta only recomputes accessibility + PCI,
not the network or amenity fetch.
"""

import os
import sys
import json
import copy
import uuid
import pickle
import traceback
import geopandas as gpd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import (
    Flask, request, jsonify, render_template, session, Response
)

from core.city_config import get_city_config, list_cities, get_default_user_params
from core.boundary_grid import BoundaryFetcher
from core.osm_fetcher import OSMDataFetcher, SupplierDataFetcher
from core.census_fetcher import CensusDataFetcher
from core.mass_calculator import MassCalculator
from core.network_builder import MultiModalNetworkBuilder
from core.h3_helper import HexGrid

from pci.pci_calculator import HansenAccessibilityModel, TopographicPCICalculator
from pci.pci_analysis import (
    plot_topography_layers, plot_topography_3d,
    make_pci_map, plot_pci_distribution, plot_pci_components, compute_pci_stats,
)

from bci.bci_masses import BCIMassCalculator
from bci.bci_calculator import BCIHansenAccessibility, BCICalculator
from bci.bci_analysis import (
    plot_bci_masses, plot_bci_components, plot_bci_topography,
    make_bci_map, plot_bci_distribution, compute_bci_stats,
)

from analysis.comparative_analysis import (
    both_available, compute_comparative_stats,
    plot_scatter, plot_distribution_comparison,
    plot_spatial_comparison, make_comparison_map,
)
from analysis.shared import (
    build_neighborhood_colors, compute_neighborhood_stats,
)
from analysis.sensitivity import run_pci_sensitivity, run_bci_sensitivity
from analysis.network_diagnostics import validate_network, print_mass_topography
from analysis.impact import (
    make_network_map,
    run_pci_amenity_removal, run_pci_amenity_addition,
    run_pci_edge_penalty, run_pci_edge_removal,
    run_bci_supplier_change,
    run_bci_edge_penalty, run_bci_edge_removal,
)
from analysis.isochrones import (
    run_isochrone_analysis,
    make_pci_isochrone_map, make_bci_isochrone_map,
)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET", "connectivity-pipeline-secret-key")

# Disable static-file caching so browsers always fetch the latest JS/CSS.
# Without this, browsers cache app.js and run stale code after updates.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# In-memory state store (use Redis for production)
STATE: dict = {}

# Directory for persisted results
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Keys saved to / restored from disk (excludes large network graphs)
_PERSIST_KEYS = [
    # Core results
    "pci", "bci", "city_name", "city_cfg", "user_params",
    # Census / mass data (no network refs)
    "income_by_hex", "population_by_hex", "labour_by_hex",
    "amenities", "mass_calc", "avg_mode_cost",
    # Unified BCI mass calculator (no network refs)
    "mass_calc_bci",
    # Grid carries all computed columns (PCI, BCI, accessibility, components)
    "grid",
    # OSM neighbourhood polygons (small GDF, safe to pickle)
    "neighborhoods_gdf",
    # NOTE: ham, pci_calc, bci_hansen, bci_calc are intentionally excluded —
    # they hold MultiModalNetworkBuilder references which are too large to pickle.
]


def _results_path(city_name: str) -> str:
    safe = city_name.replace(",", "").replace(" ", "_").replace("/", "_")
    return os.path.join(RESULTS_DIR, f"{safe}.pkl")


def save_results(city_name: str, s: dict):
    """Persist computed PCI/BCI state to disk (network excluded for size)."""
    try:
        to_save = {k: s[k] for k in _PERSIST_KEYS if k in s}
        path = _results_path(city_name)
        with open(path, "wb") as f:
            pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"   💾 Results saved → {path}")
    except Exception as e:
        print(f"   ⚠  Could not save results: {e}")


def load_results(city_name: str) -> dict:
    """Load persisted state from disk. Returns empty dict if not found."""
    path = _results_path(city_name)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"   📂 Results loaded ← {path}")
        return data
    except Exception as e:
        print(f"   ⚠  Could not load results: {e}")
        return {}


def get_state(sid: str) -> dict:
    if sid not in STATE:
        STATE[sid] = {}
    return STATE[sid]


def sid() -> str:
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


# ---------------------------------------------------------------------------
# Routes — general
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    import time
    return render_template("index.html", cities=list_cities(),
                           js_version=int(time.time()))


@app.route("/api/cities")
def api_cities():
    return jsonify(list_cities())


@app.route("/api/session/status", methods=["GET"])
def session_status():
    """Return which results are currently live in this server-side session.
    Called on page load so the JS flags (_hasPCI, _hasBCI) can be restored
    without requiring the user to click Restore."""
    s = get_state(sid())
    return jsonify({
        "has_pci":     "pci"     in s,
        "has_bci":     "bci"     in s,
        "has_network": "network" in s,
        "has_grid":    "grid"    in s,
    })


@app.route("/api/default_params")
def api_default_params():
    return jsonify(get_default_user_params())


@app.route("/api/about")
def api_about():
    """Return the about.md content as plain text."""
    about_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "about.md")
    try:
        with open(about_path, "r") as f:
            return jsonify({"status": "ok", "markdown": f.read()})
    except FileNotFoundError:
        return jsonify({"status": "error", "message": "about.md not found"}), 404


@app.route("/api/restore", methods=["POST"])
def restore_results():
    """
    Restore previously saved PCI/BCI results from disk into the current session.
    Body: { city_name }
    """
    data      = request.get_json() or {}
    city_name = data.get("city_name", "")
    if not city_name:
        return jsonify({"status": "error", "message": "city_name required"}), 400

    saved = load_results(city_name)
    if not saved:
        return jsonify({"status": "error", "message": "No saved results found for this city."}), 404

    s = get_state(sid())
    s.update(saved)

    has_pci = "pci" in s
    has_bci = "bci" in s
    return jsonify({
        "status":   "ok",
        "has_pci":  has_pci,
        "has_bci":  has_bci,
        "city_name": s.get("city_name", city_name),
    })


@app.route("/api/saved_cities")
def saved_cities():
    """List city names for which saved results exist on disk."""
    cities = []
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith(".pkl"):
            cities.append(fname[:-4].replace("_", " "))
    return jsonify({"status": "ok", "cities": cities})


# ---------------------------------------------------------------------------
# Routes — PCI
# ---------------------------------------------------------------------------

@app.route("/api/pci/init", methods=["POST"])
def pci_init():
    """
    Step 1 of PCI: fetch boundary, build grid, fetch amenities, build mass.
    Slow — runs once per city selection.
    Body: { city_name, user_params }
    """
    data = request.get_json()
    city_name   = data.get("city_name", "San Francisco, California, USA")
    user_params = {**get_default_user_params(), **(data.get("user_params") or {})}

    s = get_state(sid())

    # If same city and session already has grid + amenities, just update params
    # and recompute mass (fast) — skip all OSM fetching and grid building
    if (s.get("city_name") == city_name
            and "grid" in s
            and "amenities" in s
            and "mass_calc" in s):
        s["user_params"] = user_params
        grid      = s["grid"]
        amenities = s["amenities"]
        mass_calc = s["mass_calc"]
        mass_calc.amenity_weights = user_params["amenity_weights"]
        mass = mass_calc.compute_composite_mass()
        grid.attach_data(mass, "mass")
        for name, layer in mass_calc.layers.items():
            grid.attach_data(layer.normalized_values, f"{name}_norm")
        print("   ♻  Reusing grid & amenities already in session")
        return jsonify({"status": "ok", "n_hexagons": len(grid)})

    try:
        city_cfg = get_city_config(city_name)
        s["city_name"] = city_name
        s["city_cfg"]  = city_cfg
        s["user_params"] = user_params

        # Boundary + grid
        fetcher = BoundaryFetcher(city_name)
        fetcher.get_boundary(
            use_local=city_cfg.get("use_local_polygon", False),
            local_path=city_cfg.get("local_polygon_path"),
        )
        grid = fetcher.build_grid(resolution=city_cfg["h3_resolution"])

        # Preserve BCI columns from the previous grid (same city/resolution = same hex IDs).
        # Safety net for when the session guard didn't fire (e.g. after server restart).
        old_grid = s.get("grid")
        if old_grid is not None:
            for col in ["BCI", "A_market", "A_labour", "A_supplier",
                        "market_mass", "labour_mass", "supplier_mass"]:
                if col in old_grid.gdf.columns:
                    grid.attach_data(old_grid.gdf.set_index("hex_id")[col], col)

        s["grid"]    = grid
        s["fetcher"] = fetcher

        # Amenities
        osm = OSMDataFetcher(fetcher.boundary_polygon, cache_dir=CACHE_DIR)
        osm.set_enabled_tags(user_params["enabled_amenity_tags"])
        amenities = osm.fetch_all()
        s["amenities"] = amenities
        s["osm_fetcher"] = osm

        # Mass + topography
        mass_calc = MassCalculator(
            grid,
            amenity_weights=user_params["amenity_weights"],
            decay_coefficients=user_params["decay_coefficients"],
        )
        for name, gdf in amenities.items():
            mass_calc.add_amenity_layer(name, gdf, use_area=(name == "parks"))
        mass = mass_calc.compute_composite_mass()
        grid.attach_data(mass, "mass")
        for name, layer in mass_calc.layers.items():
            grid.attach_data(layer.normalized_values, f"{name}_norm")
        s["mass_calc"] = mass_calc

        return jsonify({"status": "ok", "n_hexagons": len(grid)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/pci/build_network", methods=["POST"])
def pci_build_network():
    """
    Step 2 of PCI: build multi-modal network + fetch census income.
    Slow — runs once per city selection.
    """
    s = get_state(sid())
    if "grid" not in s:
        return jsonify({"status": "error", "message": "Run /api/pci/init first"}), 400

    try:
        city_cfg    = s["city_cfg"]
        user_params = s["user_params"]
        grid        = s["grid"]
        fetcher     = s["fetcher"]

        # Network — build once per session, reuse on subsequent PCI runs
        if "network" not in s:
            net = MultiModalNetworkBuilder(
                fetcher.boundary_polygon,
                gtfs_path=city_cfg.get("gtfs_path"),
                travel_speeds=city_cfg["travel_speeds"],
                travel_costs=city_cfg["travel_costs"],
                time_penalties=city_cfg["time_penalties"],
                median_hourly_wage=city_cfg["median_hourly_wage"],
            )
            net.build_all_networks(cache_dir=CACHE_DIR)
            s["network"] = net
        else:
            net = s["network"]
            print("   ♻  Reusing network already in session")

        # Census — fetch once per session
        if "income_by_hex" not in s:
            census = CensusDataFetcher(
                year=city_cfg["census_year"],
                state_fips=city_cfg["state_fips"],
                county_fips=city_cfg["county_fips"],
                cache_dir=CACHE_DIR,
            )
            all_census = census.assign_all_to_hexes(grid)
            s["income_by_hex"]     = all_census["median_income"]
            s["population_by_hex"] = all_census["population"]
            s["labour_by_hex"]     = all_census["labour"]
            grid.attach_data(all_census["median_income"], "median_income")
            grid.attach_data(all_census["population"],    "population")
            s["census"] = census
        else:
            census = s.get("census")
            print("   ♻  Reusing census data already in session")

        # Neighbourhood polygons — build once per session
        if "neighborhoods_gdf" not in s:
            nb_file     = city_cfg.get("neighborhoods_file")
            use_custom  = nb_file and os.path.isfile(nb_file)
            nb_gdf      = None
            try:
                if use_custom:
                    nb_gdf = (gpd.read_file(nb_file)[["name", "geometry"]]
                              .dropna(subset=["geometry", "name"])
                              .reset_index(drop=True))
                    print(f"   🏘  Loaded {len(nb_gdf)} neighbourhoods from {os.path.basename(nb_file)}")
                else:
                    tracts   = census.fetch_tiger_tracts()
                    name_col = "NAMELSAD" if "NAMELSAD" in tracts.columns else "NAME"
                    nb_gdf   = (tracts[[name_col, "geometry"]]
                                .copy()
                                .rename(columns={name_col: "name"})
                                .dropna(subset=["geometry", "name"])
                                .loc[lambda d: ~d["name"].astype(str).str.contains("9902", na=False)]
                                .reset_index(drop=True))
                s["neighborhoods_gdf"] = nb_gdf
            except Exception as _nb_err:
                print(f"   ⚠  Neighbourhood polygon extraction skipped: {_nb_err}")
                s.setdefault("neighborhoods_gdf", None)

        # Assign neighbourhood names to hexes — only if not already on the grid
        if "neighborhood" not in grid.gdf.columns:
            nb_gdf        = s.get("neighborhoods_gdf")
            nb_file       = city_cfg.get("neighborhoods_file")
            use_custom    = nb_file and os.path.isfile(nb_file)
            safe_city     = s.get("city_name", "unknown").replace(",", "").replace(" ", "_").replace("/", "_")
            nb_cache_path = os.path.join(CACHE_DIR, f"neighborhoods_{safe_city}.pkl")
            try:
                if os.path.exists(nb_cache_path):
                    with open(nb_cache_path, "rb") as f:
                        neighborhoods = pickle.load(f)
                    print("   📦 Neighborhoods loaded from cache")
                else:
                    neighborhoods = census.assign_neighborhoods_to_hexes(
                        grid, osm_neighborhoods_gdf=nb_gdf if use_custom else None
                    )
                    try:
                        with open(nb_cache_path, "wb") as f:
                            pickle.dump(neighborhoods, f, protocol=pickle.HIGHEST_PROTOCOL)
                    except Exception as _ce:
                        print(f"   ⚠  Could not cache neighborhoods: {_ce}")
                grid.attach_data(neighborhoods, "neighborhood")
            except Exception as _nb_err:
                print(f"   ⚠  Neighborhood assignment skipped: {_nb_err}")

        diag = validate_network(net, grid, verbose=False)
        return jsonify({"status": "ok", "network_stats": diag["unified"]})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/pci/compute", methods=["POST"])
def pci_compute():
    """
    Step 3 of PCI: compute travel times, accessibility, final PCI.
    Re-run this when beta / lambda / weights change.
    Body: { user_params }  (partial — merged with stored params)
    """
    s = get_state(sid())
    required = ["grid", "network", "mass_calc", "income_by_hex"]
    for k in required:
        if k not in s:
            return jsonify({"status": "error",
                            "message": f"Missing '{k}'. Run init and build_network first."}), 400

    data = request.get_json() or {}
    # Merge partial user params
    up = s["user_params"]
    if "user_params" in data:
        up = {**up, **data["user_params"]}
        s["user_params"] = up

    try:
        grid      = s["grid"]
        net       = s["network"]
        mass_calc = s["mass_calc"]
        city_cfg  = s["city_cfg"]

        # Recompute mass if weights changed
        mass_calc.amenity_weights = up["amenity_weights"]
        mass = mass_calc.compute_composite_mass()
        grid.attach_data(mass, "mass")

        # Travel-time cache path (depends only on city/network, not on params)
        safe_city     = s.get("city_name", "unknown").replace(",", "").replace(" ", "_").replace("/", "_")
        tt_cache_path = os.path.join(CACHE_DIR, f"tt_pci_{safe_city}.pkl")
        avg_cost      = sum(city_cfg["travel_costs"].values()) / len(city_cfg["travel_costs"])

        existing_ham = s.get("ham")
        if existing_ham is not None and existing_ham._travel_times is not None:
            # Travel times already in session — just reuse them
            ham = existing_ham
            ham.mass = mass_calc
            print("   ♻  Reusing travel times from session")
        elif os.path.exists(tt_cache_path):
            # Load travel times from disk (survives server restarts)
            ham = HansenAccessibilityModel(grid, net, mass_calc)
            with open(tt_cache_path, "rb") as f:
                ham._travel_times = pickle.load(f)
            n = sum(len(v) for v in ham._travel_times.values())
            print(f"   📦 Travel times loaded from cache ({n:,} hex-pairs)")
        else:
            # First run — compute Dijkstra and save to disk
            ham = HansenAccessibilityModel(grid, net, mass_calc)
            ham.compute_travel_times(max_time=city_cfg["max_travel_time"])
            try:
                with open(tt_cache_path, "wb") as f:
                    pickle.dump(ham._travel_times, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"   💾 Travel times cached → {os.path.basename(tt_cache_path)}")
            except Exception as exc:
                print(f"   ⚠  Could not cache travel times: {exc}")
        acc = ham.compute_accessibility(
            beta=up["hansen_beta"],
            income_data=s["income_by_hex"],
            mode_cost=avg_cost,
        )
        grid.attach_data(acc, "accessibility")
        s["ham"] = ham
        s["avg_mode_cost"] = avg_cost
        s["city_cfg"] = city_cfg

        # Final PCI
        pci_calc = TopographicPCICalculator(grid, ham, mass_calc)
        pci = pci_calc.compute_pci(
            active_lambda=up["active_street_lambda"],
            mask_parks=up["mask_parks"],
            park_threshold=city_cfg["park_threshold"],
        )
        grid.attach_data(pci, "PCI")
        s["pci"] = pci
        s["pci_calc"] = pci_calc

        stats = compute_pci_stats(pci, grid)

        # Invalidate cached network map (now has updated PCI scores)
        s.pop("network_map_html", None)

        # Persist results so diagnostics / sensitivity can reload without rerun
        save_results(s.get("city_name", "unknown"), s)

        return jsonify({"status": "ok", "stats": stats})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/pci/visualize", methods=["GET"])
def pci_visualize():
    """Return all PCI visualisation artifacts as base64 PNGs + folium HTML strings."""
    s = get_state(sid())
    if "pci" not in s:
        return jsonify({"status": "error", "message": "Run /api/pci/compute first"}), 400

    city_name      = s.get("city_name", "")
    grid           = s["grid"]
    mass_calc      = s["mass_calc"]
    pci            = s["pci"]
    neighborhoods_gdf = s.get("neighborhoods_gdf")
    # network may be absent when results are restored from disk
    net = s.get("network")

    # Neighbourhood stats: avg PCI per neighbourhood (any overlap counts)
    nb_colors = {}
    nb_stats  = []
    if neighborhoods_gdf is not None and not neighborhoods_gdf.empty:
        nb_colors = build_neighborhood_colors(neighborhoods_gdf["name"].tolist())
        nb_stats  = compute_neighborhood_stats(grid.gdf, "PCI", neighborhoods_gdf, nb_colors)

    try:
        return jsonify({
            "status":              "ok",
            "topography_layers":   plot_topography_layers(grid, mass_calc, city_name),
            "topography_3d":       plot_topography_3d(grid, mass_calc, city_name),
            "pci_components":      plot_pci_components(grid, city_name),
            "pci_distribution":    plot_pci_distribution(pci, city_name),
            "pci_map":             make_pci_map(
                                       grid, pci, net, city_name,
                                       neighborhoods_gdf=neighborhoods_gdf,
                                   )._repr_html_(),
            "stats":               compute_pci_stats(pci, grid),
            "neighborhoods":       nb_stats,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Visualization failed: {e}"}), 500


# ---------------------------------------------------------------------------
# Routes — BCI
# ---------------------------------------------------------------------------

@app.route("/api/bci/init", methods=["POST"])
def bci_init():
    """
    BCI Step 1: fetch suppliers and compute the three masses.
    Reuses boundary, grid, census, and network from PCI if already built.
    """
    s = get_state(sid())
    data = request.get_json() or {}
    city_name   = data.get("city_name") or s.get("city_name", "San Francisco, California, USA")
    user_params = {**get_default_user_params(), **(data.get("user_params") or {})}

    try:
        # Bootstrap grid/boundary if not already done
        if "grid" not in s:
            city_cfg = get_city_config(city_name)
            s["city_name"] = city_name
            s["city_cfg"]  = city_cfg
            fetcher = BoundaryFetcher(city_name)
            fetcher.get_boundary(
                use_local=city_cfg.get("use_local_polygon", False),
                local_path=city_cfg.get("local_polygon_path"),
            )
            grid = fetcher.build_grid(resolution=city_cfg["h3_resolution"])
            s["grid"] = grid
            s["fetcher"] = fetcher
        else:
            city_cfg = s["city_cfg"]
            grid     = s["grid"]
            fetcher  = s["fetcher"]

        s["user_params"] = user_params

        # Supplier data from OSM
        supplier_fetcher = SupplierDataFetcher(fetcher.boundary_polygon, cache_dir=CACHE_DIR)
        supplier_fetcher.set_enabled_tags(user_params["enabled_supplier_tags"])
        suppliers_gdf    = supplier_fetcher.fetch_suppliers()
        supplier_counts  = supplier_fetcher.compute_supplier_density(grid.gdf, suppliers_gdf)
        s["supplier_fetcher"] = supplier_fetcher

        # Census if not loaded
        if "population_by_hex" not in s:
            census = CensusDataFetcher(
                year=city_cfg["census_year"],
                state_fips=city_cfg["state_fips"],
                county_fips=city_cfg["county_fips"],
                cache_dir=CACHE_DIR,
            )
            all_census = census.assign_all_to_hexes(grid)
            s["income_by_hex"]     = all_census["median_income"]
            s["population_by_hex"] = all_census["population"]
            s["labour_by_hex"]     = all_census["labour"]
            grid.attach_data(all_census["median_income"], "median_income")
            grid.attach_data(all_census["population"],    "population")

            # Neighbourhood polygons (BCI-first run: mirror pci_build_network logic)
            nb_file    = city_cfg.get("neighborhoods_file")
            use_custom = nb_file and os.path.isfile(nb_file)
            nb_gdf     = s.get("neighborhoods_gdf")
            if nb_gdf is None:
                try:
                    if use_custom:
                        nb_gdf = (gpd.read_file(nb_file)[["name", "geometry"]]
                                  .dropna(subset=["geometry", "name"])
                                  .reset_index(drop=True))
                        print(f"   🏘  Loaded {len(nb_gdf)} neighbourhoods from {os.path.basename(nb_file)}")
                    else:
                        tracts   = census.fetch_tiger_tracts()
                        name_col = "NAMELSAD" if "NAMELSAD" in tracts.columns else "NAME"
                        nb_gdf   = (tracts[[name_col, "geometry"]]
                                    .copy()
                                    .rename(columns={name_col: "name"})
                                    .dropna(subset=["geometry", "name"])
                                    .loc[lambda d: ~d["name"].astype(str).str.contains("9902", na=False)]
                                    .reset_index(drop=True))
                    s["neighborhoods_gdf"] = nb_gdf
                except Exception as _nb_err:
                    print(f"   ⚠  Neighbourhood polygon extraction skipped: {_nb_err}")
                    s.setdefault("neighborhoods_gdf", None)

            # Assign neighbourhood names to hexes (only if PCI didn't already do it)
            if "neighborhood" not in grid.gdf.columns:
                try:
                    neighborhoods = census.assign_neighborhoods_to_hexes(
                        grid, osm_neighborhoods_gdf=nb_gdf if use_custom else None
                    )
                    grid.attach_data(neighborhoods, "neighborhood")
                except Exception as _nb_err:
                    print(f"   ⚠  Neighborhood assignment skipped: {_nb_err}")

        # Single unified mass calculator — matches notebook BCIMassCalculator
        mass_calc = BCIMassCalculator(grid)
        mass_calc.load_census_data(
            population=s["population_by_hex"],
            income=s["income_by_hex"],
            labour=s["labour_by_hex"],
        )
        mass_calc.compute_market_mass(normalize_income=True)
        mass_calc.compute_labour_mass()
        mass_calc.compute_supplier_mass(supplier_counts)

        # Urban interface bonus
        if user_params["use_urban_interface"]:
            mass_calc.compute_urban_interface(
                boundary_polygon=fetcher.boundary_polygon,
                airport_locations=city_cfg.get("airport_locations"),
            )

        # Attach raw masses to grid for display (matches notebook scale)
        grid.attach_data(mass_calc.market_mass,   "market_mass")
        grid.attach_data(mass_calc.labour_mass,   "labour_mass")
        grid.attach_data(mass_calc.supplier_mass, "supplier_mass")

        s["mass_calc_bci"] = mass_calc

        return jsonify({"status": "ok", "n_hexagons": len(grid)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/bci/build_network", methods=["POST"])
def bci_build_network():
    """BCI Step 2: build component-specific networks (reuses PCI network if built)."""
    s = get_state(sid())
    if "grid" not in s:
        return jsonify({"status": "error", "message": "Run /api/bci/init first"}), 400

    try:
        city_cfg = s["city_cfg"]
        user_params = s["user_params"]

        # Reuse PCI network or build fresh
        if "network" not in s:
            fetcher = s["fetcher"]
            net = MultiModalNetworkBuilder(
                fetcher.boundary_polygon,
                gtfs_path=city_cfg.get("gtfs_path"),
                travel_speeds=city_cfg["travel_speeds"],
                travel_costs=city_cfg["travel_costs"],
                time_penalties=city_cfg["time_penalties"],
                median_hourly_wage=city_cfg["median_hourly_wage"],
            )
            net.build_all_networks(cache_dir=CACHE_DIR)
            s["network"] = net

        grid = s["grid"]
        net  = s["network"]

        # BCI Hansen model — build component graphs once per session
        if "bci_hansen" not in s:
            bci_hansen = BCIHansenAccessibility(
                grid, net,
                beta_params={
                    "market":   user_params["beta_market"],
                    "labour":   user_params["beta_labour"],
                    "supplier": user_params["beta_supplier"],
                },
            )
            bci_hansen.build_component_graphs()
            s["bci_hansen"] = bci_hansen
        else:
            bci_hansen = s["bci_hansen"]
            print("   ♻  Reusing BCI Hansen model from session")

        return jsonify({"status": "ok"})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/bci/compute", methods=["POST"])
def bci_compute():
    """BCI Step 3: compute travel times, accessibility, final BCI."""
    s = get_state(sid())
    required = ["grid", "bci_hansen", "mass_calc_bci"]
    for k in required:
        if k not in s:
            return jsonify({"status": "error",
                            "message": f"Missing '{k}'. Run bci/init and bci/build_network first."}), 400

    data = request.get_json() or {}
    up   = {**s["user_params"], **(data.get("user_params") or {})}
    s["user_params"] = up

    try:
        grid       = s["grid"]
        city_cfg   = s["city_cfg"]
        bci_hansen = s["bci_hansen"]
        mass_calc  = s["mass_calc_bci"]

        # Update betas
        bci_hansen.beta = {
            "market":   up["beta_market"],
            "labour":   up["beta_labour"],
            "supplier": up["beta_supplier"],
        }

        # BCI travel-time cache path
        safe_city      = s.get("city_name", "unknown").replace(",", "").replace(" ", "_").replace("/", "_")
        tt_bci_path    = os.path.join(CACHE_DIR, f"tt_bci_{safe_city}.pkl")
        _tt_populated  = all(
            comp in bci_hansen._travel_times and bci_hansen._travel_times[comp]
            for comp in ("market", "labour", "supplier")
        )

        if _tt_populated:
            print("   ♻  Reusing BCI travel times from session")
        elif os.path.exists(tt_bci_path):
            with open(tt_bci_path, "rb") as f:
                bci_hansen._travel_times = pickle.load(f)
            print(f"   📦 BCI travel times loaded from cache")
        else:
            bci_hansen.compute_all_travel_times(max_time=city_cfg["max_travel_time"])
            try:
                with open(tt_bci_path, "wb") as f:
                    pickle.dump(bci_hansen._travel_times, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"   💾 BCI travel times cached → {os.path.basename(tt_bci_path)}")
            except Exception as exc:
                print(f"   ⚠  Could not cache BCI travel times: {exc}")

        # Pass raw masses directly — matches notebook BCIHansenAccessibility.compute_accessibility()
        bci_hansen.compute_all_accessibility(
            market_mass=mass_calc.market_mass,
            labour_mass=mass_calc.labour_mass,
            supplier_mass=mass_calc.supplier_mass,
        )

        # Attach raw accessibility to grid (visualizer will normalise for display)
        for comp, series in bci_hansen.accessibility.items():
            grid.attach_data(series, f"A_{comp}")

        # Final BCI — pass unified mass calc
        bci_calc = BCICalculator(grid, bci_hansen, mass_calc)
        bci = bci_calc.compute_bci(
            method=up["bci_method"],
            market_weight=up["market_weight"],
            labour_weight=up["labour_weight"],
            supplier_weight=up["supplier_weight"],
            use_interface=up["use_urban_interface"],
            interface_lambda=up["interface_lambda"],
        )
        grid.attach_data(bci, "BCI")
        s["bci"]      = bci
        s["bci_calc"] = bci_calc

        # Attach normalised component series to grid so they survive disk persistence
        # (bci_calc itself is not pickled because it holds network references)
        for _comp_name, _comp_series in bci_calc.components.items():
            grid.attach_data(_comp_series, _comp_name)

        stats = compute_bci_stats(bci, grid, bci_calc)

        # Invalidate cached network map (now has updated BCI scores)
        s.pop("network_map_html", None)

        # Persist results
        save_results(s.get("city_name", "unknown"), s)

        return jsonify({"status": "ok", "stats": stats})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/bci/visualize", methods=["GET"])
def bci_visualize():
    s = get_state(sid())
    if "bci" not in s:
        return jsonify({"status": "error", "message": "Run /api/bci/compute first"}), 400

    city_name         = s.get("city_name", "")
    grid              = s["grid"]
    bci               = s["bci"]
    neighborhoods_gdf = s.get("neighborhoods_gdf")
    # bci_calc may be absent when results are restored from disk
    # (it holds network references and is excluded from the pickle)
    bci_calc      = s.get("bci_calc")
    mass_calc_bci = s.get("mass_calc_bci")

    # Neighbourhood stats: avg BCI per neighbourhood (any overlap counts)
    nb_colors = {}
    nb_stats  = []
    if neighborhoods_gdf is not None and not neighborhoods_gdf.empty:
        nb_colors = build_neighborhood_colors(neighborhoods_gdf["name"].tolist())
        nb_stats  = compute_neighborhood_stats(grid.gdf, "BCI", neighborhoods_gdf, nb_colors)

    return jsonify({
        "status":           "ok",
        "bci_masses":       plot_bci_masses(grid, city_name),
        "bci_topography":   plot_bci_topography(mass_calc_bci, city_name) if mass_calc_bci else "",
        "bci_components":   plot_bci_components(grid, city_name),
        "bci_distribution": plot_bci_distribution(bci, city_name),
        "bci_map":          make_bci_map(
                                grid, bci, city_name,
                                neighborhoods_gdf=neighborhoods_gdf,
                            )._repr_html_(),
        "stats":            compute_bci_stats(bci, grid, bci_calc),
        "neighborhoods":    nb_stats,
    })


# ---------------------------------------------------------------------------
# Routes — Comparative
# ---------------------------------------------------------------------------

@app.route("/api/compare/stats", methods=["GET"])
def compare_stats():
    s = get_state(sid())
    if not both_available(s.get("grid", HexGrid())):
        return jsonify({"status": "error",
                        "message": "Both PCI and BCI must be computed first."}), 400
    stats = compute_comparative_stats(s["grid"])
    return jsonify({"status": "ok", "stats": stats})


@app.route("/api/compare/visualize", methods=["GET"])
def compare_visualize():
    s = get_state(sid())
    grid = s.get("grid")
    if not both_available(grid):
        return jsonify({"status": "error",
                        "message": "Both PCI and BCI must be computed first."}), 400
    city_name = s.get("city_name", "")
    try:
        return jsonify({
            "status":               "ok",
            "scatter":              plot_scatter(grid, city_name),
            "distribution":         plot_distribution_comparison(grid, city_name),
            "spatial":              plot_spatial_comparison(grid, city_name),
            "comparison_map":       make_comparison_map(grid)._repr_html_(),
            "stats":                compute_comparative_stats(grid),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Compare visualization failed: {e}"}), 500


# ---------------------------------------------------------------------------
# Routes — Isochrones & Diagnostics
# ---------------------------------------------------------------------------

@app.route("/api/diagnostics/network", methods=["GET"])
def diagnostics_network():
    s = get_state(sid())
    if "network" not in s:
        return jsonify({"status": "error", "message": "Build network first"}), 400
    result = validate_network(s["network"], s["grid"], verbose=True)
    return jsonify({"status": "ok", "diagnostics": result})


@app.route("/api/diagnostics/topography", methods=["GET"])
def diagnostics_topography():
    s = get_state(sid())
    if "mass_calc" not in s:
        return jsonify({"status": "error", "message": "Run PCI init first"}), 400
    print_mass_topography(s["grid"], s["mass_calc"], s.get("city_name", ""))
    summary = s["mass_calc"].summary().to_dict(orient="records")
    return jsonify({"status": "ok", "summary": summary})


@app.route("/api/isochrones/run", methods=["POST"])
def isochrones_run():
    s = get_state(sid())
    if "grid" not in s:
        return jsonify({"status": "error", "message": "Build grid first"}), 400

    data = request.get_json() or {}
    max_origins = min(int(data.get("max_origins", 5)), 10)  # hard cap at 10

    # Guarantee census columns are on the grid regardless of build order or
    # session age — attach from already-fetched session data if missing.
    grid = s["grid"]
    if "population_by_hex" in s and "population" not in grid.gdf.columns:
        grid.attach_data(s["population_by_hex"], "population")
    if "income_by_hex" in s and "median_income" not in grid.gdf.columns:
        grid.attach_data(s["income_by_hex"], "median_income")

    results = run_isochrone_analysis(
        grid=grid,
        city_name=s.get("city_name", ""),
        amenities=s.get("amenities"),
        max_origins=max_origins,
    )

    s["iso_results"] = results

    def _to_records(df):
        return df.to_dict(orient="records") if df is not None and not df.empty else []

    return jsonify({
        "status":          "ok",
        "pci_per_origin":  _to_records(results["pci_per_origin"]),
        "bci_per_origin":  _to_records(results["bci_per_origin"]),
        "pci_summary":     _to_records(results["pci_summary"]),
        "bci_pop_summary": _to_records(results["bci_pop_summary"]),
        "bci_biz_summary": _to_records(results["bci_biz_summary"]),
    })


@app.route("/api/isochrones/maps", methods=["GET"])
def isochrones_maps():
    s   = get_state(sid())
    res = s.get("iso_results")
    if res is None:
        return jsonify({"status": "error", "message": "Run /api/isochrones/run first"}), 400

    grid = s["grid"]
    out  = {"status": "ok"}

    stored_origins = res.get("origins", {})

    if "pci" in s and not res["iso_gdf"].empty:
        pci_origins = stored_origins.get("PCI", {})
        out["pci_iso_map"] = make_pci_isochrone_map(
            grid, res["iso_gdf"], res["pci_counts_df"], pci_origins, s["pci"]
        )._repr_html_()

    if "bci" in s and not res["iso_gdf"].empty:
        bci_origins = stored_origins.get("BCI", {})
        out["bci_iso_map"] = make_bci_isochrone_map(
            grid, res["iso_gdf"], res["bci_counts_df"], bci_origins, s["bci"]
        )._repr_html_()

    return jsonify(out)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sensitivity Analysis endpoints
# ---------------------------------------------------------------------------

@app.route("/api/sensitivity/pci", methods=["POST"])
def sensitivity_pci():
    """Run OAT sensitivity analysis for PCI parameters."""
    s = get_state(sid())
    required = ["grid", "ham", "mass_calc", "amenities", "income_by_hex", "pci", "user_params"]
    missing  = [k for k in required if k not in s]
    if missing:
        step = "compute" if "pci" not in s else ("build network" if "income_by_hex" not in s else "init")
        return jsonify({"error": f"Please complete all PCI steps first (missing: {step})."}), 400
    try:
        results, tornado_b64, table_html = run_pci_sensitivity(s)
        return jsonify({"status": "ok", "tornado_png": tornado_b64, "table_html": table_html})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/sensitivity/bci", methods=["POST"])
def sensitivity_bci():
    """Run OAT sensitivity analysis for BCI parameters."""
    s = get_state(sid())
    required = ["grid", "bci_hansen", "mass_calc_bci", "bci"]
    missing  = [k for k in required if k not in s]
    if missing:
        return jsonify({"error": "Please complete all BCI steps first."}), 400
    try:
        results, tornado_b64, table_html = run_bci_sensitivity(s)
        return jsonify({"status": "ok", "tornado_png": tornado_b64, "table_html": table_html})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ---------------------------------------------------------------------------
# Routes — Scenario Testing
# NOTE: these routes are intentionally non-persistent.  They never call
# save_results() and they never overwrite session keys that the restore
# button depends on (pci, bci, grid, ham, bci_hansen, …).
# ---------------------------------------------------------------------------

@app.route("/api/scenario/network_map", methods=["GET"])
def scenario_network_map():
    """
    Render the network visualisation map (hex grid + walk/transit/drive
    edges).  Requires at least a grid in session; network is optional.
    """
    s = get_state(sid())
    if "grid" not in s:
        return jsonify({
            "status":  "error",
            "message": "No grid in session — run PCI or BCI init first.",
        }), 400

    try:
        net = s.get("network")
        m   = make_network_map(
            grid      = s["grid"],
            net       = net,
            pci       = s.get("pci"),
            bci       = s.get("bci"),
            city_name = s.get("city_name", ""),
        )
        return jsonify({"status": "ok", "map_html": m._repr_html_()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/scenario/network_map_view", methods=["GET"])
def scenario_network_map_view():
    """
    Serve the network map as a raw HTML page so the scenario iframe can
    load it via src= (same origin as the parent page).  This gives the
    iframe a real HTTP origin and makes window.parent.scAddHex() /
    window.parent.scReceiveEdge() reliable without cross-origin guards.
    """
    s = get_state(sid())
    if "grid" not in s:
        return Response(
            "<html><body style='font-family:sans-serif;padding:24px'>"
            "No grid in session — run PCI or BCI first.</body></html>",
            status=400, mimetype="text/html",
        )
    try:
        if "network_map_html" not in s:
            m = make_network_map(
                grid      = s["grid"],
                net       = s.get("network"),
                pci       = s.get("pci"),
                bci       = s.get("bci"),
                city_name = s.get("city_name", ""),
            )
            s["network_map_html"] = m._repr_html_()
            print("   🗺  Network map rendered and cached in session")
        else:
            print("   ♻  Reusing cached network map")

        return Response(s["network_map_html"], status=200, mimetype="text/html")
    except Exception as e:
        traceback.print_exc()
        return Response(
            f"<html><body>Error: {e}</body></html>",
            status=500, mimetype="text/html",
        )


@app.route("/api/scenario/amenity_types", methods=["GET"])
def scenario_amenity_types():
    """Return the amenity layers available in this session, with per-unit info."""
    s = get_state(sid())
    mass_calc = s.get("mass_calc")
    if not mass_calc:
        return jsonify({"status": "error",
                        "message": "Run PCI first to see available amenity types."}), 400
    UNITS = {
        "parks":       "m² of park area",
        "education":   "schools / facilities",
        "health":      "clinics / hospitals",
        "community":   "community centres",
    }
    total_weight = sum(
        l.weight for l in mass_calc.layers.values() if l.weight > 0
    ) or 1.0
    types = []
    for name, layer in mass_calc.layers.items():
        if layer.weight <= 0:
            continue
        raw_range = float(layer.raw_values.max() - layer.raw_values.min())
        types.append({
            "name":         name,
            "label":        name.replace("_", " ").title(),
            "unit":         UNITS.get(name, "units"),
            "weight":       round(layer.weight, 3),
            "raw_range":    max(raw_range, 1.0),
            "total_weight": round(total_weight, 3),
        })
    return jsonify({"status": "ok", "types": types})


@app.route("/api/scenario/run_pci", methods=["POST"])
def scenario_run_pci():
    """
    Run a single PCI scenario modification and return impact stats +
    delta map.  Session state is NOT modified — no save_results() call.

    Body:
        scenario_type : "amenity_remove" | "amenity_add"
                      | "edge_penalty"  | "edge_remove"
        hex_ids       : [str, ...]
        radius        : int (k-ring, default 0)
        factor        : float (edge_penalty only)
        amenity_type  : str  (amenity_add only, e.g. "education")
        amenity_count : float (amenity_add only, e.g. 3)
    """
    s = get_state(sid())
    required = ["grid", "ham", "mass_calc", "pci", "user_params", "city_cfg"]
    missing  = [k for k in required if k not in s]
    if missing:
        return jsonify({
            "status":  "error",
            "message": f"Run PCI compute first (missing: {', '.join(missing)}).",
        }), 400

    data          = request.get_json() or {}
    scenario_type = data.get("scenario_type", "amenity_remove")
    hex_ids       = list(data.get("hex_ids", []))
    radius        = int(data.get("radius", 0))
    factor        = float(data.get("factor", 2.0))
    amenity_type  = data.get("amenity_type", "education")
    amenity_count = float(data.get("amenity_count", 1.0))
    for en in data.get("edge_nodes", []):
        if en.get("u"): hex_ids.append(str(en["u"]))
        if en.get("v"): hex_ids.append(str(en["v"]))

    h3_only = [h for h in hex_ids if not str(h).isdigit()]
    if scenario_type in ("amenity_remove", "amenity_add"):
        if not h3_only:
            return jsonify({"status": "error",
                            "message": "Select a hex cell (not just edges) for amenity scenarios."}), 400
    elif not hex_ids:
        return jsonify({"status": "error", "message": "No hexes or edges selected."}), 400

    try:
        if scenario_type == "amenity_remove":
            result = run_pci_amenity_removal(s, hex_ids, radius)
        elif scenario_type == "amenity_add":
            result = run_pci_amenity_addition(s, h3_only[0], amenity_type, amenity_count)
        elif scenario_type == "edge_penalty":
            result = run_pci_edge_penalty(s, hex_ids, factor, radius)
        elif scenario_type == "edge_remove":
            result = run_pci_edge_removal(s, hex_ids, radius)
        else:
            return jsonify({"status": "error",
                            "message": f"Unknown scenario type: {scenario_type}"}), 400

        return jsonify({"status": "ok", **result})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/scenario/run_bci", methods=["POST"])
def scenario_run_bci():
    """
    Run a single BCI scenario modification and return impact stats +
    delta map.  Session state is NOT modified — no save_results() call.

    Body:
        scenario_type : "supplier_remove" | "supplier_add"
                      | "edge_penalty"   | "edge_remove"
        hex_ids       : [str, ...]
        radius        : int (k-ring, default 0)
        factor        : float (edge_penalty only, default 2.0)
        strength      : float (supplier_add only, default 1.0)
    """
    s = get_state(sid())
    required = ["grid", "bci_hansen", "mass_calc_bci", "bci", "user_params"]
    missing  = [k for k in required if k not in s]
    if missing:
        return jsonify({
            "status":  "error",
            "message": f"Run BCI compute first (missing: {', '.join(missing)}).",
        }), 400

    data          = request.get_json() or {}
    scenario_type = data.get("scenario_type", "supplier_remove")
    hex_ids       = list(data.get("hex_ids", []))
    radius        = int(data.get("radius", 0))
    strength      = float(data.get("strength", 1.0))
    factor        = float(data.get("factor", 2.0))
    for en in data.get("edge_nodes", []):
        if en.get("u"): hex_ids.append(str(en["u"]))
        if en.get("v"): hex_ids.append(str(en["v"]))

    h3_only = [h for h in hex_ids if not str(h).isdigit()]
    if scenario_type in ("supplier_remove", "supplier_add"):
        if not h3_only:
            return jsonify({"status": "error",
                            "message": "Select a hex cell (not just edges) for supplier scenarios."}), 400
    elif not hex_ids:
        return jsonify({"status": "error", "message": "No hexes or edges selected."}), 400

    try:
        if scenario_type == "supplier_remove":
            result = run_bci_supplier_change(s, hex_ids, "remove", strength, radius)
        elif scenario_type == "supplier_add":
            result = run_bci_supplier_change(s, hex_ids, "add",    strength, radius)
        elif scenario_type == "edge_penalty":
            result = run_bci_edge_penalty(s, hex_ids, factor, radius)
        elif scenario_type == "edge_remove":
            result = run_bci_edge_removal(s, hex_ids, radius)
        else:
            return jsonify({"status": "error",
                            "message": f"Unknown BCI scenario type: {scenario_type}"}), 400

        return jsonify({"status": "ok", **result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
