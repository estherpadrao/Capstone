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
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import (
    Flask, request, jsonify, render_template, session
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
    plot_topography_layers, plot_topography_3d, make_mass_network_map,
    make_pci_map, plot_pci_distribution, plot_pci_components, compute_pci_stats,
)

from bci.bci_masses import MarketMassCalculator, LabourMassCalculator, SupplierMassCalculator
from bci.bci_calculator import BCIHansenAccessibility, BCICalculator
from bci.bci_analysis import (
    plot_bci_masses, plot_bci_components, make_market_map, make_labour_map,
    make_supplier_map, make_bci_map, plot_bci_distribution, compute_bci_stats,
)

from analysis.comparative_analysis import (
    both_available, compute_comparative_stats,
    plot_scatter, plot_distribution_comparison,
    plot_spatial_comparison, make_comparison_map,
)
from analysis.isochrones import (
    validate_network, print_mass_topography, run_isochrone_analysis,
    make_pci_isochrone_map, make_bci_isochrone_map,
)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET", "connectivity-pipeline-secret-key")

# In-memory state store (use Redis for production)
STATE: dict = {}


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
    return render_template("index.html", cities=list_cities())


@app.route("/api/cities")
def api_cities():
    return jsonify(list_cities())


@app.route("/api/default_params")
def api_default_params():
    return jsonify(get_default_user_params())


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
        s["grid"]    = grid
        s["fetcher"] = fetcher

        # Amenities
        osm = OSMDataFetcher(fetcher.boundary_polygon)
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

        # Network
        net = MultiModalNetworkBuilder(
            fetcher.boundary_polygon,
            gtfs_path=city_cfg.get("gtfs_path"),
            travel_speeds=city_cfg["travel_speeds"],
            travel_costs=city_cfg["travel_costs"],
            time_penalties=city_cfg["time_penalties"],
            median_hourly_wage=city_cfg["median_hourly_wage"],
        )
        net.build_all_networks()
        s["network"] = net

        # Census income
        census = CensusDataFetcher(
            year=city_cfg["census_year"],
            state_fips=city_cfg["state_fips"],
            county_fips=city_cfg["county_fips"],
        )
        all_census = census.assign_all_to_hexes(grid)
        s["income_by_hex"]     = all_census["median_income"]
        s["population_by_hex"] = all_census["population"]
        s["labour_by_hex"]     = all_census["labour"]
        grid.attach_data(all_census["median_income"], "median_income")
        s["census"] = census

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

        # Hansen model
        ham = HansenAccessibilityModel(grid, net, mass_calc)
        ham.compute_travel_times(max_time=city_cfg["max_travel_time"])
        avg_cost = sum(city_cfg["travel_costs"].values()) / len(city_cfg["travel_costs"])
        acc = ham.compute_accessibility(
            beta=up["hansen_beta"],
            income_data=s["income_by_hex"],
            mode_cost=avg_cost,
        )
        grid.attach_data(acc, "accessibility")
        s["ham"] = ham

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

    city_name = s.get("city_name", "")
    grid      = s["grid"]
    mass_calc = s["mass_calc"]
    pci       = s["pci"]
    net       = s["network"]

    return jsonify({
        "status":              "ok",
        "topography_layers":   plot_topography_layers(grid, mass_calc, city_name),
        "topography_3d":       plot_topography_3d(grid, mass_calc, city_name),
        "pci_components":      plot_pci_components(grid, city_name),
        "pci_distribution":    plot_pci_distribution(pci, city_name),
        "mass_network_map":    make_mass_network_map(grid, net, city_name)._repr_html_(),
        "pci_map":             make_pci_map(grid, pci, net, city_name)._repr_html_(),
        "stats":               compute_pci_stats(pci, grid),
    })


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
        supplier_fetcher = SupplierDataFetcher(fetcher.boundary_polygon)
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
            )
            all_census = census.assign_all_to_hexes(grid)
            s["income_by_hex"]     = all_census["median_income"]
            s["population_by_hex"] = all_census["population"]
            s["labour_by_hex"]     = all_census["labour"]

        hex_ids = grid.gdf["hex_id"].tolist()

        # Three mass calculators
        market_calc   = MarketMassCalculator(hex_ids)
        labour_calc   = LabourMassCalculator(hex_ids)
        supplier_calc = SupplierMassCalculator(hex_ids)

        market_mass   = market_calc.compute(s["population_by_hex"], s["income_by_hex"])
        labour_mass   = labour_calc.compute(s["labour_by_hex"])
        supplier_mass = supplier_calc.compute(supplier_counts)

        # Urban interface
        if user_params["use_urban_interface"]:
            supplier_calc.compute_urban_interface(
                grid,
                boundary_polygon=fetcher.boundary_polygon,
                airport_locations=city_cfg.get("airport_locations"),
            )

        grid.attach_data(market_mass,   "market_mass")
        grid.attach_data(labour_mass,   "labour_mass")
        grid.attach_data(supplier_mass, "supplier_mass")

        s["market_calc"]   = market_calc
        s["labour_calc"]   = labour_calc
        s["supplier_calc"] = supplier_calc

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
            net.build_all_networks()
            s["network"] = net

        # BCI Hansen model
        grid = s["grid"]
        net  = s["network"]
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

        return jsonify({"status": "ok"})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/bci/compute", methods=["POST"])
def bci_compute():
    """BCI Step 3: compute travel times, accessibility, final BCI."""
    s = get_state(sid())
    required = ["grid", "bci_hansen", "market_calc", "labour_calc", "supplier_calc"]
    for k in required:
        if k not in s:
            return jsonify({"status": "error",
                            "message": f"Missing '{k}'. Run bci/init and bci/build_network first."}), 400

    data = request.get_json() or {}
    up   = {**s["user_params"], **(data.get("user_params") or {})}
    s["user_params"] = up

    try:
        grid          = s["grid"]
        city_cfg      = s["city_cfg"]
        bci_hansen    = s["bci_hansen"]
        market_calc   = s["market_calc"]
        labour_calc   = s["labour_calc"]
        supplier_calc = s["supplier_calc"]

        # Update betas
        bci_hansen.beta = {
            "market":   up["beta_market"],
            "labour":   up["beta_labour"],
            "supplier": up["beta_supplier"],
        }

        bci_hansen.compute_all_travel_times(max_time=city_cfg["max_travel_time"])
        bci_hansen.compute_all_accessibility(
            market_mass=market_calc.normalised,
            labour_mass=labour_calc.normalised,
            supplier_mass=supplier_calc.normalised,
        )

        # Attach accessibility to grid
        for comp, series in bci_hansen.accessibility.items():
            grid.attach_data(series, f"A_{comp}")

        # Final BCI
        bci_calc = BCICalculator(grid, bci_hansen, market_calc, labour_calc, supplier_calc)
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

        stats = compute_bci_stats(bci, grid, bci_calc)
        return jsonify({"status": "ok", "stats": stats})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/bci/visualize", methods=["GET"])
def bci_visualize():
    s = get_state(sid())
    if "bci" not in s:
        return jsonify({"status": "error", "message": "Run /api/bci/compute first"}), 400

    city_name = s.get("city_name", "")
    grid      = s["grid"]
    bci       = s["bci"]
    bci_calc  = s["bci_calc"]

    return jsonify({
        "status":           "ok",
        "bci_masses":       plot_bci_masses(grid, city_name),
        "bci_components":   plot_bci_components(grid, city_name),
        "bci_distribution": plot_bci_distribution(bci, city_name),
        "market_map":       make_market_map(grid, city_name)._repr_html_(),
        "labour_map":       make_labour_map(grid, city_name)._repr_html_(),
        "supplier_map":     make_supplier_map(grid, city_name)._repr_html_(),
        "bci_map":          make_bci_map(grid, bci, city_name)._repr_html_(),
        "stats":            compute_bci_stats(bci, grid, bci_calc),
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
    return jsonify({
        "status":               "ok",
        "scatter":              plot_scatter(grid, city_name),
        "distribution":         plot_distribution_comparison(grid, city_name),
        "spatial":              plot_spatial_comparison(grid, city_name),
        "comparison_map":       make_comparison_map(grid)._repr_html_(),
        "stats":                compute_comparative_stats(grid),
    })


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
    max_origins = int(data.get("max_origins", 5))

    results = run_isochrone_analysis(
        grid=s["grid"],
        city_name=s.get("city_name", ""),
        amenities=s.get("amenities"),
        max_origins=max_origins,
    )

    s["iso_results"] = results
    # Return summary tables (not full GeoDataFrame)
    return jsonify({
        "status": "ok",
        "pci_summary": results["pci_summary"].reset_index().to_dict(orient="records") if not results["pci_summary"].empty else [],
        "bci_pop_summary": results["bci_pop_summary"].reset_index().to_dict(orient="records") if not results["bci_pop_summary"].empty else [],
        "bci_biz_summary": results["bci_biz_summary"].reset_index().to_dict(orient="records") if not results["bci_biz_summary"].empty else [],
    })


@app.route("/api/isochrones/maps", methods=["GET"])
def isochrones_maps():
    s   = get_state(sid())
    res = s.get("iso_results")
    if res is None:
        return jsonify({"status": "error", "message": "Run /api/isochrones/run first"}), 400

    grid = s["grid"]
    out  = {"status": "ok"}

    if "pci" in s and not res["iso_gdf"].empty:
        pci_origins = {
            "top":    grid.gdf[grid.gdf["PCI"] >= grid.gdf["PCI"].quantile(0.90)],
            "bottom": grid.gdf[grid.gdf["PCI"] <= grid.gdf["PCI"].quantile(0.10)],
        }
        out["pci_iso_map"] = make_pci_isochrone_map(
            grid, res["iso_gdf"], res["pci_counts_df"], pci_origins, s["pci"]
        )._repr_html_()

    if "bci" in s and not res["iso_gdf"].empty:
        bci_origins = {
            "top":    grid.gdf[grid.gdf["BCI"] >= grid.gdf["BCI"].quantile(0.90)],
            "bottom": grid.gdf[grid.gdf["BCI"] <= grid.gdf["BCI"].quantile(0.10)],
        }
        out["bci_iso_map"] = make_bci_isochrone_map(
            grid, res["iso_gdf"], res["bci_counts_df"], bci_origins, s["bci"]
        )._repr_html_()

    return jsonify(out)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
