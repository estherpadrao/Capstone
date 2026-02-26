"""
City Configuration Registry
============================
City-level parameters are LOCKED here and cannot be changed by the user.
The user selects a city; the app loads the corresponding config automatically.

Parameters the user CAN change (in the webapp):
  - HANSEN_BETA (global or per-mode)
  - AMENITY_WEIGHTS (per category)
  - ACTIVE_STREET_LAMBDA
  - MASK_PARKS / PARK_THRESHOLD
  - BCI beta per component (beta_market, beta_labour, beta_supplier)
  - BCI combination method and weights
  - INTERFACE_LAMBDA (urban interface bonus)
  - OSM tag toggles

Parameters the user CANNOT change (locked to the city):
  - H3_RESOLUTION
  - STATE_FIPS / COUNTY_FIPS / CENSUS_YEAR
  - GTFS path / URL
  - TRAVEL_SPEEDS
  - TRAVEL_COSTS
  - Time penalties (transit wait, parking, bike unlock)
  - MEDIAN_HOURLY_WAGE
  - PARK_THRESHOLD (override)
  - MAX_TRAVEL_TIME
"""

from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# City registry
# ---------------------------------------------------------------------------

CITY_CONFIGS: Dict[str, Dict[str, Any]] = {

    "San Francisco, California, USA": {
        "display_name":     "San Francisco, CA",
        "h3_resolution":    8,
        "state_fips":       "06",
        "county_fips":      "075",
        "census_year":      2022,
        "gtfs_path":        "data/muni_gtfs-current.zip",   # place GTFS zip here
        "gtfs_url":         None,
        "travel_speeds": {
            "walk":    4.8,
            "bike":    15.0,
            "drive":   24.0,
            "transit": 13.0,
        },
        "travel_costs": {
            "walk":    0.00,
            "bike":    3.99,
            "drive":   9.00,
            "transit": 2.75,
        },
        "time_penalties": {
            "transit_wait":   9.0,
            "transit_board":  0.5,
            "parking_search": 13.0,
            "bike_unlock":    1.0,
        },
        "median_hourly_wage": 46.07,
        "max_travel_time":    90,
        "park_threshold":     0.90,
        "airport_locations":  [(37.6213, -122.3790)],  # SFO
        "use_local_polygon":  True,
        "local_polygon_path": "data/sf_polygon.geojson",
    },

    "New York City, New York, USA": {
        "display_name":     "New York City, NY",
        "h3_resolution":    8,
        "state_fips":       "36",
        "county_fips":      "061",   # Manhattan; override if you want all 5 boroughs
        "census_year":      2022,
        "gtfs_path":        "data/nyc_gtfs.zip",
        "gtfs_url":         None,
        "travel_speeds": {
            "walk":    4.8,
            "bike":    14.0,
            "drive":   18.0,
            "transit": 16.0,
        },
        "travel_costs": {
            "walk":    0.00,
            "bike":    4.49,
            "drive":   15.00,
            "transit": 2.90,
        },
        "time_penalties": {
            "transit_wait":   7.0,
            "transit_board":  0.5,
            "parking_search": 20.0,
            "bike_unlock":    1.0,
        },
        "median_hourly_wage": 38.00,
        "max_travel_time":    90,
        "park_threshold":     0.90,
        "airport_locations":  [(40.6413, -73.7781), (40.7769, -73.8740)],  # JFK, LGA
        "use_local_polygon":  False,
        "local_polygon_path": None,
    },

    "Chicago, Illinois, USA": {
        "display_name":     "Chicago, IL",
        "h3_resolution":    8,
        "state_fips":       "17",
        "county_fips":      "031",
        "census_year":      2022,
        "gtfs_path":        "data/chicago_gtfs.zip",
        "gtfs_url":         None,
        "travel_speeds": {
            "walk":    4.8,
            "bike":    14.0,
            "drive":   22.0,
            "transit": 15.0,
        },
        "travel_costs": {
            "walk":    0.00,
            "bike":    3.30,
            "drive":   10.00,
            "transit": 2.50,
        },
        "time_penalties": {
            "transit_wait":   10.0,
            "transit_board":   0.5,
            "parking_search": 10.0,
            "bike_unlock":     1.0,
        },
        "median_hourly_wage": 30.00,
        "max_travel_time":    90,
        "park_threshold":     0.90,
        "airport_locations":  [(41.9742, -87.9073), (41.7868, -87.7522)],  # ORD, MDW
        "use_local_polygon":  False,
        "local_polygon_path": None,
    },

    # Add more cities following the same pattern
}


# ---------------------------------------------------------------------------
# Default user-tunable parameters
# ---------------------------------------------------------------------------

DEFAULT_USER_PARAMS: Dict[str, Any] = {
    # PCI
    "hansen_beta":          0.08,
    "per_mode_beta": {          # optional per-mode override (null = use global beta)
        "walk":    None,
        "bike":    None,
        "drive":   None,
        "transit": None,
    },
    "amenity_weights": {
        "health":     0.319,
        "education":  0.276,
        "parks":      0.255,
        "community":  0.148,
    },
    "decay_coefficients": {
        "health":     0.08,
        "education":  0.08,
        "parks":      0.08,
        "community":  0.08,
    },
    "active_street_lambda": 0.30,
    "mask_parks":           False,

    # OSM tag toggles (PCI amenities)
    "enabled_amenity_tags": {
        "health":      True,
        "education":   True,
        "parks":       True,
        "transit":     True,
        "food_retail": True,
        "community":   True,
    },

    # BCI
    "beta_market":   0.12,
    "beta_labour":   0.05,
    "beta_supplier": 0.10,
    "bci_method":    "weight_free",   # or "weighted"
    "market_weight":   0.40,
    "labour_weight":   0.35,
    "supplier_weight": 0.25,
    "use_urban_interface": True,
    "interface_lambda":    0.15,

    # OSM tag toggles (BCI suppliers)
    "enabled_supplier_tags": {
        "offices":                True,
        "industrial_commercial":  True,
        "commercial_buildings":   True,
        "wholesale":              True,
        "finance":                True,
    },
}


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_city_config(city_name: str) -> Dict[str, Any]:
    """Return the locked city configuration."""
    if city_name not in CITY_CONFIGS:
        raise KeyError(
            f"City '{city_name}' not in registry. "
            f"Available: {list(CITY_CONFIGS.keys())}"
        )
    return CITY_CONFIGS[city_name].copy()


def list_cities() -> list:
    return [{"key": k, "display": v["display_name"]} for k, v in CITY_CONFIGS.items()]


def get_default_user_params() -> Dict[str, Any]:
    import copy
    return copy.deepcopy(DEFAULT_USER_PARAMS)
