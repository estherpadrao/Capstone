# Connectivity Pipeline

A modular, web-deployable pipeline for computing and visualising two urban accessibility indices:

- **PCI** — *People Connectivity Index*: measures residential livability based on how well people can reach amenities by walking, cycling, driving, and transit.
- **BCI** — *Business Connectivity Index*: measures commercial viability by quantifying access to customers (market), workers (labour), and suppliers across three component-specific transport networks.

Both indices use a Hansen gravity model with Gaussian distance decay, multi-modal travel-time networks, and H3 hexagonal grids.

---

## Quick Start

```bash
# 1. Clone the repository
cd Capstone

# 2. Install dependencies
pip install -r connectivity_pipeline/requirements.txt

# 3. (Optional) Pre-warm caches before first run — eliminates all live API calls
python connectivity_pipeline/scripts/precompute.py

# 4. Run the web app
python connectivity_pipeline/webapp/app.py

# 5. Open http://localhost:5000
```

---

## Project Structure

```
Capstone/
│
├── connectivity_pipeline/
│   │
│   ├── core/                       # Shared building blocks (PCI + BCI)
│   │   ├── h3_helper.py            # H3 hexagonal grid utilities
│   │   ├── osm_fetcher.py          # OSM amenity + supplier fetcher (disk-cached)
│   │   ├── boundary_grid.py        # City boundary fetch + grid construction
│   │   ├── census_fetcher.py       # US Census ACS + TIGER data (disk-cached)
│   │   ├── mass_calculator.py      # PCI topographic mass surface
│   │   ├── network_builder.py      # Multi-modal network (disk-cached)
│   │   └── city_config.py          # City configuration registry
│   │
│   ├── pci/
│   │   ├── pci_calculator.py       # Hansen accessibility + PCI score
│   │   ├── pci_plots.py            # Matplotlib charts (topography, distribution)
│   │   ├── pci_maps.py             # Folium interactive maps
│   │   ├── pci_stats.py            # Descriptive statistics
│   │   └── pci_analysis.py         # Compatibility shim (re-exports all of the above)
│   │
│   ├── bci/
│   │   ├── bci_masses.py           # Market, Labour, Supplier mass calculators
│   │   ├── bci_calculator.py       # BCI Hansen model + final score
│   │   └── bci_analysis.py         # BCI maps, distributions, statistics
│   │
│   ├── analysis/
│   │   ├── comparative_analysis.py # PCI vs BCI comparison
│   │   ├── sensitivity.py          # Parameter sensitivity (tornado charts)
│   │   ├── network_diagnostics.py  # Network validation + mass topography
│   │   ├── impact.py               # Scenario testing (edge/amenity/supplier changes)
│   │   ├── isochrones.py           # Isochrone analysis
│   │   └── shared.py               # Shared plot helpers + neighbourhood stats
│   │
│   ├── webapp/
│   │   ├── app.py                  # Flask web application (all API routes)
│   │   ├── about.md                # About tab content (markdown)
│   │   ├── templates/
│   │   │   └── index.html          # Single-page shell + tab structure
│   │   ├── static/
│   │   │   ├── css/index.css       # Dark-theme stylesheet
│   │   │   └── js/app.js           # All frontend logic
│   │   └── results/                # Persisted PCI/BCI results (gitignored)
│   │       └── <City>.pkl
│   │
│   ├── data/                       # City data files (gitignored)
│   │   ├── cache/                  # Pre-warmed data cache (network, OSM, Census)
│   │   ├── muni_gtfs-current.zip   # GTFS transit feed
│   │   └── sf_polygon.geojson      # Optional local boundary polygon
│   │
│   ├── scripts/
│   │   └── precompute.py           # CLI: warm all caches before deployment
│   │
│   └── requirements.txt
│
└── README.md
```

---

## Pre-Warming Caches (Recommended Before Deployment)

The first run for a city fetches data from OpenStreetMap, the US Census API, and builds the transport network. This takes 5–20 minutes. Running `precompute.py` once locally saves everything to `data/cache/` so the server never makes live API calls:

```bash
# Warm all configured cities
python connectivity_pipeline/scripts/precompute.py

# Single city only
python connectivity_pipeline/scripts/precompute.py --city "San Francisco, California, USA"

# Skip the network build (fastest; still caches OSM + Census)
python connectivity_pipeline/scripts/precompute.py --skip-network
```

Cache files written:
| File | Contents |
|------|----------|
| `network_{hash}.pkl` | Multi-modal transport graph |
| `amenity_{hash}_{name}.pkl` | OSM amenity GeoDataFrame per category |
| `supplier_{hash}_{name}.pkl` | OSM supplier GeoDataFrame per category |
| `acs_{key}.pkl` | Census ACS income/population/labour |
| `tiger_{key}.pkl` | Census TIGER tract geometries |
| `tt_pci_{city}.pkl` | PCI Dijkstra travel times |
| `tt_bci_{city}.pkl` | BCI Dijkstra travel times (3 components) |
| `neighborhoods_{city}.pkl` | Hex→neighbourhood assignment |

Upload `data/cache/` to your server alongside the rest of the project.

---

## Parameter Guide

### City-locked parameters (`core/city_config.py`)

| Parameter | Description |
|-----------|-------------|
| `h3_resolution` | Hex grid resolution (8 = ~460 m hexes) |
| `state_fips` / `county_fips` | US Census FIPS codes |
| `census_year` | ACS 5-year survey year |
| `gtfs_path` | Path to GTFS transit zip |
| `travel_speeds` | Mode-specific speeds (km/h) |
| `travel_costs` | Mode-specific trip costs (USD) |
| `time_penalties` | Transit wait, parking search, bike unlock (min) |
| `median_hourly_wage` | Used for income-adjusted cost-of-time |
| `park_threshold` | Park coverage fraction above which hex is masked |
| `max_travel_time` | Dijkstra cutoff (minutes) |
| `airport_locations` | (lat, lng) pairs for BCI urban interface bonus |

### User-editable parameters (webapp sidebar)

#### PCI
| Parameter | Default | Description |
|-----------|---------|-------------|
| `hansen_beta` | 0.08 | Distance decay β — higher = people value only nearby amenities |
| `active_street_lambda` | 0.30 | Bonus weight for walkable/cyclable streets |
| `amenity_weights` | health 0.319, edu 0.276, parks 0.255, community 0.148 | Importance per amenity type (Zheng et al. 2021) |
| OSM tag toggles | all on | Enable/disable amenity categories |

#### BCI
| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta_market` | 0.12 | Decay for customer access |
| `beta_labour` | 0.05 | Decay for worker access |
| `beta_supplier` | 0.10 | Decay for supplier access |
| `interface_lambda` | 0.15 | Airport/urban-edge proximity bonus weight |
| `bci_method` | weight_free | `weight_free` / `weighted` / `min` |
| `market_weight` | 0.40 | (weighted mode only) |
| `labour_weight` | 0.35 | (weighted mode only) |
| `supplier_weight` | 0.25 | (weighted mode only) |
| Supplier tag toggles | all on | Toggle OSM supplier categories |

---

## Webapp Features

### Tabs (default landing: About)
| Tab | Description |
|-----|-------------|
| **About** | Project overview, methodology summary (default on open) |
| **PCI** | Run PCI, view maps, topography, distributions, neighbourhood table |
| **BCI** | Run BCI, view maps, component breakdown, neighbourhood table |
| **Compare** | Auto-loads after both PCI and BCI are computed; scatter, spatial, correlations |
| **Diagnostics** | Network validation stats and mass topography summary |
| **Sensitivity** | Tornado charts for PCI and BCI — results stack (PCI + BCI shown simultaneously) |
| **Scenario Testing** | Add/remove amenities or suppliers, penalise/remove network edges; see delta maps |

### Caching behaviour
- **Same session, same city**: grid, amenities, network, census, and Dijkstra travel times are all reused — changing parameters only reruns the fast scoring step.
- **After server restart**: travel times are reloaded from disk (`tt_pci_*.pkl`, `tt_bci_*.pkl`); a full Dijkstra is only needed if no cache exists.
- **Scenario network map**: the folium network map is cached in session memory after first render; re-opening the scenario tab is instant.

### Browser notifications
The app requests notification permission on first load. When a computation finishes while you are on another browser tab, a system notification is sent for PCI, BCI, sensitivity, and scenario runs.

---

## Modular Recomputation

| Change | What reruns |
|--------|-------------|
| `hansen_beta` | Accessibility → PCI only |
| `amenity_weights` | Mass → Accessibility → PCI |
| OSM tag toggle | Re-fetch category → Mass → Accessibility → PCI |
| `beta_market/labour/supplier` | BCI accessibility → BCI |
| City change | Full rebuild (all caches still reused where applicable) |

---

## Adding a New City

1. Open `core/city_config.py` and add an entry to `CITY_CONFIGS`.
2. Place the GTFS zip at `gtfs_path` (under `data/`).
3. If you have a boundary GeoJSON, set `use_local_polygon: True` and `local_polygon_path`.
4. Run `python connectivity_pipeline/scripts/precompute.py --city "Your City, State, Country"` to warm caches.
5. Restart the Flask app — the city appears in the dropdown automatically.

---

## Methodology

### PCI — People Connectivity Index

1. **H3 grid**: City clipped to H3 resolution-8 hexagons (~460 m).
2. **Amenity fetch**: Health, education, parks, community, food/retail, transit from OSM.
3. **Mass surface**: Weighted composite (Zheng et al. 2021 weights), Gaussian-smoothed.
4. **Network**: Walk + bike + drive + transit (GTFS if available, OSM stops as fallback).
5. **Income adjustment**: Census ACS tract-level median household income → cost-of-time penalty per hex.
6. **Hansen accessibility**: `A_i = Σ_j M_j × exp(−β × t_ij) × CostAdj_i`
7. **Active street bonus**: `PCI_raw = A_i × (1 + λ × Degree_i)`
8. **Normalisation**: Min-max to [0, 100]. Park-dominated hexes optionally masked.
9. **City score**: Area-weighted mean. Gini coefficient for equity.

### BCI — Business Connectivity Index

1. **Three masses**: Market (Population × Income), Labour (employed pop.), Supplier (OSM business density).
2. **Component networks**: Market = walk+transit; Labour = drive+transit; Supplier = drive only.
3. **Per-component Hansen**: Separate β per component.
4. **Urban interface**: Airport proximity + city-edge score × λ.
5. **Combination**: Weight-free (`BCI = ΣA_k/max`), weighted average, or min.
6. **Normalisation**: [0, 100].

### Comparative Analysis

Runs automatically after both PCI and BCI are computed in the same session:
- Pearson, Spearman, Kendall correlations
- Distribution comparisons (histogram, box, violin, Q-Q)
- Spatial maps: PCI, BCI, difference (PCI−BCI), quadrant classification
- Interactive folium layer-control map

---

## Data Sources

| Source | Used for |
|--------|----------|
| OpenStreetMap (via OSMnx) | City boundary, street networks, amenities, suppliers |
| Uber H3 | Hexagonal spatial indexing |
| US Census ACS 5-year | Median income, population, employed population |
| Census TIGER | Tract-level geometries for spatial join |
| GTFS (agency-provided) | Transit network and stop locations |

---

## Interpretation Guide

| PCI | Meaning |
|-----|---------|
| 70–100 | Excellent — high multi-modal amenity access |
| 50–70  | Good — moderate connectivity |
| 30–50  | Fair — gaps in coverage or affordability |
| 0–30   | Poor — major connectivity deficits |

**Gini coefficient**: 0 = perfectly equal; 1 = entirely concentrated.

**BCI hotspots** (top 10%): locations most attractive to businesses due to combined access.

**Quadrant analysis** (Compare tab):
- High PCI + High BCI: mixed-use live-work neighbourhoods
- High PCI + Low BCI: primarily residential
- Low PCI + High BCI: commercial or industrial zones
- Low PCI + Low BCI: underserved areas

---

## License

MIT
