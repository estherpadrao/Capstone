# Connectivity Pipeline

A modular, web-deployable pipeline for computing and visualising two urban accessibility indices across H3 hexagonal grids:

- **PCI — People Connectivity Index**: measures residential livability based on how well people can reach amenities by walking, cycling, driving, and transit.
- **BCI — Business Connectivity Index**: measures commercial viability by quantifying access to customers (market), workers (labour), and business suppliers across three component-specific transport networks.

Both indices use a **Hansen gravity model** with exponential distance decay, multi-modal Dijkstra travel-time networks, and **Uber H3 hexagonal grids**.

---

## Demo

> 📹 **Video walkthrough:** *(Loom link coming soon)*

---

## What It Does

### People Connectivity Index (PCI)

For every hex cell in a city, PCI answers: *"how accessible are daily amenities — health, education, parks, food, transit, and community — when weighted by how far a person is realistically willing to travel?"*

The score accounts for:
- Multi-modal travel (walk, bike, drive, transit via GTFS)
- Income-adjusted cost-of-time (richer areas are penalised for high opportunity cost)
- Active street bonus (denser intersections reward walkability)
- Amenity importance weights sourced from Zheng et al. (2021)

### Business Connectivity Index (BCI)

For every hex cell, BCI answers: *"how attractive is this location for businesses, given access to customers, workers, and suppliers?"*

Three independent accessibility components are computed:
- **Market access** — reach weighted by population × purchasing power (walk + transit)
- **Labour access** — reach weighted by employed population (drive + transit)
- **Supplier access** — reach weighted by commercial/industrial density (drive)

Each uses its own decay rate (β) and is combined into a single normalised score.

---

## Full Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                               │
│         City selection · Parameters · OSM tag toggles           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                   ┌────────▼────────┐
                   │   STEP 1: INIT  │
                   │─────────────────│
                   │ • Fetch city    │
                   │   boundary      │
                   │   (OSM/GeoJSON) │
                   │ • Build H3 hex  │
                   │   grid (res 8,  │
                   │   ~460 m cells) │
                   │ • Fetch OSM     │
                   │   amenities +   │
                   │   suppliers     │
                   │ • Compute PCI   │
                   │   mass surface  │
                   │   (weighted +   │
                   │   Gaussian      │
                   │   smoothed)     │
                   │ • Compute BCI   │
                   │   market/labour │
                   │   /supplier     │
                   │   masses        │
                   └────────┬────────┘
                            │
               ┌────────────▼────────────┐
               │   STEP 2: BUILD NETWORK │
               │─────────────────────────│
               │ • Walk network (OSMnx)  │
               │ • Bike network (OSMnx)  │
               │ • Drive network (OSMnx) │
               │ • Transit network       │
               │   (GTFS → stop graph)   │
               │ • Fetch Census ACS:     │
               │   income · population   │
               │   · employed labour     │
               └────────────┬────────────┘
                            │
               ┌────────────▼────────────┐
               │   STEP 3: COMPUTE       │
               │─────────────────────────│
               │ Dijkstra travel times   │
               │ (cached per mode)       │
               │          ↓              │
               │ Hansen accessibility    │
               │ A_i = Σ M_j ×          │
               │   exp(−β × t_ij) ×     │
               │   CostAdj_i            │
               │          ↓              │
               │ PCI: active-street      │
               │ bonus + normalise       │
               │ → score [0–100]         │
               │          ↓              │
               │ BCI: 3 components ×     │
               │ own β → combine         │
               │ → score [0–100]         │
               └────────────┬────────────┘
                            │
          ┌─────────────────▼──────────────────┐
          │              RESULTS                │
          │────────────────────────────────────│
          │  Interactive maps (Folium/Leaflet)  │
          │  Topography & component plots       │
          │  Statistics: mean · Gini · quantile │
          │  Neighbourhood-level breakdown      │
          │  Comparative analysis (PCI vs BCI)  │
          │  Sensitivity & scenario testing     │
          │  Saved to webapp/results/*.pkl      │
          └────────────────────────────────────┘
```

---

## How to Run It

### 1. Install dependencies

```bash
cd connectivity_pipeline
pip install -r requirements.txt
```

Or use the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate connectivity_pipeline
```

### 2. (Optional) Add city data

Place files in `connectivity_pipeline/data/`:

| File | Purpose |
|------|---------|
| `muni_gtfs-current.zip` | GTFS transit feed (highly recommended for transit routing) |
| `sf_polygon.geojson` | Local boundary polygon (optional — fetched from OSM otherwise) |

### 3. Start the web app

```bash
python webapp/app.py
```

Open **http://localhost:5001** in your browser.

---

## What to Expect

### First run — full pipeline (~5–15 min depending on city size)

When you click **▶ PCI** or **▶ BCI** for the first time on a city, the pipeline runs all three steps in sequence:

| Step | What happens | Typical time |
|------|-------------|--------------|
| Init | Boundary fetch, H3 grid, OSM amenity/supplier download, mass surface | 1–3 min |
| Build Network | Walk/bike/drive/transit graph construction, Census ACS fetch | 2–8 min |
| Compute | Dijkstra travel times (all hex pairs), accessibility model, scoring | 1–5 min |

A **status bar** at the bottom of the page shows the current stage and completion.

### Subsequent runs — parameter changes only (~10–60 sec)

After the network is built, changing parameters (β, weights, tag toggles) only reruns the affected downstream steps:

| Change | Recomputes |
|--------|-----------|
| `hansen_beta` | Accessibility → score only |
| Amenity weights | Mass surface → accessibility → score |
| OSM tag toggle | Re-fetch category → mass → accessibility → score |
| BCI β values | That component's accessibility → BCI |
| City change | Everything — full rebuild |

Click **↺ Recompute** in the sidebar after changing parameters.

### Restoring saved results (~5–15 sec)

Click **Restore Saved Results** to reload a previously computed run from disk without re-running the pipeline.

---

## Web App Tour

The app is a single-page interface with a **sidebar** (controls) and **tab panel** (results).

### Sidebar

| Section | What it does |
|---------|-------------|
| **City** | Select the city to analyse |
| **Run Index** | ▶ PCI / ▶ BCI / ▶ Run Both / Restore Saved Results |
| **PCI Parameters** | Hansen β, active street λ, amenity weights, OSM tag toggles |
| **BCI Parameters** | β market/labour/supplier, interface λ, combination method, supplier tags |

Each section has a collapsible **ⓘ explanation box** — click to read what the parameters do.

### Tabs

| Tab | Contents |
|-----|---------|
| **About** | Methodology overview and data sources |
| **PCI** | Topography plots, 3D surface, component breakdown, interactive map, neighbourhood table, distribution |
| **BCI** | Mass maps (market/labour/supplier), component plots, interactive BCI map, distribution |
| **Compare** | Correlation stats, scatter plot, distribution comparison, spatial side-by-side map — unlocks after both PCI and BCI are run |
| **Diagnostics** | Network validation (edge/node counts, mode coverage), mass layer summaries |
| **Sensitivity** | One-at-a-time parameter sensitivity: tornado chart + score-change table. PCI and BCI results stack on top of each other |
| **Scenario Testing** | Interactive Leaflet map; modify amenities, suppliers, or travel-time edges and see the delta on the score map |

---

## Where to Find Results

### In the web app

All results are displayed interactively in the relevant tab immediately after computation.

### On disk

Saved results (pickled session state) are written to:

```
connectivity_pipeline/webapp/results/<City_Name>.pkl
```

The pickle contains the full session dict including the H3 grid with all computed columns, scores, mass layers, and visualisation-ready data. It **does not** include the network graph (too large — rebuilt from OSM/GTFS on restore).

### Outputs directory

If you export any static plots outside the webapp, they land in `Outputs/<CityName>/`.

---

## Project Structure

```
connectivity_pipeline/
│
├── core/                       # Shared building blocks (used by both PCI and BCI)
│   ├── h3_helper.py            # H3 hex grid utilities
│   ├── osm_fetcher.py          # OSM amenity + supplier data fetcher
│   ├── boundary_grid.py        # City boundary fetch + grid construction
│   ├── census_fetcher.py       # US Census ACS + TIGER data
│   ├── mass_calculator.py      # PCI weighted mass surface + Gaussian smoothing
│   ├── network_builder.py      # Multi-modal network (walk/bike/drive/transit)
│   └── city_config.py          # City configuration registry
│
├── pci/
│   ├── pci_calculator.py       # Hansen accessibility model + PCI scoring
│   ├── pci_plots.py            # Topography, 3D surface, component, distribution plots
│   ├── pci_maps.py             # Interactive Folium map + neighbourhood overlays
│   ├── pci_stats.py            # Mean, Gini, quantiles
│   └── pci_analysis.py         # Unified analysis interface (imports from above)
│
├── bci/
│   ├── bci_masses.py           # Market, Labour, Supplier mass calculators
│   ├── bci_calculator.py       # Per-component Hansen models + BCI final score
│   ├── bci_plots.py            # Mass maps, component plots, distribution
│   ├── bci_maps.py             # Interactive BCI Folium map
│   ├── bci_stats.py            # Mean, Gini, quantiles
│   └── bci_analysis.py         # Unified analysis interface
│
├── analysis/
│   ├── comparative_analysis.py # PCI vs BCI: correlations, spatial maps, quadrant
│   ├── sensitivity.py          # One-at-a-time (OAT) parameter sensitivity
│   ├── network_diagnostics.py  # Network validation + topography diagnostics
│   ├── isochrones.py           # Isochrone-based accessibility analysis
│   ├── impact.py               # Scenario testing (fast/slow paths, non-destructive)
│   └── shared.py               # Shared helpers (neighbourhood stats, colour palettes)
│
├── webapp/
│   ├── app.py                  # Flask app — 40+ API routes, session management
│   ├── about.md                # About tab markdown content
│   ├── results/                # Persisted results (city-specific .pkl files)
│   ├── templates/
│   │   ├── index.html          # Single-page app shell
│   │   └── partials/           # One HTML file per tab + sidebar
│   └── static/
│       ├── css/index.css       # Global stylesheet
│       └── js/app.js           # Client-side state + API wrappers + map logic
│
├── data/                       # City data files (gitignored)
│   └── muni_gtfs-current/      # San Francisco GTFS transit feed
│
├── requirements.txt
└── environment.yml
```

---

## Parameter Guide

### City-locked parameters (`core/city_config.py`)

These are set per city and are not exposed in the UI.

| Parameter | Description |
|-----------|-------------|
| `h3_resolution` | Hex grid resolution (8 ≈ 460 m cells) |
| `state_fips` / `county_fips` | US Census FIPS codes |
| `census_year` | ACS 5-year survey year |
| `gtfs_path` | Path to GTFS transit zip |
| `travel_speeds` | Mode-specific speeds (km/h) |
| `travel_costs` | Mode-specific trip costs (USD) |
| `time_penalties` | Transit wait, parking, bike unlock (min) |
| `median_hourly_wage` | For income-adjusted cost-of-time |
| `park_threshold` | Park coverage fraction above which hex is masked |
| `max_travel_time` | Dijkstra cutoff (minutes) |
| `airport_locations` | (lat, lng) pairs for BCI urban interface bonus |

### User-editable parameters (sidebar)

#### PCI

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hansen_beta` | 0.08 | Distance decay β — higher = people prefer closer destinations |
| `active_street_lambda` | 0.30 | Bonus for streets with high intersection density |
| `amenity_weights` | health 0.319, education 0.276, parks 0.255, community 0.148 | Per-category importance weights (Zheng et al. 2021) |
| OSM tag toggles | all on | Enable/disable individual amenity categories |

#### BCI

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta_market` | 0.12 | Customer access decay — customers are more distance-sensitive |
| `beta_labour` | 0.05 | Worker commute decay — workers tolerate longer travel |
| `beta_supplier` | 0.10 | Business service decay |
| `interface_lambda` | 0.15 | Airport/urban-edge proximity bonus weight |
| `bci_method` | weight_free | `weight_free`: equal sum; `weighted`: custom proportions |
| Supplier tag toggles | all on | Toggle OSM commercial categories: offices, industrial, commercial, wholesale, finance |

---

## Adding a New City

1. Open `core/city_config.py`.
2. Add an entry to `CITY_CONFIGS` following the pattern of an existing city.
3. Place the GTFS zip at the path set in `gtfs_path` (under `data/`).
4. For a custom boundary: set `use_local_polygon: True` and point `local_polygon_path` to a GeoJSON file. Otherwise the boundary is fetched automatically from OpenStreetMap.
5. Restart the Flask app — the city appears in the dropdown automatically.

---

## API Reference (key routes)

The Flask app exposes a REST API used by the frontend. You can also call it directly.

```bash
# Run PCI pipeline
curl -X POST http://localhost:5001/api/pci/init \
  -H "Content-Type: application/json" \
  -d '{"city_name": "San Francisco, California, USA"}'
curl -X POST http://localhost:5001/api/pci/build_network
curl -X POST http://localhost:5001/api/pci/compute
curl http://localhost:5001/api/pci/visualize

# Run BCI (reuses PCI network if already built)
curl -X POST http://localhost:5001/api/bci/init
curl -X POST http://localhost:5001/api/bci/build_network
curl -X POST http://localhost:5001/api/bci/compute
curl http://localhost:5001/api/bci/visualize

# Check session state
curl http://localhost:5001/api/session/status

# Restore saved results
curl -X POST http://localhost:5001/api/restore \
  -H "Content-Type: application/json" \
  -d '{"city_name": "San Francisco, California, USA"}'
```

---

## Interpretation Guide

### PCI Score

| Score | Meaning |
|-------|---------|
| 70–100 | Excellent — high multi-modal amenity access |
| 50–70 | Good — moderate connectivity |
| 30–50 | Fair — gaps in coverage or affordability |
| 0–30 | Poor — major connectivity deficits |

### BCI Score

| Score | Meaning |
|-------|---------|
| 70–100 | Prime commercial location — strong access to customers, labour, and suppliers |
| 40–70 | Viable — strong in one or two components |
| 0–40 | Limited commercial potential from an accessibility standpoint |

### Gini Coefficient

0 = perfectly equal distribution of accessibility across all hexes.
1 = entirely concentrated in a single hex. Values above 0.4 indicate significant spatial inequality.

### Quadrant Analysis (Compare tab)

| Quadrant | Interpretation |
|----------|---------------|
| High PCI + High BCI | Mixed-use live-work neighbourhoods |
| High PCI + Low BCI | Residential neighbourhoods with good amenities |
| Low PCI + High BCI | Commercial or industrial zones |
| Low PCI + Low BCI | Underserved areas — connectivity deficits for both residents and businesses |

---

## Data Sources

| Source | Used for |
|--------|---------|
| OpenStreetMap (via OSMnx) | City boundary, street networks, amenities, suppliers |
| Uber H3 | Hexagonal spatial indexing |
| US Census ACS 5-year | Median income, population, employed population |
| Census TIGER | Tract-level geometries for spatial join |
| GTFS (agency-provided) | Transit network and stop locations |

---

## Methodology Summary

### PCI

```
Amenities (OSM) → Weighted mass surface (Gaussian smoothed)
Network (OSMnx + GTFS) → Multi-modal travel times t_ij
Census ACS → Income-adjusted cost-of-time CostAdj_i

A_i = Σ_j  M_j × exp(−β × t_ij) × CostAdj_i
PCI_raw_i = A_i × (1 + λ × StreetDegree_i)
PCI_i = MinMax(PCI_raw) × 100
```

### BCI

```
Market mass  = Population_j × NormIncome_j      → β_market,  walk+transit
Labour mass  = EmployedPop_j                     → β_labour,  drive+transit
Supplier mass = OSM commercial density_j         → β_supplier, drive

BCI_i = f(A_market_i, A_labour_i, A_supplier_i) + Interface_i × λ
      [weight-free: sum of normalised components | weighted: custom mix]
BCI_i = MinMax(BCI_raw) × 100
```

---

## License

MIT
