# Connectivity Pipeline

A modular, web-deployable pipeline for computing and visualising two urban accessibility indices:

- **PCI** — *People Connectivity Index*: measures residential livability based on how well people can reach amenities by walking, cycling, driving, and transit.
- **BCI** — *Business Connectivity Index*: measures commercial viability by quantifying access to customers (market), workers (labour), and suppliers across three component-specific transport networks.

Both indices use a Hansen gravity model with Gaussian distance decay, multi-modal travel-time networks, and H3 hexagonal grids.

---

## Quick Start

```bash
# 1. Clone / unzip the project
cd connectivity_pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Place city data in data/
#    - GTFS zip:        data/muni_gtfs-current.zip  (or whatever filename you configure)
#    - Local boundary:  data/sf_polygon.geojson

# 4. Run the web app
python webapp/app.py

# 5. Open http://localhost:5000 in your browser
```

---

## Project Structure

```
connectivity_pipeline/
│
├── core/                       # Shared building blocks (PCI + BCI both use these)
│   ├── h3_helper.py            # File 1  — H3 hexagonal grid utilities
│   ├── osm_fetcher.py          # File 2  — OSM amenity + supplier data fetcher
│   ├── boundary_grid.py        # File 3  — City boundary fetch + grid construction
│   ├── census_fetcher.py       # File 4  — US Census ACS + TIGER data
│   ├── mass_calculator.py      # File 5  — PCI topographic mass surface
│   ├── network_builder.py      # File 6  — Multi-modal network (walk/bike/drive/transit)
│   └── city_config.py          #          — City configuration registry
│
├── pci/
│   ├── pci_calculator.py       # File 7  — Hansen accessibility + PCI score
│   └── pci_analysis.py         # File 8  — PCI maps, distributions, statistics
│
├── bci/
│   ├── bci_masses.py           # Files 9–11 — Market, Labour, Supplier mass calculators
│   ├── bci_calculator.py       # File 12 — BCI Hansen model + final score
│   └── bci_analysis.py         # File 13 — BCI maps, distributions, statistics
│
├── analysis/
│   ├── comparative_analysis.py # File 14 — PCI vs BCI comparison (only if both run)
│   └── isochrones.py           # File 15 — Isochrone validation + network diagnostics
│
├── webapp/
│   ├── app.py                  # Flask web application (API + UI)
│   ├── templates/
│   │   └── index.html          # Single-page frontend
│   └── static/                 # CSS/JS assets (optional overrides)
│
├── data/                       # City data files (gitignored)
│   ├── muni_gtfs-current.zip   # GTFS transit feed
│   └── sf_polygon.geojson      # Optional local boundary polygon
│
├── requirements.txt
└── README.md
```

---

## Parameter Guide

### City-locked parameters (configured in `core/city_config.py`, not user-editable)

| Parameter | Description |
|-----------|-------------|
| `h3_resolution` | Hex grid resolution (8 = ~460m hexes) |
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

### User-editable parameters (available in the webapp sidebar)

#### PCI
| Parameter | Default | Description |
|-----------|---------|-------------|
| `hansen_beta` | 0.08 | Global distance decay β — higher = people prefer closer destinations |
| `active_street_lambda` | 0.30 | Bonus weight for streets with high intersection density |
| `amenity_weights` | health 0.319, education 0.276, parks 0.255, community 0.148 | Importance weights per amenity type (Zheng et al. 2021) |
| OSM tag toggles | all on | Enable/disable individual amenity categories from OSM fetch |

#### BCI
| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta_market` | 0.12 | Decay for customer access — customers are more distance-sensitive |
| `beta_labour` | 0.05 | Decay for worker access — workers tolerate longer commutes |
| `beta_supplier` | 0.10 | Decay for supplier/business service access |
| `interface_lambda` | 0.15 | Weight of airport/urban-edge proximity bonus |
| `bci_method` | weight_free | `weight_free`: BCI = sum of normalised components; `weighted`: custom weights |
| `market_weight` | 0.40 | (weighted mode only) |
| `labour_weight` | 0.35 | (weighted mode only) |
| `supplier_weight` | 0.25 | (weighted mode only) |
| Supplier tag toggles | all on | Toggle OSM supplier categories: offices, industrial, commercial, wholesale, finance |

---

## Modular Recomputation

The pipeline is designed so that changing a parameter only rerenders what is necessary:

| Change | What is recomputed |
|--------|-------------------|
| `hansen_beta` | Accessibility scores → PCI only |
| `amenity_weights` | Mass surface → Accessibility → PCI |
| OSM tag toggle | Re-fetch affected category → Mass → Accessibility → PCI |
| `beta_market/labour/supplier` | BCI travel times for affected component → BCI |
| Network (city change) | Everything — full rebuild |

The **↺ Recompute** button in the sidebar appears when any parameter is changed. It skips the slow network and OSM fetch steps and only reruns the accessibility and scoring computations.

---

## Adding a New City

1. Open `core/city_config.py`.
2. Add a new entry to `CITY_CONFIGS` following the pattern of an existing city.
3. Place the GTFS zip at the path specified in `gtfs_path` (under `data/`).
4. If you have a boundary GeoJSON, set `use_local_polygon: True` and point `local_polygon_path` to it; otherwise the boundary is fetched from OpenStreetMap.
5. Restart the Flask app — the new city appears in the dropdown automatically.

---

## Running PCI and BCI Independently

PCI and BCI can be run independently. They share the boundary, grid, network, and census data when run together (BCI reuses all of PCI's expensive infrastructure), but each can be triggered separately from the webapp or the API.

**PCI only:**
```bash
curl -X POST http://localhost:5000/api/pci/init -H "Content-Type: application/json" \
  -d '{"city_name": "San Francisco, California, USA"}'
curl -X POST http://localhost:5000/api/pci/build_network
curl -X POST http://localhost:5000/api/pci/compute
curl http://localhost:5000/api/pci/visualize
```

**BCI only (after PCI has been run, the network is reused):**
```bash
curl -X POST http://localhost:5000/api/bci/init
curl -X POST http://localhost:5000/api/bci/build_network
curl -X POST http://localhost:5000/api/bci/compute
curl http://localhost:5000/api/bci/visualize
```

---

## Methodology

### PCI — People Connectivity Index

1. **H3 grid**: City clipped to H3 resolution-8 hexagons (~460m).
2. **Amenity fetch**: Health, education, parks, community, food/retail, transit from OSM.
3. **Mass surface**: Weighted composite (Zheng et al. 2021 weights), Gaussian-smoothed.
4. **Network**: Walk + bike + drive + transit (GTFS if available, OSM stops as fallback).
5. **Income adjustment**: Census ACS tract-level median household income → cost-of-time penalty per hex.
6. **Hansen accessibility**: `A_i = Σ_j M_j × exp(−β × t_ij) × CostAdj_i`
7. **Active street bonus**: `PCI_raw = A_i × (1 + λ × Degree_i)` where Degree is street network intersection density.
8. **Normalisation**: Min-max to [0, 100]. Park-dominated hexes optionally masked.
9. **City score**: Area-weighted mean. Gini coefficient for equity.

### BCI — Business Connectivity Index

1. **Three masses**:
   - Market Mass = Population × Normalised Income (purchasing power)
   - Labour Mass = Employed population 16+ (Census ACS)
   - Supplier Mass = OSM business/commercial feature density
2. **Component networks**: Market uses walk+transit; Labour uses drive+transit; Supplier uses drive only.
3. **Per-component Hansen**: Separate β for each component.
4. **Urban interface**: Airport proximity + city-edge score × λ.
5. **Combination**: Weight-free (`BCI = ΣA_k/max`) or weighted.
6. **Normalisation**: [0, 100].

### Comparative Analysis

Available when both indices are computed. Includes:
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

**Gini coefficient**: 0 = perfectly equal distribution across hexes; 1 = entirely concentrated.

**BCI hotspots** (top 10%): locations most attractive to businesses due to combined customer, worker, and supplier access.

**Quadrant analysis**:
- High PCI + High BCI: mixed-use live-work neighbourhoods
- High PCI + Low BCI: residential neighbourhoods
- Low PCI + High BCI: commercial or industrial zones
- Low PCI + Low BCI: underserved areas

---

## License

MIT
