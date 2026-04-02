# Connectivity Pipeline

> See the full documentation in the [root README](../README.md).

This directory contains all source code for the Connectivity Pipeline:

```
core/        Shared building blocks (grid, network, OSM, Census)
pci/         People Connectivity Index — calculator, plots, maps, stats
bci/         Business Connectivity Index — masses, calculator, plots, maps, stats
analysis/    Cross-module analysis: comparative, sensitivity, diagnostics, scenarios
webapp/      Flask web application (app.py, templates, static assets, results)
data/        City data files — GTFS feeds, local boundary polygons (gitignored)
```

## Quick start

```bash
pip install -r requirements.txt
python webapp/app.py
# → http://localhost:5001
```
