PyClimaExplorer 🌍 – HackItOut 2026
===================================

Interactive Streamlit dashboard to explore NetCDF climate datasets (ensembles, forced runs, observations) with quick visualizations for spatial fields, time-series trends, model-vs-observation comparisons, and warming hotspots.

Table of Contents
-----------------
- Overview
- Features
- Project Structure
- Setup
- Running the App
- Usage Guide (Analysis Modes)
- Datasets
- Troubleshooting
- Acknowledgements

Overview
--------
PyClimaExplorer was built for the HackItOut hackathon to make climate diagnostics fast and approachable. Drop in a NetCDF file (or use the bundled sample) and interactively explore trends, spatial patterns, and hotspots without writing code.

Features
--------
- Upload or use bundled NetCDF sample; automatic caching and CF decoding.
- Four analysis modes:
	- Global Climate Map (yearly slice visualized on lat/lon).
	- Time Series Trend at a chosen lat/lon.
	- Model vs Observation Comparison on matching variables.
	- Climate Hotspots (recent minus baseline warming).
- Smart coordinate detection (time/date dims, lat/lon) with safe fallbacks.
- Interactive controls: variable picker, year slider, lat/lon sliders, custom baseline/recent periods.
- Plotly charts + data tables for top hotspot regions.
- Streamlit sidebar summaries of dataset dimensions, coords, variables, and attributes.

Project Structure
-----------------
- app.py — Streamlit entrypoint and UI flow.
- modules/
	- data_loader.py — dataset I/O, caching, coord helpers, summaries.
	- global_map.py — spatial field visualization.
	- time_series.py — point time-series trends.
	- comparison.py — model vs observation line comparison.
	- hotspots.py — warming hotspots and leaderboard table.
- datasets/ — place NetCDF files here (a sample is expected at datasets/sample.nc if available).
- requirements.txt — Python dependencies.

Setup
-----
1) Install Python 3.10+.
2) (Recommended) Create and activate a virtual environment:
	 - Windows (PowerShell): `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
	 - macOS/Linux: `python -m venv .venv && source .venv/bin/activate`
3) Install dependencies: `pip install -r requirements.txt`

Running the App
---------------
From the project root:
```
streamlit run app.py
```
Then open the local URL Streamlit prints (typically http://localhost:8501).

Usage Guide (Analysis Modes)
----------------------------
- Global Climate Map: Pick a variable with lat/lon and a target year; view spatial distribution.
- Time Series Trend: Pick a variable with time + lat/lon; choose the time coordinate if multiple exist; view trend at nearest point.
- Model vs Observation Comparison: Provide two datasets (or reuse primary) and compare the selected variable at the chosen point.
- Climate Hotspots: Define baseline and recent periods; see Δ between means and a ranked table of hottest grid points.

Datasets
--------
- You can upload NetCDF (.nc) directly or .tar/.tgz bundles.
- Sample lookup: the app looks for datasets/sample.nc; otherwise it loads the first .nc inside datasets/.
- Included folder lists CESM1/2 ensemble NetCDF files (large); keep only what you need to save space.

Troubleshooting
---------------
- Missing h5netcdf/h5py error: `pip install -r requirements.txt` to ensure backends are present.
- No time coordinate found: ensure your variable has a time-like coord (time/date). Use Time Series/Hotspots only on variables with time.
- No lat/lon dims: Global Map and Hotspots require both; switch to Time Series if analyzing a non-spatial variable.
- Cache issues: modify the dataset file (touch) or restart Streamlit to invalidate cached loads.

Acknowledgements
----------------
- Built with Streamlit, xarray, NetCDF4/h5netcdf, pandas, numpy, plotly.
- Climate datasets courtesy of CESM/BEST collections (see datasets/ for filenames).
# PyClimaExplorer
# PyClimaExplorer
