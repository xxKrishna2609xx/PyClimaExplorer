from __future__ import annotations
from pathlib import Path
from typing import Tuple

import streamlit as st

from modules.comparison import show_model_comparison
from modules.data_loader import (
	dataset_lat_bounds,
	dataset_lon_bounds,
	dataset_variables,
	dataset_year_bounds,
	load_dataset,
	summarize_dataset,
	variables_with_lat_lon,
	variables_with_time_dim,
	time_coord_candidates,
)
from modules.global_map import show_global_map
from modules.hotspots import show_hotspots
from modules.time_series import show_time_series


SAMPLE_DATA_PATH = Path("datasets/sample.nc")


def _preferred_local_dataset() -> tuple[str, Path] | tuple[None, None]:
	if SAMPLE_DATA_PATH.exists():
		return SAMPLE_DATA_PATH.name, SAMPLE_DATA_PATH
	alt_files = list(Path("datasets").glob("*.nc"))
	if alt_files:
		path = alt_files[0]
		return path.name, path
	return (None, None)


def _render_dataset_summary(ds_label: str, ds) -> None:
	summary = summarize_dataset(ds)
	with st.expander(f"Dataset summary • {ds_label}"):
		st.write("Dimensions", summary["dimensions"])
		st.write("Coordinates", summary["coords"])
		st.write("Variables", summary["variables"])
		if summary["attributes"]:
			st.write("Attributes")
			st.json(summary["attributes"])


def _prepare_dataset(source_choice: str, uploaded_file) -> Tuple[str, object]:
	"""Load primary dataset based on sidebar selection."""

	if source_choice == "Sample dataset":
		label, path = _preferred_local_dataset()
		if path is None:
			st.error("No sample dataset found in the datasets folder. Please upload a NetCDF file.")
			st.stop()
		ds = load_dataset(path)
		return label, ds
	if uploaded_file is None:
		st.info("Upload a NetCDF (.nc) file or a .tar containing NetCDF data to begin.")
		st.stop()
	content = uploaded_file.read()
	ds = load_dataset(content)
	return uploaded_file.name, ds


def _load_optional_dataset(title: str, fallback_ds, uploader_key: str):
	st.sidebar.markdown(f"**{title}**")
	use_primary = st.sidebar.radio("Source", ["Use primary dataset", "Upload"], key=f"{uploader_key}_choice")
	if use_primary == "Use primary dataset":
		return fallback_ds, "Primary"
	uploaded = st.sidebar.file_uploader("Upload NetCDF or .tar", type=["nc", "tar", "tgz", "gz"], key=uploader_key)
	if uploaded is None:
		st.warning(f"Please upload a file for {title.lower()}.")
		st.stop()
	content = uploaded.read()
	return load_dataset(content), uploaded.name


def main() -> None:
	st.set_page_config(page_title="PyClimaExplorer", layout="wide")
	st.title("🌍 PyClimaExplorer – Interactive Climate Data Dashboard")
	st.caption("Explore climate fields, trends, and hotspots interactively.")

	st.sidebar.header("Controls")
	dataset_choice = st.sidebar.radio("Dataset", ["Sample dataset", "Upload dataset"], index=0)
	uploaded_main = st.sidebar.file_uploader("Upload NetCDF or .tar", type=["nc", "tar", "tgz", "gz"])

	try:
		ds_label, ds = _prepare_dataset(dataset_choice, uploaded_main)
	except Exception as exc:  # defensive: show helpful errors
		st.error(f"Failed to load dataset: {exc}")
		st.stop()

	analysis_mode = st.sidebar.selectbox(
		"Analysis mode",
		[
			"Global Climate Map",
			"Time Series Trend",
			"Model vs Observation Comparison",
			"Climate Hotspots",
		],
	)

	# Filter variables based on chosen analysis mode
	if analysis_mode == "Global Climate Map":
		variables = variables_with_lat_lon(ds)
		if not variables:
			st.error("No variables with lat/lon dimensions are available for mapping.")
			st.stop()
	elif analysis_mode in {"Time Series Trend", "Model vs Observation Comparison"}:
		variables = variables_with_time_dim(ds)
		if not variables:
			st.error("No variables with a time dimension are available for this analysis.")
			st.stop()
	elif analysis_mode == "Climate Hotspots":
		variables = variables_with_lat_lon(ds)
		if not variables:
			st.error("No variables with lat/lon dimensions are available for hotspots.")
			st.stop()
	else:
		variables = dataset_variables(ds)
		if not variables:
			st.error("No numeric variables found in this dataset.")
			st.stop()

	variable = st.sidebar.selectbox("Variable", variables)

	min_year, max_year = dataset_year_bounds(ds)
	if min_year == max_year:
		year = min_year
		st.sidebar.info(f"Single year detected: {year}")
	else:
		year_default = min(max_year, max(min_year, 2000))
		year = st.sidebar.slider("Year", min_year, max_year, value=year_default)

	lat_min, lat_max = dataset_lat_bounds(ds)
	lon_min, lon_max = dataset_lon_bounds(ds)
	lat = st.sidebar.slider("Latitude", float(lat_min), float(lat_max), value=float((lat_min + lat_max) / 2))
	lon = st.sidebar.slider("Longitude", float(lon_min), float(lon_max), value=float((lon_min + lon_max) / 2))

	_render_dataset_summary(ds_label, ds)

	if analysis_mode == "Global Climate Map":
		try:
			with st.spinner("Rendering map..."):
				fig = show_global_map(ds, variable, year)
			st.plotly_chart(fig, use_container_width=True)
		except Exception as exc:
			st.error(f"Unable to render map: {exc}")

	elif analysis_mode == "Time Series Trend":
		data_var = ds[variable]
		candidates = time_coord_candidates(ds, data=data_var)
		if not candidates:
			st.error("Selected variable has no time-like dimension. Choose another variable or add a time axis.")
			st.stop()
		time_coord = st.sidebar.selectbox("Time coordinate", candidates, index=0)
		try:
			with st.spinner("Building time series..."):
				fig = show_time_series(ds, variable, lat, lon, time_coord=time_coord)
			st.plotly_chart(fig, use_container_width=True)
		except Exception as exc:
			st.error(f"Unable to build time series: {exc}")

	elif analysis_mode == "Model vs Observation Comparison":
		model_ds, model_label = _load_optional_dataset("Model dataset", ds, "model_upload")
		obs_ds, obs_label = _load_optional_dataset("Observed dataset", ds, "obs_upload")
		try:
			with st.spinner("Comparing time series..."):
				fig = show_model_comparison(model_ds, obs_ds, variable, lat, lon)
			st.plotly_chart(fig, use_container_width=True)
			st.caption(f"Model: {model_label} • Observed: {obs_label}")
		except Exception as exc:
			st.error(f"Unable to compare datasets: {exc}")

	elif analysis_mode == "Climate Hotspots":
		if min_year == max_year:
			st.sidebar.info("Single-year dataset; hotspot periods collapse to the available year.")
			baseline_years = (min_year, max_year)
			recent_years = (min_year, max_year)
		else:
			baseline_start = max(min_year, min(max_year, 1950))
			baseline_end = max(baseline_start + 1, min(max_year, 1970))
			recent_start = max(min_year, min(max_year, 2000))
			recent_end = max(recent_start + 1, max_year)
			baseline_years = st.sidebar.slider(
				"Baseline period",
				min_year,
				max_year,
				value=(baseline_start, baseline_end) if baseline_start < baseline_end else (min_year, max_year),
			)
			recent_years = st.sidebar.slider(
				"Recent period",
				min_year,
				max_year,
				value=(recent_start, recent_end) if recent_start < recent_end else (min_year, max_year),
			)
		try:
			with st.spinner("Detecting hotspots..."):
				fig, table_df = show_hotspots(ds, variable, baseline_years, recent_years)
			st.plotly_chart(fig, use_container_width=True)
			if table_df is not None:
				st.subheader("Top warming regions")
				st.dataframe(table_df, use_container_width=True)
		except Exception as exc:
			st.error(f"Unable to detect hotspots: {exc}")


if __name__ == "__main__":
	main()
