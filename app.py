from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

from modules.data_loader import (
	dataset_lat_bounds,
	dataset_lon_bounds,
	dataset_variables,
	dataset_year_bounds,
	get_time_index,
	load_dataset,
	summarize_dataset,
	time_coord_candidates,
	variables_with_lat_lon,
	variables_with_time_dim,
)
from modules.global_map import show_global_map
from modules.hotspots import show_hotspots


SAMPLE_DATA_PATH = Path("datasets/sample.nc")


def _inject_custom_css() -> None:
	st.markdown(
		"""
		<style>
			.stApp {
				background: radial-gradient(circle at top left, #0d1b2a 0%, #0a101f 45%, #070b14 100%);
				color: #e7ecf3;
			}
			section.main > div {
				padding-top: 0.8rem;
			}
			div.block-container {
				padding-top: 0.9rem;
				padding-right: 1.4rem;
				padding-left: 1.4rem;
				padding-bottom: 1.2rem;
				max-width: 100%;
			}
			section[data-testid="stSidebar"] {
				background: linear-gradient(180deg, rgba(9, 20, 36, 0.94) 0%, rgba(6, 13, 24, 0.92) 100%);
				border-right: 1px solid rgba(120, 145, 170, 0.2);
			}
			[data-testid="stMetric"] {
				background: rgba(14, 26, 43, 0.7);
				border: 1px solid rgba(88, 115, 145, 0.32);
				border-radius: 12px;
				padding: 0.55rem 0.75rem;
			}
			[data-testid="stMetricLabel"] {
				color: #9fb3c8;
			}
			[data-testid="stMetricValue"] {
				color: #f3f7fb;
			}
		</style>
		""",
		unsafe_allow_html=True,
	)


def _preferred_local_dataset() -> tuple[str | None, Path | None]:
	if SAMPLE_DATA_PATH.exists():
		return SAMPLE_DATA_PATH.name, SAMPLE_DATA_PATH
	alt_files = list(Path("datasets").glob("*.nc"))
	if alt_files:
		path = alt_files[0]
		return path.name, path
	return (None, None)


def _prepare_dataset(source_choice: str, uploaded_file) -> tuple[str, xr.Dataset]:
	if source_choice == "Sample dataset":
		label, path = _preferred_local_dataset()
		if path is None:
			st.error("No sample dataset found in the datasets folder. Please upload a NetCDF file.")
			st.stop()
		ds = load_dataset(path)
		return str(label), ds

	if uploaded_file is None:
		st.info("Upload a NetCDF (.nc) file or a .tar containing NetCDF data to begin.")
		st.stop()

	content = uploaded_file.read()
	ds = load_dataset(content)
	return uploaded_file.name, ds


def _load_optional_dataset(title: str, fallback_ds: xr.Dataset, uploader_key: str) -> tuple[xr.Dataset, str]:
	st.sidebar.markdown(f"**{title}**")
	source_choice = st.sidebar.radio("Source", ["Use primary dataset", "Upload"], key=f"{uploader_key}_source")
	if source_choice == "Use primary dataset":
		return fallback_ds, "Primary"

	uploaded = st.sidebar.file_uploader("Upload NetCDF or .tar", type=["nc", "tar", "tgz", "gz"], key=uploader_key)
	if uploaded is None:
		st.warning(f"Please upload a file for {title.lower()}.")
		st.stop()

	return load_dataset(uploaded.read()), uploaded.name


def _render_dataset_summary(ds_label: str, ds: xr.Dataset) -> None:
	summary = summarize_dataset(ds)
	with st.expander(f"Dataset summary - {ds_label}"):
		st.write("Dimensions", summary["dimensions"])
		st.write("Coordinates", summary["coords"])
		st.write("Variables", summary["variables"])
		if summary["attributes"]:
			st.write("Attributes")
			st.json(summary["attributes"])


def _is_time_like(name: str) -> bool:
	lower = name.lower()
	return lower in {"time", "date", "datetime", "timestamp"} or "time" in lower or "date" in lower


def _pick_time_coord(ds: xr.Dataset, data: xr.DataArray) -> str:
	candidates = [name for name in time_coord_candidates(ds, data=data) if name in data.dims or name in data.coords]
	for candidate in candidates:
		idx = get_time_index(ds, coord_name=candidate, data=data)
		if idx is not None and len(idx) > 0:
			return candidate

	for dim in data.dims:
		if _is_time_like(dim):
			return dim

	raise ValueError("Could not find a valid time coordinate for this variable")


def _extract_series(
	ds: xr.Dataset,
	variable: str,
	lat: float,
	lon: float,
	time_coord: str | None = None,
) -> pd.Series:
	if variable not in ds:
		raise ValueError(f"Variable '{variable}' was not found in the selected dataset")

	data = ds[variable]
	if "lat" in data.dims:
		data = data.sel(lat=lat, method="nearest")
	if "lon" in data.dims:
		data = data.sel(lon=lon, method="nearest")

	selected_time_coord = time_coord or _pick_time_coord(ds, data)
	if selected_time_coord not in data.dims and selected_time_coord not in data.coords:
		raise ValueError(f"Time coordinate '{selected_time_coord}' is not present for variable '{variable}'")

	for dim in list(data.dims):
		if dim == selected_time_coord:
			continue
		if dim in {"lat", "lon"}:
			continue
		data = data.mean(dim=dim, skipna=True)

	data = data.sortby(selected_time_coord)
	idx = get_time_index(ds, coord_name=selected_time_coord, data=data)
	if idx is None:
		raise ValueError("Dataset time coordinate could not be interpreted as dates")

	values = np.asarray(data.values, dtype="float64").reshape(-1)
	if len(values) != len(idx):
		values = values[: len(idx)]

	series = pd.Series(values, index=pd.DatetimeIndex(idx), name=variable)
	series = series.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
	if series.empty:
		raise ValueError("No valid values found for the selected variable and location")
	return series


def _align_series(model_series: pd.Series, obs_series: pd.Series) -> pd.DataFrame:
	df = pd.concat(
		[
			model_series.rename("Model"),
			obs_series.rename("Observed"),
		],
		axis=1,
		join="inner",
	).dropna()
	if df.empty:
		raise ValueError("No overlapping timestamps between model and observed series")
	df["Difference"] = df["Model"] - df["Observed"]
	return df.sort_index()


def _compute_trend(series: pd.Series) -> pd.Series:
	y = np.asarray(series.values, dtype="float64")
	x = np.arange(len(y), dtype="float64")
	valid = np.isfinite(y)
	if valid.sum() < 2:
		return pd.Series(np.full_like(y, np.nan), index=series.index)

	slope, intercept = np.polyfit(x[valid], y[valid], deg=1)
	trend = slope * x + intercept
	return pd.Series(trend, index=series.index)


def _variable_category(variable: str) -> str:
	name = variable.lower()
	if "tas" in name or "temp" in name:
		return "Temperature"
	if name.startswith("pr") or "precip" in name or "rain" in name:
		return "Precipitation"
	if "sst" in name or "sea_surface" in name:
		return "Sea Surface"
	if "wind" in name:
		return "Wind"
	if "humidity" in name or "rh" in name:
		return "Humidity"
	return "Climate"


def _variable_label_map(variables: list[str]) -> dict[str, str]:
	label_map: dict[str, str] = {}
	for var in variables:
		base = f"{_variable_category(var)} - {var}"
		label = base
		i = 2
		while label in label_map:
			label = f"{base} ({i})"
			i += 1
		label_map[label] = var
	return label_map


def _intersect_bounds(primary_bounds: tuple[float, float], secondary_bounds: tuple[float, float]) -> tuple[float, float]:
	p_min, p_max = sorted(primary_bounds)
	s_min, s_max = sorted(secondary_bounds)
	low = max(p_min, s_min)
	high = min(p_max, s_max)
	if low >= high:
		return (p_min, p_max)
	return (low, high)


def _build_comparison_figure(
	df: pd.DataFrame,
	model_variable: str,
	obs_variable: str,
	lat: float,
	lon: float,
	show_trend: bool,
) -> go.Figure:
	fig = go.Figure()

	fig.add_trace(
		go.Scatter(
			x=df.index,
			y=df["Model"],
			name="Model",
			mode="lines+markers",
			line={"color": "#4FC3F7", "width": 2.7, "shape": "spline", "smoothing": 0.8},
			marker={"size": 5},
			hovertemplate="Time: %{x|%Y-%m-%d}<br>Model: %{y:.3f}<extra></extra>",
		)
	)

	fig.add_trace(
		go.Scatter(
			x=df.index,
			y=df["Observed"],
			name="Observed",
			mode="lines+markers",
			line={"color": "#FF8A65", "width": 2.7, "shape": "spline", "smoothing": 0.8},
			marker={"size": 5},
			hovertemplate="Time: %{x|%Y-%m-%d}<br>Observed: %{y:.3f}<extra></extra>",
		)
	)

	lower = np.minimum(df["Model"].values, df["Observed"].values)
	upper = np.maximum(df["Model"].values, df["Observed"].values)

	fig.add_trace(
		go.Scatter(
			x=df.index,
			y=lower,
			name="Difference lower",
			mode="lines",
			line={"width": 0},
			showlegend=False,
			hoverinfo="skip",
		)
	)

	fig.add_trace(
		go.Scatter(
			x=df.index,
			y=upper,
			name="Difference (Model - Observed)",
			mode="lines",
			line={"width": 0},
			fill="tonexty",
			fillcolor="rgba(255, 193, 7, 0.17)",
			customdata=df["Difference"],
			hovertemplate="Time: %{x|%Y-%m-%d}<br>Model - Observed: %{customdata:.3f}<extra></extra>",
		)
	)

	if show_trend:
		model_trend = _compute_trend(df["Model"])
		obs_trend = _compute_trend(df["Observed"])
		fig.add_trace(
			go.Scatter(
				x=df.index,
				y=model_trend,
				name="Model trend",
				mode="lines",
				line={"color": "#7FDBFF", "dash": "dash", "width": 2},
				hovertemplate="Time: %{x|%Y-%m-%d}<br>Model trend: %{y:.3f}<extra></extra>",
			)
		)
		fig.add_trace(
			go.Scatter(
				x=df.index,
				y=obs_trend,
				name="Observed trend",
				mode="lines",
				line={"color": "#FFC1A6", "dash": "dash", "width": 2},
				hovertemplate="Time: %{x|%Y-%m-%d}<br>Observed trend: %{y:.3f}<extra></extra>",
			)
		)

	trace_count = len(fig.data)

	def _visibility(show_model: bool, show_observed: bool) -> list[bool]:
		visible = [False] * trace_count
		visible[0] = show_model
		visible[1] = show_observed
		visible[2] = show_model and show_observed
		visible[3] = show_model and show_observed
		if show_trend:
			visible[4] = show_model
			visible[5] = show_observed
		return visible

	fig.update_layout(
		template="plotly_dark",
		height=660,
		hovermode="x unified",
		title={
			"text": f"Model ({model_variable}) vs Observed ({obs_variable}) @ lat {lat:.2f}, lon {lon:.2f}",
			"x": 0.01,
			"xanchor": "left",
		},
		xaxis={
			"title": "Time",
			"showgrid": True,
			"gridcolor": "rgba(170, 185, 205, 0.15)",
			"rangeslider": {"visible": True},
			"showspikes": True,
			"spikemode": "across",
			"spikecolor": "#9ec5ff",
		},
		yaxis={
			"title": model_variable if model_variable == obs_variable else "Value",
			"showgrid": True,
			"gridcolor": "rgba(170, 185, 205, 0.15)",
			"zeroline": False,
		},
		legend={
			"orientation": "h",
			"yanchor": "bottom",
			"y": 1.02,
			"xanchor": "left",
			"x": 0.01,
		},
		margin={"l": 30, "r": 20, "t": 75, "b": 20},
		updatemenus=[
			{
				"type": "buttons",
				"direction": "right",
				"x": 0.01,
				"y": 1.2,
				"showactive": True,
				"buttons": [
					{"label": "Both", "method": "update", "args": [{"visible": _visibility(True, True)}]},
					{"label": "Model", "method": "update", "args": [{"visible": _visibility(True, False)}]},
					{"label": "Observed", "method": "update", "args": [{"visible": _visibility(False, True)}]},
				],
			}
		],
	)

	return fig


def _format_value(value: float, unit: str) -> str:
	if pd.isna(value):
		return "N/A"
	return f"{value:,.3f}{unit}"


def _reference_year_difference(df: pd.DataFrame, year: int) -> float:
	values = df.loc[df.index.year == year, "Difference"]
	if values.empty:
		return float("nan")
	return float(values.mean())


def _render_comparison_dashboard(primary_ds: xr.Dataset) -> None:
	model_ds, model_label = _load_optional_dataset("Model dataset", primary_ds, "model_upload")
	obs_ds, obs_label = _load_optional_dataset("Observed dataset", primary_ds, "obs_upload")

	model_vars = variables_with_time_dim(model_ds)
	obs_vars = variables_with_time_dim(obs_ds)
	if not model_vars or not obs_vars:
		st.error("Model and observed datasets must both include at least one time-based variable.")
		st.stop()

	model_label_map = _variable_label_map(model_vars)
	obs_label_map = _variable_label_map(obs_vars)
	model_labels = list(model_label_map)
	obs_labels = list(obs_label_map)
	preferred = "tas_global_avg_ann"
	if preferred in model_vars:
		default_model_var = preferred
	else:
		default_model_var = next((v for v in model_vars if "tas" in v.lower() or "temp" in v.lower()), model_vars[0])
	if default_model_var in obs_vars:
		default_obs_var = default_model_var
	elif preferred in obs_vars:
		default_obs_var = preferred
	else:
		default_obs_var = next((v for v in obs_vars if "tas" in v.lower() or "temp" in v.lower()), obs_vars[0])

	default_model_label = next(label for label, var in model_label_map.items() if var == default_model_var)
	default_obs_label = next(label for label, var in obs_label_map.items() if var == default_obs_var)

	selected_model_label = st.sidebar.selectbox(
		"Model variable",
		model_labels,
		index=model_labels.index(default_model_label),
		key="model_variable_select",
	)
	selected_obs_label = st.sidebar.selectbox(
		"Observed variable",
		obs_labels,
		index=obs_labels.index(default_obs_label),
		key="obs_variable_select",
	)
	model_variable = model_label_map[selected_model_label]
	obs_variable = obs_label_map[selected_obs_label]

	lat_bounds = _intersect_bounds(dataset_lat_bounds(model_ds), dataset_lat_bounds(obs_ds))
	lon_bounds = _intersect_bounds(dataset_lon_bounds(model_ds), dataset_lon_bounds(obs_ds))
	lat = st.sidebar.slider(
		"Latitude",
		float(lat_bounds[0]),
		float(lat_bounds[1]),
		value=float((lat_bounds[0] + lat_bounds[1]) / 2),
	)
	lon = st.sidebar.slider(
		"Longitude",
		float(lon_bounds[0]),
		float(lon_bounds[1]),
		value=float((lon_bounds[0] + lon_bounds[1]) / 2),
	)
	show_trend = st.sidebar.toggle("Show trend lines", value=True)

	with st.spinner("Building advanced comparison dashboard..."):
		model_series = _extract_series(model_ds, model_variable, lat, lon)
		obs_series = _extract_series(obs_ds, obs_variable, lat, lon)
		df = _align_series(model_series, obs_series)

	years = sorted(df.index.year.unique().tolist())
	reference_year = st.sidebar.select_slider("Reference year", options=years, value=years[-1])

	fig = _build_comparison_figure(df, model_variable, obs_variable, lat, lon, show_trend)
	st.plotly_chart(
		fig,
		use_container_width=True,
		config={
			"displaylogo": False,
			"scrollZoom": True,
			"modeBarButtonsToRemove": ["lasso2d", "select2d"],
		},
	)
	st.caption(
		f"Model source: {model_label} ({model_variable}) | Observed source: {obs_label} ({obs_variable})"
	)

	model_unit_raw = str(model_ds[model_variable].attrs.get("units", "")).strip()
	obs_unit_raw = str(obs_ds[obs_variable].attrs.get("units", "")).strip()
	unit_raw = model_unit_raw if model_unit_raw == obs_unit_raw else model_unit_raw
	unit = f" {unit_raw}" if unit_raw else ""

	model_mean = float(df["Model"].mean())
	observed_mean = float(df["Observed"].mean())
	diff_mean = float(df["Difference"].mean())
	diff_min = float(df["Difference"].min())
	diff_max = float(df["Difference"].max())
	diff_ref = _reference_year_difference(df, reference_year)

	prev_year = next((year for year in reversed(years) if year < reference_year), None)
	prev_diff = _reference_year_difference(df, prev_year) if prev_year is not None else float("nan")
	delta_ref = diff_ref - prev_diff if pd.notna(diff_ref) and pd.notna(prev_diff) else None

	stats_row_1 = st.columns(4)
	stats_row_1[0].metric("Model mean", _format_value(model_mean, unit))
	stats_row_1[1].metric("Observed mean", _format_value(observed_mean, unit))
	stats_row_1[2].metric("Difference mean", _format_value(diff_mean, unit))
	stats_row_1[3].metric(
		f"Difference @ {reference_year}",
		_format_value(diff_ref, unit),
		delta=None if delta_ref is None else _format_value(delta_ref, unit),
	)

	stats_row_2 = st.columns(3)
	stats_row_2[0].metric("Difference min", _format_value(diff_min, unit))
	stats_row_2[1].metric("Difference max", _format_value(diff_max, unit))
	stats_row_2[2].metric("Total samples", f"{len(df):,}")

	with st.expander("Difference values preview"):
		preview = df[["Difference"]].copy()
		preview.index.name = "Time"
		st.dataframe(preview.tail(25), use_container_width=True)


def _build_time_series_figure(
	series: pd.Series,
	variable: str,
	lat: float,
	lon: float,
	show_trend: bool,
) -> go.Figure:
	fig = go.Figure()

	fig.add_trace(
		go.Scatter(
			x=series.index,
			y=series.values,
			name=variable,
			mode="lines+markers",
			line={"color": "#4FC3F7", "width": 2.9, "shape": "spline", "smoothing": 0.8},
			marker={"size": 5},
			hovertemplate="Time: %{x|%Y-%m-%d}<br>Value: %{y:.3f}<extra></extra>",
		)
	)

	if show_trend:
		trend = _compute_trend(series)
		fig.add_trace(
			go.Scatter(
				x=series.index,
				y=trend.values,
				name="Trend",
				mode="lines",
				line={"color": "#FFC1A6", "dash": "dash", "width": 2.2},
				hovertemplate="Time: %{x|%Y-%m-%d}<br>Trend: %{y:.3f}<extra></extra>",
			)
		)

	trace_count = len(fig.data)

	def _visibility(show_series: bool, show_trend_trace: bool) -> list[bool]:
		visible = [False] * trace_count
		visible[0] = show_series
		if show_trend and trace_count > 1:
			visible[1] = show_trend_trace
		return visible

	buttons = [{"label": "Series", "method": "update", "args": [{"visible": _visibility(True, False)}]}]
	if show_trend:
		buttons = [
			{"label": "Both", "method": "update", "args": [{"visible": _visibility(True, True)}]},
			{"label": "Series", "method": "update", "args": [{"visible": _visibility(True, False)}]},
			{"label": "Trend", "method": "update", "args": [{"visible": _visibility(False, True)}]},
		]

	fig.update_layout(
		template="plotly_dark",
		height=660,
		hovermode="x unified",
		title={
			"text": f"{variable} trend @ lat {lat:.2f}, lon {lon:.2f}",
			"x": 0.01,
			"xanchor": "left",
		},
		xaxis={
			"title": "Time",
			"showgrid": True,
			"gridcolor": "rgba(170, 185, 205, 0.15)",
			"rangeslider": {"visible": True},
			"showspikes": True,
			"spikemode": "across",
			"spikecolor": "#9ec5ff",
		},
		yaxis={
			"title": variable,
			"showgrid": True,
			"gridcolor": "rgba(170, 185, 205, 0.15)",
			"zeroline": False,
		},
		legend={
			"orientation": "h",
			"yanchor": "bottom",
			"y": 1.02,
			"xanchor": "left",
			"x": 0.01,
		},
		margin={"l": 30, "r": 20, "t": 75, "b": 20},
		updatemenus=[
			{
				"type": "buttons",
				"direction": "right",
				"x": 0.01,
				"y": 1.2,
				"showactive": True,
				"buttons": buttons,
			}
		],
	)

	return fig


def _render_time_series_dashboard(ds: xr.Dataset) -> None:
	variables = variables_with_time_dim(ds)
	if not variables:
		st.error("No variables with a time dimension are available for this analysis.")
		return

	label_map = _variable_label_map(variables)
	labels = list(label_map)
	preferred = "tas_global_avg_ann"
	if preferred in variables:
		default_var = preferred
	else:
		default_var = next((v for v in variables if "tas" in v.lower() or "temp" in v.lower()), variables[0])
	default_label = next(label for label, var in label_map.items() if var == default_var)

	selected_label = st.sidebar.selectbox("Variable", labels, index=labels.index(default_label), key="trend_variable")
	variable = label_map[selected_label]

	lat_min, lat_max = dataset_lat_bounds(ds)
	lon_min, lon_max = dataset_lon_bounds(ds)
	lat = st.sidebar.slider("Latitude", float(lat_min), float(lat_max), value=float((lat_min + lat_max) / 2), key="trend_lat")
	lon = st.sidebar.slider("Longitude", float(lon_min), float(lon_max), value=float((lon_min + lon_max) / 2), key="trend_lon")

	data_var = ds[variable]
	candidates = time_coord_candidates(ds, data=data_var)
	if not candidates:
		st.error("Selected variable has no time-like dimension. Choose another variable or add a time axis.")
		return
	time_coord = st.sidebar.selectbox("Time coordinate", candidates, index=0, key="trend_time_coord")
	show_trend = st.sidebar.toggle("Show trend line", value=True, key="trend_show_line")

	with st.spinner("Building advanced time series dashboard..."):
		series = _extract_series(ds, variable, lat, lon, time_coord=time_coord)

	years = sorted(series.index.year.unique().tolist())
	reference_year = st.sidebar.select_slider("Reference year", options=years, value=years[-1], key="trend_reference_year")

	fig = _build_time_series_figure(series, variable, lat, lon, show_trend)
	st.plotly_chart(
		fig,
		use_container_width=True,
		config={
			"displaylogo": False,
			"scrollZoom": True,
			"modeBarButtonsToRemove": ["lasso2d", "select2d"],
		},
	)

	unit_raw = str(ds[variable].attrs.get("units", "")).strip()
	unit = f" {unit_raw}" if unit_raw else ""
	current_values = series.loc[series.index.year == reference_year]
	current_mean = float(current_values.mean()) if not current_values.empty else float("nan")
	prev_year = next((year for year in reversed(years) if year < reference_year), None)
	prev_values = series.loc[series.index.year == prev_year] if prev_year is not None else pd.Series(dtype="float64")
	prev_mean = float(prev_values.mean()) if not prev_values.empty else float("nan")
	delta = current_mean - prev_mean if pd.notna(current_mean) and pd.notna(prev_mean) else None

	stats_row_1 = st.columns(4)
	stats_row_1[0].metric("Mean", _format_value(float(series.mean()), unit))
	stats_row_1[1].metric("Min", _format_value(float(series.min()), unit))
	stats_row_1[2].metric("Max", _format_value(float(series.max()), unit))
	stats_row_1[3].metric(
		f"Value @ {reference_year}",
		_format_value(current_mean, unit),
		delta=None if delta is None else _format_value(delta, unit),
	)

	stats_row_2 = st.columns(2)
	stats_row_2[0].metric("Total samples", f"{len(series):,}")
	stats_row_2[1].metric("Std dev", _format_value(float(series.std()), unit))

	with st.expander("Time series values preview"):
		preview = pd.DataFrame({"Value": series})
		preview.index.name = "Time"
		st.dataframe(preview.tail(25), use_container_width=True)


def main() -> None:
	st.set_page_config(page_title="PyClimaExplorer", layout="wide")
	_inject_custom_css()

	st.title("PyClimaExplorer - Climate Analytics Dashboard")
	st.caption("Interactive NetCDF analytics with model/observation comparison, trends, and hotspots.")

	st.sidebar.header("Controls")
	dataset_choice = st.sidebar.radio("Primary dataset", ["Sample dataset", "Upload dataset"], index=0)
	uploaded_main = st.sidebar.file_uploader("Upload NetCDF or .tar", type=["nc", "tar", "tgz", "gz"])

	try:
		ds_label, ds = _prepare_dataset(dataset_choice, uploaded_main)
	except Exception as exc:
		st.error(f"Failed to load dataset: {exc}")
		st.stop()

	analysis_mode = st.sidebar.selectbox(
		"Analysis mode",
		[
			"Model vs Observation Comparison",
			"Global Climate Map",
			"Time Series Trend",
			"Climate Hotspots",
		],
	)

	_render_dataset_summary(ds_label, ds)

	if analysis_mode == "Model vs Observation Comparison":
		try:
			_render_comparison_dashboard(ds)
		except Exception as exc:
			st.error(f"Unable to build comparison dashboard: {exc}")
		return

	if analysis_mode == "Global Climate Map":
		variables = variables_with_lat_lon(ds)
		if not variables:
			st.error("No variables with lat/lon dimensions are available for mapping.")
			return
		variable = st.sidebar.selectbox("Variable", variables)
		min_year, max_year = dataset_year_bounds(ds)
		if min_year == max_year:
			year = min_year
			st.sidebar.info(f"Single year detected: {year}")
		else:
			year_default = min(max_year, max(min_year, 2000))
			year = st.sidebar.slider("Year", min_year, max_year, value=year_default)
		try:
			fig = show_global_map(ds, variable, year)
			st.plotly_chart(fig, use_container_width=True)
		except Exception as exc:
			st.error(f"Unable to render map: {exc}")
		return

	if analysis_mode == "Time Series Trend":
		try:
			_render_time_series_dashboard(ds)
		except Exception as exc:
			st.error(f"Unable to build time series dashboard: {exc}")
		return

	if analysis_mode == "Climate Hotspots":
		variables = variables_with_lat_lon(ds)
		if not variables:
			st.error("No variables with lat/lon dimensions are available for hotspots.")
			return
		variable = st.sidebar.selectbox("Variable", variables)
		min_year, max_year = dataset_year_bounds(ds)
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
			fig, table_df = show_hotspots(ds, variable, baseline_years, recent_years)
			st.plotly_chart(fig, use_container_width=True)
			if table_df is not None:
				st.subheader("Top warming regions")
				st.dataframe(table_df, use_container_width=True)
		except Exception as exc:
			st.error(f"Unable to detect hotspots: {exc}")
		return

	available = dataset_variables(ds)
	if available:
		st.info("No matching analysis mode selected. Choose a mode from the sidebar.")


if __name__ == "__main__":
	main()
