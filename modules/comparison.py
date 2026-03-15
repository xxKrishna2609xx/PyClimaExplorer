"""Model vs observation comparison."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import xarray as xr

from .data_loader import get_time_index


def _extract_series(ds: xr.Dataset, variable: str, lat: float, lon: float) -> pd.Series:
	data = ds[variable]
	if "lat" in data.dims:
		data = data.sel(lat=lat, method="nearest")
	if "lon" in data.dims:
		data = data.sel(lon=lon, method="nearest")
	if "time" in data.dims:
		data = data.sortby("time")
	idx = get_time_index(data.to_dataset())
	if idx is None:
		raise ValueError("Time coordinate missing in dataset")
	return pd.Series(data.values, index=idx)


def show_model_comparison(model_ds: xr.Dataset, obs_ds: xr.Dataset, variable: str, lat: float, lon: float):
	"""Return Plotly line chart comparing model vs observed time series."""

	model_series = _extract_series(model_ds, variable, lat, lon)
	obs_series = _extract_series(obs_ds, variable, lat, lon)
	df = pd.DataFrame({"Model": model_series, "Observed": obs_series})
	df = df.sort_index()
	fig = px.line(df, labels={"index": "Time", "value": variable, "variable": "Source"}, title=f"{variable}: model vs observed @ lat {lat:.2f}, lon {lon:.2f}")
	fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
	return fig
