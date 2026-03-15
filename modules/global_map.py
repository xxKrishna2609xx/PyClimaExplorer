"""Global map visualization."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr

from .data_loader import get_time_index


def _select_time_for_year(ds: xr.Dataset, year: int) -> Optional[pd.Timestamp]:
	idx = get_time_index(ds)
	if idx is None or len(idx) == 0:
		return None
	target = pd.Timestamp(year=year, month=1, day=1)
	nearest_idx = int(np.argmin(np.abs(idx - target)))
	return pd.Timestamp(idx[nearest_idx])


def show_global_map(ds: xr.Dataset, variable: str, year: int):
	"""Return a Plotly figure for the selected year's global field."""

	timestamp = _select_time_for_year(ds, year)
	data = ds[variable]
	if timestamp is not None and "time" in data.dims:
		data = data.sel(time=timestamp, method="nearest")
	if "lat" in data.dims:
		data = data.sortby("lat")
	if "lon" in data.dims:
		data = data.sortby("lon")
	data = data.squeeze()
	if "lat" in data.dims and "lon" in data.dims:
		data = data.transpose("lat", "lon")
	array = data.values
	mask = np.isfinite(array)
	array = np.where(mask, array, np.nan)
	fig = px.imshow(
		array,
		x=data["lon"].values if "lon" in data.coords else None,
		y=data["lat"].values if "lat" in data.coords else None,
		color_continuous_scale="RdBu_r",
		origin="lower",
	)
	fig.update_layout(
		title=f"{variable} spatial distribution ({year})",
		xaxis_title="Longitude",
		yaxis_title="Latitude",
		coloraxis_colorbar=dict(title=variable),
		margin=dict(l=10, r=10, t=40, b=10),
	)
	return fig
