"""Time series visualization."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import xarray as xr

from .data_loader import get_time_index


def show_time_series(ds: xr.Dataset, variable: str, lat: float, lon: float, time_coord: str | None = None):
	"""Return a Plotly time series at the nearest lat/lon."""

	data = ds[variable]
	# Select nearest point if spatial dims exist
	if "lat" in data.dims:
		data = data.sel(lat=lat, method="nearest")
	if "lon" in data.dims:
		data = data.sel(lon=lon, method="nearest")
	# Sort by time dimension (case-insensitive)
	for dim in data.dims:
		if dim.lower() == "time":
			data = data.sortby(dim)
			break
	idx = get_time_index(data.to_dataset(), coord_name=time_coord, data=data)
	if idx is None:
		raise ValueError("Dataset is missing a valid time coordinate (looked for time/date-like coords)")
	series = pd.Series(data.values, index=idx)
	fig = px.line(series, labels={"index": "Time", "value": variable}, title=f"{variable} trend @ lat {lat:.2f}, lon {lon:.2f}")
	fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
	return fig
