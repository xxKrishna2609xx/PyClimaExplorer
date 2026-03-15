"""Dataset loading helpers for PyClimaExplorer."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import streamlit as st
import xarray as xr


def _hash_bytes(data: bytes) -> str:
	"""Return a short hash for cache keys."""

	return hashlib.md5(data).hexdigest()  # nosec - used for caching only


def _path_cache_buster(path: Path) -> tuple[int, int]:
	"""Return mtime/size tuple for cache invalidation."""

	try:
		stat = path.stat()
	except FileNotFoundError as exc:
		raise FileNotFoundError(f"Dataset not found at {path}") from exc
	return (int(stat.st_mtime_ns), int(stat.st_size))


def _decode_cf_safe(ds: xr.Dataset) -> xr.Dataset:
	"""Decode CF metadata with cftime fallback; leave raw if decoding fails."""

	try:
		return xr.decode_cf(ds, use_cftime=True)
	except Exception:
		try:
			return xr.decode_cf(ds, decode_times=False)
		except Exception:
			return ds


@st.cache_resource(show_spinner=False)
def _open_dataset_from_path(path_str: str, cache_buster: tuple[int, int]) -> xr.Dataset:
	_ = cache_buster  # ensures cache invalidates when file changes
	try:
		ds = xr.open_dataset(path_str, decode_times=False)
	except Exception as exc:
		try:
			ds = xr.open_dataset(path_str, engine="h5netcdf", decode_times=False)
		except ImportError as imp_exc:
			raise ValueError("h5py/h5netcdf backend missing; install h5py to read this file") from imp_exc
		except Exception:
			raise exc
	return _decode_cf_safe(ds)


@st.cache_resource(show_spinner=False)
def _open_dataset_from_bytes(content: bytes, cache_key: str) -> xr.Dataset:
	file_like = io.BytesIO(content)
	try:
		ds = xr.open_dataset(file_like, decode_times=False)
	except Exception as exc:
		file_like.seek(0)
		try:
			ds = xr.open_dataset(file_like, engine="h5netcdf", decode_times=False)
		except ImportError as imp_exc:
			raise ValueError("h5py/h5netcdf backend missing; install h5py to read this file") from imp_exc
		except Exception:
			raise exc
	ds = _decode_cf_safe(ds)
	ds.attrs["_cache_key"] = cache_key
	return ds


def load_dataset(path: Path | str | bytes) -> xr.Dataset:
	"""Load a NetCDF dataset from a path or raw bytes with caching."""

	if isinstance(path, (str, Path)):
		path_obj = Path(path).expanduser().resolve()
		cache_buster = _path_cache_buster(path_obj)
		return _open_dataset_from_path(str(path_obj), cache_buster)
	if isinstance(path, bytes):
		cache_key = _hash_bytes(path)
		return _open_dataset_from_bytes(path, cache_key)
	raise TypeError("Unsupported dataset source")


def dataset_variables(ds: xr.Dataset) -> list[str]:
	"""Return numeric data variable names suitable for plotting."""

	numeric_vars: Iterable[str] = (
		name
		for name, var in ds.data_vars.items()
		if var.dtype.kind in {"i", "u", "f"} and var.ndim >= 1
	)
	return sorted(numeric_vars)


def time_coord_candidates(ds: xr.Dataset, data: xr.DataArray | None = None) -> list[str]:
	"""Return coordinate/dimension names that look like time/date, optionally filtered to a data array's dims."""

	def looks_like_time(name: str) -> bool:
		lname = name.lower()
		return lname in {"time", "date", "datetime", "timestamp"} or "time" in lname or "date" in lname

	allowed_dims = set(data.dims) if data is not None else None

	candidates: list[str] = []
	for name in ds.coords:
		if allowed_dims is not None and name not in allowed_dims:
			continue
		if looks_like_time(name):
			candidates.append(name)
	for dim in ds.dims:
		if allowed_dims is not None and dim not in allowed_dims:
			continue
		if looks_like_time(dim) and dim not in candidates:
			candidates.append(dim)
	return candidates


def variables_with_time_dim(ds: xr.Dataset) -> list[str]:
	"""Variables that include a time-like dimension."""

	result: list[str] = []
	for name, var in ds.data_vars.items():
		for dim in var.dims:
			lname = dim.lower()
			if lname in {"time", "date", "datetime", "timestamp"} or "time" in lname or "date" in lname:
				result.append(name)
				break
	return sorted(result)


def variables_with_lat_lon(ds: xr.Dataset) -> list[str]:
	"""Variables that include both lat and lon dimensions."""

	result: list[str] = []
	for name, var in ds.data_vars.items():
		if "lat" in var.dims and "lon" in var.dims:
			result.append(name)
	return sorted(result)


def get_time_index(ds: xr.Dataset, coord_name: Optional[str] = None, data: xr.DataArray | None = None) -> Optional[pd.DatetimeIndex]:
	"""Return a timezone-naive DatetimeIndex for the given or first time-like coord.

	If the coordinate is missing but the dimension exists on the provided data array,
	construct a simple RangeIndex as a last resort.
	"""

	def _as_years_index(values) -> Optional[pd.DatetimeIndex]:
		"""If values look like YYYY, return DatetimeIndex at Jan 1 of that year."""
		try:
			as_float = pd.to_numeric(values)
		except Exception:
			return None
		if not pd.api.types.is_numeric_dtype(as_float):
			return None
		as_int = as_float.astype(int)
		if (as_int >= 1000).all() and (as_int <= 3000).all():
			idx = pd.to_datetime(as_int, format="%Y", errors="coerce")
			if idx.notna().all():
				return pd.DatetimeIndex(idx).tz_localize(None)
		return None

	candidates = time_coord_candidates(ds, data=data)
	name = coord_name or (candidates[0] if candidates else None)
	if name is None:
		return None
	coord = ds[name] if name in ds else None
	if coord is None and data is not None and name in data.dims:
		return pd.RangeIndex(len(data[name])) if name in data.coords else pd.RangeIndex(data.sizes[name])
	if coord is None:
		return None
	# First try xarray's own index (handles CFTimeIndex)
	try:
		xr_index = coord.to_index()
		if hasattr(xr_index, "to_datetimeindex"):
			idx = xr_index.to_datetimeindex()
		else:
			idx = pd.DatetimeIndex(xr_index)
		# If the index is numeric years (e.g., 1950), reinterpret as calendar years
		year_idx = _as_years_index(idx.values)
		if year_idx is not None:
			return year_idx
		idx = idx.tz_localize(None)
		return idx
	except Exception:
		pass
	# Fallback: direct pandas conversion
	time_values = coord.values
	# Heuristic: if values look like plain years (e.g., 1950, 2023), convert to Jan 1 of that year
	year_idx = _as_years_index(time_values)
	if year_idx is not None:
		return year_idx
	try:
		idx = pd.to_datetime(time_values)
		idx = idx.tz_localize(None)
		return idx
	except Exception:
		return None


def dataset_year_bounds(ds: xr.Dataset) -> tuple[int, int]:
	"""Return min and max years, falling back to (2000, 2000) if missing."""

	idx = get_time_index(ds)
	if idx is None or len(idx) == 0:
		return (2000, 2000)
	years = idx.year
	return int(years.min()), int(years.max())


def dataset_lat_bounds(ds: xr.Dataset) -> tuple[float, float]:
	if "lat" not in ds:
		return (-90.0, 90.0)
	vals = ds["lat"].values
	return float(vals.min()), float(vals.max())


def dataset_lon_bounds(ds: xr.Dataset) -> tuple[float, float]:
	if "lon" not in ds:
		return (-180.0, 180.0)
	vals = ds["lon"].values
	return float(vals.min()), float(vals.max())


def summarize_dataset(ds: xr.Dataset) -> dict:
	"""Return a compact summary for UI display."""

	return {
		"dimensions": {name: int(size) for name, size in ds.sizes.items()},
		"variables": dataset_variables(ds),
		"coords": list(ds.coords),
		"attributes": {k: str(v) for k, v in ds.attrs.items()},
	}
