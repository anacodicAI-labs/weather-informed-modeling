#!/usr/bin/env python3
"""Build capacity-weighted hourly weather from locally staged ERA5 files.

The finest ERA5 grid is downloaded once by ``stage_era5_cds.py`` or
``stage_era5_arco.py`` into ``data/era5_native_0p25deg``. This script does no
network access. At each requested
resolution it genuinely coarsens the weather grid, maps full-precision
generator locations to the nearest resulting cell, and reduces the field using
technology-specific capacity weights.

The preferred capacity layout is annual::

    data/capacity_weights_post_by_year/<year>/<code>.csv

The capacity map is changed at each January boundary inside a multi-year ERA5
run.  A legacy flat layout (``<capacity-dir>/<code>.csv``) remains supported
for reproducibility of the earlier static-capacity experiment.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from country_registry import BBOX


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
# The data root enables automatic per-country selection between the traditional
# CDS staging tree and the ARCO staging tree.  Passing either tree explicitly is
# still supported.
DEFAULT_ERA5_DIR = DATA_DIR
DEFAULT_CAPACITY_DIR = DATA_DIR / "capacity_weights_post_by_year"
DEFAULT_OUT_DIR = DATA_DIR / "country_weather_post_covid"
DEFAULT_RESOLUTION_CACHE_DIR = DATA_DIR / "era5_coarse_post_covid"
LOW_COVERAGE_POINT_THRESHOLD = 10

# The Portuguese, Spanish, and French Energy-Charts targets represent their
# continental interconnected systems. Island/overseas generators must not be
# weighted against those targets. Keep these exclusions explicit and auditable
# instead of treating them as accidental ERA5-domain misses. REN describes its
# transmission grid as covering all of mainland Portugal:
# https://www.ren.pt/en-gb/activity/electricity
SYSTEM_SCOPE_BOUNDS = {
    "pt": {
        "label": "mainland_portugal",
        "bounds": BBOX["pt"],
    },
    "es": {
        "label": "mainland_spain",
        "bounds": BBOX["es"],
    },
    "fr": {
        "label": "metropolitan_france",
        "bounds": BBOX["fr"],
    },
}

ALIASES = {
    "u100": ("u100", "100m_u_component_of_wind"),
    "v100": ("v100", "100m_v_component_of_wind"),
    "t2m": ("t2m", "2m_temperature"),
    "ssrd": ("ssrd", "surface_solar_radiation_downwards"),
}


def inclusive_end(value: str | pd.Timestamp) -> pd.Timestamp:
    """Treat a date-only end value as the end of that UTC calendar day."""
    timestamp = pd.Timestamp(value)
    if isinstance(value, str) and len(value.strip()) == 10:
        timestamp += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return timestamp


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024


def canonicalize(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for canonical, candidates in {
        "time": ("time", "valid_time"),
        "latitude": ("latitude", "lat"),
        "longitude": ("longitude", "lon"),
    }.items():
        found = next((name for name in candidates if name in ds.coords), None)
        if found is None:
            raise KeyError(f"Missing {canonical} coordinate; coordinates={list(ds.coords)}")
        if found != canonical:
            rename[found] = canonical
    ds = ds.rename(rename)

    variable_rename = {}
    for canonical, candidates in ALIASES.items():
        found = next((name for name in candidates if name in ds.data_vars), None)
        if found is None:
            raise KeyError(f"Missing {canonical}; data variables={list(ds.data_vars)}")
        if found != canonical:
            variable_rename[found] = canonical
    ds = ds.rename(variable_rename)[list(ALIASES)]

    if "expver" in ds.dims:
        # ERA5/ERA5T sometimes arrive as complementary expver slices.
        ds = ds.max("expver", skipna=True)
    # CDS can retain scalar metadata coordinates such as ``number`` and
    # ``expver`` after merging instant and accumulated streams. They are not
    # model features and should not leak into the output CSV.
    extra_coords = [
        name for name in ds.coords
        if name not in {"time", "latitude", "longitude"}
    ]
    if extra_coords:
        ds = ds.drop_vars(extra_coords)
    if np.nanmax(ds.longitude.values) > 180:
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)
        ds = ds.sortby("longitude")
    ds = ds.sortby("latitude").sortby("time")
    return ds


def region_directories(code: str, era5_dir: Path) -> list[Path]:
    """Return explicit or auto-discovered ERA5 directories for one region."""
    candidates = [era5_dir / code]
    candidates.extend(
        [
            era5_dir / "era5_native_0p25deg" / code,
            era5_dir / "era5_regional" / code,
        ]
    )
    result = []
    for candidate in candidates:
        if candidate not in result:
            result.append(candidate)
    return result


def naive_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def open_paths(paths: list[Path]) -> xr.Dataset:
    try:
        import dask  # noqa: F401
        use_dask = True
    except ImportError:
        use_dask = False

    datasets = []
    # Prefer the larger, usually full-period file when a smoke-test file and a
    # complete file overlap. Duplicate timestamps retain the first occurrence.
    paths = sorted(paths, key=lambda path: path.stat().st_size, reverse=True)
    for path in paths:
        if use_dask:
            # Preserve the file's native chunks.  Forcing 168-hour boundaries
            # through an existing chunk produced warnings and unnecessary I/O.
            raw = xr.open_dataset(path, chunks={})
        else:
            raw = xr.open_dataset(path)
        datasets.append(canonicalize(raw))
    combined = xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="outer",
    )
    _, unique_index = np.unique(combined.time.values, return_index=True)
    return combined.isel(time=np.sort(unique_index)).sortby("time")


def open_local_region(
    code: str, era5_dir: Path, start: str, end: str
) -> tuple[xr.Dataset, list[Path]]:
    requested_start = naive_timestamp(start)
    requested_end = naive_timestamp(inclusive_end(end)).floor("h")
    expected = pd.date_range(requested_start, requested_end, freq="h")
    diagnostics = []

    for directory in region_directories(code, era5_dir):
        paths = [
            Path(path)
            for path in sorted(glob.glob(str(directory / "era5_*.nc")))
        ]
        if not paths:
            diagnostics.append(f"{directory}: no NetCDF files")
            continue
        ds = open_paths(paths)
        ds = ds.sel(time=slice(requested_start, requested_end))
        _, unique_index = np.unique(ds.time.values, return_index=True)
        ds = ds.isel(time=np.sort(unique_index))
        actual = pd.DatetimeIndex(ds.time.values)
        missing = expected.difference(actual)
        if len(actual) == len(expected) and not len(missing):
            return ds, paths
        diagnostics.append(
            f"{directory}: found {len(actual)}/{len(expected)} requested hours; "
            f"missing={len(missing)}"
        )
        ds.close()

    detail = "\n  ".join(diagnostics)
    raise ValueError(
        f"No complete staged ERA5 source for {code} in {start}..{end}. Checked:\n  {detail}"
    )


def grid_step(values: np.ndarray) -> float:
    unique = np.unique(np.asarray(values, dtype=float))
    if len(unique) < 2:
        raise ValueError("Cannot determine grid spacing from fewer than two coordinates")
    return float(np.median(np.abs(np.diff(unique))))


def resolution_tag(resolution: float) -> str:
    return f"{resolution:g}".replace(".", "p")


def resolution_cache_path(
    cache_dir: Path,
    code: str,
    start: str,
    end: str,
    resolution: float,
) -> Path:
    return (
        Path(cache_dir)
        / code
        / f"era5_fields_{code}_{start}_{end}_{resolution_tag(resolution)}deg.nc"
    )


def derived_fields(ds: xr.Dataset) -> xr.Dataset:
    wind = np.hypot(ds.u100, ds.v100)
    temperature = ds.t2m
    temp_units = str(temperature.attrs.get("units", "")).lower()
    if "k" in temp_units or not temp_units:
        temperature = temperature - 273.15

    radiation = ds.ssrd
    rad_units = str(radiation.attrs.get("units", "")).lower()
    if "j" in rad_units or not rad_units:
        # CDS hourly ERA5 SSRD is energy accumulated over the preceding hour.
        radiation = radiation / 3600.0
    radiation = radiation.clip(min=0)
    return xr.Dataset(
        {
            "wind_speed_100m": wind,
            "shortwave_radiation": radiation,
            "temperature_2m": temperature,
        }
    )


def coarsen_to(ds: xr.Dataset, resolution: float) -> tuple[xr.Dataset, float, int, int]:
    lat_step = grid_step(ds.latitude.values)
    lon_step = grid_step(ds.longitude.values)
    native = max(lat_step, lon_step)
    fy = int(round(resolution / lat_step))
    fx = int(round(resolution / lon_step))
    if fy < 1 or fx < 1:
        raise ValueError(f"Requested {resolution:g}° is finer than native grid {native:g}°")
    if not np.isclose(fy * lat_step, resolution, atol=max(0.01, lat_step * 0.08)):
        raise ValueError(f"Resolution {resolution:g}° is not an integer multiple of {lat_step:g}°")
    if not np.isclose(fx * lon_step, resolution, atol=max(0.01, lon_step * 0.08)):
        raise ValueError(f"Resolution {resolution:g}° is not an integer multiple of {lon_step:g}°")
    if fy == 1 and fx == 1:
        return ds, native, fy, fx
    return (
        # Pad incomplete edge windows so coarse resolutions retain the entire
        # staged national domain.  ``trim`` silently discarded coastal cells
        # when a grid dimension was not divisible by the coarsening factor.
        ds.coarsen(latitude=fy, longitude=fx, boundary="pad").mean(skipna=True),
        native,
        fy,
        fx,
    )


def capacity_path(code: str, capacity_dir: Path, year: int) -> tuple[Path, str]:
    """Resolve an annual map, with explicit support for the old flat layout."""
    annual = capacity_dir / str(year) / f"{code}.csv"
    if annual.exists():
        return annual, "annual"
    legacy = capacity_dir / f"{code}.csv"
    if legacy.exists():
        return legacy, "static"
    raise FileNotFoundError(
        f"Missing capacity map for {code} in {year}. Tried {annual} and {legacy}"
    )


def load_capacity(code: str, capacity_dir: Path, year: int) -> tuple[pd.DataFrame, Path, str]:
    path, layout = capacity_path(code, capacity_dir, year)
    cap = pd.read_csv(path)
    required = {"lat", "lon", "wind_mw", "solar_mw"}
    if not required.issubset(cap.columns):
        raise KeyError(f"{path} must contain {sorted(required)}")
    for column in required:
        cap[column] = pd.to_numeric(cap[column], errors="coerce")
    cap = cap.dropna(subset=list(required)).copy()
    if not cap["lat"].between(-90, 90).all() or not cap["lon"].between(-180, 180).all():
        raise ValueError(f"{path} contains invalid latitude or longitude values")
    if (cap[["wind_mw", "solar_mw"]] < 0).any().any():
        raise ValueError(f"{path} contains negative capacity values")
    return cap, path, layout


def apply_system_scope(code: str, cap: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Restrict capacity to the electrical system represented by the target.

    Most country series and capacity maps use the same national footprint.
    Portugal, Spain, and France use continental-system targets while GEM may
    also contain island/overseas plants. Excluded capacity is reported
    separately from genuine ERA5-domain misses.
    """
    scope = SYSTEM_SCOPE_BOUNDS.get(code)
    if scope is None:
        return cap, {
            "system_scope": "country",
            "wind_out_of_system_scope_mw": 0.0,
            "solar_out_of_system_scope_mw": 0.0,
            "wind_out_of_system_scope_points": 0,
            "solar_out_of_system_scope_points": 0,
        }

    south, north, west, east = scope["bounds"]
    inside = cap["lat"].between(south, north) & cap["lon"].between(west, east)
    excluded = cap.loc[~inside]
    return cap.loc[inside].copy(), {
        "system_scope": scope["label"],
        "wind_out_of_system_scope_mw": float(excluded["wind_mw"].sum()),
        "solar_out_of_system_scope_mw": float(excluded["solar_mw"].sum()),
        "wind_out_of_system_scope_points": int((excluded["wind_mw"] > 0).sum()),
        "solar_out_of_system_scope_points": int((excluded["solar_mw"] > 0).sum()),
    }


def map_capacity_to_grid(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    cap: pd.DataFrame,
    column: str,
    domain_bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, float, float, int]:
    """Map in-domain capacity to its nearest grid cell.

    Capacity outside the staged ERA5 domain is excluded and reported instead
    of being silently snapped to a boundary cell.
    """
    weights = np.zeros((len(latitudes), len(longitudes)), dtype=np.float64)
    positive = cap[cap[column] > 0]
    if positive.empty:
        return weights, 0.0, 0.0, 0
    south, north, west, east = domain_bounds
    in_domain = positive["lat"].between(south, north) & positive["lon"].between(west, east)
    outside = positive.loc[~in_domain]
    positive = positive.loc[in_domain]
    if not positive.empty:
        lat_index = np.abs(latitudes[:, None] - positive.lat.to_numpy()[None, :]).argmin(axis=0)
        lon_index = np.abs(longitudes[:, None] - positive.lon.to_numpy()[None, :]).argmin(axis=0)
        np.add.at(weights, (lat_index, lon_index), positive[column].to_numpy())
    return (
        weights,
        float(positive[column].sum()),
        float(outside[column].sum()),
        int(len(outside)),
    )


def weighted_spatial_mean(field: xr.DataArray, weights: np.ndarray) -> xr.DataArray:
    total = float(weights.sum())
    if total <= 0:
        return field.mean(("latitude", "longitude"))
    w = xr.DataArray(
        weights,
        dims=("latitude", "longitude"),
        coords={"latitude": field.latitude, "longitude": field.longitude},
    )
    return (field * w).sum(("latitude", "longitude")) / total


def effective_cell_count(weights: np.ndarray) -> float:
    """Kish effective count of nonuniformly weighted grid cells."""
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    squared = float(np.square(weights).sum())
    return total * total / squared if total > 0 and squared > 0 else 0.0


def native_domain_bounds(ds: xr.Dataset) -> tuple[float, float, float, float]:
    """Return native-grid cell-edge bounds used to audit capacity coverage."""
    lat = ds.latitude.values.astype(float)
    lon = ds.longitude.values.astype(float)
    lat_half = grid_step(lat) / 2
    lon_half = grid_step(lon) / 2
    return (
        float(lat.min() - lat_half),
        float(lat.max() + lat_half),
        float(lon.min() - lon_half),
        float(lon.max() + lon_half),
    )


def annual_aggregation(
    fields: xr.Dataset,
    code: str,
    scheme: str,
    capacity_dir: Path,
    domain_bounds: tuple[float, float, float, float],
) -> tuple[xr.Dataset, list[dict]]:
    """Aggregate each calendar year with the matching capacity snapshot."""
    timestamps = pd.DatetimeIndex(fields.time.values)
    years = sorted(set(timestamps.year))
    pieces: list[xr.Dataset] = []
    audit_rows: list[dict] = []
    latitudes = fields.latitude.values.astype(float)
    longitudes = fields.longitude.values.astype(float)

    for year in years:
        selector = timestamps.year == year
        yearly_fields = fields.isel(time=np.flatnonzero(selector))
        cap, path, layout = load_capacity(code, capacity_dir, year)
        cap, scope_audit = apply_system_scope(code, cap)
        wind_w, wind_inside, wind_outside, wind_outside_points = map_capacity_to_grid(
            latitudes, longitudes, cap, "wind_mw", domain_bounds
        )
        solar_w, solar_inside, solar_outside, solar_outside_points = map_capacity_to_grid(
            latitudes, longitudes, cap, "solar_mw", domain_bounds
        )
        wind_fallback = wind_w.sum() <= 0
        solar_fallback = solar_w.sum() <= 0
        wind_points = int((cap["wind_mw"] > 0).sum())
        solar_points = int((cap["solar_mw"] > 0).sum())
        wind_mapped_points = wind_points - wind_outside_points
        solar_mapped_points = solar_points - solar_outside_points

        if scheme == "uniform":
            wind_used = np.ones_like(wind_w)
            solar_used = np.ones_like(solar_w)
            temperature_used = np.ones_like(wind_w)
        elif scheme == "capacity":
            # An absent technology has no defensible capacity centroid.  Use a
            # documented national-box mean for that feature rather than failing
            # or producing NaNs (relevant to wind in Slovenia and Slovakia).
            wind_used = np.ones_like(wind_w) if wind_fallback else wind_w
            solar_used = np.ones_like(solar_w) if solar_fallback else solar_w
            combined = wind_w + solar_w
            temperature_used = np.ones_like(combined) if combined.sum() <= 0 else combined
        else:
            raise ValueError("scheme must be 'capacity' or 'uniform'")

        pieces.append(
            xr.Dataset(
                {
                    "wind_speed_100m": weighted_spatial_mean(
                        yearly_fields.wind_speed_100m, wind_used
                    ),
                    "shortwave_radiation": weighted_spatial_mean(
                        yearly_fields.shortwave_radiation, solar_used
                    ),
                    "temperature_2m": weighted_spatial_mean(
                        yearly_fields.temperature_2m, temperature_used
                    ),
                }
            )
        )
        audit_rows.append(
            {
                "year": year,
                "hours": int(selector.sum()),
                "layout": layout,
                "capacity_path": str(path),
                "wind_capacity_mw": wind_inside + wind_outside,
                "solar_capacity_mw": solar_inside + solar_outside,
                "wind_mapped_mw": wind_inside,
                "solar_mapped_mw": solar_inside,
                "wind_outside_grid_mw": wind_outside,
                "solar_outside_grid_mw": solar_outside,
                "wind_outside_grid_points": wind_outside_points,
                "solar_outside_grid_points": solar_outside_points,
                "wind_mapped_points": wind_mapped_points,
                "solar_mapped_points": solar_mapped_points,
                "wind_occupied_cells": int(np.count_nonzero(wind_w)),
                "solar_occupied_cells": int(np.count_nonzero(solar_w)),
                "combined_occupied_cells": int(np.count_nonzero(wind_w + solar_w)),
                "wind_effective_cells": effective_cell_count(wind_w),
                "solar_effective_cells": effective_cell_count(solar_w),
                "combined_effective_cells": effective_cell_count(wind_w + solar_w),
                "wind_low_coverage": bool(
                    0 < wind_mapped_points <= LOW_COVERAGE_POINT_THRESHOLD
                ),
                "solar_low_coverage": bool(
                    0 < solar_mapped_points <= LOW_COVERAGE_POINT_THRESHOLD
                ),
                "wind_capacity_absent": bool(wind_fallback),
                "solar_capacity_absent": bool(solar_fallback),
                "wind_weight_fallback": bool(wind_fallback and scheme == "capacity"),
                "solar_weight_fallback": bool(solar_fallback and scheme == "capacity"),
                **scope_audit,
            }
        )

    return xr.concat(pieces, dim="time").sortby("time"), audit_rows


def time_weighted_metric(audit_rows: list[dict], column: str) -> float:
    hours = np.asarray([row["hours"] for row in audit_rows], dtype=float)
    values = np.asarray([row[column] for row in audit_rows], dtype=float)
    return float(np.average(values, weights=hours))


def run(
    code: str,
    resolution: float,
    scheme: str = "capacity",
    start: str = "2022-01-01",
    end: str = "2026-04-30",
    era5_dir: Path = DEFAULT_ERA5_DIR,
    capacity_dir: Path = DEFAULT_CAPACITY_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    output: Path | None = None,
    resolution_cache_dir: Path = DEFAULT_RESOLUTION_CACHE_DIR,
    use_resolution_cache: bool = True,
) -> tuple[Path, dict]:
    started = time.perf_counter()
    cache_path = resolution_cache_path(
        Path(resolution_cache_dir), code, start, end, resolution
    )
    source_open_started = time.perf_counter()
    if use_resolution_cache and cache_path.exists():
        fields = xr.open_dataset(cache_path, chunks={})
        required_fields = {"wind_speed_100m", "shortwave_radiation", "temperature_2m"}
        missing_fields = required_fields.difference(fields.data_vars)
        if missing_fields:
            fields.close()
            raise KeyError(f"{cache_path} is missing cached fields {sorted(missing_fields)}")
        fields = fields[list(sorted(required_fields))].sortby("latitude").sortby("time")
        fields = fields.sel(
            time=slice(naive_timestamp(start), naive_timestamp(inclusive_end(end)).floor("h"))
        )
        expected_hours = len(pd.date_range(
            naive_timestamp(start), naive_timestamp(inclusive_end(end)).floor("h"), freq="h"
        ))
        if fields.sizes.get("time", 0) != expected_hours:
            fields.close()
            raise ValueError(
                f"Incomplete resolution cache {cache_path}: "
                f"{fields.sizes.get('time', 0)}/{expected_hours} hours"
            )
        source_paths = [cache_path]
        source_kind = "resolution_cache"
        native_resolution = float(fields.attrs.get("native_resolution_deg", 0.25))
        fy = int(fields.attrs.get("coarsen_lat_factor", round(resolution / native_resolution)))
        fx = int(fields.attrs.get("coarsen_lon_factor", round(resolution / native_resolution)))
        native_points = int(fields.attrs.get(
            "native_points", fields.sizes["latitude"] * fields.sizes["longitude"]
        ))
        domain_bounds = tuple(
            float(fields.attrs[name])
            for name in ("domain_south", "domain_north", "domain_west", "domain_east")
        )
    else:
        ds, source_paths = open_local_region(code, Path(era5_dir), start, end)
        fields = derived_fields(ds)
        native_points = fields.sizes["latitude"] * fields.sizes["longitude"]
        domain_bounds = native_domain_bounds(fields)
        fields, native_resolution, fy, fx = coarsen_to(fields, resolution)
        source_kind = "native_era5"
    source_open_s = time.perf_counter() - source_open_started
    source_bytes = sum(path.stat().st_size for path in source_paths)
    coarse_points = fields.sizes["latitude"] * fields.sizes["longitude"]

    graph_started = time.perf_counter()
    aggregated, capacity_audit = annual_aggregation(
        fields=fields,
        code=code,
        scheme=scheme,
        capacity_dir=Path(capacity_dir),
        domain_bounds=domain_bounds,
    )
    graph_build_s = time.perf_counter() - graph_started
    compute_started = time.perf_counter()
    aggregated = aggregated.compute()
    compute_s = time.perf_counter() - compute_started

    frame = aggregated.to_dataframe().reset_index()
    frame["timestamp"] = pd.to_datetime(frame.pop("time"), utc=True)
    frame = frame.set_index("timestamp").sort_index()
    frame = frame.loc[
        pd.Timestamp(start, tz="UTC") : inclusive_end(end).tz_localize("UTC")
    ]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if output is None:
        tag = str(resolution).replace(".", "p")
        output = out_dir / f"weather_era5_{code}_{scheme}_{tag}deg.csv"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_started = time.perf_counter()
    frame.to_csv(output)
    write_s = time.perf_counter() - write_started
    total_s = time.perf_counter() - started

    metrics = {
        "code": code,
        "scheme": scheme,
        "resolution_deg": resolution,
        "native_resolution_deg": native_resolution,
        "coarsen_lat_factor": fy,
        "coarsen_lon_factor": fx,
        "hours": len(frame),
        "native_points": native_points,
        "processed_points": coarse_points,
        "point_hours": int(coarse_points * len(frame)),
        "source_bytes": source_bytes,
        "source_kind": source_kind,
        "source_open_s": source_open_s,
        "graph_build_s": graph_build_s,
        "capacity_dir": str(Path(capacity_dir).resolve()),
        "capacity_layout": ",".join(sorted({row["layout"] for row in capacity_audit})),
        "capacity_years": ",".join(str(row["year"]) for row in capacity_audit),
        "capacity_files": ";".join(row["capacity_path"] for row in capacity_audit),
        "wind_capacity_mw": time_weighted_metric(capacity_audit, "wind_capacity_mw"),
        "solar_capacity_mw": time_weighted_metric(capacity_audit, "solar_capacity_mw"),
        "wind_capacity_mw_start": capacity_audit[0]["wind_capacity_mw"],
        "wind_capacity_mw_end": capacity_audit[-1]["wind_capacity_mw"],
        "solar_capacity_mw_start": capacity_audit[0]["solar_capacity_mw"],
        "solar_capacity_mw_end": capacity_audit[-1]["solar_capacity_mw"],
        "wind_occupied_cells": time_weighted_metric(capacity_audit, "wind_occupied_cells"),
        "solar_occupied_cells": time_weighted_metric(capacity_audit, "solar_occupied_cells"),
        "combined_occupied_cells": time_weighted_metric(capacity_audit, "combined_occupied_cells"),
        "wind_effective_cells": time_weighted_metric(capacity_audit, "wind_effective_cells"),
        "solar_effective_cells": time_weighted_metric(capacity_audit, "solar_effective_cells"),
        "combined_effective_cells": time_weighted_metric(capacity_audit, "combined_effective_cells"),
        "wind_outside_grid_mw_max": max(row["wind_outside_grid_mw"] for row in capacity_audit),
        "solar_outside_grid_mw_max": max(row["solar_outside_grid_mw"] for row in capacity_audit),
        "wind_outside_grid_points_max": max(row["wind_outside_grid_points"] for row in capacity_audit),
        "solar_outside_grid_points_max": max(row["solar_outside_grid_points"] for row in capacity_audit),
        "system_scope": ",".join(sorted({row["system_scope"] for row in capacity_audit})),
        "wind_out_of_system_scope_mw_max": max(
            row["wind_out_of_system_scope_mw"] for row in capacity_audit
        ),
        "solar_out_of_system_scope_mw_max": max(
            row["solar_out_of_system_scope_mw"] for row in capacity_audit
        ),
        "wind_out_of_system_scope_points_max": max(
            row["wind_out_of_system_scope_points"] for row in capacity_audit
        ),
        "solar_out_of_system_scope_points_max": max(
            row["solar_out_of_system_scope_points"] for row in capacity_audit
        ),
        "wind_weight_fallback": any(row["wind_weight_fallback"] for row in capacity_audit),
        "solar_weight_fallback": any(row["solar_weight_fallback"] for row in capacity_audit),
        "wind_capacity_absent": any(row["wind_capacity_absent"] for row in capacity_audit),
        "solar_capacity_absent": any(row["solar_capacity_absent"] for row in capacity_audit),
        "wind_low_coverage": any(row["wind_low_coverage"] for row in capacity_audit),
        "solar_low_coverage": any(row["solar_low_coverage"] for row in capacity_audit),
        "wind_fallback_years": ",".join(
            str(row["year"]) for row in capacity_audit if row["wind_weight_fallback"]
        ),
        "solar_fallback_years": ",".join(
            str(row["year"]) for row in capacity_audit if row["solar_weight_fallback"]
        ),
        "capacity_audit": capacity_audit,
        "compute_s": compute_s,
        "write_s": write_s,
        "total_s": total_s,
        "point_hours_per_s": coarse_points * len(frame) / total_s if total_s else np.nan,
        "aggregation_point_hours_per_s": (
            coarse_points * len(frame) / compute_s if compute_s else np.nan
        ),
        "peak_rss_mb": peak_rss_mb(),
        "output": str(output),
    }
    return output, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--scheme", choices=("capacity", "uniform"), default="capacity")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--era5-dir", default=str(DEFAULT_ERA5_DIR))
    parser.add_argument("--capacity-dir", default=str(DEFAULT_CAPACITY_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--resolution-cache-dir", default=str(DEFAULT_RESOLUTION_CACHE_DIR)
    )
    parser.add_argument("--no-resolution-cache", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--metrics-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, metrics = run(
        code=args.code,
        resolution=args.resolution,
        scheme=args.scheme,
        start=args.start,
        end=args.end,
        era5_dir=Path(args.era5_dir).expanduser().resolve(),
        capacity_dir=Path(args.capacity_dir).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve() if args.output else None,
        resolution_cache_dir=Path(args.resolution_cache_dir).expanduser().resolve(),
        use_resolution_cache=not args.no_resolution_cache,
    )
    if args.metrics_json:
        path = Path(args.metrics_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
