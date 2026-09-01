#!/usr/bin/env python3
"""Materialize resolution-native ERA5 feature files for fair scaling tests.

The science sweep can always fall back to the native 0.25-degree ERA5 files.
For performance experiments, however, making every task reread the native grid
creates an I/O floor that hides the work reduction at coarser resolutions.
This one-time preprocessing step writes the three derived weather fields at
each requested resolution.  ``run_spatial_resolution_ladder.py`` discovers
these files automatically and records ``source_kind=resolution_cache``.

Preprocessing time is written to its own manifest and is therefore never
silently omitted from the computational accounting.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import xarray as xr
import dask

import build_weighted_weather_local as builder
from country_registry import EUROPE_CODES


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results" / "post_covid_spatial_resolution"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", nargs="+", choices=EUROPE_CODES + ("tx",), required=True)
    parser.add_argument("--resolutions", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--era5-dir", default=str(builder.DEFAULT_ERA5_DIR))
    parser.add_argument("--out-dir", default=str(builder.DEFAULT_RESOLUTION_CACHE_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate(path: Path, expected_hours: int) -> None:
    with xr.open_dataset(path) as ds:
        required = {"wind_speed_100m", "shortwave_radiation", "temperature_2m"}
        missing = required.difference(ds.data_vars)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        if ds.sizes.get("time", 0) != expected_hours:
            raise ValueError(
                f"{path} has {ds.sizes.get('time', 0)}/{expected_hours} hours"
            )


def main() -> None:
    args = parse_args()
    era5_dir = Path(args.era5_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    expected_hours = len(pd.date_range(
        builder.naive_timestamp(args.start),
        builder.naive_timestamp(builder.inclusive_end(args.end)).floor("h"),
        freq="h",
    ))
    rows: list[dict] = []

    for code in args.codes:
        opened = time.perf_counter()
        ds, source_paths = builder.open_local_region(code, era5_dir, args.start, args.end)
        source_open_s = time.perf_counter() - opened
        fields = builder.derived_fields(ds)
        native_points = fields.sizes["latitude"] * fields.sizes["longitude"]
        domain_bounds = builder.native_domain_bounds(fields)
        source_bytes = sum(path.stat().st_size for path in source_paths)

        for resolution in args.resolutions:
            target = builder.resolution_cache_path(
                out_dir, code, args.start, args.end, resolution
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "code": code,
                "resolution_deg": resolution,
                "path": str(target),
                "source_paths": ";".join(map(str, source_paths)),
                "source_bytes": source_bytes,
                "source_open_s": source_open_s,
            }
            if target.exists() and not args.force:
                validate(target, expected_hours)
                row.update(
                    status="skipped_existing",
                    output_bytes=target.stat().st_size,
                    build_wall_s=0.0,
                )
                rows.append(row)
                print(f"[skip] {code} {resolution:g}°: {target}", flush=True)
                continue

            started = time.perf_counter()
            coarse, native_resolution, fy, fx = builder.coarsen_to(fields, resolution)
            coarse = coarse.transpose("time", "latitude", "longitude").assign_attrs(
                native_resolution_deg=native_resolution,
                coarsen_lat_factor=fy,
                coarsen_lon_factor=fx,
                native_points=native_points,
                domain_south=domain_bounds[0],
                domain_north=domain_bounds[1],
                domain_west=domain_bounds[2],
                domain_east=domain_bounds[3],
                preprocessing="derived from native ERA5; padded spatial coarsening",
            )
            coarse = coarse.chunk({"time": 168, "latitude": -1, "longitude": -1})
            encoding = {
                name: {
                    "chunksizes": (
                        min(168, coarse.sizes["time"]),
                        coarse.sizes["latitude"],
                        coarse.sizes["longitude"],
                    ),
                    "zlib": True,
                    "complevel": 1,
                    "shuffle": True,
                }
                for name in coarse.data_vars
            }
            tmp = target.with_suffix(target.suffix + ".part")
            if tmp.exists():
                tmp.unlink()
            print(
                f"[build] {code} {resolution:g}°: "
                f"{coarse.sizes['latitude']}x{coarse.sizes['longitude']} -> {target}",
                flush=True,
            )
            try:
                # netCDF4/HDF5 writes are not reliably thread-safe on every
                # workstation build.  A threaded Dask store can deadlock while
                # reading the source NetCDF and writing the cache.  The cache
                # builder is parallelized across countries at the job level;
                # keep each individual file write deterministic and serial.
                with dask.config.set(scheduler="synchronous"):
                    coarse.to_netcdf(tmp, engine="netcdf4", encoding=encoding)
                validate(tmp, expected_hours)
                os.replace(tmp, target)
                wall_s = time.perf_counter() - started
                row.update(
                    status="built",
                    native_points=native_points,
                    processed_points=coarse.sizes["latitude"] * coarse.sizes["longitude"],
                    point_hours=(
                        coarse.sizes["latitude"]
                        * coarse.sizes["longitude"]
                        * coarse.sizes["time"]
                    ),
                    output_bytes=target.stat().st_size,
                    build_wall_s=wall_s,
                )
                print(
                    f"[done] {code} {resolution:g}°: "
                    f"{target.stat().st_size / 1e6:.1f} MB in {wall_s:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                if tmp.exists():
                    tmp.unlink()
                row.update(
                    status="failed",
                    error=repr(exc),
                    build_wall_s=time.perf_counter() - started,
                )
                print(f"[FAIL] {code} {resolution:g}°: {exc}", flush=True)
            rows.append(row)
        ds.close()

    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = results_dir / f"resolution_cache_manifest_{stamp}.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(manifest, index=False)
    print(f"Wrote cache manifest: {manifest}")
    if frame.status.eq("failed").any():
        raise SystemExit("One or more cache builds failed")


if __name__ == "__main__":
    main()
