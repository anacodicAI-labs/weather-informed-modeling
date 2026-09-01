#!/usr/bin/env python3
"""Stage regional ERA5 data by reading the official geo-chunked ARCO store.

Unlike ``stage_era5_cds.py``, this script does not submit queued CDS retrieval
jobs.  It reads only the required cloud chunks and writes one resumable NetCDF
file per country.  The output is compatible with
``build_weighted_weather_local.py --era5-dir ...``.

The CDS personal-access token is read from ``CDSAPI_KEY`` or ``~/.cdsapirc``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from country_registry import BBOX


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_OUT_DIR = PROJECT_DIR / "data" / "era5_native_0p25deg"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results" / "post_covid_spatial_resolution"

ARCO_URL = (
    "https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/"
    "reanalysis_era5_single_levels/sfc/geoChunked.zarr"
)

VARIABLES = ("u100", "v100", "t2m", "ssrd")


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", nargs="+", choices=sorted(BBOX), required=True)
    parser.add_argument("--start", type=parse_date, default=parse_date("2022-01-01"))
    parser.add_argument("--end", type=parse_date, default=parse_date("2026-04-30"))
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_cds_key() -> str:
    key = os.environ.get("CDSAPI_KEY", "").strip()
    if key:
        return key

    config = Path.home() / ".cdsapirc"
    if config.exists():
        for line in config.read_text().splitlines():
            if line.strip().startswith("key:"):
                key = line.split(":", 1)[1].strip()
                if key:
                    return key
    raise RuntimeError(
        "CDS API key not found. Set CDSAPI_KEY or add key: ... to ~/.cdsapirc"
    )


def latitude_slice(latitude, south: float, north: float):
    first = float(latitude.values[0])
    last = float(latitude.values[-1])
    return slice(south, north) if first < last else slice(north, south)


def validate_output(path: Path) -> tuple[int, str, str]:
    import xarray as xr

    with xr.open_dataset(path) as ds:
        missing = [name for name in VARIABLES if name not in ds]
        if missing:
            raise ValueError(f"{path} is missing variables {missing}")
        if "time" not in ds.coords or not ds.sizes.get("time", 0):
            raise ValueError(f"{path} has no time records")
        times = pd.to_datetime(ds.time.values)
        return len(times), str(times.min()), str(times.max())


def main() -> None:
    args = parse_args()
    if args.end < args.start:
        raise ValueError("--end must be on or after --start")
    if args.padding < 0:
        raise ValueError("--padding cannot be negative")

    out_root = Path(args.out_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for code in args.codes:
            print(json.dumps({
                "code": code,
                "bbox_south_north_west_east": BBOX[code],
                "start": str(args.start),
                "end": str(args.end),
                "variables": VARIABLES,
                "source": ARCO_URL,
            }, indent=2))
        return

    try:
        import dask  # noqa: F401
        import xarray as xr
        import zarr  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Install ARCO dependencies first: pip install 'xarray>=2024.10' "
            "'zarr<3' dask fsspec netCDF4"
        ) from exc

    key = read_cds_key()
    print("Opening official ERA5 geo-chunked ARCO store...", flush=True)
    ds = xr.open_zarr(
        ARCO_URL,
        consolidated=True,
        storage_options={"headers": {"Authorization": f"Bearer {key}"}},
    )
    missing = [name for name in VARIABLES if name not in ds]
    if missing:
        raise KeyError(f"ARCO store is missing required variables {missing}")

    requested_start = pd.Timestamp(args.start)
    requested_end = pd.Timestamp(args.end) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    available_start = pd.Timestamp(ds.time.values[0])
    available_end = pd.Timestamp(ds.time.values[-1])
    if requested_start < available_start or requested_end > available_end:
        raise ValueError(
            f"Requested {requested_start}..{requested_end}, but ARCO currently has "
            f"{available_start}..{available_end}"
        )

    manifest_rows = []
    for code in args.codes:
        code_dir = out_root / code
        code_dir.mkdir(parents=True, exist_ok=True)
        target = code_dir / (
            f"era5_{code}_arco_{args.start.isoformat()}_{args.end.isoformat()}_0.25deg.nc"
        )
        tmp = target.with_suffix(target.suffix + ".part")
        row = {"code": code, "path": str(target), "source": ARCO_URL}

        if target.exists() and not args.force:
            records, first, last = validate_output(target)
            row.update(
                status="skipped_existing", records=records, first_time=first,
                last_time=last, bytes=target.stat().st_size, wall_s=0.0,
            )
            manifest_rows.append(row)
            print(f"[skip] {code}: {target}", flush=True)
            continue

        south, north, west, east = BBOX[code]
        south -= args.padding
        north += args.padding
        west -= args.padding
        east += args.padding
        subset = ds[list(VARIABLES)].sel(
            time=slice(requested_start, requested_end),
            latitude=latitude_slice(ds.latitude, south, north),
            longitude=slice(west, east),
        )
        if not subset.sizes.get("latitude", 0) or not subset.sizes.get("longitude", 0):
            raise ValueError(f"Empty ARCO spatial selection for {code}")

        # Write one week per output chunk, matching the downstream sweep.
        subset = subset.chunk({"time": 168, "latitude": -1, "longitude": -1})
        encoding = {
            name: {
                "chunksizes": (
                    min(168, subset.sizes["time"]),
                    subset.sizes["latitude"],
                    subset.sizes["longitude"],
                ),
                "zlib": True,
                "complevel": 1,
                "shuffle": True,
            }
            for name in VARIABLES
        }

        if tmp.exists():
            tmp.unlink()
        started = time.perf_counter()
        print(
            f"[stream] {code}: {subset.sizes['time']} hours, "
            f"{subset.sizes['latitude']}x{subset.sizes['longitude']} grid -> {target}",
            flush=True,
        )
        try:
            subset.to_netcdf(tmp, engine="netcdf4", encoding=encoding)
            records, first, last = validate_output(tmp)
            os.replace(tmp, target)
            wall_s = time.perf_counter() - started
            row.update(
                status="downloaded", records=records, first_time=first,
                last_time=last, bytes=target.stat().st_size, wall_s=wall_s,
            )
            print(
                f"[done] {code}: {target.stat().st_size / 1e6:.1f} MB in "
                f"{wall_s:.1f}s",
                flush=True,
            )
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            row.update(
                status="failed", error=repr(exc),
                wall_s=time.perf_counter() - started,
            )
            print(f"[FAIL] {code}: {exc}", flush=True)
        manifest_rows.append(row)

    stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest_path = results_dir / f"era5_arco_manifest_{stamp}.csv"
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote ARCO manifest: {manifest_path}")
    if len(manifest) and manifest.status.eq("failed").any():
        raise SystemExit("One or more ARCO streams failed; rerun to resume")


if __name__ == "__main__":
    main()
