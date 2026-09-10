#!/usr/bin/env python3
"""Download spatially subsetted ERA5 data from the Copernicus CDS.

This replaces repeated small-country reads from the globally chunked Google
ARCO native-grid store. CDS performs the spatial subset before transfer. Each
country-year is an independent, resumable file. Download/staging time is kept
separate from the later HPC timing experiment.

Prerequisites
-------------
1. Create a free CDS account and accept the ERA5 single-level dataset licence.
2. Configure ``~/.cdsapirc`` as described by the CDS.
3. ``pip install cdsapi xarray netCDF4``

Examples
--------
python stage_era5_cds.py --codes dk --start 2022-01-01 --end 2022-01-31
python stage_era5_cds.py --codes dk ie nl
python stage_era5_cds.py --codes tx --dry-run
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import random
import tempfile
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_OUT_DIR = PROJECT_DIR / "data" / "era5_regional"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results" / "post_covid_spatial_resolution"

DATASET = "reanalysis-era5-single-levels"
VARIABLES = [
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "2m_temperature",
    "surface_solar_radiation_downwards",
]

BBOX = {
    # Denmark and the Netherlands extend slightly farther west to retain the
    # geolocated offshore-wind phases in the GEM capacity maps.
    "dk": (54.5, 57.8, 7.5, 15.2), "ie": (51.4, 55.4, -10.6, -5.9),
    # PT intentionally stages mainland Portugal only. REN's interconnected
    # transmission/load series does not cover the separate Madeira/Azores
    # island systems; their GEM plants are audited as out-of-system scope.
    "nl": (50.7, 53.6, 3.0, 7.2), "pt": (36.9, 42.2, -9.6, -6.2),
    "gr": (34.8, 41.8, 19.3, 28.3), "be": (49.5, 51.5, 2.5, 6.4),
    "lt": (53.9, 56.5, 20.9, 26.9), "hr": (42.4, 46.6, 13.5, 19.4),
    "bg": (41.2, 44.2, 22.4, 28.6), "lv": (55.7, 58.1, 21.0, 28.2),
    "si": (45.4, 46.9, 13.4, 16.6), "rs": (42.2, 46.2, 18.8, 23.0),
    "sk": (47.7, 49.6, 16.8, 22.6), "tx": (25.8, 36.5, -106.6, -93.5),
}


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def is_temporary_cds_error(exc: Exception) -> bool:
    """Return True for CDS queue/network failures that are safe to retry."""
    message = str(exc).lower()
    retryable_messages = (
        "number queued requests for this dataset is temporarily limited",
        "temporarily limited",
        "too many requests",
        "429 client error",
        "502 bad gateway",
        "503 service unavailable",
        "504 gateway timeout",
        "connection reset",
        "connection aborted",
        "read timed out",
        "connect timeout",
        "timed out",
        "timeout",
    )
    return any(text in message for text in retryable_messages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", nargs="+", choices=sorted(BBOX), default=sorted(BBOX))
    parser.add_argument("--start", type=parse_date, default=parse_date("2022-01-01"))
    parser.add_argument("--end", type=parse_date, default=parse_date("2026-04-30"))
    parser.add_argument("--grid", type=float, default=0.25)
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument(
        "--months-per-request",
        type=int,
        default=1,
        help="Number of consecutive months per CDS request (default: 1)",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Retries for temporary CDS errors; 0 means keep retrying (default: 0)",
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=120.0,
        help="Initial wait after a temporary CDS error (default: 120 seconds)",
    )
    parser.add_argument(
        "--retry-max-seconds",
        type=float,
        default=1800.0,
        help="Maximum wait between retries (default: 1800 seconds)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def months_for_year(start: date, end: date, year: int) -> list[int]:
    first = start.month if year == start.year else 1
    last = end.month if year == end.year else 12
    return list(range(first, last + 1))


def request_for(code: str, year: int, months: list[int], grid: float, padding: float) -> dict:
    south, north, west, east = BBOX[code]
    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [f"{month:02d}" for month in months],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        # CDS order is north, west, south, east.
        "area": [north + padding, west - padding, south - padding, east + padding],
        "grid": [grid, grid],
    }


def validate_netcdf(path: Path) -> tuple[int, str, str]:
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("Install xarray and netCDF4 to validate downloads") from exc
    with xr.open_dataset(path) as ds:
        time_name = "valid_time" if "valid_time" in ds.coords else "time"
        if time_name not in ds.coords:
            raise ValueError(f"No time coordinate in {path}; coordinates={list(ds.coords)}")
        aliases = [
            ("u100", "100m_u_component_of_wind"),
            ("v100", "100m_v_component_of_wind"),
            ("t2m", "2m_temperature"),
            ("ssrd", "surface_solar_radiation_downwards"),
        ]
        missing = [pair for pair in aliases if not any(name in ds for name in pair)]
        if missing:
            raise ValueError(f"Missing ERA5 variables {missing}; present={list(ds.data_vars)}")
        values = pd.to_datetime(ds[time_name].values)
        if len(values) == 0:
            raise ValueError(f"No time records in {path}")
        return len(values), str(values.min()), str(values.max())


def normalize_cds_download(path: Path) -> None:
    """Convert CDS zip responses into the single NetCDF file expected downstream."""
    with path.open("rb") as handle:
        signature = handle.read(4)
    if signature != b"PK\x03\x04":
        return

    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("Install xarray and netCDF4 to normalize zipped CDS downloads") from exc

    normalized = path.with_suffix(path.suffix + ".unzipped")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        extracted: list[Path] = []
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.lower().endswith((".nc", ".netcdf")):
                    continue
                out_path = tmp_dir_path / Path(member.filename).name
                with archive.open(member) as source, out_path.open("wb") as dest:
                    dest.write(source.read())
                extracted.append(out_path)

        if not extracted:
            raise ValueError(f"CDS returned a zip archive with no NetCDF files: {path}")

        datasets = []
        try:
            for extracted_path in extracted:
                with xr.open_dataset(extracted_path) as ds:
                    datasets.append(ds.load())
            merged = xr.merge(datasets, compat="override", join="outer")
            # Store one week per time chunk. The downstream spatial sweep also
            # works in week-sized blocks, so this avoids repeatedly reading a
            # full country-year chunk for every Dask task.
            encoding = {}
            for name, variable in merged.data_vars.items():
                if not variable.dims:
                    continue
                chunksizes = tuple(
                    min(168, variable.sizes[dim])
                    if dim in {"time", "valid_time"}
                    else variable.sizes[dim]
                    for dim in variable.dims
                )
                encoding[name] = {
                    "chunksizes": chunksizes,
                    "zlib": True,
                    "complevel": 1,
                    "shuffle": True,
                }
            merged.to_netcdf(normalized, engine="netcdf4", encoding=encoding)
            merged.close()
        finally:
            for ds in datasets:
                ds.close()

    os.replace(normalized, path)


def main() -> None:
    args = parse_args()
    if args.end < args.start:
        raise ValueError("--end must be on or after --start")
    if args.grid <= 0:
        raise ValueError("--grid must be positive")
    if args.months_per_request < 1:
        raise ValueError("--months-per-request must be at least 1")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be zero or positive")
    if args.retry_base_seconds <= 0 or args.retry_max_seconds <= 0:
        raise ValueError("retry wait times must be positive")
    if args.retry_max_seconds < args.retry_base_seconds:
        raise ValueError("--retry-max-seconds must be at least --retry-base-seconds")

    client = None

    out_root = Path(args.out_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    for code in args.codes:
        code_dir = out_root / code
        code_dir.mkdir(parents=True, exist_ok=True)
        for year in range(args.start.year, args.end.year + 1):
            year_months = months_for_year(args.start, args.end, year)
            month_batches = [
                year_months[index : index + args.months_per_request]
                for index in range(0, len(year_months), args.months_per_request)
            ]
            for months in month_batches:
                request = request_for(code, year, months, args.grid, args.padding)
                month_tag = f"m{months[0]:02d}-{months[-1]:02d}"
                target = code_dir / f"era5_{code}_{year}_{month_tag}_{args.grid:g}deg.nc"
                tmp = target.with_suffix(target.suffix + ".part")
                row = {
                    "code": code,
                    "year": year,
                    "months": month_tag,
                    "grid_deg": args.grid,
                    "path": str(target),
                    "status": "pending",
                }

                if target.exists() and not args.force:
                    n, first, last = validate_netcdf(target)
                    row.update(
                        status="skipped_existing", records=n, first_time=first,
                        last_time=last, bytes=target.stat().st_size, wall_s=0.0,
                    )
                    manifest_rows.append(row)
                    print(f"[skip] {code} {year} {month_tag}: {target}", flush=True)
                    continue

                print(f"[request] {code} {year} {month_tag} -> {target}", flush=True)
                if args.dry_run:
                    print(json.dumps(request, indent=2))
                    row["status"] = "dry_run"
                    manifest_rows.append(row)
                    continue

                started = time.perf_counter()
                retries = 0
                while True:
                    if tmp.exists():
                        tmp.unlink()
                    try:
                        if client is None:
                            try:
                                import cdsapi
                            except ImportError as exc:
                                raise RuntimeError(
                                    "Install the CDS client first: pip install cdsapi"
                                ) from exc
                            client = cdsapi.Client(quiet=False, progress=True)
                        client.retrieve(DATASET, request, str(tmp))
                        normalize_cds_download(tmp)
                        n, first, last = validate_netcdf(tmp)
                        os.replace(tmp, target)
                        row.update(
                            status="downloaded", records=n, first_time=first,
                            last_time=last, bytes=target.stat().st_size,
                            wall_s=time.perf_counter() - started, retries=retries,
                        )
                        print(
                            f"[done] {code} {year} {month_tag}: "
                            f"{target.stat().st_size / 1e6:.1f} MB, "
                            f"{row['wall_s']:.1f}s, {n} timestamps, "
                            f"{retries} retries",
                            flush=True,
                        )
                        break
                    except Exception as exc:
                        retryable = is_temporary_cds_error(exc)
                        retry_available = (
                            args.max_retries == 0 or retries < args.max_retries
                        )
                        if retryable and retry_available:
                            retries += 1
                            exponential_wait = args.retry_base_seconds * (2 ** (retries - 1))
                            capped_wait = min(args.retry_max_seconds, exponential_wait)
                            wait_seconds = capped_wait * random.uniform(0.8, 1.2)
                            client = None
                            print(
                                f"[retry] {code} {year} {month_tag}: CDS is temporarily "
                                f"limiting queued requests; retry {retries} in "
                                f"{wait_seconds:.0f}s (same month)",
                                flush=True,
                            )
                            time.sleep(wait_seconds)
                            continue

                        row.update(
                            status="failed", error=repr(exc), retries=retries,
                            wall_s=time.perf_counter() - started,
                        )
                        print(f"[FAIL] {code} {year} {month_tag}: {exc}", flush=True)
                        break
                manifest_rows.append(row)

    manifest = pd.DataFrame(manifest_rows)
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest_path = results_dir / f"era5_download_manifest_{stamp}.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote download manifest: {manifest_path}")
    failures = manifest[manifest.status.eq("failed")] if len(manifest) else manifest
    if len(failures):
        raise SystemExit(f"{len(failures)} download(s) failed; rerun to resume")


if __name__ == "__main__":
    main()
