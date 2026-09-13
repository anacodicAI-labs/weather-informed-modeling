#!/usr/bin/env python3
"""Audit whether country-era data are ready for the spatial model ladder.

The audit checks electricity targets, model-input links, complete native ERA5,
annual capacity maps, and every requested resolution cache. It writes one row
per country-era and exits with status 2 when anything required is incomplete.
Use ``--no-fail`` for an informational inventory during downloads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weather_informed import weather_build as builder
from weather_informed.regions import COUNTRIES, EUROPE_CODES
import run_era_spatial_resolution as runner


DEFAULT_OUTPUT = (
    runner.ROOT
    / "results"
    / "covid_era_spatial_resolution"
    / "data_readiness_audit.csv"
)
REQUIRED_ENERGY_COLUMNS = ("Load", "Renewable_share_of_load")
RECOMMENDED_ENERGY_COVERAGE = 0.98
MINIMUM_MODEL_TARGET_HOURS = 24 * 365
REQUIRED_CACHE_VARIABLES = {
    "wind_speed_100m", "shortwave_radiation", "temperature_2m"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", nargs="+", choices=EUROPE_CODES, default=list(EUROPE_CODES))
    parser.add_argument("--eras", nargs="+", choices=runner.ERAS, default=list(runner.ERAS))
    parser.add_argument("--resolutions", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--unknown-start-policy", choices=("include", "exclude"), default="include")
    parser.add_argument(
        "--minimum-energy-coverage", type=float, default=0.75,
        help=(
            "minimum target coverage needed to fit a model (default: 0.75; "
            "coverage below 0.98 remains explicitly flagged)"
        ),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-fail", action="store_true")
    return parser.parse_args()


def expected_hours(era: str) -> int:
    config = runner.ERAS[era]
    return len(pd.date_range(config["start"], f'{config["end"]} 23:00:00', freq="h"))


def audit_energy(path: Path, era: str, minimum_coverage: float) -> dict:
    result = {
        "energy_source_exists": path.exists(),
        "energy_rows": 0,
        "energy_target_hours": 0,
        "energy_missing_target_hours": 0,
        "energy_longest_missing_run_hours": 0,
        "energy_missing_hours_by_month": "{}",
        "energy_coverage_fraction": 0.0,
        "energy_columns_good": False,
        "energy_mix_components_good": False,
        "energy_mix_problem": "",
        "energy_time_good": False,
        "energy_recommended_coverage": False,
        "energy_source_good": False,
        "energy_problem": "missing file" if not path.exists() else "",
        "energy_warning": "",
    }
    if not path.exists():
        return result
    try:
        frame = pd.read_csv(path)
        if "timestamp" not in frame:
            result["energy_problem"] = "missing timestamp column"
            return result
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        valid_time = timestamps.notna()
        result["energy_rows"] = int(valid_time.sum())
        present = [
            column for column in REQUIRED_ENERGY_COLUMNS
            if column in frame and pd.to_numeric(frame[column], errors="coerce").notna().any()
        ]
        wind_columns = [
            column for column in frame
            if "wind" in column.casefold() and "speed" not in column.casefold()
        ]
        wind_good = any(
            pd.to_numeric(frame[column], errors="coerce").notna().any()
            for column in wind_columns
        )
        solar_good = (
            "Solar" in frame
            and pd.to_numeric(frame["Solar"], errors="coerce").notna().any()
        )
        result["energy_columns_good"] = len(present) == len(REQUIRED_ENERGY_COLUMNS)
        result["energy_mix_components_good"] = bool(solar_good and wind_good)
        missing_mix = []
        if not solar_good:
            missing_mix.append("Solar")
        if not wind_good:
            missing_mix.append("Wind_*")
        result["energy_mix_problem"] = ";".join(missing_mix)
        config = runner.ERAS[era]
        expected_index = pd.date_range(
            config["start"], f'{config["end"]} 23:00:00', freq="h", tz="UTC"
        )
        if "Renewable_share_of_load" in frame:
            target = pd.Series(
                pd.to_numeric(
                    frame["Renewable_share_of_load"], errors="coerce"
                ).to_numpy(),
                index=timestamps,
            )
            target = target[target.index.notna()]
            target = target.groupby(level=0).last().reindex(expected_index)
        else:
            target = pd.Series(np.nan, index=expected_index)
        target_good = target.notna()
        target_valid = int(target_good.sum())
        missing = ~target_good
        missing_groups = missing.ne(missing.shift(fill_value=False)).cumsum()
        missing_runs = missing.groupby(missing_groups).sum()
        missing_by_month = (
            missing.astype(int)
            .groupby(missing.index.strftime("%Y-%m"))
            .sum()
        )
        result["energy_target_hours"] = int(target_valid)
        result["energy_missing_target_hours"] = int(missing.sum())
        result["energy_longest_missing_run_hours"] = int(
            missing_runs.max() if len(missing_runs) else 0
        )
        result["energy_missing_hours_by_month"] = json.dumps(
            {
                month: int(hours)
                for month, hours in missing_by_month.items()
                if hours
            },
            sort_keys=True,
        )
        coverage = float(target_valid / expected_hours(era))
        result["energy_coverage_fraction"] = coverage
        result["energy_recommended_coverage"] = bool(
            coverage >= RECOMMENDED_ENERGY_COVERAGE
        )
        warnings = []
        if not result["energy_recommended_coverage"]:
            warnings.append(
                f"target coverage below recommended 0.98 ({coverage:.3f})"
            )
        if not result["energy_mix_components_good"]:
            warnings.append(
                "wind/solar mix unavailable: " + result["energy_mix_problem"]
            )
        result["energy_warning"] = "; ".join(warnings)
        if valid_time.any():
            start_ok = timestamps[valid_time].min() <= pd.Timestamp(config["start"], tz="UTC") + pd.Timedelta(hours=2)
            end_ok = timestamps[valid_time].max() >= pd.Timestamp(config["end"], tz="UTC") + pd.Timedelta(hours=20)
            enough_targets = (
                coverage >= minimum_coverage
                or target_valid >= MINIMUM_MODEL_TARGET_HOURS
            )
            result["energy_time_good"] = bool(
                start_ok and end_ok and enough_targets
            )
        result["energy_source_good"] = bool(
            result["energy_columns_good"] and result["energy_time_good"]
        )
        if not result["energy_source_good"]:
            problems = []
            if not result["energy_columns_good"]:
                problems.append("missing/non-numeric required energy series")
            if not result["energy_time_good"]:
                problems.append(
                    "insufficient target data "
                    f"({target_valid} hours, coverage={coverage:.3f}; need "
                    f"coverage>={minimum_coverage:.2f} or at least "
                    f"{MINIMUM_MODEL_TARGET_HOURS} hours)"
                )
            result["energy_problem"] = "; ".join(problems)
    except Exception as exc:
        result["energy_problem"] = f"read failure: {exc!r}"
    return result


def audit_native(code: str, era: str) -> tuple[bool, str]:
    config = runner.ERAS[era]
    try:
        dataset, _ = builder.open_local_region(
            code, runner.ERA5_ROOT, config["start"], config["end"]
        )
        dataset.close()
        return True, ""
    except Exception as exc:
        return False, str(exc).splitlines()[0]


def audit_capacity(code: str, era: str, policy: str) -> tuple[bool, str]:
    root = runner.capacity_dir_for(era, policy)
    missing = [
        str(year) for year in runner.ERAS[era]["years"]
        if not (root / str(year) / f"{code}.csv").exists()
    ]
    return not missing, ",".join(missing)


def audit_caches(code: str, era: str, resolutions: list[float]) -> tuple[bool, str]:
    config = runner.ERAS[era]
    cache_root = runner.era_paths(era)["cache"]
    expected = expected_hours(era)
    problems = []
    for resolution in resolutions:
        path = builder.resolution_cache_path(
            cache_root, code, config["start"], config["end"], resolution
        )
        if not path.exists():
            problems.append(f"{resolution:g}° missing")
            continue
        try:
            with xr.open_dataset(path) as dataset:
                missing_vars = REQUIRED_CACHE_VARIABLES.difference(dataset.data_vars)
                hours = dataset.sizes.get("time", 0)
                if missing_vars or hours != expected:
                    problems.append(
                        f"{resolution:g}° invalid(vars={sorted(missing_vars)},hours={hours}/{expected})"
                    )
        except Exception as exc:
            problems.append(f"{resolution:g}° unreadable:{exc!r}")
    return not problems, "; ".join(problems)


def main() -> None:
    args = parse_args()
    if not 0 < args.minimum_energy_coverage <= 1:
        raise ValueError("--minimum-energy-coverage must be in (0, 1]")
    rows = []
    for code in args.codes:
        for era in args.eras:
            source = runner.source_input(code, era)
            energy = audit_energy(source, era, args.minimum_energy_coverage)
            model_input = runner.era_paths(era)["inputs"] / f"weather_energy_merged_{code}.csv"
            native_good, native_problem = audit_native(code, era)
            capacity_good, missing_capacity_years = audit_capacity(
                code, era, args.unknown_start_policy
            )
            cache_good, cache_problem = audit_caches(code, era, args.resolutions)
            input_good = model_input.exists()
            problems = []
            if not energy["energy_source_good"]:
                problems.append(f"energy: {energy['energy_problem']}")
            if not input_good:
                problems.append("model input link/file missing")
            if not native_good:
                problems.append(f"native ERA5: {native_problem}")
            if not capacity_good:
                problems.append(f"capacity years missing: {missing_capacity_years}")
            if not cache_good:
                problems.append(f"cache: {cache_problem}")
            rows.append({
                "code": code,
                "country": COUNTRIES[code],
                "era": era,
                **energy,
                "model_input_good": input_good,
                "native_era5_good": native_good,
                "capacity_maps_good": capacity_good,
                "resolution_caches_good": cache_good,
                "ready_for_analysis": bool(
                    energy["energy_source_good"]
                    and input_good
                    and native_good
                    and capacity_good
                    and cache_good
                ),
                "problems": " | ".join(problems),
            })

    audit = pd.DataFrame(rows).sort_values(["era", "code"]).reset_index(drop=True)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)
    status = (
        audit.groupby("era", as_index=False)
        .agg(country_eras=("code", "size"), ready=("ready_for_analysis", "sum"))
    )
    status["missing_or_invalid"] = status.country_eras - status.ready
    print(status.to_string(index=False))
    failed = audit[~audit.ready_for_analysis]
    if not failed.empty:
        print("\nMissing or invalid country-era data:")
        print(failed[["code", "era", "problems"]].to_string(index=False))
    print(f"\nWrote {output}")
    if not args.no_fail and not failed.empty:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
