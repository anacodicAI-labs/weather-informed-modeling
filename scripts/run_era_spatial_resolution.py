#!/usr/bin/env python3
"""Run the spatial-resolution experiment for all three COVID-defined eras.

This is an orchestration layer around the project's validated spatial scripts;
it does not reimplement weather coarsening or model evaluation.  It prepares
the historical target files in the naming layout expected by
``evaluate_spatial_weather.py``, builds year-specific GEM capacity maps, stages
ERA5, materializes resolution-native caches, runs the ladder, and invokes the
same summarizer used for the post-COVID experiment.

Examples
--------
Show the commands without running them::

    python run_era_spatial_resolution.py --eras pre covid --dry-run

Run the complete pre-COVID, COVID, and post-COVID workflow::

    python run_era_spatial_resolution.py --eras pre covid post --workers 8

Run in resumable phases::

    python run_era_spatial_resolution.py --eras pre covid post \
        --steps fetch inputs capacity stage
    python run_era_spatial_resolution.py --eras pre covid post \
        --steps cache analyze summarize compare --workers 8

The ERA5 staging step requires ``CDSAPI_KEY`` or ``~/.cdsapirc``.  Existing
valid ERA5 and cache files are skipped unless ``--force-era5`` or
``--force-cache`` is supplied.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from country_registry import EUROPE_CODES


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = SCRIPT_DIR.parent
PYTHON = sys.executable

CODES = EUROPE_CODES
ERAS = {
    "pre": {
        "start": "2016-01-01",
        "end": "2019-12-31",
        "years": range(2016, 2020),
    },
    "covid": {
        "start": "2020-01-01",
        "end": "2021-12-31",
        "years": range(2020, 2022),
    },
    "post": {
        "start": "2022-01-01",
        "end": "2026-04-30",
        "years": range(2022, 2027),
    },
}
ALL_STEPS = (
    "fetch", "inputs", "capacity", "stage", "cache", "validate",
    "analyze", "summarize", "compare",
)

SOURCE_TARGET_ROOT = ROOT / "data" / "energy_targets_source_by_era"
ERA_INPUT_ROOT = ROOT / "data" / "energy_targets_by_era"
CAPACITY_HISTORICAL_ROOT = ROOT / "data" / "capacity_weights_pre_and_covid_by_year"
CAPACITY_POST_ROOT = ROOT / "data" / "capacity_weights_post_by_year"
CAPACITY_EXCLUDED_ROOT = ROOT / "data" / "capacity_weights_unknown_start_excluded"
ERA5_ROOT = ROOT / "data" / "era5_native_0p25deg"
CACHE_ROOT = ROOT / "data" / "era5_coarse_by_era"
POST_CACHE_ROOT = ROOT / "data" / "era5_coarse_post_covid"
WEATHER_ROOT = ROOT / "data" / "country_weather_by_era"
POST_WEATHER_ROOT = ROOT / "data" / "country_weather_post_covid"
RESULTS_ROOT = ROOT / "results" / "covid_era_spatial_resolution"
FIGURES_ROOT = ROOT / "figures" / "era_spatial_resolution"
BASELINE_POST_SUMMARY = (
    ROOT
    / "results"
    / "baseline_country_models"
    / "summary_tables"
    / "country_summary.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eras", nargs="+", choices=sorted(ERAS), default=["pre", "covid", "post"])
    parser.add_argument("--codes", nargs="+", choices=CODES, default=list(CODES))
    parser.add_argument("--resolutions", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--schemes", nargs="+", choices=("capacity", "uniform"), default=["capacity", "uniform"])
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=list(ALL_STEPS))
    parser.add_argument("--unknown-start-policy", choices=("include", "exclude"), default="include")
    parser.add_argument("--force-era5", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source_input(code: str, era: str) -> Path:
    return SOURCE_TARGET_ROOT / f"weather_energy_merged_{code}_{era}.csv"


def available_codes(requested: list[str], era: str) -> list[str]:
    found = [code for code in requested if source_input(code, era).exists()]
    missing = sorted(set(requested).difference(found))
    if missing:
        print(f"[{era}] skipping missing historical target files: {', '.join(missing)}")
    if not found:
        raise FileNotFoundError(f"No historical target files found for era={era}")
    return found


def print_command(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)


def run_command(command: list[str], dry_run: bool) -> None:
    print_command(command)
    if not dry_run:
        subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def link_or_copy(source: Path, target: Path) -> None:
    """Atomically expose a historical target under the standard filename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        try:
            if target.resolve() == source.resolve():
                return
        except FileNotFoundError:
            pass
        target.unlink()
    try:
        relative = os.path.relpath(source, target.parent)
        target.symlink_to(relative)
    except OSError:
        shutil.copy2(source, target)


def generation_mix(path: Path) -> tuple[float, float]:
    frame = pd.read_csv(path)
    if "Load" not in frame:
        return np.nan, np.nan
    load = pd.to_numeric(frame["Load"], errors="coerce").sum(min_count=1)
    wind_columns = [
        column for column in frame.columns
        if "wind" in column.lower() and "speed" not in column.lower()
    ]
    solar_columns = [column for column in frame.columns if column.lower() == "solar"]
    if not np.isfinite(load) or load <= 0:
        return np.nan, np.nan
    wind = (
        frame[wind_columns].apply(pd.to_numeric, errors="coerce").sum().sum()
        if wind_columns else np.nan
    )
    solar = (
        frame[solar_columns].apply(pd.to_numeric, errors="coerce").sum().sum()
        if solar_columns else np.nan
    )
    return 100.0 * wind / load, 100.0 * solar / load


def prepare_inputs(era: str, codes: list[str], dry_run: bool) -> Path:
    target_dir = ERA_INPUT_ROOT / era
    rows = []
    for code in codes:
        source = source_input(code, era)
        target = target_dir / f"weather_energy_merged_{code}.csv"
        print(f"[{era}] input {code}: {source} -> {target}")
        if not dry_run:
            link_or_copy(source, target)
            wind, solar = generation_mix(source)
            rows.append(
                {
                    "code": code,
                    "actual_wind_share_pct": wind,
                    "actual_solar_share_pct": solar,
                    "source": str(source),
                }
            )
    if not dry_run:
        result_dir = era_paths(era)["results"]
        result_dir.mkdir(parents=True, exist_ok=True)
        summary_path = result_dir / "country_summary.csv"
        new_rows = pd.DataFrame(rows)

        # Incremental country runs must extend the saved mix summary rather
        # than replacing the original-country rows. The correlation analysis
        # merges model gains to this table by country code.
        if summary_path.exists():
            existing = pd.read_csv(summary_path)
        elif era == "post" and BASELINE_POST_SUMMARY.exists():
            existing = pd.read_csv(BASELINE_POST_SUMMARY)
        else:
            existing = pd.DataFrame()

        keep = [
            "code", "actual_wind_share_pct", "actual_solar_share_pct", "source"
        ]
        for column in keep:
            if column not in existing:
                existing[column] = np.nan
            if column not in new_rows:
                new_rows[column] = np.nan
        combined = (
            pd.concat([existing[keep], new_rows[keep]], ignore_index=True)
            .drop_duplicates("code", keep="last")
            .sort_values("code")
            .reset_index(drop=True)
        )
        combined.to_csv(summary_path, index=False)
        print(
            f"[{era}] merged country summary: {len(combined)} countries -> "
            f"{summary_path}"
        )
    return target_dir


def capacity_command(
    eras: list[str],
    codes: list[str],
    policy: str,
    out_dir: Path,
    results_dir: Path,
) -> list[str]:
    years = sorted({year for era in eras for year in ERAS[era]["years"]})
    return [
        PYTHON,
        str(SCRIPT_DIR / "import_gem_capacity.py"),
        "--codes", *codes,
        "--years", *map(str, years),
        "--unknown-start-policy", policy,
        "--out-dir", str(out_dir),
        "--results-dir", str(results_dir),
    ]


def capacity_dir_for(era: str, policy: str) -> Path:
    if policy == "exclude":
        return CAPACITY_EXCLUDED_ROOT
    return CAPACITY_POST_ROOT if era == "post" else CAPACITY_HISTORICAL_ROOT


def era_paths(era: str) -> dict[str, Path]:
    if era == "post":
        return {
            "inputs": ERA_INPUT_ROOT / era,
            "cache": POST_CACHE_ROOT,
            "weather": POST_WEATHER_ROOT,
            "results": ROOT / "results" / "post_covid_spatial_resolution",
            "figures": ROOT / "figures" / "spatial_resolution",
        }
    return {
        "inputs": ERA_INPUT_ROOT / era,
        "cache": CACHE_ROOT / era,
        "weather": WEATHER_ROOT / era,
        "results": RESULTS_ROOT / era,
        "figures": FIGURES_ROOT / era,
    }


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not 0 < args.split < 1:
        raise ValueError("--split must be between 0 and 1")
    if "fetch" in args.steps:
        run_command(
            [
                PYTHON,
                str(SCRIPT_DIR / "fetch_era_energy_inputs.py"),
                "--eras", *args.eras,
                "--codes", *args.codes,
            ],
            args.dry_run,
        )

    if args.dry_run and "fetch" in args.steps:
        # The fetch command is only printed in a dry run, so its future files
        # do not exist yet. Keep every requested code in the command preview.
        selected = {era: list(args.codes) for era in args.eras}
    else:
        selected = {era: available_codes(args.codes, era) for era in args.eras}

    if "inputs" in args.steps:
        for era in args.eras:
            prepare_inputs(era, selected[era], args.dry_run)

    if "capacity" in args.steps:
        capacity_codes = sorted({code for codes in selected.values() for code in codes})
        if args.unknown_start_policy == "exclude":
            run_command(
                capacity_command(
                    args.eras,
                    capacity_codes,
                    "exclude",
                    CAPACITY_EXCLUDED_ROOT,
                    (
                        era_paths("post")["results"] / "unknown_start_excluded"
                        if args.eras == ["post"]
                        else RESULTS_ROOT / "capacity_maps_unknown_start_excluded"
                    ),
                ),
                args.dry_run,
            )
        else:
            historical = [era for era in args.eras if era != "post"]
            if historical:
                run_command(
                    capacity_command(
                        historical,
                        capacity_codes,
                        "include",
                        CAPACITY_HISTORICAL_ROOT,
                        RESULTS_ROOT / "capacity_maps",
                    ),
                    args.dry_run,
                )
            if "post" in args.eras:
                run_command(
                    capacity_command(
                        ["post"],
                        capacity_codes,
                        "include",
                        CAPACITY_POST_ROOT,
                        era_paths("post")["results"],
                    ),
                    args.dry_run,
                )

    for era in args.eras:
        config = ERAS[era]
        paths = era_paths(era)
        codes = selected[era]

        # Input links and resource-mix summaries are cheap and required by the
        # analysis/summarizer, so ensure they exist when those later steps are
        # selected directly in a resumed run.
        if not args.dry_run and any(step in args.steps for step in ("analyze", "summarize")):
            prepare_inputs(era, codes, False)

        if "stage" in args.steps:
            command = [
                PYTHON,
                str(SCRIPT_DIR / "stage_era5_arco.py"),
                "--codes", *codes,
                "--start", config["start"],
                "--end", config["end"],
                "--out-dir", str(ERA5_ROOT),
                "--results-dir", str(paths["results"] / "era5_staging"),
            ]
            if args.force_era5:
                command.append("--force")
            run_command(command, args.dry_run)

        if "cache" in args.steps:
            command = [
                PYTHON,
                str(SCRIPT_DIR / "prepare_era5_resolution_cache.py"),
                "--codes", *codes,
                "--resolutions", *map(str, args.resolutions),
                "--start", config["start"],
                "--end", config["end"],
                "--era5-dir", str(ERA5_ROOT),
                "--out-dir", str(paths["cache"]),
                "--results-dir", str(paths["results"] / "resolution_cache_manifests"),
            ]
            if args.force_cache:
                command.append("--force")
            run_command(command, args.dry_run)

        if "validate" in args.steps:
            run_command(
                [
                    PYTHON,
                    str(SCRIPT_DIR / "audit_spatial_pipeline_data.py"),
                    "--codes", *codes,
                    "--eras", era,
                    "--resolutions", *map(str, args.resolutions),
                    "--unknown-start-policy", args.unknown_start_policy,
                    "--output", str(paths["results"] / "data_readiness_audit.csv"),
                ],
                args.dry_run,
            )

        if "analyze" in args.steps:
            run_command(
                [
                    PYTHON,
                    str(SCRIPT_DIR / "run_spatial_resolution_ladder.py"),
                    "--codes", *codes,
                    "--resolutions", *map(str, args.resolutions),
                    "--schemes", *args.schemes,
                    "--workers", str(args.workers),
                    "--start", config["start"],
                    "--end", config["end"],
                    "--split", str(args.split),
                    "--era5-dir", str(ERA5_ROOT),
                    "--capacity-dir", str(
                        capacity_dir_for(era, args.unknown_start_policy)
                    ),
                    "--weather-out-dir", str(paths["weather"]),
                    "--resolution-cache-dir", str(paths["cache"]),
                    "--data-dir", str(paths["inputs"]),
                    "--results-dir", str(paths["results"]),
                ],
                args.dry_run,
            )

        if "summarize" in args.steps:
            run_command(
                [
                    PYTHON,
                    str(SCRIPT_DIR / "summarize_spatial_resolution.py"),
                    "--results-dir", str(paths["results"]),
                    "--country-summary", str(paths["results"] / "country_summary.csv"),
                    "--figure-dir", str(paths["figures"]),
                ],
                args.dry_run,
            )

    if "compare" in args.steps:
        run_command(
            [
                PYTHON,
                str(SCRIPT_DIR / "compare_era_spatial_resolution.py"),
                "--eras", *dict.fromkeys(args.eras),
            ],
            args.dry_run,
        )


if __name__ == "__main__":
    main()
