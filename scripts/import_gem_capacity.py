#!/usr/bin/env python3
"""Import GEM wind and solar trackers into annual capacity-point maps.

The importer reads the February 2026 Global Wind Power Tracker and Global
Solar Power Tracker workbooks, keeps geolocated operating phases, and writes
one capacity map per region and year.  Wind and solar phases at identical
coordinates are combined without rounding; the ERA5 weighting stage remains
responsible for assigning these full-precision locations to its requested grid.

Outputs
-------
data/capacity_weights_post_by_year/<year>/<code>.csv
    Columns: lat, lon, wind_mw, solar_mw
results/post_covid_spatial_resolution/gem_capacity_coverage.csv
    Per-region/year coverage and quality statistics.
results/post_covid_spatial_resolution/gem_capacity_records.csv
    Normalized phase-level audit trail used to build the maps.
results/post_covid_spatial_resolution/gem_distributed_solar_reference.csv
    National distributed-solar totals from GEM.  These have no coordinates and
    therefore are reported but never placed into the spatial weights.

By default, operating phases with an unknown start year are included in every
requested year and explicitly reported.  Use ``--unknown-start-policy exclude``
for a stricter sensitivity run.  Solar capacities labelled MWp/dc are converted
to MWac-equivalent using the configurable ``--solar-dc-to-ac`` factor.  Unknown
solar capacity ratings are retained as reported and flagged in the audit files.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from country_registry import COUNTRIES


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "post_covid_spatial_resolution"
DEFAULT_GEM_DIR = DATA_DIR / "raw" / "gem"
DEFAULT_WIND_XLSX = DEFAULT_GEM_DIR / "Global-Wind-Power-Tracker-February-2026.xlsx"
DEFAULT_SOLAR_XLSX = DEFAULT_GEM_DIR / "Global-Solar-Power-Tracker-February-2026.xlsx"
DEFAULT_OUT_DIR = DATA_DIR / "capacity_weights_post_by_year"
DEFAULT_COMPAT_DIR = DATA_DIR / "capacity_weights_static_legacy"

REGIONS = {**COUNTRIES, "tx": "Texas"}
EUROPE_CODES = tuple(code for code in REGIONS if code != "tx")
COUNTRY_TO_CODE = {
    country.casefold(): code
    for code, country in REGIONS.items()
    if code != "tx"
}

WIND_SHEETS = ("Data", "Below Threshold")
SOLAR_SHEET = "Utility-Scale (1 MW+)"
DISTRIBUTED_SOLAR_SHEET = "Distributed (<1 MW)"

COLUMN_ALIASES = {
    "country": ("Country/Area", "Country", "Country or Area"),
    "project": ("Project Name", "Project"),
    "phase": ("Phase Name", "Phase"),
    "capacity": ("Capacity (MW)", "Capacity MW", "Capacity"),
    "capacity_rating": ("Capacity Rating", "Rating"),
    "status": ("Status", "Project Status"),
    "start_year": ("Start year", "Start Year", "Operating year"),
    "retired_year": ("Retired year", "Retired Year"),
    "latitude": ("Latitude", "lat"),
    "longitude": ("Longitude", "lon", "lng"),
    "location_accuracy": ("Location accuracy", "Location Accuracy"),
    "state": ("State/Province", "State", "Province"),
    "gem_location_id": ("GEM location ID", "GEM Location ID"),
    "gem_phase_id": ("GEM phase ID", "GEM Phase ID"),
    "date_researched": ("Date Last Researched", "Last Researched"),
    "wiki_url": ("Wiki URL", "URL"),
}


def normalized_name(value: object) -> str:
    return " ".join(str(value).strip().casefold().replace("_", " ").split())


def resolve_column(frame: pd.DataFrame, key: str, required: bool = True) -> str | None:
    lookup = {normalized_name(column): str(column) for column in frame.columns}
    for alias in COLUMN_ALIASES[key]:
        match = lookup.get(normalized_name(alias))
        if match is not None:
            return match
    if required:
        raise KeyError(
            f"Missing GEM column for {key!r}; tried {COLUMN_ALIASES[key]}; "
            f"available columns={list(frame.columns)}"
        )
    return None


def read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet, dtype=object)
    except ValueError as exc:
        book = pd.ExcelFile(path)
        lookup = {normalized_name(name): name for name in book.sheet_names}
        actual = lookup.get(normalized_name(sheet))
        if actual is None:
            raise KeyError(
                f"Missing sheet {sheet!r} in {path}; available={book.sheet_names}"
            ) from exc
        return pd.read_excel(path, sheet_name=actual, dtype=object)


def numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def assign_codes(country: pd.Series, state: pd.Series) -> pd.Series:
    country_norm = country.fillna("").map(normalized_name)
    state_norm = state.fillna("").map(normalized_name)
    codes = country_norm.map(COUNTRY_TO_CODE)
    united_states = country_norm.isin(
        {"united states", "united states of america", "usa", "u.s.", "us"}
    )
    codes.loc[united_states & state_norm.str.contains(r"\btexas\b", regex=True)] = "tx"
    return codes


def normalize_phases(
    frame: pd.DataFrame,
    technology: str,
    source_sheet: str,
    source_path: Path,
    codes: set[str],
    solar_dc_to_ac: float,
) -> tuple[pd.DataFrame, list[dict]]:
    columns = {
        key: resolve_column(
            frame,
            key,
            required=key
            in {
                "country",
                "project",
                "capacity",
                "status",
                "start_year",
                "latitude",
                "longitude",
            },
        )
        for key in COLUMN_ALIASES
    }

    def values(key: str, default: object = "") -> pd.Series:
        column = columns[key]
        if column is None:
            return pd.Series(default, index=frame.index, dtype=object)
        return frame[column]

    out = pd.DataFrame(index=frame.index)
    out["country"] = values("country").astype(str).str.strip()
    out["state_province"] = values("state").astype(str).str.strip()
    out["code"] = assign_codes(out["country"], out["state_province"])
    out["status"] = values("status").astype(str).str.strip()
    out["project_name"] = values("project").astype(str).str.strip()
    out["phase_name"] = values("phase").astype(str).str.strip()
    out["capacity_mw_reported"] = numeric(values("capacity"))
    out["capacity_rating"] = values("capacity_rating", "MW").astype(str).str.strip()
    out["start_year"] = numeric(values("start_year"))
    out["retired_year"] = numeric(values("retired_year"))
    out["latitude"] = numeric(values("latitude"))
    out["longitude"] = numeric(values("longitude"))
    out["location_accuracy"] = values("location_accuracy").astype(str).str.strip()
    out["gem_location_id"] = values("gem_location_id").astype(str).str.strip()
    out["gem_phase_id"] = values("gem_phase_id").astype(str).str.strip()
    out["date_last_researched"] = values("date_researched").astype(str).str.strip()
    out["wiki_url"] = values("wiki_url").astype(str).str.strip()
    out["technology"] = technology
    out["source_sheet"] = source_sheet
    out["source_file"] = str(source_path)

    relevant = out["code"].isin(codes) & out["status"].map(normalized_name).eq("operating")
    out = out.loc[relevant].copy()
    positive_capacity = out["capacity_mw_reported"].gt(0) & np.isfinite(
        out["capacity_mw_reported"]
    )
    valid_coordinates = (
        out["latitude"].between(-90, 90)
        & out["longitude"].between(-180, 180)
        & np.isfinite(out["latitude"])
        & np.isfinite(out["longitude"])
    )

    diagnostics = []
    for code in sorted(codes):
        subset = out[out["code"].eq(code)]
        diagnostics.append(
            {
                "code": code,
                "country": REGIONS[code],
                "technology": technology,
                "source_sheet": source_sheet,
                "operating_rows": int(len(subset)),
                "valid_rows": int(
                    (positive_capacity.loc[subset.index] & valid_coordinates.loc[subset.index]).sum()
                ),
                "invalid_or_missing_capacity_rows": int(
                    (~positive_capacity.loc[subset.index]).sum()
                ),
                "invalid_or_missing_coordinate_rows": int(
                    (~valid_coordinates.loc[subset.index]).sum()
                ),
                "unknown_start_rows": int(subset["start_year"].isna().sum()),
                "unknown_start_reported_mw": float(
                    subset.loc[subset["start_year"].isna(), "capacity_mw_reported"].sum()
                ),
            }
        )

    out = out.loc[positive_capacity & valid_coordinates].copy()
    rating_norm = out["capacity_rating"].map(normalized_name)
    is_dc = rating_norm.str.contains("dc", regex=False) | rating_norm.str.contains(
        "mwp", regex=False
    )
    out["capacity_mw_weight"] = out["capacity_mw_reported"]
    if technology == "solar":
        out.loc[is_dc, "capacity_mw_weight"] *= solar_dc_to_ac
    out["solar_dc_converted"] = technology == "solar"
    if technology == "solar":
        out["solar_dc_converted"] = is_dc
    out["unknown_capacity_rating"] = rating_norm.isin({"", "nan", "none", "unknown"})
    out["unknown_start_year"] = out["start_year"].isna()

    fallback_key = (
        out["source_sheet"].astype(str)
        + "|"
        + out["country"].astype(str)
        + "|"
        + out["project_name"].astype(str)
        + "|"
        + out["phase_name"].astype(str)
        + "|"
        + out["latitude"].astype(str)
        + "|"
        + out["longitude"].astype(str)
    )
    phase_id = out["gem_phase_id"].replace({"": np.nan, "nan": np.nan, "None": np.nan})
    out["record_key"] = phase_id.fillna(fallback_key)
    return out.reset_index(drop=True), diagnostics


def active_in_year(
    records: pd.DataFrame, year: int, unknown_start_policy: str
) -> pd.DataFrame:
    known_started = records["start_year"].notna() & records["start_year"].le(year)
    unknown_started = (
        records["start_year"].isna()
        if unknown_start_policy == "include"
        else pd.Series(False, index=records.index)
    )
    not_retired = records["retired_year"].isna() | records["retired_year"].gt(year)
    return records.loc[(known_started | unknown_started) & not_retired].copy()


def build_map(records: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for technology, output_column in (("wind", "wind_mw"), ("solar", "solar_mw")):
        subset = records[records["technology"].eq(technology)]
        if subset.empty:
            continue
        grouped = (
            subset.groupby(["latitude", "longitude"], as_index=False)["capacity_mw_weight"]
            .sum()
            .rename(
                columns={
                    "latitude": "lat",
                    "longitude": "lon",
                    "capacity_mw_weight": output_column,
                }
            )
        )
        pieces.append(grouped)

    if not pieces:
        return pd.DataFrame(columns=["lat", "lon", "wind_mw", "solar_mw"])
    result = pieces[0]
    for piece in pieces[1:]:
        result = result.merge(piece, on=["lat", "lon"], how="outer")
    for column in ("wind_mw", "solar_mw"):
        if column not in result:
            result[column] = 0.0
    return (
        result[["lat", "lon", "wind_mw", "solar_mw"]]
        .fillna(0.0)
        .sort_values(["lat", "lon"])
        .reset_index(drop=True)
    )


def coverage_row(
    records: pd.DataFrame,
    code: str,
    year: int,
    unknown_start_policy: str,
    solar_dc_to_ac: float,
) -> dict:
    row: dict[str, object] = {
        "code": code,
        "country": REGIONS[code],
        "year": year,
        "unknown_start_policy": unknown_start_policy,
        "solar_dc_to_ac": solar_dc_to_ac,
    }
    for technology in ("wind", "solar"):
        subset = records[records["technology"].eq(technology)]
        row[f"{technology}_phases"] = int(len(subset))
        row[f"{technology}_points"] = int(
            subset[["latitude", "longitude"]].drop_duplicates().shape[0]
        )
        row[f"{technology}_reported_mw"] = float(subset["capacity_mw_reported"].sum())
        row[f"{technology}_weight_mw"] = float(subset["capacity_mw_weight"].sum())
        row[f"{technology}_unknown_start_phases"] = int(
            subset["unknown_start_year"].sum()
        )
        row[f"{technology}_unknown_start_weight_mw"] = float(
            subset.loc[subset["unknown_start_year"], "capacity_mw_weight"].sum()
        )
        row[f"{technology}_approximate_location_phases"] = int(
            subset["location_accuracy"].map(normalized_name).ne("exact").sum()
        )
        row[f"{technology}_unknown_rating_weight_mw"] = float(
            subset.loc[subset["unknown_capacity_rating"], "capacity_mw_weight"].sum()
        )
    return row


def distributed_solar_reference(
    solar_path: Path, codes: set[str]
) -> pd.DataFrame:
    try:
        frame = read_sheet(solar_path, DISTRIBUTED_SOLAR_SHEET)
    except KeyError:
        return pd.DataFrame(
            columns=["code", "country", "year", "capacity_mw", "capacity_rating", "source_file"]
        )
    country_col = resolve_column(frame, "country")
    capacity_col = resolve_column(frame, "capacity")
    rating_col = resolve_column(frame, "capacity_rating", required=False)
    state_col = resolve_column(frame, "state", required=False)
    year_col = next(
        (column for column in frame.columns if normalized_name(column) == "year"), None
    )
    state = frame[state_col] if state_col else pd.Series("", index=frame.index)
    out = pd.DataFrame(
        {
            "country": frame[country_col].astype(str).str.strip(),
            "capacity_mw": numeric(frame[capacity_col]),
            "capacity_rating": frame[rating_col].astype(str).str.strip()
            if rating_col
            else "unknown",
            "year": numeric(frame[year_col]) if year_col else np.nan,
        }
    )
    out["code"] = assign_codes(out["country"], state)
    out = out[out["code"].isin(codes) & out["capacity_mw"].gt(0)].copy()
    out["source_file"] = str(solar_path)
    return out[["code", "country", "year", "capacity_mw", "capacity_rating", "source_file"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build annual geolocated wind/solar capacity maps from GEM workbooks."
    )
    parser.add_argument("--wind-xlsx", default=str(DEFAULT_WIND_XLSX))
    parser.add_argument("--solar-xlsx", default=str(DEFAULT_SOLAR_XLSX))
    parser.add_argument(
        "--codes",
        nargs="+",
        choices=sorted(REGIONS),
        default=list(EUROPE_CODES),
        help="Default: the expanded European sample. Add tx explicitly if desired.",
    )
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2022, 2027)))
    parser.add_argument(
        "--unknown-start-policy",
        choices=("include", "exclude"),
        default="include",
        help="Include unknown-start operating phases in every year, or exclude them.",
    )
    parser.add_argument(
        "--solar-dc-to-ac",
        type=float,
        default=0.87,
        help="Multiplier for GEM capacities labelled MWp/dc (default: 0.87).",
    )
    parser.add_argument(
        "--no-below-threshold-wind",
        action="store_true",
        help="Do not import GEM's separate below-10-MW wind sheet.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument(
        "--compat-year",
        type=int,
        help=(
            "Also copy this year's maps to data/capacity_weights_static_legacy for the legacy "
            "static-weight builder. Omitted by default to prevent accidental look-ahead."
        ),
    )
    parser.add_argument("--compat-dir", default=str(DEFAULT_COMPAT_DIR))
    return parser.parse_args()


def existing_path(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} workbook not found: {path}")
    return path


def main() -> None:
    args = parse_args()
    if not 0 < args.solar_dc_to_ac <= 1.5:
        raise ValueError("--solar-dc-to-ac must be greater than 0 and at most 1.5")
    years = sorted(set(args.years))
    if not years:
        raise ValueError("At least one year is required")
    if args.compat_year is not None and args.compat_year not in years:
        raise ValueError("--compat-year must also be present in --years")

    wind_path = existing_path(args.wind_xlsx, "Wind")
    solar_path = existing_path(args.solar_xlsx, "Solar")
    codes = set(args.codes)
    out_dir = Path(args.out_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    records = []
    diagnostics: list[dict] = []
    wind_sheets: Iterable[str] = (
        (WIND_SHEETS[0],)
        if args.no_below_threshold_wind
        else WIND_SHEETS
    )
    for sheet in wind_sheets:
        normalized, sheet_diagnostics = normalize_phases(
            read_sheet(wind_path, sheet),
            technology="wind",
            source_sheet=sheet,
            source_path=wind_path,
            codes=codes,
            solar_dc_to_ac=args.solar_dc_to_ac,
        )
        records.append(normalized)
        diagnostics.extend(sheet_diagnostics)
    solar, solar_diagnostics = normalize_phases(
        read_sheet(solar_path, SOLAR_SHEET),
        technology="solar",
        source_sheet=SOLAR_SHEET,
        source_path=solar_path,
        codes=codes,
        solar_dc_to_ac=args.solar_dc_to_ac,
    )
    records.append(solar)
    diagnostics.extend(solar_diagnostics)

    all_records = pd.concat(records, ignore_index=True)
    duplicate_count = int(all_records.duplicated(["technology", "record_key"]).sum())
    all_records = (
        all_records.drop_duplicates(["technology", "record_key"], keep="first")
        .sort_values(["code", "technology", "project_name", "phase_name"])
        .reset_index(drop=True)
    )

    audit_columns = [
        "code",
        "country",
        "state_province",
        "technology",
        "project_name",
        "phase_name",
        "status",
        "start_year",
        "retired_year",
        "latitude",
        "longitude",
        "location_accuracy",
        "capacity_mw_reported",
        "capacity_rating",
        "capacity_mw_weight",
        "solar_dc_converted",
        "unknown_capacity_rating",
        "unknown_start_year",
        "gem_location_id",
        "gem_phase_id",
        "date_last_researched",
        "source_sheet",
        "source_file",
        "wiki_url",
    ]
    audit_path = results_dir / "gem_capacity_records.csv"
    all_records[audit_columns].to_csv(audit_path, index=False)

    diagnostics_frame = pd.DataFrame(diagnostics)
    diagnostics_frame["duplicates_removed_after_sheet_merge"] = duplicate_count
    diagnostics_path = results_dir / "gem_import_diagnostics.csv"
    diagnostics_frame.to_csv(diagnostics_path, index=False)

    coverage = []
    written = 0
    for year in years:
        year_dir = out_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        for code in args.codes:
            region_records = all_records[all_records["code"].eq(code)]
            active = active_in_year(region_records, year, args.unknown_start_policy)
            capacity_map = build_map(active)
            capacity_map.to_csv(year_dir / f"{code}.csv", index=False)
            coverage.append(
                coverage_row(
                    active,
                    code=code,
                    year=year,
                    unknown_start_policy=args.unknown_start_policy,
                    solar_dc_to_ac=args.solar_dc_to_ac,
                )
            )
            written += 1

    coverage_frame = pd.DataFrame(coverage)
    distributed = distributed_solar_reference(solar_path, codes)
    if not distributed.empty:
        distributed_lookup = (
            distributed.sort_values("year")
            .drop_duplicates("code", keep="last")
            .set_index("code")["capacity_mw"]
        )
        coverage_frame["distributed_solar_reference_mw_latest"] = (
            coverage_frame["code"].map(distributed_lookup)
        )
    else:
        coverage_frame["distributed_solar_reference_mw_latest"] = np.nan
    coverage_path = results_dir / "gem_capacity_coverage.csv"
    coverage_frame.to_csv(coverage_path, index=False)

    distributed_path = results_dir / "gem_distributed_solar_reference.csv"
    distributed.to_csv(distributed_path, index=False)

    if args.compat_year is not None:
        compat_dir = Path(args.compat_dir).expanduser().resolve()
        compat_dir.mkdir(parents=True, exist_ok=True)
        for code in args.codes:
            shutil.copyfile(
                out_dir / str(args.compat_year) / f"{code}.csv",
                compat_dir / f"{code}.csv",
            )

    latest = coverage_frame[coverage_frame["year"].eq(max(years))]
    display = latest[
        ["code", "wind_weight_mw", "solar_weight_mw", "wind_points", "solar_points"]
    ].copy()
    print(display.to_string(index=False, float_format=lambda value: f"{value:,.1f}"))
    print(f"\nWrote {written} annual capacity maps to {out_dir}")
    print(f"Wrote coverage report to {coverage_path}")
    print(f"Wrote normalized audit trail to {audit_path}")
    print(f"Wrote import diagnostics to {diagnostics_path}")
    print(f"Wrote distributed-solar reference to {distributed_path}")
    if args.compat_year is not None:
        print(f"Wrote legacy-compatible {args.compat_year} snapshot to {args.compat_dir}")
    print(
        "Distributed solar is not geolocated and was not inserted into spatial weights. "
        "Review gem_capacity_coverage.csv before running the weather ladder."
    )


if __name__ == "__main__":
    main()
