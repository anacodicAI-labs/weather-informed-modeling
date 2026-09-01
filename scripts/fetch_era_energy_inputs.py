#!/usr/bin/env python3
"""Fetch hourly Energy-Charts target inputs for the COVID-era analysis.

Only energy/target columns are written. The spatial-resolution evaluator joins
these files to independently staged ERA5 weather, so downloading a second
flat-city weather product would be redundant.
"""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from country_registry import COUNTRIES, EUROPE_CODES


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "data" / "energy_targets_source_by_era"
DEFAULT_AUDIT = ROOT / "results" / "covid_era_spatial_resolution" / "input_coverage.csv"
API_URL = "https://api.energy-charts.info/public_power"
ERAS = {
    "pre": ("2016-01-01", "2019-12-31"),
    "covid": ("2020-01-01", "2021-12-31"),
    "post": ("2022-01-01", "2026-04-30"),
}
MODEL_REQUIRED = ("Load", "Renewable_share_of_load")
MIN_COMPLETE_COVERAGE = 0.98


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eras", nargs="+", choices=ERAS, default=list(ERAS))
    parser.add_argument("--codes", nargs="+", choices=EUROPE_CODES, default=list(EUROPE_CODES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--chunk-days", type=int, default=90)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--request-delay", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_name(value: str) -> str:
    return value.strip().replace(" ", "_").replace("/", "_")


def covers(path: Path, start: str, end: str) -> bool:
    if not path.exists():
        return False
    try:
        timestamps = pd.read_csv(path, usecols=["timestamp"])["timestamp"]
        timestamps = (
            pd.to_datetime(timestamps, utc=True, errors="coerce")
            .dropna()
            .drop_duplicates()
        )
    except Exception:
        return False
    if timestamps.empty:
        return False

    # Energy-Charts timestamps can start one hour before the requested UTC
    # interval and end one or two hours before 23:00 UTC because the API is
    # based on European local-market intervals. Requiring one exact final
    # timestamp therefore caused essentially complete files (for example,
    # Denmark post-COVID at 99.9% coverage) to be downloaded on every run.
    expected = pd.date_range(
        start,
        f"{end} 23:00:00",
        freq="h",
        tz="UTC",
    )
    in_era = timestamps[(timestamps >= expected.min()) & (timestamps <= expected.max())]
    coverage = in_era.nunique() / len(expected)
    reaches_start = timestamps.min() <= expected.min() + pd.Timedelta(hours=1)
    reaches_end = timestamps.max() >= expected.max() - pd.Timedelta(hours=3)
    return reaches_start and reaches_end and coverage >= MIN_COMPLETE_COVERAGE


def fetch_chunk(
    session: requests.Session,
    code: str,
    start: date,
    end: date,
    retries: int,
) -> pd.DataFrame:
    params = {"country": code, "start": start.isoformat(), "end": end.isoformat()}
    for attempt in range(1, retries + 1):
        response = session.get(API_URL, params=params, timeout=90)
        if response.status_code == 429:
            wait = 5 * attempt
            print(f"    rate limited; retrying in {wait}s", flush=True)
            time.sleep(wait)
            continue
        response.raise_for_status()
        payload = response.json()
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(payload.get("unix_seconds", []), unit="s", utc=True)
        })
        for series in payload.get("production_types", []):
            frame[clean_name(series["name"])] = series.get("data", [])
        if frame.empty:
            raise ValueError(f"Energy-Charts returned no timestamps for {code} {start}..{end}")
        return frame
    raise RuntimeError(f"Energy-Charts rate limit persisted for {code} {start}..{end}")


def read_or_fetch_chunk(
    args: argparse.Namespace,
    session: requests.Session,
    code: str,
    era: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Load a completed chunk checkpoint or fetch and save it atomically."""
    checkpoint_dir = (
        Path(args.out_dir).expanduser().resolve()
        / ".energy_download_chunks"
        / f"{code}_{era}"
    )
    checkpoint = checkpoint_dir / f"{start.isoformat()}__{end.isoformat()}.csv"

    if checkpoint.exists():
        try:
            frame = pd.read_csv(checkpoint, parse_dates=["timestamp"])
            timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            if not frame.empty and timestamps.notna().any():
                frame["timestamp"] = timestamps
                print(
                    f"[resume] {code} {era}: {start}..{end} from checkpoint",
                    flush=True,
                )
                return frame
        except Exception:
            # A truncated checkpoint can result from a killed process. Replace
            # it with a fresh, atomically written download below.
            pass

    print(f"[fetch] {code} {era}: {start}..{end}", flush=True)
    frame = fetch_chunk(session, code, start, end, args.retries)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(checkpoint)
    time.sleep(args.request_delay)
    return frame


def fix_short_load_failures(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Interpolate extreme local load collapses and preserve the share numerator."""
    if "Load" not in frame or "Renewable_share_of_load" not in frame:
        return frame, 0
    median = frame["Load"].rolling(24 * 14, center=True, min_periods=24).median()
    bad = frame["Load"] < 0.30 * median
    if not bad.any():
        return frame, 0
    frame = frame.copy()
    old_load = frame["Load"].copy()
    frame.loc[bad, "Load"] = np.nan
    frame["Load"] = frame["Load"].interpolate(method="time")
    frame.loc[bad, "Renewable_share_of_load"] *= (
        old_load.loc[bad] / frame.loc[bad, "Load"]
    )
    return frame, int(bad.sum())


def fetch_one(args: argparse.Namespace, session: requests.Session, code: str, era: str) -> dict:
    start_text, end_text = ERAS[era]
    target = Path(args.out_dir).expanduser().resolve() / f"weather_energy_merged_{code}_{era}.csv"
    if target.exists() and not args.force and covers(target, start_text, end_text):
        print(f"[skip] {code} {era}: complete {target}", flush=True)
        frame = pd.read_csv(target, parse_dates=["timestamp"]).set_index("timestamp")
        status = "existing"
        anomalies = np.nan
        write_pending = False
    elif args.dry_run:
        print(f"[fetch] {code} {era}: {start_text}..{end_text} -> {target}")
        return {"code": code, "country": COUNTRIES[code], "era": era, "status": "dry_run"}
    else:
        start, end = date.fromisoformat(start_text), date.fromisoformat(end_text)
        pieces = []
        current = start
        while current <= end:
            chunk_end = min(current + timedelta(days=args.chunk_days - 1), end)
            pieces.append(
                read_or_fetch_chunk(
                    args, session, code, era, current, chunk_end
                )
            )
            current = chunk_end + timedelta(days=1)
        frame = (
            pd.concat(pieces, ignore_index=True)
            .drop_duplicates("timestamp", keep="last")
            .set_index("timestamp")
            .sort_index()
            .apply(pd.to_numeric, errors="coerce")
            .resample("h").mean()
        )
        frame, anomalies = fix_short_load_failures(frame)
        status = "fetched"
        write_pending = True

    wind_columns = [c for c in frame if "Wind_" in c or c.startswith("Wind_")]
    missing = [
        column for column in MODEL_REQUIRED
        if column not in frame or not frame[column].notna().any()
    ]
    missing_components = []
    if "Solar" not in frame or not frame["Solar"].notna().any():
        missing_components.append("Solar")
    if not wind_columns or not any(frame[column].notna().any() for column in wind_columns):
        missing_components.append("Wind_*")
    if write_pending and not missing:
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index_label="timestamp")
        checkpoint_dir = target.parent / ".energy_download_chunks" / f"{code}_{era}"
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
    expected = len(pd.date_range(start_text, f"{end_text} 23:00:00", freq="h", tz="UTC"))
    return {
        "code": code,
        "country": COUNTRIES[code],
        "era": era,
        "status": "invalid" if missing else status,
        "rows": len(frame),
        "expected_hours": expected,
        "coverage_fraction": (
            pd.to_numeric(frame.get("Renewable_share_of_load"), errors="coerce")
            .notna().sum() / expected
            if "Renewable_share_of_load" in frame else 0.0
        ),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "load_anomalies_fixed": anomalies,
        "missing_required_series": ";".join(missing),
        "missing_mix_component_series": ";".join(missing_components),
        "path": str(target),
    }


def main() -> None:
    args = parse_args()
    if args.chunk_days < 1 or args.retries < 1 or args.request_delay < 0:
        raise ValueError("chunk-days/retries must be positive and request-delay nonnegative")
    session = requests.Session()
    rows = [
        fetch_one(args, session, code, era)
        for code in args.codes
        for era in args.eras
    ]
    audit = pd.DataFrame(rows)
    audit_path = Path(args.audit).expanduser().resolve()
    if not args.dry_run:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(audit_path, index=False)
        print(f"Wrote {audit_path}")
    print(audit.to_string(index=False))
    if not args.dry_run and audit.status.eq("invalid").any():
        failed = audit.loc[audit.status.eq("invalid"), ["code", "era"]]
        print(
            "WARNING: unavailable country-era inputs were not written and will "
            "be skipped by the orchestrator:\n" + failed.to_string(index=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
