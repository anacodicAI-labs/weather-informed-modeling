#!/usr/bin/env python3
"""Weekly block-bootstrap uncertainty for all post-COVID prediction models.

This script uses the canonical 0.25-degree capacity-weighted ERA5 inputs for
all 19 European systems. For each country it:

1. creates the same chronological 80/20 split as the main model pipeline;
2. fits CALENDAR and CALENDAR+WEATHER versions of each requested model once;
3. predicts every held-out hour;
4. resamples complete held-out ISO weeks (or fixed multi-day blocks) with
   replacement; and
5. recomputes weather-added skill for every bootstrap draw.

The resulting intervals quantify sampling uncertainty in held-out predictive
gain, conditional on the fitted models. They do not repeatedly refit models;
the legacy ``bootstrap_core.py`` remains available for the distinct and much
more expensive training-variability experiment.

Positive values always mean that adding weather helped:

* ``auc_gain`` = AUC(BOTH) - AUC(CALENDAR)
* ``brier_reduction`` = Brier(CALENDAR) - Brier(BOTH)
* ``r2_gain`` = R2(BOTH) - R2(CALENDAR)
* ``mae_reduction`` = MAE(CALENDAR) - MAE(BOTH)

Outputs are anchored to the project directory, never the current working
directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler

from country_registry import COUNTRIES, EUROPE_CODES
from evaluate_spatial_weather import (
    CALENDAR_COLUMNS,
    SHARE_COL,
    WEATHER_COLUMNS,
    load_inputs,
    make_features,
)


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_ENERGY_DIR = DATA_DIR / "energy_targets_by_era" / "post"
DEFAULT_WEATHER_DIR = DATA_DIR / "country_weather_post_covid"
DEFAULT_RESULTS_DIR = (
    PROJECT_DIR / "results" / "post_covid_spatial_resolution" / "block_bootstrap"
)
DEFAULT_COUNTRY_SUMMARY = (
    PROJECT_DIR / "results" / "post_covid_spatial_resolution" / "country_summary.csv"
)
DEFAULT_CANONICAL_RESULTS = (
    PROJECT_DIR
    / "results"
    / "post_covid_spatial_resolution"
    / "model_results_latest.csv"
)

CLASSIFICATION_MODELS = ("LogReg", "RandForest")
REGRESSION_MODELS = ("GradientBoosting", "LightGBM")
ALL_MODELS = CLASSIFICATION_MODELS + REGRESSION_MODELS
MODEL_TASK = {
    "LogReg": "classification",
    "RandForest": "classification",
    "GradientBoosting": "regression",
    "LightGBM": "regression",
}
MODEL_METRICS = {
    "classification": ("auc_gain", "brier_reduction"),
    "regression": ("r2_gain", "mae_reduction"),
}
PRIMARY_METRIC = {
    "classification": "auc_gain",
    "regression": "r2_gain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codes",
        nargs="+",
        choices=EUROPE_CODES,
        default=list(EUROPE_CODES),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=list(ALL_MODELS),
    )
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--backend",
        choices=("process", "thread"),
        default="process",
        help=(
            "Use process workers for normal local/SCC runs; thread is a "
            "fallback for environments that prohibit process semaphores."
        ),
    )
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument(
        "--block-mode",
        choices=("iso_week", "fixed_days"),
        default="iso_week",
        help="Resample whole ISO weeks (default) or consecutive fixed-day blocks.",
    )
    parser.add_argument(
        "--block-days",
        type=int,
        default=7,
        help="Block length for --block-mode fixed_days (default: 7).",
    )
    parser.add_argument("--energy-dir", default=str(DEFAULT_ENERGY_DIR))
    parser.add_argument("--weather-dir", default=str(DEFAULT_WEATHER_DIR))
    parser.add_argument("--country-summary", default=str(DEFAULT_COUNTRY_SUMMARY))
    parser.add_argument(
        "--canonical-results",
        default=str(DEFAULT_CANONICAL_RESULTS),
        help="Canonical score table used to verify bootstrap point estimates.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--save-draws",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the per-draw gzip CSV used for correlation intervals.",
    )
    return parser.parse_args()


def stable_seed(base_seed: int, code: str) -> int:
    digest = hashlib.sha256(code.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "little")
    return int((base_seed + offset) % (2**32 - 1))


def weather_path(weather_dir: Path, code: str) -> Path:
    return weather_dir / f"weather_era5_{code}_capacity_0p25deg.csv"


def load_country_data(
    code: str,
    energy_dir: Path,
    weather_dir: Path,
    split_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = make_features(
        load_inputs(code, weather_path(weather_dir, code), energy_dir)
    )
    split = int(len(data) * split_fraction)
    train, test = data.iloc[:split].copy(), data.iloc[split:].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"{code}: insufficient chronological rows "
            f"(train={len(train)}, test={len(test)})"
        )
    return train, test


def make_blocks(
    index: pd.DatetimeIndex,
    mode: str,
    block_days: int,
) -> tuple[list[np.ndarray], list[str]]:
    if mode == "iso_week":
        iso = index.isocalendar()
        labels = np.asarray(
            [
                f"{int(year):04d}-W{int(week):02d}"
                for year, week in zip(iso.year, iso.week)
            ],
            dtype=object,
        )
    else:
        origin = index.min().floor("D")
        labels = np.asarray(
            ((index - origin) // pd.Timedelta(days=block_days)).astype(int),
            dtype=np.int64,
        )

    blocks: list[np.ndarray] = []
    block_labels: list[str] = []
    for label in pd.unique(labels):
        positions = np.flatnonzero(labels == label).astype(np.int64)
        if len(positions):
            blocks.append(positions)
            block_labels.append(str(label))
    if len(blocks) < 2:
        raise ValueError(f"Need at least two temporal blocks; found {len(blocks)}")
    return blocks, block_labels


def classification_predictions(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    y_train = (train[SHARE_COL] > 50).astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"{model_name}: classification training target has one class")
    outputs = []
    for columns in (CALENDAR_COLUMNS, WEATHER_COLUMNS + CALENDAR_COLUMNS):
        x_train = train[columns].to_numpy()
        x_test = test[columns].to_numpy()
        if model_name == "LogReg":
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
            model = LogisticRegression(
                max_iter=2000, random_state=42
            )
        elif model_name == "RandForest":
            model = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=1,
            )
        else:
            raise ValueError(f"Unknown classification model {model_name}")
        model.fit(x_train, y_train)
        outputs.append(model.predict_proba(x_test)[:, 1])
    return outputs[0], outputs[1]


def regression_predictions(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    y_train = train[SHARE_COL].to_numpy()
    outputs = []
    for columns in (CALENDAR_COLUMNS, WEATHER_COLUMNS + CALENDAR_COLUMNS):
        if model_name == "GradientBoosting":
            model = HistGradientBoostingRegressor(random_state=42)
        elif model_name == "LightGBM":
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:
                raise RuntimeError(
                    "LightGBM was requested but is not installed"
                ) from exc
            model = LGBMRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=1,
                verbose=-1,
            )
        else:
            raise ValueError(f"Unknown regression model {model_name}")
        model.fit(train[columns], y_train)
        outputs.append(model.predict(test[columns]))
    return outputs[0], outputs[1]


def classification_gains(
    y: np.ndarray,
    calendar: np.ndarray,
    both: np.ndarray,
) -> dict[str, float]:
    auc_gain = np.nan
    if len(np.unique(y)) >= 2:
        auc_gain = float(
            roc_auc_score(y, both) - roc_auc_score(y, calendar)
        )
    brier_reduction = float(
        np.mean((y - calendar) ** 2) - np.mean((y - both) ** 2)
    )
    return {
        "auc_gain": auc_gain,
        "brier_reduction": brier_reduction,
    }


def regression_gains(
    y: np.ndarray,
    calendar: np.ndarray,
    both: np.ndarray,
) -> dict[str, float]:
    return {
        "r2_gain": float(r2_score(y, both) - r2_score(y, calendar)),
        "mae_reduction": float(
            np.mean(np.abs(y - calendar)) - np.mean(np.abs(y - both))
        ),
    }


def summarize_draws(
    code: str,
    model: str,
    task: str,
    metric: str,
    point_estimate: float,
    values: np.ndarray,
    confidence: float,
    common: dict,
) -> dict:
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError(f"{code}/{model}/{metric}: no valid bootstrap draws")
    alpha = 1.0 - confidence
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    negative_tail = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
    positive_tail = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
    p_two_sided = min(1.0, 2.0 * min(negative_tail, positive_tail))
    direction = (
        "positive"
        if low > 0
        else "negative"
        if high < 0
        else "overlaps_zero"
    )
    return {
        **common,
        "code": code,
        "country": COUNTRIES[code],
        "task": task,
        "model": model,
        "metric": metric,
        "primary_metric": metric == PRIMARY_METRIC[task],
        "point_estimate": point_estimate,
        "bootstrap_mean": float(np.mean(values)),
        "bootstrap_median": float(np.median(values)),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": confidence,
        "valid_resamples": int(len(values)),
        "positive_ci": bool(low > 0),
        "excludes_zero": bool(low > 0 or high < 0),
        "direction": direction,
        "p_two_sided": float(p_two_sided),
    }


def run_country(task: dict) -> dict:
    started = time.perf_counter()
    code = task["code"]
    try:
        train, test = load_country_data(
            code,
            Path(task["energy_dir"]),
            Path(task["weather_dir"]),
            task["split"],
        )
        blocks, block_labels = make_blocks(
            test.index,
            task["block_mode"],
            task["block_days"],
        )
        rng = np.random.default_rng(stable_seed(task["seed"], code))
        chosen_blocks = rng.integers(
            0,
            len(blocks),
            size=(task["n_resamples"], len(blocks)),
        )

        predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for model in task["models"]:
            if MODEL_TASK[model] == "classification":
                predictions[model] = classification_predictions(model, train, test)
            else:
                predictions[model] = regression_predictions(model, train, test)

        y_class = (test[SHARE_COL] > 50).astype(int).to_numpy()
        y_regression = test[SHARE_COL].to_numpy()
        point: dict[tuple[str, str], float] = {}
        draws = {
            (model, metric): np.full(task["n_resamples"], np.nan)
            for model in task["models"]
            for metric in MODEL_METRICS[MODEL_TASK[model]]
        }

        for model, (calendar, both) in predictions.items():
            if MODEL_TASK[model] == "classification":
                scores = classification_gains(y_class, calendar, both)
            else:
                scores = regression_gains(y_regression, calendar, both)
            for metric, value in scores.items():
                point[(model, metric)] = value

        for draw_id, selected in enumerate(chosen_blocks):
            positions = np.concatenate([blocks[i] for i in selected])
            for model, (calendar, both) in predictions.items():
                if MODEL_TASK[model] == "classification":
                    scores = classification_gains(
                        y_class[positions],
                        calendar[positions],
                        both[positions],
                    )
                else:
                    scores = regression_gains(
                        y_regression[positions],
                        calendar[positions],
                        both[positions],
                    )
                for metric, value in scores.items():
                    draws[(model, metric)][draw_id] = value

        common = {
            "split_fraction": task["split"],
            "train_rows": len(train),
            "test_rows": len(test),
            "test_start": test.index.min().isoformat(),
            "test_end": test.index.max().isoformat(),
            "block_mode": task["block_mode"],
            "block_days": task["block_days"] if task["block_mode"] == "fixed_days" else 7,
            "temporal_blocks": len(blocks),
            "requested_resamples": task["n_resamples"],
            "seed": stable_seed(task["seed"], code),
        }
        summaries = []
        draw_rows = []
        for (model, metric), values in draws.items():
            summaries.append(
                summarize_draws(
                    code=code,
                    model=model,
                    task=MODEL_TASK[model],
                    metric=metric,
                    point_estimate=point[(model, metric)],
                    values=values,
                    confidence=task["confidence"],
                    common=common,
                )
            )
            draw_rows.extend(
                {
                    "code": code,
                    "country": COUNTRIES[code],
                    "task": MODEL_TASK[model],
                    "model": model,
                    "metric": metric,
                    "draw": draw_id,
                    "gain": value,
                }
                for draw_id, value in enumerate(values)
                if np.isfinite(value)
            )
        return {
            "status": "ok",
            "code": code,
            "summaries": summaries,
            "draws": draw_rows,
            "wall_s": time.perf_counter() - started,
            "blocks": block_labels,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "code": code,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "wall_s": time.perf_counter() - started,
        }


def correlation_intervals(
    summaries: pd.DataFrame,
    draws: pd.DataFrame,
    country_summary_path: Path,
    confidence: float,
) -> pd.DataFrame:
    mix = pd.read_csv(country_summary_path)
    mix["wind_minus_solar"] = (
        mix["actual_wind_share_pct"] - mix["actual_solar_share_pct"]
    )
    mix = mix.dropna(subset=["wind_minus_solar"])
    d_lookup = mix.set_index("code")["wind_minus_solar"]
    alpha = 1.0 - confidence
    rows = []
    for (task, model, metric), local_summary in summaries.groupby(
        ["task", "model", "metric"]
    ):
        local_draws = draws[
            draws["task"].eq(task)
            & draws["model"].eq(model)
            & draws["metric"].eq(metric)
        ]
        for relationship, excluded in (
            ("all_europe", set()),
            ("without_lithuania", {"lt"}),
        ):
            codes = [
                code
                for code in local_summary["code"]
                if code in d_lookup.index and code not in excluded
            ]
            point = (
                local_summary.set_index("code")
                .reindex(codes)["point_estimate"]
                .astype(float)
            )
            x = d_lookup.reindex(codes).astype(float)
            point_r = float(x.corr(point)) if len(codes) >= 3 else np.nan
            if len(codes) >= 3:
                pivot = (
                    local_draws[local_draws["code"].isin(codes)]
                    .pivot_table(index="code", columns="draw", values="gain")
                    .reindex(codes)
                )
                correlations = pivot.apply(
                    lambda column: x.corr(column.astype(float)), axis=0
                ).dropna()
            else:
                correlations = pd.Series(dtype=float)
            low, high = (
                np.quantile(correlations, [alpha / 2, 1 - alpha / 2])
                if len(correlations)
                else (np.nan, np.nan)
            )
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "metric": metric,
                    "primary_metric": metric == PRIMARY_METRIC[task],
                    "relationship": relationship,
                    "countries": len(codes),
                    "point_correlation": point_r,
                    "bootstrap_mean_correlation": (
                        float(correlations.mean()) if len(correlations) else np.nan
                    ),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "confidence": confidence,
                    "valid_resamples": int(len(correlations)),
                    "positive_ci": bool(low > 0) if np.isfinite(low) else False,
                }
            )
    return pd.DataFrame(rows)


def atomic_csv(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False, **kwargs)
    os.replace(tmp, path)


def validate_point_estimates(
    summaries: pd.DataFrame,
    canonical_path: Path,
    out_dir: Path,
    split_fraction: float,
) -> dict:
    """Reconcile primary bootstrap point estimates to the main score table."""
    if not canonical_path.exists() or not np.isclose(split_fraction, 0.8):
        return {
            "performed": False,
            "reason": (
                "canonical table missing"
                if not canonical_path.exists()
                else "noncanonical split fraction"
            ),
        }
    canonical = pd.read_csv(canonical_path)
    expected = canonical[
        canonical["scheme"].eq("capacity")
        & np.isclose(canonical["resolution_deg"], 0.25)
    ][["code", "model", "gain"]].rename(columns={"gain": "canonical_gain"})
    observed = summaries[summaries["primary_metric"]][
        ["code", "model", "metric", "point_estimate"]
    ]
    comparison = observed.merge(
        expected,
        on=["code", "model"],
        how="left",
        validate="one_to_one",
    )
    comparison["absolute_difference"] = (
        comparison["point_estimate"] - comparison["canonical_gain"]
    ).abs()
    atomic_csv(
        comparison,
        out_dir / "block_bootstrap_point_validation.csv",
    )
    missing = int(comparison["canonical_gain"].isna().sum())
    maximum = float(comparison["absolute_difference"].max())
    if missing or maximum > 1e-8:
        raise RuntimeError(
            "Bootstrap point estimates did not reconcile to canonical results: "
            f"missing={missing}, max absolute difference={maximum:.3g}. "
            "See block_bootstrap_point_validation.csv."
        )
    return {
        "performed": True,
        "rows": len(comparison),
        "maximum_absolute_difference": maximum,
    }


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.n_resamples < 20:
        raise ValueError("--n-resamples must be at least 20")
    if not 0 < args.confidence < 1:
        raise ValueError("--confidence must be between 0 and 1")
    if not 0 < args.split < 1:
        raise ValueError("--split must be between 0 and 1")
    if args.block_days < 1:
        raise ValueError("--block-days must be at least 1")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "models": list(dict.fromkeys(args.models)),
        "n_resamples": args.n_resamples,
        "confidence": args.confidence,
        "seed": args.seed,
        "split": args.split,
        "block_mode": args.block_mode,
        "block_days": args.block_days,
        "energy_dir": str(Path(args.energy_dir).expanduser().resolve()),
        "weather_dir": str(Path(args.weather_dir).expanduser().resolve()),
    }
    tasks = [{**common, "code": code} for code in dict.fromkeys(args.codes)]
    started = time.perf_counter()
    results = []
    if args.workers == 1:
        for index, task in enumerate(tasks, 1):
            print(f"[{index}/{len(tasks)}] {task['code']}", flush=True)
            result = run_country(task)
            print(
                f"  {result['status']} in {result['wall_s']:.1f}s",
                flush=True,
            )
            results.append(result)
    else:
        executor = (
            ProcessPoolExecutor
            if args.backend == "process"
            else ThreadPoolExecutor
        )
        with executor(max_workers=args.workers) as pool:
            future_map = {pool.submit(run_country, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_map), 1):
                result = future.result()
                print(
                    f"[{index}/{len(tasks)}] {result['status']} "
                    f"{result['code']} in {result['wall_s']:.1f}s",
                    flush=True,
                )
                results.append(result)

    failures = [result for result in results if result["status"] != "ok"]
    timing = pd.DataFrame(
        [
            {
                "code": result["code"],
                "country": COUNTRIES[result["code"]],
                "status": result["status"],
                "wall_s": result["wall_s"],
                "error": result.get("error", ""),
            }
            for result in results
        ]
    ).sort_values("code")
    atomic_csv(timing, out_dir / "block_bootstrap_timing.csv")
    if failures:
        failure_path = out_dir / "block_bootstrap_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2) + "\n")
        for failure in failures:
            print(failure["traceback"], flush=True)
        raise SystemExit(
            f"{len(failures)} country job(s) failed; see {failure_path}"
        )

    summaries = pd.DataFrame(
        row
        for result in results
        for row in result["summaries"]
    ).sort_values(["task", "model", "metric", "code"])
    draws = pd.DataFrame(
        row
        for result in results
        for row in result["draws"]
    ).sort_values(["task", "model", "metric", "code", "draw"])
    correlations = correlation_intervals(
        summaries,
        draws,
        Path(args.country_summary).expanduser().resolve(),
        args.confidence,
    )
    point_validation = validate_point_estimates(
        summaries,
        Path(args.canonical_results).expanduser().resolve(),
        out_dir,
        args.split,
    )

    atomic_csv(
        summaries,
        out_dir / "block_bootstrap_country_metrics.csv",
    )
    atomic_csv(
        correlations,
        out_dir / "block_bootstrap_correlations.csv",
    )
    if args.save_draws:
        draws_path = out_dir / "block_bootstrap_draws.csv.gz"
        draws.to_csv(draws_path, index=False, compression="gzip")

    elapsed = time.perf_counter() - started
    metadata = {
        "method": "held-out temporal block bootstrap conditional on fitted models",
        "codes": list(dict.fromkeys(args.codes)),
        "models": list(dict.fromkeys(args.models)),
        "n_resamples": args.n_resamples,
        "confidence": args.confidence,
        "split_fraction": args.split,
        "block_mode": args.block_mode,
        "block_days": args.block_days if args.block_mode == "fixed_days" else 7,
        "workers": args.workers,
        "backend": args.backend,
        "wall_s": elapsed,
        "country_jobs": len(tasks),
        "summary_rows": len(summaries),
        "draw_rows": len(draws),
        "point_validation": point_validation,
        "output_dir": str(out_dir),
    }
    (out_dir / "block_bootstrap_run.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        f"Done in {elapsed:.1f}s. Wrote {len(summaries)} interval rows and "
        f"{len(correlations)} correlation rows to {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
