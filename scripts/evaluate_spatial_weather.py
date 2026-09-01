#!/usr/bin/env python3
"""Evaluate one spatial-weather CSV with the project's fixed ML protocol."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "figures" / ".cache" / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
SHARE_COL = "Renewable_share_of_load"
WEATHER_COLUMNS = ["wind_speed_100m", "shortwave_radiation", "temperature_2m"]
CALENDAR_COLUMNS = ["hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend"]


def utc_index(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def load_inputs(code: str, weather_csv: Path, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    merged_path = Path(data_dir) / f"weather_energy_merged_{code}.csv"
    if not merged_path.exists():
        raise FileNotFoundError(f"Missing baseline energy/target file: {merged_path}")
    merged = pd.read_csv(merged_path, index_col="timestamp", parse_dates=["timestamp"])
    weather = pd.read_csv(weather_csv, index_col="timestamp", parse_dates=["timestamp"])
    merged, weather = utc_index(merged), utc_index(weather)
    missing = [column for column in WEATHER_COLUMNS if column not in weather]
    if missing:
        raise KeyError(f"{weather_csv} missing weather columns {missing}")
    base = merged.drop(columns=[column for column in WEATHER_COLUMNS if column in merged])
    joined = base.join(weather[WEATHER_COLUMNS], how="inner")
    if SHARE_COL not in joined:
        raise KeyError(f"{merged_path} missing target {SHARE_COL}")
    return joined


def make_features(merged: pd.DataFrame) -> pd.DataFrame:
    frame = merged[WEATHER_COLUMNS + [SHARE_COL]].copy()
    frame["hour_sin"] = np.sin(2 * np.pi * frame.index.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame.index.hour / 24)
    frame["month_sin"] = np.sin(2 * np.pi * frame.index.month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame.index.month / 12)
    frame["is_weekend"] = (frame.index.dayofweek >= 5).astype(int)
    return frame.dropna().sort_index()


def evaluate(
    code: str,
    weather_csv: Path,
    split_fraction: float = 0.8,
    data_dir: Path = DATA_DIR,
) -> list[dict]:
    started = time.perf_counter()
    data = make_features(load_inputs(code, weather_csv, data_dir))
    split = int(len(data) * split_fraction)
    train, test = data.iloc[:split], data.iloc[split:]
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"Insufficient joined rows: {len(data)}")
    feature_sets = {
        "CALENDAR": CALENDAR_COLUMNS,
        "BOTH": WEATHER_COLUMNS + CALENDAR_COLUMNS,
    }
    rows: list[dict] = []

    y_train_class = (train[SHARE_COL] > 50).astype(int).to_numpy()
    y_test_class = (test[SHARE_COL] > 50).astype(int).to_numpy()
    classification_scores: dict[tuple[str, str], float] = {}
    for model_name in ("LogReg", "RandForest"):
        for feature_set, columns in feature_sets.items():
            x_train = train[columns].to_numpy()
            x_test = test[columns].to_numpy()
            if model_name == "LogReg":
                scaler = StandardScaler().fit(x_train)
                x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
                model = LogisticRegression(max_iter=2000, random_state=42)
            else:
                model = RandomForestClassifier(
                    n_estimators=300, random_state=42, n_jobs=1
                )
            if len(np.unique(y_train_class)) < 2 or len(np.unique(y_test_class)) < 2:
                score = np.nan
            else:
                model.fit(x_train, y_train_class)
                score = roc_auc_score(y_test_class, model.predict_proba(x_test)[:, 1])
            classification_scores[(model_name, feature_set)] = score

    y_train_reg = train[SHARE_COL].to_numpy()
    y_test_reg = test[SHARE_COL].to_numpy()
    regression_scores: dict[tuple[str, str], tuple[float, float]] = {}
    model_factories = {
        "GradientBoosting": lambda: HistGradientBoostingRegressor(random_state=42),
    }
    try:
        from lightgbm import LGBMRegressor

        model_factories["LightGBM"] = lambda: LGBMRegressor(
            n_estimators=200, random_state=42, n_jobs=1, verbose=-1
        )
    except ImportError:
        pass
    for model_name, factory in model_factories.items():
        for feature_set, columns in feature_sets.items():
            model = factory()
            model.fit(train[columns], y_train_reg)
            prediction = model.predict(test[columns])
            regression_scores[(model_name, feature_set)] = (
                r2_score(y_test_reg, prediction),
                mean_absolute_error(y_test_reg, prediction),
            )

    common = {
        "code": code,
        "weather_csv": str(weather_csv),
        "rows": len(data),
        "train_rows": len(train),
        "test_rows": len(test),
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
    }
    for model_name in ("LogReg", "RandForest"):
        calendar = classification_scores[(model_name, "CALENDAR")]
        both = classification_scores[(model_name, "BOTH")]
        rows.append(
            {
                **common,
                "task": "classification",
                "model": model_name,
                "metric": "auc",
                "calendar_score": calendar,
                "both_score": both,
                "gain": both - calendar,
            }
        )
    for model_name in model_factories:
        cal_r2, cal_mae = regression_scores[(model_name, "CALENDAR")]
        both_r2, both_mae = regression_scores[(model_name, "BOTH")]
        rows.append(
            {
                **common,
                "task": "regression",
                "model": model_name,
                "metric": "r2",
                "calendar_score": cal_r2,
                "both_score": both_r2,
                "gain": both_r2 - cal_r2,
                "calendar_mae": cal_mae,
                "both_mae": both_mae,
                "mae_reduction": cal_mae - both_mae,
            }
        )
    elapsed = time.perf_counter() - started
    for row in rows:
        row["evaluation_wall_s"] = elapsed
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--weather-csv", required=True)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = evaluate(
        args.code,
        Path(args.weather_csv).expanduser().resolve(),
        args.split,
        Path(args.data_dir).expanduser().resolve(),
    )
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".json":
            output.write_text(json.dumps(rows, indent=2))
        else:
            pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
