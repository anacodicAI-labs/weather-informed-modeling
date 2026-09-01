#!/usr/bin/env python3
"""Generate a diagnostic figure suite for the original prediction study.

This script intentionally excludes every HPC-scheduling result.  It reads the
existing Energy-Charts/ERA5 inputs and spatial-resolution model results, then
creates figures that answer four questions:

1. What data are available, and how different are the 19 electricity systems?
2. Does weather improve held-out prediction beyond calendar features?
3. Is weather-added gain consistently associated with wind-minus-solar share?
4. How sensitive is that conclusion to spatial weighting and resolution?

The default run also refits two small diagnostics with the same chronological
80/20 protocol: an aggregate Logistic Regression calibration curve and a
Gradient Boosting time-series example for Denmark and Ireland.  Use
``--skip-refit`` to generate only figures based on existing result tables.

All paths are anchored to this file, not the current working directory.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

_SCRIPT_DIR_EARLY = Path(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_DIR_EARLY = _SCRIPT_DIR_EARLY.parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(_PROJECT_DIR_EARLY / "figures" / ".cache" / "matplotlib")
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from evaluate_spatial_weather import (
    CALENDAR_COLUMNS,
    SHARE_COL,
    WEATHER_COLUMNS,
    load_inputs,
    make_features,
)
from country_registry import COUNTRIES, EUROPE_CODES
from poster_figure_style import apply_poster_style


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
BASELINE_SUMMARY_DIR = RESULTS_DIR / "baseline_country_models" / "summary_tables"
SPATIAL_RESULTS_DIR = RESULTS_DIR / "post_covid_spatial_resolution"
POST_ENERGY_DIR = DATA_DIR / "energy_targets_by_era" / "post"
EXPANDED_BOOTSTRAP_PATH = (
    SPATIAL_RESULTS_DIR
    / "block_bootstrap"
    / "block_bootstrap_country_metrics.csv"
)
FIGURE_DIR = PROJECT_DIR / "figures" / "original_study_diagnostics"
ADDON_FIGURE_DIR = PROJECT_DIR / "figures" / "research_paper_figure_addons"
ADDON_FIGURE_STEMS = {
    "01_data_coverage",
    "02_wind_solar_balance",
    "03_renewable_share_distributions",
    "04_monthly_renewable_share_heatmap",
    "05_weather_target_correlations",
    "06_calendar_vs_weather_model_scores",
    "07_gain_vs_wind_minus_solar",
    "08_country_model_gain_heatmap",
    "09_capacity_weighting_advantage",
    "10_resolution_correlation_stability",
    "11_resolution_gain_drift",
    "12_compute_fidelity_frontier",
    "13_classification_calibration",
    "14_regression_timeseries_examples",
    "15_block_bootstrap_gain_intervals",
    "16_leave_one_country_out_correlations",
}
OUTPUT_DIR = RESULTS_DIR / "diagnostics" / "original_study"

MODEL_ORDER = ["LogReg", "RandForest", "GradientBoosting", "LightGBM"]
MODEL_LABELS = {
    "LogReg": "Logistic Regression (AUC)",
    "RandForest": "Random Forest (AUC)",
    "GradientBoosting": "Gradient Boosting (R2)",
    "LightGBM": "LightGBM (R2)",
}
MODEL_SHORT = {
    "LogReg": "LR AUC gain",
    "RandForest": "RF AUC gain",
    "GradientBoosting": "GB R2 gain",
    "LightGBM": "LGBM R2 gain",
}
MODEL_NAMES = {
    "LogReg": "Logistic Regression",
    "RandForest": "Random Forest",
    "GradientBoosting": "Gradient Boosting",
    "LightGBM": "LightGBM",
}

CALENDAR_COLOR = "#6B7280"
WEATHER_COLOR = "#087E8B"
CAPACITY_COLOR = "#087E8B"
UNIFORM_COLOR = "#E07A2D"
NEGATIVE_COLOR = "#B33A3A"
GRID_COLOR = "#D5D9DE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
        help="Output formats for each figure (default: png pdf).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--skip-refit",
        action="store_true",
        help="Skip calibration and time-series diagnostics that refit models.",
    )
    parser.add_argument(
        "--example-days",
        type=int,
        default=14,
        help="Length of held-out time-series examples (default: 14 days).",
    )
    parser.add_argument(
        "--only-capacity-weighting",
        action="store_true",
        help="Generate only the compact capacity-versus-uniform figure.",
    )
    return parser.parse_args()


def setup_style() -> None:
    apply_poster_style()
    mpl.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str, formats: tuple[str, ...], dpi: int) -> list[str]:
    target_dirs = [FIGURE_DIR]
    if stem in ADDON_FIGURE_STEMS:
        target_dirs.append(ADDON_FIGURE_DIR)

    paths: list[str] = []
    for target_dir in target_dirs:
        target_dir.mkdir(parents=True, exist_ok=True)
        for extension in formats:
            path = target_dir / f"{stem}.{extension}"
            fig.savefig(path, dpi=dpi if extension == "png" else None, bbox_inches="tight")
            paths.append(str(path))
    plt.close(fig)
    print(f"[figure] {stem}: {', '.join(paths)}")
    return paths


def load_energy_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict] = []
    mix_rows: list[dict] = []
    for code in EUROPE_CODES:
        country = COUNTRIES[code]
        path = POST_ENERGY_DIR / f"weather_energy_merged_{code}.csv"
        if not path.exists():
            warnings.warn(f"Missing merged data for {country}: {path}")
            continue
        frame = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        needed = [SHARE_COL, "Load"]
        missing = [column for column in needed if column not in frame]
        if missing:
            warnings.warn(f"Skipping {country}; missing {missing}")
            continue
        frames[code] = frame

        usable = frame[[SHARE_COL, "Load"]].dropna()
        expected = int((frame.index.max() - frame.index.min()) / pd.Timedelta(hours=1)) + 1
        audit_rows.append(
            {
                "code": code,
                "country": country,
                "start": frame.index.min(),
                "end": frame.index.max(),
                "rows": len(frame),
                "usable_target_rows": len(usable),
                "expected_hourly_rows": expected,
                "coverage_pct": 100 * len(frame) / expected if expected else np.nan,
                "target_missing_pct": 100 * frame[SHARE_COL].isna().mean(),
                "share_above_50_pct": 100 * (usable[SHARE_COL] > 50).mean(),
                "share_median_pct": usable[SHARE_COL].median(),
                "share_p95_pct": usable[SHARE_COL].quantile(0.95),
            }
        )

        load = pd.to_numeric(frame["Load"], errors="coerce")
        wind_columns = [
            column
            for column in frame.columns
            if "wind" in column.lower() and column not in WEATHER_COLUMNS
        ]
        solar_columns = [column for column in frame.columns if column.lower() == "solar"]
        valid_load = load.notna() & (load > 0)
        denominator = load.loc[valid_load].sum()
        wind_total = (
            frame.loc[valid_load, wind_columns]
            .apply(pd.to_numeric, errors="coerce")
            .sum(axis=1, min_count=1)
            .sum(min_count=1)
            if wind_columns
            else np.nan
        )
        solar_total = (
            frame.loc[valid_load, solar_columns]
            .apply(pd.to_numeric, errors="coerce")
            .sum(axis=1, min_count=1)
            .sum(min_count=1)
            if solar_columns
            else np.nan
        )
        wind_share = (
            100 * wind_total / denominator
            if denominator and np.isfinite(wind_total)
            else np.nan
        )
        solar_share = (
            100 * solar_total / denominator
            if denominator and np.isfinite(solar_total)
            else np.nan
        )
        mix_rows.append(
            {
                "code": code,
                "country": country,
                "wind_share_pct": wind_share,
                "solar_share_pct": solar_share,
                "wind_minus_solar": wind_share - solar_share,
                "mix_complete": bool(
                    np.isfinite(wind_share) and np.isfinite(solar_share)
                ),
            }
        )

    audit = pd.DataFrame(audit_rows).sort_values("country")
    mix = pd.DataFrame(mix_rows).sort_values("country")
    if len(frames) < 2:
        raise RuntimeError("Too few country data files were available to make figures.")
    return frames, audit, mix


def load_canonical_model_results() -> pd.DataFrame:
    latest_path = SPATIAL_RESULTS_DIR / "model_results_latest.csv"
    path = (
        latest_path
        if latest_path.exists()
        else SPATIAL_RESULTS_DIR / "model_results.csv"
    )
    frame = pd.read_csv(path)
    required = {
        "code",
        "resolution_deg",
        "scheme",
        "task",
        "model",
        "calendar_score",
        "both_score",
        "gain",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing columns: {missing}")
    if "run_id" not in frame:
        frame["run_id"] = "unknown"
    keys = ["code", "resolution_deg", "scheme", "task", "model"]
    frame = frame.sort_values("run_id").drop_duplicates(keys, keep="last")
    frame["country"] = frame["code"].map(COUNTRIES)
    frame = frame[frame["country"].notna()].copy()
    expected = len(COUNTRIES) * 4 * 2 * 4
    if len(frame) != expected:
        warnings.warn(
            f"Canonical table has {len(frame)} rows; expected {expected}. "
            "Figures will use the available rows."
        )
    return frame


def add_resource_mix(results: pd.DataFrame, mix: pd.DataFrame) -> pd.DataFrame:
    columns = ["code", "wind_share_pct", "solar_share_pct", "wind_minus_solar"]
    return results.merge(mix[columns], on="code", how="left", validate="many_to_one")


def canonical_slice(results: pd.DataFrame) -> pd.DataFrame:
    return results[
        np.isclose(results["resolution_deg"], 0.25)
        & results["scheme"].eq("capacity")
    ].copy()


def figure_data_coverage(audit: pd.DataFrame, formats: tuple[str, ...], dpi: int) -> None:
    ordered = audit.sort_values("start", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13.5, 7.5), constrained_layout=True)
    for y, row in ordered.iterrows():
        ax.plot([row.start, row.end], [y, y], color=CAPACITY_COLOR, lw=7, solid_capstyle="round")
        ax.scatter([row.start, row.end], [y, y], color=CAPACITY_COLOR, s=35, zorder=3)
        ax.text(
            row.end + pd.Timedelta(days=18),
            y,
            f"{int(row.usable_target_rows):,} h  |  {row.coverage_pct:.1f}% coverage",
            va="center",
            fontsize=11,
        )
    ax.set_yticks(range(len(ordered)), ordered["country"])
    ax.set_xlabel("Available hourly observations (UTC)")
    ax.set_title("Data coverage by country")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_ylim(-0.7, len(ordered) - 0.3)
    save_figure(fig, "01_data_coverage", formats, dpi)


def figure_resource_mix(mix: pd.DataFrame, formats: tuple[str, ...], dpi: int) -> None:
    ordered = mix.sort_values("wind_minus_solar").reset_index(drop=True)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(12.5, 8.5), constrained_layout=True)
    ax.barh(y, -ordered["solar_share_pct"], color=UNIFORM_COLOR, alpha=0.88, label="Solar share of load")
    ax.barh(y, ordered["wind_share_pct"], color=CAPACITY_COLOR, alpha=0.92, label="Wind share of load")
    ax.axvline(0, color="#30343B", lw=1.2)
    for i, row in ordered.iterrows():
        label = (
            f"D={row.wind_minus_solar:+.1f}"
            if row.mix_complete
            else "solar unavailable"
        )
        ax.text(
            row.wind_share_pct + 0.7,
            i,
            label,
            va="center",
            fontsize=10,
            color="#20242A" if row.mix_complete else NEGATIVE_COLOR,
        )
    ax.set_yticks(y, ordered["country"])
    ax.set_xlabel("Share of national load (%)  |  solar shown left, wind shown right")
    ax.set_title(
        f"Observed wind-solar balance across {len(COUNTRIES)} grids "
        f"({int(mix['mix_complete'].sum())} with both components)"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    save_figure(fig, "02_wind_solar_balance", formats, dpi)


def figure_share_distributions(
    frames: dict[str, pd.DataFrame], mix: pd.DataFrame, formats: tuple[str, ...], dpi: int
) -> None:
    order = mix.sort_values("wind_minus_solar")["code"].tolist()
    values = [frames[code][SHARE_COL].dropna().clip(-20, 180).to_numpy() for code in order]
    labels = [COUNTRIES[code] for code in order]
    fig, ax = plt.subplots(figsize=(12.5, 9), constrained_layout=True)
    box = ax.boxplot(
        values,
        vert=False,
        tick_labels=labels,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#1F2933", "linewidth": 2.2},
        whiskerprops={"color": "#6B7280"},
        capprops={"color": "#6B7280"},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(CAPACITY_COLOR)
        patch.set_alpha(0.68)
    ax.axvline(50, color=NEGATIVE_COLOR, linestyle="--", lw=2, label="Classification threshold (50%)")
    ax.set_xlabel("Renewable generation relative to load (%)")
    ax.set_title("Hourly renewable-share distributions")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.legend(loc="lower right")
    save_figure(fig, "03_renewable_share_distributions", formats, dpi)


def figure_monthly_heatmap(
    frames: dict[str, pd.DataFrame], mix: pd.DataFrame, formats: tuple[str, ...], dpi: int
) -> None:
    order = mix.sort_values("wind_minus_solar", ascending=False)["code"].tolist()
    matrix = np.array(
        [frames[code][SHARE_COL].groupby(frames[code].index.month).mean().reindex(range(1, 13)) for code in order]
    )
    fig, ax = plt.subplots(figsize=(15, 8), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=np.nanpercentile(matrix, 3), vmax=np.nanpercentile(matrix, 97))
    ax.set_xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(order)), [COUNTRIES[code] for code in order])
    ax.set_xlabel("Month")
    ax.set_title("Mean renewable share by country and month")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Mean renewable generation / load (%)", fontweight="bold")
    save_figure(fig, "04_monthly_renewable_share_heatmap", formats, dpi)


def figure_weather_target_correlations(
    frames: dict[str, pd.DataFrame], mix: pd.DataFrame, formats: tuple[str, ...], dpi: int
) -> None:
    order = mix.sort_values("wind_minus_solar", ascending=False)["code"].tolist()
    rows = []
    for code in order:
        # Use the same corrected 0.25-degree capacity-weighted ERA5 inputs as
        # the headline model experiment, rather than the older city average
        # still present in weather_energy_merged_<code>.csv.
        local = load_inputs(code, canonical_weather_path(code), POST_ENERGY_DIR)[
            WEATHER_COLUMNS + [SHARE_COL]
        ].dropna()
        correlations = local.corr(method="pearson")[SHARE_COL]
        rows.append([correlations[column] for column in WEATHER_COLUMNS])
    matrix = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(10.5, 8.5), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
    labels = ["100-m wind speed", "Solar radiation", "2-m temperature"]
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(len(order)), [COUNTRIES[code] for code in order])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if abs(matrix[i, j]) > 0.55 else "#20242A"
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", color=color, fontsize=11)
    ax.set_title("Hourly weather-target correlations")
    colorbar = fig.colorbar(image, ax=ax, pad=0.025)
    colorbar.set_label("Pearson correlation with renewable share", fontweight="bold")
    save_figure(fig, "05_weather_target_correlations", formats, dpi)


def country_order(
    mix: pd.DataFrame, require_complete_mix: bool = False
) -> list[str]:
    local = mix
    if require_complete_mix:
        local = local[local["mix_complete"]]
    return local.sort_values("wind_minus_solar")["code"].tolist()


def figure_model_dumbbells(
    canonical: pd.DataFrame, mix: pd.DataFrame, formats: tuple[str, ...], dpi: int
) -> None:
    order = country_order(mix)
    fig, axes = plt.subplots(2, 2, figsize=(19, 14), sharey=True, constrained_layout=True)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        local = canonical[canonical["model"].eq(model)].set_index("code").reindex(order)
        y = np.arange(len(order))
        ax.hlines(y, local["calendar_score"], local["both_score"], color="#AEB4BB", lw=2.2)
        ax.scatter(local["calendar_score"], y, s=70, color=CALENDAR_COLOR, marker="o", label="CALENDAR", zorder=3)
        ax.scatter(local["both_score"], y, s=80, color=WEATHER_COLOR, marker="D", label="CALENDAR + WEATHER", zorder=3)
        ax.axvline(0.5 if local["task"].iloc[0] == "classification" else 0, color="#8B9199", lw=1, ls="--")
        ax.set_yticks(y, [COUNTRIES[code] for code in order])
        ax.set_xlabel("Held-out AUC" if local["task"].iloc[0] == "classification" else "Held-out R2")
        ax.set_title(MODEL_LABELS[model])
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Absolute model performance: calendar features versus added weather", fontsize=22, fontweight="bold")
    save_figure(fig, "06_calendar_vs_weather_model_scores", formats, dpi)


def figure_gain_scatter(
    canonical: pd.DataFrame, mix: pd.DataFrame, formats: tuple[str, ...], dpi: int
) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 2, figsize=(19, 12), constrained_layout=True)
    rows: list[dict] = []
    for ax, model in zip(axes.flat, MODEL_ORDER):
        local = canonical[canonical["model"].eq(model)].dropna(subset=["wind_minus_solar", "gain"]).copy()
        x = local["wind_minus_solar"].to_numpy()
        y = local["gain"].to_numpy()
        r = float(np.corrcoef(x, y)[0, 1])
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min() - 2, x.max() + 2, 200)
        ax.axhline(0, color="#8B9199", lw=1)
        ax.axvline(0, color="#8B9199", lw=1)
        ax.scatter(x, y, s=95, color=CAPACITY_COLOR, edgecolor="white", linewidth=0.9, zorder=3)
        ax.plot(x_line, slope * x_line + intercept, color="#30343B", lw=2)
        for i, row in enumerate(local.itertuples()):
            dy = 6 if i % 2 == 0 else -11
            ax.annotate(row.code.upper(), (row.wind_minus_solar, row.gain), xytext=(5, dy), textcoords="offset points", fontsize=10)
        ax.text(
            0.03,
            0.95,
            f"r = {r:.3f}  |  {len(local)} countries",
            transform=ax.transAxes,
            va="top",
            fontsize=14,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#B8BEC5"},
        )
        ax.set_xlabel("Wind share − solar share (percentage points)")
        ax.set_ylabel(
            "Improvement from adding weather (AUC)"
            if local["task"].iloc[0] == "classification"
            else r"Improvement from adding weather ($R^2$)"
        )
        ax.set_title(MODEL_NAMES[model])
        ax.grid(color=GRID_COLOR, linewidth=0.8)
        rows.append({"model": model, "countries": len(local), "correlation": r, "slope": slope, "intercept": intercept})
    fig.suptitle(
        "Weather Helps More in Wind-Heavy Grids",
        fontsize=22,
        fontweight="bold",
    )
    save_figure(fig, "07_gain_vs_wind_minus_solar", formats, dpi)
    return pd.DataFrame(rows)


def figure_gain_heatmap(
    canonical: pd.DataFrame, mix: pd.DataFrame, formats: tuple[str, ...], dpi: int
) -> None:
    order = mix.sort_values("wind_minus_solar", ascending=False)["code"].tolist()
    pivot = canonical.pivot(index="code", columns="model", values="gain").reindex(index=order, columns=MODEL_ORDER)
    matrix = pivot.to_numpy()
    bound = float(np.nanmax(np.abs(matrix)))
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    fig, ax = plt.subplots(figsize=(12.5, 9), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(4), ["LR\nAUC", "RF\nAUC", "GB\nR2", "LightGBM\nR2"])
    ax.set_yticks(range(len(order)), [COUNTRIES[code] for code in order])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if abs(matrix[i, j]) > 0.55 * bound else "#20242A"
            ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", color=color, fontsize=10)
    ax.set_title("Per-country weather-added gain at 0.25-degree capacity weighting")
    colorbar = fig.colorbar(image, ax=ax, pad=0.025)
    colorbar.set_label("BOTH score - CALENDAR score", fontweight="bold")
    save_figure(fig, "08_country_model_gain_heatmap", formats, dpi)


def figure_capacity_vs_uniform(
    results: pd.DataFrame, mix: pd.DataFrame, formats: tuple[str, ...], dpi: int
) -> pd.DataFrame:
    local = results[np.isclose(results["resolution_deg"], 0.25)].copy()
    paired = local.pivot_table(index=["code", "model", "task"], columns="scheme", values="gain").reset_index()
    paired["capacity_minus_uniform"] = paired["capacity"] - paired["uniform"]
    paired["country"] = paired["code"].map(COUNTRIES)

    # Compact poster layout: retain every country as a point, but summarize the
    # four models in two metric-specific panels instead of repeating all country
    # labels in four large lollipop plots.
    panel_specs = [
        (
            "Classification",
            ["LogReg", "RandForest"],
            ["Logistic regression", "Random forest"],
            "Prediction improvement from farm weighting (AUC)",
        ),
        (
            "Regression",
            ["GradientBoosting", "LightGBM"],
            ["Gradient boosting", "LightGBM"],
            r"Prediction improvement from farm weighting ($R^2$)",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    rng = np.random.default_rng(20260724)

    for ax, (panel_title, models, labels, xlabel) in zip(axes, panel_specs):
        plotted_values: list[np.ndarray] = []
        for yi, model in enumerate(models):
            values = (
                paired.loc[paired["model"].eq(model), "capacity_minus_uniform"]
                .dropna()
                .to_numpy(dtype=float)
            )
            plotted_values.append(values)

            # Small deterministic vertical jitter reveals overlapping countries.
            jitter = rng.uniform(-0.10, 0.10, size=len(values))
            ax.scatter(
                values,
                yi + jitter,
                s=34,
                facecolor="#C8CDD3",
                edgecolor="#6B7280",
                linewidth=0.7,
                alpha=0.90,
                zorder=2,
                label="Country" if yi == 0 else None,
            )

            mean_value = float(np.mean(values))
            bootstrap_indices = rng.integers(
                0, len(values), size=(10_000, len(values))
            )
            bootstrap_means = values[bootstrap_indices].mean(axis=1)
            ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
            ax.errorbar(
                mean_value,
                yi,
                xerr=[[mean_value - ci_low], [ci_high - mean_value]],
                fmt="D",
                markersize=8,
                markerfacecolor=CAPACITY_COLOR,
                markeredgecolor="white",
                markeredgewidth=0.9,
                ecolor=CAPACITY_COLOR,
                elinewidth=3,
                capsize=5,
                capthick=2,
                zorder=4,
                label="Average and 95% CI" if yi == 0 else None,
            )
            label_above = yi == 0
            ax.annotate(
                f"{mean_value:+.3f}",
                (mean_value, yi),
                xytext=(0, 24 if label_above else -26),
                textcoords="offset points",
                ha="center",
                va="bottom" if label_above else "top",
                color="#20242A",
                fontsize=12,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.20",
                    "facecolor": "white",
                    "edgecolor": "#8B9199",
                    "linewidth": 0.8,
                    "alpha": 0.96,
                },
                zorder=5,
            )

        all_values = np.concatenate(plotted_values)
        span = max(float(np.ptp(all_values)), 0.01)
        ax.set_xlim(float(all_values.min() - 0.10 * span), float(all_values.max() + 0.18 * span))
        ax.axvline(0, color="#3B4046", lw=1.5, zorder=1)
        ax.set_yticks(np.arange(len(models)), labels)
        ax.set_ylim(len(models) - 0.55, -0.55)
        ax.set_xlabel(f"{xlabel}\n(positive = better than equal averaging)")
        ax.set_title(panel_title, loc="left")
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        fontsize=10,
    )
    fig.suptitle(
        "Farm-Weighted Weather Predicts Better",
        fontsize=18,
        fontweight="bold",
    )
    save_figure(fig, "09_capacity_weighting_advantage", formats, dpi)
    return paired


def compute_resolution_correlations(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (scheme, resolution, task, model), group in results.groupby(
        ["scheme", "resolution_deg", "task", "model"]
    ):
        valid = group.dropna(subset=["wind_minus_solar", "gain"])
        rows.append(
            {
                "scheme": scheme,
                "resolution_deg": resolution,
                "task": task,
                "model": model,
                "countries": len(valid),
                "correlation": valid["wind_minus_solar"].corr(valid["gain"]),
            }
        )
    return pd.DataFrame(rows)


def figure_resolution_correlations(
    correlations: pd.DataFrame, formats: tuple[str, ...], dpi: int
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), sharex=True, sharey=True, constrained_layout=True)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        local = correlations[correlations["model"].eq(model)]
        for scheme, color, marker, linestyle, display_label in (
            ("capacity", CAPACITY_COLOR, "o", "-", "Farm-weighted"),
            ("uniform", UNIFORM_COLOR, "s", "--", "All locations equal"),
        ):
            series = local[local["scheme"].eq(scheme)].sort_values("resolution_deg")
            ax.plot(
                series["resolution_deg"],
                series["correlation"],
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=8,
                label=scheme.capitalize(),
            )
        ax.set_ylim(-0.05, 1.02)
        ax.set_xticks([0.25, 0.5, 1.0, 2.0])
        ax.set_xlabel("ERA5 grid resolution (degrees)")
        ax.set_ylabel("Correlation: gain vs. wind-minus-solar")
        ax.set_title(MODEL_LABELS[model])
        ax.grid(color=GRID_COLOR, linewidth=0.8)
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("Stability of the cross-country relationship across spatial representations", fontsize=21, fontweight="bold")
    save_figure(fig, "10_resolution_correlation_stability", formats, dpi)


def compute_gain_drift(results: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        results[np.isclose(results["resolution_deg"], 0.25)]
        .set_index(["code", "scheme", "task", "model"])["gain"]
        .rename("gain_0p25")
    )
    drift = results.join(baseline, on=["code", "scheme", "task", "model"])
    drift["gain_drift"] = drift["gain"] - drift["gain_0p25"]
    drift["absolute_gain_drift"] = drift["gain_drift"].abs()
    return drift


def figure_gain_drift(
    drift: pd.DataFrame, formats: tuple[str, ...], dpi: int
) -> None:
    resolutions = [0.5, 1.0, 2.0]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True, constrained_layout=True)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        local = drift[drift["model"].eq(model)]
        for scheme, color, offset in (("capacity", CAPACITY_COLOR, -0.16), ("uniform", UNIFORM_COLOR, 0.16)):
            arrays = [
                local[local["scheme"].eq(scheme) & np.isclose(local["resolution_deg"], resolution)][
                    "absolute_gain_drift"
                ].dropna().to_numpy()
                for resolution in resolutions
            ]
            positions = np.arange(len(resolutions)) + offset
            box = ax.boxplot(
                arrays,
                positions=positions,
                widths=0.27,
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "#20242A", "linewidth": 2},
            )
            for patch in box["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.65)
            for element in ("whiskers", "caps"):
                for artist in box[element]:
                    artist.set_color(color)
        ax.set_xticks(range(3), ["0.5", "1.0", "2.0"])
        ax.set_xlabel("Coarser ERA5 resolution (degrees)")
        ax.set_ylabel("Absolute gain change from 0.25 degrees")
        ax.set_title(MODEL_LABELS[model])
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    legend = [
        Line2D([0], [0], color=CAPACITY_COLOR, lw=8, alpha=0.65, label="Capacity weighted"),
        Line2D([0], [0], color=UNIFORM_COLOR, lw=8, alpha=0.65, label="Uniform"),
    ]
    axes[0, 0].legend(handles=legend, loc="upper left")
    fig.suptitle("Per-country predictive-gain drift as weather grids become coarser", fontsize=21, fontweight="bold")
    save_figure(fig, "11_resolution_gain_drift", formats, dpi)


def figure_compute_fidelity(
    drift: pd.DataFrame, formats: tuple[str, ...], dpi: int
) -> pd.DataFrame:
    metric_path = SPATIAL_RESULTS_DIR / "resolution_summary.csv"
    metrics = pd.read_csv(metric_path)
    metrics = metrics[metrics["relationship"].eq("all_europe")].copy()
    metrics["relative_point_hours_pct"] = metrics.groupby("scheme")["total_point_hours"].transform(
        lambda series: 100 * series / series.max()
    )
    empirical = (
        drift.groupby(["scheme", "resolution_deg", "task", "model"], as_index=False)["absolute_gain_drift"]
        .median()
        .rename(columns={"absolute_gain_drift": "median_absolute_gain_drift_recomputed"})
    )
    metrics = metrics.merge(empirical, on=["scheme", "resolution_deg", "task", "model"], how="left")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        local = metrics[metrics["model"].eq(model)]
        for scheme, color, marker, linestyle, display_label in (
            ("capacity", CAPACITY_COLOR, "o", "-", "Farm-weighted"),
            ("uniform", UNIFORM_COLOR, "s", "--", "All locations equal"),
        ):
            series = local[local["scheme"].eq(scheme)].sort_values("relative_point_hours_pct")
            ax.plot(
                series["relative_point_hours_pct"],
                series["median_absolute_gain_drift_recomputed"],
                marker=marker,
                color=color,
                linestyle=linestyle,
                markersize=8,
                label=display_label,
            )
            for row in series.itertuples():
                label_above = scheme == "capacity"
                ax.annotate(
                    f"{row.resolution_deg:g}° grid",
                    (
                        row.relative_point_hours_pct,
                        row.median_absolute_gain_drift_recomputed,
                    ),
                    xytext=(4, 7 if label_above else -11),
                    textcoords="offset points",
                    va="bottom" if label_above else "top",
                    fontsize=9,
                )
        ax.set_xscale("log")
        ax.set_xlabel(
            "Weather data processed (% of detailed 0.25° grid; log scale)"
        )
        task = local["task"].iloc[0]
        ax.set_ylabel(
            "Typical change from 0.25° result (AUC)"
            if task == "classification"
            else r"Typical change from 0.25° result ($R^2$)"
        )
        ax.set_title(MODEL_NAMES[model])
        ax.grid(color=GRID_COLOR, linewidth=0.8)
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(
        "Less Detailed Weather Data Still Predicts Well",
        fontsize=22,
        fontweight="bold",
    )
    save_figure(fig, "12_compute_fidelity_frontier", formats, dpi)
    return metrics


def canonical_weather_path(code: str) -> Path:
    return DATA_DIR / "country_weather_post_covid" / f"weather_era5_{code}_capacity_0p25deg.csv"


def fit_logistic_diagnostics(code: str) -> dict:
    data = make_features(
        load_inputs(code, canonical_weather_path(code), POST_ENERGY_DIR)
    )
    split = int(0.8 * len(data))
    train, test = data.iloc[:split], data.iloc[split:]
    y_train = (train[SHARE_COL] > 50).astype(int).to_numpy()
    y_test = (test[SHARE_COL] > 50).astype(int).to_numpy()
    output = {"code": code, "y_true": y_test}
    for label, columns in (("CALENDAR", CALENDAR_COLUMNS), ("BOTH", WEATHER_COLUMNS + CALENDAR_COLUMNS)):
        scaler = StandardScaler().fit(train[columns])
        model = LogisticRegression(max_iter=2000, random_state=42)
        model.fit(scaler.transform(train[columns]), y_train)
        probability = model.predict_proba(scaler.transform(test[columns]))[:, 1]
        output[label] = probability
        output[f"brier_{label.lower()}"] = brier_score_loss(y_test, probability)
    return output


def figure_calibration(formats: tuple[str, ...], dpi: int) -> pd.DataFrame:
    diagnostics = [fit_logistic_diagnostics(code) for code in COUNTRIES if canonical_weather_path(code).exists()]
    rows = []
    for item in diagnostics:
        rows.append(
            {
                "code": item["code"],
                "country": COUNTRIES[item["code"]],
                "calendar_brier": item["brier_calendar"],
                "both_brier": item["brier_both"],
                "brier_reduction": item["brier_calendar"] - item["brier_both"],
            }
        )
    scores = pd.DataFrame(rows)
    y = np.concatenate([item["y_true"] for item in diagnostics])
    p_calendar = np.concatenate([item["CALENDAR"] for item in diagnostics])
    p_both = np.concatenate([item["BOTH"] for item in diagnostics])

    fig, (ax_curve, ax_country) = plt.subplots(1, 2, figsize=(18, 7.5), constrained_layout=True)
    ax_curve.plot([0, 1], [0, 1], color="#555B62", ls="--", lw=1.5, label="Perfect calibration")
    for probabilities, label, color, marker in (
        (p_calendar, "CALENDAR", CALENDAR_COLOR, "o"),
        (p_both, "CALENDAR + WEATHER", WEATHER_COLOR, "D"),
    ):
        observed, predicted = calibration_curve(y, probabilities, n_bins=12, strategy="quantile")
        ax_curve.plot(predicted, observed, color=color, marker=marker, markersize=7, label=label)
    ax_curve.set_xlim(0, 1)
    ax_curve.set_ylim(0, 1)
    ax_curve.set_xlabel("Predicted probability of renewable share > 50%")
    ax_curve.set_ylabel("Observed frequency")
    ax_curve.set_title("Pooled held-out reliability curve")
    ax_curve.grid(color=GRID_COLOR, linewidth=0.8)
    ax_curve.legend(loc="upper left")

    ordered = scores.sort_values("brier_reduction")
    colors = np.where(ordered["brier_reduction"] >= 0, WEATHER_COLOR, NEGATIVE_COLOR)
    ax_country.barh(ordered["country"], ordered["brier_reduction"], color=colors, alpha=0.85)
    ax_country.axvline(0, color="#3B4046", lw=1.2)
    ax_country.set_xlabel("Brier-score reduction (positive is better)")
    ax_country.set_title("Probability-error improvement by country")
    ax_country.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    fig.suptitle("Does weather improve probability quality, not only AUC?", fontsize=22, fontweight="bold")
    save_figure(fig, "13_classification_calibration", formats, dpi)
    return scores


def fit_regression_example(code: str, days: int) -> tuple[pd.DataFrame, dict]:
    data = make_features(
        load_inputs(code, canonical_weather_path(code), POST_ENERGY_DIR)
    )
    split = int(0.8 * len(data))
    train, test = data.iloc[:split], data.iloc[split:]
    predictions = pd.DataFrame(index=test.index)
    predictions["Actual"] = test[SHARE_COL]
    for label, columns in (("CALENDAR", CALENDAR_COLUMNS), ("CALENDAR + WEATHER", WEATHER_COLUMNS + CALENDAR_COLUMNS)):
        model = HistGradientBoostingRegressor(random_state=42)
        model.fit(train[columns], train[SHARE_COL])
        predictions[label] = model.predict(test[columns])

    hours = max(24, int(days * 24))
    rolling_std = predictions["Actual"].rolling(hours, min_periods=hours).std()
    end = rolling_std.idxmax() if rolling_std.notna().any() else predictions.index[min(hours - 1, len(predictions) - 1)]
    start = end - pd.Timedelta(hours=hours - 1)
    window = predictions.loc[start:end]
    metrics = {
        "code": code,
        "country": COUNTRIES[code],
        "window_start": str(window.index.min()),
        "window_end": str(window.index.max()),
        "calendar_mae": mean_absolute_error(predictions["Actual"], predictions["CALENDAR"]),
        "both_mae": mean_absolute_error(predictions["Actual"], predictions["CALENDAR + WEATHER"]),
    }
    metrics["mae_reduction"] = metrics["calendar_mae"] - metrics["both_mae"]
    return window, metrics


def figure_timeseries(days: int, formats: tuple[str, ...], dpi: int) -> pd.DataFrame:
    examples = []
    metrics = []
    for code in ("dk", "ie"):
        window, row = fit_regression_example(code, days)
        examples.append((code, window))
        metrics.append(row)
    fig, axes = plt.subplots(2, 1, figsize=(19, 10), sharex=False, constrained_layout=True)
    for ax, (code, window), row in zip(axes, examples, metrics):
        ax.plot(window.index, window["Actual"], color="#1F2933", lw=2.0, label="Actual")
        ax.plot(window.index, window["CALENDAR"], color=CALENDAR_COLOR, lw=1.7, alpha=0.92, label="CALENDAR")
        ax.plot(window.index, window["CALENDAR + WEATHER"], color=WEATHER_COLOR, lw=2.0, alpha=0.92, label="CALENDAR + WEATHER")
        ax.axhline(50, color=NEGATIVE_COLOR, lw=1.2, ls="--")
        ax.set_ylabel("Renewable generation / load (%)")
        ax.set_title(
            f"{COUNTRIES[code]}: held-out {days}-day high-variability window | "
            f"full-test MAE reduction = {row['mae_reduction']:.2f} points"
        )
        ax.grid(color=GRID_COLOR, linewidth=0.7)
    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Test-set timestamp (UTC)")
    fig.suptitle("What added weather changes in the hourly regression forecast", fontsize=22, fontweight="bold")
    save_figure(fig, "14_regression_timeseries_examples", formats, dpi)
    return pd.DataFrame(metrics)


def figure_bootstrap_intervals(mix: pd.DataFrame, formats: tuple[str, ...], dpi: int) -> None:
    if EXPANDED_BOOTSTRAP_PATH.exists():
        frame = pd.read_csv(EXPANDED_BOOTSTRAP_PATH)
        primary = {
            "LogReg": "auc_gain",
            "RandForest": "auc_gain",
            "GradientBoosting": "r2_gain",
            "LightGBM": "r2_gain",
        }
        fig, axes = plt.subplots(
            2, 2, figsize=(18, 14), constrained_layout=True
        )
        for ax, model in zip(axes.flat, MODEL_ORDER):
            local = (
                frame[
                    frame["model"].eq(model)
                    & frame["metric"].eq(primary[model])
                ]
                .sort_values("point_estimate")
                .reset_index(drop=True)
            )
            y = np.arange(len(local))
            ax.axvline(0, color="#3B4046", lw=1.1)
            for yi, row in enumerate(local.itertuples()):
                color = WEATHER_COLOR if row.positive_ci else UNIFORM_COLOR
                ax.hlines(yi, row.ci_low, row.ci_high, color=color, lw=1.4)
                ax.vlines(
                    [row.ci_low, row.ci_high],
                    yi - 0.12,
                    yi + 0.12,
                    color=color,
                    lw=1.1,
                )
                ax.scatter(row.point_estimate, yi, color=color, s=42, zorder=3)
            ax.set_yticks(y, local["country"])
            ax.set_xlabel(
                "Improvement from adding weather (AUC)"
                if model in {"LogReg", "RandForest"}
                else r"Improvement from adding weather ($R^2$)"
            )
            ax.set_title(
                f"{MODEL_NAMES[model]} — clear improvement in "
                f"{int(local['positive_ci'].sum())} of {len(local)} countries"
            )
            ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        confidence = 100 * float(frame["confidence"].iloc[0])
        fig.suptitle(
            "Weather Improvements After Resampling Whole Weeks\n"
            f"({confidence:g}% confidence intervals)",
            fontsize=21,
            fontweight="bold",
        )
        save_figure(fig, "15_block_bootstrap_gain_intervals", formats, dpi)
        return

    path = BASELINE_SUMMARY_DIR / "bootstrap_gains.csv"
    if not path.exists():
        print(f"[skip] bootstrap intervals: {path} does not exist")
        return
    frame = pd.read_csv(path)
    name_to_code = {name: code for code, name in COUNTRIES.items()}
    frame["code"] = frame["country"].map(name_to_code)
    frame = frame.merge(mix[["code", "wind_minus_solar"]], on="code", how="left")
    order = frame.groupby("country")["wind_minus_solar"].first().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5), sharey=True, constrained_layout=True)
    for ax, model in zip(axes, ["LogReg", "RandForest"]):
        local = frame[frame["model"].eq(model)].set_index("country").reindex(order).reset_index()
        y = np.arange(len(local))
        xerr = np.vstack([local["gain_mean"] - local["gain_lo95"], local["gain_hi95"] - local["gain_mean"]])
        colors = np.where(local["excludes_zero"], WEATHER_COLOR, UNIFORM_COLOR)
        ax.axvline(0, color="#3B4046", lw=1.2)
        for yi, row, color, low, high in zip(y, local.itertuples(), colors, xerr[0], xerr[1]):
            ax.errorbar(row.gain_mean, yi, xerr=np.array([[low], [high]]), fmt="o", color=color, capsize=4, markersize=7)
        ax.set_yticks(y, local["country"])
        ax.set_xlabel(
            "Improvement from adding weather (AUC)\n"
            "95% confidence interval"
        )
        ax.set_title(MODEL_LABELS[model])
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    fig.suptitle(
        "Weather Improvements After Resampling Whole Weeks\n"
        "(95% confidence intervals)",
        fontsize=21,
        fontweight="bold",
    )
    save_figure(fig, "15_block_bootstrap_gain_intervals", formats, dpi)


def compute_leave_one_country_out(canonical: pd.DataFrame) -> pd.DataFrame:
    """Measure whether the headline correlation depends on one country."""
    rows: list[dict] = []
    for model in MODEL_ORDER:
        local = canonical[canonical["model"].eq(model)].dropna(
            subset=["wind_minus_solar", "gain"]
        )
        full_r = local["wind_minus_solar"].corr(local["gain"])
        for omitted_code in local["code"]:
            kept = local[local["code"].ne(omitted_code)]
            leave_one_out_r = kept["wind_minus_solar"].corr(kept["gain"])
            rows.append(
                {
                    "model": model,
                    "omitted_code": omitted_code,
                    "omitted_country": COUNTRIES[omitted_code],
                    "full_correlation": full_r,
                    "leave_one_out_correlation": leave_one_out_r,
                    "change_from_full": leave_one_out_r - full_r,
                }
            )
    return pd.DataFrame(rows)


def figure_leave_one_out(
    influence: pd.DataFrame, mix: pd.DataFrame, formats: tuple[str, ...], dpi: int
) -> None:
    valid_codes = set(influence["omitted_code"])
    order = [
        code
        for code in country_order(mix, require_complete_mix=True)
        if code in valid_codes
    ]
    fig, axes = plt.subplots(
        2, 2, figsize=(18, 13), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, model in zip(axes.flat, MODEL_ORDER):
        local = (
            influence[influence["model"].eq(model)]
            .set_index("omitted_code")
            .reindex(order)
        )
        y = np.arange(len(order))
        full_r = local["full_correlation"].iloc[0]
        ax.axvline(
            full_r,
            color="#3B4046",
            lw=2,
            ls="--",
            label=f"All {len(local)}: r={full_r:.3f}",
        )
        ax.scatter(
            local["leave_one_out_correlation"],
            y,
            color=CAPACITY_COLOR,
            s=75,
            zorder=3,
        )
        for yi, value in zip(y, local["leave_one_out_correlation"]):
            ax.plot([full_r, value], [yi, yi], color="#AEB4BB", lw=1.7, zorder=1)
        ax.set_yticks(y, [COUNTRIES[code] for code in order])
        ax.set_xlabel("Correlation after omitting the named country")
        ax.set_title(MODEL_LABELS[model])
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        ax.legend(loc="lower left")
    all_values = influence["leave_one_out_correlation"].dropna()
    axes[0, 0].set_xlim(
        max(-1, all_values.min() - 0.04), min(1, all_values.max() + 0.04)
    )
    fig.suptitle(
        "Leave-one-country-out influence on the headline relationship",
        fontsize=22,
        fontweight="bold",
    )
    save_figure(fig, "16_leave_one_country_out_correlations", formats, dpi)


def write_manifest(formats: tuple[str, ...], refit_enabled: bool) -> None:
    descriptions = [
        ("01_data_coverage", "Do countries have comparable hourly coverage and missingness?"),
        ("02_wind_solar_balance", "Does the sample span solar-dominant and wind-dominant grids?"),
        ("03_renewable_share_distributions", "How different are target distributions and threshold prevalence?"),
        ("04_monthly_renewable_share_heatmap", "How much seasonality can calendar features already capture?"),
        ("05_weather_target_correlations", "Which meteorological variables track renewable share within each grid?"),
        ("06_calendar_vs_weather_model_scores", "What are the absolute held-out scores, not only score gains?"),
        ("07_gain_vs_wind_minus_solar", "Does weather-added gain rise with wind-minus-solar share across models?"),
        ("08_country_model_gain_heatmap", "Is the headline pattern broad or driven by a few countries?"),
        ("09_capacity_weighting_advantage", "Does capacity weighting improve gain over uniform averaging?"),
        ("10_resolution_correlation_stability", "Does the cross-country relationship survive coarser grids?"),
        ("11_resolution_gain_drift", "Which countries/models lose predictive value as resolution decreases?"),
        ("12_compute_fidelity_frontier", "How much processing volume can be removed before gain changes?"),
        ("13_classification_calibration", "Does weather improve calibrated probability error as well as AUC?"),
        ("14_regression_timeseries_examples", "What do calendar-only and weather-aware predictions look like hourly?"),
        ("15_block_bootstrap_gain_intervals", "Which country gains are temporally robust under weekly resampling?"),
        ("16_leave_one_country_out_correlations", "Does the headline correlation survive removal of any one country?"),
    ]
    rows = []
    for stem, question in descriptions:
        if not refit_enabled and stem in {"13_classification_calibration", "14_regression_timeseries_examples"}:
            continue
        for extension in formats:
            rows.append(
                {
                    "figure": stem,
                    "format": extension,
                    "path": str(FIGURE_DIR / f"{stem}.{extension}"),
                    "research_paper_addon_path": str(
                        ADDON_FIGURE_DIR / f"{stem}.{extension}"
                    ),
                    "question": question,
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUTPUT_DIR / "figure_manifest.csv", index=False)
    ADDON_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(ADDON_FIGURE_DIR / "figure_manifest.csv", index=False)


def main() -> None:
    args = parse_args()
    formats = tuple(args.formats)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    frames, audit, mix = load_energy_data()
    results = add_resource_mix(load_canonical_model_results(), mix)
    canonical = canonical_slice(results)

    audit.to_csv(OUTPUT_DIR / "country_data_audit.csv", index=False)
    mix.to_csv(OUTPUT_DIR / "country_resource_mix.csv", index=False)
    canonical.to_csv(OUTPUT_DIR / "canonical_0p25_capacity_model_results.csv", index=False)

    if args.only_capacity_weighting:
        weighting = figure_capacity_vs_uniform(results, mix, formats, args.dpi)
        weighting.to_csv(
            OUTPUT_DIR / "capacity_minus_uniform_by_country.csv", index=False
        )
        print(f"\nDone. Compact capacity-weighting figure: {FIGURE_DIR}")
        return

    figure_data_coverage(audit, formats, args.dpi)
    figure_resource_mix(mix, formats, args.dpi)
    figure_share_distributions(frames, mix, formats, args.dpi)
    figure_monthly_heatmap(frames, mix, formats, args.dpi)
    figure_weather_target_correlations(frames, mix, formats, args.dpi)
    figure_model_dumbbells(canonical, mix, formats, args.dpi)
    correlations = figure_gain_scatter(canonical, mix, formats, args.dpi)
    figure_gain_heatmap(canonical, mix, formats, args.dpi)
    weighting = figure_capacity_vs_uniform(results, mix, formats, args.dpi)
    resolution_correlations = compute_resolution_correlations(results)
    figure_resolution_correlations(resolution_correlations, formats, args.dpi)
    drift = compute_gain_drift(results)
    figure_gain_drift(drift, formats, args.dpi)
    frontier = figure_compute_fidelity(drift, formats, args.dpi)
    figure_bootstrap_intervals(mix, formats, args.dpi)
    influence = compute_leave_one_country_out(canonical)
    figure_leave_one_out(influence, mix, formats, args.dpi)

    correlations.to_csv(OUTPUT_DIR / "headline_correlations.csv", index=False)
    weighting.to_csv(OUTPUT_DIR / "capacity_minus_uniform_by_country.csv", index=False)
    resolution_correlations.to_csv(OUTPUT_DIR / "resolution_correlations.csv", index=False)
    drift.to_csv(OUTPUT_DIR / "resolution_gain_drift.csv", index=False)
    frontier.to_csv(OUTPUT_DIR / "compute_fidelity_frontier.csv", index=False)
    influence.to_csv(OUTPUT_DIR / "leave_one_country_out_correlations.csv", index=False)

    diagnostic_summary: dict[str, object] = {"refit_enabled": not args.skip_refit}
    if not args.skip_refit:
        calibration = figure_calibration(formats, args.dpi)
        timeseries = figure_timeseries(args.example_days, formats, args.dpi)
        calibration.to_csv(OUTPUT_DIR / "classification_calibration_scores.csv", index=False)
        timeseries.to_csv(OUTPUT_DIR / "regression_timeseries_example_metrics.csv", index=False)
        diagnostic_summary["mean_brier_reduction"] = float(calibration["brier_reduction"].mean())
        diagnostic_summary["countries_with_positive_brier_reduction"] = int((calibration["brier_reduction"] > 0).sum())

    write_manifest(formats, refit_enabled=not args.skip_refit)
    metadata = {
        "countries": list(COUNTRIES),
        "country_count": len(COUNTRIES),
        "wind_minus_solar_country_count": int(mix["mix_complete"].sum()),
        "canonical_weather": "0.25-degree capacity weighted ERA5",
        "split": "chronological 80/20",
        "figure_count": len(list(FIGURE_DIR.glob("*.png"))) if "png" in formats else len(list(FIGURE_DIR.glob("*.pdf"))),
        **diagnostic_summary,
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nDone. Figures: {FIGURE_DIR}")
    print(f"Derived tables and manifest: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
