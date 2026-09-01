#!/usr/bin/env python3
"""Summarize the spatial sweep and create poster-ready payoff figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results" / "post_covid_spatial_resolution"
FIGURE_DIR = PROJECT_DIR / "figures" / "spatial_resolution"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "figures" / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


def latest_science_rows(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["code", "scheme", "resolution_deg", "task", "model"]
    return frame.sort_values("run_id").drop_duplicates(keys, keep="last")


def latest_build_rows(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["code", "scheme", "resolution_deg"]
    return frame.sort_values("run_id").drop_duplicates(keys, keep="last")


def write_canonical_tables(
    results_dir: Path,
    models: pd.DataFrame,
    builds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write unambiguous latest-result tables while preserving run history."""
    latest_models = latest_science_rows(models).reset_index(drop=True)
    latest_builds = latest_build_rows(builds).reset_index(drop=True)
    latest_models.to_csv(results_dir / "model_results_latest.csv", index=False)
    latest_builds.to_csv(
        results_dir / "weather_build_metrics_latest.csv", index=False
    )

    audit_path = results_dir / "capacity_mapping_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        keys = ["code", "scheme", "resolution_deg", "year"]
        latest_audit = (
            audit.sort_values("run_id", kind="stable")
            .drop_duplicates(keys, keep="last")
            .reset_index(drop=True)
        )
        latest_audit.to_csv(
            results_dir / "capacity_mapping_audit_latest.csv", index=False
        )
    return latest_models, latest_builds


def correlation_summary(model_results: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    data = latest_science_rows(model_results).merge(
        shares[["code", "actual_wind_share_pct", "actual_solar_share_pct"]],
        on="code",
        how="left",
    )
    data["wind_minus_solar"] = (
        data.actual_wind_share_pct - data.actual_solar_share_pct
    )
    # ERCOT has no row in the European country summary and is external validation,
    # not a point in the European cross-country fit.
    data = data.dropna(subset=["gain", "wind_minus_solar"])
    rows = []
    group_columns = ["scheme", "resolution_deg", "task", "model"]
    for key, group in data.groupby(group_columns):
        scheme, resolution, task, model = key
        for exclusion, subset in (
            ("all_europe", group),
            ("without_lithuania", group[group.code != "lt"]),
        ):
            correlation = (
                subset[["wind_minus_solar", "gain"]].corr().iloc[0, 1]
                if len(subset) >= 3
                else np.nan
            )
            rows.append(
                {
                    "scheme": scheme,
                    "resolution_deg": resolution,
                    "task": task,
                    "model": model,
                    "relationship": exclusion,
                    "countries": len(subset),
                    "correlation": correlation,
                }
            )
    return pd.DataFrame(rows)


def fidelity_summary(model_results: pd.DataFrame) -> pd.DataFrame:
    data = latest_science_rows(model_results).copy()
    keys = ["code", "scheme", "task", "model"]
    finest = (
        data.sort_values("resolution_deg")
        .groupby(keys, as_index=False)
        .first()[keys + ["resolution_deg", "gain"]]
        .rename(columns={"resolution_deg": "reference_resolution_deg", "gain": "reference_gain"})
    )
    data = data.merge(finest, on=keys, how="left")
    data["absolute_gain_drift"] = (data.gain - data.reference_gain).abs()
    data["relative_gain_drift"] = data.absolute_gain_drift / data.reference_gain.abs().clip(lower=1e-6)
    return (
        data.groupby(["scheme", "resolution_deg", "task", "model"], as_index=False)
        .agg(
            fidelity_regions=("code", "nunique"),
            median_absolute_gain_drift=("absolute_gain_drift", "median"),
            max_absolute_gain_drift=("absolute_gain_drift", "max"),
            median_relative_gain_drift=("relative_gain_drift", "median"),
        )
    )


def cost_summary(build_metrics: pd.DataFrame) -> pd.DataFrame:
    # Point-hours are the implementation-independent spatial-work measure.
    # Wall time is also retained, but it is not used as the fidelity-frontier
    # x-axis unless every resolution reads a resolution-native cache.
    per_region = latest_build_rows(build_metrics)
    if "source_kind" not in per_region:
        per_region["source_kind"] = "native_era5"
    return (
        per_region.groupby(["scheme", "resolution_deg"], as_index=False)
        .agg(
            cost_regions=("code", "nunique"),
            total_wall_s=("total_s", "sum"),
            total_compute_s=("compute_s", "sum"),
            total_processed_points=("processed_points", "sum"),
            total_point_hours=("point_hours", "sum"),
            max_peak_rss_mb=("peak_rss_mb", "max"),
            resolution_cached_regions=(
                "source_kind", lambda x: int((x == "resolution_cache").sum())
            ),
        )
    )


def data_quality_summary(build_metrics: pd.DataFrame) -> pd.DataFrame:
    """One auditable row per country at the finest capacity-weighted level."""
    data = latest_build_rows(build_metrics)
    data = data[data.scheme.eq("capacity")]
    finest = data.groupby("code").resolution_deg.transform("min")
    data = data[data.resolution_deg.eq(finest)].copy()
    desired = [
        "code", "resolution_deg", "system_scope",
        "wind_low_coverage", "solar_low_coverage",
        "wind_capacity_absent", "solar_capacity_absent",
        "wind_weight_fallback", "solar_weight_fallback",
        "wind_outside_grid_mw_max", "solar_outside_grid_mw_max",
        "wind_out_of_system_scope_mw_max", "solar_out_of_system_scope_mw_max",
        "source_kind", "run_id",
    ]
    for column in desired:
        if column not in data:
            data[column] = np.nan
    return data[desired].sort_values("code").reset_index(drop=True)


def grid_cell_summary(build_metrics: pd.DataFrame) -> pd.DataFrame:
    """Expose domain size and effective capacity-cell counts by country."""
    data = latest_build_rows(build_metrics).copy()
    desired = [
        "code", "scheme", "resolution_deg", "native_resolution_deg",
        "coarsen_lat_factor", "coarsen_lon_factor", "native_points",
        "processed_points", "wind_occupied_cells", "solar_occupied_cells",
        "combined_occupied_cells", "wind_effective_cells",
        "solar_effective_cells", "combined_effective_cells", "point_hours",
    ]
    for column in desired:
        if column not in data:
            data[column] = np.nan
    return data[desired].sort_values(["scheme", "code", "resolution_deg"]).reset_index(drop=True)


def _correlation(x: pd.Series, y: pd.Series) -> float:
    return float(x.corr(y)) if len(x) >= 3 and x.nunique() > 1 and y.nunique() > 1 else np.nan


def _partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Pearson partial correlation of x and y after linear adjustment for z."""
    design = np.column_stack([np.ones(len(z)), z])
    x_resid = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.std(x_resid) == 0 or np.std(y_resid) == 0:
        return np.nan
    return float(np.corrcoef(x_resid, y_resid)[0, 1])


def country_size_sensitivity(
    model_results: pd.DataFrame,
    shares: pd.DataFrame,
    build_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Test whether weather gain is explained by staged country-domain size."""
    models = latest_science_rows(model_results).merge(
        shares[["code", "actual_wind_share_pct", "actual_solar_share_pct"]],
        on="code",
        how="left",
    )
    models["wind_minus_solar"] = (
        models.actual_wind_share_pct - models.actual_solar_share_pct
    )
    sizes = latest_build_rows(build_metrics)[
        ["code", "scheme", "resolution_deg", "native_points", "processed_points"]
    ]
    data = models.merge(sizes, on=["code", "scheme", "resolution_deg"], how="left")
    data = data.dropna(subset=["gain", "wind_minus_solar", "native_points"])
    rows = []
    for key, group in data.groupby(["scheme", "resolution_deg", "task", "model"]):
        scheme, resolution, task, model = key
        group = group[group.native_points > 0]
        log_size = np.log(group.native_points.to_numpy(float))
        rows.append({
            "scheme": scheme,
            "resolution_deg": resolution,
            "task": task,
            "model": model,
            "countries": len(group),
            "gain_vs_wind_minus_solar_r": _correlation(group.gain, group.wind_minus_solar),
            "gain_vs_log_native_points_r": _correlation(group.gain, pd.Series(log_size, index=group.index)),
            "wind_minus_solar_vs_log_native_points_r": _correlation(
                group.wind_minus_solar, pd.Series(log_size, index=group.index)
            ),
            "partial_gain_vs_wind_minus_solar_controlling_log_native_points_r": (
                _partial_correlation(
                    group.wind_minus_solar.to_numpy(float),
                    group.gain.to_numpy(float),
                    log_size,
                ) if len(group) >= 4 else np.nan
            ),
        })
    return pd.DataFrame(rows)


def plot_compute_fidelity(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    frontier = summary[summary.relationship.eq("all_europe")]
    for (scheme, task, model), group in frontier.groupby(["scheme", "task", "model"]):
        group = group.sort_values("resolution_deg")
        label = f"{scheme}: {task} {model}"
        axes[0].plot(
            group.total_point_hours,
            group.median_absolute_gain_drift,
            marker="o",
            linewidth=1.8,
            label=label,
        )
        for row in group.itertuples():
            axes[0].annotate(
                f"{row.resolution_deg:g}°",
                (row.total_compute_s, row.median_absolute_gain_drift),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Spatial work (grid-point hours; log scale)")
    axes[0].set_ylabel("Median |gain − finest-grid gain|")
    axes[0].set_title("Spatial-work–fidelity frontier")
    axes[0].grid(alpha=0.25)

    corr = summary[summary.relationship.eq("all_europe")]
    for (scheme, task, model), group in corr.groupby(["scheme", "task", "model"]):
        group = group.sort_values("resolution_deg")
        axes[1].plot(
            group.resolution_deg,
            group.correlation,
            marker="o",
            linewidth=1.8,
            label=f"{scheme}: {task} {model}",
        )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Weather-grid resolution (degrees; coarser →)")
    axes[1].set_ylabel("Gain vs. wind-minus-solar correlation")
    axes[1].set_title("Does the scientific conclusion stabilize?")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scaling(scaling: pd.DataFrame, output: Path) -> pd.DataFrame:
    scaling = scaling.copy()
    if "backend" in scaling:
        # Runs created before the explicit backend field used ProcessPoolExecutor.
        scaling["backend"] = scaling["backend"].fillna("process")
    else:
        scaling["backend"] = "process"
    if "method_version" not in scaling:
        # The corrected run predating this field used the restricted thread
        # backend; older process rows used native-input edge trimming.
        scaling["method_version"] = np.where(
            scaling.backend.eq("thread"), "padded_cache_v2", "legacy_trim_native"
        )
    else:
        scaling["method_version"] = scaling["method_version"].fillna(
            "legacy_trim_native"
        )
    if "resolution_cache_enabled" not in scaling:
        scaling["resolution_cache_enabled"] = scaling.method_version.eq(
            "padded_cache_v2"
        )
    signature = ["codes", "resolutions", "schemes", "start", "end", "evaluation_included"]
    signature.extend(["backend", "method_version", "resolution_cache_enabled"])
    groups = list(scaling.groupby(signature, dropna=False))
    if not groups:
        return pd.DataFrame()
    # Prefer the most complete scaling series; break ties in favor of the most
    # recent method so a legacy one-point run cannot replace corrected results.
    selected_key, data = max(
        groups,
        key=lambda item: (
            item[1].workers.nunique(),
            pd.to_datetime(item[1].timestamp_utc).max(),
        ),
    )
    if not isinstance(selected_key, tuple):
        selected_key = (selected_key,)
    selected_signature = dict(zip(signature, selected_key))
    # Keep a balanced repeated-measures set. An earlier one-worker baseline may
    # coexist with a later scaling matrix; using the latest common number of
    # repetitions avoids silently giving one worker count extra weight.
    repetitions = int(data.groupby("workers").size().min())
    data = (
        data.assign(_timestamp=pd.to_datetime(data.timestamp_utc))
        .sort_values("_timestamp")
        .groupby("workers", group_keys=False)
        .tail(repetitions)
    )
    data["mean_task_s"] = data.sum_task_wall_s / data.tasks
    data = (
        data.groupby("workers", as_index=False)
        .agg(
            batch_wall_s=("batch_wall_s", "median"),
            runtime_min_s=("batch_wall_s", "min"),
            runtime_max_s=("batch_wall_s", "max"),
            throughput=("aggregate_point_hours_per_s", "median"),
            worker_utilization=("worker_utilization", "median"),
            mean_task_s=("mean_task_s", "median"),
        )
        .sort_values("workers")
    )
    data["repetitions"] = repetitions
    baseline = data.iloc[0]
    data["speedup"] = baseline.batch_wall_s / data.batch_wall_s
    data["parallel_efficiency"] = data.speedup / (data.workers / baseline.workers)
    data["speedup_low"] = baseline.batch_wall_s / data.runtime_max_s
    data["speedup_high"] = baseline.batch_wall_s / data.runtime_min_s
    data["efficiency_low"] = data.speedup_low / (data.workers / baseline.workers)
    data["efficiency_high"] = data.speedup_high / (data.workers / baseline.workers)
    data["scaling_ready"] = data.workers.nunique() >= 2
    if "backend" in selected_signature:
        data["backend"] = selected_signature["backend"]
    data["method_version"] = selected_signature["method_version"]
    data["resolution_cache_enabled"] = selected_signature[
        "resolution_cache_enabled"
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if data.workers.nunique() < 2:
        worker_count = int(baseline.workers)
        worker_label = f"{worker_count} worker" + ("" if worker_count == 1 else "s")
        for axis in axes:
            axis.axis("off")
        axes[0].text(
            0.5, 0.5,
            "Strong-scaling figure withheld\nNeed at least two worker counts\nfor the same workload",
            ha="center", va="center", fontsize=14,
        )
        axes[1].text(
            0.5, 0.5,
            f"Available {selected_signature.get('backend', 'process')}-backend measurement: "
            f"{worker_label}, "
            f"{baseline.batch_wall_s:.1f} s",
            ha="center", va="center", fontsize=12,
        )
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return data
    axes[0].errorbar(
        data.workers,
        data.speedup,
        yerr=np.vstack(
            [data.speedup - data.speedup_low, data.speedup_high - data.speedup]
        ),
        fmt="o-",
        capsize=4,
        label=f"measured median (n={repetitions})",
    )
    axes[0].plot(
        data.workers,
        data.workers / baseline.workers,
        "--",
        color="gray",
        label="ideal",
    )
    axes[0].set_xlabel("Workers")
    axes[0].set_ylabel("Speedup")
    axes[0].set_title("Strong scaling")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].errorbar(
        data.workers,
        100 * data.parallel_efficiency,
        yerr=100 * np.vstack(
            [
                data.parallel_efficiency - data.efficiency_low,
                data.efficiency_high - data.parallel_efficiency,
            ]
        ),
        fmt="o-",
        capsize=4,
    )
    axes[1].set_xlabel("Workers")
    axes[1].set_ylabel("Parallel efficiency (%)")
    axes[1].set_ylim(0, max(105, 105 * data.parallel_efficiency.max()))
    axes[1].set_title("Parallel efficiency")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument(
        "--country-summary",
        default=str(
            PROJECT_DIR
            / "results"
            / "baseline_country_models"
            / "summary_tables"
            / "country_summary.csv"
        ),
    )
    parser.add_argument("--figure-dir", default=str(FIGURE_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    figure_dir = Path(args.figure_dir).expanduser().resolve()
    models = pd.read_csv(results_dir / "model_results.csv")
    builds = pd.read_csv(results_dir / "weather_build_metrics.csv")
    models, builds = write_canonical_tables(results_dir, models, builds)
    shares = pd.read_csv(Path(args.country_summary).expanduser().resolve())

    correlations = correlation_summary(models, shares)
    fidelity = fidelity_summary(models)
    costs = cost_summary(builds)
    summary = fidelity.merge(costs, on=["scheme", "resolution_deg"], how="left")
    summary = summary.merge(
        correlations,
        on=["scheme", "resolution_deg", "task", "model"],
        how="left",
        suffixes=("_fidelity", "_correlation"),
    )
    summary_path = results_dir / "resolution_summary.csv"
    summary.to_csv(summary_path, index=False)
    quality_path = results_dir / "data_quality_summary.csv"
    data_quality_summary(builds).to_csv(quality_path, index=False)
    cells_path = results_dir / "grid_cell_summary.csv"
    grid_cell_summary(builds).to_csv(cells_path, index=False)
    size_path = results_dir / "country_size_sensitivity.csv"
    country_size_sensitivity(models, shares, builds).to_csv(size_path, index=False)
    plot_compute_fidelity(summary, figure_dir / "compute_fidelity_frontier.png")

    scaling_path = results_dir / "scaling_runs.csv"
    if scaling_path.exists():
        scaling_summary = plot_scaling(
            pd.read_csv(scaling_path), figure_dir / "parallel_scaling.png"
        )
        scaling_summary.to_csv(results_dir / "scaling_summary.csv", index=False)
    print(f"Wrote {summary_path}")
    print(f"Wrote {quality_path}")
    print(f"Wrote {cells_path}")
    print(f"Wrote {size_path}")
    print(f"Wrote figures to {figure_dir}")


if __name__ == "__main__":
    main()
