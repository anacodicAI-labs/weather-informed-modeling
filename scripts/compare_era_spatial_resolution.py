#!/usr/bin/env python3
"""Compare spatial-resolution fidelity across pre-, during-, and post-COVID eras.

The pre/COVID inputs are produced by ``run_era_spatial_resolution.py``.  The
post-COVID reference is the project's existing ``results/post_covid_spatial_resolution``
run.  Recommendations are threshold-based and therefore auditable rather than
being hard-coded conclusions.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(ROOT / "figures" / ".cache" / "matplotlib")
)
ERA_RESULTS_ROOT = ROOT / "results" / "covid_era_spatial_resolution"
POST_RESULTS = ROOT / "results" / "post_covid_spatial_resolution"
OUT_DIR = ERA_RESULTS_ROOT / "comparison"
FIGURE_DIR = ROOT / "figures" / "era_spatial_resolution" / "comparison"
ERA_ORDER = ("pre", "covid", "post")
ERA_LABELS = {
    "pre": "Pre-COVID (2016-2019)",
    "covid": "COVID (2020-2021)",
    "post": "Post-COVID (2022-Apr 2026)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eras", nargs="+", choices=ERA_ORDER, default=list(ERA_ORDER))
    parser.add_argument("--gain-tolerance", type=float, default=0.01)
    parser.add_argument("--correlation-tolerance", type=float, default=0.05)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--figure-dir", default=str(FIGURE_DIR))
    parser.add_argument(
        "--figure-formats", nargs="+", choices=("png", "pdf"),
        default=("png", "pdf"), help="formats for the complete figure suite",
    )
    parser.add_argument(
        "--skip-figure-suite", action="store_true",
        help="write only the original four-panel comparison figure",
    )
    return parser.parse_args()


def summary_path(era: str) -> Path:
    if era == "post":
        return POST_RESULTS / "resolution_summary.csv"
    return ERA_RESULTS_ROOT / era / "resolution_summary.csv"


def model_results_path(era: str) -> Path:
    if era == "post":
        return POST_RESULTS / "model_results.csv"
    return ERA_RESULTS_ROOT / era / "model_results.csv"


def load_summaries(eras: list[str]) -> pd.DataFrame:
    frames = []
    for era in eras:
        path = summary_path(era)
        if not path.exists():
            print(f"[skip] no completed summary for {era}: {path}")
            continue
        frame = pd.read_csv(path)
        frame["era"] = era
        frame["era_label"] = ERA_LABELS[era]
        frame["source_summary"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No era resolution_summary.csv files were found")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_model_results(eras: list[str]) -> pd.DataFrame:
    """Collect the latest country-level result for every configuration."""
    frames = []
    keys = ["code", "scheme", "resolution_deg", "task", "model"]
    for era in eras:
        path = model_results_path(era)
        if not path.exists():
            print(f"[skip] no completed model results for {era}: {path}")
            continue
        frame = pd.read_csv(path)
        if "run_id" in frame:
            frame = frame.sort_values("run_id").drop_duplicates(keys, keep="last")
        else:
            frame = frame.drop_duplicates(keys, keep="last")
        frame["era"] = era
        frame["era_label"] = ERA_LABELS[era]
        frame["source_model_results"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values(["era", "scheme", "task", "model", "resolution_deg", "code"])
        .reset_index(drop=True)
    )


def friendly_resolution_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a readable table without discarding any summary configuration."""
    desired = {
        "era_label": "Era",
        "scheme": "Weighting",
        "task": "Task",
        "model": "Model",
        "resolution_deg": "Resolution (deg)",
        "relationship": "Correlation sample",
        "fidelity_regions": "Countries with gain",
        "countries": "Countries in correlation",
        "median_absolute_gain_drift": "Median absolute gain drift",
        "max_absolute_gain_drift": "Maximum absolute gain drift",
        "correlation": "Gain vs. wind-minus-solar r",
        "total_point_hours": "Total grid-point hours",
        "total_compute_s": "Spatial compute time (s)",
        "max_peak_rss_mb": "Maximum peak memory (MB)",
    }
    columns = [column for column in desired if column in frame]
    table = frame[columns].rename(columns=desired).copy()
    for column in (
        "Median absolute gain drift",
        "Maximum absolute gain drift",
        "Gain vs. wind-minus-solar r",
        "Spatial compute time (s)",
        "Maximum peak memory (MB)",
    ):
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(6)
    return table.sort_values(
        ["Era", "Weighting", "Task", "Model", "Resolution (deg)", "Correlation sample"]
    ).reset_index(drop=True)


def primary_capacity_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the compact 24-row table corresponding exactly to the figure."""
    data = frame.copy()
    if "relationship" in data:
        data = data[data["relationship"].eq("all_europe")]
    data = data[
        data["scheme"].eq("capacity")
        & (
            (data["task"].eq("classification") & data["model"].eq("RandForest"))
            | (data["task"].eq("regression") & data["model"].eq("GradientBoosting"))
        )
    ].copy()
    data["Analysis"] = np.where(
        data["task"].eq("classification"),
        "Random Forest classification",
        "Gradient Boosting regression",
    )
    table = data[
        [
            "era_label", "Analysis", "resolution_deg", "fidelity_regions",
            "countries", "median_absolute_gain_drift",
            "max_absolute_gain_drift", "correlation",
        ]
    ].rename(
        columns={
            "era_label": "Era",
            "resolution_deg": "Resolution (deg)",
            "fidelity_regions": "Countries with gain",
            "countries": "Countries in correlation",
            "median_absolute_gain_drift": "Median absolute gain drift",
            "max_absolute_gain_drift": "Maximum absolute gain drift",
            "correlation": "Gain vs. wind-minus-solar r",
        }
    )
    numeric = [
        "Median absolute gain drift",
        "Maximum absolute gain drift",
        "Gain vs. wind-minus-solar r",
    ]
    table[numeric] = table[numeric].apply(pd.to_numeric, errors="coerce").round(6)
    era_rank = {label: index for index, label in enumerate(ERA_LABELS.values())}
    return (
        table.assign(_era_order=table["Era"].map(era_rank))
        .sort_values(["Analysis", "_era_order", "Resolution (deg)"])
        .drop(columns="_era_order")
        .reset_index(drop=True)
    )


def write_markdown_table(table: pd.DataFrame, path: Path) -> None:
    """Write a dependency-free Markdown rendering of a compact table."""
    def display(value: object) -> str:
        if pd.isna(value):
            return "NA"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(map(str, table.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in table.columns) + " |"
    body = [
        "| " + " | ".join(display(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    path.write_text("\n".join([header, rule, *body]) + "\n")


def recommendations(
    frame: pd.DataFrame, gain_tolerance: float, correlation_tolerance: float
) -> pd.DataFrame:
    data = frame.copy()
    if "relationship" in data:
        data = data[data["relationship"].eq("all_europe")]
    rows = []
    keys = ["era", "scheme", "task", "model"]
    for key, group in data.groupby(keys, dropna=False):
        group = group.sort_values("resolution_deg")
        finest = group.iloc[0]
        reference_corr = finest.get("correlation", np.nan)
        eligible = group[
            group["median_absolute_gain_drift"].le(gain_tolerance)
            & (
                group["correlation"].sub(reference_corr).abs().le(correlation_tolerance)
                | (group["correlation"].isna() & pd.isna(reference_corr))
            )
        ]
        chosen = eligible.iloc[-1] if len(eligible) else finest
        rows.append(
            {
                **dict(zip(keys, key)),
                "reference_resolution_deg": float(finest["resolution_deg"]),
                "reference_correlation": reference_corr,
                "recommended_coarsest_resolution_deg": float(chosen["resolution_deg"]),
                "median_absolute_gain_drift": chosen["median_absolute_gain_drift"],
                "correlation": chosen.get("correlation", np.nan),
                "gain_tolerance": gain_tolerance,
                "correlation_tolerance": correlation_tolerance,
                "criterion": (
                    "median |gain-finest gain| <= gain_tolerance and "
                    "|correlation-finest correlation| <= correlation_tolerance"
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(frame: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    data = frame.copy()
    if "relationship" in data:
        data = data[data["relationship"].eq("all_europe")]
    data = data[data["scheme"].eq("capacity")]
    panels = [
        ("classification", "RandForest", "Random Forest classification"),
        ("regression", "GradientBoosting", "Gradient Boosting regression"),
    ]
    colors = {"pre": "#3975b7", "covid": "#d08a2e", "post": "#3b966b"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, (task, model, title) in enumerate(panels):
        subset = data[data["task"].eq(task) & data["model"].eq(model)]
        for era in ERA_ORDER:
            group = subset[subset["era"].eq(era)].sort_values("resolution_deg")
            if group.empty:
                continue
            axes[0, column].plot(
                group["resolution_deg"], group["correlation"], marker="o",
                color=colors[era], label=ERA_LABELS[era], linewidth=2,
            )
            axes[1, column].plot(
                group["resolution_deg"], group["median_absolute_gain_drift"],
                marker="o", color=colors[era], label=ERA_LABELS[era], linewidth=2,
            )
        axes[0, column].set_title(title, fontweight="bold")
        axes[0, column].set_ylabel("Gain vs. wind-minus-solar correlation")
        axes[1, column].set_ylabel("Median |gain - 0.25-degree gain|")
        axes[1, column].set_xlabel("Weather-grid resolution (degrees)")
        for row in range(2):
            axes[row, column].set_xscale("log", base=2)
            axes[row, column].set_xticks([0.25, 0.5, 1.0, 2.0], ["0.25", "0.5", "1", "2"])
            axes[row, column].grid(alpha=0.25)
    axes[0, 1].legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Spatial weather fidelity across COVID eras", fontsize=15, fontweight="bold")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.gain_tolerance < 0 or args.correlation_tolerance < 0:
        raise ValueError("tolerances must be non-negative")
    out_dir = Path(args.out_dir).expanduser().resolve()
    figure_dir = Path(args.figure_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_summaries(args.eras)
    frame.to_csv(out_dir / "resolution_summary_all_eras.csv", index=False)
    all_models = load_model_results(args.eras)
    if not all_models.empty:
        all_models.to_csv(out_dir / "model_results_all_eras.csv", index=False)
    friendly = friendly_resolution_table(frame)
    friendly.to_csv(out_dir / "resolution_results_table.csv", index=False)
    primary = primary_capacity_table(frame)
    primary.to_csv(out_dir / "capacity_weighted_primary_results_table.csv", index=False)
    write_markdown_table(
        primary, out_dir / "capacity_weighted_primary_results_table.md"
    )
    recommendation = recommendations(
        frame, args.gain_tolerance, args.correlation_tolerance
    )
    recommendation.to_csv(out_dir / "coarsest_resolution_by_era.csv", index=False)
    plot_comparison(frame, figure_dir / "era_resolution_comparison.png")
    extra_figures = []
    if not args.skip_figure_suite and not all_models.empty:
        from plot_era_spatial_resolution_results import generate_figure_suite

        extra_figures = generate_figure_suite(
            out_dir / "model_results_all_eras.csv",
            out_dir / "resolution_summary_all_eras.csv",
            figure_dir,
            tuple(args.figure_formats),
            statistics_dir=out_dir,
        )
    print(f"Wrote {out_dir / 'resolution_summary_all_eras.csv'}")
    if not all_models.empty:
        print(f"Wrote {out_dir / 'model_results_all_eras.csv'}")
    print(f"Wrote {out_dir / 'resolution_results_table.csv'}")
    print(f"Wrote {out_dir / 'capacity_weighted_primary_results_table.csv'}")
    print(f"Wrote {out_dir / 'capacity_weighted_primary_results_table.md'}")
    print(f"Wrote {out_dir / 'coarsest_resolution_by_era.csv'}")
    print(f"Wrote {figure_dir / 'era_resolution_comparison.png'}")
    for path in extra_figures:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
