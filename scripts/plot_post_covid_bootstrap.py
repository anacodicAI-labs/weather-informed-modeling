#!/usr/bin/env python3
"""Plot four-model weekly block-bootstrap intervals."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = SCRIPT_DIR.parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_DIR / "figures" / ".cache" / "matplotlib")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from poster_figure_style import apply_poster_style


DEFAULT_INPUT = (
    PROJECT_DIR
    / "results"
    / "post_covid_spatial_resolution"
    / "block_bootstrap"
    / "block_bootstrap_country_metrics.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "figures" / "original_study_diagnostics"
MODELS = ("LogReg", "RandForest", "GradientBoosting", "LightGBM")
MODEL_LABELS = {
    "LogReg": "Logistic Regression",
    "RandForest": "Random Forest",
    "GradientBoosting": "Gradient Boosting",
    "LightGBM": "LightGBM",
}
METRICS = {
    "score": {
        "classification": "auc_gain",
        "regression": "r2_gain",
        "label": {
            "classification": "Improvement from adding weather (AUC)",
            "regression": r"Improvement from adding weather ($R^2$)",
        },
    },
    "error": {
        "classification": "brier_reduction",
        "regression": "mae_reduction",
        "label": {
            "classification": "Reduction in probability error (Brier score)",
            "regression": "Reduction in average error (MAE, share points)",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--metric-family", choices=METRICS, default="score")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(Path(args.input).expanduser().resolve())
    required = {
        "country", "task", "model", "metric", "point_estimate",
        "ci_low", "ci_high", "positive_ci",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Bootstrap input is missing columns {missing}")

    apply_poster_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 14),
        constrained_layout=True,
    )
    selection = METRICS[args.metric_family]
    for ax, model in zip(axes.flat, MODELS):
        task = "classification" if model in {"LogReg", "RandForest"} else "regression"
        metric = selection[task]
        local = (
            frame[frame["model"].eq(model) & frame["metric"].eq(metric)]
            .sort_values("point_estimate")
            .reset_index(drop=True)
        )
        if local.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No {model} results", ha="center", va="center")
            continue
        y = np.arange(len(local))
        colors = np.where(local["positive_ci"], "#087E8B", "#E07A2D")
        ax.axvline(0, color="#3B4046", linewidth=1.1)
        for yi, row, color in zip(y, local.itertuples(), colors):
            ax.hlines(yi, row.ci_low, row.ci_high, color=color, linewidth=1.4)
            ax.vlines(
                [row.ci_low, row.ci_high],
                yi - 0.12,
                yi + 0.12,
                color=color,
                linewidth=1.1,
            )
            ax.scatter(
                row.point_estimate,
                yi,
                color=color,
                s=42,
                zorder=3,
            )
        ax.set_yticks(y, local["country"])
        ax.set_xlabel(selection["label"][task])
        ax.set_title(
            f"{MODEL_LABELS[model]} — clear gain in "
            f"{int(local['positive_ci'].sum())} of {len(local)} countries"
        )
        ax.grid(axis="x", color="#D5D9DE", linewidth=0.7)

    fig.suptitle(
        "Weather Improvements After Resampling Whole Weeks\n"
        "(95% confidence intervals)",
        fontsize=22,
        fontweight="bold",
        y=1.015,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"15_block_bootstrap_all_models_{args.metric_family}"
    for extension in args.formats:
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(
            path,
            dpi=args.dpi if extension == "png" else None,
            bbox_inches="tight",
        )
        print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
