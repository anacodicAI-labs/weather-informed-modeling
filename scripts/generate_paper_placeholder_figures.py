#!/usr/bin/env python3
"""Generate Figures 1-7 requested by the original prediction paper.

Outputs are written to ``figures/paper_placeholders`` as both PNG and PDF.
The analysis uses the corrected 0.25-degree capacity-weighted model results
unless a figure explicitly examines spatial resolution or uncertainty.

Figure 1: analysis pipeline
Figure 2: study-country map colored by wind-minus-solar share
Figure 3: Denmark capacity-weighting map
Figure 4: weather-added gain versus wind-minus-solar share
Figure 5: Denmark held-out prediction time series
Figure 6: resolution and weighting sensitivity
Figure 7: weekly-block bootstrap confidence intervals

All paths are anchored to this file rather than the current working directory.
"""

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

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from generate_original_study_figures import (
    CALENDAR_COLOR,
    CAPACITY_COLOR,
    COUNTRIES,
    GRID_COLOR,
    MODEL_LABELS,
    MODEL_NAMES,
    MODEL_ORDER,
    NEGATIVE_COLOR,
    UNIFORM_COLOR,
    WEATHER_COLOR,
    add_resource_mix,
    canonical_slice,
    compute_resolution_correlations,
    fit_regression_example,
    load_canonical_model_results,
    load_energy_data,
    setup_style,
)


DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
BASELINE_SUMMARY_DIR = RESULTS_DIR / "baseline_country_models" / "summary_tables"
FIGURE_DIR = PROJECT_DIR / "figures" / "paper_placeholders"
OUTPUT_DIR = RESULTS_DIR / "manuscript" / "placeholder_data"
EXPANDED_BOOTSTRAP_PATH = (
    RESULTS_DIR
    / "post_covid_spatial_resolution"
    / "block_bootstrap"
    / "block_bootstrap_country_metrics.csv"
)

ISO3 = {
    "at": "AUT",
    "be": "BEL",
    "bg": "BGR",
    "cz": "CZE",
    "de": "DEU",
    "hr": "HRV",
    "dk": "DNK",
    "es": "ESP",
    "fr": "FRA",
    "gr": "GRC",
    "ie": "IRL",
    "lv": "LVA",
    "lt": "LTU",
    "nl": "NLD",
    "pt": "PRT",
    "ro": "ROU",
    "rs": "SRB",
    "sk": "SVK",
    "si": "SVN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--example-days",
        type=int,
        default=7,
        help="Length of the Figure 5 Denmark example (default: 7 days).",
    )
    return parser.parse_args()


def save(fig: plt.Figure, stem: str, formats: tuple[str, ...], dpi: int) -> list[str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = FIGURE_DIR / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi if extension == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    print(f"[figure] {stem}: {', '.join(paths)}")
    return paths


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    fontsize: int = 13,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor="#27536A",
        linewidth=1.8,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="#17212B",
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2,
            color="#27536A",
            connectionstyle="arc3,rad=0",
        )
    )


def figure1_pipeline(formats: tuple[str, ...], dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(18, 7.5), constrained_layout=True)
    ax.set_axis_off()
    source_color = "#DCEEF5"
    feature_color = "#E7F3E8"
    model_color = "#FFF0D9"
    result_color = "#E9E3F4"

    rounded_box(ax, (0.02, 0.69), 0.18, 0.18, "Energy-Charts\nhourly generation + load", source_color)
    rounded_box(ax, (0.02, 0.41), 0.18, 0.18, "ERA5 gridded weather\nwind + radiation + temperature", source_color)
    rounded_box(ax, (0.02, 0.13), 0.18, 0.18, "Global Energy Monitor\nwind + solar facility maps", source_color)

    rounded_box(ax, (0.28, 0.39), 0.19, 0.24, "Capacity-weighted\nnational hourly features\n0.25, 0.5, 1, and 2 degrees", feature_color)
    arrow(ax, (0.20, 0.78), (0.28, 0.57))
    arrow(ax, (0.20, 0.50), (0.28, 0.51))
    arrow(ax, (0.20, 0.22), (0.28, 0.45))

    rounded_box(ax, (0.54, 0.58), 0.16, 0.17, "CALENDAR\nhour + month + weekend", model_color)
    rounded_box(ax, (0.54, 0.27), 0.16, 0.21, "CALENDAR + WEATHER\ncalendar + wind +\nradiation + temperature", model_color)
    arrow(ax, (0.47, 0.54), (0.54, 0.67))
    arrow(ax, (0.47, 0.48), (0.54, 0.37))

    rounded_box(ax, (0.73, 0.39), 0.15, 0.24, "Chronological 80/20 split\nClassification: AUC\nRegression: R2 and MAE", result_color)
    arrow(ax, (0.70, 0.67), (0.73, 0.56))
    arrow(ax, (0.70, 0.37), (0.73, 0.46))

    rounded_box(ax, (0.91, 0.39), 0.08, 0.24, "Weather-added\ngain", "#D8F1E1", fontsize=12)
    arrow(ax, (0.88, 0.51), (0.91, 0.51))

    ax.text(0.11, 0.94, "INPUT DATA", transform=ax.transAxes, ha="center", fontweight="bold", fontsize=15)
    ax.text(0.375, 0.74, "SPATIAL AGGREGATION", transform=ax.transAxes, ha="center", fontweight="bold", fontsize=15)
    ax.text(0.62, 0.86, "CONTROLLED MODEL COMPARISON", transform=ax.transAxes, ha="center", fontweight="bold", fontsize=15)
    ax.text(0.805, 0.74, "HELD-OUT EVALUATION", transform=ax.transAxes, ha="center", fontweight="bold", fontsize=15)
    save(fig, "figure1_pipeline", formats, dpi)


def natural_earth_world() -> gpd.GeoDataFrame:
    fixture = (
        Path(pyogrio.__file__).resolve().parent
        / "tests"
        / "fixtures"
        / "naturalearth_lowres"
        / "naturalearth_lowres.shp"
    )
    if not fixture.exists():
        raise FileNotFoundError(
            "Natural Earth fixture not found. Install geopandas and pyogrio, "
            "or set up a local Natural Earth country shapefile."
        )
    return gpd.read_file(fixture)


def figure2_country_map(
    mix: pd.DataFrame, formats: tuple[str, ...], dpi: int
) -> None:
    world = natural_earth_world()
    europe = world[
        world["continent"].eq("Europe")
        | world["name"].isin(["Turkey", "Cyprus"])
    ].copy()
    mix = mix.copy()
    mix["iso_a3"] = mix["code"].map(ISO3)
    europe = europe.merge(
        mix[["iso_a3", "wind_minus_solar"]], on="iso_a3", how="left"
    )
    included = europe[europe["iso_a3"].isin(set(ISO3.values()))].copy()
    study = europe[europe["wind_minus_solar"].notna()].copy()
    missing_mix = included[included["wind_minus_solar"].isna()].copy()
    bound = float(np.ceil(np.nanmax(np.abs(study["wind_minus_solar"])) / 5) * 5)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    cmap = mpl.colormaps["RdBu"]

    fig, ax = plt.subplots(figsize=(13.5, 10), constrained_layout=True)
    europe.plot(ax=ax, color="#E9ECEF", edgecolor="white", linewidth=0.6)
    study.plot(
        ax=ax,
        column="wind_minus_solar",
        cmap=cmap,
        norm=norm,
        edgecolor="#30343B",
        linewidth=0.9,
    )
    if not missing_mix.empty:
        missing_mix.plot(
            ax=ax,
            color="#F7E7B5",
            edgecolor="#30343B",
            linewidth=0.9,
            hatch="///",
        )
    for row in included.itertuples():
        point = row.geometry.representative_point()
        code = next(code for code, iso in ISO3.items() if iso == row.iso_a3)
        ax.text(
            point.x,
            point.y,
            code.upper(),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlim(-12, 32)
    ax.set_ylim(34, 60.5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Study countries colored by wind share minus solar share", pad=12)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=ax, orientation="horizontal", shrink=0.62, pad=0.02)
    colorbar.set_label("Wind share minus solar share of load (percentage points)", fontweight="bold")
    if not missing_mix.empty:
        ax.text(
            0.01,
            0.01,
            "Hatched: included in modeling, excluded from wind-minus-solar "
            "correlation because hourly solar is unavailable",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
        )
    save(fig, "figure2_country_map", formats, dpi)


def nearest_grid_capacity(
    plants: pd.DataFrame,
    value_column: str,
    lat_centers: np.ndarray,
    lon_centers: np.ndarray,
) -> np.ndarray:
    grid = np.zeros((len(lat_centers), len(lon_centers)), dtype=float)
    for row in plants[plants[value_column] > 0].itertuples():
        lat_index = int(np.abs(lat_centers - row.lat).argmin())
        lon_index = int(np.abs(lon_centers - row.lon).argmin())
        grid[lat_index, lon_index] += float(getattr(row, value_column))
    return grid


def figure3_capacity_map(formats: tuple[str, ...], dpi: int) -> None:
    plants_path = DATA_DIR / "capacity_weights_post_by_year" / "2025" / "dk.csv"
    plants = pd.read_csv(plants_path)
    # Matches the expanded Danish ERA5 staging rectangle plus 0.25-degree padding.
    south, north, west, east = 54.25, 58.05, 7.25, 15.45
    resolution = 0.25
    lat_centers = np.arange(np.ceil(south / resolution) * resolution, north + 1e-9, resolution)
    lon_centers = np.arange(np.ceil(west / resolution) * resolution, east + 1e-9, resolution)
    lat_edges = np.r_[lat_centers - resolution / 2, lat_centers[-1] + resolution / 2]
    lon_edges = np.r_[lon_centers - resolution / 2, lon_centers[-1] + resolution / 2]

    world = natural_earth_world()
    denmark = world[world["iso_a3"].eq("DNK")]
    fig, axes = plt.subplots(1, 2, figsize=(17, 9), constrained_layout=True)
    for ax, technology, value_column, cmap_name in (
        (axes[0], "Wind", "wind_mw", "Blues"),
        (axes[1], "Solar", "solar_mw", "Oranges"),
    ):
        cell_capacity = nearest_grid_capacity(
            plants, value_column, lat_centers, lon_centers
        )
        positive = cell_capacity[cell_capacity > 0]
        masked = np.ma.masked_where(cell_capacity <= 0, cell_capacity)
        norm = LogNorm(vmin=max(0.5, positive.min()), vmax=positive.max())
        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            masked,
            cmap=cmap_name,
            norm=norm,
            shading="flat",
            alpha=0.78,
        )
        denmark.plot(ax=ax, color="none", edgecolor="#20242A", linewidth=1.3)
        active = plants[plants[value_column] > 0]
        maximum = active[value_column].max()
        sizes = 13 + 95 * np.sqrt(active[value_column] / maximum)
        ax.scatter(
            active["lon"],
            active["lat"],
            s=sizes,
            facecolor="none",
            edgecolor="#20242A",
            linewidth=0.65,
            alpha=0.8,
        )
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{technology}: mapped capacity assigned to 0.25-degree ERA5 cells")
        ax.grid(color=GRID_COLOR, linewidth=0.55)
        colorbar = fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.82)
        colorbar.set_label(f"Mapped {technology.lower()} capacity per cell (MW)", fontweight="bold")
    fig.suptitle(
        "Denmark capacity-weighting example: cell color is total MW; circle size is facility MW",
        fontsize=21,
        fontweight="bold",
    )
    save(fig, "figure3_capacity_weighting_denmark", formats, dpi)


def figure4_gain_scatter(
    canonical: pd.DataFrame, formats: tuple[str, ...], dpi: int
) -> pd.DataFrame:
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 11.5), constrained_layout=True)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        local = canonical[canonical["model"].eq(model)].dropna(
            subset=["wind_minus_solar", "gain"]
        )
        x = local["wind_minus_solar"].to_numpy()
        y = local["gain"].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        line_x = np.linspace(x.min() - 2, x.max() + 2, 200)
        ax.axhline(0, color="#8B9199", linewidth=1)
        ax.axvline(0, color="#8B9199", linewidth=1)
        ax.scatter(x, y, s=90, color=CAPACITY_COLOR, edgecolor="white", linewidth=0.8, zorder=3)
        ax.plot(line_x, slope * line_x + intercept, color="#30343B", linewidth=2)
        for index, row in enumerate(local.itertuples()):
            ax.annotate(
                row.code.upper(),
                (row.wind_minus_solar, row.gain),
                xytext=(5, 6 if index % 2 == 0 else -11),
                textcoords="offset points",
                fontsize=9,
            )
        ax.text(
            0.03,
            0.95,
            f"r = {r:.3f}  |  n = {len(local)}",
            transform=ax.transAxes,
            va="top",
            fontsize=13,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#B8BEC5"},
        )
        ax.set_xlabel("Wind share minus solar share (percentage points)")
        ax.set_ylabel(
            "Weather-added AUC gain"
            if local["task"].iloc[0] == "classification"
            else "Weather-added R2 gain"
        )
        ax.set_title(MODEL_LABELS[model])
        ax.grid(color=GRID_COLOR, linewidth=0.7)
        rows.append({"model": model, "correlation": r, "slope": slope})
    fig.suptitle(
        "Weather-added predictive gain versus wind-minus-solar share",
        fontsize=21,
        fontweight="bold",
    )
    save(fig, "figure4_gain_vs_wind_minus_solar", formats, dpi)
    return pd.DataFrame(rows)


def figure5_timeseries(
    example_days: int, formats: tuple[str, ...], dpi: int
) -> pd.DataFrame:
    window, metrics = fit_regression_example("dk", example_days)
    fig, ax = plt.subplots(figsize=(17, 7), constrained_layout=True)
    ax.plot(window.index, window["Actual"], color="#20242A", linewidth=2.2, label="Actual renewable share")
    ax.plot(window.index, window["CALENDAR"], color=CALENDAR_COLOR, linewidth=1.8, label="CALENDAR")
    ax.plot(window.index, window["CALENDAR + WEATHER"], color=WEATHER_COLOR, linewidth=2, label="CALENDAR + WEATHER")
    ax.axhline(50, color=NEGATIVE_COLOR, linestyle="--", linewidth=1.4, label="50% threshold")
    ax.set_xlabel("Held-out test timestamp (UTC)")
    ax.set_ylabel("Renewable generation relative to load (%)")
    ax.set_title(
        f"Denmark held-out example: weather reduces full-test MAE by {metrics['mae_reduction']:.2f} points"
    )
    ax.grid(color=GRID_COLOR, linewidth=0.7)
    ax.legend(loc="upper right", ncol=2)
    save(fig, "figure5_denmark_prediction_timeseries", formats, dpi)
    return pd.DataFrame([metrics])


def figure6_resolution_sensitivity(
    correlations: pd.DataFrame, formats: tuple[str, ...], dpi: int
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5), sharex=True, sharey=True, constrained_layout=True)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        local = correlations[correlations["model"].eq(model)]
        for scheme, color, marker, linestyle in (
            ("capacity", CAPACITY_COLOR, "o", "-"),
            ("uniform", UNIFORM_COLOR, "s", "--"),
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
        ax.set_xticks([0.25, 0.5, 1.0, 2.0])
        ax.set_ylim(0.72, 0.94)
        ax.set_xlabel("ERA5 grid resolution (degrees)")
        ax.set_ylabel("Correlation: gain vs. wind-minus-solar")
        ax.set_title(MODEL_LABELS[model])
        ax.grid(color=GRID_COLOR, linewidth=0.7)
    axes[0, 0].legend(loc="lower left")
    fig.suptitle(
        "Sensitivity of the cross-country relationship to resolution and weighting",
        fontsize=21,
        fontweight="bold",
    )
    save(fig, "figure6_resolution_sensitivity", formats, dpi)


def figure7_bootstrap(
    mix: pd.DataFrame, formats: tuple[str, ...], dpi: int
) -> None:
    if EXPANDED_BOOTSTRAP_PATH.exists():
        frame = pd.read_csv(EXPANDED_BOOTSTRAP_PATH)
        families = {
            "score": {
                "metrics": {
                    "LogReg": "auc_gain",
                    "RandForest": "auc_gain",
                    "GradientBoosting": "r2_gain",
                    "LightGBM": "r2_gain",
                },
                "labels": {
                    "classification": "Improvement from adding weather (AUC)",
                    "regression": r"Improvement from adding weather ($R^2$)",
                },
                "stem": "figure7_block_bootstrap_intervals",
                "title": "Weather Improvements After Resampling Whole Weeks",
            },
            "error": {
                "metrics": {
                    "LogReg": "brier_reduction",
                    "RandForest": "brier_reduction",
                    "GradientBoosting": "mae_reduction",
                    "LightGBM": "mae_reduction",
                },
                "labels": {
                    "classification": "Reduction in probability error (Brier score)",
                    "regression": "Reduction in average error (MAE, share points)",
                },
                "stem": "figure7b_block_bootstrap_error_intervals",
                "title": "Prediction-Error Reductions After Resampling Whole Weeks",
            },
        }
        confidence = 100 * float(frame["confidence"].iloc[0])
        for family in families.values():
            fig, axes = plt.subplots(
                2, 2, figsize=(18, 14), constrained_layout=True
            )
            for ax, model in zip(axes.flat, MODEL_ORDER):
                task = (
                    "classification"
                    if model in {"LogReg", "RandForest"}
                    else "regression"
                )
                local = (
                    frame[
                        frame["model"].eq(model)
                        & frame["metric"].eq(family["metrics"][model])
                    ]
                    .sort_values("point_estimate")
                    .reset_index(drop=True)
                )
                y = np.arange(len(local))
                ax.axvline(0, color="#30343B", linewidth=1.1)
                for yi, row in enumerate(local.itertuples()):
                    color = WEATHER_COLOR if row.positive_ci else UNIFORM_COLOR
                    ax.hlines(
                        yi, row.ci_low, row.ci_high, color=color, linewidth=1.4
                    )
                    ax.vlines(
                        [row.ci_low, row.ci_high],
                        yi - 0.12,
                        yi + 0.12,
                        color=color,
                        linewidth=1.1,
                    )
                    ax.scatter(
                        row.point_estimate, yi, color=color, s=42, zorder=3
                    )
                ax.set_yticks(y, local["country"])
                ax.set_xlabel(family["labels"][task])
                ax.set_title(
                    f"{MODEL_NAMES[model]} — clear improvement in "
                    f"{int(local['positive_ci'].sum())} of {len(local)} countries"
                )
                ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7)
            fig.suptitle(
                f"{family['title']}\n"
                f"({confidence:g}% confidence intervals)",
                fontsize=21,
                fontweight="bold",
            )
            save(fig, family["stem"], formats, dpi)
        return

    path = BASELINE_SUMMARY_DIR / "bootstrap_gains.csv"
    frame = pd.read_csv(path)
    name_to_code = {country: code for code, country in COUNTRIES.items()}
    frame["code"] = frame["country"].map(name_to_code)
    frame = frame.merge(mix[["code", "wind_minus_solar"]], on="code", how="left")
    order = frame.groupby("country")["wind_minus_solar"].first().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5), sharey=True, constrained_layout=True)
    for ax, model in zip(axes, ["LogReg", "RandForest"]):
        local = frame[frame["model"].eq(model)].set_index("country").reindex(order).reset_index()
        y = np.arange(len(local))
        ax.axvline(0, color="#30343B", linewidth=1.2)
        for yi, row in enumerate(local.itertuples()):
            color = WEATHER_COLOR if bool(row.excludes_zero) else UNIFORM_COLOR
            ax.errorbar(
                row.gain_mean,
                yi,
                xerr=np.array([[row.gain_mean - row.gain_lo95], [row.gain_hi95 - row.gain_mean]]),
                fmt="o",
                color=color,
                capsize=4,
                markersize=7,
            )
        ax.set_yticks(y, local["country"])
        ax.set_xlabel("Weather-added AUC gain (95% weekly-block interval)")
        ax.set_title(MODEL_LABELS[model])
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7)
    fig.suptitle(
        "Block-bootstrap uncertainty in country-level weather-added gain",
        fontsize=21,
        fontweight="bold",
    )
    save(fig, "figure7_block_bootstrap_intervals", formats, dpi)


def main() -> None:
    args = parse_args()
    formats = tuple(args.formats)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    _, _, mix = load_energy_data()
    results = add_resource_mix(load_canonical_model_results(), mix)
    canonical = canonical_slice(results)
    correlations = compute_resolution_correlations(results)

    figure1_pipeline(formats, args.dpi)
    figure2_country_map(mix, formats, args.dpi)
    figure3_capacity_map(formats, args.dpi)
    headline = figure4_gain_scatter(canonical, formats, args.dpi)
    example = figure5_timeseries(args.example_days, formats, args.dpi)
    figure6_resolution_sensitivity(correlations, formats, args.dpi)
    figure7_bootstrap(mix, formats, args.dpi)

    headline.to_csv(OUTPUT_DIR / "figure4_correlations.csv", index=False)
    example.to_csv(OUTPUT_DIR / "figure5_example_metrics.csv", index=False)
    correlations.to_csv(OUTPUT_DIR / "figure6_resolution_correlations.csv", index=False)
    manifest = pd.DataFrame(
        [
            (1, "figure1_pipeline", "Analysis pipeline"),
            (2, "figure2_country_map", "Study countries and wind-minus-solar share"),
            (3, "figure3_capacity_weighting_denmark", "Denmark capacity-weighting example"),
            (4, "figure4_gain_vs_wind_minus_solar", "Headline cross-country relationship"),
            (5, "figure5_denmark_prediction_timeseries", "Held-out Denmark time series"),
            (6, "figure6_resolution_sensitivity", "Resolution and weighting sensitivity"),
            (7, "figure7_block_bootstrap_intervals", "Country-level AUC/R2 bootstrap intervals"),
            ("7b", "figure7b_block_bootstrap_error_intervals", "Country-level Brier/MAE bootstrap intervals"),
        ],
        columns=["figure_number", "file_stem", "description"],
    )
    manifest.to_csv(OUTPUT_DIR / "paper_figure_manifest.csv", index=False)
    print(f"\nDone. Paper figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
