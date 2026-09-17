#!/usr/bin/env python3
"""Create the complete COVID-era spatial-resolution figure suite.

The script uses only completed pipeline tables; it never downloads or rebuilds
weather data.  By default it writes publication-ready PNG and PDF files beside
``era_resolution_comparison.png``.
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
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "figures" / ".cache" / "matplotlib"))

COMPARISON_DIR = ROOT / "results" / "covid_era_spatial_resolution" / "comparison"
DEFAULT_FIGURE_DIR = ROOT / "figures" / "era_spatial_resolution" / "comparison"
ERA_ORDER = ("pre", "covid", "post")
ERA_LABELS = {"pre": "Pre-COVID", "covid": "COVID", "post": "Post-COVID"}
ERA_LONG_LABELS = {
    "pre": "Pre-COVID\n(2016-2019)",
    "covid": "COVID\n(2020-2021)",
    "post": "Post-COVID\n(2022-Apr 2026)",
}
ERA_COLORS = {"pre": "#3975b7", "covid": "#d08a2e", "post": "#3b966b"}
COUNTRIES = {
    "at": "Austria", "be": "Belgium", "bg": "Bulgaria", "cz": "Czechia",
    "de": "Germany", "dk": "Denmark", "es": "Spain", "fr": "France",
    "gr": "Greece", "hr": "Croatia", "ie": "Ireland", "lt": "Lithuania",
    "lv": "Latvia", "nl": "Netherlands", "pt": "Portugal", "ro": "Romania",
    "rs": "Serbia", "si": "Slovenia", "sk": "Slovakia",
}
PRIMARY = (
    ("classification", "RandForest", "Random Forest classification", "AUC"),
    ("regression", "GradientBoosting", "Gradient Boosting regression", r"$R^2$"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-results",
        default=str(COMPARISON_DIR / "model_results_all_eras.csv"),
        help="combined country-level model results",
    )
    parser.add_argument(
        "--resolution-summary",
        default=str(COMPARISON_DIR / "resolution_summary_all_eras.csv"),
        help="combined spatial-resolution summary",
    )
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf"))
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def configure_plotting() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "figure.titlesize": 17,
            "figure.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.facecolor": "white",
        }
    )


def load_primary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["code"] = frame["code"].astype(str).str.lower()
    frame["country"] = frame["code"].map(COUNTRIES).fillna(frame["code"].str.upper())
    mask = frame["scheme"].eq("capacity") & np.isclose(frame["resolution_deg"], 0.25)
    primary = frame[mask].copy()
    keep = pd.Series(False, index=primary.index)
    for task, model, _, _ in PRIMARY:
        keep |= primary["task"].eq(task) & primary["model"].eq(model)
    return primary[keep].copy()


def save_figure(fig, figure_dir: Path, stem: str, formats: tuple[str, ...], dpi: int) -> list[Path]:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = figure_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi if fmt == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def country_order(primary: pd.DataFrame) -> list[str]:
    regression = primary[
        primary["task"].eq("regression") & primary["model"].eq("GradientBoosting")
    ]
    post = regression[regression["era"].eq("post")].set_index("code")["both_score"]
    fallback = regression.groupby("code")["both_score"].median()
    score = post.combine_first(fallback).sort_values(ascending=False)
    return score.index.tolist()


def annotated_heatmap(ax, matrix: pd.DataFrame, *, cmap, norm, value_format: str) -> None:
    masked = np.ma.masked_invalid(matrix.to_numpy(float))
    image = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(matrix.columns)), [ERA_LONG_LABELS[x] for x in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            label = "NA" if pd.isna(value) else format(value, value_format)
            if pd.isna(value):
                color = "#555555"
            else:
                rgba = cmap(norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                color = "black" if luminance > 0.58 else "white"
            ax.text(column, row, label, ha="center", va="center", fontsize=7.2, color=color)
    return image


def figure_country_scores(primary: pd.DataFrame, order: list[str], figure_dir: Path, formats, dpi):
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 9.4), constrained_layout=True)
    settings = ((0.45, 1.0, "viridis"), (-0.20, 1.0, "RdYlGn"))
    for ax, (task, model, title, metric), (lower, upper, cmap_name) in zip(axes, PRIMARY, settings):
        data = primary[primary["task"].eq(task) & primary["model"].eq(model)]
        matrix = data.pivot(index="code", columns="era", values="both_score").reindex(
            index=order, columns=ERA_ORDER
        )
        matrix.index = [f"{code.upper()}  {COUNTRIES.get(code, code)}" for code in matrix.index]
        cmap = colormaps[cmap_name].copy()
        cmap.set_bad("#dedede")
        image = annotated_heatmap(
            ax, matrix, cmap=cmap, norm=Normalize(lower, upper, clip=True), value_format=".2f"
        )
        ax.set_title(f"{title}: weather + calendar {metric}")
        fig.colorbar(image, ax=ax, shrink=0.72, pad=0.02, label=metric)
    fig.suptitle("Country-level predictive accuracy at 0.25° (capacity weighted)")
    return save_figure(fig, figure_dir, "country_model_accuracy_by_era", formats, dpi)


def figure_country_gains(primary: pd.DataFrame, order: list[str], figure_dir: Path, formats, dpi):
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import TwoSlopeNorm

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 9.4), constrained_layout=True)
    for ax, (task, model, title, metric) in zip(axes, PRIMARY):
        data = primary[primary["task"].eq(task) & primary["model"].eq(model)]
        matrix = data.pivot(index="code", columns="era", values="gain").reindex(
            index=order, columns=ERA_ORDER
        )
        matrix.index = [f"{code.upper()}  {COUNTRIES.get(code, code)}" for code in matrix.index]
        bound = max(0.05, float(np.nanmax(np.abs(matrix.to_numpy(float)))))
        cmap = colormaps["RdBu"].copy()
        cmap.set_bad("#dedede")
        image = annotated_heatmap(
            ax,
            matrix,
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
            value_format="+.2f",
        )
        ax.set_title(f"{title}: weather-added {metric} gain")
        fig.colorbar(image, ax=ax, shrink=0.72, pad=0.02, label=f"Weather-added {metric} gain")
    fig.suptitle("Where weather features improve prediction")
    return save_figure(fig, figure_dir, "country_weather_gain_by_era", formats, dpi)


def figure_gain_distributions(primary: pd.DataFrame, figure_dir: Path, formats, dpi):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20260722)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for ax, (task, model, title, metric) in zip(axes, PRIMARY):
        subset = primary[primary["task"].eq(task) & primary["model"].eq(model)]
        arrays = []
        for position, era in enumerate(ERA_ORDER, start=1):
            values = subset.loc[subset["era"].eq(era), "gain"].dropna().to_numpy(float)
            arrays.append(values)
            jitter = rng.normal(0, 0.045, len(values))
            ax.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=24,
                color=ERA_COLORS[era],
                alpha=0.72,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
            positive = int((values > 0).sum())
            ax.text(
                position,
                0.97,
                f"{positive}/{len(values)} positive",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8.5,
            )
        boxes = ax.boxplot(arrays, positions=range(1, 4), widths=0.45, patch_artist=True, showfliers=False)
        for patch, era in zip(boxes["boxes"], ERA_ORDER):
            patch.set_facecolor(ERA_COLORS[era]); patch.set_alpha(0.18)
            patch.set_edgecolor(ERA_COLORS[era])
        for median in boxes["medians"]:
            median.set_color("#222222"); median.set_linewidth(2)
        ax.axhline(0, color="#8b2f2f", linestyle="--", linewidth=1)
        ax.set_xticks(range(1, 4), [ERA_LABELS[x] for x in ERA_ORDER])
        ax.set_ylabel(f"Weather-added {metric} gain")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Weather features help most countries, but the size of the gain varies")
    return save_figure(fig, figure_dir, "weather_gain_distributions_by_era", formats, dpi)


def figure_capacity_advantage(all_models: pd.DataFrame, figure_dir: Path, formats, dpi):
    import matplotlib.pyplot as plt

    data = all_models[np.isclose(all_models["resolution_deg"], 0.25)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    rng = np.random.default_rng(220726)
    for ax, (task, model, title, metric) in zip(axes, PRIMARY):
        sub = data[data["task"].eq(task) & data["model"].eq(model)]
        paired = sub.pivot_table(index=["code", "era"], columns="scheme", values="both_score").dropna()
        paired["difference"] = paired["capacity"] - paired["uniform"]
        arrays = []
        for position, era in enumerate(ERA_ORDER, start=1):
            values = paired.xs(era, level="era")["difference"].to_numpy(float) if era in paired.index.get_level_values("era") else np.array([])
            arrays.append(values)
            ax.scatter(
                np.full(len(values), position) + rng.normal(0, 0.045, len(values)), values,
                s=25, color=ERA_COLORS[era], alpha=0.7, edgecolor="white", linewidth=0.35, zorder=3,
            )
            mean = float(np.mean(values)) if len(values) else np.nan
            better = int((values > 0).sum())
            ax.scatter(position, mean, marker="D", s=56, color="#202020", zorder=5)
            ax.text(
                position, 0.97, f"mean {mean:+.3f}\n{better}/{len(values)} better",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.3,
            )
        boxes = ax.boxplot(arrays, positions=range(1, 4), widths=0.45, patch_artist=True, showfliers=False)
        for patch, era in zip(boxes["boxes"], ERA_ORDER):
            patch.set_facecolor(ERA_COLORS[era]); patch.set_alpha(0.16); patch.set_edgecolor(ERA_COLORS[era])
        for median in boxes["medians"]:
            median.set_color("#222222"); median.set_linewidth(2)
        ax.axhline(0, color="#555555", linewidth=1)
        ax.set_xticks(range(1, 4), [ERA_LABELS[x] for x in ERA_ORDER])
        ax.set_ylabel(f"Capacity − uniform {metric}")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Capacity weighting consistently improves predictive accuracy")
    return save_figure(fig, figure_dir, "capacity_weighting_advantage", formats, dpi)


def figure_resolution_tradeoff(summary: pd.DataFrame, figure_dir: Path, formats, dpi):
    import matplotlib.pyplot as plt

    data = summary.copy()
    if "relationship" in data:
        data = data[data["relationship"].eq("all_europe")]
    data = data[data["scheme"].eq("capacity")]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), constrained_layout=True)

    cost = data.drop_duplicates(["era", "resolution_deg"])[["era", "resolution_deg", "total_point_hours"]]
    cost["relative_pct"] = cost.groupby("era")["total_point_hours"].transform(lambda x: 100 * x / x.iloc[np.argmin(cost.loc[x.index, "resolution_deg"].to_numpy())])
    for era in ERA_ORDER:
        group = cost[cost["era"].eq(era)].sort_values("resolution_deg")
        axes[0, 0].plot(group["resolution_deg"], group["relative_pct"], marker="o", color=ERA_COLORS[era], linewidth=2, label=ERA_LABELS[era])
    axes[0, 0].set_title("Computational workload")
    axes[0, 0].set_ylabel("Grid-point hours (% of 0.25°)")
    axes[0, 0].set_yscale("log")

    line_styles = {"RandForest": "-", "GradientBoosting": "--"}
    for era in ERA_ORDER:
        for task, model, _, _ in PRIMARY:
            group = data[data["era"].eq(era) & data["task"].eq(task) & data["model"].eq(model)].sort_values("resolution_deg")
            label = f"{ERA_LABELS[era]} · {'RF' if model == 'RandForest' else 'GB'}"
            axes[0, 1].plot(group["resolution_deg"], group["median_absolute_gain_drift"], marker="o", color=ERA_COLORS[era], linestyle=line_styles[model], linewidth=1.8, label=label)
            axes[1, 0].plot(group["resolution_deg"], group["max_absolute_gain_drift"], marker="o", color=ERA_COLORS[era], linestyle=line_styles[model], linewidth=1.8)
            axes[1, 1].plot(group["resolution_deg"], group["correlation"], marker="o", color=ERA_COLORS[era], linestyle=line_styles[model], linewidth=1.8)
    axes[0, 1].axhline(0.01, color="#555555", linestyle=":", linewidth=1.3)
    axes[0, 1].set_title("Typical country fidelity")
    axes[0, 1].set_ylabel("Median |gain − 0.25° gain|")
    axes[1, 0].set_title("Worst-country sensitivity")
    axes[1, 0].set_ylabel("Maximum |gain − 0.25° gain|")
    axes[1, 1].set_title("Cross-country relationship stability")
    axes[1, 1].set_ylabel("Gain vs. wind-minus-solar r")

    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1, 2], ["0.25", "0.5", "1", "2"])
        ax.set_xlabel("Weather-grid resolution (degrees)")
        ax.grid(alpha=0.22)
        ax.axvline(1.0, color="#777777", linestyle=":", linewidth=1)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=7.5, ncol=2)
    fig.suptitle("A 1° grid preserves most predictive information at far lower cost")
    return save_figure(fig, figure_dir, "resolution_cost_fidelity_tradeoff", formats, dpi)


def load_size_sensitivity() -> pd.DataFrame:
    paths = {
        "pre": ROOT / "results" / "covid_era_spatial_resolution" / "pre" / "country_size_sensitivity.csv",
        "covid": ROOT / "results" / "covid_era_spatial_resolution" / "covid" / "country_size_sensitivity.csv",
        "post": ROOT / "results" / "post_covid_spatial_resolution" / "country_size_sensitivity.csv",
    }
    frames = []
    for era, path in paths.items():
        if path.exists():
            frame = pd.read_csv(path); frame["era"] = era; frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def figure_size_control(size: pd.DataFrame, figure_dir: Path, formats, dpi):
    import matplotlib.pyplot as plt

    if size.empty:
        return []
    size = size[size["scheme"].eq("capacity") & np.isclose(size["resolution_deg"], 0.25)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    width = 0.34
    for ax, (task, model, title, _) in zip(axes, PRIMARY):
        sub = size[size["task"].eq(task) & size["model"].eq(model)].set_index("era").reindex(ERA_ORDER)
        x = np.arange(3)
        raw = sub["gain_vs_wind_minus_solar_r"].to_numpy(float)
        partial = sub["partial_gain_vs_wind_minus_solar_controlling_log_native_points_r"].to_numpy(float)
        ax.bar(x - width / 2, raw, width, label="Raw correlation", color="#607ea8")
        ax.bar(x + width / 2, partial, width, label="Controlling country grid size", color="#43a17b")
        for xpos, values in ((x - width / 2, raw), (x + width / 2, partial)):
            for xx, value in zip(xpos, values):
                ax.text(xx, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        for xx, count in zip(x, sub["countries"].to_numpy()):
            ax.text(xx, 0.02, f"n={int(count)}", ha="center", va="bottom", fontsize=8, color="#333333")
        ax.set_xticks(x, [ERA_LABELS[e] for e in ERA_ORDER])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Correlation coefficient")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Country size does not explain the wind-dominance relationship")
    return save_figure(fig, figure_dir, "country_size_sensitivity", formats, dpi)


def safe_wilcoxon(values: np.ndarray) -> float:
    try:
        from scipy.stats import wilcoxon

        if len(values) and not np.allclose(values, 0):
            return float(wilcoxon(values).pvalue)
    except (ImportError, ValueError):
        pass
    return np.nan


def figure_pre_post_changes(primary: pd.DataFrame, figure_dir: Path, formats, dpi):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7.3), constrained_layout=True)
    for ax, (task, model, title, metric) in zip(axes, PRIMARY):
        sub = primary[primary["task"].eq(task) & primary["model"].eq(model)]
        paired = sub[sub["era"].isin(["pre", "post"])].pivot(index="code", columns="era", values="gain").dropna()
        paired["change"] = paired["post"] - paired["pre"]
        paired = paired.sort_values("change")
        y = np.arange(len(paired))
        for yy, row in enumerate(paired.itertuples()):
            color = "#3b966b" if row.change >= 0 else "#b84d4d"
            ax.plot([row.pre, row.post], [yy, yy], color=color, alpha=0.55, linewidth=2)
        ax.scatter(paired["pre"], y, color=ERA_COLORS["pre"], s=34, label="Pre-COVID", zorder=3)
        ax.scatter(paired["post"], y, color=ERA_COLORS["post"], s=34, label="Post-COVID", zorder=3)
        ax.axvline(0, color="#777777", linestyle=":", linewidth=1)
        ax.set_yticks(y, [f"{code.upper()}  {COUNTRIES.get(code, code)}" for code in paired.index])
        ax.set_xlabel(f"Weather-added {metric} gain")
        p_value = safe_wilcoxon(paired["change"].to_numpy(float))
        p_text = "NA" if pd.isna(p_value) else f"{p_value:.3f}"
        ax.set_title(
            f"{title}\nmean change {paired['change'].mean():+.3f}; "
            f"median {paired['change'].median():+.3f}; Wilcoxon p={p_text}"
        )
        ax.grid(axis="x", alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("No uniform pre- to post-COVID shift in weather-added gain")
    return save_figure(fig, figure_dir, "pre_to_post_country_gain_changes", formats, dpi)


def load_audits() -> pd.DataFrame:
    paths = {
        "pre": ROOT / "results" / "covid_era_spatial_resolution" / "pre" / "data_readiness_audit.csv",
        "covid": ROOT / "results" / "covid_era_spatial_resolution" / "covid" / "data_readiness_audit.csv",
        "post": ROOT / "results" / "post_covid_spatial_resolution" / "data_readiness_audit.csv",
    }
    frames = []
    for era, path in paths.items():
        if path.exists():
            frame = pd.read_csv(path); frame["era"] = era; frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def figure_data_coverage(audits: pd.DataFrame, order: list[str], figure_dir: Path, formats, dpi):
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    if audits.empty:
        return []
    audits["code"] = audits["code"].astype(str).str.lower()
    matrix = audits.pivot(index="code", columns="era", values="energy_coverage_fraction").reindex(index=order, columns=ERA_ORDER)
    mix = audits.pivot(index="code", columns="era", values="energy_mix_components_good").reindex(index=order, columns=ERA_ORDER)
    matrix.index = [f"{code.upper()}  {COUNTRIES.get(code, code)}" for code in matrix.index]
    fig, ax = plt.subplots(figsize=(7.0, 8.8), constrained_layout=True)
    cmap = colormaps["YlGn"].copy(); cmap.set_bad("#dedede")
    image = annotated_heatmap(ax, matrix, cmap=cmap, norm=Normalize(0, 1), value_format=".0%")
    for row in range(mix.shape[0]):
        for column in range(mix.shape[1]):
            value = mix.iloc[row, column]
            if pd.notna(value) and not bool(value):
                ax.text(column + 0.42, row - 0.32, "×mix", ha="right", va="top", fontsize=6.5, color="#7d1e1e")
    fig.colorbar(image, ax=ax, shrink=0.78, pad=0.03, label="Renewable-share target coverage")
    ax.set_title("Target coverage by country and era\n×mix = wind/solar breakdown unavailable (model target still usable)")
    return save_figure(fig, figure_dir, "data_coverage_by_country_and_era", formats, dpi)


def build_statistics(primary: pd.DataFrame, all_models: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task, model, title, metric in PRIMARY:
        sub = primary[primary["task"].eq(task) & primary["model"].eq(model)]
        for era in ERA_ORDER:
            group = sub[sub["era"].eq(era)]
            valid = group.dropna(subset=["gain", "both_score"])
            rows.extend(
                [
                    {"analysis": "primary", "model": model, "era": era, "metric": f"median_weather_plus_calendar_{metric}", "value": valid["both_score"].median(), "n": len(valid)},
                    {"analysis": "primary", "model": model, "era": era, "metric": f"median_weather_gain_{metric}", "value": valid["gain"].median(), "n": len(valid)},
                    {"analysis": "primary", "model": model, "era": era, "metric": "countries_positive_gain", "value": int((valid["gain"] > 0).sum()), "n": len(valid)},
                ]
            )
        paired = sub[sub["era"].isin(["pre", "post"])].pivot(index="code", columns="era", values="gain").dropna()
        changes = paired["post"] - paired["pre"]
        rows.extend(
            [
                {"analysis": "pre_post", "model": model, "era": "post-minus-pre", "metric": f"mean_gain_change_{metric}", "value": changes.mean(), "n": len(changes)},
                {"analysis": "pre_post", "model": model, "era": "post-minus-pre", "metric": f"median_gain_change_{metric}", "value": changes.median(), "n": len(changes)},
                {"analysis": "pre_post", "model": model, "era": "post-minus-pre", "metric": "wilcoxon_p", "value": safe_wilcoxon(changes.to_numpy(float)), "n": len(changes)},
            ]
        )

    models025 = all_models[np.isclose(all_models["resolution_deg"], 0.25)]
    for task, model, _, metric in PRIMARY:
        sub = models025[models025["task"].eq(task) & models025["model"].eq(model)]
        paired = sub.pivot_table(index=["code", "era"], columns="scheme", values="both_score").dropna()
        paired["difference"] = paired["capacity"] - paired["uniform"]
        for era in ERA_ORDER:
            values = paired.xs(era, level="era")["difference"]
            rows.extend(
                [
                    {"analysis": "weighting", "model": model, "era": era, "metric": f"mean_capacity_minus_uniform_{metric}", "value": values.mean(), "n": len(values)},
                    {"analysis": "weighting", "model": model, "era": era, "metric": "countries_capacity_better", "value": int((values > 0).sum()), "n": len(values)},
                ]
            )

    data = summary.copy()
    if "relationship" in data:
        data = data[data["relationship"].eq("all_europe")]
    data = data[data["scheme"].eq("capacity")].drop_duplicates(["era", "resolution_deg"])
    for era in ERA_ORDER:
        group = data[data["era"].eq(era)].sort_values("resolution_deg")
        if group.empty:
            continue
        reference = group.iloc[0]["total_point_hours"]
        for row in group.itertuples():
            rows.append({"analysis": "workload", "model": "all", "era": era, "metric": f"point_hours_pct_at_{row.resolution_deg:g}deg", "value": 100 * row.total_point_hours / reference, "n": int(row.cost_regions)})
    return pd.DataFrame(rows)


def generate_figure_suite(
    model_results_path: Path,
    resolution_summary_path: Path,
    figure_dir: Path,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 240,
    statistics_dir: Path | None = None,
) -> list[Path]:
    configure_plotting()
    all_models = pd.read_csv(model_results_path)
    summary = pd.read_csv(resolution_summary_path)
    primary = load_primary(model_results_path)
    order = country_order(primary)
    outputs: list[Path] = []
    outputs += figure_country_scores(primary, order, figure_dir, formats, dpi)
    outputs += figure_country_gains(primary, order, figure_dir, formats, dpi)
    outputs += figure_gain_distributions(primary, figure_dir, formats, dpi)
    outputs += figure_capacity_advantage(all_models, figure_dir, formats, dpi)
    outputs += figure_resolution_tradeoff(summary, figure_dir, formats, dpi)
    outputs += figure_size_control(load_size_sensitivity(), figure_dir, formats, dpi)
    outputs += figure_pre_post_changes(primary, figure_dir, formats, dpi)
    outputs += figure_data_coverage(load_audits(), order, figure_dir, formats, dpi)

    statistics_dir = statistics_dir or model_results_path.parent
    statistics_dir.mkdir(parents=True, exist_ok=True)
    statistics = build_statistics(primary, all_models, summary)
    stats_path = statistics_dir / "figure_statistics.csv"
    statistics.to_csv(stats_path, index=False)
    outputs.append(stats_path)
    manifest = pd.DataFrame(
        {
            "file": [str(path) for path in outputs],
            "kind": [path.suffix.lstrip(".") for path in outputs],
        }
    )
    manifest_path = statistics_dir / "figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    outputs.append(manifest_path)
    return outputs


def main() -> None:
    args = parse_args()
    model_results_path = Path(args.model_results).expanduser().resolve()
    resolution_summary_path = Path(args.resolution_summary).expanduser().resolve()
    figure_dir = Path(args.figure_dir).expanduser().resolve()
    for path in (model_results_path, resolution_summary_path):
        if not path.exists():
            raise FileNotFoundError(path)
    outputs = generate_figure_suite(
        model_results_path,
        resolution_summary_path,
        figure_dir,
        tuple(args.formats),
        args.dpi,
        model_results_path.parent,
    )
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
