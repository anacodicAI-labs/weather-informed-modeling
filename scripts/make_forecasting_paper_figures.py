#!/usr/bin/env python3
"""Figures for the Forecasting manuscript Results section.

Paper Figure 2: per-country calendar vs calendar+weather, by era   (fig1)
Paper Figure 3: wind dominance vs weather-added gain, post-COVID    (fig2)
Paper Figure 4: predictive drift against workload reduction        (fig3)

Reads only the curated tables in results/paper/, so it runs from a clean clone:
    python scripts/make_forecasting_paper_figures.py
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.text import Text

COMP = "results/paper"   # curated, path-free tables committed with the repo
ERAS = [("pre", "Pre-COVID (2016-2019)"), ("post", "Post-COVID (2022-2026)")]
CAL, BOTH = "#9AA5B1", "#0F766E"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 10, "axes.titlesize": 11, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 9, "pdf.fonttype": 42,
    "savefig.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
})


def load():
    M = pd.read_csv(f"{COMP}/model_results_all_eras.csv")
    return M[(M.scheme == "capacity") & (M.resolution_deg == 0.25)]


def _draw_arrows(fig, pending, r_pt=3.0, head_pt=6.5, width_pt=5.6, lw=1.9):
    """Draw calendar -> calendar+weather arrows after layout is final.

    matplotlib's arrowheads have a fixed size, so an arrow shorter than its
    own head drew the head's wide base on the wrong side of the dot: a gain
    of -0.012 looked like an arrow pointing right. Here the geometry is
    computed in points. The shaft starts at the edge of the calendar dot and
    the head always points from the dot toward the weather score. When the
    change is smaller than one arrowhead, the head is drawn at minimum length
    just outside the dot, so the direction is always visible.
    """
    from matplotlib.patches import Polygon
    fig.canvas.draw()
    px = fig.dpi / 72.0
    r, H, W = r_pt * px, head_pt * px, width_pt * px
    for ax, a, b, y, col in pending:
        ax.set_autoscale_on(False)
        T, inv = ax.transData, ax.transData.inverted()
        (xa, yd), (xb, _) = T.transform([(a, y), (b, y)])
        d = xb - xa
        if a == b:                             # exactly no change: no arrow
            continue
        sgn = 1.0 if d > 0 else -1.0
        start = xa + sgn * r
        base, tip = xb - sgn * H, xb
        if sgn * (base - start) < 0:           # shorter than its own head
            base, tip = start, start + sgn * H
        if sgn * (base - start) > 0:
            (x0, _), (x1, _) = inv.transform([(start, yd), (base, yd)])
            ax.plot([x0, x1], [y, y], color=col, lw=lw,
                    solid_capstyle="butt", zorder=3)
        verts = inv.transform([(tip, yd), (base, yd + W / 2), (base, yd - W / 2)])
        ax.add_patch(Polygon(verts, closed=True, color=col, lw=0, zorder=3))


def fig1(ref, out):
    """Per-country dumbbells.

    Both era panels in a row share one country ordering (by the post-COVID
    score) and one y-axis, so a country can be tracked across eras and the
    labels are drawn once per row instead of four times. Connectors are
    coloured by the sign of the gain, which is what makes the handful of
    systems where weather does not help legible at a glance.
    """
    UP, DOWN = "#0F766E", "#C2410C"
    panels = [("classification", "RandForest", "AUC", 0.5),
              ("regression", "GradientBoosting", "$R^2$", 0.0)]
    pending = []
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.4), sharey="row")
    for r, (task, model, lab, refline) in enumerate(panels):
        sub = ref[(ref.task == task) & (ref.model == model)]
        order = (sub[sub.era == "post"].dropna(subset=["both_score"])
                 .sort_values("both_score")["code"].tolist())
        pos = {c: i for i, c in enumerate(order)}
        # No truncation (Reviewer 1): the full R^2 range is shown, so every
        # country -- including Croatia's pre-COVID baseline below -1 -- is
        # drawn as an ordinary arrow rather than an off-axis marker.
        lo = None
        # Axis range from the plotted eras only. The source table also holds the
        # 2020-21 COVID era, which is not shown here; including it stretched
        # the R^2 axis to -3.9 for a value (Slovenia, COVID era) never drawn.
        shown = sub[sub.era.isin([e for e, _ in ERAS])]
        vals = pd.concat([shown.calendar_score, shown.both_score]).dropna()
        if lo is not None:
            vals = vals.clip(lower=lo)
        pad = 0.06 * (vals.max() - vals.min())
        xlim = (vals.min() - pad, vals.max() + pad)
        for c, (era, title) in enumerate(ERAS):
            ax = axes[r, c]
            d = sub[sub.era == era].dropna(subset=["both_score"]).copy()
            d["y"] = d["code"].map(pos)
            d = d.dropna(subset=["y"])
            cal, both = d.calendar_score.copy(), d.both_score.copy()
            clipped = []
            if lo is not None:
                clipped = sorted(d.loc[(cal < lo) | (both < lo), "code"].str.upper())
                cal, both = cal.clip(lower=lo), both.clip(lower=lo)
            ax.axvline(refline, color="#D5D9DE", lw=1.0, zorder=0)
            for (_, row), a, b in zip(d.iterrows(), cal, both):
                col = UP if row.gain > 0 else DOWN
                pending.append((ax, a, b, row.y, col))
                ax.scatter(a, row.y, s=20, color="#FFFFFF", edgecolor=col,
                           lw=1.4, zorder=4)
            ax.set_xlim(*xlim)
            ax.set_yticks(range(len(order)), [x.upper() for x in order])
            ax.tick_params(axis="y", length=0)
            ax.set_xlabel(lab)
            if r == 0:
                ax.set_title(title, fontsize=11, pad=8)
            n_missing = len(order) - len(d)
            note = f"n = {len(d)}" + (f"  ({n_missing} undefined)" if n_missing else "")
            ax.text(0.98, 0.02, note, transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=8.5, color="#5A6069")
            if clipped:
                # Reviewer 1: the axis truncation was only noted in a corner
                # caption. Mark each truncated row on the axis itself and print
                # the value that falls outside, so the break cannot be missed.
                for code in clipped:
                    # Neutral grey: orange is reserved for "weather does not
                    # help", and a truncated country may well have improved.
                    y = pos[code.lower()]
                    row = d.loc[d.code == code.lower()].iloc[0]
                    c0, c1 = row.calendar_score, row.both_score
                    ax.plot([xlim[0]], [y], marker="<", ms=7, color="#6B7178",
                            clip_on=False, zorder=5)
                    ax.text(xlim[0] + 0.012 * (xlim[1] - xlim[0]), y + 0.34,
                            f"off axis: {c0:.2f} → {c1:.2f}".replace("-", "−"),
                            fontsize=7.6, color="#4A4F57", va="bottom", zorder=5)
            ax.grid(axis="x", color="#F1F3F5", lw=0.8)
            ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], marker="o", ls="none", mfc="white", mec="#6B7178",
                   mew=1.4, ms=6, label="calendar only"),
        plt.Line2D([], [], color=UP, lw=2.2, label="weather improves"),
        plt.Line2D([], [], color=DOWN, lw=2.2, label="weather does not help"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.995), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    _draw_arrows(fig, pending)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/fig_results_percountry.{ext}", dpi=300)
    plt.close(fig)


def _label_points(ax, xs, ys, labels, marker_pt=3.6, line=None, crowd_pt=9.0):
    """Non-overlapping point labels, placed a fixed few points from each dot.

    Reviewer 1 noted that countries with similar wind dominance and similar
    gain collide. Each label is tried at a ring of offsets measured in points
    (so the gap to its own dot stays the same at any figure size). A position
    is kept only if the rendered text box clears every placed label, every
    dot, and the axes. Labels that cannot sit beside their dot move further
    out with a thin leader line, so ownership is never ambiguous.
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    r = marker_pt * px_per_pt + 1.5
    ax_box = ax.get_window_extent(rend)

    near = [(5, 3), (-5, 3), (5, -3), (-5, -3), (0, 6), (0, -6), (7, 0), (-7, 0)]
    far = [(dx * k, dy * k) for k in (2.6, 3.6) for dx, dy in
           [(1, 1), (-1, 1), (1, -1), (-1, -1), (0, 1.3), (0, -1.3), (1.4, 0), (-1.4, 0)]
           for dx, dy in [(dx * 6, dy * 5)]]

    line_px = None
    if line is not None:                          # fitted line, sampled densely
        lx = np.linspace(line[0][0], line[0][-1], 400)
        ly = np.interp(lx, line[0], line[1])
        line_px = ax.transData.transform(np.column_stack([lx, ly]))
    crowd = crowd_pt * px_per_pt
    nearest = [min((np.hypot(*(pts[i] - pts[j])) for j in range(len(pts)) if j != i),
                   default=np.inf) for i in range(len(pts))]

    r_other = r + 7 * px_per_pt                       # extra room around other countries' dots

    def clear(bb, own, avoid_dots=True, avoid_line=True):
        if (bb.x0 < ax_box.x0 or bb.x1 > ax_box.x1 or
                bb.y0 < ax_box.y0 or bb.y1 > ax_box.y1):
            return False
        if any(bb.overlaps(b) for b in placed):      # labels never overlap
            return False
        if avoid_dots:
            for k, (qx, qy) in enumerate(pts):
                rad = r if k == own else r_other   # a label must not read as another dot's
                cx = min(max(qx, bb.x0), bb.x1); cy = min(max(qy, bb.y0), bb.y1)
                if (cx - qx) ** 2 + (cy - qy) ** 2 < rad * rad:
                    return False
        if avoid_line and line_px is not None and np.any(
                (line_px[:, 0] > bb.x0 - 2) & (line_px[:, 0] < bb.x1 + 2)
                & (line_px[:, 1] > bb.y0 - 2) & (line_px[:, 1] < bb.y1 + 2)):
            return False
        return True

    farther = [(dx * 1.7, dy * 1.7) for dx, dy in far]
    placed = []
    for i in np.argsort(-np.asarray(ys)):
        crowded = nearest[i] < crowd
        rings = ([] if crowded else [(near, False)]) + [(far, True), (farther, True)]
        if crowded:
            # Try offsets on the side facing away from the neighbouring dot first,
            # so the two leader lines of a crowded pair can never cross.
            j = min((k for k in range(len(pts)) if k != i),
                    key=lambda k: np.hypot(*(pts[i] - pts[k])))
            v = pts[i] - pts[j]
            away = lambda ring: sorted(ring, key=lambda o: -(o[0] * v[0] + o[1] * v[1]))
            rings = [(away(rg), ld) for rg, ld in rings]
        # Every label is placed. Rules relax in order only when nothing passes:
        # first the fitted line may be crossed (the white halo keeps the text
        # legible), then other dots; labels themselves never overlap.
        for avoid_line, avoid_dots in ((True, True), (False, True), (False, False)):
            done = False
            for ring, leader in rings:
                for ox, oy in ring:
                    ha = "left" if ox > 0 else "right" if ox < 0 else "center"
                    va = "bottom" if oy > 0 else "top" if oy < 0 else "center"
                    kw = dict(arrowprops=dict(arrowstyle="-", lw=0.6, color="#8B9199",
                                              shrinkA=0, shrinkB=r / px_per_pt)) if leader else {}
                    t = ax.annotate(labels[i], (xs[i], ys[i]), xytext=(ox, oy),
                                    textcoords="offset points", ha=ha, va=va,
                                    fontsize=7.5, color="#30343B", zorder=4, **kw)
                    t.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])
                    # Annotation.get_window_extent includes the leader line, which
                    # always touches the label's own dot; measure the text alone and
                    # test the leader separately.
                    t.update_positions(rend)            # resolve the offset-points position
                    bb = Text.get_window_extent(t, rend).expanded(1.06, 1.12)
                    ok = clear(bb, i, avoid_dots, avoid_line)
                    if ok and leader and avoid_dots:
                        tx, ty = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
                        seg = np.column_stack([np.linspace(tx, pts[i][0], 24),
                                               np.linspace(ty, pts[i][1], 24)])
                        # ignore the stretch next to the label's own dot, which a
                        # near-coincident neighbour necessarily shares
                        seg = seg[np.hypot(*(seg - pts[i]).T) > 2.5 * r]
                        for k, q in enumerate(pts):
                            if k != i and np.min(np.hypot(*(seg - q).T)) < r:
                                ok = False                 # leader would cross another dot
                        if any(np.any((seg[:, 0] > b.x0) & (seg[:, 0] < b.x1) &
                                      (seg[:, 1] > b.y0) & (seg[:, 1] < b.y1)) for b in placed):
                            ok = False                     # or run through another label
                    if ok:
                        placed.append(bb); done = True
                        break
                    t.remove()
                if done:
                    break
            if done:
                break
        else:
            raise RuntimeError(f"could not place label {labels[i]}")


def fig2(ref, D, out):
    # (task, data key, display name, y label) -- the data key is the string the
    # results CSV uses; the display name must match the model names in the tables.
    panels = [("classification", "RandForest", "Random Forest",
               "Weather-added gain in AUC"),
              ("regression", "GradientBoosting", "Gradient Boosting",
               "Weather-added gain in $R^2$")]
    # Stacked rather than side by side: each panel gets the full text width,
    # which roughly doubles the horizontal room between country labels.
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.6))
    for ax, (task, model, label, ylab) in zip(axes, panels):
        d = (ref[(ref.era == "post") & (ref.task == task) & (ref.model == model)]
             .merge(D[D.era == "post"], on="code")
             .dropna(subset=["gain", "D", "native_points"]))
        ax.scatter(d.D, d.gain, s=46, color=BOTH, edgecolor="white", lw=0.8, zorder=3)
        b, a = np.polyfit(d.D, d.gain, 1)
        xs = np.linspace(d.D.min() - 2, d.D.max() + 2, 100)
        ax.plot(xs, a + b * xs, color="#30343B", lw=1.6, zorder=2)
        _label_points(ax, d.D.values, d.gain.values,
                      [c.upper() for c in d.code.values], line=(xs, a + b * xs))

        z = np.log(d.native_points.values)
        eD = d.D.values - np.polyval(np.polyfit(z, d.D.values, 1), z)
        eg = d.gain.values - np.polyval(np.polyfit(z, d.gain.values, 1), z)
        rp = np.corrcoef(eD, eg)[0, 1]
        ax.text(0.04, 0.96, f"r = {np.corrcoef(d.D, d.gain)[0,1]:.3f}\n"
                            f"partial r (log N) = {rp:.3f}\nn = {len(d)}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CBD2D9"))
        ax.axhline(0, color="#8B9199", lw=0.9)
        ax.axvline(0, color="#8B9199", lw=0.9)
        ax.set_xlabel("Wind dominance $D$ (wind − solar share of load, pp)")
        ax.set_ylabel(ylab)
        ax.set_title(label, fontsize=11)
        ax.grid(color="#EDF0F2", lw=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/fig_results_winddominance.{ext}", dpi=300)
    plt.close(fig)


def fig3(out):
    R = pd.read_csv(f"{COMP}/resolution_summary_all_eras.csv")
    cap = R[R.scheme == "capacity"].copy()
    base = (cap[cap.resolution_deg == 0.25]
            .set_index(["era", "task", "model"]).total_point_hours.rename("H0"))
    cap = cap.join(base, on=["era", "task", "model"])
    cap["red"] = 100 * (1 - cap.total_point_hours / cap.H0)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    marks = {"pre": "o", "post": "^"}
    pts = []
    cols = {"pre": "#0F766E", "post": "#2C5F7C"}
    for era, title in ERAS:
        g = (cap[(cap.era == era) & (cap.resolution_deg != 0.25)]
             .groupby("resolution_deg")
             .agg(drift=("median_absolute_gain_drift", "median"),
                  red=("red", "first")).reset_index().sort_values("red"))
        ax.plot(g.red, g.drift, marker=marks[era], color=cols[era], lw=1.8,
                ms=7, label=title)
        pts.append(g)
    # Both eras share the 1 degree point (93.0%, 0.0045), so per-era labels
    # collide there. Label each resolution once, below its lowest marker.
    ax.set_ylim(bottom=0)        # drift is a magnitude; also leaves room for the 0.5 label
    allp = pd.concat(pts, ignore_index=True)
    for res, g in allp.groupby("resolution_deg"):
        low = g.loc[g.drift.idxmin()]
        t = ax.annotate(f"{res:g}°", (low.red, low.drift), xytext=(0, -11),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=9, color="#4A4F57")
        t.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])
    ax.set_xlabel("Workload reduction vs 0.25° (%, grid-point hours)")
    ax.set_ylabel("Median predictive drift $d_r$")
    ax.grid(color="#EDF0F2", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/fig_results_drift_workload.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures/paper")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ref = load()
    D = []
    n = pd.read_csv(f"{COMP}/native_grid_cells.csv")   # identical in both eras
    for era in ("pre", "post"):
        cs = pd.read_csv(f"{COMP}/country_summary_{era}.csv").drop_duplicates("code", keep="last")
        cs["D"] = cs.actual_wind_share_pct - cs.actual_solar_share_pct
        cs["era"] = era
        D.append(cs[["era", "code", "D"]].merge(n, on="code", how="left"))
    D = pd.concat(D)
    fig1(ref, args.out_dir); fig2(ref, D, args.out_dir); fig3(args.out_dir)
    print("wrote figures to", args.out_dir)
