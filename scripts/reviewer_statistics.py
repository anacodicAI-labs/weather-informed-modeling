#!/usr/bin/env python3
"""Statistics added to the manuscript during peer review.

Every number below is computed from the curated tables in results/paper/, so the
script runs from a clean clone. The one exception is the retired-capacity count,
which needs the raw GEM workbooks and the 2019 capacity maps produced by the
pipeline; pass --data-root to include it.

Outputs (results/paper/):
    reviewer_capacity_tests.csv        capacity vs uniform, per era and primary model
    reviewer_era_matched_correlation.csv  wind dominance vs gain on matched countries
    reviewer_resolution_percountry.csv 0.25 vs 1 degree gain, per country (Table 7)
    reviewer_drift_drivers.csv         which country/model sets each maximum drift
    reviewer_test_windows.csv          length of each held-out test interval
    reviewer_unknown_start_all19.csv   unknown-start sensitivity across all 19 countries

    python scripts/reviewer_statistics.py [--data-root data]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "paper"
PRIMARY = {"classification": "RandForest", "regression": "GradientBoosting"}
KEY = ["era", "code", "task", "model", "resolution_deg", "scheme"]


def load_models():
    m = pd.read_csv(RES / "model_results_all_eras.csv")
    m = m[(m.capacity_layout == "annual") & m.era.isin(["pre", "post"])]
    return m.drop_duplicates(subset=KEY, keep="last")


def wind_dominance(era):
    cs = pd.read_csv(RES / f"country_summary_{era}.csv").drop_duplicates("code", keep="last")
    cs = cs.set_index("code")
    return cs.actual_wind_share_pct, cs.actual_solar_share_pct


def gains(m, era, task, model, res=0.25, scheme="capacity"):
    s = m[(m.era == era) & (m.task == task) & (m.model == model)
          & (m.resolution_deg == res) & (m.scheme == scheme)]
    return s.set_index("code").gain


def partial_r(x, y, z):
    ex = x - np.polyval(np.polyfit(z, x, 1), z)
    ey = y - np.polyval(np.polyfit(z, y, 1), z)
    return np.corrcoef(ex, ey)[0, 1]


def capacity_tests(m):
    """One paired difference per country, per era and primary model (Sec. 4.4)."""
    rows = []
    for era in ("pre", "post"):
        for task, model in PRIMARY.items():
            s = m[(m.era == era) & (m.model == model) & (m.resolution_deg == 0.25)]
            p = s.pivot(index="code", columns="scheme", values="both_score").dropna()
            d = p["capacity"] - p["uniform"]
            w = wilcoxon(d)
            rows.append(dict(era=era, model=model, n=len(d), capacity_wins=int((d > 0).sum()),
                             mean_diff=d.mean(), W=w.statistic, p=w.pvalue))
    t = pd.DataFrame(rows)
    t.to_csv(RES / "reviewer_capacity_tests.csv", index=False)
    print("== capacity vs uniform, per era and primary model ==")
    print(t.round(5).to_string(index=False))


def era_comparison(m):
    """Wind dominance vs gain, both eras restricted to the same countries (Sec. 4.3)."""
    wpre, spre = wind_dominance("pre")
    wpost, spost = wind_dominance("post")
    Dpre, Dpost = (wpre - spre).rename("D_pre"), (wpost - spost).rename("D_post")
    rng = np.random.default_rng(42)          # one stream, RandForest then GradientBoosting
    rows = []
    for task, model in PRIMARY.items():
        t = pd.concat([gains(m, "pre", task, model).rename("g_pre"),
                       gains(m, "post", task, model).rename("g_post"), Dpre, Dpost], axis=1).dropna()
        t = t.sort_index()                   # fixed order, so the bootstrap is reproducible
        r_pre = np.corrcoef(t.D_pre, t.g_pre)[0, 1]
        r_post = np.corrcoef(t.D_post, t.g_post)[0, 1]
        diffs = []
        for _ in range(5000):
            s = t.iloc[rng.integers(0, len(t), len(t))]
            if s.D_pre.std() == 0 or s.g_pre.std() == 0:
                continue
            diffs.append(np.corrcoef(s.D_post, s.g_post)[0, 1] - np.corrcoef(s.D_pre, s.g_pre)[0, 1])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append(dict(model=model, n=len(t), countries=" ".join(sorted(t.index)),
                         r_pre=r_pre, r_post=r_post, diff=r_post - r_pre, ci_low=lo, ci_high=hi))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "reviewer_era_matched_correlation.csv", index=False)
    print("\n== era comparison on matched countries (country bootstrap, 5000) ==")
    print(out.drop(columns="countries").round(3).to_string(index=False))

    print("\n== post-COVID correlation: Fisher-z interval, and Ireland/Serbia ==")
    n_nat = pd.read_csv(RES / "native_grid_cells.csv").set_index("code").native_points
    for task, model in PRIMARY.items():
        g = gains(m, "post", task, model)
        d = pd.concat([g.rename("g"), Dpost.rename("D"), n_nat.rename("N")], axis=1).dropna()
        r = np.corrcoef(d.D, d.g)[0, 1]
        z, se = np.arctanh(r), 1 / np.sqrt(len(d) - 3)
        pr = partial_r(d.D.values, d.g.values, np.log(d.N.values))
        D0 = (wpost - spost.fillna(0)).rename("D")          # unreported solar set to zero
        d0 = pd.concat([g.rename("g"), D0], axis=1).dropna()
        r0 = np.corrcoef(d0.D, d0.g)[0, 1]
        print(f"  {model:17s} n={len(d)} r={r:.3f}  Fisher 95% [{np.tanh(z-1.96*se):.3f}, "
              f"{np.tanh(z+1.96*se):.3f}]  partial r(log N)={pr:.3f}  "
              f"with IE+RS (solar=0, n={len(d0)}) r={r0:.3f}")


def interconnection():
    """Uses the output of reviewer_decomposition.py (Sec. 4.3)."""
    f = RES / "reviewer_decomposition.csv"
    if not f.exists():
        print("\n(skip interconnection: run reviewer_decomposition.py first)")
        return
    t = pd.read_csv(f).set_index("code")
    w, s = wind_dominance("post")
    d = t.join((w - s).rename("D")).dropna(subset=["D", "xborder", "share_gain"])
    print("\n== interconnection (gradient boosting, post-COVID, n=%d) ==" % len(d))
    print(f"  proxy range {t.xborder.min():.1f}%-{t.xborder.max():.1f}% (median {t.xborder.median():.1f}%)")
    print(f"  r(D, gain) = {np.corrcoef(d.D, d.share_gain)[0,1]:+.3f}   "
          f"r(D, gain | interconnection) = {partial_r(d.D.values, d.share_gain.values, d.xborder.values):+.3f}")
    print(f"  r(interconnection, gain) = {np.corrcoef(d.xborder, d.share_gain)[0,1]:+.3f}   "
          f"r(interconnection, D) = {np.corrcoef(d.xborder, d.D)[0,1]:+.3f}")
    print("\n== decomposition medians (19 countries) ==")
    for k in ("load", "gen", "genm", "share"):
        if f"{k}_gain" in t:
            print(f"  {k:6s} calendar {t[k+'_cal'].median():.3f} -> weather {t[k+'_wx'].median():.3f}"
                  f"  gain {t[k+'_gain'].median():+.3f}")


def resolution_table(m):
    """Per-country gain at 0.25 vs 1 degree, post-COVID primary models (Table 7)."""
    parts = {}
    for task, model in PRIMARY.items():
        lab = "AUC" if task == "classification" else "R2"
        a, b = gains(m, "post", task, model, 0.25), gains(m, "post", task, model, 1.0)
        parts[lab] = pd.DataFrame({f"{lab}_g025": a, f"{lab}_g1": b})
        parts[lab][f"{lab}_d"] = (parts[lab][f"{lab}_g1"] - parts[lab][f"{lab}_g025"]).abs()
    t = parts["AUC"].join(parts["R2"])
    t.index = t.index.str.upper()
    t.sort_index().round(4).to_csv(RES / "reviewer_resolution_percountry.csv")
    print("\n== 0.25 vs 1 degree (post-COVID) ==")
    print(f"  median |d| AUC {t.AUC_d.median():.4f}, R2 {t.R2_d.median():.4f}; sign flips "
          f"AUC {int(((t.AUC_g025 > 0) != (t.AUC_g1 > 0)).sum())}, "
          f"R2 {int(((t.R2_g025 > 0) != (t.R2_g1 > 0)).sum())}")


def drift_drivers(m):
    """Which country and model set each maximum drift in Table 6."""
    rows = []
    c = m[m.scheme == "capacity"]
    for era in ("pre", "post"):
        ref = c[(c.era == era) & (c.resolution_deg == 0.25)].set_index(["code", "task", "model"]).gain
        for res in (0.5, 1.0, 2.0):
            cur = c[(c.era == era) & (c.resolution_deg == res)].set_index(["code", "task", "model"]).gain
            dr = (cur - ref).abs().dropna()
            code, task, model = dr.idxmax()
            rows.append(dict(era=era, resolution_deg=res, max_drift=dr.max(), code=code, model=model))
    t = pd.DataFrame(rows)
    t.to_csv(RES / "reviewer_drift_drivers.csv", index=False)
    print("\n== drift maxima (all four models, as in Table 6; medians are in resolution_summary_all_eras.csv) ==")
    print(t.round(4).to_string(index=False))


def test_windows(m):
    s = m[(m.resolution_deg == 0.25) & (m.scheme == "capacity")].drop_duplicates(["era", "code"])
    a = pd.to_datetime(s.test_start, utc=True)
    b = pd.to_datetime(s.test_end, utc=True)
    t = pd.DataFrame({"era": s.era.values, "code": s.code.values, "start": a.dt.date.values,
                      "end": b.dt.date.values, "days": (b - a).dt.days.values})
    t["calendar_months"] = [len(pd.period_range(x, y, freq="M")) for x, y in zip(a, b)]
    t.sort_values(["era", "days"]).to_csv(RES / "reviewer_test_windows.csv", index=False)
    print("\n== held-out test windows ==")
    for era, g in t.groupby("era"):
        print(f"  {era}: {g.days.min()}-{g.days.max()} days (median {int(g.days.median())}), "
              f"{g.calendar_months.min()}-{g.calendar_months.max()} calendar months; "
              f"shortest {g.loc[g.days.idxmin(), 'code'].upper()}")


def distributed_solar():
    cov = pd.read_csv(RES / "gem_capacity_coverage_post.csv")
    c = cov[cov.unknown_start_policy == "include"].sort_values("year").groupby("code").last()
    c = c.dropna(subset=["distributed_solar_reference_mw_latest"])
    frac = 100 * c.distributed_solar_reference_mw_latest / (
        c.distributed_solar_reference_mw_latest + c.solar_weight_mw)
    w, s = wind_dominance("post")
    solar_dom = sorted((w - s)[(w - s) < 0].index)
    print("\n== distributed (unmapped) solar ==")
    print(f"  {len(frac)} countries publish a total: median {frac.median():.0f}% of solar capacity, "
          f"range {frac.min():.0f}% ({frac.idxmin().upper()}) to {frac.max():.0f}% ({frac.idxmax().upper()})")
    print(f"  solar-dominant (D<0): {', '.join(x.upper() for x in solar_dom)}; "
          f"with a published total: {', '.join(x.upper() for x in solar_dom if x in frac.index) or 'none'}")


def unknown_start(m):
    """Sensitivity to unknown-start facilities across all 19 countries (Sec. 4.6)."""
    orig = pd.read_csv(RES / "unknown_start_model_comparison.csv")
    ex = pd.read_csv(RES / "unknown_start_expansion_model_results.csv")
    k = ["code", "task", "model"]
    ex = ex[(ex.resolution_deg == 1.0) & (ex.scheme == "capacity")].drop_duplicates(k, keep="last")
    inc = m[(m.era == "post") & (m.resolution_deg == 1.0) & (m.scheme == "capacity")]
    j = ex.set_index(k).gain.rename("gain_exclude").to_frame().join(
        inc.set_index(k).gain.rename("gain_include")).dropna().reset_index()
    j["abs_gain_delta"] = (j.gain_exclude - j.gain_include).abs()
    j["batch"] = "second batch (6)"
    o = orig[k + ["abs_gain_delta"]].assign(batch="first batch (13)")
    t = pd.concat([o, j[k + ["abs_gain_delta", "batch"]]], ignore_index=True)
    t.to_csv(RES / "reviewer_unknown_start_all19.csv", index=False)
    six = t[t.batch == "second batch (6)"]
    worst = six.loc[six.abs_gain_delta.idxmax()]
    print("\n== unknown-start sensitivity, all 19 countries at 1 degree ==")
    print(f"  {len(t)} comparisons ({t.code.nunique()} countries): median {t.abs_gain_delta.median():.5f}, "
          f"max {t.abs_gain_delta.max():.4f}; second batch max {worst.abs_gain_delta:.4f} "
          f"({worst.code.upper()} {worst.model})")


def without_croatia(m):
    """Croatia's pre-COVID series covers 2019 only; the era comparisons without it."""
    w, sp = wind_dominance("pre"); wq, sq = wind_dominance("post")
    Dpre, Dpost = (w - sp).rename("dp"), (wq - sq).rename("dq")
    rng = np.random.default_rng(42)
    print("\n== era comparisons with Croatia omitted ==")
    for task, model in PRIMARY.items():
        g = pd.concat([gains(m, "pre", task, model).rename("gp"),
                       gains(m, "post", task, model).rename("gq")], axis=1).dropna().drop("hr")
        p = wilcoxon(g.gq - g.gp).pvalue
        t = g.join(Dpre).join(Dpost).dropna().sort_index()
        diffs = []
        for _ in range(5000):
            s_ = t.iloc[rng.integers(0, len(t), len(t))]
            if s_.dp.std() == 0 or s_.gp.std() == 0:
                continue
            diffs.append(np.corrcoef(s_.dq, s_.gq)[0, 1] - np.corrcoef(s_.dp, s_.gp)[0, 1])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"  {model:17s} paired Wilcoxon p={p:.3f}; matched r {np.corrcoef(t.dp, t.gp)[0,1]:.3f}"
              f" -> {np.corrcoef(t.dq, t.gq)[0,1]:.3f} (n={len(t)}), increase 95% [{lo:+.3f}, {hi:+.3f}]")


def lithuania_and_exceptions(m):
    pre = gains(m, "pre", "regression", "GradientBoosting")
    s = m[(m.task == "regression") & (m.model == "GradientBoosting") &
          (m.resolution_deg == 0.25) & (m.scheme == "capacity")]
    cw = s.pivot(index="code", columns="era", values="both_score")
    drop = (cw.post - cw.pre).sort_values()
    print("\n== calendar+weather R2 change between eras (gradient boosting) ==")
    print("  largest falls:", ", ".join(f"{c.upper()} {cw.pre[c]:.3f}->{cw.post[c]:.3f}"
                                         for c in drop.index[:3]))
    print("\n== post-COVID countries where weather did not help ==")
    for task, model in PRIMARY.items():
        g = gains(m, "post", task, model)
        neg = g[g < 0].sort_values()
        print(f"  {model:17s}", ", ".join(f"{c.upper()} {v:+.3f}" for c, v in neg.items()))


def retired_capacity(data_root):
    """Wind/solar phases GEM lists as retired since 2016, absent from pre-COVID maps."""
    if data_root is None:
        print("\n(skip retired capacity: pass --data-root to include it)")
        return
    root = Path(data_root)
    countries = {"Austria", "Belgium", "Bulgaria", "Croatia", "Czech Republic", "Czechia", "Denmark",
                 "France", "Germany", "Greece", "Ireland", "Latvia", "Lithuania", "Netherlands",
                 "Portugal", "Romania", "Serbia", "Slovakia", "Slovenia", "Spain"}
    codes = {"Austria": "at", "Germany": "de", "Netherlands": "nl", "Spain": "es",
             "Denmark": "dk", "Ireland": "ie"}
    print("\n== retired or mothballed phases since 2016 (absent from pre-COVID weights) ==")
    for kind in ("Wind", "Solar"):
        xl = next((root / "raw" / "gem").glob(f"Global-{kind}-Power-Tracker-*.xlsx"))
        sheets = pd.read_excel(xl, sheet_name=None)
        d = next(v for v in sheets.values() if "Status" in v.columns)
        cc = next(c for c in d.columns if c.lower() in ("country/area", "country"))
        st = next(c for c in d.columns if c.lower() == "status")
        cap = next(c for c in d.columns if "capacity" in c.lower() and "mw" in c.lower())
        ry = next(c for c in d.columns if "retired year" in c.lower() or "retirement year" in c.lower())
        d = d[d[cc].isin(countries) & d[st].astype(str).str.lower().isin(["retired", "mothballed"])]
        d = d[pd.to_numeric(d[ry], errors="coerce") >= 2016]
        print(f"  {kind}: {len(d)} phases, {d[cap].sum():,.0f} MW")
        if kind == "Wind":
            by = d.groupby(cc)[cap].sum()
            for name, code in codes.items():
                fleet = pd.read_csv(root / "capacity_weights_pre_and_covid_by_year" / "2019" / f"{code}.csv").wind_mw.sum()
                print(f"    {code.upper()}: {by.get(name, 0):,.0f} MW = {100 * by.get(name, 0) / fleet:.1f}% "
                      f"of mapped 2019 wind ({fleet:,.0f} MW)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None,
                    help="pipeline data directory; enables the retired-capacity count")
    args = ap.parse_args()
    m = load_models()
    capacity_tests(m)
    era_comparison(m)
    interconnection()
    resolution_table(m)
    drift_drivers(m)
    test_windows(m)
    distributed_solar()
    unknown_start(m)
    lithuania_and_exceptions(m)
    without_croatia(m)
    retired_capacity(args.data_root)
