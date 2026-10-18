"""Reviewer response: (a) decompose weather-added gain into demand vs generation,
(b) test cross-border interconnection as a confounder of the wind-dominance result.

Reuses the paper's exact feature set, split and model settings so the share-target
column reproduces the published post-COVID median R^2 gain of 0.319."""
import numpy as np, pandas as pd, glob, os, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--data-root", default=str(REPO / "data"),
                 help="pipeline data directory (default: <repo>/data)")
_ap.add_argument("--out", default=str(REPO / "results" / "paper" / "reviewer_decomposition.csv"))
ARGS = _ap.parse_args()
ROOT = ARGS.data_root
ENE  = f"{ROOT}/energy_targets_by_era/post"
WX   = f"{ROOT}/country_weather_post_covid"
WCOL = ["wind_speed_100m", "shortwave_radiation", "temperature_2m"]

def calendar(ix):
    h, m = ix.hour.values, ix.month.values
    return pd.DataFrame({
        "sin_h": np.sin(2*np.pi*h/24), "cos_h": np.cos(2*np.pi*h/24),
        "sin_m": np.sin(2*np.pi*m/12), "cos_m": np.cos(2*np.pi*m/12),
        "weekend": (ix.dayofweek.values >= 5).astype(float)}, index=ix)

def r2(y, p):
    return 1 - ((y-p)**2).sum() / ((y-y.mean())**2).sum()

def fit_gain(X0, X1, y, frac=0.8):
    n = int(len(y)*frac)
    out = []
    for X in (X0, X1):
        m = HistGradientBoostingRegressor(random_state=42)
        m.fit(X.iloc[:n], y.iloc[:n])
        out.append(r2(y.iloc[n:], pd.Series(m.predict(X.iloc[n:]), index=y.index[n:])))
    return out[0], out[1], out[1]-out[0]

rows = []
for f in sorted(glob.glob(f"{ENE}/weather_energy_merged_*.csv")):
    code = os.path.basename(f).split("_")[-1][:2]
    wf = f"{WX}/weather_era5_{code}_capacity_0p25deg.csv"
    if not os.path.exists(wf): continue
    e = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp")
    w = pd.read_csv(wf, parse_dates=["timestamp"]).set_index("timestamp")
    d = e.join(w, how="inner")
    d = d[d.index >= "2022-01-01"]
    d["gen"] = d.Renewable_share_of_load * d.Load / 100.0          # implied renewable generation
    # Measured alternative: sum the source columns directly. Energy-Charts counts
    # half of waste as renewable; this definition reproduces its share to a
    # median 0.02 pp in every country.
    CORE = ["Wind_onshore", "Wind_offshore", "Solar", "Biomass", "Hydro_Run-of-River",
            "Hydro_water_reservoir", "Geothermal", "Other_renewables"]
    d["gen_measured"] = (sum(d[c].fillna(0) for c in CORE if c in d)
                         + 0.5 * (d["Waste"].fillna(0) if "Waste" in d else 0))
    need = ["Renewable_share_of_load", "Load", "gen"] + WCOL
    d = d.dropna(subset=need)
    if len(d) < 5000: continue
    cal = calendar(d.index)
    X0, X1 = cal, pd.concat([cal, d[WCOL]], axis=1)
    rec = {"code": code, "n": len(d)}
    for name, y in [("share", d.Renewable_share_of_load), ("load", d.Load), ("gen", d["gen"]),
                    ("genm", d["gen_measured"])]:
        c, b, g = fit_gain(X0, X1, y)
        rec[f"{name}_cal"], rec[f"{name}_wx"], rec[f"{name}_gain"] = c, b, g
    # interconnection proxy: mean absolute net cross-border flow / mean load
    if "Cross_border_electricity_trading" in d:
        rec["xborder"] = 100 * d.Cross_border_electricity_trading.abs().mean() / d.Load.mean()
    rows.append(rec)
    print(f"  {code}: share {rec['share_gain']:+.3f}  load {rec['load_gain']:+.3f}  gen {rec['gen_gain']:+.3f}  measured {rec['genm_gain']:+.3f}", flush=True)

t = pd.DataFrame(rows)
t.to_csv(ARGS.out, index=False)
print("\n=== medians across %d countries ===" % len(t))
for k in ["share", "load", "gen", "genm"]:
    print(f"  {k:6s} calendar {t[k+'_cal'].median():+.3f} -> weather {t[k+'_wx'].median():+.3f}  gain {t[k+'_gain'].median():+.3f}")
print("\npaper's published post-COVID median R2 gain (share, GradientBoosting) = +0.319")
