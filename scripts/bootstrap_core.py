"""
bootstrap_core.py  --  block-bootstrap significance for the weather-added gain.

WHAT IT DOES
------------
For a set of (country, model) work items assigned to this SGE array task,
estimates a confidence interval on the weather-added skill (BOTH minus
CALENDAR) by BLOCK bootstrap:

  * blocks = ISO weeks (respects hourly autocorrelation -- resampling
    individual hours would pretend adjacent hours are independent and give
    falsely narrow intervals)
  * each resample: draw weeks of the TRAINING period with replacement,
    refit CALENDAR and BOTH models, evaluate the gain on the FIXED test set
  * repeat n_resamples times -> distribution of the gain -> CI

Refitting inside the loop is deliberate: it makes the interval reflect real
training variability AND makes the workload genuinely CPU-bound (the reason
this belongs on the cluster rather than a laptop).

Feature engineering, the >50% target, and the 80/20 chronological split are
replicated EXACTLY from run_country_pipeline.py so the bootstrap gain matches
the pipeline's point estimate.

Reads its assignment from a manifest file (built by build_manifest.py) keyed
to $SGE_TASK_ID, so the SGE array is load-balanced rather than one-country-
per-task. Writes one partial-results file + one timing record per task.

Usage (normally invoked by the .qsub array script):
    python bootstrap_core.py --task-id $SGE_TASK_ID \
        --manifest manifest --out partials --data ../data
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# --- must match run_country_pipeline.py exactly --------------------------
FEATURE_SETS = {
    "CALENDAR": ["hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend"],
    "WEATHER": ["wind_speed_100m", "shortwave_radiation", "temperature_2m"],
    "BOTH": ["wind_speed_100m", "shortwave_radiation", "temperature_2m",
             "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend"],
}
SHARE_COL = "Renewable_share_of_load"
SPLIT_FRACTION = 0.8
THRESHOLD = 50


def build_features(merged):
    """Replicated from run_country_pipeline.build_features (identical logic)."""
    df = pd.DataFrame(index=merged.index)
    df["wind_speed_100m"] = merged["wind_speed_100m"]
    df["shortwave_radiation"] = merged["shortwave_radiation"]
    df["temperature_2m"] = merged["temperature_2m"]
    df["hour_sin"] = np.sin(2 * np.pi * merged.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * merged.index.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * merged.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * merged.index.month / 12)
    df["is_weekend"] = (merged.index.dayofweek >= 5).astype(int)
    df["y"] = (merged[SHARE_COL] > THRESHOLD).astype(int)
    return df.dropna().sort_index()


def load_country(code, data_dir):
    path = os.path.join(data_dir, f"weather_energy_merged_{code}.csv")
    merged = pd.read_csv(path, index_col=0, parse_dates=True)
    df = build_features(merged)
    split = int(len(df) * SPLIT_FRACTION)
    train, test = df.iloc[:split], df.iloc[split:]
    # label each TRAIN row by ISO (year, week) so we can resample whole weeks
    iso = train.index.isocalendar()
    train = train.copy()
    train["_block"] = list(zip(iso.year, iso.week))
    return train, test


def make_model(name):
    if name == "LogReg":
        return ("scale", LogisticRegression(max_iter=1000))
    if name == "RandForest":
        return ("raw", RandomForestClassifier(n_estimators=300, random_state=42,
                                              n_jobs=1))
    raise ValueError(f"unknown model {name}")


def fit_auc(model_kind, model, train, test, feats):
    Xtr, ytr = train[feats].values, train["y"].values
    Xte, yte = test[feats].values, test["y"].values
    if len(np.unique(yte)) < 2:      # AUC undefined on single-class test
        return np.nan
    if model_kind == "scale":
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    model.fit(Xtr, ytr)
    return roc_auc_score(yte, model.predict_proba(Xte)[:, 1])


def one_resample(train, test, model_name, blocks, rng):
    """Draw whole weeks with replacement -> refit -> gain (BOTH minus CALENDAR)."""
    chosen = rng.choice(len(blocks), size=len(blocks), replace=True)
    picked = [blocks[i] for i in chosen]
    boot = pd.concat([train[train["_block"] == b] for b in picked])
    kind, _ = make_model(model_name)
    _, m_cal = make_model(model_name)
    _, m_both = make_model(model_name)
    auc_cal = fit_auc(kind, m_cal, boot, test, FEATURE_SETS["CALENDAR"])
    auc_both = fit_auc(kind, m_both, boot, test, FEATURE_SETS["BOTH"])
    return auc_both - auc_cal


def run_item(item, data_dir):
    """item = {country, code, model, n_resamples, seed}."""
    train, test = load_country(item["code"], data_dir)
    blocks = list(dict.fromkeys(train["_block"]))   # unique weeks, order-preserving
    rng = np.random.default_rng(item["seed"])
    gains = np.array([one_resample(train, test, item["model"], blocks, rng)
                      for _ in range(item["n_resamples"])])
    gains = gains[~np.isnan(gains)]
    return {
        "country": item["country"], "model": item["model"],
        "n_resamples": int(len(gains)),
        "gain_mean": float(np.mean(gains)),
        "gain_lo95": float(np.percentile(gains, 2.5)),
        "gain_hi95": float(np.percentile(gains, 97.5)),
        "excludes_zero": bool(np.percentile(gains, 2.5) > 0),
        "gains": gains.tolist(),          # kept so the cross-country CI can be paired
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--manifest", default="manifest")
    ap.add_argument("--out", default="partials")
    ap.add_argument("--data", default="../data")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.manifest, f"task_{args.task_id}.json")
    with open(manifest_path) as f:
        items = json.load(f)

    t0 = time.time()
    results = []
    for item in items:
        ti = time.time()
        r = run_item(item, args.data)
        r["wall_s"] = round(time.time() - ti, 2)
        results.append(r)
        print(f"  task {args.task_id}: {item['country']}/{item['model']} "
              f"n={r['n_resamples']} gain={r['gain_mean']:+.3f} "
              f"[{r['gain_lo95']:+.3f},{r['gain_hi95']:+.3f}] "
              f"{'*' if r['excludes_zero'] else ''} ({r['wall_s']}s)")

    total = round(time.time() - t0, 2)
    out = {"task_id": args.task_id, "task_wall_s": total,
           "n_items": len(items), "results": results}
    with open(os.path.join(args.out, f"task_{args.task_id}.json"), "w") as f:
        json.dump(out, f)
    print(f"task {args.task_id} done: {len(items)} items in {total}s")


if __name__ == "__main__":
    main()
