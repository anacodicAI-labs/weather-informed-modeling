"""
build_manifest.py  --  split the bootstrap workload into LOAD-BALANCED bundles,
one per SGE array task.

WHY THIS EXISTS (the HPC contribution)
--------------------------------------
The work items are (country x model) bootstrap jobs. Their cost varies a lot:
  * countries differ ~4x in data volume (Netherlands ~150k rows vs ~38k)
  * models differ ~10x+ in per-fit cost (RandomForest n=300 >> LogReg)
so assigning one country per task, or round-robin, leaves some tasks finishing
in minutes while others run for an hour -- the cluster sits idle and makespan
(the thing you actually wait for) is set by the single slowest task.

This script estimates each item's cost and uses Longest-Processing-Time-first
(LPT) greedy bin-packing to balance total estimated cost across N tasks. The
--naive flag instead does round-robin, so you can measure the makespan/
efficiency improvement for the poster (balanced vs naive is the money figure).

    python build_manifest.py --tasks 32 --resamples 2000
    python build_manifest.py --tasks 32 --resamples 2000 --naive   # baseline
"""

import os
import json
import argparse
import pandas as pd

# keep in sync with the pipeline
COUNTRIES = {
    "Denmark": "dk", "Ireland": "ie", "Netherlands": "nl", "Portugal": "pt",
    "Greece": "gr", "Belgium": "be", "Lithuania": "lt", "Croatia": "hr",
    "Bulgaria": "bg", "Latvia": "lv", "Slovenia": "si", "Serbia": "rs",
    "Slovakia": "sk",
}
MODELS = ["LogReg", "RandForest"]
# relative per-fit cost weights (RandomForest 300 trees is much heavier)
MODEL_COST = {"LogReg": 1.0, "RandForest": 12.0}


def country_rows(code, data_dir):
    """Cheap row count to estimate per-fit cost; falls back to a constant."""
    path = os.path.join(data_dir, f"weather_energy_merged_{code}.csv")
    try:
        # count lines without loading the whole frame
        with open(path) as f:
            return sum(1 for _ in f) - 1
    except FileNotFoundError:
        return 38000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, required=True)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--data", default="../data")
    ap.add_argument("--manifest", default="manifest")
    ap.add_argument("--naive", action="store_true",
                    help="round-robin instead of balanced (baseline for scaling study)")
    args = ap.parse_args()
    os.makedirs(args.manifest, exist_ok=True)

    # one work item per (country, model); each carries the FULL resample count
    items = []
    for i, (country, code) in enumerate(COUNTRIES.items()):
        rows = country_rows(code, args.data)
        for model in MODELS:
            cost = rows * MODEL_COST[model] * args.resamples
            items.append({"country": country, "code": code, "model": model,
                          "n_resamples": args.resamples,
                          "seed": 1000 * i + hash(model) % 997,
                          "_cost": cost})

    # assign items to tasks
    buckets = [[] for _ in range(args.tasks)]
    loads = [0.0] * args.tasks
    if args.naive:
        for j, it in enumerate(items):
            buckets[j % args.tasks].append(it)
            loads[j % args.tasks] += it["_cost"]
    else:  # LPT: heaviest first, always to the currently-lightest task
        for it in sorted(items, key=lambda x: -x["_cost"]):
            k = min(range(args.tasks), key=lambda t: loads[t])
            buckets[k].append(it)
            loads[k] += it["_cost"]

    for t in range(args.tasks):
        for it in buckets[t]:
            it.pop("_cost", None)
        with open(os.path.join(args.manifest, f"task_{t + 1}.json"), "w") as f:
            json.dump(buckets[t], f)

    imb = max(loads) / (sum(loads) / len(loads)) if sum(loads) else 0
    print(f"{'NAIVE' if args.naive else 'BALANCED'}: {len(items)} items -> "
          f"{args.tasks} tasks")
    print(f"  estimated load imbalance (max/mean) = {imb:.2f}x  "
          f"(1.00 = perfect; lower is better)")
    print(f"  wrote {args.tasks} manifests to {args.manifest}/  "
          f"(array range: 1-{args.tasks})")


if __name__ == "__main__":
    main()
