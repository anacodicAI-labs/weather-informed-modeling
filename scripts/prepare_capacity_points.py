"""
prepare_capacity_points.py  --  STAGE 1 (run once, on a login node).

Turns a downloaded plant-location dataset into capacity-weighted weather
sample points, one set per country. Run and READ THE COVERAGE REPORT before
committing to the (slow) weather fetch -- it tells you whether the data
actually covers each country.

INPUT: a plant CSV you download yourself. Two supported sources:
  * Dunnett et al. "Harmonised global wind/solar farm locations" (recommended)
  * WRI Global Power Plant Database (global_power_plant_database.csv)
Set PLANT_CSV and the COLUMN NAMES below to match whichever you downloaded --
run once; if the coverage numbers look wrong, the column mapping is wrong.

METHOD: bin plants into a coarse grid (default 0.5 deg). For each non-empty
cell, sum wind MW and solar MW separately (wind and solar farms sit in
different places, so they get weighted separately downstream). Output one
row per cell: cell centroid + wind_mw + solar_mw. This keeps the number of
weather fetches to dozens-per-country instead of thousands-of-plants.

    python prepare_capacity_points.py
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "data", "capacity_points")

# >>> EDIT THESE to match your downloaded file <<<
PLANT_CSV = os.path.join(HERE, "..", "data", "plants.csv")
COL_LAT = "latitude"
COL_LON = "longitude"
COL_MW = "capacity_mw"
COL_FUEL = "fuel"            # column holding technology / primary fuel
WIND_LABELS = {"wind", "Wind", "onshore", "offshore"}
SOLAR_LABELS = {"solar", "Solar", "pv", "PV"}

GRID_DEG = 0.5              # cell size; smaller = higher resolution = more fetches

# approximate bounding boxes (lat_min, lat_max, lon_min, lon_max) -- verify once
BBOX = {
    "dk": (54.5, 57.8, 8.0, 15.2),  "ie": (51.4, 55.4, -10.6, -5.9),
    "nl": (50.7, 53.6, 3.3, 7.2),   "pt": (36.9, 42.2, -9.6, -6.2),
    "gr": (34.8, 41.8, 19.3, 28.3), "be": (49.5, 51.5, 2.5, 6.4),
    "lt": (53.9, 56.5, 20.9, 26.9), "hr": (42.4, 46.6, 13.5, 19.4),
    "bg": (41.2, 44.2, 22.4, 28.6), "lv": (55.7, 58.1, 21.0, 28.2),
    "si": (45.4, 46.9, 13.4, 16.6), "rs": (42.2, 46.2, 18.8, 23.0),
    "sk": (47.7, 49.6, 16.8, 22.6), "tx": (25.8, 36.5, -106.6, -93.5),
}


def classify(fuel):
    f = str(fuel).lower()
    if any(w.lower() in f for w in WIND_LABELS):
        return "wind"
    if any(s.lower() in f for s in SOLAR_LABELS):
        return "solar"
    return None


def cellify(df):
    """Bin to grid cells; return per-cell wind_mw / solar_mw with centroids."""
    df = df.copy()
    df["clat"] = (np.floor(df[COL_LAT] / GRID_DEG) + 0.5) * GRID_DEG
    df["clon"] = (np.floor(df[COL_LON] / GRID_DEG) + 0.5) * GRID_DEG
    df["tech"] = df[COL_FUEL].map(classify)
    df = df[df.tech.notna()]
    piv = (df.pivot_table(index=["clat", "clon"], columns="tech",
                          values=COL_MW, aggfunc="sum", fill_value=0.0)
             .reset_index())
    for c in ("wind", "solar"):
        if c not in piv:
            piv[c] = 0.0
    return piv.rename(columns={"wind": "wind_mw", "solar": "solar_mw"})


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(PLANT_CSV):
        raise FileNotFoundError(
            f"{PLANT_CSV} not found. Download the Dunnett or WRI plant dataset, "
            f"put it there, and set the COLUMN NAMES at the top of this script.")

    plants = pd.read_csv(PLANT_CSV)
    for need in (COL_LAT, COL_LON, COL_MW, COL_FUEL):
        if need not in plants.columns:
            raise KeyError(f"column '{need}' not in file; columns are "
                           f"{list(plants.columns)} -- fix the COL_* settings.")

    print(f"{'country':8}{'plants':>8}{'cells':>7}{'wind_MW':>11}{'solar_MW':>11}")
    print("-" * 45)
    for code, (la0, la1, lo0, lo1) in BBOX.items():
        sub = plants[(plants[COL_LAT].between(la0, la1)) &
                     (plants[COL_LON].between(lo0, lo1))]
        cells = cellify(sub)
        cells = cells[(cells.wind_mw + cells.solar_mw) > 0]
        cells.to_csv(os.path.join(OUT_DIR, f"{code}.csv"), index=False)
        n_plants = len(sub[sub[COL_FUEL].map(classify).notna()])
        print(f"{code:8}{n_plants:>8}{len(cells):>7}"
              f"{cells.wind_mw.sum():>11.0f}{cells.solar_mw.sum():>11.0f}")

    print(f"\nwrote per-country cell files to {OUT_DIR}/")
    print("COVERAGE CHECK: compare the wind_MW / solar_MW totals above against "
          "known national installed capacities. If a country reads ~0 or way "
          "off, the dataset doesn't cover it well -- decide before fetching.")


if __name__ == "__main__":
    main()
