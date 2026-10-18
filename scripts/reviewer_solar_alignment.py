"""Effect of the one-hour ERA5 radiation offset on the headline result.

ERA5 surface radiation is an accumulation stamped at the END of its hour;
Energy-Charts stamps the START. So ERA5 at T+1 describes the same hour as
Energy-Charts at T, i.e. the radiation series must be shifted by -1.
Same pipeline as reviewer_decomposition.py (which reproduces Table 2)."""
import numpy as np, pandas as pd, glob, os, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
import argparse
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--data-root", default=str(REPO / "data"),
                 help="pipeline data directory (default: <repo>/data)")
_ap.add_argument("--out", default=str(REPO / "results" / "paper" / "reviewer_solar_alignment.csv"))
ARGS = _ap.parse_args()
ROOT = ARGS.data_root
W=["wind_speed_100m","shortwave_radiation","temperature_2m"]
def cal(ix):
    h,m=ix.hour.values,ix.month.values
    return pd.DataFrame({"sh":np.sin(2*np.pi*h/24),"ch":np.cos(2*np.pi*h/24),
        "sm":np.sin(2*np.pi*m/12),"cm":np.cos(2*np.pi*m/12),
        "we":(ix.dayofweek.values>=5).astype(float)},index=ix)
def r2(y,p): return 1-((y-p)**2).sum()/((y-y.mean())**2).sum()
def score(X,y):
    n=int(len(y)*0.8); m=HistGradientBoostingRegressor(random_state=42).fit(X.iloc[:n],y.iloc[:n])
    return r2(y.iloc[n:].values,m.predict(X.iloc[n:]))
rows=[]
for f in sorted(glob.glob(f"{ROOT}/energy_targets_by_era/post/weather_energy_merged_*.csv")):
    c=os.path.basename(f)[-6:-4]
    e=pd.read_csv(f,parse_dates=["timestamp"]).set_index("timestamp")
    w=pd.read_csv(f"{ROOT}/country_weather_post_covid/weather_era5_{c}_capacity_0p25deg.csv",parse_dates=["timestamp"]).set_index("timestamp")
    w2=w.copy(); w2["shortwave_radiation"]=w2["shortwave_radiation"].shift(-1)
    out={"code":c}
    for tag,ww in (("orig",w),("fixed",w2)):
        d=e.join(ww,how="inner"); d=d[d.index>="2022-01-01"].dropna(subset=["Renewable_share_of_load"]+W)
        X0=cal(d.index); X1=pd.concat([X0,d[W]],axis=1); y=d.Renewable_share_of_load
        out[f"{tag}_cal"],out[f"{tag}_wx"]=score(X0,y),score(X1,y)
        out[f"{tag}_gain"]=out[f"{tag}_wx"]-out[f"{tag}_cal"]
    rows.append(out); print(f"  {c}: gain {out['orig_gain']:+.3f} -> {out['fixed_gain']:+.3f}",flush=True)
t=pd.DataFrame(rows); t.to_csv(ARGS.out,index=False)
print("\nmedian gain  orig %.3f  fixed %.3f" % (t.orig_gain.median(), t.fixed_gain.median()))
print("median +wx R2 orig %.3f  fixed %.3f" % (t.orig_wx.median(), t.fixed_wx.median()))
