# weather-informed-modeling

Code for "Weather-Informed Modeling of Renewable Electricity Share in Europe: Effects of Renewable
Mix and Spatial Resolution." Random Forest and gradient-boosting models predict hourly renewable
generation as a share of electricity load in 19 European countries, isolating the added predictive
value of ERA5 weather beyond calendar patterns, and comparing installed-capacity-weighted vs. uniform
weather aggregation across four spatial resolutions.

The paper itself lives in a separate, private planning repo (`my-local/weather-informed-modeling/`),
not here — this repo is the public, citable code artifact referenced by the paper's Data Availability
statement.

## Layout

```
weather-informed-modeling/
├── weather_informed/         # shared library code, imported by scripts/ below
│   ├── regions.py             # country codes, names, bounding boxes
│   ├── weather_build.py       # ERA5 coarsening + capacity-weighted/uniform aggregation
│   ├── evaluate.py            # calendar-only vs. calendar+weather model fitting and gain
│   └── plotting_style.py      # shared matplotlib rcParams
├── scripts/                   # standalone, single-purpose pipeline steps (run in the order below)
│   ├── fetch_era_energy_inputs.py        # Energy-Charts API -> hourly load/renewable-share targets
│   ├── import_gem_capacity.py            # GEM wind/solar Excel trackers -> annual capacity maps
│   ├── stage_era5_arco.py                # ERA5 (ARCO/Zarr) -> per-country native-grid NetCDF
│   ├── prepare_era5_resolution_cache.py  # coarsens staged ERA5 to 0.25/0.5/1/2 degree caches
│   ├── prepare_capacity_points.py        # LEGACY precursor to import_gem_capacity.py; kept for
│   │                                      #   reference, not part of the current pipeline
│   ├── run_era_spatial_resolution.py     # orchestrates the steps above across eras (see below)
│   ├── bootstrap_post_covid_all_models.py  # current block-bootstrap CI script (cheap, no refit),
│   │                                        #   uses weather_informed.evaluate for model fitting
│   ├── bootstrap_core.py                 # legacy refit-based bootstrap, SGE-cluster array job
│   ├── summarize_spatial_resolution.py   # aggregates run outputs into summary tables + 1 figure
│   ├── plot_post_covid_bootstrap.py      # 2x2 bootstrap CI figure
│   ├── plot_era_spatial_resolution_results.py # era-comparison figure suite + figure_statistics.csv
│   ├── generate_original_study_figures.py    # main 16-figure diagnostic suite
│   └── generate_paper_placeholder_figures.py # the 7 figures assembled for the paper itself
├── notebooks/                 # narrative, teaching-annotated walkthroughs of the pipeline
│   ├── 01_fetch_energy_and_capacity_data.ipynb
│   ├── 02_stage_and_aggregate_weather.ipynb
│   ├── 03_evaluate_calendar_vs_weather_models.ipynb
│   ├── 04_resolution_and_capacity_sensitivity.ipynb
│   └── 05_figures_and_bootstrap_ci.ipynb
├── data/                       # inputs and intermediate files (gitignored; see below)
├── results/                    # run outputs (gitignored) except:
│   └── paper/                  #   the curated tables the paper's numbers come from
├── figures/                    # run outputs (gitignored) except:
│   └── paper/                  #   the three figures in the Results section
├── CITATION.cff
├── requirements.txt
└── .env.example
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in CDSAPI_KEY, or use ~/.cdsapirc instead
```

## What runs offline vs. needs a key

| Step | Needs a key? |
|---|---|
| `fetch_era_energy_inputs.py` (Energy-Charts) | No -- public API, no key |
| `import_gem_capacity.py` (GEM trackers) | No -- but you must manually download the GEM Excel workbooks first (not redistributed here); see the script's docstring for the expected filenames under `data/raw/gem/` |
| `stage_era5_arco.py` (ERA5) | Yes -- `CDSAPI_KEY` (Copernicus CDS) |
| everything downstream of the above three | No -- reads local files only |

## Committed outputs

`results/paper/` and `figures/paper/` hold the artifacts the paper's Results section is
built from, so the reported numbers can be checked without re-running the pipeline (which
needs ~20 GB of ERA5 and a Copernicus key). Absolute paths in provenance columns have been
rewritten as repo-relative.

| File | Backs |
|---|---|
| `model_results_all_eras.csv` | per-country scores and weather-added gain, all eras x tasks x models x resolutions x schemes |
| `resolution_summary_all_eras.csv` | predictive drift and grid-point-hour workload per resolution |
| `figure_statistics.csv` | Wilcoxon p-values for the pre/post comparison |
| `capacity_weighted_primary_results_table.csv` | headline capacity-weighted summary per era |
| `block_bootstrap_correlations.csv` | block-bootstrap CIs on the wind-dominance correlation, with and without Lithuania |
| `country_size_sensitivity_post.csv` | partial correlations controlling for log grid-cell count |
| `headline_correlations.csv`, `leave_one_country_out_correlations.csv` | correlation and its leave-one-country-out range |
| `country_summary_pre.csv`, `country_summary_post.csv` | per-country wind and solar shares of load |

| Figure | Shows |
|---|---|
| `fig_results_percountry` | per-country calendar vs. calendar+weather, both eras |
| `fig_results_winddominance` | wind dominance vs. weather-added gain, post-COVID |
| `fig_results_drift_workload` | predictive drift against workload reduction |

Everything else under `results/` and `figures/` remains gitignored.

## Citing this work

See `CITATION.cff`. GitHub renders it as a "Cite this repository" button; `cffconvert` will
turn it into BibTeX. The preferred citation is the paper, not the repo.

## Known gaps

`prepare_capacity_points.py` is legacy and does not import `weather_informed.regions` -- it has its
own smaller, duplicated bounding-box dict. It has been superseded by `import_gem_capacity.py` for the
current pipeline; kept for reference only.

## License

MIT, see `LICENSE`. The same terms are recorded in `CITATION.cff`.
