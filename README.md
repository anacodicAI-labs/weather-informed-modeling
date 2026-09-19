# weather-informed-modeling

How much does weather data actually improve a model of renewable electricity supply — and how much
of that data do you really need?

This repository contains the full analysis for **"Weather-Informed Modeling of Renewable Electricity
Share in Europe: Effects of Renewable Mix and Spatial Resolution."** It predicts hourly renewable
generation as a share of national electricity demand across 19 European power systems, measures what
ERA5 weather adds over a calendar-only baseline, and tests two design choices: how weather is
averaged across a country, and how finely it is resolved.

## What the analysis found

| | |
|---|---|
| Weather more than doubles explained variance | median R² 0.10 → 0.56 (pre-COVID), 0.23 → 0.63 (post-COVID) |
| It helps almost everywhere | 15 of 19 countries pre-COVID, 16 of 19 post-COVID |
| The benefit concentrates in wind-led systems | r = 0.88 between wind dominance and weather gain |
| Where capacity sits beats how finely weather is resolved | capacity weighting wins 55/64 and 65/76 comparisons; a 1° grid drops 93% of the data for 0.005 drift |

The practical consequence: coarsen the weather grid, and spend the effort on locating capacity
accurately instead.

## Start here

Read the notebooks in order. They narrate the pipeline stage by stage and explain why each step
exists, with no credentials required to follow along.

| Notebook | Covers |
|---|---|
| `01_fetch_energy_and_capacity_data.ipynb` | where the electricity and power-plant data come from |
| `02_stage_and_aggregate_weather.ipynb` | ERA5 staging, coarsening, capacity-weighted aggregation |
| `03_evaluate_calendar_vs_weather_models.ipynb` | the calendar-vs-weather comparison |
| `04_resolution_and_capacity_sensitivity.ipynb` | the resolution and weighting experiments |
| `05_figures_and_bootstrap_ci.ipynb` | figures and uncertainty intervals |

## Checking the reported numbers

Every number and figure in the paper comes from a committed file — you do not need to re-run the
pipeline (which needs a Copernicus key and roughly 20 GB of ERA5) to verify them.

**`results/paper/`**

| File | What it backs |
|---|---|
| `model_results_all_eras.csv` | per-country scores and weather-added gain, across every era, task, model, resolution and weighting scheme. This is the master table. |
| `capacity_weighted_primary_results_table.csv` | the headline capacity-weighted summary per era |
| `resolution_summary_all_eras.csv` | predictive drift and grid-point-hour workload at each resolution |
| `headline_correlations.csv` | wind dominance vs. weather-added gain, per model |
| `leave_one_country_out_correlations.csv` | how much that correlation moves when each country is dropped |
| `country_size_sensitivity_post.csv` | the same correlation after controlling for country domain size |
| `figure_statistics.csv` | Wilcoxon results for the pre/post-COVID comparison |
| `block_bootstrap_correlations.csv` | bootstrap confidence intervals on the correlation |
| `country_summary_pre.csv`, `country_summary_post.csv` | per-country wind and solar shares of load |

**`figures/paper/`** — PDF and PNG of each

| Figure | Shows |
|---|---|
| `fig_results_percountry` | calendar vs. calendar-plus-weather, every country, both eras |
| `fig_results_winddominance` | wind dominance against weather-added gain |
| `fig_results_drift_workload` | accuracy drift against the data volume saved by coarsening |

## Running it yourself

```bash
pip install -r requirements.txt
cp .env.example .env          # add CDSAPI_KEY, or use ~/.cdsapirc
```

`run_era_spatial_resolution.py` orchestrates the whole pipeline and is resumable — re-running skips
work already on disk. Start with `--dry-run` to see the commands, then one country before all 19:

```bash
python scripts/run_era_spatial_resolution.py --eras pre post --dry-run
python scripts/run_era_spatial_resolution.py --eras pre post --codes dk
```

Only one step needs a credential:

| Step | Credential |
|---|---|
| Energy-Charts download | none — public API |
| GEM capacity trackers | none, but download the Excel workbooks manually into `data/raw/gem/` (not redistributed here) |
| ERA5 staging | `CDSAPI_KEY` from Copernicus |
| everything downstream | none — reads local files |

## Code layout

```
weather_informed/            importable library, shared by every script
  regions.py                 country codes, names, bounding boxes
  weather_build.py           ERA5 coarsening and capacity-weighted aggregation
  evaluate.py                the model-fitting protocol and weather-added gain
  plotting_style.py          shared figure styling

scripts/
  fetch_era_energy_inputs.py          Energy-Charts -> hourly load and renewable share
  import_gem_capacity.py              GEM trackers -> annual capacity maps per country
  stage_era5_arco.py                  ERA5 via the ARCO/Zarr store (default route)
  stage_era5_cds.py                   ERA5 via queued CDS requests (alternative route)
  prepare_era5_resolution_cache.py    coarsen to 0.5/1/2 degrees, cached once
  prepare_capacity_points.py          builds capacity maps from a plant CSV instead of GEM

  run_era_spatial_resolution.py       orchestrates everything below, across eras
  audit_spatial_pipeline_data.py      verifies a country-era is complete before running
  run_spatial_resolution_ladder.py    every resolution x scheme combination, in parallel
  compare_era_spatial_resolution.py   compares eras, writes the comparison tables
  summarize_spatial_resolution.py     aggregates runs into the summary tables

  bootstrap_post_covid_all_models.py  block-bootstrap confidence intervals
  bootstrap_core.py                   refit-based bootstrap, for SGE cluster array jobs
  build_manifest.py                   task manifest consumed by bootstrap_core.py

  plot_era_spatial_resolution_results.py   era-comparison figure suite
  plot_post_covid_bootstrap.py             bootstrap interval figure
  generate_original_study_figures.py       16-figure diagnostic suite
  generate_paper_placeholder_figures.py    the figures used in the paper
```

`data/`, and everything in `results/` and `figures/` other than the `paper/` subfolders, are
generated locally and not tracked.

## Citing

See `CITATION.cff` — GitHub renders it as a "Cite this repository" button, and `cffconvert` turns it
into BibTeX. Please cite the paper rather than the repository.

## License

MIT, see `LICENSE`.
