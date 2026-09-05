"""Shared library code for the weather-informed renewable-share pipeline.

Submodules (import explicitly, e.g. ``from weather_informed import regions``,
rather than relying on re-exports here -- some submodules pull in heavy
geospatial/ML dependencies that shouldn't load just to read a constant):

- ``regions``: country codes, names, and bounding boxes used throughout the
  pipeline (formerly ``scripts/country_registry.py``).
- ``weather_build``: coarsens ERA5 fields and aggregates them per country,
  either uniformly or capacity-weighted (formerly
  ``scripts/build_weighted_weather_local.py``).
- ``evaluate``: fits the calendar-only and calendar-plus-weather
  classification/regression models and computes weather-added gain
  (formerly ``scripts/evaluate_spatial_weather.py``).
- ``plotting_style``: shared matplotlib rcParams for figure consistency
  (formerly ``scripts/poster_figure_style.py``).
"""
