#!/usr/bin/env python3
"""Shared country definitions for the European weather-resolution analyses.

The expansion set was fixed from source-coverage checks before evaluating any
model gains.  Every expansion country has load, renewable-share, wind, and
solar data in sampled pre-COVID, COVID, and post-COVID windows from the
Energy-Charts API, plus usable geolocated GEM wind/solar records.
"""

from __future__ import annotations


CORE_CODES = (
    "dk", "ie", "nl", "pt", "gr", "be", "lt", "hr", "bg", "lv", "si", "rs", "sk",
)

# Italy and Poland are not included because most historical GEM solar phases
# lack commissioning years. Hungary, Sweden, Finland, and Norway are not
# included because the sampled historical Energy-Charts solar series is absent.
EXPANSION_CODES = ("de", "es", "fr", "at", "cz", "ro")
EUROPE_CODES = CORE_CODES + EXPANSION_CODES

COUNTRIES = {
    "at": "Austria",
    "be": "Belgium",
    "bg": "Bulgaria",
    "cz": "Czech Republic",
    "de": "Germany",
    "dk": "Denmark",
    "es": "Spain",
    "fr": "France",
    "gr": "Greece",
    "hr": "Croatia",
    "ie": "Ireland",
    "lt": "Lithuania",
    "lv": "Latvia",
    "nl": "Netherlands",
    "pt": "Portugal",
    "ro": "Romania",
    "rs": "Serbia",
    "si": "Slovenia",
    "sk": "Slovakia",
}

# South, north, west, east. Domains intentionally include a small coastal
# margin so offshore wind sites are not silently clipped. Spain and France use
# their continental interconnected-system footprints; any GEM phases outside
# those scopes are explicitly audited by build_weighted_weather_local.py.
BBOX = {
    "at": (46.3, 49.1, 9.4, 17.2),
    "be": (49.5, 51.5, 2.5, 6.4),
    "bg": (41.2, 44.2, 22.4, 28.6),
    "cz": (48.5, 51.1, 12.0, 18.9),
    "de": (47.2, 55.2, 5.5, 15.5),
    "dk": (54.5, 57.8, 7.5, 15.2),
    "es": (35.7, 43.9, -9.7, 4.5),
    "fr": (41.2, 51.3, -5.5, 9.8),
    "gr": (34.8, 41.8, 19.3, 28.3),
    "hr": (42.4, 46.6, 13.5, 19.4),
    "ie": (51.4, 55.4, -10.6, -5.9),
    "lt": (53.9, 56.5, 20.9, 26.9),
    "lv": (55.7, 58.1, 21.0, 28.2),
    "nl": (50.7, 53.6, 3.0, 7.2),
    "pt": (36.9, 42.2, -9.6, -6.2),
    "ro": (43.5, 48.3, 20.1, 29.9),
    "rs": (42.2, 46.2, 18.8, 23.0),
    "si": (45.4, 46.9, 13.4, 16.6),
    "sk": (47.7, 49.6, 16.8, 22.6),
    "tx": (25.8, 36.5, -106.6, -93.5),
}
