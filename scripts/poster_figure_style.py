#!/usr/bin/env python3
"""Shared typography and export settings for poster figures."""

import matplotlib as mpl

PANEL_TITLE_SIZE = 16
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 13
ANNOTATION_SIZE = 13
NOTE_SIZE = 12


def apply_poster_style() -> None:
    """Use Arial consistently in raster and vector poster outputs."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.titleweight": "bold",
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.labelweight": "bold",
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": TICK_LABEL_SIZE,
            "lines.linewidth": 2.6,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
