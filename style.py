"""Shared matplotlib styling for research-quality figures.

The default matplotlib look (DejaVu Sans) reads as non-publication. We switch to a
Times-like serif with Computer Modern math to match the paper's newtxtext body and
LaTeX math, tighten sizes, and lighten the frame.
"""
import matplotlib as mpl


def apply_style():
    mpl.rcParams.update({
        "pdf.fonttype": 42,                # embed TrueType, not Type 3 (AAAI PDF rule)
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",          # LaTeX-like math
        # Figures are 5in wide but printed at ~3.3in column width (ratio ~0.66),
        # so set sizes ~1.5x larger than the target on-page point size.
        "font.size": 14,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "legend.fontsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.1,
        "axes.spines.top": False,          # single-axis plots; twin axes re-enable
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "lines.linewidth": 2.8,
        "lines.markersize": 7.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
    })
