"""House style for every figure in the paper (IEEE compsoc, two-column, Times body).

Draw at FINAL print size and include with width equal to the drawn width, so 8 pt here is 8 pt on the page (IEEE: figure text 8-10 pt):
  TEXT_W = 7.16 in (\\textwidth), COL_W = 3.5 in (\\columnwidth).
Font: STIX (Times-compatible, ships with matplotlib) for text and math, embedded as TrueType (Type 42).
Palette (validated with the dataviz validator on white: lightness, chroma, CVD dE 24.7, normal-vision dE 33.6,
contrast >= 3:1, ALL PASS): BLUE = defender-side quantity (operating points, prevented attacks),
ORANGE = attacker-side effect (detected miss increase, gate-only success, vacuous block-all), GRAY = not
significant / reference. Text is always INK/INK2, never a series color. No bold in-figure titles: panels get a
small '(a)' label; descriptions belong in the caption. Usage: import paper_style as ps; ps.apply()."""
import matplotlib as mpl

TEXT_W, COL_W = 7.16, 3.5
BLUE, ORANGE, GRAY, LIGHTGRAY = "#2a78d6", "#eb6834", "#8a8984", "#e6e6e6"
INK, INK2 = "#0b0b0b", "#52514e"

def apply():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 8.0, "axes.labelsize": 8.0, "axes.titlesize": 8.0, "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0, "legend.fontsize": 8.0, "axes.titleweight": "normal",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
        "axes.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.labelpad": 2.0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.0, "lines.markersize": 3.5, "lines.markeredgewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.2, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2,
        "grid.color": LIGHTGRAY, "grid.linewidth": 0.4,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    })

def panel_label(ax, letter, x=-0.02, y=1.02):
    """Small regular-weight '(a)' at the top-left of an axes (no bold titles inside figures)."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0, color=INK)
