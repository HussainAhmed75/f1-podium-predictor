"""One visual system for every chart in the project (dark, F1-style)."""
import matplotlib.pyplot as plt

SURFACE = "#1a1a19"
INK = "#f0efec"
INK_2 = "#c3c2b7"
MUTED = "#8a8980"
GRID = "#33332f"
BLUE = "#3987e5"     # series 1 (dark-mode step)
ORANGE = "#d95926"   # series 2 / highlight
AQUA = "#199e70"     # series 3
GRAY = "#55554f"     # de-emphasised marks
SERIES = [BLUE, ORANGE, AQUA, "#c98500", "#d55181"]


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 26,
        "font.size": 11, "font.family": "DejaVu Sans", "legend.frameon": False, "lines.linewidth": 2,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=10.5, va="bottom")


def source(fig, text="Data: Jolpica-F1 / Ergast via Kaggle (Samuel_Pbq), CC BY-NC-SA 4.0 · Hussain Ahmed"):
    fig.text(0.01, -0.02, text, color=MUTED, fontsize=8.5, ha="left", va="top")
