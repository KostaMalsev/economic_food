"""Shared article-ready Matplotlib styling for FoodNorm analyses."""

from pathlib import Path

import matplotlib.pyplot as plt


FONT_SIZE_PT = 10
FIGURE_WIDTH_IN = 7.1


def apply_publication_style():
    """Use one consistent 10-point font throughout every figure element."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": FONT_SIZE_PT,
        "axes.titlesize": FONT_SIZE_PT,
        "axes.labelsize": FONT_SIZE_PT,
        "xtick.labelsize": FONT_SIZE_PT,
        "ytick.labelsize": FONT_SIZE_PT,
        "legend.fontsize": FONT_SIZE_PT,
        "figure.titlesize": FONT_SIZE_PT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_png_and_pdf(fig, png_path, *, dpi=300):
    """Save a high-resolution preview and a scalable publication PDF."""
    png_path = Path(png_path)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
