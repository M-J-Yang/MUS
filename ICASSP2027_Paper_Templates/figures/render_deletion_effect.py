"""Render the frozen-head reversion plot with paper-facing model labels."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).with_name("deletion_effect.pdf")


def main() -> None:
    labels = ["W2V2-F1", "W2V2-F2", "D2V-F0"]
    undo_high = np.array([7.10, 9.86, 3.25])
    undo_low = np.array([0.22, 0.08, -0.43])
    y = np.arange(len(labels))

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(244.08 / 72, 133.2 / 72))
    fig.subplots_adjust(left=0.30, right=0.99, bottom=0.25, top=0.82)

    height = 0.22
    ax.barh(
        y - 0.13,
        undo_high,
        height=height,
        color="#2b6c96",
        edgecolor="#20252a",
        linewidth=0.7,
        label="Undo high 25%",
    )
    ax.barh(
        y + 0.13,
        undo_low,
        height=height,
        color="#c5c9cc",
        edgecolor="#20252a",
        linewidth=0.7,
        hatch="//",
        label="Undo low 25%",
    )

    for yi, value in zip(y - 0.13, undo_high):
        ax.text(value + 0.12, yi, f"{value:+.2f}", va="center", ha="left")
    for yi, value in zip(y + 0.13, undo_low):
        if value >= 0:
            ax.text(value + 0.12, yi, f"{value:+.2f}", va="center", ha="left")
        else:
            ax.text(0.12, yi, f"{value:+.2f}", va="center", ha="left")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.7, 11.7)
    ax.set_xticks([0, 3, 6, 9])
    ax.set_xlabel("WER change (percentage points)")
    ax.grid(axis="x", color="#d9dde0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#20252a")
    ax.spines["bottom"].set_color("#20252a")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=3, color="#20252a")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(-0.18, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.5,
        borderaxespad=0,
    )

    fig.savefig(OUT, format="pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
