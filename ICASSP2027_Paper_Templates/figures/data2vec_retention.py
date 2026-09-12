"""Render the data2vec Audio Large retained-shift sweep."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).with_name("data2vec_retention.pdf")


def main() -> None:
    budgets = np.array([25, 50, 75])
    values = {
        "DADS": np.array([15.20, 12.94, 10.96]),
        "Magnitude": np.array([13.30, 13.75, 11.10]),
        "Gradient": np.array([15.41, 13.68, 11.12]),
        "Random": np.array([13.00, 15.08, 11.35]),
    }
    random_sd = np.array([0.03, 0.31, 0.08])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(244.08 / 72, 153 / 72))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.24, top=0.97)

    styles = {
        "DADS": {"color": "#1f4e79", "marker": "o", "linewidth": 2.1, "zorder": 4},
        "Magnitude": {"color": "#d17c28", "marker": "s", "linewidth": 1.25, "zorder": 3},
        "Gradient": {"color": "#9a4e5a", "marker": "^", "linewidth": 1.25, "zorder": 3},
        "Random": {"color": "#666666", "marker": "D", "linewidth": 1.15, "linestyle": "--", "zorder": 2},
    }

    for name in ["Magnitude", "Gradient", "Random", "DADS"]:
        style = styles[name]
        ax.plot(budgets, values[name], label=name, markersize=4.2, **style)
        if name == "Random":
            ax.errorbar(
                budgets,
                values[name],
                yerr=random_sd,
                fmt="none",
                ecolor=style["color"],
                elinewidth=0.8,
                capsize=2,
                zorder=style["zorder"],
            )

    ax.plot(
        [100],
        [11.39],
        marker="*",
        markersize=7,
        color="#20252a",
        linestyle="None",
        label="Full",
        zorder=5,
    )
    ax.plot(
        [0],
        [13.77],
        marker="X",
        markersize=5.5,
        color="#20252a",
        linestyle="None",
        label="NoShift",
        zorder=5,
    )

    ax.set_xlim(-3, 103)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(10.4, 16.1)
    ax.set_xlabel("Retained shift (%)")
    ax.set_ylabel("Test WER (%)")
    ax.grid(axis="y", color="#d9dde0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#20252a")
    ax.spines["bottom"].set_color("#20252a")
    ax.tick_params(axis="both", length=3, color="#20252a")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.8,
        borderaxespad=0,
    )

    fig.savefig(OUT, format="pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
