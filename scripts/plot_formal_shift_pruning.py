#!/usr/bin/env python3
"""Render the primary frozen-head shift-pruning figure."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("iclr2027/figures/formal_shift_pruning.pdf"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.2, 3.35))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.55], wspace=0.30)

    broken = grid[0, 0].subgridspec(2, 1, height_ratios=[1, 2.15], hspace=0.06)
    ax_top = fig.add_subplot(broken[0, 0])
    ax_bottom = fig.add_subplot(broken[1, 0], sharex=ax_top)
    labels = ["No\nshift", "Utility\n50%", "Full\nshift"]
    values = [118.912, 11.713, 10.499]
    colors = ["#9aa0a6", "#e08e3c", "#3973ac"]
    x = range(len(labels))

    top_bars = ax_top.bar([0], [values[0]], color=[colors[0]], width=0.66,
                          edgecolor="#30343b", linewidth=0.55)
    bottom_bars = ax_bottom.bar([1, 2], [values[1], values[2]],
                                color=colors[1:], width=0.66,
                                edgecolor="#30343b", linewidth=0.55)
    ax_top.set_ylim(105, 123)
    ax_bottom.set_ylim(0, 20.5)
    ax_top.set_yticks([110, 120])
    ax_bottom.set_yticks([0, 5, 10, 15, 20])
    ax_bottom.set_xticks(list(x), labels)
    ax_bottom.set_ylabel("Test WER (%)", fontsize=8)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for axis in (ax_top, ax_bottom):
        axis.tick_params(axis="both", labelsize=7, length=2)
        axis.grid(axis="y", color="#d9dde3", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.set_title("Frozen CTC readout", fontsize=8.5, pad=4)

    ax_top.text(0, values[0] + 0.8, f"{values[0]:.3f}",
                ha="center", va="bottom", fontsize=7.2)
    for bar, value in zip(bottom_bars, values[1:]):
        ax_bottom.text(bar.get_x() + bar.get_width() / 2, value + 0.45,
                       f"{value:.3f}", ha="center", va="bottom", fontsize=7.2)
    ax_bottom.annotate(
        "98.9% gain\nretained", xy=(1, values[1]), xytext=(1.62, 17.2),
        ha="center", va="center", fontsize=7.1, color="#9b4e10",
        arrowprops=dict(arrowstyle="->", color="#9b4e10", lw=0.8,
                        connectionstyle="arc3,rad=-0.18"),
    )
    # Break marks make the 118.912% counterfactual and the 10--12% operating
    # range legible without hiding either scale.
    diagonal = 0.018
    kwargs = dict(transform=ax_top.transAxes, color="#30343b", clip_on=False,
                  linewidth=0.8)
    ax_top.plot((-diagonal, diagonal), (-diagonal, diagonal), **kwargs)
    ax_top.plot((1 - diagonal, 1 + diagonal), (-diagonal, diagonal), **kwargs)
    kwargs["transform"] = ax_bottom.transAxes
    ax_bottom.plot((-diagonal, diagonal), (1 - diagonal, 1 + diagonal), **kwargs)
    ax_bottom.plot((1 - diagonal, 1 + diagonal), (1 - diagonal, 1 + diagonal), **kwargs)

    flow = fig.add_subplot(grid[0, 1])
    flow.axis("off")
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    boxes = [
        (0.02, 0.73, 0.24, 0.14, r"$f_{\mathrm{pt}}(x)\!\to\!E_{\mathrm{pt}}$"),
        (0.30, 0.73, 0.24, 0.14, r"$f_{\mathrm{ft}}(x)\!\to\!E_{\mathrm{ft}}$"),
        (0.62, 0.73, 0.35, 0.14, r"$\Delta=E_{\mathrm{ft}}-E_{\mathrm{pt}}$"),
        (0.11, 0.47, 0.38, 0.15,
         r"$U_i=\mathbb{E}|\Delta_i\,\partial\mathcal{L}/\partial\Delta_i|$"),
        (0.60, 0.47, 0.28, 0.15, r"$M_K=\operatorname{TopK}(U)$"),
    ]
    for x, y, width, height, label in boxes:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#f4f6f8", edgecolor="#4b5563", linewidth=0.8,
        )
        flow.add_patch(patch)
        flow.text(x + width / 2, y + height / 2, label,
                  ha="center", va="center", fontsize=7.1)
    arrows = [
        ((0.26, 0.80), (0.61, 0.80)),
        ((0.54, 0.80), (0.61, 0.80)),
        ((0.79, 0.73), (0.49, 0.62)),
        ((0.49, 0.545), (0.59, 0.545)),
    ]
    for start, end in arrows:
        flow.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=8,
            linewidth=0.8, color="#4b5563",
        ))

    patch = FancyBboxPatch(
        (0.13, 0.16), 0.74, 0.20,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor="#eaf2fb", edgecolor="#3973ac", linewidth=0.9,
    )
    flow.add_patch(patch)
    flow.text(0.50, 0.285, r"$\widetilde E_K=E_{\mathrm{pt}}+M_K\odot\Delta$",
              ha="center", va="center", fontsize=9)
    flow.text(0.50, 0.205,
              "original fine-tuned CTC readout; no retraining",
              ha="center", va="center", fontsize=7.0, color="#274c77")
    flow.add_patch(FancyArrowPatch(
        (0.74, 0.47), (0.74, 0.36), arrowstyle="-|>", mutation_scale=8,
        linewidth=0.8, color="#3973ac", connectionstyle="arc3,rad=-0.15",
    ))
    flow.text(0.91, 0.40, "direct\nintervention", ha="center", va="center",
              fontsize=6.8, color="#274c77")
    flow.text(0.50, 0.055, r"$g_{\mathrm{ft}}\;\to\;\operatorname{Dec}$ (fixed)",
              ha="center", va="center", fontsize=7.5, color="#274c77")
    flow.set_title("Measure, rank, intervene", fontsize=8.5, pad=5)

    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
