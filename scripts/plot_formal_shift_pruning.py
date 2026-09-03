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

    fig = plt.figure(figsize=(7.2, 3.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.42], wspace=0.32)

    ax = fig.add_subplot(grid[0, 0])
    labels = ["No\nshift", "50%\nutility", "Full\nshift"]
    values = [22.75, 17.51, 16.48]
    colors = ["#9aa0a6", "#e08e3c", "#3973ac"]
    bars = ax.bar(labels, values, color=colors, width=0.66,
                  edgecolor="#30343b", linewidth=0.55)
    ax.set_ylabel("Test WER (%)", fontsize=8)
    ax.set_ylim(14, 24.8)
    ax.set_yticks([15, 18, 21, 24])
    ax.tick_params(axis="both", labelsize=7, length=2)
    ax.grid(axis="y", color="#d9dde3", linewidth=0.55)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.28,
                f"{value:.2f}", ha="center", va="bottom", fontsize=7.2)
    ax.annotate(
        "83.5% gain\nretained", xy=(1, 17.51), xytext=(1.98, 19.8),
        ha="center", va="center", fontsize=7.2, color="#9b4e10",
        arrowprops=dict(arrowstyle="->", color="#9b4e10", lw=0.8,
                        connectionstyle="arc3,rad=-0.18"),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Same model, frozen CTC head", fontsize=8.5, pad=5)

    flow = fig.add_subplot(grid[0, 1])
    flow.axis("off")
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    boxes = [
        (0.02, 0.60, 0.17, 0.22, r"$E_0$"),
        (0.225, 0.60, 0.23, 0.22, r"$\Delta=E_{ft}-E_0$"),
        (0.505, 0.60, 0.27, 0.22,
         r"$U_i=\mathbb{E}|\Delta_i\,\partial L/\partial\Delta_i|$"),
        (0.82, 0.60, 0.16, 0.22, "Top 50%"),
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
    for x1, x2 in [(0.19, 0.22), (0.46, 0.50), (0.79, 0.815)]:
        flow.add_patch(FancyArrowPatch(
            (x1, 0.71), (x2, 0.71), arrowstyle="-|>", mutation_scale=8,
            linewidth=0.8, color="#4b5563",
        ))

    patch = FancyBboxPatch(
        (0.20, 0.16), 0.60, 0.22,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor="#eaf2fb", edgecolor="#3973ac", linewidth=0.9,
    )
    flow.add_patch(patch)
    flow.text(0.50, 0.27, r"$E_0+M_{512}\odot\Delta$",
              ha="center", va="center", fontsize=9)
    flow.text(0.50, 0.205,
              "original fine-tuned CTC head; no retraining",
              ha="center", va="center", fontsize=7.0, color="#274c77")
    flow.add_patch(FancyArrowPatch(
        (0.90, 0.59), (0.73, 0.39), arrowstyle="-|>", mutation_scale=8,
        linewidth=0.8, color="#3973ac", connectionstyle="arc3,rad=-0.15",
    ))
    flow.text(0.92, 0.47, "direct\nintervention", ha="center", va="center",
              fontsize=6.8, color="#274c77")
    flow.set_title("Taylor utility identifies functional shift coordinates",
                   fontsize=8.5, pad=5)

    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
