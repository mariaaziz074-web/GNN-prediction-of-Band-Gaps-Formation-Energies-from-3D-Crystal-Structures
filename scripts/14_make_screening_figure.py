from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TABLE_DIR = Path("results/tables")
FIGURE_DIR = Path("results/figures")

SCREEN_PATH = (
    TABLE_DIR
    / "screened_test_candidates.csv"
)

OUTPUT_NAME = (
    "figure_05_candidate_screening"
)

COBALT = "#004CFF"
CRIMSON = "#E6003D"
EMERALD = "#009B55"
VIOLET = "#7200C9"
ORANGE = "#F47A00"

INK = "#080A0F"
GRID = "#C4CBD4"
WHITE = "#FFFFFF"

DPI = 400


def style_axis(ax):
    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.spines[
        "left"
    ].set_linewidth(1.4)

    ax.spines[
        "bottom"
    ].set_linewidth(1.4)

    ax.tick_params(
        width=1.1,
        length=5,
        colors=INK,
    )

    ax.grid(
        color=GRID,
        linestyle=":",
        linewidth=0.8,
        alpha=0.45,
        zorder=0,
    )


def save_figure(fig):
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png = (
        FIGURE_DIR
        / f"{OUTPUT_NAME}.png"
    )

    pdf = (
        FIGURE_DIR
        / f"{OUTPUT_NAME}.pdf"
    )

    fig.savefig(
        png,
        dpi=DPI,
        bbox_inches="tight",
        facecolor=WHITE,
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor=WHITE,
    )

    plt.close(fig)

    print(f"[OK] {png}")
    print(f"[OK] {pdf}")


def main():
    plt.rcParams.update(
        {
            "font.family": (
                "DejaVu Sans"
            ),
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )

    print(
        "[INFO] Loading screened "
        "candidates..."
    )

    data = pd.read_csv(
        SCREEN_PATH
    )

    if len(data) == 0:
        raise ValueError(
            "No screened candidates "
            "were found."
        )

    confirmed = data[
        data[
            "dft_meets_screen"
        ].astype(bool)
    ].copy()

    false_positive = data[
        ~data[
            "dft_meets_screen"
        ].astype(bool)
    ].copy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            13.2,
            5.6,
        ),
        facecolor=WHITE,
    )

    # --------------------------------------------------------
    # Panel A:
    # predicted screening space
    # --------------------------------------------------------

    ax = axes[0]

    style_axis(ax)

    ax.scatter(
        false_positive[
            "bandgap_pred"
        ],
        false_positive[
            "formation_pred"
        ],
        s=58,
        marker="X",
        color=CRIMSON,
        alpha=0.78,
        edgecolor=WHITE,
        linewidth=0.5,
        label=(
            "Not DFT-confirmed"
        ),
        zorder=3,
    )

    ax.scatter(
        confirmed[
            "bandgap_pred"
        ],
        confirmed[
            "formation_pred"
        ],
        s=64,
        marker="o",
        color=EMERALD,
        alpha=0.90,
        edgecolor=INK,
        linewidth=0.45,
        label="DFT-confirmed",
        zorder=4,
    )

    ax.axvline(
        1.0,
        color=VIOLET,
        linestyle="--",
        linewidth=1.6,
    )

    ax.axvline(
        3.0,
        color=VIOLET,
        linestyle="--",
        linewidth=1.6,
    )

    ax.axhline(
        -1.0,
        color=ORANGE,
        linestyle="--",
        linewidth=1.6,
    )

    ax.axvspan(
        1.0,
        3.0,
        color=COBALT,
        alpha=0.055,
        zorder=0,
    )

    ax.set_xlabel(
        "Predicted band gap (eV)",
        fontweight="bold",
    )

    ax.set_ylabel(
        "Predicted formation energy "
        "(eV/atom)",
        fontweight="bold",
    )

    ax.set_title(
        "A  Predicted Screening Space",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )

    ax.legend(
        frameon=True,
        facecolor=WHITE,
        edgecolor=INK,
        framealpha=0.96,
    )

    # --------------------------------------------------------
    # Panel B:
    # predicted vs true gap for selected candidates
    # --------------------------------------------------------

    ax = axes[1]

    style_axis(ax)

    maximum = max(
        float(
            data[
                "bandgap_pred"
            ].max()
        ),
        float(
            data[
                "bandgap_true"
            ].max()
        ),
        3.2,
    )

    ax.plot(
        [0, maximum],
        [0, maximum],
        color=INK,
        linestyle="--",
        linewidth=1.8,
        label="Ideal prediction",
        zorder=1,
    )

    ax.axhspan(
        1.0,
        3.0,
        color=COBALT,
        alpha=0.055,
        zorder=0,
    )

    ax.scatter(
        false_positive[
            "bandgap_pred"
        ],
        false_positive[
            "bandgap_true"
        ],
        s=58,
        marker="X",
        color=CRIMSON,
        alpha=0.78,
        edgecolor=WHITE,
        linewidth=0.5,
        label=(
            "Not DFT-confirmed"
        ),
        zorder=3,
    )

    ax.scatter(
        confirmed[
            "bandgap_pred"
        ],
        confirmed[
            "bandgap_true"
        ],
        s=64,
        marker="o",
        color=EMERALD,
        alpha=0.90,
        edgecolor=INK,
        linewidth=0.45,
        label="DFT-confirmed",
        zorder=4,
    )

    ax.set_xlabel(
        "Predicted band gap (eV)",
        fontweight="bold",
    )

    ax.set_ylabel(
        "DFT band gap (eV)",
        fontweight="bold",
    )

    ax.set_title(
        "B  Retrospective DFT Check",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )

    # --------------------------------------------------------
    # Summary annotation
    # --------------------------------------------------------

    precision = (
        len(confirmed)
        / len(data)
        * 100.0
    )

    summary = (
        f"Selected: {len(data)}\n"
        f"DFT-confirmed: "
        f"{len(confirmed)}\n"
        f"Precision: "
        f"{precision:.1f}%"
    )

    ax.text(
        0.97,
        0.05,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        bbox={
            "boxstyle": (
                "round,pad=0.45"
            ),
            "facecolor": WHITE,
            "edgecolor": INK,
            "linewidth": 1.0,
            "alpha": 0.94,
        },
        zorder=8,
    )

    fig.suptitle(
        "Retrospective GNN Screening of "
        "Withheld JARVIS-DFT Materials",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout()

    save_figure(fig)

    print(
        "[INFO] Selected candidates: "
        f"{len(data)}"
    )

    print(
        "[INFO] DFT-confirmed: "
        f"{len(confirmed)}"
    )

    print(
        "[INFO] Retrospective precision: "
        f"{precision:.1f}%"
    )

    print(
        "[OK] Candidate-screening figure "
        "complete."
    )


if __name__ == "__main__":
    main()