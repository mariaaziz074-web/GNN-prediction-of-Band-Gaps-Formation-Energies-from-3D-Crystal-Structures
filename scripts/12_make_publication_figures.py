from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TABLE_DIR = Path("results/tables")
FIGURE_DIR = Path("results/figures")

DATA_PATH = (
    TABLE_DIR
    / "test_error_analysis.csv"
)

COMPARISON_PATH = (
    TABLE_DIR
    / "detailed_model_comparison.csv"
)

HISTORY_PATH = (
    TABLE_DIR
    / "gnn_training_history.csv"
)

DPI = 400


# ============================================================
# FINAL PUBLICATION PALETTE
# Bright, saturated, deep, high contrast.
# ============================================================

COBALT = "#004CFF"
CRIMSON = "#E6003D"
EMERALD = "#009B55"
VIOLET = "#7200C9"
ORANGE = "#F47A00"

INK = "#080A0F"
CHARCOAL = "#30343B"
GRID = "#C4CBD4"
WHITE = "#FFFFFF"


# ============================================================
# GENERAL STYLE
# ============================================================


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def style_axis(
    ax,
    grid=True,
):
    ax.set_facecolor(
        WHITE
    )

    ax.spines[
        "left"
    ].set_color(
        INK
    )

    ax.spines[
        "bottom"
    ].set_color(
        INK
    )

    ax.spines[
        "left"
    ].set_linewidth(
        1.45
    )

    ax.spines[
        "bottom"
    ].set_linewidth(
        1.45
    )

    ax.tick_params(
        axis="both",
        colors=INK,
        width=1.2,
        length=5,
        direction="out",
    )

    if grid:
        ax.grid(
            color=GRID,
            alpha=0.48,
            linestyle=":",
            linewidth=0.8,
            zorder=0,
        )


def save_figure(
    fig,
    filename,
):
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        FIGURE_DIR
        / f"{filename}.png"
    )

    pdf_path = (
        FIGURE_DIR
        / f"{filename}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor=WHITE,
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor=WHITE,
    )

    plt.close(
        fig
    )

    print(
        f"[OK] {png_path}"
    )

    print(
        f"[OK] {pdf_path}"
    )


# ============================================================
# FIGURE 1 — PARITY
# ============================================================


def parity_limits(
    y_true,
    gnn_pred,
    baseline_pred,
):
    values = np.concatenate(
        [
            np.asarray(
                y_true
            ),
            np.asarray(
                gnn_pred
            ),
            np.asarray(
                baseline_pred
            ),
        ]
    )

    minimum = float(
        np.min(values)
    )

    maximum = float(
        np.max(values)
    )

    span = (
        maximum
        - minimum
    )

    padding = (
        0.05 * span
        if span > 0
        else 1.0
    )

    return (
        minimum - padding,
        maximum + padding,
    )


def add_metric_box(
    ax,
    gnn_mae,
    gnn_r2,
    baseline_mae,
    baseline_r2,
    unit,
):
    text = (
        r"$\bf{Crystal\ GNN}$"
        "\n"
        f"MAE = {gnn_mae:.3f} {unit}"
        "\n"
        f"R² = {gnn_r2:.3f}"
        "\n\n"
        r"$\bf{ExtraTrees}$"
        "\n"
        f"MAE = {baseline_mae:.3f} {unit}"
        "\n"
        f"R² = {baseline_r2:.3f}"
    )

    ax.text(
        0.97,
        0.04,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=INK,
        bbox={
            "boxstyle": (
                "round,pad=0.45"
            ),
            "facecolor": WHITE,
            "edgecolor": INK,
            "linewidth": 1.0,
            "alpha": 0.92,
        },
        zorder=10,
    )


def draw_parity_panel(
    ax,
    y_true,
    gnn_pred,
    baseline_pred,
    title,
    unit,
    gnn_mae,
    gnn_r2,
    baseline_mae,
    baseline_r2,
):
    style_axis(
        ax
    )

    # Draw baseline first.
    ax.scatter(
        y_true,
        baseline_pred,
        s=23,
        marker="D",
        color=CRIMSON,
        alpha=0.46,
        edgecolor=WHITE,
        linewidth=0.28,
        label="ExtraTrees",
        zorder=2,
    )

    # GNN visually dominant.
    ax.scatter(
        y_true,
        gnn_pred,
        s=27,
        marker="o",
        color=COBALT,
        alpha=0.74,
        edgecolor=WHITE,
        linewidth=0.32,
        label="Crystal GNN",
        zorder=3,
    )

    low, high = parity_limits(
        y_true,
        gnn_pred,
        baseline_pred,
    )

    ax.plot(
        [low, high],
        [low, high],
        color=INK,
        linewidth=1.8,
        linestyle="--",
        label="Ideal prediction",
        zorder=1,
    )

    ax.set_xlim(
        low,
        high,
    )

    ax.set_ylim(
        low,
        high,
    )

    ax.set_xlabel(
        f"DFT value ({unit})",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_ylabel(
        f"Predicted value ({unit})",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=11,
    )

    add_metric_box(
        ax=ax,
        gnn_mae=gnn_mae,
        gnn_r2=gnn_r2,
        baseline_mae=baseline_mae,
        baseline_r2=baseline_r2,
        unit=unit,
    )


def make_parity_figure(
    data,
    comparison,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            13.2,
            5.7,
        ),
        facecolor=WHITE,
    )

    formation_gnn = comparison[
        (
            comparison["model"]
            == "CrystalGNN"
        )
        & (
            comparison["target"]
            == "formation_energy_peratom"
        )
        & (
            comparison["subset"]
            == "all_test"
        )
    ].iloc[0]

    formation_baseline = comparison[
        (
            comparison["model"]
            == "ExtraTrees"
        )
        & (
            comparison["target"]
            == "formation_energy_peratom"
        )
        & (
            comparison["subset"]
            == "all_test"
        )
    ].iloc[0]

    bandgap_gnn = comparison[
        (
            comparison["model"]
            == "CrystalGNN"
        )
        & (
            comparison["target"]
            == "optb88vdw_bandgap"
        )
        & (
            comparison["subset"]
            == "all_test"
        )
    ].iloc[0]

    bandgap_baseline = comparison[
        (
            comparison["model"]
            == "ExtraTrees"
        )
        & (
            comparison["target"]
            == "optb88vdw_bandgap"
        )
        & (
            comparison["subset"]
            == "all_test"
        )
    ].iloc[0]

    draw_parity_panel(
        ax=axes[0],
        y_true=data[
            "formation_true"
        ],
        gnn_pred=data[
            "formation_pred"
        ],
        baseline_pred=data[
            "formation_pred_base"
        ],
        title="Formation Energy",
        unit="eV/atom",
        gnn_mae=float(
            formation_gnn["mae"]
        ),
        gnn_r2=float(
            formation_gnn["r2"]
        ),
        baseline_mae=float(
            formation_baseline["mae"]
        ),
        baseline_r2=float(
            formation_baseline["r2"]
        ),
    )

    draw_parity_panel(
        ax=axes[1],
        y_true=data[
            "bandgap_true"
        ],
        gnn_pred=data[
            "bandgap_pred"
        ],
        baseline_pred=data[
            "bandgap_pred_base"
        ],
        title="OptB88-vdW Band Gap",
        unit="eV",
        gnn_mae=float(
            bandgap_gnn["mae"]
        ),
        gnn_r2=float(
            bandgap_gnn["r2"]
        ),
        baseline_mae=float(
            bandgap_baseline["mae"]
        ),
        baseline_r2=float(
            bandgap_baseline["r2"]
        ),
    )

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.935,
        ),
        ncol=3,
        frameon=True,
        facecolor=WHITE,
        edgecolor=INK,
        framealpha=0.96,
        fontsize=10,
    )

    fig.suptitle(
        "Frozen-Test Prediction Performance",
        fontsize=18,
        fontweight="bold",
        color=INK,
        y=1.01,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.90,
        ]
    )

    save_figure(
        fig,
        "figure_01_test_parity",
    )


# ============================================================
# FIGURE 2 — TRAINING
# ============================================================


def make_training_figure(
    history,
):
    fig, ax = plt.subplots(
        figsize=(
            8.8,
            5.6,
        ),
        facecolor=WHITE,
    )

    style_axis(
        ax
    )

    ax.plot(
        history["epoch"],
        history["train_loss"],
        color=COBALT,
        linewidth=3.0,
        marker="o",
        markersize=3.5,
        markevery=2,
        label="Training loss",
        zorder=3,
    )

    ax.plot(
        history["epoch"],
        history["val_loss"],
        color=CRIMSON,
        linewidth=3.0,
        marker="s",
        markersize=3.5,
        markevery=2,
        label="Validation loss",
        zorder=3,
    )

    best_index = (
        history[
            "val_loss"
        ].idxmin()
    )

    best_epoch = int(
        history.loc[
            best_index,
            "epoch",
        ]
    )

    best_loss = float(
        history.loc[
            best_index,
            "val_loss",
        ]
    )

    ax.axvspan(
        best_epoch - 0.55,
        best_epoch + 0.55,
        color=EMERALD,
        alpha=0.15,
        zorder=1,
    )

    ax.axvline(
        best_epoch,
        color=EMERALD,
        linewidth=2.2,
        linestyle="--",
        zorder=2,
    )

    ax.scatter(
        best_epoch,
        best_loss,
        s=155,
        color=EMERALD,
        edgecolor=INK,
        linewidth=1.2,
        zorder=6,
        label=(
            f"Best epoch = "
            f"{best_epoch}"
        ),
    )

    ax.annotate(
        (
            f"Best validation loss\n"
            f"{best_loss:.3f}"
        ),
        xy=(
            best_epoch,
            best_loss,
        ),
        xytext=(
            best_epoch + 2.3,
            best_loss + 0.22,
        ),
        fontsize=10,
        fontweight="bold",
        color=EMERALD,
        arrowprops={
            "arrowstyle": "->",
            "color": EMERALD,
            "linewidth": 1.8,
        },
    )

    ax.set_xlabel(
        "Epoch",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Combined standardized MSE",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_title(
        "GNN Training Dynamics and Model Selection",
        fontsize=17,
        fontweight="bold",
        pad=13,
    )

    ax.legend(
        frameon=True,
        facecolor=WHITE,
        edgecolor=INK,
        framealpha=0.96,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_02_training_curves",
    )


# ============================================================
# FIGURE 3 — ERROR DISTRIBUTIONS
# ============================================================


def make_error_figure(
    data,
):
    formation_gnn = data[
        "formation_abs_error_gnn"
    ].to_numpy()

    formation_baseline = data[
        "formation_abs_error_baseline"
    ].to_numpy()

    gap_gnn = data[
        "bandgap_abs_error_gnn"
    ].to_numpy()

    gap_baseline = data[
        "bandgap_abs_error_baseline"
    ].to_numpy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            13.0,
            5.5,
        ),
        facecolor=WHITE,
    )

    for ax in axes:
        style_axis(
            ax
        )

    formation_limit = float(
        np.percentile(
            np.concatenate(
                [
                    formation_gnn,
                    formation_baseline,
                ]
            ),
            98,
        )
    )

    gap_limit = float(
        np.percentile(
            np.concatenate(
                [
                    gap_gnn,
                    gap_baseline,
                ]
            ),
            98,
        )
    )

    formation_bins = np.linspace(
        0,
        formation_limit,
        34,
    )

    gap_bins = np.linspace(
        0,
        gap_limit,
        34,
    )

    axes[0].hist(
        formation_baseline,
        bins=formation_bins,
        color=CRIMSON,
        alpha=0.55,
        edgecolor=INK,
        linewidth=0.35,
        label="ExtraTrees",
        zorder=2,
    )

    axes[0].hist(
        formation_gnn,
        bins=formation_bins,
        color=COBALT,
        alpha=0.76,
        edgecolor=WHITE,
        linewidth=0.35,
        label="Crystal GNN",
        zorder=3,
    )

    axes[1].hist(
        gap_baseline,
        bins=gap_bins,
        color=ORANGE,
        alpha=0.68,
        edgecolor=INK,
        linewidth=0.35,
        label="ExtraTrees",
        zorder=2,
    )

    axes[1].hist(
        gap_gnn,
        bins=gap_bins,
        color=VIOLET,
        alpha=0.76,
        edgecolor=WHITE,
        linewidth=0.35,
        label="Crystal GNN",
        zorder=3,
    )

    axes[0].set_title(
        "Formation-Energy Error",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    axes[1].set_title(
        "Band-Gap Error",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    axes[0].set_xlabel(
        "Absolute error (eV/atom)",
        fontsize=11,
        fontweight="bold",
    )

    axes[1].set_xlabel(
        "Absolute error (eV)",
        fontsize=11,
        fontweight="bold",
    )

    for ax in axes:
        ax.set_ylabel(
            "Number of test materials",
            fontsize=11,
            fontweight="bold",
        )

        ax.legend(
            frameon=True,
            facecolor=WHITE,
            edgecolor=INK,
            framealpha=0.96,
        )

    fig.suptitle(
        "Frozen-Test Absolute-Error Distributions",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_03_error_distributions",
    )


# ============================================================
# FIGURE 4 — BAND-GAP SUBSETS
# ============================================================


def get_bandgap_mae(
    comparison,
    model,
    subset,
):
    row = comparison[
        (
            comparison["model"]
            == model
        )
        & (
            comparison["target"]
            == "optb88vdw_bandgap"
        )
        & (
            comparison["subset"]
            == subset
        )
    ]

    if len(row) != 1:
        raise ValueError(
            "Expected exactly one "
            f"{model}/{subset} row."
        )

    return float(
        row[
            "mae"
        ].iloc[0]
    )


def make_bandgap_comparison(
    comparison,
):
    subsets = [
        "all_test",
        "zero_gap_test",
        "nonzero_gap_test",
    ]

    labels = [
        "All test\nn = 800",
        "Zero gap\nn = 564",
        "Nonzero gap\nn = 236",
    ]

    gnn_values = [
        get_bandgap_mae(
            comparison,
            "CrystalGNN",
            subset,
        )
        for subset in subsets
    ]

    baseline_values = [
        get_bandgap_mae(
            comparison,
            "ExtraTrees",
            subset,
        )
        for subset in subsets
    ]

    x = np.arange(
        len(subsets)
    )

    width = 0.34

    fig, ax = plt.subplots(
        figsize=(
            9.4,
            5.9,
        ),
        facecolor=WHITE,
    )

    style_axis(
        ax
    )

    gnn_bars = ax.bar(
        x - width / 2,
        gnn_values,
        width,
        color=COBALT,
        edgecolor=INK,
        linewidth=1.05,
        label="Crystal GNN",
        zorder=3,
    )

    baseline_bars = ax.bar(
        x + width / 2,
        baseline_values,
        width,
        color=CRIMSON,
        edgecolor=INK,
        linewidth=1.05,
        label="ExtraTrees",
        zorder=3,
    )

    ax.bar_label(
        gnn_bars,
        fmt="%.3f",
        padding=5,
        fontsize=10.5,
        fontweight="bold",
        color=COBALT,
    )

    ax.bar_label(
        baseline_bars,
        fmt="%.3f",
        padding=5,
        fontsize=10.5,
        fontweight="bold",
        color=CRIMSON,
    )

    for index, (
        gnn_value,
        baseline_value,
    ) in enumerate(
        zip(
            gnn_values,
            baseline_values,
        )
    ):
        improvement = (
            (
                baseline_value
                - gnn_value
            )
            / baseline_value
            * 100.0
        )

        ax.text(
            index,
            max(
                gnn_value,
                baseline_value,
            )
            + 0.095,
            (
                f"↓ {improvement:.1f}% "
                "MAE"
            ),
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=EMERALD,
        )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_ylabel(
        "Band-gap MAE (eV)",
        fontsize=11.5,
        fontweight="bold",
    )

    ax.set_ylim(
        0,
        max(
            baseline_values
        )
        * 1.28,
    )

    ax.set_title(
        "Band-Gap Performance Across Test Regimes",
        fontsize=17,
        fontweight="bold",
        pad=13,
    )

    ax.text(
        0.5,
        1.01,
        (
            "Lower MAE is better; "
            "percentages show GNN error reduction"
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color=CHARCOAL,
    )

    ax.legend(
        frameon=True,
        facecolor=WHITE,
        edgecolor=INK,
        framealpha=0.96,
        loc="upper left",
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_04_bandgap_subset_comparison",
    )


# ============================================================
# MAIN
# ============================================================


def main():
    configure_style()

    print(
        "[INFO] Loading finalized "
        "analysis tables..."
    )

    data = pd.read_csv(
        DATA_PATH
    )

    comparison = pd.read_csv(
        COMPARISON_PATH
    )

    history = pd.read_csv(
        HISTORY_PATH
    )

    print(
        "[INFO] Generating Figure 1: "
        "test parity..."
    )

    make_parity_figure(
        data,
        comparison,
    )

    print(
        "[INFO] Generating Figure 2: "
        "training curves..."
    )

    make_training_figure(
        history
    )

    print(
        "[INFO] Generating Figure 3: "
        "error distributions..."
    )

    make_error_figure(
        data
    )

    print(
        "[INFO] Generating Figure 4: "
        "band-gap subsets..."
    )

    make_bandgap_comparison(
        comparison
    )

    print()
    print(
        "[OK] Final publication figures "
        "generated."
    )


if __name__ == "__main__":
    main()