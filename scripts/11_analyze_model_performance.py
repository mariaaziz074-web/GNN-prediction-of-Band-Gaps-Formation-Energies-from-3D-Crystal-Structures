from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


GNN_PATH = Path(
    "results/tables/"
    "gnn_test_predictions.csv"
)

BASELINE_PATH = Path(
    "results/tables/"
    "classical_baseline_predictions.csv"
)

METADATA_PATH = Path(
    "data/processed/"
    "jarvis_dft_8000_metadata.csv"
)

OUTPUT_METRICS = Path(
    "results/tables/"
    "detailed_model_comparison.csv"
)

OUTPUT_ERRORS = Path(
    "results/tables/"
    "test_error_analysis.csv"
)

OUTPUT_WORST = Path(
    "results/tables/"
    "worst_gnn_predictions.csv"
)


def metrics(
    y_true,
    y_pred,
):
    return {
        "mae": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),
        "r2": r2_score(
            y_true,
            y_pred,
        ),
    }


def find_column(
    frame,
    candidates,
):
    for column in candidates:
        if column in frame.columns:
            return column

    raise ValueError(
        "None of the expected columns "
        f"were found: {candidates}. "
        f"Available columns: "
        f"{list(frame.columns)}"
    )


def main():
    print(
        "[INFO] Loading prediction "
        "tables..."
    )

    gnn = pd.read_csv(
        GNN_PATH
    )

    baseline = pd.read_csv(
        BASELINE_PATH
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    test_metadata = (
        metadata[
            metadata["split"]
            .astype(str)
            .eq("test")
        ]
        .copy()
    )

    if len(gnn) != 800:
        raise ValueError(
            "Expected 800 GNN test "
            "predictions."
        )

    print(
        "[INFO] Classical prediction "
        f"columns: {list(baseline.columns)}"
    )

    # Locate baseline columns robustly.
    jid_col = find_column(
        baseline,
        [
            "jid",
            "JID",
        ],
    )

    split_col = find_column(
        baseline,
        [
            "split",
            "Split",
        ],
    )

    target_col = find_column(
        baseline,
        [
            "target",
            "property",
        ],
    )

    true_col = find_column(
        baseline,
        [
            "y_true",
            "true",
            "actual",
        ],
    )

    pred_col = find_column(
        baseline,
        [
            "y_pred",
            "pred",
            "prediction",
            "predicted",
        ],
    )

    baseline_test = baseline[
        baseline[split_col]
        .astype(str)
        .str.lower()
        .eq("test")
    ].copy()

    formation_base = baseline_test[
        baseline_test[target_col]
        == "formation_energy_peratom"
    ][
        [
            jid_col,
            true_col,
            pred_col,
        ]
    ].copy()

    formation_base.columns = [
        "jid",
        "formation_true_base",
        "formation_pred_base",
    ]

    bandgap_base = baseline_test[
        baseline_test[target_col]
        == "optb88vdw_bandgap"
    ][
        [
            jid_col,
            true_col,
            pred_col,
        ]
    ].copy()

    bandgap_base.columns = [
        "jid",
        "bandgap_true_base",
        "bandgap_pred_base",
    ]

    combined = (
        gnn
        .merge(
            formation_base,
            on="jid",
            how="left",
            validate="one_to_one",
        )
        .merge(
            bandgap_base,
            on="jid",
            how="left",
            validate="one_to_one",
        )
        .merge(
            test_metadata[
                [
                    "jid",
                    "formula",
                    "nat",
                    "crystal_system",
                ]
            ],
            on="jid",
            how="left",
            validate="one_to_one",
        )
    )

    if len(combined) != 800:
        raise ValueError(
            "Merged test dataset does "
            "not contain 800 rows."
        )

    required = [
        "formation_pred_base",
        "bandgap_pred_base",
    ]

    if combined[
        required
    ].isna().any().any():
        raise ValueError(
            "Missing classical predictions "
            "after JID merge."
        )

    # Verify both model files reference
    # identical ground-truth values.
    if not np.allclose(
        combined[
            "formation_true"
        ],
        combined[
            "formation_true_base"
        ],
        atol=1e-5,
    ):
        raise ValueError(
            "Formation target mismatch "
            "between model tables."
        )

    if not np.allclose(
        combined[
            "bandgap_true"
        ],
        combined[
            "bandgap_true_base"
        ],
        atol=1e-5,
    ):
        raise ValueError(
            "Band-gap target mismatch "
            "between model tables."
        )

    combined[
        "formation_abs_error_gnn"
    ] = np.abs(
        combined["formation_true"]
        - combined["formation_pred"]
    )

    combined[
        "formation_abs_error_baseline"
    ] = np.abs(
        combined["formation_true"]
        - combined[
            "formation_pred_base"
        ]
    )

    combined[
        "bandgap_abs_error_gnn"
    ] = np.abs(
        combined["bandgap_true"]
        - combined["bandgap_pred"]
    )

    combined[
        "bandgap_abs_error_baseline"
    ] = np.abs(
        combined["bandgap_true"]
        - combined[
            "bandgap_pred_base"
        ]
    )

    combined[
        "gap_subset"
    ] = np.where(
        combined["bandgap_true"] > 0,
        "nonzero",
        "zero",
    )

    comparison_rows = []

    model_specs = {
        "CrystalGNN": {
            "formation": (
                "formation_pred"
            ),
            "bandgap": (
                "bandgap_pred"
            ),
        },
        "ExtraTrees": {
            "formation": (
                "formation_pred_base"
            ),
            "bandgap": (
                "bandgap_pred_base"
            ),
        },
    }

    for model_name, columns in (
        model_specs.items()
    ):
        formation_result = metrics(
            combined[
                "formation_true"
            ],
            combined[
                columns["formation"]
            ],
        )

        comparison_rows.append(
            {
                "model": model_name,
                "target": (
                    "formation_energy_peratom"
                ),
                "subset": "all_test",
                "n": 800,
                **formation_result,
            }
        )

        for subset_name, mask in [
            (
                "all_test",
                np.ones(
                    len(combined),
                    dtype=bool,
                ),
            ),
            (
                "zero_gap_test",
                combined[
                    "bandgap_true"
                ].eq(0).to_numpy(),
            ),
            (
                "nonzero_gap_test",
                combined[
                    "bandgap_true"
                ].gt(0).to_numpy(),
            ),
        ]:
            result = metrics(
                combined.loc[
                    mask,
                    "bandgap_true",
                ],
                combined.loc[
                    mask,
                    columns["bandgap"],
                ],
            )

            comparison_rows.append(
                {
                    "model": model_name,
                    "target": (
                        "optb88vdw_bandgap"
                    ),
                    "subset": (
                        subset_name
                    ),
                    "n": int(
                        mask.sum()
                    ),
                    **result,
                }
            )

    comparison = pd.DataFrame(
        comparison_rows
    )

    OUTPUT_METRICS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_METRICS,
        index=False,
    )

    combined.to_csv(
        OUTPUT_ERRORS,
        index=False,
    )

    worst_formation = (
        combined
        .nlargest(
            10,
            "formation_abs_error_gnn",
        )
        .assign(
            error_target=(
                "formation_energy_peratom"
            )
        )
    )

    worst_bandgap = (
        combined
        .nlargest(
            10,
            "bandgap_abs_error_gnn",
        )
        .assign(
            error_target=(
                "optb88vdw_bandgap"
            )
        )
    )

    worst = pd.concat(
        [
            worst_formation,
            worst_bandgap,
        ],
        ignore_index=True,
    )

    worst.to_csv(
        OUTPUT_WORST,
        index=False,
    )

    print()
    print(
        "[RESULT] Detailed test "
        "comparison:"
    )

    for _, row in (
        comparison.iterrows()
    ):
        print(
            f"{row['model']:10s} | "
            f"{row['target']:28s} | "
            f"{row['subset']:16s} | "
            f"n={int(row['n']):3d} | "
            f"MAE={row['mae']:.4f} | "
            f"RMSE={row['rmse']:.4f} | "
            f"R2={row['r2']:.4f}"
        )

    print()
    print(
        f"[OK] Comparison: "
        f"{OUTPUT_METRICS}"
    )

    print(
        f"[OK] Error analysis: "
        f"{OUTPUT_ERRORS}"
    )

    print(
        f"[OK] Worst predictions: "
        f"{OUTPUT_WORST}"
    )


if __name__ == "__main__":
    main()