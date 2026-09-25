from pathlib import Path

import numpy as np
import pandas as pd


TABLE_DIR = Path("results/tables")
DATA_DIR = Path("data/processed")

PREDICTIONS_PATH = (
    TABLE_DIR
    / "gnn_test_predictions.csv"
)

METADATA_PATH = (
    DATA_DIR
    / "jarvis_dft_8000_metadata.csv"
)

OUTPUT_PATH = (
    TABLE_DIR
    / "screened_test_candidates.csv"
)

TOP_PATH = (
    TABLE_DIR
    / "top_screened_candidates.csv"
)


FORMATION_THRESHOLD = -1.0
GAP_MIN = 1.0
GAP_MAX = 3.0
TARGET_GAP = 2.0
TOP_N = 20


def main():
    print(
        "[INFO] Loading frozen-test "
        "predictions..."
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
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

    if len(predictions) != 800:
        raise ValueError(
            "Expected exactly 800 "
            "test predictions."
        )

    if len(test_metadata) != 800:
        raise ValueError(
            "Expected exactly 800 "
            "test metadata rows."
        )

    data = predictions.merge(
        test_metadata[
            [
                "jid",
                "formula",
                "nat",
                "space_group_number",
                "crystal_system",
            ]
        ],
        on="jid",
        how="left",
        validate="one_to_one",
    )

    if data["formula"].isna().any():
        raise ValueError(
            "Missing metadata after "
            "JID merge."
        )

    selected = data[
        (
            data["formation_pred"]
            < FORMATION_THRESHOLD
        )
        & (
            data["bandgap_pred"]
            >= GAP_MIN
        )
        & (
            data["bandgap_pred"]
            <= GAP_MAX
        )
    ].copy()

    # Transparent ranking:
    #
    # More-negative formation energy is rewarded.
    # Distance from a 2 eV band gap is penalized.
    #
    # The score is for demonstration/ranking only,
    # not a physical observable.
    selected[
        "screening_score"
    ] = (
        -selected[
            "formation_pred"
        ]
        - np.abs(
            selected[
                "bandgap_pred"
            ]
            - TARGET_GAP
        )
    )

    selected[
        "formation_abs_error"
    ] = np.abs(
        selected[
            "formation_true"
        ]
        - selected[
            "formation_pred"
        ]
    )

    selected[
        "bandgap_abs_error"
    ] = np.abs(
        selected[
            "bandgap_true"
        ]
        - selected[
            "bandgap_pred"
        ]
    )

    # Retrospective DFT confirmation.
    selected[
        "dft_meets_screen"
    ] = (
        (
            selected[
                "formation_true"
            ]
            < FORMATION_THRESHOLD
        )
        & (
            selected[
                "bandgap_true"
            ]
            >= GAP_MIN
        )
        & (
            selected[
                "bandgap_true"
            ]
            <= GAP_MAX
        )
    )

    selected = selected.sort_values(
        by=[
            "screening_score",
            "formation_pred",
        ],
        ascending=[
            False,
            True,
        ],
    )

    selected.insert(
        0,
        "rank",
        np.arange(
            1,
            len(selected) + 1,
        ),
    )

    selected.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    top = selected.head(
        TOP_N
    ).copy()

    top.to_csv(
        TOP_PATH,
        index=False,
    )

    confirmed = int(
        selected[
            "dft_meets_screen"
        ].sum()
    )

    precision = (
        confirmed
        / len(selected)
        if len(selected) > 0
        else float("nan")
    )

    print()
    print(
        "[SCREEN] Criteria"
    )

    print(
        "[SCREEN] Predicted formation "
        f"energy < "
        f"{FORMATION_THRESHOLD:.1f} "
        "eV/atom"
    )

    print(
        "[SCREEN] Predicted band gap "
        f"{GAP_MIN:.1f}-"
        f"{GAP_MAX:.1f} eV"
    )

    print()
    print(
        f"[RESULT] Candidates selected: "
        f"{len(selected)} / 800"
    )

    print(
        "[RESULT] Also satisfy criteria "
        f"using DFT truth: "
        f"{confirmed}"
    )

    if len(selected) > 0:
        print(
            "[RESULT] Retrospective "
            f"precision: "
            f"{precision * 100:.1f}%"
        )

    print()

    if len(top) > 0:
        display_columns = [
            "rank",
            "jid",
            "formula",
            "formation_pred",
            "bandgap_pred",
            "formation_true",
            "bandgap_true",
            "dft_meets_screen",
        ]

        print(
            "[RESULT] Top candidates:"
        )

        print(
            top[
                display_columns
            ]
            .to_string(
                index=False,
            )
        )

    print()
    print(
        f"[OK] Full screen: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"[OK] Top {TOP_N}: "
        f"{TOP_PATH}"
    )

    print()
    print(
        "[NOTE] This is retrospective "
        "screening of materials with "
        "existing JARVIS-DFT labels, "
        "not prospective materials "
        "discovery."
    )


if __name__ == "__main__":
    main()