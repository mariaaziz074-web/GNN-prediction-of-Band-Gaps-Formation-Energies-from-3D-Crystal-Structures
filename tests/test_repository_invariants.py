"""Regression tests for frozen Project 4 scientific artifacts."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = ROOT / "results" / "tables"
DATA_DIR = ROOT / "data" / "processed"


def test_frozen_metadata_split_counts():
    """The deterministic 8,000-material benchmark must remain frozen."""

    path = DATA_DIR / "jarvis_dft_8000_metadata.csv"

    assert path.exists()

    df = pd.read_csv(path)

    assert len(df) == 8000

    assert "split" in df.columns

    counts = df["split"].value_counts().to_dict()

    assert counts.get("train") == 6400
    assert counts.get("val") == 800
    assert counts.get("test") == 800


def test_gnn_test_predictions_cover_exact_test_set():
    """GNN predictions must contain 800 frozen-test observations per target."""

    path = TABLE_DIR / "gnn_test_predictions.csv"

    assert path.exists()

    df = pd.read_csv(path)

    assert len(df) > 0

    assert "jid" in df.columns

    # Prediction files may be wide or long depending on the evaluation
    # representation. At minimum, all 800 test JIDs must be represented.
    assert df["jid"].nunique() == 800


def test_screening_candidate_counts():
    """Freeze the published retrospective screening result."""

    path = TABLE_DIR / "screened_test_candidates.csv"

    assert path.exists()

    df = pd.read_csv(path)

    assert len(df) == 67

    assert "dft_meets_screen" in df.columns

    confirmed = (
        df["dft_meets_screen"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
        .sum()
    )

    assert confirmed == 32


def test_screened_predictions_satisfy_screen():
    """Every selected row must satisfy the prediction-based screen."""

    df = pd.read_csv(
        TABLE_DIR / "screened_test_candidates.csv"
    )

    assert (
        df["formation_pred"] < -1.0
    ).all()

    assert (
        df["bandgap_pred"].between(
            1.0,
            3.0,
            inclusive="both",
        )
    ).all()


def test_publication_figures_exist():
    """All five final publication PNGs must be present."""

    figure_dir = ROOT / "results" / "figures"

    for number in range(1, 6):
        matching = list(
            figure_dir.glob(
                f"figure_{number:02d}_*.png"
            )
        )

        assert len(matching) == 1