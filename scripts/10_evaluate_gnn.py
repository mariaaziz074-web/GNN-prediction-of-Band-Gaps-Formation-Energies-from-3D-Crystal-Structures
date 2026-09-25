from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch.utils.data import DataLoader

from gnn_materials.data.batching import (
    collate_graphs,
)
from gnn_materials.models.crystal_gnn import (
    CrystalGNN,
)


GRAPH_PATH = Path(
    "data/processed/"
    "jarvis_dft_8000_graphs.pt"
)

MODEL_PATH = Path(
    "results/models/"
    "crystal_gnn_best.pt"
)

METRICS_PATH = Path(
    "results/tables/"
    "gnn_test_metrics.csv"
)

PREDICTIONS_PATH = Path(
    "results/tables/"
    "gnn_test_predictions.csv"
)

COMPARISON_PATH = Path(
    "results/tables/"
    "gnn_vs_classical_test_metrics.csv"
)

BASELINE_METRICS_PATH = Path(
    "results/tables/"
    "classical_baseline_metrics.csv"
)

BATCH_SIZE = 64


def regression_metrics(
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


def main():
    torch.manual_seed(42)

    print(
        "[INFO] Loading frozen graph "
        "cache and best checkpoint..."
    )

    graphs = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    test_graphs = [
        graph
        for graph in graphs
        if graph["split"] == "test"
    ]

    if len(test_graphs) != 800:
        raise ValueError(
            "Expected exactly 800 "
            f"test graphs, found "
            f"{len(test_graphs)}."
        )

    stats = checkpoint[
        "target_stats"
    ]

    model = CrystalGNN(
        max_atomic_number=100,
        hidden_dim=checkpoint[
            "hidden_dim"
        ],
        edge_dim=checkpoint[
            "edge_dim"
        ],
        num_layers=checkpoint[
            "num_layers"
        ],
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    loader = DataLoader(
        test_graphs,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_graphs,
    )

    rows = []

    print(
        f"[INFO] Evaluating best "
        f"epoch {checkpoint['best_epoch']} "
        "on 800 frozen test graphs..."
    )

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                z=batch["z"],
                edge_index=batch[
                    "edge_index"
                ],
                edge_attr=batch[
                    "edge_attr"
                ],
                batch=batch[
                    "batch"
                ],
                num_graphs=batch[
                    "num_graphs"
                ],
            )

            pred_formation = (
                outputs["formation"]
                * stats[
                    "formation_std"
                ]
                + stats[
                    "formation_mean"
                ]
            )

            pred_bandgap = (
                outputs["bandgap"]
                * stats[
                    "bandgap_std"
                ]
                + stats[
                    "bandgap_mean"
                ]
            )

            true_formation = batch[
                "y_formation"
            ]

            true_bandgap = batch[
                "y_bandgap"
            ]

            for i, jid in enumerate(
                batch["jids"]
            ):
                rows.append(
                    {
                        "jid": jid,
                        "formation_true": (
                            true_formation[
                                i
                            ].item()
                        ),
                        "formation_pred": (
                            pred_formation[
                                i
                            ].item()
                        ),
                        "bandgap_true": (
                            true_bandgap[
                                i
                            ].item()
                        ),
                        "bandgap_pred": (
                            pred_bandgap[
                                i
                            ].item()
                        ),
                    }
                )

    predictions = pd.DataFrame(
        rows
    )

    if len(predictions) != 800:
        raise ValueError(
            "Prediction count mismatch."
        )

    formation_metrics = (
        regression_metrics(
            predictions[
                "formation_true"
            ].to_numpy(),
            predictions[
                "formation_pred"
            ].to_numpy(),
        )
    )

    bandgap_metrics = (
        regression_metrics(
            predictions[
                "bandgap_true"
            ].to_numpy(),
            predictions[
                "bandgap_pred"
            ].to_numpy(),
        )
    )

    nonzero = (
        predictions[
            "bandgap_true"
        ] > 0.0
    )

    n_nonzero = int(
        nonzero.sum()
    )

    if n_nonzero < 2:
        raise ValueError(
            "Too few nonzero-gap test "
            "materials for evaluation."
        )

    nonzero_bandgap_metrics = (
        regression_metrics(
            predictions.loc[
                nonzero,
                "bandgap_true",
            ].to_numpy(),
            predictions.loc[
                nonzero,
                "bandgap_pred",
            ].to_numpy(),
        )
    )

    metric_rows = [
        {
            "model": "CrystalGNN",
            "target": (
                "formation_energy_peratom"
            ),
            "subset": "all_test",
            "n": 800,
            **formation_metrics,
        },
        {
            "model": "CrystalGNN",
            "target": (
                "optb88vdw_bandgap"
            ),
            "subset": "all_test",
            "n": 800,
            **bandgap_metrics,
        },
        {
            "model": "CrystalGNN",
            "target": (
                "optb88vdw_bandgap"
            ),
            "subset": (
                "nonzero_gap_test"
            ),
            "n": n_nonzero,
            **nonzero_bandgap_metrics,
        },
    ]

    metrics = pd.DataFrame(
        metric_rows
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    metrics.to_csv(
        METRICS_PATH,
        index=False,
    )

    print()
    print(
        "[TEST] Formation energy"
    )
    print(
        f"MAE="
        f"{formation_metrics['mae']:.4f} "
        f"eV/atom | "
        f"RMSE="
        f"{formation_metrics['rmse']:.4f} "
        f"| R2="
        f"{formation_metrics['r2']:.4f}"
    )

    print()
    print(
        "[TEST] Band gap — all 800"
    )
    print(
        f"MAE="
        f"{bandgap_metrics['mae']:.4f} "
        f"eV | "
        f"RMSE="
        f"{bandgap_metrics['rmse']:.4f} "
        f"| R2="
        f"{bandgap_metrics['r2']:.4f}"
    )

    print()
    print(
        "[TEST] Band gap — "
        f"nonzero subset (n={n_nonzero})"
    )
    print(
        f"MAE="
        f"{nonzero_bandgap_metrics['mae']:.4f} "
        f"eV | "
        f"RMSE="
        f"{nonzero_bandgap_metrics['rmse']:.4f} "
        f"| R2="
        f"{nonzero_bandgap_metrics['r2']:.4f}"
    )

    # Direct comparison with the already frozen
    # ExtraTrees baseline.
    if BASELINE_METRICS_PATH.exists():
        baseline = pd.read_csv(
            BASELINE_METRICS_PATH
        )

        test_baseline = baseline[
            baseline["split"]
            .astype(str)
            .str.lower()
            .eq("test")
        ].copy()

        comparison_rows = []

        for target in (
            "formation_energy_peratom",
            "optb88vdw_bandgap",
        ):
            gnn_row = metrics[
                (
                    metrics["target"]
                    == target
                )
                & (
                    metrics["subset"]
                    == "all_test"
                )
            ].iloc[0]

            baseline_row = (
                test_baseline[
                    test_baseline[
                        "target"
                    ]
                    == target
                ]
            )

            if len(baseline_row) == 1:
                baseline_row = (
                    baseline_row.iloc[0]
                )

                comparison_rows.append(
                    {
                        "target": target,
                        "gnn_mae": (
                            gnn_row["mae"]
                        ),
                        "classical_mae": (
                            baseline_row[
                                "mae"
                            ]
                        ),
                        "gnn_rmse": (
                            gnn_row["rmse"]
                        ),
                        "classical_rmse": (
                            baseline_row[
                                "rmse"
                            ]
                        ),
                        "gnn_r2": (
                            gnn_row["r2"]
                        ),
                        "classical_r2": (
                            baseline_row[
                                "r2"
                            ]
                        ),
                    }
                )

        if comparison_rows:
            pd.DataFrame(
                comparison_rows
            ).to_csv(
                COMPARISON_PATH,
                index=False,
            )

            print()
            print(
                "[OK] Classical/GNN "
                "comparison table saved."
            )

    print()
    print(
        f"[OK] Metrics: "
        f"{METRICS_PATH}"
    )
    print(
        f"[OK] Predictions: "
        f"{PREDICTIONS_PATH}"
    )


if __name__ == "__main__":
    main()