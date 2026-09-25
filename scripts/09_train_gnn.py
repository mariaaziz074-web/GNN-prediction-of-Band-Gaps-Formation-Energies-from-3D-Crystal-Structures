import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from gnn_materials.data.batching import (
    collate_graphs,
)
from gnn_materials.models.crystal_gnn import (
    CrystalGNN,
)


SEED = 42

GRAPH_PATH = Path(
    "data/processed/"
    "jarvis_dft_8000_graphs.pt"
)

MODEL_DIR = Path(
    "results/models"
)

TABLE_DIR = Path(
    "results/tables"
)

MODEL_PATH = (
    MODEL_DIR
    / "crystal_gnn_best.pt"
)

HISTORY_PATH = (
    TABLE_DIR
    / "gnn_training_history.csv"
)

SCALER_PATH = (
    MODEL_DIR
    / "target_scaler.json"
)

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

MAX_EPOCHS = 100
PATIENCE = 15

HIDDEN_DIM = 64
NUM_LAYERS = 3
EDGE_DIM = 32


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_target_stats(
    graphs,
):
    formation = torch.tensor(
        [
            float(
                graph["y_formation"]
            )
            for graph in graphs
        ],
        dtype=torch.float32,
    )

    bandgap = torch.tensor(
        [
            float(
                graph["y_bandgap"]
            )
            for graph in graphs
        ],
        dtype=torch.float32,
    )

    return {
        "formation_mean": float(
            formation.mean()
        ),
        "formation_std": float(
            formation.std(
                unbiased=False
            )
        ),
        "bandgap_mean": float(
            bandgap.mean()
        ),
        "bandgap_std": float(
            bandgap.std(
                unbiased=False
            )
        ),
    }


def standardized_targets(
    batch,
    stats,
):
    formation = (
        batch["y_formation"]
        - stats["formation_mean"]
    ) / stats["formation_std"]

    bandgap = (
        batch["y_bandgap"]
        - stats["bandgap_mean"]
    ) / stats["bandgap_std"]

    return formation, bandgap


def run_epoch(
    model,
    loader,
    optimizer,
    stats,
    training,
):
    if training:
        model.train()
    else:
        model.eval()

    criterion = nn.MSELoss()

    total_loss = 0.0
    total_graphs = 0

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for batch in loader:
            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            outputs = model(
                z=batch["z"],
                edge_index=batch[
                    "edge_index"
                ],
                edge_attr=batch[
                    "edge_attr"
                ],
                batch=batch["batch"],
                num_graphs=batch[
                    "num_graphs"
                ],
            )

            target_formation, target_bandgap = (
                standardized_targets(
                    batch,
                    stats,
                )
            )

            formation_loss = criterion(
                outputs["formation"],
                target_formation,
            )

            bandgap_loss = criterion(
                outputs["bandgap"],
                target_bandgap,
            )

            loss = (
                formation_loss
                + bandgap_loss
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

            batch_size = int(
                batch["num_graphs"]
            )

            total_loss += (
                loss.detach().item()
                * batch_size
            )

            total_graphs += batch_size

    return (
        total_loss
        / total_graphs
    )


def main():
    set_seed(SEED)

    # CPU-only project configuration.
    torch.set_num_threads(
        max(
            1,
            min(
                4,
                torch.get_num_threads(),
            ),
        )
    )

    print(
        "[INFO] Loading validated "
        "graph cache..."
    )

    graphs = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )

    train_graphs = [
        graph
        for graph in graphs
        if graph["split"] == "train"
    ]

    val_graphs = [
        graph
        for graph in graphs
        if graph["split"] == "val"
    ]

    test_graphs = [
        graph
        for graph in graphs
        if graph["split"] == "test"
    ]

    print(
        f"[INFO] Train/val/test: "
        f"{len(train_graphs):,}/"
        f"{len(val_graphs):,}/"
        f"{len(test_graphs):,}"
    )

    stats = compute_target_stats(
        train_graphs
    )

    if (
        stats["formation_std"] <= 0
        or stats["bandgap_std"] <= 0
    ):
        raise ValueError(
            "Invalid training target "
            "standard deviation."
        )

    print(
        "[INFO] Training-only target "
        "statistics:"
    )

    print(
        "[INFO] Formation: "
        f"mean="
        f"{stats['formation_mean']:.6f}, "
        f"std="
        f"{stats['formation_std']:.6f}"
    )

    print(
        "[INFO] Band gap:  "
        f"mean="
        f"{stats['bandgap_mean']:.6f}, "
        f"std="
        f"{stats['bandgap_std']:.6f}"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SCALER_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            stats,
            handle,
            indent=2,
        )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_graphs,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_graphs,
        generator=generator,
    )

    val_loader = DataLoader(
        val_graphs,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_graphs,
    )

    model = CrystalGNN(
        max_atomic_number=100,
        hidden_dim=HIDDEN_DIM,
        edge_dim=EDGE_DIM,
        num_layers=NUM_LAYERS,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"[INFO] Trainable parameters: "
        f"{parameter_count:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
        )
    )

    history = []

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(
        "[INFO] Starting CPU training..."
    )

    training_start = time.time()

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        epoch_start = time.time()

        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            stats=stats,
            training=True,
        )

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            stats=stats,
            training=False,
        )

        scheduler.step(
            val_loss
        )

        learning_rate = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": (
                    train_loss
                ),
                "val_loss": val_loss,
                "learning_rate": (
                    learning_rate
                ),
                "epoch_seconds": (
                    epoch_seconds
                ),
            }
        )

        improved = (
            val_loss
            < best_val_loss - 1e-6
        )

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "target_stats": stats,
                    "seed": SEED,
                    "hidden_dim": (
                        HIDDEN_DIM
                    ),
                    "num_layers": (
                        NUM_LAYERS
                    ),
                    "edge_dim": EDGE_DIM,
                    "best_epoch": (
                        best_epoch
                    ),
                    "best_val_loss": (
                        best_val_loss
                    ),
                },
                MODEL_PATH,
            )

        else:
            epochs_without_improvement += 1

        print(
            f"[EPOCH {epoch:03d}] "
            f"train={train_loss:.5f} | "
            f"val={val_loss:.5f} | "
            f"lr={learning_rate:.2e} | "
            f"{epoch_seconds:.1f}s"
        )

        pd.DataFrame(
            history
        ).to_csv(
            HISTORY_PATH,
            index=False,
        )

        if (
            epochs_without_improvement
            >= PATIENCE
        ):
            print(
                "[INFO] Early stopping "
                f"after epoch {epoch}."
            )
            break

    total_minutes = (
        time.time()
        - training_start
    ) / 60.0

    print()
    print(
        "[OK] Training complete."
    )

    print(
        f"[INFO] Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"[INFO] Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"[INFO] Total training time: "
        f"{total_minutes:.2f} min"
    )

    print(
        f"[OK] Best model: "
        f"{MODEL_PATH}"
    )

    print(
        f"[OK] Training history: "
        f"{HISTORY_PATH}"
    )

    # The frozen test set is deliberately not evaluated here.
    # Test evaluation is performed separately after model selection.


if __name__ == "__main__":
    main()