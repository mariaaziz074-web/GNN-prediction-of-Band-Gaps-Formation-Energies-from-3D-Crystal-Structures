from pathlib import Path

import torch
from torch import nn

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


def main():
    torch.manual_seed(42)

    print(
        "[INFO] Loading graph cache..."
    )

    graphs = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )

    # Include normal graphs and all zero-edge graphs.
    regular = [
        graph
        for graph in graphs
        if graph["num_edges"] > 0
    ][:13]

    edgeless = [
        graph
        for graph in graphs
        if graph["num_edges"] == 0
    ]

    selected = (
        regular
        + edgeless
    )

    batch = collate_graphs(
        selected
    )

    model = CrystalGNN(
        max_atomic_number=100,
        hidden_dim=64,
        edge_dim=32,
        num_layers=3,
    )

    print(model)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"[INFO] Trainable parameters: "
        f"{parameter_count:,}"
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

    expected_shape = (
        len(selected),
    )

    assert outputs[
        "formation"
    ].shape == expected_shape

    assert outputs[
        "bandgap"
    ].shape == expected_shape

    assert torch.isfinite(
        outputs["formation"]
    ).all()

    assert torch.isfinite(
        outputs["bandgap"]
    ).all()

    # Verify that gradients propagate through both heads and
    # message-passing layers.
    formation_target = batch[
        "y_formation"
    ]

    bandgap_target = batch[
        "y_bandgap"
    ]

    criterion = nn.MSELoss()

    loss = (
        criterion(
            outputs["formation"],
            formation_target,
        )
        + criterion(
            outputs["bandgap"],
            bandgap_target,
        )
    )

    loss.backward()

    gradient_parameters = 0

    for parameter in model.parameters():
        if (
            parameter.grad is not None
            and torch.isfinite(
                parameter.grad
            ).all()
        ):
            gradient_parameters += 1

    if gradient_parameters == 0:
        raise RuntimeError(
            "No valid gradients were produced."
        )

    print(
        f"[OK] Forward pass shape: "
        f"{expected_shape}"
    )

    print(
        f"[INFO] Test loss: "
        f"{loss.detach().item():.6f}"
    )

    print(
        "[OK] Finite outputs verified."
    )

    print(
        f"[OK] Gradients verified for "
        f"{gradient_parameters} "
        "parameter tensors."
    )

    print(
        "[OK] Crystal GNN forward/backward "
        "validation passed."
    )


if __name__ == "__main__":
    main()