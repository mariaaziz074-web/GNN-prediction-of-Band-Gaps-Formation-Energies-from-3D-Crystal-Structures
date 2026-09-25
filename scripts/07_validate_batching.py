from pathlib import Path

import torch

from gnn_materials.data.batching import (
    collate_graphs,
)


GRAPH_PATH = Path(
    "data/processed/"
    "jarvis_dft_8000_graphs.pt"
)


def main():
    print(
        "[INFO] Loading graph cache..."
    )

    graphs = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )

    regular = [
        graph
        for graph in graphs
        if graph["num_edges"] > 0
    ][:5]

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

    expected_nodes = sum(
        graph["num_nodes"]
        for graph in selected
    )

    expected_edges = sum(
        graph["num_edges"]
        for graph in selected
    )

    assert batch["num_graphs"] == len(
        selected
    )

    assert batch["z"].shape == (
        expected_nodes,
    )

    assert batch["batch"].shape == (
        expected_nodes,
    )

    assert batch["edge_index"].shape == (
        2,
        expected_edges,
    )

    assert batch["edge_attr"].shape == (
        expected_edges,
        32,
    )

    assert batch["y_formation"].shape == (
        len(selected),
    )

    assert batch["y_bandgap"].shape == (
        len(selected),
    )

    if expected_edges > 0:
        assert int(
            batch["edge_index"].max()
        ) < expected_nodes

    # Test a batch containing only zero-edge graphs.
    zero_batch = collate_graphs(
        edgeless
    )

    assert zero_batch[
        "edge_index"
    ].shape == (
        2,
        0,
    )

    assert zero_batch[
        "edge_attr"
    ].shape == (
        0,
        32,
    )

    assert zero_batch[
        "num_graphs"
    ] == 3

    print(
        f"[OK] Mixed batch graphs: "
        f"{batch['num_graphs']}"
    )

    print(
        f"[INFO] Mixed batch nodes: "
        f"{expected_nodes}"
    )

    print(
        f"[INFO] Mixed batch edges: "
        f"{expected_edges}"
    )

    print(
        "[OK] All-zero-edge batch "
        "validated."
    )

    print(
        "[OK] Native PyTorch crystal "
        "batching validation passed."
    )


if __name__ == "__main__":
    main()