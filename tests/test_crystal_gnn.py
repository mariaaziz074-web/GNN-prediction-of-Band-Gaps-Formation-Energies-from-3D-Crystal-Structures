"""Unit tests for the CrystalGNN architecture."""

import torch

from gnn_materials.models.crystal_gnn import (
    CrystalGNN,
    MessagePassingBlock,
)


EDGE_DIM = 32


def test_message_passing_supports_zero_edges():
    """Message passing must remain finite for isolated nodes."""

    torch.manual_seed(42)

    layer = MessagePassingBlock(
        hidden_dim=16,
        edge_dim=EDGE_DIM,
    )

    x = torch.randn(
        3,
        16,
    )

    edge_index = torch.empty(
        (2, 0),
        dtype=torch.long,
    )

    edge_attr = torch.empty(
        (0, EDGE_DIM),
        dtype=torch.float32,
    )

    output = layer(
        x,
        edge_index,
        edge_attr,
    )

    assert output.shape == (3, 16)
    assert torch.isfinite(output).all()


def test_crystal_gnn_multitask_output_shapes():
    """The model must emit one value per graph for both targets."""

    torch.manual_seed(42)

    model = CrystalGNN(
        max_atomic_number=100,
        hidden_dim=32,
        edge_dim=EDGE_DIM,
        num_layers=2,
    )

    z = torch.tensor(
        [14, 14, 8, 22, 8],
        dtype=torch.long,
    )

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 4],
        ],
        dtype=torch.long,
    )

    edge_attr = torch.randn(
        4,
        EDGE_DIM,
    )

    batch = torch.tensor(
        [0, 0, 1, 1, 1],
        dtype=torch.long,
    )

    output = model(
        z=z,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        num_graphs=2,
    )

    assert set(output) == {
        "formation",
        "bandgap",
    }

    assert output["formation"].shape == (2,)
    assert output["bandgap"].shape == (2,)

    assert torch.isfinite(
        output["formation"]
    ).all()

    assert torch.isfinite(
        output["bandgap"]
    ).all()


def test_crystal_gnn_handles_all_zero_edge_batch():
    """The complete model must support batches with no bonds."""

    torch.manual_seed(42)

    model = CrystalGNN(
        max_atomic_number=100,
        hidden_dim=32,
        edge_dim=EDGE_DIM,
        num_layers=2,
    )

    z = torch.tensor(
        [14, 15],
        dtype=torch.long,
    )

    edge_index = torch.empty(
        (2, 0),
        dtype=torch.long,
    )

    edge_attr = torch.empty(
        (0, EDGE_DIM),
        dtype=torch.float32,
    )

    batch = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    output = model(
        z=z,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        num_graphs=2,
    )

    assert output["formation"].shape == (2,)
    assert output["bandgap"].shape == (2,)

    assert torch.isfinite(
        output["formation"]
    ).all()

    assert torch.isfinite(
        output["bandgap"]
    ).all()


def test_crystal_gnn_backward_pass():
    """Both regression heads must participate in gradient computation."""

    torch.manual_seed(42)

    model = CrystalGNN(
        max_atomic_number=100,
        hidden_dim=32,
        edge_dim=EDGE_DIM,
        num_layers=2,
    )

    z = torch.tensor(
        [6, 8, 14],
        dtype=torch.long,
    )

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=torch.long,
    )

    edge_attr = torch.randn(
        4,
        EDGE_DIM,
    )

    batch = torch.zeros(
        3,
        dtype=torch.long,
    )

    output = model(
        z=z,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        num_graphs=1,
    )

    loss = (
        output["formation"].pow(2).mean()
        + output["bandgap"].pow(2).mean()
    )

    loss.backward()

    assert model.embedding.weight.grad is not None

    assert (
        model.formation_head.weight.grad
        is not None
    )

    assert (
        model.bandgap_head.weight.grad
        is not None
    )

    assert torch.isfinite(
        model.embedding.weight.grad
    ).all()