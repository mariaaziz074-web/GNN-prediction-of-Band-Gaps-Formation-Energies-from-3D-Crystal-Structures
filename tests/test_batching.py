"""Unit tests for native-PyTorch crystal graph batching."""

import torch

from gnn_materials.data.batching import collate_graphs


RBF_DIM = 32


def make_graph(
    jid,
    atomic_numbers,
    edges,
    formation,
    bandgap,
):
    """Construct a minimal graph dictionary for batching tests."""

    z = torch.tensor(
        atomic_numbers,
        dtype=torch.long,
    )

    if edges:
        edge_index = torch.tensor(
            edges,
            dtype=torch.long,
        ).T.contiguous()

        edge_attr = torch.ones(
            (len(edges), RBF_DIM),
            dtype=torch.float32,
        )
    else:
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

        edge_attr = torch.empty(
            (0, RBF_DIM),
            dtype=torch.float32,
        )

    return {
        "jid": jid,
        "z": z,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y_formation": torch.tensor(
            formation,
            dtype=torch.float32,
        ),
        "y_bandgap": torch.tensor(
            bandgap,
            dtype=torch.float32,
        ),
    }


def test_collate_graphs_offsets_edges_and_labels():
    """Edges must be offset correctly between disconnected graphs."""

    graph_a = make_graph(
        jid="A",
        atomic_numbers=[14, 14],
        edges=[
            [0, 1],
            [1, 0],
        ],
        formation=-1.0,
        bandgap=1.5,
    )

    graph_b = make_graph(
        jid="B",
        atomic_numbers=[8, 22, 8],
        edges=[
            [0, 1],
            [1, 2],
        ],
        formation=-2.0,
        bandgap=2.5,
    )

    batch = collate_graphs(
        [graph_a, graph_b]
    )

    assert batch["num_graphs"] == 2

    assert batch["z"].tolist() == [
        14,
        14,
        8,
        22,
        8,
    ]

    assert batch["batch"].tolist() == [
        0,
        0,
        1,
        1,
        1,
    ]

    expected_edges = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 4],
        ],
        dtype=torch.long,
    )

    assert torch.equal(
        batch["edge_index"],
        expected_edges,
    )

    assert batch["edge_attr"].shape == (
        4,
        RBF_DIM,
    )

    assert batch["jids"] == [
        "A",
        "B",
    ]

    assert torch.allclose(
        batch["y_formation"],
        torch.tensor(
            [-1.0, -2.0]
        ),
    )

    assert torch.allclose(
        batch["y_bandgap"],
        torch.tensor(
            [1.5, 2.5]
        ),
    )


def test_collate_all_zero_edge_graphs():
    """A batch containing only edgeless graphs must remain valid."""

    graph_a = make_graph(
        jid="isolated-A",
        atomic_numbers=[14],
        edges=[],
        formation=-0.5,
        bandgap=0.0,
    )

    graph_b = make_graph(
        jid="isolated-B",
        atomic_numbers=[15],
        edges=[],
        formation=-0.2,
        bandgap=1.0,
    )

    batch = collate_graphs(
        [graph_a, graph_b]
    )

    assert batch["num_graphs"] == 2
    assert batch["edge_index"].shape == (2, 0)
    assert batch["edge_attr"].shape == (0, RBF_DIM)
    assert batch["batch"].tolist() == [0, 1]


def test_collate_mixed_edged_and_edgeless_graphs():
    """Edgeless graphs must not corrupt node offsets."""

    graph_a = make_graph(
        jid="edgeless",
        atomic_numbers=[17],
        edges=[],
        formation=-0.1,
        bandgap=0.0,
    )

    graph_b = make_graph(
        jid="connected",
        atomic_numbers=[8, 12],
        edges=[
            [0, 1],
            [1, 0],
        ],
        formation=-1.5,
        bandgap=3.0,
    )

    batch = collate_graphs(
        [graph_a, graph_b]
    )

    expected_edges = torch.tensor(
        [
            [1, 2],
            [2, 1],
        ],
        dtype=torch.long,
    )

    assert torch.equal(
        batch["edge_index"],
        expected_edges,
    )

    assert batch["batch"].tolist() == [
        0,
        1,
        1,
    ]


def test_collate_empty_list_raises():
    """An empty list is not a meaningful graph batch."""

    try:
        collate_graphs([])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for an empty graph list."
        )