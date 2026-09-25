from pathlib import Path

import pandas as pd
import torch


GRAPH_PATH = Path(
    "data/processed/"
    "jarvis_dft_8000_graphs.pt"
)

METADATA_PATH = Path(
    "data/processed/"
    "jarvis_dft_8000_metadata.csv"
)

EDGELESS_PATH = Path(
    "data/processed/"
    "graph_edgeless_structures.csv"
)

EXPECTED_TOTAL = 8000
EXPECTED_TRAIN = 6400
EXPECTED_VAL = 800
EXPECTED_TEST = 800
EXPECTED_RBF = 32
CUTOFF = 5.0


def main():
    print(
        "[INFO] Loading graph cache..."
    )

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing graph cache: "
            f"{GRAPH_PATH}"
        )

    graphs = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"[INFO] Graphs loaded: "
        f"{len(graphs):,}"
    )

    if len(graphs) != EXPECTED_TOTAL:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL} "
            f"graphs, found {len(graphs)}."
        )

    if len(metadata) != EXPECTED_TOTAL:
        raise ValueError(
            "Metadata count mismatch."
        )

    split_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    total_nodes = 0
    total_edges = 0
    zero_edge_jids = []

    max_distance = 0.0
    max_atomic_number = 0

    for i, graph in enumerate(graphs):
        row = metadata.iloc[i]

        if (
            str(graph["jid"])
            != str(row["jid"])
        ):
            raise ValueError(
                f"JID mismatch at index {i}."
            )

        if (
            int(graph["sample_index"])
            != int(row["sample_index"])
        ):
            raise ValueError(
                "Sample-index mismatch "
                f"at index {i}."
            )

        split = str(
            graph["split"]
        )

        if split not in split_counts:
            raise ValueError(
                f"Invalid split '{split}' "
                f"at index {i}."
            )

        split_counts[split] += 1

        z = graph["z"]
        edge_index = graph[
            "edge_index"
        ]
        edge_attr = graph[
            "edge_attr"
        ]
        distances = graph[
            "distances"
        ]

        if z.dtype != torch.long:
            raise TypeError(
                f"Invalid z dtype "
                f"at index {i}."
            )

        if z.ndim != 1:
            raise ValueError(
                f"Invalid z shape "
                f"at index {i}: "
                f"{tuple(z.shape)}"
            )

        num_nodes = len(z)

        if (
            int(graph["num_nodes"])
            != num_nodes
        ):
            raise ValueError(
                "Node-count mismatch "
                f"at index {i}."
            )

        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
        ):
            raise ValueError(
                "Invalid edge_index shape "
                f"at index {i}: "
                f"{tuple(edge_index.shape)}"
            )

        num_edges = int(
            edge_index.shape[1]
        )

        if (
            int(graph["num_edges"])
            != num_edges
        ):
            raise ValueError(
                "Edge-count mismatch "
                f"at index {i}."
            )

        if (
            edge_attr.shape
            != (
                num_edges,
                EXPECTED_RBF,
            )
        ):
            raise ValueError(
                "Invalid edge_attr shape "
                f"at index {i}: "
                f"{tuple(edge_attr.shape)}"
            )

        if distances.shape != (
            num_edges,
        ):
            raise ValueError(
                "Distance shape mismatch "
                f"at index {i}."
            )

        if num_edges > 0:
            if int(
                edge_index.min()
            ) < 0:
                raise ValueError(
                    "Negative node index "
                    f"at graph {i}."
                )

            if int(
                edge_index.max()
            ) >= num_nodes:
                raise ValueError(
                    "Edge references invalid "
                    f"node at graph {i}."
                )

            if not torch.all(
                distances > 0
            ):
                raise ValueError(
                    "Non-positive distance "
                    f"at graph {i}."
                )

            if not torch.all(
                distances
                <= CUTOFF + 1e-6
            ):
                raise ValueError(
                    "Distance exceeds cutoff "
                    f"at graph {i}."
                )

            max_distance = max(
                max_distance,
                float(
                    distances.max()
                ),
            )

        else:
            zero_edge_jids.append(
                str(graph["jid"])
            )

        y_formation = float(
            graph[
                "y_formation"
            ]
        )

        y_bandgap = float(
            graph[
                "y_bandgap"
            ]
        )

        expected_formation = float(
            row[
                "formation_energy_peratom"
            ]
        )

        expected_bandgap = float(
            row[
                "optb88vdw_bandgap"
            ]
        )

        if abs(
            y_formation
            - expected_formation
        ) > 1e-5:
            raise ValueError(
                "Formation-energy label "
                f"mismatch at index {i}."
            )

        if abs(
            y_bandgap
            - expected_bandgap
        ) > 1e-5:
            raise ValueError(
                "Band-gap label mismatch "
                f"at index {i}."
            )

        total_nodes += num_nodes
        total_edges += num_edges

        max_atomic_number = max(
            max_atomic_number,
            int(z.max()),
        )

    expected_splits = {
        "train": EXPECTED_TRAIN,
        "val": EXPECTED_VAL,
        "test": EXPECTED_TEST,
    }

    if split_counts != expected_splits:
        raise ValueError(
            "Split-count mismatch: "
            f"{split_counts}"
        )

    if EDGELESS_PATH.exists():
        edgeless = pd.read_csv(
            EDGELESS_PATH
        )

        reported_jids = (
            edgeless["jid"]
            .astype(str)
            .tolist()
        )

        if reported_jids != zero_edge_jids:
            raise ValueError(
                "Zero-edge CSV does not "
                "match graph cache."
            )

    print(
        "[OK] All 8,000 graph identities "
        "and labels match metadata."
    )

    print(
        "[OK] Tensor dimensions and "
        "edge indices validated."
    )

    print(
        "[OK] Split counts: "
        f"{split_counts}"
    )

    print(
        f"[INFO] Total nodes: "
        f"{total_nodes:,}"
    )

    print(
        f"[INFO] Total directed edges: "
        f"{total_edges:,}"
    )

    print(
        f"[INFO] Maximum edge distance: "
        f"{max_distance:.4f} Å"
    )

    print(
        f"[INFO] Maximum atomic number: "
        f"{max_atomic_number}"
    )

    print(
        f"[INFO] Zero-edge graphs: "
        f"{len(zero_edge_jids)}"
    )

    for jid in zero_edge_jids:
        print(
            f"[INFO]   {jid}"
        )

    print(
        "[OK] Graph cache validation "
        "passed."
    )


if __name__ == "__main__":
    main()