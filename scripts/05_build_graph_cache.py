import json
from pathlib import Path

import pandas as pd
import torch

from gnn_materials.data.graph_builder import (
    build_crystal_graph,
)


STRUCTURES_PATH = Path(
    "data/processed/"
    "jarvis_dft_8000_structures.json"
)

METADATA_PATH = Path(
    "data/processed/"
    "jarvis_dft_8000_metadata.csv"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "jarvis_dft_8000_graphs.pt"
)

EDGELESS_PATH = Path(
    "data/processed/"
    "graph_edgeless_structures.csv"
)

CUTOFF = 5.0
MAX_NEIGHBORS = 12
N_RBF = 32


def main():
    print(
        "[INFO] Loading frozen structures "
        "and metadata..."
    )

    with STRUCTURES_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        structures = json.load(
            handle
        )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    if len(structures) != len(metadata):
        raise ValueError(
            "Structure/metadata length "
            "mismatch."
        )

    structure_jids = [
        str(item["jid"])
        for item in structures
    ]

    metadata_jids = (
        metadata["jid"]
        .astype(str)
        .tolist()
    )

    if structure_jids != metadata_jids:
        raise ValueError(
            "JID order mismatch between "
            "structures and metadata."
        )

    print(
        "[OK] Structure/metadata JID "
        "alignment verified."
    )

    print(
        f"[INFO] Building periodic graphs "
        f"for {len(structures):,} "
        "structures..."
    )

    graphs = []
    edgeless_records = []

    for i, item in enumerate(
        structures
    ):
        row = metadata.iloc[i]

        graph = build_crystal_graph(
            item["atoms"],
            cutoff=CUTOFF,
            max_neighbors=MAX_NEIGHBORS,
            n_rbf=N_RBF,
        )

        if graph["num_edges"] == 0:
            edgeless_records.append(
                {
                    "sample_index": int(
                        row["sample_index"]
                    ),
                    "jid": str(
                        row["jid"]
                    ),
                    "formula": str(
                        row["formula"]
                    ),
                    "nat": int(
                        row["nat"]
                    ),
                    "split": str(
                        row["split"]
                    ),
                }
            )

            print(
                f"[WARN] Zero-edge graph: "
                f"{row['jid']} "
                f"({row['formula']}, "
                f"nat={row['nat']})"
            )

        graph["jid"] = str(
            row["jid"]
        )

        graph["split"] = str(
            row["split"]
        )

        graph["sample_index"] = int(
            row["sample_index"]
        )

        graph[
            "neighbor_cutoff"
        ] = CUTOFF

        graph[
            "y_formation"
        ] = torch.tensor(
            float(
                row[
                    "formation_energy_peratom"
                ]
            ),
            dtype=torch.float32,
        )

        graph[
            "y_bandgap"
        ] = torch.tensor(
            float(
                row[
                    "optb88vdw_bandgap"
                ]
            ),
            dtype=torch.float32,
        )

        graphs.append(
            graph
        )

        if (i + 1) % 500 == 0:
            print(
                f"[INFO] Built "
                f"{i + 1:,}/"
                f"{len(structures):,} "
                "graphs"
            )

    train_count = sum(
        graph["split"] == "train"
        for graph in graphs
    )

    val_count = sum(
        graph["split"] == "val"
        for graph in graphs
    )

    test_count = sum(
        graph["split"] == "test"
        for graph in graphs
    )

    if (
        train_count,
        val_count,
        test_count,
    ) != (
        6400,
        800,
        800,
    ):
        raise ValueError(
            "Frozen split counts changed: "
            f"{train_count}/"
            f"{val_count}/"
            f"{test_count}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        graphs,
        OUTPUT_PATH,
    )

    edgeless_columns = [
        "sample_index",
        "jid",
        "formula",
        "nat",
        "split",
    ]

    pd.DataFrame(
        edgeless_records,
        columns=edgeless_columns,
    ).to_csv(
        EDGELESS_PATH,
        index=False,
    )

    total_nodes = sum(
        graph["num_nodes"]
        for graph in graphs
    )

    total_edges = sum(
        graph["num_edges"]
        for graph in graphs
    )

    print()
    print(
        "[OK] Graph cache created."
    )
    print(
        f"[INFO] Total: "
        f"{len(graphs):,}"
    )
    print(
        f"[INFO] Train: "
        f"{train_count:,}"
    )
    print(
        f"[INFO] Val:   "
        f"{val_count:,}"
    )
    print(
        f"[INFO] Test:  "
        f"{test_count:,}"
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
        "[INFO] Zero-edge structures: "
        f"{len(edgeless_records):,}"
    )
    print(
        f"[OK] Graph cache: "
        f"{OUTPUT_PATH}"
    )
    print(
        f"[OK] Edge-case report: "
        f"{EDGELESS_PATH}"
    )


if __name__ == "__main__":
    main()