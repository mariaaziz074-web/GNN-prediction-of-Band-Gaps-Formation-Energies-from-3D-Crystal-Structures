import json
from pathlib import Path

import torch

from gnn_materials.data.graph_builder import build_crystal_graph


STRUCTURES_PATH = Path(
    "data/processed/jarvis_dft_8000_structures.json"
)


def main():
    print("[INFO] Loading frozen structures...")

    with STRUCTURES_PATH.open("r", encoding="utf-8") as f:
        structures = json.load(f)

    print(f"[INFO] Structures available: {len(structures):,}")

    n_test = min(100, len(structures))

    node_counts = []
    edge_counts = []
    max_distances = []

    for i in range(n_test):
        item = structures[i]

        # Support either direct atom dictionaries or records
        # containing an "atoms" field.
        atom_dict = item.get("atoms", item)

        graph = build_crystal_graph(atom_dict)

        assert graph["z"].dtype == torch.long
        assert graph["edge_index"].shape[0] == 2
        assert graph["edge_attr"].shape[1] == 32
        assert graph["edge_index"].shape[1] == graph["distances"].shape[0]
        assert graph["edge_attr"].shape[0] == graph["distances"].shape[0]
        assert torch.all(graph["distances"] > 0)
        assert torch.all(graph["distances"] <= 5.0 + 1e-6)

        node_counts.append(graph["num_nodes"])
        edge_counts.append(graph["edge_index"].shape[1])
        max_distances.append(float(graph["distances"].max()))

    print(f"[OK] Validated {n_test} periodic crystal graphs.")
    print(
        f"[INFO] Nodes: min={min(node_counts)}, "
        f"max={max(node_counts)}"
    )
    print(
        f"[INFO] Edges: min={min(edge_counts)}, "
        f"max={max(edge_counts)}"
    )
    print(
        f"[INFO] Largest retained distance: "
        f"{max(max_distances):.4f} Å"
    )
    print("[OK] Periodic graph builder validation passed.")


if __name__ == "__main__":
    main()