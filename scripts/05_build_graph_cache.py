import json
from pathlib import Path

import pandas as pd
import torch

from gnn_materials.data.graph_builder import build_crystal_graph


STRUCTURES_PATH = Path(
    "data/processed/jarvis_dft_8000_structures.json"
)
METADATA_PATH = Path(
    "data/processed/jarvis_dft_8000_metadata.csv"
)
OUTPUT_PATH = Path(
    "data/processed/jarvis_dft_8000_graphs.pt"
)


def main():
    print("[INFO] Loading frozen structures and metadata...")

    with STRUCTURES_PATH.open("r", encoding="utf-8") as f:
        structures = json.load(f)

    metadata = pd.read_csv(METADATA_PATH)

    if len(structures) != len(metadata):
        raise ValueError(
            f"Structure/metadata mismatch: "
            f"{len(structures)} vs {len(metadata)}"
        )

    graphs = []

    print(
        f"[INFO] Building periodic graphs for "
        f"{len(structures):,} structures..."
    )

    for i, item in enumerate(structures):
        atom_dict = item.get("atoms", item)

        try:
            graph = build_crystal_graph(
                atom_dict,
                cutoff=5.0,
                max_neighbors=12,
                n_rbf=32,
            )
        except ValueError as exc:
            row = metadata.iloc[i]

            print()
            print(
                f"[ERROR] Graph construction failed "
                f"at index {i}"
            )
            print(f"[ERROR] JID: {row['jid']}")
            print(f"[ERROR] Formula: {row['formula']}")
            print(
                f"[ERROR] Number of atoms: "
                f"{row['nat']}"
            )
            print(f"[ERROR] Reason: {exc}")

            raise

        row = metadata.iloc[i]

        graph["jid"] = str(row["jid"])
        graph["split"] = str(row["split"])

        graph["y_formation"] = torch.tensor(
            float(row["formation_energy_peratom"]),
            dtype=torch.float32,
        )

        graph["y_bandgap"] = torch.tensor(
            float(row["optb88vdw_bandgap"]),
            dtype=torch.float32,
        )

        graphs.append(graph)

        if (i + 1) % 500 == 0:
            print(
                f"[INFO] Built {i + 1:,}/"
                f"{len(structures):,} graphs"
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(graphs, OUTPUT_PATH)

    train_count = sum(
        g["split"] == "train" for g in graphs
    )
    val_count = sum(
        g["split"] == "val" for g in graphs
    )
    test_count = sum(
        g["split"] == "test" for g in graphs
    )

    print()
    print("[OK] Graph cache created.")
    print(f"[INFO] Total: {len(graphs):,}")
    print(f"[INFO] Train: {train_count:,}")
    print(f"[INFO] Val:   {val_count:,}")
    print(f"[INFO] Test:  {test_count:,}")
    print(f"[OK] Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()