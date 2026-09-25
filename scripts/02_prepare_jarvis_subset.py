import json
from pathlib import Path

import numpy as np
import pandas as pd
from jarvis.db.figshare import data


SEED = 42
N_SAMPLES = 8000

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("[INFO] Loading JARVIS-DFT dft_3d...")
    records = data("dft_3d")

    print(f"[INFO] Full dataset: {len(records):,}")

    # All 93,902 were verified to contain both primary targets.
    indices = np.arange(len(records))

    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)

    selected = indices[:N_SAMPLES]

    # Freeze split BEFORE model development.
    n_train = int(0.80 * N_SAMPLES)
    n_val = int(0.10 * N_SAMPLES)

    split = np.full(N_SAMPLES, "test", dtype=object)
    split[:n_train] = "train"
    split[n_train:n_train + n_val] = "val"

    rows = []
    structure_records = []

    for local_idx, source_idx in enumerate(selected):
        rec = records[int(source_idx)]

        rows.append(
            {
                "sample_index": local_idx,
                "source_index": int(source_idx),
                "jid": rec["jid"],
                "formula": rec.get("formula", ""),
                "nat": rec.get("nat", ""),
                "space_group_number": rec.get(
                    "spg_number", rec.get("spg", "")
                ),
                "crystal_system": rec.get("crys", ""),
                "formation_energy_peratom": float(
                    rec["formation_energy_peratom"]
                ),
                "optb88vdw_bandgap": float(
                    rec["optb88vdw_bandgap"]
                ),
                "split": split[local_idx],
            }
        )

        structure_records.append(
            {
                "sample_index": local_idx,
                "source_index": int(source_idx),
                "jid": rec["jid"],
                "atoms": rec["atoms"],
            }
        )

    df = pd.DataFrame(rows)

    csv_path = PROCESSED_DIR / "jarvis_dft_8000_metadata.csv"
    df.to_csv(csv_path, index=False)

    structures_path = PROCESSED_DIR / "jarvis_dft_8000_structures.json"
    with structures_path.open("w", encoding="utf-8") as f:
        json.dump(structure_records, f)

    manifest = {
        "dataset": "JARVIS-DFT dft_3d",
        "source_population": len(records),
        "subset_size": N_SAMPLES,
        "seed": SEED,
        "split": {
            "train": int((df["split"] == "train").sum()),
            "validation": int((df["split"] == "val").sum()),
            "test": int((df["split"] == "test").sum()),
        },
        "targets": {
            "formation_energy_peratom": "eV/atom",
            "optb88vdw_bandgap": "eV",
        },
        "identifier": "jid",
        "structure_field": "atoms",
        "selection": "deterministic random subset using numpy default_rng(seed=42)",
    }

    manifest_path = PROCESSED_DIR / "jarvis_dft_8000_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n[OK] Dataset frozen.")
    print(df["split"].value_counts())
    print()
    print(df[
        [
            "formation_energy_peratom",
            "optb88vdw_bandgap"
        ]
    ].describe())

    print(f"\n[OK] Metadata:   {csv_path}")
    print(f"[OK] Structures: {structures_path}")
    print(f"[OK] Manifest:   {manifest_path}")


if __name__ == "__main__":
    main()