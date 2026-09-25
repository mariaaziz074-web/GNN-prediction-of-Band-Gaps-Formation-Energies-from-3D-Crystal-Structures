import json
from pathlib import Path

import pandas as pd


STRUCTURES_PATH = Path(
    "data/processed/jarvis_dft_8000_structures.json"
)
METADATA_PATH = Path(
    "data/processed/jarvis_dft_8000_metadata.csv"
)


def main():
    with STRUCTURES_PATH.open("r", encoding="utf-8") as f:
        structures = json.load(f)

    metadata = pd.read_csv(METADATA_PATH)

    print(f"[INFO] Structures: {len(structures):,}")
    print(f"[INFO] Metadata:   {len(metadata):,}")
    print(
        "[INFO] Metadata columns:",
        list(metadata.columns),
    )

    if len(structures) != len(metadata):
        raise ValueError("Dataset lengths do not match.")

    first = structures[0]
    print(
        "[INFO] First structure keys:",
        list(first.keys()) if isinstance(first, dict) else type(first),
    )

    if isinstance(first, dict) and "jid" in first:
        structure_jids = [str(x["jid"]) for x in structures]
        metadata_jids = metadata["jid"].astype(str).tolist()

        if structure_jids != metadata_jids:
            raise ValueError(
                "JID order mismatch between structures and metadata."
            )

        print("[OK] All 8,000 JIDs are aligned in identical order.")
    else:
        print(
            "[WARN] Structure JSON does not contain JIDs; "
            "alignment cannot be independently verified here."
        )


if __name__ == "__main__":
    main()