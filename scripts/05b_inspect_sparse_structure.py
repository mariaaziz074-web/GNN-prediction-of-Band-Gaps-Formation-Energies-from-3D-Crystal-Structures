import json
from pathlib import Path

import numpy as np
from jarvis.core.atoms import Atoms


STRUCTURES_PATH = Path(
    "data/processed/jarvis_dft_8000_structures.json"
)

TARGET_JID = "JVASP-7577"


def main():
    with STRUCTURES_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        records = json.load(f)

    record = next(
        (
            item
            for item in records
            if str(item["jid"]) == TARGET_JID
        ),
        None,
    )

    if record is None:
        raise ValueError(
            f"{TARGET_JID} not found."
        )

    atoms = Atoms.from_dict(
        record["atoms"]
    )

    lattice = np.asarray(
        atoms.lattice_mat,
        dtype=float,
    )

    frac = np.asarray(
        atoms.frac_coords,
        dtype=float,
    )

    print(
        f"[INFO] JID: {TARGET_JID}"
    )
    print(
        f"[INFO] Number of atoms: "
        f"{len(atoms.atomic_numbers)}"
    )

    print("[INFO] Lattice matrix:")
    for row in lattice:
        print(
            "       "
            + " ".join(
                f"{value:12.6f}"
                for value in row
            )
        )

    lengths = np.linalg.norm(
        lattice,
        axis=1,
    )

    volume = abs(
        np.linalg.det(lattice)
    )

    print(
        "[INFO] Lattice-vector lengths "
        f"(Å): {lengths}"
    )

    print(
        f"[INFO] Cell volume (Å^3): "
        f"{volume:.6f}"
    )

    print(
        "[INFO] Fractional coordinates:"
    )
    print(frac)

    # For a one-atom primitive cell, the nearest
    # periodic image is a nonzero integer combination
    # of lattice vectors. Search a generous image range.
    best_distance = float("inf")
    best_shift = None
    best_vector = None

    for a in range(-4, 5):
        for b in range(-4, 5):
            for c in range(-4, 5):
                if a == 0 and b == 0 and c == 0:
                    continue

                shift = np.array(
                    [a, b, c],
                    dtype=float,
                )

                vector = shift @ lattice

                distance = float(
                    np.linalg.norm(vector)
                )

                if distance < best_distance:
                    best_distance = distance
                    best_shift = (
                        a,
                        b,
                        c,
                    )
                    best_vector = vector

    print(
        "[RESULT] Nearest periodic-image "
        f"distance: {best_distance:.6f} Å"
    )

    print(
        "[RESULT] Image shift: "
        f"{best_shift}"
    )

    print(
        "[RESULT] Cartesian vector (Å): "
        f"{best_vector}"
    )

    if best_distance > 10.0:
        print(
            "[RESULT] This explains why the "
            "10 Å fallback produced no edges."
        )


if __name__ == "__main__":
    main()