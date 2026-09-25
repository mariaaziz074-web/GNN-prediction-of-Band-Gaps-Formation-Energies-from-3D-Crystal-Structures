from pathlib import Path

import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path("data/processed")
RESULTS = Path("results/tables")
RESULTS.mkdir(parents=True, exist_ok=True)

META = DATA / "jarvis_dft_8000_metadata.csv"
STRUCTURES = DATA / "jarvis_dft_8000_structures.json"

TARGETS = [
    "formation_energy_peratom",
    "optb88vdw_bandgap",
]


def structure_features(atom_dict):
    atoms = Atoms.from_dict(atom_dict)

    z = np.asarray(atoms.atomic_numbers, dtype=float)
    coords = np.asarray(atoms.cart_coords, dtype=float)
    lattice = np.asarray(atoms.lattice_mat, dtype=float)

    nat = len(z)
    volume = abs(np.linalg.det(lattice))

    # Lightweight structural/compositional baseline descriptors.
    return [
        nat,
        volume,
        volume / max(nat, 1),
        np.mean(z),
        np.std(z),
        np.min(z),
        np.max(z),
        len(np.unique(z)),
        np.mean(np.linalg.norm(coords - coords.mean(axis=0), axis=1)),
    ]


def metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def main():
    print("[INFO] Loading frozen Project 4 dataset...")

    df = pd.read_csv(META)
    structures = pd.read_json(STRUCTURES)

    structure_map = {
        row["sample_index"]: row["atoms"]
        for _, row in structures.iterrows()
    }

    feature_rows = []

    for i, row in df.iterrows():
        if (i + 1) % 1000 == 0:
            print(f"[INFO] Featurized {i + 1:,}/{len(df):,}")

        feats = structure_features(
            structure_map[row["sample_index"]]
        )
        feature_rows.append(feats)

    feature_names = [
        "num_atoms",
        "cell_volume",
        "volume_per_atom",
        "mean_atomic_number",
        "std_atomic_number",
        "min_atomic_number",
        "max_atomic_number",
        "n_element_types",
        "mean_radial_spread",
    ]

    X = pd.DataFrame(feature_rows, columns=feature_names)

    train_mask = df["split"].eq("train").to_numpy()
    val_mask = df["split"].eq("val").to_numpy()
    test_mask = df["split"].eq("test").to_numpy()

    results = []
    predictions = []

    for target in TARGETS:
        print(f"\n[INFO] Training baseline: {target}")

        model = ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        )

        model.fit(
            X.loc[train_mask],
            df.loc[train_mask, target],
        )

        for split_name, mask in [
            ("val", val_mask),
            ("test", test_mask),
        ]:
            y_true = df.loc[mask, target].to_numpy()
            y_pred = model.predict(X.loc[mask])

            m = metrics(y_true, y_pred)

            results.append({
                "model": "ExtraTrees_structural_baseline",
                "target": target,
                "split": split_name,
                **m,
            })

            indices = np.where(mask)[0]

            for idx, true, pred in zip(indices, y_true, y_pred):
                predictions.append({
                    "sample_index": int(df.iloc[idx]["sample_index"]),
                    "jid": df.iloc[idx]["jid"],
                    "target": target,
                    "split": split_name,
                    "y_true": float(true),
                    "y_pred": float(pred),
                })

            print(
                f"[{split_name.upper()}] "
                f"MAE={m['mae']:.4f} | "
                f"RMSE={m['rmse']:.4f} | "
                f"R2={m['r2']:.4f}"
            )

    results_df = pd.DataFrame(results)
    predictions_df = pd.DataFrame(predictions)

    results_path = RESULTS / "classical_baseline_metrics.csv"
    predictions_path = RESULTS / "classical_baseline_predictions.csv"

    results_df.to_csv(results_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print(f"\n[OK] Metrics: {results_path}")
    print(f"[OK] Predictions: {predictions_path}")


if __name__ == "__main__":
    main()