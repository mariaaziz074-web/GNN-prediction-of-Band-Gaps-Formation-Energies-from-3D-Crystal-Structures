import itertools

import numpy as np
import torch
from jarvis.core.atoms import Atoms


def gaussian_expand(distances, cutoff=5.0, n_rbf=32):
    """Expand scalar distances using Gaussian radial basis functions."""
    centers = torch.linspace(
        0.0,
        cutoff,
        n_rbf,
        dtype=distances.dtype,
        device=distances.device,
    )

    if n_rbf > 1:
        spacing = centers[1] - centers[0]
    else:
        spacing = torch.tensor(
            cutoff,
            dtype=distances.dtype,
            device=distances.device,
        )

    return torch.exp(
        -((distances.unsqueeze(-1) - centers) / spacing) ** 2
    )


def build_crystal_graph(
    atom_dict,
    cutoff=5.0,
    max_neighbors=12,
    n_rbf=32,
):
    """
    Convert a periodic JARVIS crystal structure to a graph.

    Multiple periodic images of an atom are retained when they lie
    within the cutoff. This is required for small primitive cells.
    """
    atoms = Atoms.from_dict(atom_dict)

    lattice = np.asarray(atoms.lattice_mat, dtype=np.float64)
    frac = np.asarray(atoms.frac_coords, dtype=np.float64)
    atomic_numbers = np.asarray(
        atoms.atomic_numbers,
        dtype=np.int64,
    )

    n_atoms = len(atomic_numbers)

    # Conservative number of image cells required along each
    # fractional direction. Reciprocal-vector norms account for
    # non-orthogonal cells.
    reciprocal = np.linalg.inv(lattice).T
    image_limits = np.ceil(
        cutoff * np.linalg.norm(reciprocal, axis=0)
    ).astype(int) + 1

    translations = list(
        itertools.product(
            range(-image_limits[0], image_limits[0] + 1),
            range(-image_limits[1], image_limits[1] + 1),
            range(-image_limits[2], image_limits[2] + 1),
        )
    )

    sources = []
    targets = []
    edge_distances = []

    tol = 1e-8

    for i in range(n_atoms):
        neighbors = []

        for j in range(n_atoms):
            for shift in translations:
                shift = np.asarray(shift, dtype=np.float64)

                # Exclude only the atom in its own central image.
                if i == j and np.all(shift == 0):
                    continue

                delta_frac = frac[j] + shift - frac[i]
                delta_cart = delta_frac @ lattice
                distance = float(np.linalg.norm(delta_cart))

                if tol < distance <= cutoff + tol:
                    neighbors.append((distance, j))

        neighbors.sort(key=lambda item: item[0])

        if max_neighbors is not None:
            neighbors = neighbors[:max_neighbors]

        for distance, j in neighbors:
            sources.append(i)
            targets.append(j)
            edge_distances.append(distance)

    if not edge_distances:
        raise ValueError(
            "No graph edges were found. Increase cutoff or inspect structure."
        )

    z = torch.tensor(atomic_numbers, dtype=torch.long)

    edge_index = torch.tensor(
        [sources, targets],
        dtype=torch.long,
    )

    distances = torch.tensor(
        edge_distances,
        dtype=torch.float32,
    )

    edge_attr = gaussian_expand(
        distances,
        cutoff=cutoff,
        n_rbf=n_rbf,
    )

    return {
        "z": z,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "distances": distances,
        "num_nodes": n_atoms,
    }