import itertools

import numpy as np
import torch
from jarvis.core.atoms import Atoms


def gaussian_expand(
    distances,
    cutoff=5.0,
    n_rbf=32,
):
    """
    Expand distances using Gaussian radial basis functions.
    """
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
        -(
            (
                distances.unsqueeze(-1)
                - centers
            )
            / spacing
        )
        ** 2
    )


def build_crystal_graph(
    atom_dict,
    cutoff=5.0,
    max_neighbors=12,
    n_rbf=32,
):
    """
    Convert a periodic JARVIS crystal into a graph.

    Periodic images are explicitly enumerated. Multiple images of
    the same atom may therefore appear as physically distinct
    neighbors.

    Structures with no neighbors inside the cutoff are represented
    by valid zero-edge graphs rather than by artificial long-range
    bonds.
    """
    atoms = Atoms.from_dict(atom_dict)

    lattice = np.asarray(
        atoms.lattice_mat,
        dtype=np.float64,
    )

    frac = np.asarray(
        atoms.frac_coords,
        dtype=np.float64,
    )

    atomic_numbers = np.asarray(
        atoms.atomic_numbers,
        dtype=np.int64,
    )

    n_atoms = len(atomic_numbers)

    if n_atoms == 0:
        raise ValueError(
            "Crystal structure contains no atoms."
        )

    if cutoff <= 0:
        raise ValueError(
            "cutoff must be positive."
        )

    reciprocal = np.linalg.inv(
        lattice
    ).T

    image_limits = np.ceil(
        cutoff
        * np.linalg.norm(
            reciprocal,
            axis=0,
        )
    ).astype(int) + 1

    translations = list(
        itertools.product(
            range(
                -image_limits[0],
                image_limits[0] + 1,
            ),
            range(
                -image_limits[1],
                image_limits[1] + 1,
            ),
            range(
                -image_limits[2],
                image_limits[2] + 1,
            ),
        )
    )

    sources = []
    targets = []
    edge_distances = []

    tolerance = 1e-8

    for i in range(n_atoms):
        neighbors = []

        for j in range(n_atoms):
            for shift_tuple in translations:
                shift = np.asarray(
                    shift_tuple,
                    dtype=np.float64,
                )

                # Remove only the exact central-cell self edge.
                # Periodic copies of the atom remain valid.
                if (
                    i == j
                    and np.all(shift == 0)
                ):
                    continue

                delta_frac = (
                    frac[j]
                    + shift
                    - frac[i]
                )

                delta_cart = (
                    delta_frac
                    @ lattice
                )

                distance = float(
                    np.linalg.norm(
                        delta_cart
                    )
                )

                if (
                    tolerance
                    < distance
                    <= cutoff + tolerance
                ):
                    neighbors.append(
                        (
                            distance,
                            j,
                            shift_tuple,
                        )
                    )

        # Including the translation in the sort key gives
        # deterministic ordering when several images have exactly
        # the same distance.
        neighbors.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            )
        )

        if max_neighbors is not None:
            neighbors = neighbors[
                :max_neighbors
            ]

        for distance, j, _ in neighbors:
            sources.append(i)
            targets.append(j)
            edge_distances.append(
                distance
            )

    z = torch.tensor(
        atomic_numbers,
        dtype=torch.long,
    )

    if edge_distances:
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

    else:
        # Valid representation of an isolated atom or an
        # exceptionally sparse periodic structure.
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

        distances = torch.empty(
            (0,),
            dtype=torch.float32,
        )

        edge_attr = torch.empty(
            (0, n_rbf),
            dtype=torch.float32,
        )

    return {
        "z": z,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "distances": distances,
        "num_nodes": n_atoms,
        "num_edges": len(edge_distances),
    }