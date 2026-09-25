import torch


def collate_graphs(graphs):
    """
    Combine a list of crystal graph dictionaries into one
    disconnected PyTorch graph batch.

    Node indices in edge_index are shifted so each crystal occupies
    a separate component. Zero-edge graphs are fully supported.
    """
    if not graphs:
        raise ValueError(
            "Cannot collate an empty graph list."
        )

    z_parts = []
    edge_index_parts = []
    edge_attr_parts = []
    batch_parts = []

    y_formation = []
    y_bandgap = []
    jids = []

    node_offset = 0

    for graph_id, graph in enumerate(graphs):
        z = graph["z"]
        edge_index = graph["edge_index"]
        edge_attr = graph["edge_attr"]

        num_nodes = int(z.shape[0])
        num_edges = int(edge_index.shape[1])

        if num_nodes < 1:
            raise ValueError(
                "Every graph must contain at least one node."
            )

        z_parts.append(z)

        batch_parts.append(
            torch.full(
                (num_nodes,),
                graph_id,
                dtype=torch.long,
            )
        )

        if num_edges > 0:
            edge_index_parts.append(
                edge_index + node_offset
            )

            edge_attr_parts.append(
                edge_attr
            )

        y_formation.append(
            graph["y_formation"].reshape(())
        )

        y_bandgap.append(
            graph["y_bandgap"].reshape(())
        )

        jids.append(
            str(graph["jid"])
        )

        node_offset += num_nodes

    z = torch.cat(
        z_parts,
        dim=0,
    )

    batch = torch.cat(
        batch_parts,
        dim=0,
    )

    if edge_index_parts:
        edge_index = torch.cat(
            edge_index_parts,
            dim=1,
        )

        edge_attr = torch.cat(
            edge_attr_parts,
            dim=0,
        )
    else:
        rbf_dim = int(
            graphs[0]["edge_attr"].shape[1]
        )

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

        edge_attr = torch.empty(
            (0, rbf_dim),
            dtype=torch.float32,
        )

    return {
        "z": z,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": batch,
        "num_graphs": len(graphs),
        "y_formation": torch.stack(
            y_formation
        ).float(),
        "y_bandgap": torch.stack(
            y_bandgap
        ).float(),
        "jids": jids,
    }