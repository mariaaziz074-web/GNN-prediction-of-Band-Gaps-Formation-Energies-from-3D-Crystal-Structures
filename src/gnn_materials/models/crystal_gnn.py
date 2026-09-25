import torch
from torch import nn


class MessagePassingBlock(nn.Module):
    """
    Edge-conditioned message-passing block.

    Messages depend on the neighboring node state and the
    radial-basis edge representation.
    """

    def __init__(
        self,
        hidden_dim=64,
        edge_dim=32,
    ):
        super().__init__()

        self.message_mlp = nn.Sequential(
            nn.Linear(
                hidden_dim + edge_dim,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
    ):
        """
        Parameters
        ----------
        x : Tensor
            Node features [N, hidden_dim].
        edge_index : LongTensor
            Directed edges [2, E].
        edge_attr : Tensor
            Edge features [E, edge_dim].
        """
        num_nodes = x.shape[0]

        if edge_index.shape[1] == 0:
            # Preserve isolated-node information.
            return self.norm(x)

        source = edge_index[0]
        target = edge_index[1]

        message_input = torch.cat(
            [
                x[source],
                edge_attr,
            ],
            dim=-1,
        )

        messages = self.message_mlp(
            message_input
        )

        aggregated = torch.zeros(
            (
                num_nodes,
                messages.shape[-1],
            ),
            dtype=messages.dtype,
            device=messages.device,
        )

        aggregated.index_add_(
            0,
            target,
            messages,
        )

        # Degree normalization prevents nodes with many neighbors
        # from automatically receiving much larger updates.
        degree = torch.zeros(
            num_nodes,
            dtype=x.dtype,
            device=x.device,
        )

        degree.index_add_(
            0,
            target,
            torch.ones(
                target.shape[0],
                dtype=x.dtype,
                device=x.device,
            ),
        )

        degree = degree.clamp(
            min=1.0
        ).unsqueeze(-1)

        aggregated = (
            aggregated / degree
        )

        update_input = torch.cat(
            [
                x,
                aggregated,
            ],
            dim=-1,
        )

        update = self.update_mlp(
            update_input
        )

        return self.norm(
            x + update
        )


class CrystalGNN(nn.Module):
    """
    Multi-task graph neural network for crystal properties.

    Outputs standardized predictions for:
      1. formation energy per atom
      2. OptB88-vdW band gap
    """

    def __init__(
        self,
        max_atomic_number=100,
        hidden_dim=64,
        edge_dim=32,
        num_layers=3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            max_atomic_number + 1,
            hidden_dim,
        )

        self.layers = nn.ModuleList(
            [
                MessagePassingBlock(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dim,
                )
                for _ in range(
                    num_layers
                )
            ]
        )

        self.readout = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.SiLU(),
        )

        self.formation_head = nn.Linear(
            hidden_dim // 2,
            1,
        )

        self.bandgap_head = nn.Linear(
            hidden_dim // 2,
            1,
        )

    def mean_pool(
        self,
        x,
        batch,
        num_graphs,
    ):
        """
        Mean-pool node representations by graph.
        """
        pooled = torch.zeros(
            (
                num_graphs,
                x.shape[-1],
            ),
            dtype=x.dtype,
            device=x.device,
        )

        pooled.index_add_(
            0,
            batch,
            x,
        )

        counts = torch.zeros(
            num_graphs,
            dtype=x.dtype,
            device=x.device,
        )

        counts.index_add_(
            0,
            batch,
            torch.ones(
                batch.shape[0],
                dtype=x.dtype,
                device=x.device,
            ),
        )

        counts = counts.clamp(
            min=1.0
        ).unsqueeze(-1)

        return pooled / counts

    def forward(
        self,
        z,
        edge_index,
        edge_attr,
        batch,
        num_graphs,
    ):
        x = self.embedding(z)

        for layer in self.layers:
            x = layer(
                x,
                edge_index,
                edge_attr,
            )

        pooled = self.mean_pool(
            x,
            batch,
            num_graphs,
        )

        representation = self.readout(
            pooled
        )

        formation = (
            self.formation_head(
                representation
            )
            .squeeze(-1)
        )

        bandgap = (
            self.bandgap_head(
                representation
            )
            .squeeze(-1)
        )

        return {
            "formation": formation,
            "bandgap": bandgap,
        }