"""
Directed Message Passing Neural Network (D-MPNN) Implementation

This module implements a directed message passing neural network inspired by
Chemprop (Yang et al., 2019). Key features:

1. Bond-level message passing: Messages pass along bonds (not atoms)
2. Directed edges: Each bond becomes two directed edges
3. Message aggregation: Atom representations built from incoming bond messages
4. Multi-task prediction: Shared encoder, task-specific heads

Architecture rationale:
- Bond-level messaging captures richer structural information than atom-level
- Direction matters: bond A->B differs from B->A in molecular context
- Allows encoding of stereochemistry and bond directionality

Reference:
Yang et al. "Analyzing Learned Molecular Representations for Property Prediction"
J. Chem. Inf. Model. 2019, 59, 8, 3370–3388

Author: Claude Code
Date: 2025-10-20
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


class DirectedMessagePassing(nn.Module):
    """
    Directed Message Passing Layer.

    Implements bond-to-bond message passing where each undirected bond
    becomes two directed edges. Messages flow along directed edges and
    are aggregated at atoms.

    Message passing steps:
    1. Initialize bond hidden states from bond features
    2. For T steps:
        a. Aggregate messages from neighboring bonds
        b. Update bond hidden states
    3. Aggregate bond messages to atom representations
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int,
        num_message_passing_steps: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize directed message passing layer.

        Args:
            node_features: Dimension of node (atom) feature vectors
            edge_features: Dimension of edge (bond) feature vectors
            hidden_dim: Hidden dimension for message passing
            num_message_passing_steps: Number of message passing iterations
            dropout: Dropout probability
        """
        super().__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_steps = num_message_passing_steps

        # Initial bond embedding: edge_features -> hidden_dim
        self.edge_init = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Message function: transforms incoming messages
        # Input: concatenated [bond_hidden, neighbor_bond_hidden]
        self.message_nn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Atom readout: aggregate bond messages to atom representation
        # Input: concatenated [atom_features, aggregated_bond_messages]
        self.atom_readout = nn.Sequential(
            nn.Linear(node_features + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of directed message passing.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            batch: Batch assignment for nodes [num_nodes]

        Returns:
            Atom representations [num_nodes, hidden_dim]
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # Step 1: Initialize bond hidden states
        bond_hidden = self.edge_init(edge_attr)  # [num_edges, hidden_dim]

        # Step 2: Message passing for T steps
        for step in range(self.num_steps):
            # For each bond, aggregate messages from neighboring bonds
            messages = []

            # Build adjacency: for each directed edge, find incoming edges to target
            source_nodes = edge_index[0]  # Where edge starts
            target_nodes = edge_index[1]  # Where edge ends

            # For each edge, find all other edges targeting the same source node
            # These represent neighboring bonds in the molecular graph
            new_bond_hidden = torch.zeros_like(bond_hidden)

            for edge_idx in range(num_edges):
                src = source_nodes[edge_idx]
                tgt = target_nodes[edge_idx]

                # Find all edges that end at the source of current edge
                # Exclude the reverse edge (to avoid message from same bond)
                neighbor_mask = (target_nodes == src)
                neighbor_edges = torch.where(neighbor_mask)[0]

                # Remove reverse edge: edge that goes from tgt to src
                reverse_mask = (source_nodes[neighbor_edges] != tgt)
                neighbor_edges = neighbor_edges[reverse_mask]

                if len(neighbor_edges) > 0:
                    # Aggregate neighbor bond hidden states
                    neighbor_hidden = bond_hidden[neighbor_edges]  # [num_neighbors, hidden_dim]
                    aggregated = neighbor_hidden.sum(dim=0)  # [hidden_dim]
                else:
                    # No neighbors - use zeros
                    aggregated = torch.zeros(self.hidden_dim, device=x.device)

                # Concatenate current bond hidden with aggregated neighbors
                message_input = torch.cat([
                    bond_hidden[edge_idx],
                    aggregated
                ], dim=0)  # [hidden_dim * 2]

                # Transform through message neural network
                new_bond_hidden[edge_idx] = self.message_nn(message_input)

            # Update bond hidden states
            bond_hidden = new_bond_hidden + bond_hidden  # Residual connection
            bond_hidden = self.dropout(bond_hidden)

        # Step 3: Aggregate bond messages to atom representations
        atom_messages = torch.zeros(num_nodes, self.hidden_dim, device=x.device)

        # For each atom, aggregate messages from incoming bonds
        for node_idx in range(num_nodes):
            # Find all edges ending at this node
            incoming_mask = (target_nodes == node_idx)
            incoming_edges = torch.where(incoming_mask)[0]

            if len(incoming_edges) > 0:
                incoming_hidden = bond_hidden[incoming_edges]
                atom_messages[node_idx] = incoming_hidden.sum(dim=0)

        # Combine atom features with aggregated bond messages
        atom_repr = torch.cat([x, atom_messages], dim=1)
        atom_repr = self.atom_readout(atom_repr)

        return atom_repr


class DMPNNEncoder(nn.Module):
    """
    D-MPNN Encoder: Converts molecular graph to fixed-size representation.

    Architecture:
    1. Directed message passing on bonds
    2. Atom-level representations
    3. Graph-level pooling (sum/mean/max)
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 256,
        num_message_passing_steps: int = 3,
        num_ffn_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = 'sum'
    ):
        """
        Initialize D-MPNN encoder.

        Args:
            node_features: Dimension of node features
            edge_features: Dimension of edge features
            hidden_dim: Hidden dimension
            num_message_passing_steps: Number of message passing iterations
            num_ffn_layers: Number of feed-forward layers after pooling
            dropout: Dropout probability
            pooling: Graph pooling method ('sum', 'mean', or 'max')
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pooling = pooling

        # Message passing
        self.message_passing = DirectedMessagePassing(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_message_passing_steps=num_message_passing_steps,
            dropout=dropout
        )

        # Feed-forward network after pooling
        ffn_layers = []
        for i in range(num_ffn_layers):
            ffn_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.ffn = nn.Sequential(*ffn_layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode molecular graph to fixed-size vector.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            batch: Batch assignment [num_nodes]

        Returns:
            Graph representations [batch_size, hidden_dim]
        """
        # Message passing to get atom representations
        atom_repr = self.message_passing(x, edge_index, edge_attr, batch)

        # Pool to graph-level representation
        if self.pooling == 'sum':
            graph_repr = global_add_pool(atom_repr, batch)
        elif self.pooling == 'mean':
            graph_repr = global_mean_pool(atom_repr, batch)
        elif self.pooling == 'max':
            graph_repr = global_max_pool(atom_repr, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Feed-forward network
        graph_repr = self.ffn(graph_repr)

        return graph_repr


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head.

    Each task has its own prediction head (feed-forward network)
    that takes the shared graph representation as input.
    """

    def __init__(
        self,
        input_dim: int,
        task_names: List[str],
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize multi-task head.

        Args:
            input_dim: Dimension of input representations
            task_names: List of task names
            hidden_dim: Hidden dimension for task-specific heads
            dropout: Dropout probability
        """
        super().__init__()

        self.task_names = task_names
        self.num_tasks = len(task_names)

        # Task-specific prediction heads
        self.task_heads = nn.ModuleDict()
        for task in task_names:
            self.task_heads[task] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)  # Single output per task
            )

    def forward(self, graph_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict all tasks.

        Args:
            graph_repr: Graph representations [batch_size, input_dim]

        Returns:
            Dictionary mapping task names to predictions [batch_size, 1]
        """
        predictions = {}
        for task in self.task_names:
            predictions[task] = self.task_heads[task](graph_repr)

        return predictions


class DMPNNModel(nn.Module):
    """
    Complete D-MPNN model with multi-task prediction.

    Architecture:
    1. Shared encoder: D-MPNN for graph encoding
    2. Task-specific heads: Separate prediction heads per task

    This design allows:
    - Transfer learning: Shared representations across tasks
    - Task-specific adaptation: Each task can learn unique patterns
    - Efficient training: Single forward pass for all tasks
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        task_names: List[str],
        hidden_dim: int = 256,
        num_message_passing_steps: int = 3,
        num_ffn_layers: int = 2,
        task_head_hidden_dim: int = 128,
        dropout: float = 0.1,
        pooling: str = 'sum'
    ):
        """
        Initialize D-MPNN model.

        Args:
            node_features: Dimension of node features
            edge_features: Dimension of edge features
            task_names: List of prediction task names
            hidden_dim: Hidden dimension for encoder
            num_message_passing_steps: Message passing iterations
            num_ffn_layers: Feed-forward layers in encoder
            task_head_hidden_dim: Hidden dimension for task heads
            dropout: Dropout probability
            pooling: Graph pooling method
        """
        super().__init__()

        self.task_names = task_names

        # Shared encoder
        self.encoder = DMPNNEncoder(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_message_passing_steps=num_message_passing_steps,
            num_ffn_layers=num_ffn_layers,
            dropout=dropout,
            pooling=pooling
        )

        # Multi-task prediction heads
        self.heads = MultiTaskHead(
            input_dim=hidden_dim,
            task_names=task_names,
            hidden_dim=task_head_hidden_dim,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: graph to multi-task predictions.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            batch: Batch assignment [num_nodes]

        Returns:
            Dictionary of predictions per task
        """
        # Encode graph
        graph_repr = self.encoder(x, edge_index, edge_attr, batch)

        # Predict all tasks
        predictions = self.heads(graph_repr)

        return predictions

    def get_graph_representation(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Get graph-level representation (for analysis or transfer learning).

        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features
            batch: Batch assignment

        Returns:
            Graph representations [batch_size, hidden_dim]
        """
        return self.encoder(x, edge_index, edge_attr, batch)


def create_dmpnn_model(
    node_features: int,
    edge_features: int,
    task_names: List[str],
    config: Optional[Dict] = None
) -> DMPNNModel:
    """
    Factory function to create D-MPNN model with default or custom config.

    Args:
        node_features: Dimension of node features
        edge_features: Dimension of edge features
        task_names: List of task names
        config: Optional configuration dictionary

    Returns:
        Initialized DMPNNModel
    """
    default_config = {
        'hidden_dim': 256,
        'num_message_passing_steps': 3,
        'num_ffn_layers': 2,
        'task_head_hidden_dim': 128,
        'dropout': 0.1,
        'pooling': 'sum'
    }

    if config is not None:
        default_config.update(config)

    return DMPNNModel(
        node_features=node_features,
        edge_features=edge_features,
        task_names=task_names,
        **default_config
    )


if __name__ == "__main__":
    # Test model creation
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("PyTorch Geometric not available. Cannot test D-MPNN model.")
    else:
        print("Testing D-MPNN model creation...")

        # Example dimensions from fingerprints.py graph representation
        node_features = 13 + 5 + 5  # atom_type_onehot + features + hybridization
        edge_features = 6  # bond_type_onehot + features

        task_names = ['solubility', 'permeability', 'hERG']

        model = create_dmpnn_model(
            node_features=node_features,
            edge_features=edge_features,
            task_names=task_names
        )

        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        print(f"Tasks: {task_names}")

        # Test forward pass with dummy data
        batch_size = 4
        num_nodes = 20
        num_edges = 40

        x = torch.randn(num_nodes, node_features)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, edge_features)
        batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes // batch_size)

        with torch.no_grad():
            predictions = model(x, edge_index, edge_attr, batch)

        print("\nTest forward pass successful!")
        for task, pred in predictions.items():
            print(f"  {task}: {pred.shape}")
