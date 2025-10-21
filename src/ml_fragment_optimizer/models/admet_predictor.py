"""
Multi-Task ADMET Prediction Module

This module implements a comprehensive ADMET (Absorption, Distribution, Metabolism,
Excretion, Toxicity) prediction system using deep learning.

Predicted Properties:
1. Solubility (LogS): Aqueous solubility at pH 7.4
2. Permeability (Caco-2): Intestinal permeability
3. CYP3A4 Inhibition: Drug-drug interaction potential
4. hERG Liability: Cardiotoxicity risk
5. LogD: Lipophilicity at pH 7.4
6. pKa: Acid dissociation constant

Architecture Options:
1. Graph-based (D-MPNN): Uses molecular graph structure via message passing
2. Fingerprint-based: Uses traditional molecular fingerprints (ECFP + descriptors)
3. Hybrid: Combines both approaches

Multi-task Learning Benefits:
- Shared representations capture common molecular features
- Transfer learning between related properties
- More efficient than training separate models
- Better performance on data-scarce tasks

Author: Claude Code
Date: 2025-10-20
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from .fingerprints import MolecularFeaturizer, MoleculeFeatures
from .chemprop_wrapper import DMPNNModel, create_dmpnn_model
from .uncertainty import (
    EvidentialOutput, evidential_loss, nig_uncertainty,
    MCDropoutWrapper, compute_uncertainty_metrics
)

try:
    from torch_geometric.data import Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    Batch = Any  # Type placeholder when torch_geometric not available


# Standard ADMET task names and their typical ranges (for normalization)
ADMET_TASKS = {
    'solubility': (-10.0, 2.0),  # LogS (mol/L)
    'permeability': (-8.0, -4.0),  # Log Papp (cm/s)
    'cyp3a4': (0.0, 1.0),  # Binary: inhibitor or not
    'herg': (0.0, 1.0),  # Binary: liability or not
    'logd': (-2.0, 6.0),  # LogD at pH 7.4
    'pka': (0.0, 14.0),  # pKa (most acidic)
}


@dataclass
class ADMETConfig:
    """Configuration for ADMET predictor"""
    # Model architecture
    model_type: str = 'dmpnn'  # 'dmpnn', 'fingerprint', or 'hybrid'
    hidden_dim: int = 256
    num_message_passing_steps: int = 3
    num_ffn_layers: int = 2
    task_head_hidden_dim: int = 128
    dropout: float = 0.1
    pooling: str = 'sum'

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    patience: int = 10
    gradient_clip: float = 5.0

    # Uncertainty quantification
    use_evidential: bool = True
    evidential_coeff: float = 0.1
    use_mc_dropout: bool = False
    mc_samples: int = 20

    # Data
    train_split: float = 0.8
    val_split: float = 0.1
    # test_split: float = 0.1 (implied)

    # Task selection (None = use all tasks)
    active_tasks: Optional[List[str]] = None

    def __post_init__(self):
        """Validate configuration"""
        if self.model_type not in ['dmpnn', 'fingerprint', 'hybrid']:
            raise ValueError(f"Invalid model_type: {self.model_type}")

        if self.model_type == 'dmpnn' and not TORCH_GEOMETRIC_AVAILABLE:
            raise ValueError("DMPNN model requires torch_geometric")

        if self.train_split + self.val_split >= 1.0:
            raise ValueError("train_split + val_split must be < 1.0")


class ADMETDataset(Dataset):
    """
    Dataset for ADMET prediction.

    Stores molecular features and targets for multiple ADMET properties.
    """

    def __init__(
        self,
        smiles_list: List[str],
        targets: Dict[str, np.ndarray],
        featurizer: MolecularFeaturizer,
        model_type: str = 'dmpnn'
    ):
        """
        Initialize ADMET dataset.

        Args:
            smiles_list: List of SMILES strings
            targets: Dict mapping task names to target arrays [N, 1]
            featurizer: Molecular featurizer
            model_type: Model type ('dmpnn', 'fingerprint', or 'hybrid')
        """
        self.smiles_list = smiles_list
        self.targets = targets
        self.featurizer = featurizer
        self.model_type = model_type

        # Validate targets
        n_samples = len(smiles_list)
        for task, values in targets.items():
            if len(values) != n_samples:
                raise ValueError(f"Task '{task}' has {len(values)} values but expected {n_samples}")

        # Precompute features
        print(f"Featurizing {len(smiles_list)} molecules...")
        self.features, failed = featurizer.featurize_batch(
            smiles_list,
            include_morgan=(model_type in ['fingerprint', 'hybrid']),
            include_maccs=(model_type in ['fingerprint', 'hybrid']),
            include_descriptors=(model_type in ['fingerprint', 'hybrid']),
            include_graph=(model_type in ['dmpnn', 'hybrid'])
        )

        if len(failed) > 0:
            warnings.warn(f"Failed to featurize {len(failed)} molecules. These will be skipped.")

        print(f"Successfully featurized {len(self.features)} molecules")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[MoleculeFeatures, Dict[str, float]]:
        """Get molecule features and targets"""
        feat = self.features[idx]

        # Get targets for this molecule
        # Handle missing values (NaN) gracefully
        mol_targets = {}
        for task, values in self.targets.items():
            mol_targets[task] = values[idx]

        return feat, mol_targets


def collate_admet_batch(
    batch: List[Tuple[MoleculeFeatures, Dict[str, float]]],
    model_type: str = 'dmpnn'
) -> Tuple[Union[Batch, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Collate function for ADMET dataloader.

    Args:
        batch: List of (features, targets) tuples
        model_type: Model type

    Returns:
        Tuple of (batched_features, batched_targets)
    """
    features, targets = zip(*batch)

    # Collate targets
    task_names = targets[0].keys()
    batched_targets = {}
    for task in task_names:
        task_values = [t[task] for t in targets]
        batched_targets[task] = torch.tensor(task_values, dtype=torch.float32).unsqueeze(1)

    # Collate features based on model type
    if model_type == 'dmpnn':
        # Batch graphs
        graphs = [f.graph_data for f in features]
        batched_features = Batch.from_data_list(graphs)

    elif model_type == 'fingerprint':
        # Stack fingerprints
        from .fingerprints import MolecularFeaturizer
        featurizer = MolecularFeaturizer()
        fps = [featurizer.get_combined_fingerprint(f) for f in features]
        batched_features = torch.tensor(np.stack(fps), dtype=torch.float32)

    else:  # hybrid
        raise NotImplementedError("Hybrid model not yet implemented")

    return batched_features, batched_targets


class FingerprintModel(nn.Module):
    """
    Feed-forward neural network for fingerprint-based ADMET prediction.

    Simpler than graph models but can be effective with good features.
    """

    def __init__(
        self,
        input_dim: int,
        task_names: List[str],
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize fingerprint model.

        Args:
            input_dim: Dimension of input fingerprint
            task_names: List of prediction tasks
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.task_names = task_names

        # Build shared encoder
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task in task_names:
            self.task_heads[task] = nn.Sequential(
                nn.Linear(prev_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1)
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input fingerprints [batch_size, input_dim]

        Returns:
            Dict mapping task names to predictions
        """
        # Shared encoding
        encoded = self.encoder(x)

        # Task-specific predictions
        predictions = {}
        for task in self.task_names:
            predictions[task] = self.task_heads[task](encoded)

        return predictions


class ADMETPredictor:
    """
    Complete ADMET prediction system with training and inference.
    """

    def __init__(self, config: ADMETConfig):
        """
        Initialize ADMET predictor.

        Args:
            config: Configuration object
        """
        self.config = config
        self.featurizer = MolecularFeaturizer()
        self.model: Optional[nn.Module] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Determine active tasks
        if config.active_tasks is None:
            self.task_names = list(ADMET_TASKS.keys())
        else:
            self.task_names = config.active_tasks

        print(f"Initialized ADMET predictor with tasks: {self.task_names}")
        print(f"Device: {self.device}")

    def create_model(
        self,
        node_features: Optional[int] = None,
        edge_features: Optional[int] = None,
        fingerprint_dim: Optional[int] = None
    ) -> nn.Module:
        """
        Create model based on configuration.

        Args:
            node_features: Dimension of node features (for DMPNN)
            edge_features: Dimension of edge features (for DMPNN)
            fingerprint_dim: Dimension of fingerprint (for fingerprint model)

        Returns:
            Initialized model
        """
        if self.config.model_type == 'dmpnn':
            if node_features is None or edge_features is None:
                # Default from fingerprints.py
                node_features = 13 + 5 + 5  # atom_type + features + hybridization
                edge_features = 6  # bond_type + features

            model = create_dmpnn_model(
                node_features=node_features,
                edge_features=edge_features,
                task_names=self.task_names,
                config={
                    'hidden_dim': self.config.hidden_dim,
                    'num_message_passing_steps': self.config.num_message_passing_steps,
                    'num_ffn_layers': self.config.num_ffn_layers,
                    'task_head_hidden_dim': self.config.task_head_hidden_dim,
                    'dropout': self.config.dropout,
                    'pooling': self.config.pooling
                }
            )

        elif self.config.model_type == 'fingerprint':
            if fingerprint_dim is None:
                # Morgan (2048) + MACCS (166) + RDKit descriptors (20)
                fingerprint_dim = 2048 + 166 + 20

            model = FingerprintModel(
                input_dim=fingerprint_dim,
                task_names=self.task_names,
                hidden_dims=[512, 256, 128],
                dropout=self.config.dropout
            )

        else:
            raise NotImplementedError("Hybrid model not yet implemented")

        return model.to(self.device)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: Optional[Path] = None
    ) -> Dict[str, List[float]]:
        """
        Train ADMET model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_path: Path to save best model

        Returns:
            Training history dictionary
        """
        if self.model is None:
            # Infer input dimensions from first batch
            sample_batch = next(iter(train_loader))
            features, _ = sample_batch

            if self.config.model_type == 'dmpnn':
                node_features = features.x.shape[1]
                edge_features = features.edge_attr.shape[1]
                self.model = self.create_model(node_features, edge_features)
            else:
                fingerprint_dim = features.shape[1]
                self.model = self.create_model(fingerprint_dim=fingerprint_dim)

        # Optimizer and scheduler
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
        }
        for task in self.task_names:
            history[f'train_{task}_loss'] = []
            history[f'val_{task}_loss'] = []

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\nStarting training for {self.config.num_epochs} epochs...")

        for epoch in range(self.config.num_epochs):
            # Training phase
            train_loss, train_task_losses = self._train_epoch(train_loader, optimizer)

            # Validation phase
            val_loss, val_task_losses = self._validate(val_loader)

            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            for task in self.task_names:
                history[f'train_{task}_loss'].append(train_task_losses[task])
                history[f'val_{task}_loss'].append(val_task_losses[task])

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                if save_path is not None:
                    self.save_model(save_path)
                    print(f"  Saved best model to {save_path}")

            else:
                patience_counter += 1

            # Print progress
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            for task in self.task_names:
                print(f"    {task}: Train={train_task_losses[task]:.4f}, Val={val_task_losses[task]:.4f}")

            if patience_counter >= self.config.patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break

        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")

        return history

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.task_names}
        num_batches = 0

        for features, targets in train_loader:
            # Move to device
            if self.config.model_type == 'dmpnn':
                features = features.to(self.device)
            else:
                features = features.to(self.device)

            targets = {task: t.to(self.device) for task, t in targets.items()}

            # Forward pass
            optimizer.zero_grad()

            if self.config.model_type == 'dmpnn':
                predictions = self.model(
                    features.x,
                    features.edge_index,
                    features.edge_attr,
                    features.batch
                )
            else:
                predictions = self.model(features)

            # Compute loss for each task
            loss = 0.0
            for task in self.task_names:
                # Skip if all targets are NaN
                mask = ~torch.isnan(targets[task])
                if mask.sum() == 0:
                    continue

                task_pred = predictions[task][mask]
                task_target = targets[task][mask]

                # MSE loss for regression tasks
                task_loss = F.mse_loss(task_pred, task_target)
                loss += task_loss

                task_losses[task] += task_loss.item()

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Average losses
        avg_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}

        return avg_loss, avg_task_losses

    def _validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate model"""
        self.model.eval()

        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.task_names}
        num_batches = 0

        with torch.no_grad():
            for features, targets in val_loader:
                # Move to device
                if self.config.model_type == 'dmpnn':
                    features = features.to(self.device)
                else:
                    features = features.to(self.device)

                targets = {task: t.to(self.device) for task, t in targets.items()}

                # Forward pass
                if self.config.model_type == 'dmpnn':
                    predictions = self.model(
                        features.x,
                        features.edge_index,
                        features.edge_attr,
                        features.batch
                    )
                else:
                    predictions = self.model(features)

                # Compute loss
                loss = 0.0
                for task in self.task_names:
                    mask = ~torch.isnan(targets[task])
                    if mask.sum() == 0:
                        continue

                    task_pred = predictions[task][mask]
                    task_target = targets[task][mask]

                    task_loss = F.mse_loss(task_pred, task_target)
                    loss += task_loss

                    task_losses[task] += task_loss.item()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}

        return avg_loss, avg_task_losses

    def predict(
        self,
        smiles_list: List[str],
        return_uncertainty: bool = False
    ) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """
        Make predictions for list of SMILES.

        Args:
            smiles_list: List of SMILES strings
            return_uncertainty: Whether to compute uncertainties

        Returns:
            Tuple of (predictions, uncertainties)
            predictions: Dict mapping task names to prediction arrays
            uncertainties: Dict mapping task names to uncertainty arrays (or None)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Train or load a model first.")

        self.model.eval()

        # Featurize molecules
        features, failed = self.featurizer.featurize_batch(
            smiles_list,
            include_morgan=(self.config.model_type in ['fingerprint', 'hybrid']),
            include_maccs=(self.config.model_type in ['fingerprint', 'hybrid']),
            include_descriptors=(self.config.model_type in ['fingerprint', 'hybrid']),
            include_graph=(self.config.model_type in ['dmpnn', 'hybrid'])
        )

        if len(failed) > 0:
            warnings.warn(f"Failed to featurize {len(failed)} molecules")

        # Create dummy targets (not used for inference)
        dummy_targets = {task: np.zeros((len(features), 1)) for task in self.task_names}

        # Create dataset and loader
        dataset = ADMETDataset(
            [f.smiles for f in features],
            dummy_targets,
            self.featurizer,
            self.config.model_type
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_admet_batch(batch, self.config.model_type)
        )

        # Collect predictions
        all_predictions = {task: [] for task in self.task_names}

        with torch.no_grad():
            for features_batch, _ in loader:
                if self.config.model_type == 'dmpnn':
                    features_batch = features_batch.to(self.device)
                    preds = self.model(
                        features_batch.x,
                        features_batch.edge_index,
                        features_batch.edge_attr,
                        features_batch.batch
                    )
                else:
                    features_batch = features_batch.to(self.device)
                    preds = self.model(features_batch)

                for task in self.task_names:
                    all_predictions[task].append(preds[task].cpu().numpy())

        # Concatenate predictions
        predictions = {
            task: np.concatenate(preds, axis=0)
            for task, preds in all_predictions.items()
        }

        uncertainties = None
        if return_uncertainty:
            # TODO: Implement uncertainty estimation
            warnings.warn("Uncertainty estimation not yet implemented")

        return predictions, uncertainties

    def save_model(self, path: Path):
        """Save model checkpoint"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'task_names': self.task_names
        }, path)

    def load_model(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.config = checkpoint['config']
        self.task_names = checkpoint['task_names']

        # Create model with saved config
        if self.model is None:
            self.model = self.create_model()

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()


if __name__ == "__main__":
    print("ADMET Predictor Module")
    print("=" * 50)
    print("\nThis module provides comprehensive ADMET prediction capabilities.")
    print("\nExample usage:")
    print("""
    from admet_predictor import ADMETPredictor, ADMETConfig

    # Configure predictor
    config = ADMETConfig(
        model_type='dmpnn',
        hidden_dim=256,
        num_epochs=100,
        batch_size=32
    )

    # Initialize
    predictor = ADMETPredictor(config)

    # Train (requires dataset)
    # predictor.train(train_loader, val_loader, save_path='model.pt')

    # Predict
    smiles = ['CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O']
    predictions, uncertainties = predictor.predict(smiles)

    print(predictions['solubility'])
    """)
