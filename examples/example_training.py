"""
Example Training Script for ADMET Predictor

This script demonstrates:
1. Loading and preprocessing ADMET data
2. Creating train/validation/test splits
3. Training a D-MPNN model
4. Evaluating predictions with uncertainty
5. Visualizing results

Author: Claude Code
Date: 2025-10-20
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from src.models import (
    ADMETPredictor,
    ADMETConfig,
    ADMETDataset,
    MolecularFeaturizer,
    collate_admet_batch
)
from src.models.data_utils import ADMETDataProcessor, create_sample_dataset


def plot_training_history(history: dict, save_path: Path):
    """Plot training history"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Overall loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Overall Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Per-task validation loss
    task_names = [k.replace('val_', '').replace('_loss', '')
                  for k in history.keys() if k.startswith('val_') and k != 'val_loss']

    for task in task_names:
        key = f'val_{task}_loss'
        if key in history:
            axes[1].plot(history[key], label=task, linewidth=2)

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Per-Task Validation Loss', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training history plot to {save_path}")


def plot_predictions(
    predictions: dict,
    targets: dict,
    task_names: list,
    save_path: Path
):
    """Plot predicted vs actual values"""
    n_tasks = len(task_names)
    n_cols = 3
    n_rows = (n_tasks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_tasks > 1 else [axes]

    for i, task in enumerate(task_names):
        if task not in predictions or task not in targets:
            continue

        pred = predictions[task].flatten()
        tgt = targets[task].flatten()

        # Remove NaN values
        mask = ~(np.isnan(pred) | np.isnan(tgt))
        pred = pred[mask]
        tgt = tgt[mask]

        # Scatter plot
        axes[i].scatter(tgt, pred, alpha=0.5, s=30)

        # Perfect prediction line
        min_val = min(tgt.min(), pred.min())
        max_val = max(tgt.max(), pred.max())
        axes[i].plot([min_val, max_val], [min_val, max_val],
                    'r--', linewidth=2, label='Perfect Prediction')

        # Calculate metrics
        mae = np.abs(pred - tgt).mean()
        rmse = np.sqrt(((pred - tgt) ** 2).mean())
        r2 = 1 - ((pred - tgt) ** 2).sum() / ((tgt - tgt.mean()) ** 2).sum()

        axes[i].set_xlabel('Actual', fontsize=11)
        axes[i].set_ylabel('Predicted', fontsize=11)
        axes[i].set_title(f'{task}\nMAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}',
                         fontsize=12, fontweight='bold')
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(task_names), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved predictions plot to {save_path}")


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("ADMET Predictor Training Example")
    print("=" * 70)

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Create sample dataset
    print("\n[1/7] Creating sample dataset...")
    data_path = output_dir / "sample_admet_data.csv"
    create_sample_dataset(data_path, n_samples=500, random_state=42)

    # Step 2: Load and process data
    print("\n[2/7] Loading and processing data...")
    processor = ADMETDataProcessor(
        normalize=True,
        handle_missing='drop'
    )

    smiles_list, targets = processor.load_csv(
        data_path,
        smiles_column='smiles',
        validate_smiles=True
    )

    # Step 3: Split data
    print("\n[3/7] Splitting data...")
    train_smiles, train_targets, \
    val_smiles, val_targets, \
    test_smiles, test_targets = processor.split_data(
        smiles_list,
        targets,
        train_size=0.7,
        val_size=0.15,
        random_state=42
    )

    # Step 4: Configure model
    print("\n[4/7] Configuring model...")
    config = ADMETConfig(
        model_type='dmpnn',
        hidden_dim=128,           # Smaller for demo
        num_message_passing_steps=2,
        num_ffn_layers=2,
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=20,            # Fewer epochs for demo
        patience=5,
        use_evidential=False      # Use standard MSE for simplicity
    )

    # Step 5: Create datasets and loaders
    print("\n[5/7] Creating datasets...")
    featurizer = MolecularFeaturizer()

    train_dataset = ADMETDataset(
        train_smiles,
        train_targets,
        featurizer,
        config.model_type
    )

    val_dataset = ADMETDataset(
        val_smiles,
        val_targets,
        featurizer,
        config.model_type
    )

    test_dataset = ADMETDataset(
        test_smiles,
        test_targets,
        featurizer,
        config.model_type
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_admet_batch(batch, config.model_type)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_admet_batch(batch, config.model_type)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_admet_batch(batch, config.model_type)
    )

    # Step 6: Train model
    print("\n[6/7] Training model...")
    predictor = ADMETPredictor(config)

    model_path = output_dir / "best_model.pt"
    history = predictor.train(
        train_loader,
        val_loader,
        save_path=model_path
    )

    # Plot training history
    plot_training_history(history, output_dir / "training_history.png")

    # Step 7: Evaluate on test set
    print("\n[7/7] Evaluating on test set...")
    predictor.load_model(model_path)

    predictions, uncertainties = predictor.predict(
        test_smiles,
        return_uncertainty=False
    )

    # Denormalize predictions
    predictions = processor.denormalize_predictions(predictions)

    # Plot predictions
    plot_predictions(
        predictions,
        test_targets,
        predictor.task_names,
        output_dir / "test_predictions.png"
    )

    # Print summary statistics
    print("\n" + "=" * 70)
    print("Test Set Performance Summary")
    print("=" * 70)

    for task in predictor.task_names:
        if task in predictions and task in test_targets:
            pred = predictions[task].flatten()
            tgt = test_targets[task].flatten()

            # Remove NaN
            mask = ~(np.isnan(pred) | np.isnan(tgt))
            pred = pred[mask]
            tgt = tgt[mask]

            mae = np.abs(pred - tgt).mean()
            rmse = np.sqrt(((pred - tgt) ** 2).mean())
            r2 = 1 - ((pred - tgt) ** 2).sum() / ((tgt - tgt.mean()) ** 2).sum()

            print(f"\n{task.upper()}:")
            print(f"  MAE:  {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²:   {r2:.4f}")

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Plots saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
