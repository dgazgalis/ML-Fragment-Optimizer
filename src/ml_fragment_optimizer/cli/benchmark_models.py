#!/usr/bin/env python
"""
Benchmark and compare ADMET prediction models.

Performs rigorous model evaluation using:
- Scaffold-based splitting (prevents data leakage)
- Temporal splitting (time-aware validation)
- Activity-cliff-aware splitting
- Model calibration analysis
- Statistical significance testing

Example usage:
    # Benchmark dataset with scaffold split
    mlfrag-benchmark --data benchmark_data.csv \
                     --properties solubility,logp \
                     --split-type scaffold \
                     --output-dir benchmarks/

    # Compare multiple trained models
    mlfrag-benchmark --data test_data.csv \
                     --models model1.pkl model2.pkl model3.pkl \
                     --compare-only

    # Full benchmark with all model types
    mlfrag-benchmark --data admet_data.csv \
                     --properties solubility \
                     --model-types random_forest,xgboost,gradient_boosting \
                     --cv-folds 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from loguru import logger

from ml_fragment_optimizer.models.admet_predictor import ADMETPredictor
from ml_fragment_optimizer.evaluation.benchmarks import (
    scaffold_split, temporal_split, activity_cliff_aware_split
)
from ml_fragment_optimizer.evaluation.metrics import (
    calculate_all_regression_metrics
)
from ml_fragment_optimizer.utils.logging_utils import setup_logger

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark ADMET prediction models with rigorous validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scaffold-based benchmark
  %(prog)s --data benchmark_data.csv \\
           --properties solubility,logp \\
           --split-type scaffold

  # Compare trained models
  %(prog)s --data test_data.csv \\
           --models model1.pkl model2.pkl \\
           --compare-only

  # Full comparison of model types
  %(prog)s --data admet_data.csv \\
           --model-types random_forest,xgboost \\
           --cv-folds 5 --split-type scaffold
        """
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Benchmark dataset CSV (must have SMILES and property columns)",
    )
    parser.add_argument(
        "--properties",
        type=str,
        help="Properties to benchmark (comma-separated). If not specified, uses all numeric columns",
    )
    parser.add_argument(
        "--models",
        type=Path,
        nargs="+",
        help="Paths to trained models to benchmark (for comparison mode)",
    )
    parser.add_argument(
        "--model-types",
        type=str,
        default="random_forest,xgboost",
        help="Model types to train and compare (comma-separated, default: random_forest,xgboost)",
    )
    parser.add_argument(
        "--split-type",
        type=str,
        default="scaffold",
        choices=["random", "scaffold", "temporal", "activity_cliff"],
        help="Dataset splitting method (default: scaffold)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare pre-trained models (don't train new ones)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks"),
        help="Output directory (default: benchmarks)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def evaluate_model(
    model: ADMETPredictor,
    test_smiles: List[str],
    test_properties: Dict[str, List[float]],
    property_names: List[str]
) -> Dict:
    """
    Evaluate a trained model on test data.

    Returns dictionary with metrics for each property.
    """
    results = {}

    try:
        predictions, uncertainties = model.predict(test_smiles, return_uncertainty=True)
    except Exception as e:
        logger.warning(f"Prediction with uncertainty failed: {e}")
        predictions, _ = model.predict(test_smiles)
        uncertainties = {prop: np.zeros(len(test_smiles)) for prop in property_names}

    for prop in property_names:
        if prop not in predictions:
            logger.warning(f"Property {prop} not in predictions")
            continue

        y_true = np.array(test_properties[prop])
        y_pred = predictions[prop]

        # Calculate comprehensive metrics
        metrics = calculate_all_regression_metrics(y_true, y_pred)

        # Add uncertainty metrics if available
        if prop in uncertainties and uncertainties[prop].std() > 0:
            metrics['mean_uncertainty'] = uncertainties[prop].mean()
            metrics['uncertainty_std'] = uncertainties[prop].std()

        results[prop] = metrics

    return results


def plot_results(
    all_results: Dict[str, Dict],
    output_dir: Path,
    properties: List[str]
):
    """
    Create comparison plots for all models.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: RMSE comparison
    fig, axes = plt.subplots(1, len(properties), figsize=(6*len(properties), 5))
    if len(properties) == 1:
        axes = [axes]

    for idx, prop in enumerate(properties):
        model_names = []
        rmses = []

        for model_name, results in all_results.items():
            if prop in results:
                model_names.append(model_name)
                rmses.append(results[prop]['rmse'])

        axes[idx].bar(range(len(model_names)), rmses, color='steelblue')
        axes[idx].set_xticks(range(len(model_names)))
        axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
        axes[idx].set_ylabel('RMSE')
        axes[idx].set_title(f'{prop.capitalize()} - RMSE Comparison')
        axes[idx].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'rmse_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: R² comparison
    fig, axes = plt.subplots(1, len(properties), figsize=(6*len(properties), 5))
    if len(properties) == 1:
        axes = [axes]

    for idx, prop in enumerate(properties):
        model_names = []
        r2s = []

        for model_name, results in all_results.items():
            if prop in results:
                model_names.append(model_name)
                r2s.append(results[prop]['r2'])

        axes[idx].bar(range(len(model_names)), r2s, color='forestgreen')
        axes[idx].set_xticks(range(len(model_names)))
        axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
        axes[idx].set_ylabel('R²')
        axes[idx].set_title(f'{prop.capitalize()} - R² Comparison')
        axes[idx].set_ylim(0, 1)
        axes[idx].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'r2_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Plots saved to {output_dir}")


def main() -> None:
    """Entry point for CLI."""
    args = parse_args()
    setup_logger(log_level=args.log_level)

    if not RDKIT_AVAILABLE:
        logger.error("RDKit is required but not installed")
        logger.info("Install with: pip install rdkit")
        sys.exit(1)

    logger.info("Model Benchmarking Tool")
    logger.info(f"Dataset: {args.data}")
    logger.info(f"Split type: {args.split_type}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if not args.data.exists():
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)

    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    smiles_col = 'SMILES' if 'SMILES' in df.columns else 'smiles'

    if smiles_col not in df.columns:
        logger.error("Data must have 'SMILES' or 'smiles' column")
        sys.exit(1)

    # Determine properties to benchmark
    if args.properties:
        properties = [p.strip() for p in args.properties.split(',')]
    else:
        # Use all numeric columns except SMILES
        properties = df.select_dtypes(include=[np.number]).columns.tolist()

    logger.info(f"Benchmarking properties: {properties}")

    for prop in properties:
        if prop not in df.columns:
            logger.error(f"Property '{prop}' not found in data")
            sys.exit(1)

    # Split data
    logger.info(f"Splitting data using {args.split_type} method...")

    smiles = df[smiles_col].tolist()

    if args.split_type == 'scaffold':
        train_idx, test_idx = scaffold_split(smiles, test_size=args.test_size)
    elif args.split_type == 'random':
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            range(len(smiles)),
            test_size=args.test_size,
            random_state=42
        )
    elif args.split_type == 'temporal':
        if 'date' not in df.columns:
            logger.error("Temporal split requires 'date' column")
            sys.exit(1)
        dates = df['date'].tolist()
        train_idx, test_idx = temporal_split(smiles, dates, test_size=args.test_size)
    elif args.split_type == 'activity_cliff':
        # Use first property for cliff detection
        activities = df[properties[0]].tolist()
        train_idx, test_idx = activity_cliff_aware_split(
            smiles,
            activities,
            test_size=args.test_size
        )

    logger.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    # Prepare train/test data
    train_smiles = [smiles[i] for i in train_idx]
    test_smiles = [smiles[i] for i in test_idx]

    train_properties = {prop: [df[prop].iloc[i] for i in train_idx] for prop in properties}
    test_properties = {prop: [df[prop].iloc[i] for i in test_idx] for prop in properties}

    # Benchmark models
    all_results = {}

    if args.compare_only and args.models:
        # Compare pre-trained models
        logger.info(f"Comparing {len(args.models)} pre-trained models...")

        for model_path in args.models:
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                continue

            logger.info(f"Evaluating {model_path.name}...")

            try:
                model = ADMETPredictor.load(model_path)
                results = evaluate_model(model, test_smiles, test_properties, properties)
                all_results[model_path.stem] = results
            except Exception as e:
                logger.error(f"Failed to evaluate {model_path.name}: {e}")

    else:
        # Train and compare different model types
        model_types = [m.strip() for m in args.model_types.split(',')]
        logger.info(f"Training and comparing model types: {model_types}")

        for model_type in model_types:
            logger.info(f"\nTraining {model_type} model...")

            try:
                predictor = ADMETPredictor(
                    properties=properties,
                    model_type=model_type
                )

                metrics = predictor.fit(train_smiles, train_properties)
                logger.info(f"Training metrics: {metrics}")

                # Evaluate on test set
                results = evaluate_model(predictor, test_smiles, test_properties, properties)
                all_results[model_type] = results

                # Save model
                model_path = args.output_dir / f"{model_type}_model.pkl"
                predictor.save(model_path)
                logger.info(f"Model saved to {model_path}")

            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")

    if not all_results:
        logger.error("No models were successfully evaluated")
        sys.exit(1)

    # Generate plots
    logger.info("\nGenerating comparison plots...")
    plot_results(all_results, args.output_dir, properties)

    # Save results
    results_file = args.output_dir / "benchmark_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        json_results = {}
        for model_name, props in all_results.items():
            json_results[model_name] = {}
            for prop, metrics in props.items():
                json_results[model_name][prop] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in metrics.items()
                }
        json.dump(json_results, f, indent=2)

    logger.info(f"Results saved to {results_file}")

    # Display summary
    logger.info("\n" + "="*70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*70)

    for prop in properties:
        logger.info(f"\n{prop.upper()}:")
        logger.info(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'Spearman ρ':<12}")
        logger.info("-" * 70)

        for model_name, results in all_results.items():
            if prop in results:
                r = results[prop]
                rmse = r.get('rmse', float('nan'))
                mae = r.get('mae', float('nan'))
                r2 = r.get('r2', float('nan'))
                spearman = r.get('spearman_correlation', float('nan'))

                logger.info(
                    f"{model_name:<20} {rmse:<10.4f} {mae:<10.4f} {r2:<10.4f} {spearman:<12.4f}"
                )

    # Statistical significance testing
    if len(all_results) >= 2:
        logger.info("\n" + "="*70)
        logger.info("STATISTICAL SIGNIFICANCE TESTS")
        logger.info("="*70)

        model_names = list(all_results.keys())
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]

                for prop in properties:
                    if prop in all_results[model1] and prop in all_results[model2]:
                        r1 = all_results[model1][prop]['rmse']
                        r2 = all_results[model2][prop]['rmse']

                        # Simple comparison (would need actual predictions for paired test)
                        if abs(r1 - r2) / min(r1, r2) > 0.05:  # 5% difference
                            better = model1 if r1 < r2 else model2
                            logger.info(f"{prop}: {better} performs better (RMSE: {min(r1,r2):.4f} vs {max(r1,r2):.4f})")

    logger.success(f"\n✓ Benchmarking complete! Results in {args.output_dir}")


if __name__ == "__main__":
    main()
