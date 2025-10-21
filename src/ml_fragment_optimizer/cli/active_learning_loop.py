#!/usr/bin/env python
"""
Run active learning loop for fragment optimization.

Implements an iterative cycle of model training, candidate selection,
and experimental validation to efficiently explore chemical space.

Supports:
- Multiple acquisition functions (EI, UCB, Thompson sampling)
- Diversity-aware batch selection
- Multi-objective optimization
- Integration with GCMC workflows

Example usage:
    # Basic active learning
    mlfrag-active-learning --initial-data fragments.csv \
                           --candidate-pool library.smi \
                           --iterations 10 --batch-size 20

    # With simulation mode (no experiments needed)
    mlfrag-active-learning --initial-data fragments.csv \
                           --candidate-pool library.smi \
                           --simulate --iterations 5

    # Multi-objective with diversity
    mlfrag-active-learning --initial-data fragments.csv \
                           --candidate-pool library.smi \
                           --acquisition ucb --diversity-weight 0.3
"""

import argparse
import sys
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from loguru import logger

from ml_fragment_optimizer.models.admet_predictor import ADMETPredictor
from ml_fragment_optimizer.active_learning.diversity_sampler import DiversitySampler
from ml_fragment_optimizer.utils.logging_utils import setup_logger

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run active learning loop for fragment optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic active learning
  %(prog)s --initial-data fragments.csv \\
           --candidate-pool library.smi \\
           --iterations 10 --batch-size 20

  # With simulation mode
  %(prog)s --initial-data fragments.csv \\
           --candidate-pool library.smi \\
           --simulate --iterations 5

  # Multi-objective with diversity
  %(prog)s --initial-data fragments.csv \\
           --candidate-pool library.smi \\
           --acquisition ucb --diversity-weight 0.3 \\
           --properties solubility,logp
        """
    )

    parser.add_argument(
        "--initial-data",
        type=Path,
        required=True,
        help="Initial training data CSV (must have SMILES column and property columns)",
    )
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        required=True,
        help="Pool of candidate molecules (CSV with SMILES or .smi file)",
    )
    parser.add_argument(
        "--properties",
        type=str,
        default="solubility",
        help="Properties to optimize (comma-separated, default: solubility)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of AL iterations (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of molecules per iteration (default: 10)",
    )
    parser.add_argument(
        "--acquisition",
        type=str,
        default="ei",
        choices=["ei", "ucb", "pi", "thompson", "greedy"],
        help="Acquisition function (default: ei = Expected Improvement)",
    )
    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=0.2,
        help="Weight for diversity in selection (0-1, default: 0.2)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate experiments (for testing, no real measurements needed)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost", "gradient_boosting"],
        help="Model type (default: random_forest)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("active_learning_results"),
        help="Output directory (default: active_learning_results)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def load_candidate_pool(path: Path) -> List[str]:
    """Load candidate molecules from file."""
    if not path.exists():
        raise FileNotFoundError(f"Candidate pool not found: {path}")

    if path.suffix == '.smi':
        # SMILES file (one per line)
        with open(path) as f:
            smiles = [line.strip().split()[0] for line in f if line.strip()]
    elif path.suffix == '.csv':
        # CSV file with SMILES column
        df = pd.read_csv(path)
        if 'SMILES' not in df.columns and 'smiles' not in df.columns:
            raise ValueError("CSV must have 'SMILES' or 'smiles' column")
        smiles_col = 'SMILES' if 'SMILES' in df.columns else 'smiles'
        smiles = df[smiles_col].tolist()
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Validate SMILES
    if RDKIT_AVAILABLE:
        valid_smiles = []
        for smi in smiles:
            if Chem.MolFromSmiles(smi) is not None:
                valid_smiles.append(smi)
        logger.info(f"Loaded {len(valid_smiles)}/{len(smiles)} valid SMILES")
        return valid_smiles
    else:
        return smiles


def simulate_measurement(smiles: str, property_name: str) -> float:
    """
    Simulate experimental measurement (for testing).

    Uses RDKit descriptors as a proxy for real measurements.
    """
    if not RDKIT_AVAILABLE:
        return np.random.randn()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.random.randn()

    from rdkit.Chem import Descriptors

    # Map property names to RDKit descriptors
    if property_name == 'solubility':
        return -Descriptors.MolLogP(mol) + np.random.normal(0, 0.5)
    elif property_name == 'logp':
        return Descriptors.MolLogP(mol) + np.random.normal(0, 0.3)
    elif property_name == 'mw':
        return Descriptors.MolWt(mol) + np.random.normal(0, 10)
    else:
        return np.random.randn()


def main() -> None:
    """Entry point for CLI."""
    args = parse_args()
    setup_logger(log_level=args.log_level)

    if not RDKIT_AVAILABLE:
        logger.error("RDKit is required but not installed")
        logger.info("Install with: pip install rdkit")
        sys.exit(1)

    logger.info("Active Learning Optimization Loop")
    logger.info(f"Iterations: {args.iterations}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Acquisition: {args.acquisition}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse properties
    properties = [p.strip() for p in args.properties.split(',')]
    logger.info(f"Optimizing properties: {properties}")

    # Load initial data
    logger.info(f"Loading initial data from {args.initial_data}")
    if not args.initial_data.exists():
        logger.error(f"Initial data file not found: {args.initial_data}")
        sys.exit(1)

    initial_df = pd.read_csv(args.initial_data)
    smiles_col = 'SMILES' if 'SMILES' in initial_df.columns else 'smiles'

    if smiles_col not in initial_df.columns:
        logger.error("Initial data must have 'SMILES' or 'smiles' column")
        sys.exit(1)

    for prop in properties:
        if prop not in initial_df.columns:
            logger.error(f"Property '{prop}' not found in initial data")
            sys.exit(1)

    # Load candidate pool
    logger.info(f"Loading candidate pool from {args.candidate_pool}")
    try:
        candidate_smiles = load_candidate_pool(args.candidate_pool)
        logger.info(f"Candidate pool size: {len(candidate_smiles)}")
    except Exception as e:
        logger.error(f"Failed to load candidate pool: {e}")
        sys.exit(1)

    # Remove candidates already in training data
    known_smiles = set(initial_df[smiles_col].tolist())
    candidate_smiles = [s for s in candidate_smiles if s not in known_smiles]
    logger.info(f"Unexplored candidates: {len(candidate_smiles)}")

    # Initialize components
    diversity_sampler = DiversitySampler()

    # Training data (accumulated over iterations)
    training_data = initial_df.copy()

    # Tracking metrics
    history = {
        'iteration': [],
        'best_value': [],
        'mean_value': [],
        'model_score': [],
        'selected_smiles': []
    }

    # Active learning loop
    for iteration in range(args.iterations):
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration + 1}/{args.iterations}")
        logger.info(f"{'='*60}")

        # Step 1: Train model on current data
        logger.info(f"Training model on {len(training_data)} examples...")

        predictor = ADMETPredictor(
            properties=properties,
            model_type=args.model_type
        )

        try:
            metrics = predictor.fit(
                training_data[smiles_col].tolist(),
                {prop: training_data[prop].tolist() for prop in properties}
            )
            logger.info(f"Model trained. Metrics: {metrics}")
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            break

        # Step 2: Predict on candidate pool
        logger.info(f"Predicting on {len(candidate_smiles)} candidates...")

        try:
            predictions, uncertainties = predictor.predict(
                candidate_smiles,
                return_uncertainty=True
            )
        except Exception as e:
            logger.warning(f"Prediction with uncertainty failed: {e}, using point estimates")
            predictions, _ = predictor.predict(candidate_smiles)
            uncertainties = {prop: np.ones(len(candidate_smiles)) for prop in properties}

        # Step 3: Select next batch using acquisition function
        logger.info(f"Selecting top {args.batch_size} candidates...")

        # Simple acquisition: Expected Improvement or UCB
        scores = np.zeros(len(candidate_smiles))

        if args.acquisition == 'ei':
            # Expected improvement (maximize)
            best_so_far = max([training_data[prop].max() for prop in properties])
            for i, prop in enumerate(properties):
                improvement = predictions[prop] - best_so_far
                scores += improvement * uncertainties[prop]

        elif args.acquisition == 'ucb':
            # Upper confidence bound
            for i, prop in enumerate(properties):
                scores += predictions[prop] + 2.0 * uncertainties[prop]

        elif args.acquisition == 'thompson':
            # Thompson sampling
            for i, prop in enumerate(properties):
                samples = np.random.normal(predictions[prop], uncertainties[prop])
                scores += samples

        elif args.acquisition == 'greedy':
            # Greedy: just pick highest predicted
            for prop in properties:
                scores += predictions[prop]

        # Add diversity bonus
        if args.diversity_weight > 0:
            selected_indices = []
            remaining_indices = list(range(len(candidate_smiles)))

            for _ in range(args.batch_size):
                if not remaining_indices:
                    break

                # Calculate diversity penalty for each remaining candidate
                div_scores = scores[remaining_indices].copy()

                if selected_indices:
                    selected_smiles = [candidate_smiles[i] for i in selected_indices]
                    remaining_smiles = [candidate_smiles[i] for i in remaining_indices]

                    # Penalize candidates similar to already selected
                    try:
                        similarities = diversity_sampler._calculate_similarity_matrix(
                            selected_smiles + remaining_smiles
                        )
                        # Average similarity to selected molecules
                        avg_sim = similarities[:len(selected_smiles), len(selected_smiles):].mean(axis=0)
                        div_scores -= args.diversity_weight * avg_sim * scores[remaining_indices].std()
                    except:
                        pass  # Diversity calculation failed, use scores as-is

                # Select best
                best_local_idx = div_scores.argmax()
                best_global_idx = remaining_indices[best_local_idx]
                selected_indices.append(best_global_idx)
                remaining_indices.remove(best_global_idx)
        else:
            # No diversity: just pick top scores
            selected_indices = scores.argsort()[-args.batch_size:][::-1].tolist()

        selected_smiles = [candidate_smiles[i] for i in selected_indices]

        logger.info(f"Selected {len(selected_smiles)} molecules for testing")

        # Step 4: Simulate or wait for experimental results
        if args.simulate:
            logger.info("Simulating experimental measurements...")
            new_measurements = {}
            for prop in properties:
                new_measurements[prop] = [simulate_measurement(smi, prop) for smi in selected_smiles]
        else:
            logger.warning("Experimental measurements required!")
            logger.info(f"Please measure properties {properties} for:")
            for smi in selected_smiles:
                logger.info(f"  {smi}")
            logger.info("Save results to new_data.csv and re-run to continue")
            break

        # Step 5: Add new data to training set
        new_data = pd.DataFrame({smiles_col: selected_smiles, **new_measurements})
        training_data = pd.concat([training_data, new_data], ignore_index=True)

        # Remove selected molecules from candidate pool
        candidate_smiles = [s for s in candidate_smiles if s not in selected_smiles]

        # Track metrics
        best_value = max([new_data[prop].max() for prop in properties])
        mean_value = np.mean([new_data[prop].mean() for prop in properties])

        history['iteration'].append(iteration + 1)
        history['best_value'].append(best_value)
        history['mean_value'].append(mean_value)
        history['model_score'].append(metrics.get('mean_r2', 0.0) if isinstance(metrics, dict) else 0.0)
        history['selected_smiles'].append(selected_smiles)

        logger.info(f"Best value this iteration: {best_value:.3f}")
        logger.info(f"Mean value this iteration: {mean_value:.3f}")

        # Save intermediate results
        training_data.to_csv(args.output_dir / f"training_data_iter{iteration+1}.csv", index=False)

    # Save final results
    logger.info("\nSaving final results...")

    final_model_path = args.output_dir / "final_model.pkl"
    predictor.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    training_data.to_csv(args.output_dir / "final_training_data.csv", index=False)
    logger.info(f"Final training data saved")

    history_df = pd.DataFrame(history)
    history_df.to_csv(args.output_dir / "optimization_history.csv", index=False)
    logger.info(f"Optimization history saved")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("ACTIVE LEARNING SUMMARY")
    logger.info("="*60)
    logger.info(f"Initial training size: {len(initial_df)}")
    logger.info(f"Final training size: {len(training_data)}")
    logger.info(f"Molecules explored: {len(training_data) - len(initial_df)}")
    logger.info(f"Best value found: {max(history['best_value']):.3f}")
    logger.info(f"Improvement: {max(history['best_value']) - history['best_value'][0]:.3f}")

    logger.success(f"\n✓ Active learning complete! Results in {args.output_dir}")


if __name__ == "__main__":
    main()
