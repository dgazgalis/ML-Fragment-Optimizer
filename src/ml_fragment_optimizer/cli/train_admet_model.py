#!/usr/bin/env python
"""
Train multi-task ADMET prediction models.

Example usage:
    mlfrag-train --data admet_data.csv --properties solubility,logp,clearance \
                 --model-type xgboost --output-dir models/admet_v1

    mlfrag-train --config configs/admet_model.yaml --data admet_data.csv
"""

import argparse
import sys
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from loguru import logger

from ml_fragment_optimizer.models.admet_predictor import ADMETPredictor
from ml_fragment_optimizer.utils.featurizers import MolecularFeaturizer
from ml_fragment_optimizer.utils.config_loader import ADMETModelConfig
from ml_fragment_optimizer.utils.logging_utils import setup_logger, log_experiment_start, log_experiment_end


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train multi-task ADMET property prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  %(prog)s --data admet_data.csv --properties solubility,logp

  # Train XGBoost model with custom parameters
  %(prog)s --data admet_data.csv --properties solubility,logp,clearance \\
           --model-type xgboost --n-estimators 200 --learning-rate 0.05

  # Use configuration file
  %(prog)s --config configs/admet_model.yaml --data admet_data.csv

  # Train with descriptors and custom fingerprints
  %(prog)s --data admet_data.csv --properties solubility \\
           --fingerprint-type morgan --fingerprint-bits 4096 \\
           --use-descriptors

Data format:
  CSV file with columns: SMILES, <property1>, <property2>, ...
  Example:
    SMILES,solubility,logp,clearance
    CCO,-0.77,0.46,12.3
    c1ccccc1,-2.13,2.13,8.7
        """,
    )

    # Required arguments
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to training data CSV file (required columns: SMILES, <properties>)",
    )
    parser.add_argument(
        "--properties",
        type=str,
        help="Comma-separated list of properties to predict (column names in CSV)",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file (overrides CLI arguments)",
    )

    # Model architecture
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost", "gradient_boosting"],
        help="Type of ML model (default: random_forest)",
    )
    parser.add_argument(
        "--fingerprint-type",
        type=str,
        default="morgan",
        choices=["morgan", "maccs", "rdkit", "avalon"],
        help="Type of molecular fingerprint (default: morgan)",
    )
    parser.add_argument(
        "--fingerprint-radius",
        type=int,
        default=2,
        help="Radius for Morgan fingerprints (default: 2)",
    )
    parser.add_argument(
        "--fingerprint-bits",
        type=int,
        default=2048,
        help="Number of bits for fingerprints (default: 2048)",
    )
    parser.add_argument(
        "--use-descriptors",
        action="store_true",
        help="Include RDKit molecular descriptors in features",
    )

    # Training parameters
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees/estimators (default: 100)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for gradient boosting (default: 0.1)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Maximum tree depth (None = unlimited for RF, 6 for XGBoost)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/admet"),
        help="Output directory for trained models (default: models/admet)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.config and not args.properties:
        parser.error("Either --properties or --config must be specified")

    return args


def validate_training_smiles(smiles_list: List[str]) -> tuple[List[str], List[int]]:
    """
    Validate SMILES strings for training data.

    Returns:
        Tuple of (valid_smiles, valid_indices)
    """
    from rdkit import Chem

    valid_smiles = []
    valid_indices = []
    invalid_count = 0

    for i, smi in enumerate(smiles_list):
        if not smi or not isinstance(smi, str) or not smi.strip():
            invalid_count += 1
            continue

        try:
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is None:
                invalid_count += 1
                logger.debug(f"Invalid SMILES at index {i}: {smi}")
            else:
                valid_smiles.append(smi.strip())
                valid_indices.append(i)
        except Exception as e:
            invalid_count += 1
            logger.debug(f"Error parsing SMILES at index {i}: {smi} ({e})")

    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} invalid SMILES in training data")

    return valid_smiles, valid_indices


def load_data(data_path: Path, properties: List[str]) -> tuple:
    """
    Load training data from CSV with validation.

    Returns:
        Tuple of (smiles, properties_dict)

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If required columns are missing or data is invalid
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data file not found: {data_path}\n"
            f"Please provide a valid CSV file with SMILES and property columns"
        )

    logger.info(f"Loading data from {data_path}")

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    if df.empty:
        raise ValueError(f"Data file is empty: {data_path}")

    # Check required columns
    smiles_col = None
    if "SMILES" in df.columns:
        smiles_col = "SMILES"
    elif "smiles" in df.columns:
        smiles_col = "smiles"
    else:
        raise ValueError(
            f"Data must contain 'SMILES' or 'smiles' column.\n"
            f"Found columns: {', '.join(df.columns)}"
        )

    missing_props = []
    for prop in properties:
        if prop not in df.columns:
            missing_props.append(prop)

    if missing_props:
        raise ValueError(
            f"Properties not found in data: {', '.join(missing_props)}\n"
            f"Available columns: {', '.join(df.columns)}"
        )

    # Check for numeric property values
    for prop in properties:
        if not pd.api.types.is_numeric_dtype(df[prop]):
            try:
                df[prop] = pd.to_numeric(df[prop], errors='coerce')
                logger.warning(f"Converted property '{prop}' to numeric (NaN for invalid values)")
            except Exception:
                raise ValueError(
                    f"Property '{prop}' contains non-numeric values and cannot be converted"
                )

    # Remove rows with missing values
    n_initial = len(df)
    df = df.dropna(subset=[smiles_col] + properties)
    n_final = len(df)

    if n_final < n_initial:
        logger.warning(f"Removed {n_initial - n_final} rows with missing values")

    if n_final == 0:
        raise ValueError(
            f"No valid data remaining after removing rows with missing values.\n"
            f"Original data had {n_initial} rows."
        )

    # Minimum data requirement
    if n_final < 10:
        logger.warning(
            f"Very small dataset: only {n_final} molecules. "
            f"Model performance may be poor. Consider collecting more data."
        )

    # Validate SMILES
    raw_smiles = df[smiles_col].tolist()
    logger.info("Validating SMILES...")
    valid_smiles, valid_indices = validate_training_smiles(raw_smiles)

    if not valid_smiles:
        raise ValueError(
            f"No valid SMILES found in data.\n"
            f"All {len(raw_smiles)} SMILES strings were invalid."
        )

    if len(valid_smiles) < len(raw_smiles):
        logger.warning(
            f"Removed {len(raw_smiles) - len(valid_smiles)} molecules with invalid SMILES"
        )

        # Filter dataframe to only valid SMILES
        df = df.iloc[valid_indices].reset_index(drop=True)

    smiles = df[smiles_col].tolist()
    properties_dict = {prop: df[prop].values for prop in properties}

    logger.info(f"Loaded {len(smiles)} valid molecules with {len(properties)} properties")

    # Log property statistics
    for prop in properties:
        values = properties_dict[prop]
        logger.info(
            f"  {prop}: mean={np.mean(values):.3f}, "
            f"std={np.std(values):.3f}, "
            f"range=[{np.min(values):.3f}, {np.max(values):.3f}]"
        )

    return smiles, properties_dict


def train_model(args: argparse.Namespace) -> None:
    """Main training function with comprehensive error handling."""
    # Create output directory first to ensure we can write logs
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        logger.error(f"Permission denied: Cannot create output directory {args.output_dir}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        sys.exit(1)

    # Setup logging
    log_file = args.output_dir / "training.log"
    setup_logger(log_level=args.log_level, log_file=log_file)

    # Load config if provided
    if args.config:
        if not args.config.exists():
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)

        try:
            config = ADMETModelConfig.from_yaml(args.config)
            properties = config.properties
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load config from {args.config}: {e}")
            sys.exit(1)
    else:
        properties = [p.strip() for p in args.properties.split(",")]
        if not properties or any(not p for p in properties):
            logger.error("Invalid property list. Please provide comma-separated property names.")
            sys.exit(1)

        config = ADMETModelConfig(
            properties=properties,
            model_type=args.model_type,
            fingerprint_type=args.fingerprint_type,
            fingerprint_radius=args.fingerprint_radius,
            fingerprint_bits=args.fingerprint_bits,
            use_descriptors=args.use_descriptors,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
        )

    log_experiment_start("ADMET Model Training", config.to_dict())

    # Load data with error handling
    try:
        smiles, properties_dict = load_data(args.data, properties)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    # Create featurizer with error handling
    logger.info("Initializing molecular featurizer...")
    try:
        featurizer = MolecularFeaturizer(
            fingerprint_type=config.fingerprint_type,
            radius=config.fingerprint_radius,
            n_bits=config.fingerprint_bits,
            include_descriptors=config.use_descriptors,
        )
    except Exception as e:
        logger.error(f"Failed to initialize featurizer: {e}")
        logger.info(
            f"Check that fingerprint_type '{config.fingerprint_type}' is valid. "
            f"Valid options: morgan, maccs, rdkit, avalon"
        )
        sys.exit(1)

    # Create and train model with error handling
    logger.info("Initializing ADMET predictor")
    try:
        predictor = ADMETPredictor(
            properties=properties,
            model_type=config.model_type,
            featurizer=featurizer,
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
        )
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        logger.info(
            f"Check that model_type '{config.model_type}' is valid. "
            f"Valid options: random_forest, xgboost, gradient_boosting"
        )
        sys.exit(1)

    logger.info("Training models...")
    logger.info(f"  Properties: {', '.join(properties)}")
    logger.info(f"  Model type: {config.model_type}")
    logger.info(f"  Training set size: {len(smiles)}")
    logger.info(f"  Validation split: {config.test_size:.1%}")

    try:
        metrics = predictor.fit(
            smiles=smiles,
            properties_dict=properties_dict,
            validation_split=config.test_size,
        )
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        logger.info(
            "This could be due to:\n"
            "  - Insufficient data for training\n"
            "  - Invalid molecular structures\n"
            "  - Incompatible model hyperparameters\n"
            "  - Missing dependencies (check that XGBoost is installed if using xgboost)"
        )
        sys.exit(1)

    logger.success("Training completed successfully!")

    # Save model with error handling
    model_path = args.output_dir / "admet_model.pkl"
    try:
        predictor.save(model_path)
        logger.info(f"Model saved to: {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        logger.warning("Training succeeded but model could not be saved")

    # Save metrics
    try:
        metrics_df = pd.DataFrame(metrics).T
        metrics_path = args.output_dir / "metrics.csv"
        metrics_df.to_csv(metrics_path)
        logger.info(f"Metrics saved to: {metrics_path}")

        # Display metrics summary
        logger.info("\nTraining metrics summary:")
        for prop in properties:
            if prop in metrics:
                logger.info(f"  {prop}:")
                for metric_name, value in metrics[prop].items():
                    if isinstance(value, (int, float)):
                        logger.info(f"    {metric_name}: {value:.4f}")
    except Exception as e:
        logger.warning(f"Failed to save metrics: {e}")

    # Save config
    try:
        config_path = args.output_dir / "config.yaml"
        from ml_fragment_optimizer.utils.config_loader import save_config
        save_config(config.to_dict(), config_path)
        logger.info(f"Config saved to: {config_path}")
    except Exception as e:
        logger.warning(f"Failed to save config: {e}")

    log_experiment_end("ADMET Model Training", {"metrics": metrics})


def main() -> None:
    """Entry point for CLI."""
    args = parse_args()
    try:
        train_model(args)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
