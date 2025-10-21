#!/usr/bin/env python
"""
Batch prediction of ADMET properties for molecules.

Example usage:
    mlfrag-predict --model models/admet_v1/admet_model.pkl \\
                   --input molecules.smi --output predictions.csv

    mlfrag-predict --model models/admet_v1/admet_model.pkl \\
                   --input molecules.sdf --output predictions.csv \\
                   --uncertainty --flag-outliers
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from loguru import logger

from ml_fragment_optimizer.models.admet_predictor import ADMETPredictor
from ml_fragment_optimizer.utils.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict ADMET properties for molecules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction from SMILES file
  %(prog)s --model models/admet_model.pkl --input molecules.smi \\
           --output predictions.csv

  # Predict with uncertainty estimates
  %(prog)s --model models/admet_model.pkl --input molecules.smi \\
           --output predictions.csv --uncertainty

  # Flag molecules outside applicability domain
  %(prog)s --model models/admet_model.pkl --input molecules.sdf \\
           --output predictions.csv --flag-outliers

Input formats:
  - SMILES file (.smi, .txt): One SMILES per line
  - SDF file (.sdf): Multi-molecule SDF
  - CSV file (.csv): Must contain 'SMILES' column
        """,
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained ADMET model (.pkl file)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input file with molecules (SMILES, SDF, or CSV)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file for predictions",
    )
    parser.add_argument(
        "--uncertainty",
        action="store_true",
        help="Include uncertainty estimates in output",
    )
    parser.add_argument(
        "--flag-outliers",
        action="store_true",
        help="Flag molecules outside model applicability domain",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for prediction (default: 1000)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def validate_smiles(smiles_list: list[str]) -> tuple[list[str], list[int], list[str]]:
    """
    Validate SMILES strings and return valid ones with their indices.

    Returns:
        Tuple of (valid_smiles, valid_indices, invalid_smiles)
    """
    valid_smiles = []
    valid_indices = []
    invalid_smiles = []

    for i, smi in enumerate(smiles_list):
        if not smi or not smi.strip():
            invalid_smiles.append(f"Empty SMILES at index {i}")
            continue

        try:
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is None:
                invalid_smiles.append(f"Invalid SMILES at index {i}: {smi}")
            else:
                valid_smiles.append(smi.strip())
                valid_indices.append(i)
        except Exception as e:
            invalid_smiles.append(f"Error parsing SMILES at index {i}: {smi} ({e})")

    return valid_smiles, valid_indices, invalid_smiles


def load_molecules(input_path: Path) -> tuple:
    """
    Load molecules from file with validation.

    Returns:
        Tuple of (smiles_list, molecule_ids, valid_indices)

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If file format is unsupported or contains no valid molecules
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    raw_smiles = []
    raw_ids = []

    try:
        if suffix in [".smi", ".txt"]:
            # SMILES file
            with open(input_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):  # Skip empty lines and comments
                        continue
                    parts = line.split()
                    if parts:
                        raw_smiles.append(parts[0])
                        mol_id = parts[1] if len(parts) > 1 else f"mol_{i}"
                        raw_ids.append(mol_id)

        elif suffix == ".sdf":
            # SDF file
            try:
                suppl = Chem.SDMolSupplier(str(input_path), removeHs=False, sanitize=True)
            except Exception as e:
                raise ValueError(f"Failed to read SDF file: {e}")

            for i, mol in enumerate(suppl, 1):
                if mol is not None:
                    try:
                        smi = Chem.MolToSmiles(mol)
                        raw_smiles.append(smi)
                        mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}"
                        raw_ids.append(mol_id)
                    except Exception as e:
                        logger.warning(f"Failed to convert molecule {i} to SMILES: {e}")

        elif suffix == ".csv":
            # CSV file
            try:
                df = pd.read_csv(input_path)
            except Exception as e:
                raise ValueError(f"Failed to read CSV file: {e}")

            if "SMILES" not in df.columns and "smiles" not in df.columns:
                raise ValueError(
                    f"CSV must contain 'SMILES' or 'smiles' column. "
                    f"Found columns: {', '.join(df.columns)}"
                )

            smiles_col = "SMILES" if "SMILES" in df.columns else "smiles"
            raw_smiles = df[smiles_col].fillna("").tolist()

            if "ID" in df.columns:
                raw_ids = df["ID"].tolist()
            elif "id" in df.columns:
                raw_ids = df["id"].tolist()
            else:
                raw_ids = [f"mol_{i+1}" for i in range(len(df))]

        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: .smi, .txt, .sdf, .csv"
            )

    except UnicodeDecodeError:
        raise ValueError(f"Failed to read file {input_path}: encoding error. Try UTF-8 encoding.")

    if not raw_smiles:
        raise ValueError(f"No molecules found in {input_path}")

    # Validate SMILES
    logger.info(f"Loaded {len(raw_smiles)} molecules from {input_path}")
    logger.info("Validating SMILES...")

    valid_smiles, valid_indices, invalid_smiles = validate_smiles(raw_smiles)

    # Report validation results
    if invalid_smiles:
        logger.warning(f"Found {len(invalid_smiles)} invalid SMILES:")
        for msg in invalid_smiles[:5]:  # Show first 5
            logger.warning(f"  {msg}")
        if len(invalid_smiles) > 5:
            logger.warning(f"  ... and {len(invalid_smiles) - 5} more")

    if not valid_smiles:
        raise ValueError(
            f"No valid molecules found in {input_path}. "
            f"All {len(raw_smiles)} SMILES strings were invalid."
        )

    # Filter IDs to match valid SMILES
    valid_ids = [raw_ids[i] for i in valid_indices]

    logger.info(f"Validated: {len(valid_smiles)}/{len(raw_smiles)} molecules are valid")

    return valid_smiles, valid_ids, valid_indices


def detect_outliers(
    predictor: ADMETPredictor,
    smiles: list,
    threshold: float = 3.0,
) -> np.ndarray:
    """
    Detect outliers based on feature space distance.

    Args:
        predictor: Trained predictor
        smiles: List of SMILES
        threshold: Z-score threshold for outlier detection

    Returns:
        Boolean array indicating outliers
    """
    # Featurize molecules
    X = predictor.featurizer.featurize(smiles)

    # Calculate distance to training set mean
    # (simplified outlier detection)
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)

    z_scores = np.abs((mean - np.mean(mean)) / (std + 1e-10))
    is_outlier = z_scores > threshold

    return is_outlier


def predict_batch(args: argparse.Namespace) -> None:
    """Main prediction function."""
    setup_logger(log_level=args.log_level)

    # Validate inputs
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model file not found: {args.model}\n"
            f"Please provide a valid path to a trained model (.pkl file)"
        )

    # Load model
    logger.info(f"Loading model from {args.model}")
    try:
        predictor = ADMETPredictor.load(args.model)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from {args.model}\n"
            f"Error: {e}\n"
            f"Make sure the file is a valid ADMET model saved with ADMETPredictor.save()"
        )

    logger.info(f"Model predicts: {predictor.properties}")

    # Load molecules
    try:
        smiles, mol_ids, _ = load_molecules(args.input)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    # Prepare results dataframe
    results = pd.DataFrame({
        "ID": mol_ids,
        "SMILES": smiles,
    })

    # Predict in batches
    logger.info("Running predictions...")
    all_predictions = {prop: [] for prop in predictor.properties}
    all_uncertainties = {prop: [] for prop in predictor.properties} if args.uncertainty else None
    failed_predictions = []

    for i in tqdm(range(0, len(smiles), args.batch_size), desc="Predicting"):
        batch_smiles = smiles[i:i + args.batch_size]

        try:
            if args.uncertainty:
                preds, uncs = predictor.predict(batch_smiles, return_uncertainty=True)
                for prop in predictor.properties:
                    all_predictions[prop].extend(preds[prop])
                    all_uncertainties[prop].extend(uncs[prop])
            else:
                preds = predictor.predict(batch_smiles, return_uncertainty=False)
                for prop in predictor.properties:
                    all_predictions[prop].extend(preds[prop])

        except Exception as e:
            logger.warning(
                f"Batch prediction failed for molecules {i}-{i+len(batch_smiles)-1}: {e}\n"
                f"Attempting individual predictions for this batch..."
            )

            # Try individual predictions for failed batch
            for j, smi in enumerate(batch_smiles):
                try:
                    if args.uncertainty:
                        preds, uncs = predictor.predict([smi], return_uncertainty=True)
                        for prop in predictor.properties:
                            all_predictions[prop].append(preds[prop][0])
                            all_uncertainties[prop].append(uncs[prop][0])
                    else:
                        preds = predictor.predict([smi], return_uncertainty=False)
                        for prop in predictor.properties:
                            all_predictions[prop].append(preds[prop][0])

                except Exception as e_single:
                    # Record failure and add NaN
                    failed_predictions.append((i + j, smi, str(e_single)))
                    for prop in predictor.properties:
                        all_predictions[prop].append(np.nan)
                        if args.uncertainty:
                            all_uncertainties[prop].append(np.nan)

    # Report failed predictions
    if failed_predictions:
        logger.warning(
            f"\nFailed to predict {len(failed_predictions)} molecules:"
        )
        for idx, smi, error in failed_predictions[:5]:
            logger.warning(f"  Index {idx}: {smi} - {error}")
        if len(failed_predictions) > 5:
            logger.warning(f"  ... and {len(failed_predictions) - 5} more")
        logger.warning(f"These predictions are marked as NaN in the output.")

    # Add predictions to results
    for prop in predictor.properties:
        results[f"{prop}_predicted"] = all_predictions[prop]

        if args.uncertainty:
            results[f"{prop}_uncertainty"] = all_uncertainties[prop]

    # Flag outliers if requested
    if args.flag_outliers:
        logger.info("Detecting outliers...")
        is_outlier = detect_outliers(predictor, smiles)
        results["is_outlier"] = is_outlier
        n_outliers = is_outlier.sum()
        logger.warning(f"Flagged {n_outliers} molecules as outliers ({100*n_outliers/len(smiles):.1f}%)")

    # Save results
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        logger.info(f"Saved predictions to {args.output}")
    except PermissionError:
        logger.error(f"Permission denied: Cannot write to {args.output}")
        logger.info("Try specifying a different output path with --output")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        sys.exit(1)

    # Print summary statistics
    logger.info("\nPrediction summary:")
    for prop in predictor.properties:
        values = results[f"{prop}_predicted"]
        logger.info(f"  {prop}:")
        logger.info(f"    Mean: {values.mean():.3f}")
        logger.info(f"    Std:  {values.std():.3f}")
        logger.info(f"    Min:  {values.min():.3f}")
        logger.info(f"    Max:  {values.max():.3f}")


def main() -> None:
    """Entry point for CLI."""
    args = parse_args()
    try:
        predict_batch(args)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()
