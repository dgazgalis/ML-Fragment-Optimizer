"""
Example Inference Script for ADMET Predictor

This script demonstrates how to use a trained ADMET model
to make predictions on new molecules.

Author: Claude Code
Date: 2025-10-20
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.models import ADMETPredictor, ADMETConfig


def predict_admet_properties(
    smiles_list: list,
    model_path: Path,
    output_csv: Path = None
):
    """
    Predict ADMET properties for a list of SMILES.

    Args:
        smiles_list: List of SMILES strings
        model_path: Path to trained model
        output_csv: Optional path to save predictions
    """
    print(f"Loading model from {model_path}...")

    # Initialize predictor
    predictor = ADMETPredictor(ADMETConfig())

    # Load trained model
    predictor.load_model(model_path)

    print(f"\nPredicting ADMET properties for {len(smiles_list)} molecules...")

    # Make predictions
    predictions, uncertainties = predictor.predict(
        smiles_list,
        return_uncertainty=False
    )

    # Create results dataframe
    results = pd.DataFrame({'smiles': smiles_list})

    for task, values in predictions.items():
        results[task] = values.flatten()

    # Display results
    print("\n" + "=" * 80)
    print("ADMET Predictions")
    print("=" * 80)
    print(results.to_string(index=False))
    print("=" * 80)

    # Save to CSV if requested
    if output_csv is not None:
        results.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")

    return results


def interpret_predictions(results: pd.DataFrame):
    """
    Interpret ADMET predictions and flag potential issues.

    Args:
        results: DataFrame with ADMET predictions
    """
    print("\n" + "=" * 80)
    print("ADMET Interpretation")
    print("=" * 80)

    for idx, row in results.iterrows():
        smiles = row['smiles']
        print(f"\nMolecule: {smiles}")
        print("-" * 40)

        # Solubility
        if 'solubility' in row:
            sol = row['solubility']
            if sol < -6:
                flag = "⚠ POOR"
            elif sol < -4:
                flag = "○ MODERATE"
            else:
                flag = "✓ GOOD"
            print(f"  Solubility (LogS):     {sol:.2f}  {flag}")

        # Permeability
        if 'permeability' in row:
            perm = row['permeability']
            if perm < -6.5:
                flag = "⚠ LOW"
            elif perm < -5.5:
                flag = "○ MODERATE"
            else:
                flag = "✓ HIGH"
            print(f"  Permeability (Caco-2): {perm:.2f}  {flag}")

        # CYP3A4 inhibition
        if 'cyp3a4' in row:
            cyp = row['cyp3a4']
            flag = "⚠ INHIBITOR" if cyp > 0.5 else "✓ NON-INHIBITOR"
            print(f"  CYP3A4 Inhibition:     {cyp:.2f}  {flag}")

        # hERG liability
        if 'herg' in row:
            herg = row['herg']
            flag = "⚠ RISK" if herg > 0.5 else "✓ SAFE"
            print(f"  hERG Liability:        {herg:.2f}  {flag}")

        # LogD
        if 'logd' in row:
            logd = row['logd']
            if logd < 0 or logd > 5:
                flag = "⚠ SUBOPTIMAL"
            else:
                flag = "✓ OPTIMAL"
            print(f"  LogD (pH 7.4):         {logd:.2f}  {flag}")

        # pKa
        if 'pka' in row:
            pka = row['pka']
            print(f"  pKa (most acidic):     {pka:.2f}")

    print("=" * 80)


def main():
    """Main inference pipeline"""
    print("=" * 80)
    print("ADMET Predictor Inference Example")
    print("=" * 80)

    # Example molecules
    example_molecules = [
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),
        ("Atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O"),
    ]

    print("\nExample molecules:")
    for name, smiles in example_molecules:
        print(f"  {name}: {smiles}")

    smiles_list = [smi for _, smi in example_molecules]

    # Path to trained model (adjust as needed)
    model_path = Path("outputs/best_model.pt")

    if not model_path.exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please run example_training.py first to train a model.")
        return

    # Make predictions
    results = predict_admet_properties(
        smiles_list,
        model_path,
        output_csv=Path("outputs/predictions.csv")
    )

    # Interpret predictions
    interpret_predictions(results)

    print("\n" + "=" * 80)
    print("Inference completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
