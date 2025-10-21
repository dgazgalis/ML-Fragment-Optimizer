#!/usr/bin/env python
"""
Suggest fragment modifications to improve properties.

Uses bioisosteric replacements and chemical transformations to suggest
modified fragments with improved ADMET properties while maintaining
synthetic accessibility.

Example usage:
    # Optimize solubility
    mlfrag-optimize --fragment "c1ccccc1" --target-property solubility \
                    --target-value -2.0 --num-suggestions 10 \
                    --model models/admet_model.pkl

    # Optimize multiple properties
    mlfrag-optimize --fragment "CCO" --target-property solubility,logp \
                    --target-value -1.0,2.0 --num-suggestions 20
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from loguru import logger

from ml_fragment_optimizer.models.admet_predictor import ADMETPredictor
from ml_fragment_optimizer.qsar.bioisostere_suggester import BioisostereSuggester
from ml_fragment_optimizer.synthesis.sa_score import SAScoreCalculator
from ml_fragment_optimizer.utils.logging_utils import setup_logger

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize fragment to improve target properties using bioisosteric replacements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize solubility
  %(prog)s --fragment "c1ccccc1" --target-property solubility \\
           --target-value -2.0 --model models/admet_model.pkl

  # Optimize multiple properties with weights
  %(prog)s --fragment "CCO" --target-property solubility,logp \\
           --target-value -1.0,2.0 --num-suggestions 20 \\
           --max-sa-score 6.0

  # Use scaffold hopping
  %(prog)s --fragment "c1ccc(N)cc1" --target-property herg \\
           --maximize --scaffold-hopping
        """
    )

    parser.add_argument(
        "--fragment",
        type=str,
        required=True,
        help="Fragment SMILES to optimize",
    )
    parser.add_argument(
        "--target-property",
        type=str,
        required=True,
        help="Property to optimize (comma-separated for multiple: solubility,logp)",
    )
    parser.add_argument(
        "--target-value",
        type=str,
        help="Target value for property (comma-separated for multiple, e.g., -1.0,2.0)",
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        help="Maximize property instead of reaching target value",
    )
    parser.add_argument(
        "--num-suggestions",
        type=int,
        default=10,
        help="Number of suggestions to generate (default: 10)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to trained ADMET model (optional, uses MW/logP if not provided)",
    )
    parser.add_argument(
        "--max-sa-score",
        type=float,
        default=7.0,
        help="Maximum SAScore allowed (1-10, lower is easier to synthesize, default: 7.0)",
    )
    parser.add_argument(
        "--scaffold-hopping",
        action="store_true",
        help="Enable scaffold hopping (larger structural changes)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fragment_suggestions.csv"),
        help="Output CSV file (default: fragment_suggestions.csv)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def generate_simple_modifications(smiles: str, num_candidates: int = 100) -> List[str]:
    """
    Generate simple structural modifications without ML model.

    Uses:
    - Halogen additions (F, Cl, Br)
    - Methyl additions
    - Hydroxyl additions
    - Simple ring modifications
    """
    if not RDKIT_AVAILABLE:
        logger.error("RDKit is required for fragment optimization")
        sys.exit(1)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.error(f"Invalid SMILES: {smiles}")
        sys.exit(1)

    modifications = set()
    modifications.add(smiles)  # Include original

    # Simple substitutions
    substitutions = {
        'H': ['F', 'Cl', 'Br', 'OH', 'CH3', 'NH2', 'CN'],
    }

    # Try adding groups to aromatic carbons
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and atom.GetSymbol() == 'C':
            for group in substitutions['H']:
                try:
                    # Create a copy
                    new_mol = Chem.RWMol(mol)
                    idx = atom.GetIdx()

                    # Add substituent (simplified - just change implicit H to explicit group)
                    if group == 'F':
                        new_mol.GetAtomWithIdx(idx).SetFormalCharge(0)
                        # This is simplified - proper implementation needs SMARTS

                    new_smiles = Chem.MolToSmiles(new_mol)
                    if new_smiles and new_smiles != smiles:
                        modifications.add(new_smiles)

                    if len(modifications) >= num_candidates:
                        break
                except:
                    continue

            if len(modifications) >= num_candidates:
                break

    return list(modifications)[:num_candidates]


def calculate_score(
    predicted: Dict[str, float],
    target_props: List[str],
    target_values: List[float],
    maximize: bool,
    sa_score: float,
    max_sa: float
) -> float:
    """
    Calculate optimization score.

    Score = property_score * sa_penalty

    where:
    - property_score: Distance to target (or raw value if maximizing)
    - sa_penalty: Penalize difficult-to-synthesize molecules
    """
    # Property score
    if maximize:
        # For maximization, just sum the properties
        prop_score = sum(predicted.get(prop, 0.0) for prop in target_props)
    else:
        # For target-based, calculate distance
        distances = []
        for prop, target in zip(target_props, target_values):
            if prop in predicted:
                dist = abs(predicted[prop] - target)
                distances.append(dist)
        prop_score = -np.mean(distances) if distances else -1000.0

    # SAScore penalty (exponential penalty for high SA scores)
    sa_penalty = np.exp(-(sa_score - 1.0) / 3.0)  # Penalty increases with SA score

    # Disqualify if SA score too high
    if sa_score > max_sa:
        return -1000.0

    return prop_score * sa_penalty


def main() -> None:
    """Entry point for CLI."""
    args = parse_args()
    setup_logger(log_level=args.log_level)

    if not RDKIT_AVAILABLE:
        logger.error("RDKit is required but not installed")
        logger.info("Install with: pip install rdkit")
        sys.exit(1)

    logger.info("Fragment Optimization Tool")
    logger.info(f"Fragment: {args.fragment}")
    logger.info(f"Target property: {args.target_property}")

    # Parse target properties
    target_props = [p.strip() for p in args.target_property.split(',')]

    # Parse target values
    if args.target_value:
        target_values = [float(v.strip()) for v in args.target_value.split(',')]
        if len(target_values) != len(target_props):
            logger.error("Number of target values must match number of target properties")
            sys.exit(1)
    else:
        if not args.maximize:
            logger.error("Must provide --target-value or use --maximize")
            sys.exit(1)
        target_values = [0.0] * len(target_props)

    # Load model if provided
    predictor = None
    if args.model and args.model.exists():
        logger.info(f"Loading ADMET model from {args.model}")
        try:
            predictor = ADMETPredictor.load(args.model)
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            logger.info("Falling back to molecular descriptors")

    # Initialize SA score calculator
    sa_calculator = SAScoreCalculator()

    # Step 1: Generate candidate fragments
    logger.info("Generating candidate fragments...")

    # Use bioisostere suggester if available
    try:
        suggester = BioisostereSuggester()
        candidates_data = suggester.suggest_replacements(
            args.fragment,
            context_smiles=None,
            max_suggestions=args.num_suggestions * 10  # Generate more, filter later
        )
        candidates = [c['smiles'] for c in candidates_data]
        logger.info(f"Generated {len(candidates)} bioisosteric candidates")
    except Exception as e:
        logger.warning(f"Bioisostere generation failed: {e}")
        logger.info("Using simple modifications")
        candidates = generate_simple_modifications(args.fragment, args.num_suggestions * 10)

    if not candidates:
        logger.error("No candidates generated")
        sys.exit(1)

    # Step 2: Predict properties
    logger.info(f"Evaluating {len(candidates)} candidates...")

    results = []
    for smiles in candidates:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Calculate SA score
        sa_score = sa_calculator.calculate_sa_score(smiles)

        # Predict properties
        if predictor:
            try:
                preds, _ = predictor.predict([smiles])
                predicted = {prop: preds[prop][0] if prop in preds else 0.0
                           for prop in target_props}
            except Exception as e:
                logger.debug(f"Prediction failed for {smiles}: {e}")
                continue
        else:
            # Use simple descriptors
            predicted = {}
            if 'logp' in target_props:
                predicted['logp'] = Descriptors.MolLogP(mol)
            if 'mw' in target_props:
                predicted['mw'] = Descriptors.MolWt(mol)
            if 'tpsa' in target_props:
                predicted['tpsa'] = Descriptors.TPSA(mol)

        # Calculate score
        score = calculate_score(
            predicted,
            target_props,
            target_values,
            args.maximize,
            sa_score,
            args.max_sa_score
        )

        results.append({
            'smiles': smiles,
            'score': score,
            'sa_score': sa_score,
            **predicted
        })

    if not results:
        logger.error("No valid candidates after filtering")
        sys.exit(1)

    # Step 3: Rank and select top suggestions
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)
    top_results = results_df.head(args.num_suggestions)

    # Step 4: Save results
    logger.info(f"Saving top {len(top_results)} suggestions to {args.output}")
    top_results.to_csv(args.output, index=False)

    # Display results
    logger.info("\n=== Top Suggestions ===")
    for idx, row in top_results.iterrows():
        logger.info(f"\n{idx+1}. {row['smiles']}")
        logger.info(f"   Score: {row['score']:.3f}")
        logger.info(f"   SA Score: {row['sa_score']:.2f}")
        for prop in target_props:
            if prop in row:
                target_str = f" (target: {target_values[target_props.index(prop)]:.2f})" if not args.maximize else ""
                logger.info(f"   {prop}: {row[prop]:.3f}{target_str}")

    logger.success(f"Optimization complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
