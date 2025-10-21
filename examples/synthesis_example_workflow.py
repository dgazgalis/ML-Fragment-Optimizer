#!/usr/bin/env python3
"""
Complete Synthesis Analysis Workflow Example

Demonstrates the complete workflow for evaluating synthetic feasibility:
1. Generate retrosynthetic routes
2. Calculate SAScores for intermediates
3. Check building block availability
4. Score and rank routes by makeability
5. Export results

Author: ML-Fragment-Optimizer Team
"""

import logging
from pathlib import Path
import json

# Import synthesis module components
from retrosynthesis import RetrosynthesisAnalyzer
from sa_score import SAScoreCalculator
from building_blocks import BuildingBlockChecker, create_example_catalog
from route_scoring import RouteScorer


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def initialize_components(catalog_path: Path):
    """
    Initialize all synthesis analysis components

    Args:
        catalog_path: Path to building block catalog CSV

    Returns:
        Tuple of (analyzer, scorer, bb_checker, sa_calc)
    """
    logger = logging.getLogger(__name__)

    # Create example catalog if it doesn't exist
    if not catalog_path.exists():
        logger.info(f"Creating example catalog at {catalog_path}")
        create_example_catalog(catalog_path)

    # Initialize building block checker
    logger.info("Initializing building block checker...")
    bb_checker = BuildingBlockChecker(
        local_catalog_path=catalog_path,
        use_zinc=False,  # Disable for offline use
        similarity_threshold=0.80
    )

    # Initialize SAScore calculator
    logger.info("Initializing SAScore calculator...")
    sa_calc = SAScoreCalculator()

    # Initialize retrosynthesis analyzer
    logger.info("Initializing retrosynthesis analyzer...")
    analyzer = RetrosynthesisAnalyzer(
        use_aizynthfinder=False,  # Use template-based for this example
        max_routes=5,
        max_steps=8
    )

    # Initialize route scorer with custom weights
    custom_weights = {
        'steps': 0.25,
        'convergence': 0.15,
        'availability': 0.30,
        'sa_score': 0.25,
        'cost': 0.05
    }

    logger.info("Initializing route scorer...")
    scorer = RouteScorer(
        weights=custom_weights,
        building_block_checker=bb_checker,
        sa_calculator=sa_calc
    )

    logger.info("All components initialized successfully\n")

    return analyzer, scorer, bb_checker, sa_calc


def analyze_molecule(
    smiles: str,
    name: str,
    analyzer: RetrosynthesisAnalyzer,
    scorer: RouteScorer
):
    """
    Complete analysis of a single molecule

    Args:
        smiles: Target molecule SMILES
        name: Molecule name
        analyzer: RetrosynthesisAnalyzer instance
        scorer: RouteScorer instance

    Returns:
        Dictionary with analysis results
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Analyzing: {name}")
    logger.info(f"SMILES: {smiles}")
    logger.info("-" * 80)

    # Step 1: Generate retrosynthetic routes
    logger.info("Step 1: Generating retrosynthetic routes...")
    try:
        routes = analyzer.analyze(smiles, method='template')
        logger.info(f"  Generated {len(routes)} route(s)")
    except Exception as e:
        logger.error(f"  Failed to generate routes: {e}")
        return None

    if not routes:
        logger.warning("  No routes found")
        return None

    # Step 2: Score and rank routes
    logger.info("Step 2: Scoring routes...")
    try:
        scores = scorer.rank_routes(routes)
        logger.info(f"  Scored {len(scores)} route(s)")
    except Exception as e:
        logger.error(f"  Failed to score routes: {e}")
        return None

    # Step 3: Display results
    logger.info("\nResults:")
    logger.info("=" * 80)

    best_score = scores[0]
    logger.info(f"\nBest Route (Overall Score: {best_score.overall_score:.3f}):")
    logger.info(f"  {best_score.get_interpretation()}")

    logger.info("\nComponent Scores:")
    logger.info(f"  Steps:        {best_score.step_score:.3f} ({len(best_score.route.steps)} steps)")
    logger.info(f"  Convergence:  {best_score.convergence_score:.3f}")
    logger.info(f"  Availability: {best_score.availability_score:.3f} "
                f"({best_score.num_available_building_blocks}/{best_score.num_total_building_blocks} available)")
    logger.info(f"  SA Score:     {best_score.sa_score:.3f} "
                f"(avg SA: {best_score.avg_intermediate_sa_score:.2f})")

    cost_str = f"${best_score.estimated_cost_usd:.2f}" if best_score.estimated_cost_usd else "N/A"
    logger.info(f"  Cost:         {best_score.cost_score:.3f} ({cost_str})")

    # Show route details
    logger.info("\nRoute Details:")
    for step in best_score.route.steps:
        logger.info(f"  Step {step.step_number} ({step.reaction_type}):")
        logger.info(f"    Reactants: {', '.join(step.reactants)}")
        logger.info(f"    Product: {step.product}")
        logger.info(f"    Confidence: {step.confidence:.2f}")

    logger.info(f"\n  Building Blocks:")
    for bb in best_score.route.building_blocks:
        logger.info(f"    - {bb}")

    # Show recommendations
    logger.info("\nRecommendations:")
    for rec in best_score.get_recommendations():
        logger.info(f"  - {rec}")

    logger.info("\n" + "=" * 80 + "\n")

    # Return results for export
    return {
        'molecule_name': name,
        'smiles': smiles,
        'num_routes': len(routes),
        'best_route': best_score.to_dict(),
        'all_routes': [score.to_dict() for score in scores]
    }


def batch_analysis(
    molecules: list[tuple[str, str]],
    analyzer: RetrosynthesisAnalyzer,
    scorer: RouteScorer,
    output_path: Path
):
    """
    Analyze multiple molecules and export results

    Args:
        molecules: List of (SMILES, name) tuples
        analyzer: RetrosynthesisAnalyzer instance
        scorer: RouteScorer instance
        output_path: Path to save JSON results
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Starting batch analysis of {len(molecules)} molecules")
    logger.info("=" * 80 + "\n")

    results = []
    successful = 0

    for i, (smiles, name) in enumerate(molecules, 1):
        logger.info(f"[{i}/{len(molecules)}] Processing: {name}")

        result = analyze_molecule(smiles, name, analyzer, scorer)

        if result:
            results.append(result)
            successful += 1
        else:
            logger.warning(f"Skipped {name} due to errors\n")

    # Summary
    logger.info("\nBatch Analysis Summary:")
    logger.info("=" * 80)
    logger.info(f"Total molecules: {len(molecules)}")
    logger.info(f"Successfully analyzed: {successful}")
    logger.info(f"Failed: {len(molecules) - successful}")

    # Export results
    logger.info(f"\nExporting results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("Export complete!")

    # Print top results
    if results:
        logger.info("\nTop Molecules by Makeability:")
        sorted_results = sorted(
            results,
            key=lambda x: x['best_route']['overall_score'],
            reverse=True
        )

        for i, result in enumerate(sorted_results[:5], 1):
            score = result['best_route']['overall_score']
            interp = result['best_route']['interpretation']
            logger.info(f"  {i}. {result['molecule_name']}: {score:.3f} ({interp})")


def main():
    """Main workflow"""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Synthesis Analysis Workflow Example")
    logger.info("=" * 80 + "\n")

    # Configuration
    catalog_path = Path('example_building_blocks.csv')
    output_path = Path('synthesis_analysis_results.json')

    # Initialize components
    analyzer, scorer, bb_checker, sa_calc = initialize_components(catalog_path)

    # Test molecules (drug-like compounds with varying complexity)
    molecules = [
        ('CCO', 'Ethanol'),
        ('CC(C)O', 'Isopropanol'),
        ('CC(=O)O', 'Acetic Acid'),
        ('c1ccccc1', 'Benzene'),
        ('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin'),
        ('CC(C)Cc1ccc(cc1)C(C)C(O)=O', 'Ibuprofen'),
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'Caffeine'),
    ]

    # Run batch analysis
    batch_analysis(molecules, analyzer, scorer, output_path)

    logger.info("\n" + "=" * 80)
    logger.info("Workflow complete!")
    logger.info(f"Results saved to: {output_path.absolute()}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
