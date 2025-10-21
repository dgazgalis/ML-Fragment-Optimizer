#!/usr/bin/env python
"""
Plan synthesis routes for fragments using retrosynthesis analysis.

Analyzes target molecules to find synthetic routes, scores them by
feasibility (SA score, building block availability, convergence),
and ranks by overall makeability.

Example usage:
    # Basic retrosynthesis
    mlfrag-synthesis --smiles "c1ccc(CCO)cc1" --output routes.json

    # With building blocks database and detailed scoring
    mlfrag-synthesis --smiles "CC(=O)Oc1ccccc1C(=O)O" \
                     --max-routes 10 --format detailed \
                     --building-blocks building_blocks.csv

    # Batch mode for multiple molecules
    mlfrag-synthesis --input-file molecules.smi \
                     --output-dir synthesis_results/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict
from loguru import logger

from ml_fragment_optimizer.synthesis.retrosynthesis import RetrosynthesisAnalyzer
from ml_fragment_optimizer.synthesis.sa_score import SAScoreCalculator
from ml_fragment_optimizer.synthesis.route_scoring import RouteScorer
from ml_fragment_optimizer.synthesis.building_blocks import BuildingBlockChecker
from ml_fragment_optimizer.utils.logging_utils import setup_logger

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plan synthesis routes for target molecules using retrosynthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single molecule
  %(prog)s --smiles "c1ccc(CCO)cc1"

  # With building blocks catalog
  %(prog)s --smiles "CC(=O)Oc1ccccc1C(=O)O" \\
           --building-blocks building_blocks.csv \\
           --max-routes 10

  # Batch mode
  %(prog)s --input-file molecules.smi \\
           --output-dir synthesis_results/ \\
           --format detailed

  # Use AiZynthFinder (if installed)
  %(prog)s --smiles "CCO" --use-aizynthfinder \\
           --aizynthfinder-config config.yml
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--smiles",
        type=str,
        help="Target molecule SMILES (single molecule mode)",
    )
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="File with SMILES (one per line, batch mode)",
    )

    parser.add_argument(
        "--max-routes",
        type=int,
        default=5,
        help="Maximum number of routes to return per molecule (default: 5)",
    )
    parser.add_argument(
        "--use-aizynthfinder",
        action="store_true",
        help="Use AiZynthFinder for ML-based retrosynthesis (if installed)",
    )
    parser.add_argument(
        "--aizynthfinder-config",
        type=Path,
        help="AiZynthFinder configuration file",
    )
    parser.add_argument(
        "--building-blocks",
        type=Path,
        help="Path to building blocks database (CSV with SMILES column)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synthesis_routes.json"),
        help="Output JSON file (default: synthesis_routes.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for batch mode (default: current directory)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["simple", "detailed", "json"],
        default="detailed",
        help="Output format (default: detailed)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def analyze_molecule(
    smiles: str,
    analyzer: RetrosynthesisAnalyzer,
    scorer: RouteScorer,
    max_routes: int
) -> Dict:
    """
    Analyze a single molecule for synthesis routes.

    Returns dictionary with routes and scores.
    """
    if not RDKIT_AVAILABLE:
        logger.error("RDKit is required")
        return {"smiles": smiles, "error": "RDKit not available"}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"Invalid SMILES: {smiles}")
        return {"smiles": smiles, "error": "Invalid SMILES"}

    # Analyze routes
    try:
        routes = analyzer.analyze(smiles, max_routes=max_routes)
    except Exception as e:
        logger.error(f"Retrosynthesis failed for {smiles}: {e}")
        return {"smiles": smiles, "error": str(e)}

    if not routes:
        logger.info(f"No routes found for {smiles}")
        return {
            "smiles": smiles,
            "routes": [],
            "synthesizable": False,
            "message": "No synthesis routes found"
        }

    # Score routes
    try:
        scored_routes = scorer.rank_routes(routes)
    except Exception as e:
        logger.warning(f"Route scoring failed: {e}, using unscored routes")
        scored_routes = routes

    # Format results
    results = {
        "smiles": smiles,
        "synthesizable": True,
        "num_routes": len(scored_routes),
        "routes": []
    }

    for idx, route in enumerate(scored_routes[:max_routes]):
        route_info = {
            "rank": idx + 1,
            "steps": route.num_steps if hasattr(route, 'num_steps') else len(route.reactions) if hasattr(route, 'reactions') else 0,
            "type": route.route_type.value if hasattr(route, 'route_type') else "unknown",
        }

        # Add scoring if available
        if hasattr(route, 'overall_score'):
            route_info["overall_score"] = round(float(route.overall_score), 3)
        if hasattr(route, 'step_score'):
            route_info["step_score"] = round(float(route.step_score), 3)
        if hasattr(route, 'availability_score'):
            route_info["availability_score"] = round(float(route.availability_score), 3)
        if hasattr(route, 'sa_score'):
            route_info["sa_score"] = round(float(route.sa_score), 3)

        # Add recommendations if available
        if hasattr(route, 'get_recommendations'):
            route_info["recommendations"] = route.get_recommendations()

        results["routes"].append(route_info)

    # Best route summary
    if scored_routes:
        results["best_route"] = {
            "steps": route_info["steps"],
            "score": route_info.get("overall_score", 0.0),
            "synthesizable": route_info.get("overall_score", 0.0) > 0.5,
        }

    return results


def main() -> None:
    """Entry point for CLI."""
    args = parse_args()
    setup_logger(log_level=args.log_level)

    if not RDKIT_AVAILABLE:
        logger.error("RDKit is required but not installed")
        logger.info("Install with: pip install rdkit")
        sys.exit(1)

    logger.info("Synthesis Route Planning Tool")

    # Initialize components
    logger.info("Initializing retrosynthesis analyzer...")
    analyzer = RetrosynthesisAnalyzer(use_aizynthfinder=args.use_aizynthfinder)

    # Initialize building blocks checker
    bb_checker = None
    if args.building_blocks and args.building_blocks.exists():
        logger.info(f"Loading building blocks from {args.building_blocks}")
        bb_checker = BuildingBlockChecker(catalog_path=str(args.building_blocks))
    else:
        logger.info("No building blocks database provided, using default availability")

    # Initialize scorers
    sa_calculator = SAScoreCalculator()
    scorer = RouteScorer(bb_checker=bb_checker, sa_calculator=sa_calculator)

    # Process molecules
    results_list = []

    if args.smiles:
        # Single molecule mode
        logger.info(f"Analyzing: {args.smiles}")
        results = analyze_molecule(args.smiles, analyzer, scorer, args.max_routes)
        results_list.append(results)

    elif args.input_file:
        # Batch mode
        if not args.input_file.exists():
            logger.error(f"Input file not found: {args.input_file}")
            sys.exit(1)

        logger.info(f"Reading molecules from {args.input_file}")
        with open(args.input_file) as f:
            smiles_list = [line.strip() for line in f if line.strip()]

        logger.info(f"Processing {len(smiles_list)} molecules...")
        for idx, smiles in enumerate(smiles_list, 1):
            logger.info(f"[{idx}/{len(smiles_list)}] {smiles}")
            results = analyze_molecule(smiles, analyzer, scorer, args.max_routes)
            results_list.append(results)

    # Save results
    output_path = args.output if args.smiles else (args.output_dir / "synthesis_results.json" if args.output_dir else Path("synthesis_results.json"))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results_list, f, indent=2)

    # Display summary
    logger.info("\n" + "="*60)
    logger.info("SYNTHESIS ROUTE SUMMARY")
    logger.info("="*60)

    for result in results_list:
        logger.info(f"\nMolecule: {result['smiles']}")
        if "error" in result:
            logger.error(f"  Error: {result['error']}")
            continue

        if not result.get('synthesizable', False):
            logger.warning(f"  Not synthesizable: {result.get('message', 'Unknown reason')}")
            continue

        logger.info(f"  Routes found: {result['num_routes']}")
        if "best_route" in result:
            best = result["best_route"]
            logger.info(f"  Best route: {best['steps']} steps (score: {best['score']:.3f})")

        if args.format == "detailed" and result.get("routes"):
            for route in result["routes"][:3]:  # Show top 3
                logger.info(f"\n  Route {route['rank']}:")
                logger.info(f"    Steps: {route['steps']}")
                if "overall_score" in route:
                    logger.info(f"    Overall score: {route['overall_score']:.3f}")
                if "sa_score" in route:
                    logger.info(f"    SA score: {route['sa_score']:.2f}")
                if "recommendations" in route:
                    for rec in route["recommendations"][:2]:
                        logger.info(f"    - {rec}")

    logger.success(f"\n✓ Analysis complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()
