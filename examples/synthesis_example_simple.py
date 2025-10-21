#!/usr/bin/env python3
"""
Simple Usage Examples

Quick examples showing basic usage of each module component.

Author: ML-Fragment-Optimizer Team
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)  # Reduce verbosity


def example_sa_score():
    """Example: Calculate SAScore for molecules"""
    from sa_score import SAScoreCalculator

    print("\n" + "=" * 80)
    print("Example 1: SAScore Calculation")
    print("=" * 80)

    calculator = SAScoreCalculator()

    molecules = [
        ('CCO', 'Ethanol'),
        ('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin'),
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'Caffeine'),
    ]

    for smiles, name in molecules:
        result = calculator.calculate(smiles)
        print(f"\n{name}:")
        print(f"  SMILES: {smiles}")
        print(f"  SAScore: {result.score:.2f} ({result.get_interpretation()})")
        print(f"  Fragment Score: {result.fragment_score:.2f}")
        print(f"  Complexity: {result.complexity_penalty:.2f}")


def example_building_blocks():
    """Example: Check building block availability"""
    from building_blocks import BuildingBlockChecker, create_example_catalog

    print("\n" + "=" * 80)
    print("Example 2: Building Block Availability")
    print("=" * 80)

    # Create example catalog
    catalog_path = Path('example_bb.csv')
    if not catalog_path.exists():
        create_example_catalog(catalog_path)
        print(f"\nCreated example catalog: {catalog_path}")

    # Initialize checker
    checker = BuildingBlockChecker(
        local_catalog_path=catalog_path,
        use_zinc=False
    )

    # Check molecules
    molecules = [
        ('CCO', 'Ethanol'),
        ('CC(C)O', 'Isopropanol'),
        ('CCCCCCCCCO', 'Nonanol'),  # Not in catalog
    ]

    for smiles, name in molecules:
        result = checker.check(smiles)
        print(f"\n{name} ({smiles}):")
        print(f"  Available: {result.is_available}")
        print(f"  Exact match: {result.exact_match}")

        if result.suppliers:
            print(f"  Suppliers: {len(result.suppliers)}")
            for supplier in result.suppliers[:2]:
                price = f"${supplier.price_usd:.2f}" if supplier.price_usd else "N/A"
                print(f"    - {supplier.name}: {price}")

        if result.similar_compounds:
            print(f"  Similar compounds: {len(result.similar_compounds)}")
            for sim_smiles, similarity in result.similar_compounds[:2]:
                print(f"    - {sim_smiles} (similarity: {similarity:.2f})")


def example_retrosynthesis():
    """Example: Generate retrosynthetic routes"""
    from retrosynthesis import RetrosynthesisAnalyzer

    print("\n" + "=" * 80)
    print("Example 3: Retrosynthesis Analysis")
    print("=" * 80)

    analyzer = RetrosynthesisAnalyzer(
        use_aizynthfinder=False,  # Use template-based
        max_routes=3
    )

    molecules = [
        ('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin'),
        ('CCO', 'Ethanol'),
    ]

    for smiles, name in molecules:
        print(f"\n{name}:")
        print(f"  Target: {smiles}")

        routes = analyzer.analyze(smiles)
        print(f"  Found {len(routes)} route(s)")

        for i, route in enumerate(routes, 1):
            metrics = route.get_metrics()
            print(f"\n  Route {i}:")
            print(f"    Steps: {metrics['num_steps']}")
            print(f"    Building blocks: {metrics['num_building_blocks']}")
            print(f"    Score: {route.overall_score:.2f}")
            print(f"    Method: {route.method}")


def example_route_scoring():
    """Example: Score retrosynthetic routes"""
    from retrosynthesis import RetrosynthesisAnalyzer
    from building_blocks import BuildingBlockChecker, create_example_catalog
    from sa_score import SAScoreCalculator
    from route_scoring import RouteScorer

    print("\n" + "=" * 80)
    print("Example 4: Route Scoring")
    print("=" * 80)

    # Setup
    catalog_path = Path('example_bb.csv')
    if not catalog_path.exists():
        create_example_catalog(catalog_path)

    bb_checker = BuildingBlockChecker(catalog_path, use_zinc=False)
    sa_calc = SAScoreCalculator()
    analyzer = RetrosynthesisAnalyzer(use_aizynthfinder=False)

    scorer = RouteScorer(
        building_block_checker=bb_checker,
        sa_calculator=sa_calc
    )

    # Analyze molecule
    target = 'CC(=O)Oc1ccccc1C(=O)O'  # Aspirin
    print(f"\nTarget: {target} (Aspirin)")

    routes = analyzer.analyze(target)
    print(f"Generated {len(routes)} route(s)")

    # Score routes
    scores = scorer.rank_routes(routes)

    for i, score in enumerate(scores, 1):
        print(f"\nRoute {i}:")
        print(f"  Overall Score: {score.overall_score:.3f}")
        print(f"  {score.get_interpretation()}")
        print(f"  Steps: {len(score.route.steps)}")
        print(f"  Building blocks: {score.num_total_building_blocks} "
              f"({score.num_available_building_blocks} available)")

        # Show recommendations for best route
        if i == 1:
            print("\n  Recommendations:")
            for rec in score.get_recommendations():
                print(f"    - {rec}")


def example_complete_workflow():
    """Example: Complete integrated workflow"""
    from retrosynthesis import RetrosynthesisAnalyzer
    from building_blocks import BuildingBlockChecker, create_example_catalog
    from sa_score import SAScoreCalculator
    from route_scoring import RouteScorer

    print("\n" + "=" * 80)
    print("Example 5: Complete Workflow")
    print("=" * 80)

    # Initialize components
    catalog_path = Path('example_bb.csv')
    if not catalog_path.exists():
        create_example_catalog(catalog_path)

    bb_checker = BuildingBlockChecker(catalog_path, use_zinc=False)
    sa_calc = SAScoreCalculator()
    analyzer = RetrosynthesisAnalyzer(use_aizynthfinder=False, max_routes=3)
    scorer = RouteScorer(bb_checker, sa_calc)

    # Analyze multiple molecules
    molecules = [
        'CCO',                              # Ethanol (very easy)
        'CC(C)Cc1ccc(cc1)C(C)C(O)=O',      # Ibuprofen (moderate)
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',    # Caffeine (moderate)
    ]

    results = []

    for smiles in molecules:
        # SAScore
        sa_result = sa_calc.calculate(smiles)

        # Availability
        bb_result = bb_checker.check(smiles)

        # Retrosynthesis
        routes = analyzer.analyze(smiles)

        # Scoring
        if routes:
            scores = scorer.rank_routes(routes)
            best_score = scores[0].overall_score
        else:
            best_score = 0.0

        results.append({
            'smiles': smiles,
            'sa_score': sa_result.score,
            'available': bb_result.is_available,
            'makeability': best_score
        })

    # Display summary
    print("\nSummary:")
    print("-" * 80)
    print(f"{'SMILES':<40} {'SAScore':>10} {'Available':>12} {'Makeability':>12}")
    print("-" * 80)

    for r in results:
        available = "Yes" if r['available'] else "No"
        print(f"{r['smiles']:<40} {r['sa_score']:>10.2f} {available:>12} {r['makeability']:>12.3f}")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("Synthesis Module - Simple Usage Examples")
    print("=" * 80)

    try:
        example_sa_score()
        example_building_blocks()
        example_retrosynthesis()
        example_route_scoring()
        example_complete_workflow()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80 + "\n")

    except ImportError as e:
        print(f"\nError: Missing dependency - {e}")
        print("Please install required packages: pip install rdkit")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
