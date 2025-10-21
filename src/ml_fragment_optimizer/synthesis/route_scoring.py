"""
Comprehensive Synthesis Route Scoring

Combines multiple factors to assess synthetic feasibility:
1. Number of synthetic steps
2. Route convergence (linear vs convergent)
3. Building block availability
4. SAScore of intermediates
5. Estimated cost (if available)
6. Overall complexity

Provides unified "makeability score" on 0-1 scale where:
- 1.0 = Very easy to synthesize
- 0.5 = Moderate difficulty
- 0.0 = Very difficult or impossible

Author: ML-Fragment-Optimizer Team
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from functools import lru_cache

try:
    from .retrosynthesis import RetrosynthesisRoute
    from .sa_score import SAScoreCalculator
    from .building_blocks import BuildingBlockChecker, AvailabilityResult
except ImportError:
    # For standalone testing
    import sys
    sys.path.append(str(Path(__file__).parent))
    from retrosynthesis import RetrosynthesisRoute
    from sa_score import SAScoreCalculator
    from building_blocks import BuildingBlockChecker, AvailabilityResult

logger = logging.getLogger(__name__)


@dataclass
class MakeabilityScore:
    """Comprehensive synthesis makeability assessment"""
    target_smiles: str
    route: RetrosynthesisRoute

    # Individual component scores (0-1, higher is better)
    step_score: float           # Fewer steps = higher score
    convergence_score: float    # More convergent = higher score
    availability_score: float   # More building blocks available = higher score
    sa_score: float            # Lower SAScore = higher score
    cost_score: float          # Lower cost = higher score

    # Overall score (0-1, weighted combination)
    overall_score: float

    # Weights used
    weights: Dict[str, float]

    # Additional metadata
    estimated_cost_usd: Optional[float] = None
    num_available_building_blocks: int = 0
    num_total_building_blocks: int = 0
    avg_intermediate_sa_score: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'target': self.target_smiles,
            'route_id': self.route.route_id,
            'overall_score': round(self.overall_score, 3),
            'interpretation': self.get_interpretation(),
            'component_scores': {
                'steps': round(self.step_score, 3),
                'convergence': round(self.convergence_score, 3),
                'availability': round(self.availability_score, 3),
                'synthetic_accessibility': round(self.sa_score, 3),
                'cost': round(self.cost_score, 3)
            },
            'weights': self.weights,
            'details': {
                'num_steps': len(self.route.steps),
                'num_building_blocks': self.num_total_building_blocks,
                'num_available': self.num_available_building_blocks,
                'estimated_cost_usd': self.estimated_cost_usd,
                'avg_intermediate_sa': round(self.avg_intermediate_sa_score, 2)
            },
            'route_info': self.route.to_dict()
        }

    def get_interpretation(self) -> str:
        """Get human-readable interpretation of overall score"""
        if self.overall_score >= 0.8:
            return "Very Easy - Highly feasible synthesis"
        elif self.overall_score >= 0.6:
            return "Easy - Feasible with standard methods"
        elif self.overall_score >= 0.4:
            return "Moderate - Requires some optimization"
        elif self.overall_score >= 0.2:
            return "Difficult - Significant challenges expected"
        else:
            return "Very Difficult - May not be practical"

    def get_recommendations(self) -> List[str]:
        """Get actionable recommendations based on scores"""
        recommendations = []

        if self.step_score < 0.5:
            recommendations.append(
                "Consider alternative routes with fewer steps or simplify target structure"
            )

        if self.availability_score < 0.5:
            recommendations.append(
                f"Only {self.num_available_building_blocks}/{self.num_total_building_blocks} "
                "building blocks are commercially available. Consider using more common starting materials"
            )

        if self.sa_score < 0.5:
            recommendations.append(
                "High synthetic complexity detected. Consider simplifying molecular structure "
                "or breaking into smaller fragments"
            )

        if self.convergence_score < 0.3 and len(self.route.steps) > 3:
            recommendations.append(
                "Route is highly linear. Explore convergent strategies to improve efficiency"
            )

        if self.cost_score < 0.4 and self.estimated_cost_usd:
            recommendations.append(
                f"Estimated cost is high (${self.estimated_cost_usd:.2f}). "
                "Consider cheaper building blocks or alternative routes"
            )

        if not recommendations:
            recommendations.append("Route looks feasible. Proceed with experimental validation")

        return recommendations


class RouteScorer:
    """
    Comprehensive synthesis route scoring system

    Evaluates retrosynthetic routes based on multiple criteria and
    provides unified makeability assessment.

    Scoring Components:
    1. Step Score: Penalizes long synthetic routes
    2. Convergence Score: Rewards convergent synthesis strategies
    3. Availability Score: Rewards use of commercially available building blocks
    4. SA Score: Rewards synthetically accessible intermediates
    5. Cost Score: Rewards economical routes (if cost data available)

    Example:
        >>> scorer = RouteScorer()
        >>> score = scorer.score(route, building_block_checker, sa_calculator)
        >>> print(f"Makeability: {score.overall_score:.2f}")
        >>> for rec in score.get_recommendations():
        ...     print(f"- {rec}")
    """

    DEFAULT_WEIGHTS = {
        'steps': 0.25,
        'convergence': 0.15,
        'availability': 0.30,
        'sa_score': 0.25,
        'cost': 0.05
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        building_block_checker: Optional[BuildingBlockChecker] = None,
        sa_calculator: Optional[SAScoreCalculator] = None
    ):
        """
        Initialize route scorer

        Args:
            weights: Custom weights for scoring components (must sum to 1.0)
            building_block_checker: BuildingBlockChecker instance (created if None)
            sa_calculator: SAScoreCalculator instance (created if None)
        """
        # Set weights
        self.weights = weights if weights else self.DEFAULT_WEIGHTS.copy()

        # Validate weights
        if abs(sum(self.weights.values()) - 1.0) > 0.01:
            logger.warning(f"Weights sum to {sum(self.weights.values())}, normalizing to 1.0")
            total = sum(self.weights.values())
            self.weights = {k: v/total for k, v in self.weights.items()}

        # Initialize checkers if not provided
        self.building_block_checker = building_block_checker
        self.sa_calculator = sa_calculator

        logger.info(f"RouteScorer initialized with weights: {self.weights}")

    def score(
        self,
        route: RetrosynthesisRoute,
        check_availability: bool = True,
        check_sa_scores: bool = True
    ) -> MakeabilityScore:
        """
        Score a retrosynthetic route

        Args:
            route: RetrosynthesisRoute to score
            check_availability: Check building block availability
            check_sa_scores: Calculate SAScores for intermediates

        Returns:
            MakeabilityScore with detailed assessment
        """
        # 1. Step score (fewer steps = better)
        step_score = self._score_steps(route)

        # 2. Convergence score (more convergent = better)
        convergence_score = self._score_convergence(route)

        # 3. Availability score
        availability_score = 0.5  # Default
        num_available = 0
        num_total = len(route.building_blocks)
        estimated_cost = None

        if check_availability and self.building_block_checker:
            availability_score, num_available, estimated_cost = self._score_availability(route)
        elif not check_availability:
            availability_score = 0.5  # Neutral score if not checking

        # 4. SA score (lower SAScore = better)
        sa_score = 0.5  # Default
        avg_sa = 5.0

        if check_sa_scores and self.sa_calculator:
            sa_score, avg_sa = self._score_synthetic_accessibility(route)
        elif not check_sa_scores:
            sa_score = 0.5  # Neutral score if not checking

        # 5. Cost score (lower cost = better)
        cost_score = 0.5  # Default
        if estimated_cost is not None:
            cost_score = self._score_cost(estimated_cost)

        # Calculate weighted overall score
        overall_score = (
            self.weights['steps'] * step_score +
            self.weights['convergence'] * convergence_score +
            self.weights['availability'] * availability_score +
            self.weights['sa_score'] * sa_score +
            self.weights['cost'] * cost_score
        )

        # Create result
        result = MakeabilityScore(
            target_smiles=route.target_smiles,
            route=route,
            step_score=step_score,
            convergence_score=convergence_score,
            availability_score=availability_score,
            sa_score=sa_score,
            cost_score=cost_score,
            overall_score=overall_score,
            weights=self.weights.copy(),
            estimated_cost_usd=estimated_cost,
            num_available_building_blocks=num_available,
            num_total_building_blocks=num_total,
            avg_intermediate_sa_score=avg_sa
        )

        logger.debug(f"Route scored: {overall_score:.3f}")
        return result

    def _score_steps(self, route: RetrosynthesisRoute) -> float:
        """
        Score based on number of steps

        Scoring:
        - 1-2 steps: 1.0 (excellent)
        - 3-4 steps: 0.8 (good)
        - 5-6 steps: 0.6 (acceptable)
        - 7-8 steps: 0.4 (challenging)
        - 9-10 steps: 0.2 (difficult)
        - >10 steps: 0.0 (very difficult)
        """
        num_steps = len(route.steps)

        if num_steps == 0:
            return 1.0  # Already a building block

        if num_steps <= 2:
            return 1.0
        elif num_steps <= 4:
            return 0.8
        elif num_steps <= 6:
            return 0.6
        elif num_steps <= 8:
            return 0.4
        elif num_steps <= 10:
            return 0.2
        else:
            return 0.0

    def _score_convergence(self, route: RetrosynthesisRoute) -> float:
        """
        Score based on route convergence

        Convergent routes are more efficient than linear routes
        Score is based on convergence degree from route metrics
        """
        if len(route.steps) <= 1:
            return 1.0  # Single step or no steps

        convergence_degree = route._calculate_convergence()

        # Convert to 0-1 score (higher convergence = better)
        # Slight preference for convergent routes
        score = 0.6 + (0.4 * convergence_degree)

        return score

    def _score_availability(
        self,
        route: RetrosynthesisRoute
    ) -> tuple[float, int, Optional[float]]:
        """
        Score based on building block availability

        Returns:
            Tuple of (score, num_available, estimated_cost)
        """
        if not route.building_blocks:
            return 0.0, 0, None

        # Check each building block
        num_available = 0
        total_cost = 0.0
        has_cost_data = False

        for bb_smiles in route.building_blocks:
            result = self.building_block_checker.check(bb_smiles)

            if result.is_available and result.exact_match:
                num_available += 1

                # Get cost if available
                best_price = result.get_best_price()
                if best_price is not None:
                    total_cost += best_price
                    has_cost_data = True

        # Calculate availability score
        availability_fraction = num_available / len(route.building_blocks)

        # Penalize if not all building blocks are available
        if availability_fraction == 1.0:
            score = 1.0
        elif availability_fraction >= 0.8:
            score = 0.8
        elif availability_fraction >= 0.6:
            score = 0.6
        elif availability_fraction >= 0.4:
            score = 0.4
        elif availability_fraction >= 0.2:
            score = 0.2
        else:
            score = 0.0

        estimated_cost = total_cost if has_cost_data else None

        return score, num_available, estimated_cost

    def _score_synthetic_accessibility(
        self,
        route: RetrosynthesisRoute
    ) -> tuple[float, float]:
        """
        Score based on SAScores of intermediates

        Returns:
            Tuple of (score, average_sa_score)
        """
        if not route.steps:
            # No intermediates, score the target
            result = self.sa_calculator.calculate(route.target_smiles)
            avg_sa = result.score
        else:
            # Calculate SAScore for all intermediates
            sa_scores = []
            for step in route.steps:
                for reactant in step.reactants:
                    try:
                        result = self.sa_calculator.calculate(reactant)
                        sa_scores.append(result.score)
                    except Exception as e:
                        logger.warning(f"Failed to calculate SAScore for {reactant}: {e}")

            avg_sa = sum(sa_scores) / len(sa_scores) if sa_scores else 5.0

        # Convert SAScore (1-10, lower is better) to 0-1 score (higher is better)
        # SAScore 1 → score 1.0
        # SAScore 5 → score 0.5
        # SAScore 10 → score 0.0
        score = 1.0 - ((avg_sa - 1.0) / 9.0)
        score = max(0.0, min(1.0, score))  # Clamp to 0-1

        return score, avg_sa

    def _score_cost(self, estimated_cost_usd: float) -> float:
        """
        Score based on estimated cost

        Scoring scale:
        - $0-100: 1.0 (very cheap)
        - $100-500: 0.8 (cheap)
        - $500-1000: 0.6 (moderate)
        - $1000-5000: 0.4 (expensive)
        - $5000-10000: 0.2 (very expensive)
        - >$10000: 0.0 (prohibitively expensive)
        """
        if estimated_cost_usd <= 100:
            return 1.0
        elif estimated_cost_usd <= 500:
            return 0.8
        elif estimated_cost_usd <= 1000:
            return 0.6
        elif estimated_cost_usd <= 5000:
            return 0.4
        elif estimated_cost_usd <= 10000:
            return 0.2
        else:
            return 0.0

    def score_batch(
        self,
        routes: List[RetrosynthesisRoute],
        **kwargs
    ) -> List[MakeabilityScore]:
        """
        Score multiple routes

        Args:
            routes: List of RetrosynthesisRoute objects
            **kwargs: Additional arguments passed to score()

        Returns:
            List of MakeabilityScore objects
        """
        scores = []
        for route in routes:
            try:
                score = self.score(route, **kwargs)
                scores.append(score)
            except Exception as e:
                logger.error(f"Failed to score route {route.route_id}: {e}")

        return scores

    def rank_routes(
        self,
        routes: List[RetrosynthesisRoute],
        **kwargs
    ) -> List[MakeabilityScore]:
        """
        Score and rank routes by makeability

        Args:
            routes: List of RetrosynthesisRoute objects
            **kwargs: Additional arguments passed to score()

        Returns:
            List of MakeabilityScore objects sorted by overall_score (descending)
        """
        scores = self.score_batch(routes, **kwargs)
        scores.sort(key=lambda s: s.overall_score, reverse=True)
        return scores


if __name__ == '__main__':
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)

    from retrosynthesis import RetrosynthesisAnalyzer, ReactionStep, RouteType
    from building_blocks import BuildingBlockChecker, create_example_catalog
    from sa_score import SAScoreCalculator

    # Create example catalog
    catalog_path = Path('example_building_blocks.csv')
    if not catalog_path.exists():
        create_example_catalog(catalog_path)

    # Initialize components
    bb_checker = BuildingBlockChecker(
        local_catalog_path=catalog_path,
        use_zinc=False
    )
    sa_calc = SAScoreCalculator()

    # Initialize scorer with custom weights (emphasize availability)
    custom_weights = {
        'steps': 0.20,
        'convergence': 0.10,
        'availability': 0.40,  # Emphasize availability
        'sa_score': 0.25,
        'cost': 0.05
    }
    scorer = RouteScorer(
        weights=custom_weights,
        building_block_checker=bb_checker,
        sa_calculator=sa_calc
    )

    # Generate routes for test molecule
    analyzer = RetrosynthesisAnalyzer(use_aizynthfinder=False)
    test_smiles = 'CC(=O)Oc1ccccc1C(=O)O'  # Aspirin

    print("\nComprehensive Route Scoring Example:")
    print("=" * 80)
    print(f"Target: {test_smiles} (Aspirin)")
    print("\nGenerating routes...")

    routes = analyzer.analyze(test_smiles, method='template')

    if routes:
        print(f"\nScoring {len(routes)} route(s)...\n")

        scores = scorer.rank_routes(routes)

        for i, score in enumerate(scores):
            print(f"\nRoute {i + 1}:")
            print(f"Overall Score: {score.overall_score:.3f} - {score.get_interpretation()}")
            print("\nComponent Scores:")
            print(f"  Steps:        {score.step_score:.3f} ({len(score.route.steps)} steps)")
            print(f"  Convergence:  {score.convergence_score:.3f}")
            print(f"  Availability: {score.availability_score:.3f} "
                  f"({score.num_available_building_blocks}/{score.num_total_building_blocks} available)")
            print(f"  SA Score:     {score.sa_score:.3f} (avg SA: {score.avg_intermediate_sa_score:.2f})")
            print(f"  Cost:         {score.cost_score:.3f}", end='')
            if score.estimated_cost_usd:
                print(f" (${score.estimated_cost_usd:.2f})")
            else:
                print(" (no cost data)")

            print("\nRecommendations:")
            for rec in score.get_recommendations():
                print(f"  - {rec}")

            if i == 0:  # Show route details for best route
                print("\nRoute Details:")
                for step in score.route.steps:
                    print(f"  Step {step.step_number} ({step.reaction_type}):")
                    print(f"    Reactants: {', '.join(step.reactants)}")
                    print(f"    Product: {step.product}")
    else:
        print("No routes found")

    print("\n" + "=" * 80)
    print("\nScoring Weights Used:")
    for component, weight in scorer.weights.items():
        print(f"  {component}: {weight:.2f}")
