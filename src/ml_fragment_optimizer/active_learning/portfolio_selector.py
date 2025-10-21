"""
Portfolio Selection for Multi-Objective Molecular Optimization

This module implements portfolio selection strategies that balance multiple
competing objectives in molecular design:
1. Predicted performance (exploitation)
2. Model uncertainty (exploration)
3. Molecular diversity
4. Synthesis feasibility
5. Budget constraints

Mathematical Background
-----------------------
Portfolio selection treats molecule selection as a multi-objective optimization:

    maximize: α₁·performance + α₂·uncertainty + α₃·diversity - α₄·cost

Subject to constraints:
    - Total cost ≤ budget
    - Synthesis feasibility ≥ threshold
    - Diversity metric ≥ minimum

Common approaches:
1. Scalarization: weighted sum of objectives
2. Pareto frontier: non-dominated solutions
3. Utility function: user-defined preferences
4. Sequential greedy: iteratively add molecules maximizing marginal utility

Author: Claude
Date: 2025-10-20
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Callable
from enum import Enum

import numpy as np
from scipy.optimize import linprog

from .diversity_sampler import compute_distance_matrix, DistanceMetric


class ObjectiveType(Enum):
    """Types of objectives for portfolio optimization."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class Objective:
    """Single objective for portfolio optimization.

    Attributes
    ----------
    name : str
        Objective name (e.g., 'affinity', 'uncertainty', 'diversity')
    values : np.ndarray
        Objective values for each candidate
    weight : float
        Importance weight (default: 1.0)
    obj_type : ObjectiveType
        Whether to maximize or minimize
    normalize : bool
        Whether to normalize to [0, 1] range
    """
    name: str
    values: np.ndarray
    weight: float = 1.0
    obj_type: ObjectiveType = ObjectiveType.MAXIMIZE
    normalize: bool = True

    def __post_init__(self):
        """Normalize values if requested."""
        self.values = np.asarray(self.values).ravel()
        if self.normalize:
            v_min = self.values.min()
            v_max = self.values.max()
            if v_max > v_min:
                self.values = (self.values - v_min) / (v_max - v_min)
            else:
                self.values = np.zeros_like(self.values)

        # Flip sign for minimization objectives
        if self.obj_type == ObjectiveType.MINIMIZE:
            self.values = 1.0 - self.values


@dataclass
class PortfolioConfig:
    """Configuration for portfolio selection.

    Attributes
    ----------
    weights : Dict[str, float]
        Weights for each objective (e.g., {'performance': 0.5, 'uncertainty': 0.3, 'diversity': 0.2})
    budget : Optional[float]
        Maximum total cost
    min_diversity : float
        Minimum average diversity (0-1)
    diversity_metric : DistanceMetric
        Distance metric for diversity
    synthesis_threshold : Optional[float]
        Minimum synthesis feasibility score
    batch_size : int
        Number of molecules to select
    strategy : str
        Selection strategy: 'greedy', 'pareto', 'scalarization', 'utility'
    diversity_penalty : float
        Penalty coefficient for selecting similar molecules
    must_include : Optional[List[int]]
        Indices that must be included
    must_exclude : Optional[List[int]]
        Indices that must be excluded
    """
    weights: Dict[str, float] = field(default_factory=dict)
    budget: Optional[float] = None
    min_diversity: float = 0.3
    diversity_metric: DistanceMetric = DistanceMetric.TANIMOTO
    synthesis_threshold: Optional[float] = None
    batch_size: int = 10
    strategy: str = "greedy"
    diversity_penalty: float = 0.5
    must_include: Optional[List[int]] = None
    must_exclude: Optional[List[int]] = None


class PortfolioSelector:
    """Multi-objective portfolio selection for molecular optimization.

    Balances multiple objectives (performance, uncertainty, diversity, cost)
    to select a diverse, high-value batch of molecules.

    Examples
    --------
    >>> # Setup objectives
    >>> predicted_affinity = np.random.rand(1000)
    >>> uncertainty = np.random.rand(1000)
    >>> synthesis_cost = np.random.uniform(100, 1000, 1000)
    >>> fingerprints = np.random.randint(0, 2, (1000, 2048))
    >>>
    >>> # Configure portfolio
    >>> config = PortfolioConfig(
    ...     weights={'performance': 0.5, 'uncertainty': 0.3, 'diversity': 0.2},
    ...     budget=5000,
    ...     batch_size=10,
    ...     strategy='greedy'
    ... )
    >>>
    >>> # Create objectives
    >>> objectives = [
    ...     Objective('performance', predicted_affinity, weight=0.5),
    ...     Objective('uncertainty', uncertainty, weight=0.3),
    ... ]
    >>>
    >>> # Select portfolio
    >>> selector = PortfolioSelector(config)
    >>> selected = selector.select(
    ...     objectives=objectives,
    ...     fingerprints=fingerprints,
    ...     costs=synthesis_cost
    ... )
    """

    def __init__(self, config: PortfolioConfig):
        """Initialize portfolio selector.

        Parameters
        ----------
        config : PortfolioConfig
            Configuration for portfolio selection
        """
        self.config = config

    def select(
        self,
        objectives: List[Objective],
        fingerprints: Optional[np.ndarray] = None,
        costs: Optional[np.ndarray] = None,
        synthesis_scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Select molecular portfolio.

        Parameters
        ----------
        objectives : List[Objective]
            List of objectives to optimize
        fingerprints : Optional[np.ndarray], shape (n_candidates, n_features)
            Molecular fingerprints for diversity calculation
        costs : Optional[np.ndarray], shape (n_candidates,)
            Cost of synthesizing/testing each molecule
        synthesis_scores : Optional[np.ndarray], shape (n_candidates,)
            Synthesis feasibility scores (0-1)

        Returns
        -------
        selected_indices : np.ndarray, shape (batch_size,)
            Indices of selected molecules
        """
        if self.config.strategy == "greedy":
            return self._select_greedy(objectives, fingerprints, costs, synthesis_scores)
        elif self.config.strategy == "pareto":
            return self._select_pareto(objectives, fingerprints, costs, synthesis_scores)
        elif self.config.strategy == "scalarization":
            return self._select_scalarization(objectives, fingerprints, costs, synthesis_scores)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

    def _compute_scalarized_objective(
        self,
        objectives: List[Objective]
    ) -> np.ndarray:
        """Compute weighted sum of objectives.

        Parameters
        ----------
        objectives : List[Objective]
            List of objectives

        Returns
        -------
        scores : np.ndarray, shape (n_candidates,)
            Scalarized objective values
        """
        n_candidates = len(objectives[0].values)
        scores = np.zeros(n_candidates)

        for obj in objectives:
            scores += obj.weight * obj.values

        return scores

    def _apply_constraints(
        self,
        available: np.ndarray,
        costs: Optional[np.ndarray],
        synthesis_scores: Optional[np.ndarray],
        selected_cost: float
    ) -> np.ndarray:
        """Apply hard constraints to available molecules.

        Parameters
        ----------
        available : np.ndarray, shape (n_candidates,)
            Boolean mask of available candidates
        costs : Optional[np.ndarray], shape (n_candidates,)
            Costs
        synthesis_scores : Optional[np.ndarray], shape (n_candidates,)
            Synthesis feasibility scores
        selected_cost : float
            Total cost already selected

        Returns
        -------
        available : np.ndarray, shape (n_candidates,)
            Updated boolean mask
        """
        available = available.copy()

        # Budget constraint
        if costs is not None and self.config.budget is not None:
            remaining_budget = self.config.budget - selected_cost
            available &= (costs <= remaining_budget)

        # Synthesis feasibility constraint
        if synthesis_scores is not None and self.config.synthesis_threshold is not None:
            available &= (synthesis_scores >= self.config.synthesis_threshold)

        # Must exclude
        if self.config.must_exclude is not None:
            available[self.config.must_exclude] = False

        return available

    def _select_greedy(
        self,
        objectives: List[Objective],
        fingerprints: Optional[np.ndarray],
        costs: Optional[np.ndarray],
        synthesis_scores: Optional[np.ndarray]
    ) -> np.ndarray:
        """Greedy portfolio selection with diversity penalty.

        Algorithm
        ---------
        1. Compute scalarized objective for all candidates
        2. Select highest-scoring candidate that satisfies constraints
        3. Apply diversity penalty: reduce scores of similar molecules
        4. Repeat until batch_size molecules selected

        Parameters
        ----------
        objectives : List[Objective]
            Objectives to optimize
        fingerprints : Optional[np.ndarray]
            Fingerprints for diversity
        costs : Optional[np.ndarray]
            Costs
        synthesis_scores : Optional[np.ndarray]
            Synthesis feasibility

        Returns
        -------
        selected_indices : np.ndarray
            Selected molecule indices
        """
        n_candidates = len(objectives[0].values)

        # Compute base scores
        scores = self._compute_scalarized_objective(objectives)

        # Initialize
        selected = []
        available = np.ones(n_candidates, dtype=bool)
        selected_cost = 0.0

        # Must include
        if self.config.must_include is not None:
            for idx in self.config.must_include:
                selected.append(idx)
                available[idx] = False
                if costs is not None:
                    selected_cost += costs[idx]

        # Greedy selection
        for _ in range(self.config.batch_size - len(selected)):
            # Apply constraints
            available = self._apply_constraints(
                available, costs, synthesis_scores, selected_cost
            )

            if not available.any():
                break  # No feasible candidates remaining

            # Select best available
            scores_masked = scores.copy()
            scores_masked[~available] = -np.inf
            idx = np.argmax(scores_masked)

            selected.append(idx)
            available[idx] = False

            if costs is not None:
                selected_cost += costs[idx]

            # Apply diversity penalty
            if fingerprints is not None and self.config.diversity_penalty > 0:
                distances = compute_distance_matrix(
                    fingerprints,
                    fingerprints[idx:idx+1],
                    metric=self.config.diversity_metric
                ).ravel()

                # Penalty inversely proportional to distance
                penalty = self.config.diversity_penalty * (1.0 - distances)
                scores -= penalty

        return np.array(selected)

    def _select_pareto(
        self,
        objectives: List[Objective],
        fingerprints: Optional[np.ndarray],
        costs: Optional[np.ndarray],
        synthesis_scores: Optional[np.ndarray]
    ) -> np.ndarray:
        """Select from Pareto frontier.

        Algorithm
        ---------
        1. Identify Pareto-optimal molecules (non-dominated)
        2. Select diverse subset from Pareto frontier
        3. If fewer than batch_size on frontier, add best remaining

        Parameters
        ----------
        objectives : List[Objective]
            Objectives
        fingerprints : Optional[np.ndarray]
            Fingerprints for diversity
        costs : Optional[np.ndarray]
            Costs
        synthesis_scores : Optional[np.ndarray]
            Synthesis feasibility

        Returns
        -------
        selected_indices : np.ndarray
            Selected indices
        """
        n_candidates = len(objectives[0].values)

        # Stack objective values
        obj_matrix = np.column_stack([obj.values for obj in objectives])

        # Find Pareto frontier
        pareto_mask = self._find_pareto_frontier(obj_matrix)
        pareto_indices = np.where(pareto_mask)[0]

        # Apply constraints to Pareto set
        available = np.ones(n_candidates, dtype=bool)
        available = self._apply_constraints(available, costs, synthesis_scores, 0.0)
        pareto_indices = pareto_indices[available[pareto_indices]]

        # If enough on frontier, select diverse subset
        if len(pareto_indices) >= self.config.batch_size:
            if fingerprints is not None:
                # Select diverse molecules from Pareto frontier
                from .diversity_sampler import select_diverse_molecules
                selected = select_diverse_molecules(
                    fingerprints[pareto_indices],
                    n_select=self.config.batch_size,
                    method="maxmin",
                    metric=self.config.diversity_metric
                )
                return pareto_indices[selected]
            else:
                # Random subset
                return np.random.choice(
                    pareto_indices,
                    size=self.config.batch_size,
                    replace=False
                )
        else:
            # Include all Pareto-optimal, then add best remaining
            selected = list(pareto_indices)
            available[pareto_indices] = False

            # Scalarized score for remaining
            scores = self._compute_scalarized_objective(objectives)
            scores[~available] = -np.inf

            remaining_needed = self.config.batch_size - len(selected)
            top_remaining = np.argsort(scores)[-remaining_needed:]

            selected.extend(top_remaining)
            return np.array(selected)

    def _find_pareto_frontier(self, obj_matrix: np.ndarray) -> np.ndarray:
        """Find Pareto-optimal solutions (all objectives maximized).

        Parameters
        ----------
        obj_matrix : np.ndarray, shape (n_candidates, n_objectives)
            Objective values

        Returns
        -------
        is_pareto : np.ndarray, shape (n_candidates,)
            Boolean mask indicating Pareto-optimal molecules
        """
        n_candidates = obj_matrix.shape[0]
        is_pareto = np.ones(n_candidates, dtype=bool)

        for i in range(n_candidates):
            if is_pareto[i]:
                # Check if any other candidate dominates i
                # j dominates i if: obj[j] >= obj[i] for all objectives
                #                   and obj[j] > obj[i] for at least one
                dominates = np.all(obj_matrix >= obj_matrix[i], axis=1)
                strictly_better = np.any(obj_matrix > obj_matrix[i], axis=1)
                is_dominated = dominates & strictly_better

                is_pareto[i] = not is_dominated.any()

        return is_pareto

    def _select_scalarization(
        self,
        objectives: List[Objective],
        fingerprints: Optional[np.ndarray],
        costs: Optional[np.ndarray],
        synthesis_scores: Optional[np.ndarray]
    ) -> np.ndarray:
        """Simple scalarization: select top batch_size by weighted score.

        Parameters
        ----------
        objectives : List[Objective]
            Objectives
        fingerprints : Optional[np.ndarray]
            Not used in this strategy
        costs : Optional[np.ndarray]
            Costs for budget constraint
        synthesis_scores : Optional[np.ndarray]
            Synthesis feasibility

        Returns
        -------
        selected_indices : np.ndarray
            Selected indices
        """
        n_candidates = len(objectives[0].values)

        # Scalarized scores
        scores = self._compute_scalarized_objective(objectives)

        # Apply constraints
        available = np.ones(n_candidates, dtype=bool)
        available = self._apply_constraints(available, costs, synthesis_scores, 0.0)

        # If budget constraint, iteratively select best within budget
        if costs is not None and self.config.budget is not None:
            selected = []
            selected_cost = 0.0

            for _ in range(self.config.batch_size):
                available = self._apply_constraints(
                    available, costs, synthesis_scores, selected_cost
                )

                if not available.any():
                    break

                scores_masked = scores.copy()
                scores_masked[~available] = -np.inf
                idx = np.argmax(scores_masked)

                selected.append(idx)
                available[idx] = False
                selected_cost += costs[idx]

            return np.array(selected)
        else:
            # Simple: select top batch_size
            scores[~available] = -np.inf
            selected = np.argsort(scores)[-self.config.batch_size:]
            return selected[::-1]  # Descending order

    def compute_portfolio_metrics(
        self,
        selected_indices: np.ndarray,
        objectives: List[Objective],
        fingerprints: Optional[np.ndarray] = None,
        costs: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute metrics for selected portfolio.

        Parameters
        ----------
        selected_indices : np.ndarray
            Indices of selected molecules
        objectives : List[Objective]
            Objectives
        fingerprints : Optional[np.ndarray]
            Fingerprints for diversity
        costs : Optional[np.ndarray]
            Costs

        Returns
        -------
        metrics : dict
            Portfolio metrics:
            - mean_<objective>: average objective value
            - total_cost: total synthesis cost
            - mean_diversity: average pairwise diversity
            - min_diversity: minimum pairwise diversity
        """
        metrics = {}

        # Objective metrics
        for obj in objectives:
            metrics[f"mean_{obj.name}"] = obj.values[selected_indices].mean()
            metrics[f"max_{obj.name}"] = obj.values[selected_indices].max()

        # Cost
        if costs is not None:
            metrics["total_cost"] = costs[selected_indices].sum()
            metrics["mean_cost"] = costs[selected_indices].mean()

        # Diversity
        if fingerprints is not None and len(selected_indices) > 1:
            selected_fps = fingerprints[selected_indices]
            dist_matrix = compute_distance_matrix(
                selected_fps,
                metric=self.config.diversity_metric,
                condensed=True
            )

            metrics["mean_diversity"] = dist_matrix.mean()
            metrics["min_diversity"] = dist_matrix.min()
            metrics["max_diversity"] = dist_matrix.max()

        return metrics


class UtilityFunction:
    """Custom utility function for portfolio selection.

    Allows users to define complex, non-linear preferences over objectives.

    Examples
    --------
    >>> # Define custom utility: prefer high affinity AND low toxicity
    >>> def utility(affinity, toxicity):
    ...     return affinity * (1 - toxicity)
    >>>
    >>> util_fn = UtilityFunction(utility)
    >>> scores = util_fn.evaluate(affinity=affinities, toxicity=toxicities)
    """

    def __init__(self, func: Callable[..., np.ndarray]):
        """Initialize utility function.

        Parameters
        ----------
        func : Callable
            Function mapping objective values to utility
            Should accept keyword arguments for each objective
        """
        self.func = func

    def evaluate(self, **objectives: np.ndarray) -> np.ndarray:
        """Evaluate utility function.

        Parameters
        ----------
        **objectives : np.ndarray
            Objective values as keyword arguments

        Returns
        -------
        utility : np.ndarray
            Utility values
        """
        return self.func(**objectives)


def select_portfolio_with_utility(
    utility_fn: UtilityFunction,
    objectives_dict: Dict[str, np.ndarray],
    batch_size: int,
    fingerprints: Optional[np.ndarray] = None,
    diversity_weight: float = 0.0,
    metric: DistanceMetric = DistanceMetric.TANIMOTO
) -> np.ndarray:
    """Select portfolio using custom utility function.

    Parameters
    ----------
    utility_fn : UtilityFunction
        Custom utility function
    objectives_dict : Dict[str, np.ndarray]
        Objective values as {name: values} dict
    batch_size : int
        Number to select
    fingerprints : Optional[np.ndarray]
        Fingerprints for diversity
    diversity_weight : float
        Weight for diversity penalty
    metric : DistanceMetric
        Distance metric

    Returns
    -------
    selected_indices : np.ndarray
        Selected molecule indices

    Examples
    --------
    >>> def utility(affinity, qed):
    ...     return affinity * qed
    >>> util_fn = UtilityFunction(utility)
    >>> selected = select_portfolio_with_utility(
    ...     util_fn,
    ...     {'affinity': affinities, 'qed': qed_scores},
    ...     batch_size=10
    ... )
    """
    # Evaluate utility
    scores = utility_fn.evaluate(**objectives_dict)

    if diversity_weight == 0.0 or fingerprints is None:
        # Simple: select top batch_size
        return np.argsort(scores)[-batch_size:][::-1]
    else:
        # Greedy with diversity penalty
        n_candidates = len(scores)
        selected = []
        available = np.ones(n_candidates, dtype=bool)

        for _ in range(batch_size):
            scores_masked = scores.copy()
            scores_masked[~available] = -np.inf
            idx = np.argmax(scores_masked)

            selected.append(idx)
            available[idx] = False

            # Diversity penalty
            distances = compute_distance_matrix(
                fingerprints,
                fingerprints[idx:idx+1],
                metric=metric
            ).ravel()

            penalty = diversity_weight * (1.0 - distances)
            scores -= penalty

        return np.array(selected)
