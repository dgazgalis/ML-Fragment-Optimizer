"""
Design of Experiments (DOE) for Active Learning

This module implements design of experiments strategies for systematic exploration
and adaptive experimentation in molecular optimization campaigns.

DOE Strategies
--------------
1. **Latin Hypercube Sampling (LHS)**: Space-filling design for initial exploration
2. **Adaptive Design**: Use model predictions to guide next experiments
3. **Sequential Design**: Multi-round campaigns with learning between rounds
4. **Information-Theoretic Design**: Maximize expected information gain
5. **Hybrid Design**: Combine exploration (LHS) and exploitation (BO)

Mathematical Background
-----------------------
Information Gain:
    IG(x) = H(y) - E[H(y|D ∪ {x})]
         = E[KL(p(y|D ∪ {x}) || p(y|D))]

For Gaussian processes:
    IG(x) ∝ log(1 + σ²(x)/σ²_noise)

Author: Claude
Date: 2025-10-20
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Callable
from enum import Enum

import numpy as np
from scipy.stats import qmc
from sklearn.preprocessing import MinMaxScaler


class DesignStrategy(Enum):
    """Design of experiments strategies."""
    LHS = "latin_hypercube"
    RANDOM = "random"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"
    UNCERTAINTY = "uncertainty"
    INFORMATION_GAIN = "information_gain"


@dataclass
class ExperimentDesignConfig:
    """Configuration for experiment design.

    Attributes
    ----------
    strategy : DesignStrategy
        DOE strategy to use
    initial_samples : int
        Number of initial samples (for LHS or random)
    samples_per_round : int
        Number of samples to select per round
    max_rounds : int
        Maximum number of experimental rounds
    exploration_weight : float
        Weight for exploration vs exploitation (0=exploit, 1=explore)
    adaptive_weight_decay : bool
        Whether to decay exploration weight over rounds
    stratify_by : Optional[str]
        Property to stratify by (e.g., molecular weight, logP)
    n_strata : int
        Number of strata for stratified sampling
    random_state : Optional[int]
        Random seed for reproducibility
    """
    strategy: DesignStrategy = DesignStrategy.LHS
    initial_samples: int = 20
    samples_per_round: int = 10
    max_rounds: int = 10
    exploration_weight: float = 0.5
    adaptive_weight_decay: bool = True
    stratify_by: Optional[str] = None
    n_strata: int = 5
    random_state: Optional[int] = None


@dataclass
class ExperimentalRound:
    """Results from one round of experimentation.

    Attributes
    ----------
    round_number : int
        Round number (0-indexed)
    selected_indices : np.ndarray
        Indices of molecules selected in this round
    predicted_values : Optional[np.ndarray]
        Model predictions for selected molecules
    observed_values : Optional[np.ndarray]
        Actual experimental outcomes
    uncertainty : Optional[np.ndarray]
        Model uncertainty for selected molecules
    metrics : Dict[str, float]
        Performance metrics for this round
    """
    round_number: int
    selected_indices: np.ndarray
    predicted_values: Optional[np.ndarray] = None
    observed_values: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class ExperimentDesigner:
    """Design of experiments for active learning campaigns.

    Orchestrates multi-round experimental campaigns with adaptive learning.

    Examples
    --------
    >>> # Initial setup
    >>> config = ExperimentDesignConfig(
    ...     strategy=DesignStrategy.HYBRID,
    ...     initial_samples=50,
    ...     samples_per_round=20,
    ...     max_rounds=5
    ... )
    >>> designer = ExperimentDesigner(config)
    >>>
    >>> # Round 0: Initial exploration (LHS)
    >>> X_candidates = np.random.randn(10000, 100)
    >>> initial_batch = designer.design_initial_experiments(X_candidates)
    >>>
    >>> # Evaluate initial batch
    >>> y_initial = evaluate_molecules(X_candidates[initial_batch])
    >>> designer.update(initial_batch, y_initial)
    >>>
    >>> # Subsequent rounds: Adaptive selection
    >>> for round_num in range(1, config.max_rounds):
    ...     # Train model on accumulated data
    ...     model.fit(designer.X_observed, designer.y_observed)
    ...
    ...     # Predict for candidates
    ...     mean, std = model.predict(X_candidates)
    ...
    ...     # Design next experiment
    ...     next_batch = designer.design_next_round(
    ...         X_candidates,
    ...         mean=mean,
    ...         std=std,
    ...         round_number=round_num
    ...     )
    ...
    ...     # Evaluate and update
    ...     y_new = evaluate_molecules(X_candidates[next_batch])
    ...     designer.update(next_batch, y_new, mean[next_batch], std[next_batch])
    """

    def __init__(self, config: ExperimentDesignConfig):
        """Initialize experiment designer.

        Parameters
        ----------
        config : ExperimentDesignConfig
            Configuration for experiment design
        """
        self.config = config
        self.rounds: List[ExperimentalRound] = []
        self.X_observed: Optional[np.ndarray] = None
        self.y_observed: Optional[np.ndarray] = None
        self.selected_indices_all: List[int] = []

    def design_initial_experiments(
        self,
        X: np.ndarray,
        stratify_values: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Design initial experiments (exploration phase).

        Parameters
        ----------
        X : np.ndarray, shape (n_candidates, n_features)
            Candidate molecules
        stratify_values : Optional[np.ndarray], shape (n_candidates,)
            Values for stratified sampling (e.g., molecular weight)

        Returns
        -------
        selected_indices : np.ndarray, shape (initial_samples,)
            Indices of molecules to evaluate
        """
        if self.config.strategy in [DesignStrategy.LHS, DesignStrategy.HYBRID]:
            selected = self._latin_hypercube_sampling(X, self.config.initial_samples)
        elif self.config.strategy == DesignStrategy.RANDOM:
            selected = self._random_sampling(X, self.config.initial_samples)
        else:
            # Default to LHS for initial exploration
            selected = self._latin_hypercube_sampling(X, self.config.initial_samples)

        # Stratified sampling if requested
        if stratify_values is not None:
            selected = self._stratified_sampling(
                stratify_values,
                self.config.initial_samples,
                self.config.n_strata
            )

        # Record round
        self.rounds.append(ExperimentalRound(
            round_number=0,
            selected_indices=selected
        ))
        self.selected_indices_all.extend(selected.tolist())

        return selected

    def design_next_round(
        self,
        X: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        round_number: int,
        exclude_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Design next round of experiments (adaptive phase).

        Parameters
        ----------
        X : np.ndarray, shape (n_candidates, n_features)
            Candidate molecules
        mean : np.ndarray, shape (n_candidates,)
            Model predictions
        std : np.ndarray, shape (n_candidates,)
            Model uncertainties
        round_number : int
            Current round number
        exclude_indices : Optional[np.ndarray]
            Indices to exclude (e.g., already evaluated)

        Returns
        -------
        selected_indices : np.ndarray, shape (samples_per_round,)
            Indices of molecules to evaluate
        """
        # Compute exploration weight for this round
        exploration_weight = self._compute_exploration_weight(round_number)

        # Exclude already selected
        available = np.ones(len(X), dtype=bool)
        if exclude_indices is not None:
            available[exclude_indices] = False
        available[self.selected_indices_all] = False

        # Select based on strategy
        if self.config.strategy == DesignStrategy.ADAPTIVE:
            selected = self._adaptive_selection(
                mean, std, exploration_weight, available
            )
        elif self.config.strategy == DesignStrategy.HYBRID:
            selected = self._hybrid_selection(
                X, mean, std, exploration_weight, available
            )
        elif self.config.strategy == DesignStrategy.UNCERTAINTY:
            selected = self._uncertainty_selection(std, available)
        elif self.config.strategy == DesignStrategy.INFORMATION_GAIN:
            selected = self._information_gain_selection(mean, std, available)
        else:
            # Default to adaptive
            selected = self._adaptive_selection(
                mean, std, exploration_weight, available
            )

        # Record round
        self.rounds.append(ExperimentalRound(
            round_number=round_number,
            selected_indices=selected,
            predicted_values=mean[selected],
            uncertainty=std[selected]
        ))
        self.selected_indices_all.extend(selected.tolist())

        return selected

    def update(
        self,
        indices: np.ndarray,
        y_observed: np.ndarray,
        y_predicted: Optional[np.ndarray] = None,
        uncertainty: Optional[np.ndarray] = None
    ) -> None:
        """Update with experimental results.

        Parameters
        ----------
        indices : np.ndarray
            Indices of evaluated molecules
        y_observed : np.ndarray
            Experimental outcomes
        y_predicted : Optional[np.ndarray]
            Model predictions (for metrics)
        uncertainty : Optional[np.ndarray]
            Model uncertainties (for metrics)
        """
        # Store results in latest round
        if self.rounds:
            self.rounds[-1].observed_values = y_observed
            if y_predicted is not None:
                self.rounds[-1].predicted_values = y_predicted
            if uncertainty is not None:
                self.rounds[-1].uncertainty = uncertainty

            # Compute metrics
            self.rounds[-1].metrics = self._compute_round_metrics(
                y_observed, y_predicted, uncertainty
            )

    def _latin_hypercube_sampling(
        self,
        X: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """Latin Hypercube Sampling in feature space.

        LHS ensures good coverage of the feature space by dividing each
        dimension into n_samples intervals and selecting one point from
        each interval.

        Parameters
        ----------
        X : np.ndarray, shape (n_candidates, n_features)
            Candidates
        n_samples : int
            Number of samples

        Returns
        -------
        selected_indices : np.ndarray
            Selected molecule indices
        """
        n_candidates, n_features = X.shape

        if n_samples > n_candidates:
            raise ValueError(f"Cannot sample {n_samples} from {n_candidates}")

        # Normalize features to [0, 1]
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Generate LHS design in [0, 1]^n_features
        sampler = qmc.LatinHypercube(
            d=n_features,
            seed=self.config.random_state
        )
        lhs_points = sampler.random(n=n_samples)

        # Find nearest candidates to LHS points
        from scipy.spatial.distance import cdist
        distances = cdist(lhs_points, X_scaled)
        selected_indices = np.argmin(distances, axis=1)

        # Remove duplicates (if any)
        selected_indices = np.unique(selected_indices)

        # If duplicates, fill with random samples
        while len(selected_indices) < n_samples:
            remaining = list(set(range(n_candidates)) - set(selected_indices))
            additional = np.random.choice(remaining, size=n_samples - len(selected_indices))
            selected_indices = np.unique(np.concatenate([selected_indices, additional]))

        return selected_indices[:n_samples]

    def _random_sampling(
        self,
        X: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """Random sampling (baseline).

        Parameters
        ----------
        X : np.ndarray
            Candidates
        n_samples : int
            Number to sample

        Returns
        -------
        selected_indices : np.ndarray
            Selected indices
        """
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)

        return np.random.choice(len(X), size=n_samples, replace=False)

    def _stratified_sampling(
        self,
        values: np.ndarray,
        n_samples: int,
        n_strata: int
    ) -> np.ndarray:
        """Stratified sampling based on continuous variable.

        Divides value range into n_strata bins and samples proportionally
        from each stratum.

        Parameters
        ----------
        values : np.ndarray, shape (n_candidates,)
            Values to stratify by
        n_samples : int
            Total number to sample
        n_strata : int
            Number of strata

        Returns
        -------
        selected_indices : np.ndarray
            Selected indices
        """
        # Assign to strata
        percentiles = np.linspace(0, 100, n_strata + 1)
        bins = np.percentile(values, percentiles)
        strata_labels = np.digitize(values, bins[1:-1])

        # Sample from each stratum
        samples_per_stratum = n_samples // n_strata
        remainder = n_samples % n_strata

        selected = []
        for stratum in range(n_strata):
            stratum_indices = np.where(strata_labels == stratum)[0]

            if len(stratum_indices) == 0:
                continue

            # Sample from this stratum
            n_from_stratum = samples_per_stratum + (1 if stratum < remainder else 0)
            n_from_stratum = min(n_from_stratum, len(stratum_indices))

            sampled = np.random.choice(
                stratum_indices,
                size=n_from_stratum,
                replace=False
            )
            selected.extend(sampled)

        return np.array(selected[:n_samples])

    def _adaptive_selection(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        exploration_weight: float,
        available: np.ndarray
    ) -> np.ndarray:
        """Adaptive selection balancing exploitation and exploration.

        Uses UCB-style acquisition:
            score = mean + exploration_weight * std

        Parameters
        ----------
        mean : np.ndarray
            Predicted values
        std : np.ndarray
            Uncertainties
        exploration_weight : float
            Weight for exploration
        available : np.ndarray
            Boolean mask of available candidates

        Returns
        -------
        selected_indices : np.ndarray
            Selected indices
        """
        # UCB-style score
        scores = mean + exploration_weight * std
        scores[~available] = -np.inf

        # Select top samples_per_round
        selected = np.argsort(scores)[-self.config.samples_per_round:]
        return selected[::-1]

    def _hybrid_selection(
        self,
        X: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        exploration_weight: float,
        available: np.ndarray
    ) -> np.ndarray:
        """Hybrid selection: combine exploitation and space-filling.

        Split budget:
        - 50% from adaptive (UCB)
        - 50% from LHS in unexplored regions

        Parameters
        ----------
        X : np.ndarray
            Candidates
        mean : np.ndarray
            Predictions
        std : np.ndarray
            Uncertainties
        exploration_weight : float
            Exploration weight
        available : np.ndarray
            Available mask

        Returns
        -------
        selected_indices : np.ndarray
            Selected indices
        """
        n_exploit = self.config.samples_per_round // 2
        n_explore = self.config.samples_per_round - n_exploit

        # Exploitation: adaptive selection
        scores = mean + exploration_weight * std
        scores[~available] = -np.inf
        exploit_indices = np.argsort(scores)[-n_exploit:]

        # Exploration: LHS in available space
        available_indices = np.where(available)[0]
        available_X = X[available_indices]

        if len(available_indices) >= n_explore:
            explore_local_indices = self._latin_hypercube_sampling(
                available_X,
                n_explore
            )
            explore_indices = available_indices[explore_local_indices]
        else:
            explore_indices = available_indices

        # Combine
        selected = np.concatenate([exploit_indices, explore_indices])
        return np.unique(selected)[:self.config.samples_per_round]

    def _uncertainty_selection(
        self,
        std: np.ndarray,
        available: np.ndarray
    ) -> np.ndarray:
        """Pure uncertainty sampling (maximum variance).

        Parameters
        ----------
        std : np.ndarray
            Uncertainties
        available : np.ndarray
            Available mask

        Returns
        -------
        selected_indices : np.ndarray
            Selected indices
        """
        std_masked = std.copy()
        std_masked[~available] = -np.inf

        selected = np.argsort(std_masked)[-self.config.samples_per_round:]
        return selected[::-1]

    def _information_gain_selection(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        available: np.ndarray,
        noise_var: float = 0.01
    ) -> np.ndarray:
        """Information gain (BALD - Bayesian Active Learning by Disagreement).

        For Gaussian processes:
            IG(x) ≈ 0.5 * log(1 + σ²(x) / σ²_noise)

        Parameters
        ----------
        mean : np.ndarray
            Predictions (not used, included for consistency)
        std : np.ndarray
            Uncertainties
        available : np.ndarray
            Available mask
        noise_var : float
            Observation noise variance

        Returns
        -------
        selected_indices : np.ndarray
            Selected indices
        """
        # Information gain (proportional to log variance)
        ig = 0.5 * np.log1p(std**2 / noise_var)
        ig[~available] = -np.inf

        selected = np.argsort(ig)[-self.config.samples_per_round:]
        return selected[::-1]

    def _compute_exploration_weight(self, round_number: int) -> float:
        """Compute exploration weight for current round.

        Parameters
        ----------
        round_number : int
            Current round

        Returns
        -------
        weight : float
            Exploration weight
        """
        if not self.config.adaptive_weight_decay:
            return self.config.exploration_weight

        # Linear decay from initial weight to 0
        progress = round_number / max(self.config.max_rounds - 1, 1)
        weight = self.config.exploration_weight * (1.0 - progress)

        return max(weight, 0.0)

    def _compute_round_metrics(
        self,
        y_observed: np.ndarray,
        y_predicted: Optional[np.ndarray],
        uncertainty: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Compute performance metrics for a round.

        Parameters
        ----------
        y_observed : np.ndarray
            Experimental outcomes
        y_predicted : Optional[np.ndarray]
            Model predictions
        uncertainty : Optional[np.ndarray]
            Uncertainties

        Returns
        -------
        metrics : dict
            Performance metrics
        """
        metrics = {
            "mean_observed": float(np.mean(y_observed)),
            "max_observed": float(np.max(y_observed)),
            "min_observed": float(np.min(y_observed)),
            "std_observed": float(np.std(y_observed)),
        }

        if y_predicted is not None:
            mae = np.mean(np.abs(y_observed - y_predicted))
            rmse = np.sqrt(np.mean((y_observed - y_predicted)**2))
            metrics["mae"] = float(mae)
            metrics["rmse"] = float(rmse)

        if uncertainty is not None:
            metrics["mean_uncertainty"] = float(np.mean(uncertainty))
            metrics["max_uncertainty"] = float(np.max(uncertainty))

        return metrics

    def get_campaign_summary(self) -> Dict[str, Any]:
        """Get summary of entire experimental campaign.

        Returns
        -------
        summary : dict
            Campaign summary with:
            - total_rounds: number of rounds completed
            - total_experiments: total molecules evaluated
            - best_observed: best experimental outcome
            - exploration_exploitation_ratio: ratio over campaign
            - per_round_metrics: list of metrics for each round
        """
        summary = {
            "total_rounds": len(self.rounds),
            "total_experiments": sum(len(r.selected_indices) for r in self.rounds),
            "per_round_metrics": [r.metrics for r in self.rounds],
        }

        # Best observed outcome
        all_observed = []
        for round_data in self.rounds:
            if round_data.observed_values is not None:
                all_observed.extend(round_data.observed_values.tolist())

        if all_observed:
            summary["best_observed"] = float(np.max(all_observed))
            summary["mean_observed"] = float(np.mean(all_observed))

        # Compute exploration/exploitation ratio
        # (based on uncertainty of selected molecules)
        uncertainties = []
        for round_data in self.rounds:
            if round_data.uncertainty is not None:
                uncertainties.append(np.mean(round_data.uncertainty))

        if uncertainties:
            summary["mean_uncertainty_per_round"] = uncertainties
            # Higher uncertainty = more exploration
            summary["exploration_trend"] = "increasing" if uncertainties[-1] > uncertainties[0] else "decreasing"

        return summary

    def get_observed_data(self) -> Tuple[List[int], List[float]]:
        """Get all observed data across all rounds.

        Returns
        -------
        indices : List[int]
            Indices of all evaluated molecules
        values : List[float]
            Experimental outcomes
        """
        indices = []
        values = []

        for round_data in self.rounds:
            if round_data.observed_values is not None:
                indices.extend(round_data.selected_indices.tolist())
                values.extend(round_data.observed_values.tolist())

        return indices, values
