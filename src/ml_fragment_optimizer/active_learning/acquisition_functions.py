"""
Acquisition Functions for Bayesian Optimization

This module provides acquisition functions for active learning in molecular optimization.
Acquisition functions balance exploitation (selecting molecules with high predicted values)
and exploration (selecting molecules with high uncertainty).

Mathematical Background
-----------------------
Given a surrogate model p(y|x, D) trained on dataset D, acquisition functions α(x)
quantify the utility of evaluating a candidate molecule x.

Common acquisition functions:
1. Expected Improvement (EI):
   α_EI(x) = E[max(0, f(x) - f_best)] = (μ(x) - f_best)Φ(Z) + σ(x)φ(Z)
   where Z = (μ(x) - f_best) / σ(x), Φ is standard normal CDF, φ is PDF

2. Upper Confidence Bound (UCB):
   α_UCB(x) = μ(x) + κ·σ(x)
   where κ controls exploration-exploitation tradeoff

3. Probability of Improvement (PI):
   α_PI(x) = P(f(x) > f_best + ξ) = Φ((μ(x) - f_best - ξ) / σ(x))
   where ξ is jitter parameter for exploration

4. Thompson Sampling:
   Sample function from posterior: f_sample ~ p(f|D)
   α_TS(x) = f_sample(x)

Author: Claude
Date: 2025-10-20
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, Tuple, Callable

import numpy as np
from scipy import stats
from scipy.special import ndtr, ndtri  # Faster than stats.norm.cdf/ppf


class AcquisitionType(Enum):
    """Supported acquisition function types."""
    EI = "expected_improvement"
    UCB = "upper_confidence_bound"
    PI = "probability_improvement"
    THOMPSON = "thompson_sampling"
    GREEDY = "greedy"  # Pure exploitation: μ(x)
    UNCERTAINTY = "uncertainty"  # Pure exploration: σ(x)


@dataclass
class AcquisitionConfig:
    """Configuration for acquisition functions.

    Attributes
    ----------
    acq_type : AcquisitionType
        Type of acquisition function to use
    kappa : float
        Exploration parameter for UCB (typical: 1.96 for 95% CI, 2.576 for 99%)
    xi : float
        Jitter parameter for EI and PI (typical: 0.01)
    minimize : bool
        Whether to minimize (True) or maximize (False) objective
    batch_size : int
        Number of molecules to select in batch mode
    batch_strategy : str
        Strategy for batch selection: 'sequential', 'hallucination', 'local_penalization'
    multi_objective : bool
        Whether to handle multi-objective predictions
    constraint_threshold : Optional[float]
        Threshold for constraint satisfaction (e.g., synthesis feasibility > 0.5)
    constraint_weight : float
        Weight for constraint violation penalty
    """
    acq_type: AcquisitionType = AcquisitionType.EI
    kappa: float = 2.0
    xi: float = 0.01
    minimize: bool = False
    batch_size: int = 1
    batch_strategy: str = "sequential"
    multi_objective: bool = False
    constraint_threshold: Optional[float] = None
    constraint_weight: float = 1000.0


class AcquisitionFunction(ABC):
    """Abstract base class for acquisition functions."""

    def __init__(self, config: AcquisitionConfig):
        """Initialize acquisition function.

        Parameters
        ----------
        config : AcquisitionConfig
            Configuration parameters
        """
        self.config = config
        self.f_best: Optional[float] = None

    def set_best_value(self, f_best: float) -> None:
        """Set current best observed value.

        Parameters
        ----------
        f_best : float
            Best observed objective value
        """
        self.f_best = f_best

    @abstractmethod
    def evaluate(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Evaluate acquisition function.

        Parameters
        ----------
        mean : np.ndarray, shape (n_samples,) or (n_samples, n_objectives)
            Predicted mean values
        std : np.ndarray, shape (n_samples,) or (n_samples, n_objectives)
            Predicted standard deviations
        **kwargs
            Additional arguments for specific acquisition functions

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Acquisition values (higher is better)
        """
        pass

    def __call__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Callable interface for acquisition function."""
        return self.evaluate(mean, std, **kwargs)

    def apply_constraints(
        self,
        acq_values: np.ndarray,
        constraint_probs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply constraint penalties to acquisition values.

        For constrained optimization, we penalize molecules that are unlikely
        to satisfy constraints (e.g., synthesis feasibility < threshold).

        Parameters
        ----------
        acq_values : np.ndarray, shape (n_samples,)
            Raw acquisition values
        constraint_probs : Optional[np.ndarray], shape (n_samples,)
            Predicted probability of constraint satisfaction

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Constrained acquisition values
        """
        if constraint_probs is None or self.config.constraint_threshold is None:
            return acq_values

        # Penalty for violating constraints
        violations = np.maximum(0, self.config.constraint_threshold - constraint_probs)
        penalty = self.config.constraint_weight * violations

        return acq_values - penalty


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function.

    Mathematical Formulation
    ------------------------
    For maximization:
        EI(x) = E[max(0, f(x) - f_best)]
              = (μ(x) - f_best - ξ)·Φ(Z) + σ(x)·φ(Z)

    where:
        Z = (μ(x) - f_best - ξ) / σ(x)
        Φ(·) is the standard normal CDF
        φ(·) is the standard normal PDF
        ξ is jitter parameter for exploration (default: 0.01)

    For minimization: replace f_best - μ(x) for μ(x) - f_best

    Properties
    ----------
    - Balances exploitation (high mean) and exploration (high uncertainty)
    - Analytical gradients available
    - Works well for noisy objectives
    - ξ parameter prevents premature convergence

    Examples
    --------
    >>> config = AcquisitionConfig(acq_type=AcquisitionType.EI, xi=0.01)
    >>> ei = ExpectedImprovement(config)
    >>> ei.set_best_value(0.8)
    >>> mean = np.array([0.7, 0.85, 0.75])
    >>> std = np.array([0.1, 0.05, 0.15])
    >>> acq_vals = ei.evaluate(mean, std)
    >>> best_idx = np.argmax(acq_vals)
    """

    def evaluate(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Evaluate Expected Improvement.

        Parameters
        ----------
        mean : np.ndarray, shape (n_samples,)
            Predicted mean values
        std : np.ndarray, shape (n_samples,)
            Predicted standard deviations

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Expected improvement values
        """
        if self.f_best is None:
            raise ValueError("Must call set_best_value() before evaluating EI")

        # Convert to numpy arrays
        mean = np.asarray(mean).ravel()
        std = np.asarray(std).ravel()

        # Handle minimization vs maximization
        if self.config.minimize:
            improvement = self.f_best - mean - self.config.xi
        else:
            improvement = mean - self.f_best - self.config.xi

        # Prevent division by zero
        std_safe = np.maximum(std, 1e-9)

        # Standardized improvement
        z = improvement / std_safe

        # Expected improvement formula
        # EI = improvement * Φ(z) + std * φ(z)
        ei = improvement * ndtr(z) + std_safe * stats.norm.pdf(z)

        # Zero EI where std is zero (deterministic predictions)
        ei = np.where(std > 1e-9, ei, 0.0)

        return ei

    def evaluate_with_gradients(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        mean_grad: np.ndarray,
        std_grad: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate EI and its gradients (for gradient-based optimization).

        Parameters
        ----------
        mean : np.ndarray, shape (n_samples,)
            Predicted mean values
        std : np.ndarray, shape (n_samples,)
            Predicted standard deviations
        mean_grad : np.ndarray, shape (n_samples, n_features)
            Gradients of mean w.r.t. input features
        std_grad : np.ndarray, shape (n_samples, n_features)
            Gradients of std w.r.t. input features

        Returns
        -------
        ei : np.ndarray, shape (n_samples,)
            Expected improvement values
        ei_grad : np.ndarray, shape (n_samples, n_features)
            Gradients of EI w.r.t. input features
        """
        mean = np.asarray(mean).ravel()
        std = np.asarray(std).ravel()

        if self.config.minimize:
            improvement = self.f_best - mean - self.config.xi
            sign = -1.0
        else:
            improvement = mean - self.f_best - self.config.xi
            sign = 1.0

        std_safe = np.maximum(std, 1e-9)
        z = improvement / std_safe

        # EI value
        phi_z = stats.norm.pdf(z)
        Phi_z = ndtr(z)
        ei = improvement * Phi_z + std_safe * phi_z
        ei = np.where(std > 1e-9, ei, 0.0)

        # EI gradient
        # d(EI)/d(mean) = sign * Φ(z)
        # d(EI)/d(std) = φ(z)
        dei_dmean = sign * Phi_z
        dei_dstd = phi_z

        # Chain rule
        ei_grad = (dei_dmean[:, None] * mean_grad +
                   dei_dstd[:, None] * std_grad)

        return ei, ei_grad


class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound acquisition function.

    Mathematical Formulation
    ------------------------
    For maximization:
        UCB(x) = μ(x) + κ·σ(x)

    For minimization:
        UCB(x) = μ(x) - κ·σ(x)

    where:
        μ(x) is the predicted mean
        σ(x) is the predicted standard deviation
        κ controls exploration-exploitation tradeoff

    Common κ values:
        κ = 1.96: 95% confidence interval
        κ = 2.576: 99% confidence interval
        κ = √(2·log(n)): GP-UCB with theoretical guarantees

    Properties
    ----------
    - Simple and interpretable
    - Direct control of exploration via κ
    - No dependence on f_best (useful early in optimization)
    - Can adapt κ over time (high early, low later)

    Examples
    --------
    >>> config = AcquisitionConfig(acq_type=AcquisitionType.UCB, kappa=2.0)
    >>> ucb = UpperConfidenceBound(config)
    >>> mean = np.array([0.7, 0.85, 0.75])
    >>> std = np.array([0.1, 0.05, 0.15])
    >>> acq_vals = ucb.evaluate(mean, std)
    """

    def evaluate(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Evaluate Upper Confidence Bound.

        Parameters
        ----------
        mean : np.ndarray, shape (n_samples,)
            Predicted mean values
        std : np.ndarray, shape (n_samples,)
            Predicted standard deviations

        Returns
        -------
        np.ndarray, shape (n_samples,)
            UCB acquisition values
        """
        mean = np.asarray(mean).ravel()
        std = np.asarray(std).ravel()

        if self.config.minimize:
            return mean - self.config.kappa * std
        else:
            return mean + self.config.kappa * std

    def set_adaptive_kappa(self, iteration: int, total_iterations: int) -> None:
        """Set κ adaptively based on iteration number.

        Early iterations: high κ (exploration)
        Late iterations: low κ (exploitation)

        Parameters
        ----------
        iteration : int
            Current iteration (0-indexed)
        total_iterations : int
            Total number of planned iterations
        """
        # Linear decay from initial kappa to kappa/2
        progress = iteration / max(total_iterations - 1, 1)
        initial_kappa = self.config.kappa
        final_kappa = initial_kappa / 2.0
        self.config.kappa = initial_kappa - progress * (initial_kappa - final_kappa)


class ProbabilityOfImprovement(AcquisitionFunction):
    """Probability of Improvement acquisition function.

    Mathematical Formulation
    ------------------------
    For maximization:
        PI(x) = P(f(x) > f_best + ξ) = Φ((μ(x) - f_best - ξ) / σ(x))

    For minimization:
        PI(x) = P(f(x) < f_best - ξ) = Φ((f_best - μ(x) - ξ) / σ(x))

    where:
        Φ(·) is the standard normal CDF
        ξ is jitter parameter (default: 0.01)

    Properties
    ----------
    - Focuses on probability rather than magnitude of improvement
    - More greedy than EI (less exploration)
    - Useful when any improvement is valuable
    - ξ parameter prevents over-exploitation

    Examples
    --------
    >>> config = AcquisitionConfig(acq_type=AcquisitionType.PI, xi=0.01)
    >>> pi = ProbabilityOfImprovement(config)
    >>> pi.set_best_value(0.8)
    >>> mean = np.array([0.7, 0.85, 0.75])
    >>> std = np.array([0.1, 0.05, 0.15])
    >>> acq_vals = pi.evaluate(mean, std)
    """

    def evaluate(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Evaluate Probability of Improvement.

        Parameters
        ----------
        mean : np.ndarray, shape (n_samples,)
            Predicted mean values
        std : np.ndarray, shape (n_samples,)
            Predicted standard deviations

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Probability of improvement values
        """
        if self.f_best is None:
            raise ValueError("Must call set_best_value() before evaluating PI")

        mean = np.asarray(mean).ravel()
        std = np.asarray(std).ravel()

        # Handle minimization vs maximization
        if self.config.minimize:
            improvement = self.f_best - mean - self.config.xi
        else:
            improvement = mean - self.f_best - self.config.xi

        # Prevent division by zero
        std_safe = np.maximum(std, 1e-9)

        # Probability of improvement
        z = improvement / std_safe
        pi = ndtr(z)

        return pi


class ThompsonSampling(AcquisitionFunction):
    """Thompson Sampling acquisition function.

    Mathematical Formulation
    ------------------------
    Sample a function from the posterior:
        f_sample ~ p(f | D)

    Then select:
        x* = argmax f_sample(x)

    For Gaussian processes:
        f_sample(x) ~ N(μ(x), σ²(x))

    Properties
    ----------
    - Naturally balances exploration and exploitation
    - Randomized (different samples each time)
    - Theoretical optimality guarantees
    - Works well for multi-armed bandits and BO

    Examples
    --------
    >>> config = AcquisitionConfig(acq_type=AcquisitionType.THOMPSON)
    >>> ts = ThompsonSampling(config)
    >>> mean = np.array([0.7, 0.85, 0.75])
    >>> std = np.array([0.1, 0.05, 0.15])
    >>> acq_vals = ts.evaluate(mean, std, seed=42)
    """

    def evaluate(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Evaluate Thompson Sampling (sample from posterior).

        Parameters
        ----------
        mean : np.ndarray, shape (n_samples,)
            Predicted mean values
        std : np.ndarray, shape (n_samples,)
            Predicted standard deviations
        seed : Optional[int]
            Random seed for reproducibility

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Sampled function values
        """
        mean = np.asarray(mean).ravel()
        std = np.asarray(std).ravel()

        if seed is not None:
            np.random.seed(seed)

        # Sample from posterior: f ~ N(μ, σ²)
        samples = np.random.normal(mean, std)

        # For minimization, negate samples
        if self.config.minimize:
            samples = -samples

        return samples


def create_acquisition_function(
    config: AcquisitionConfig
) -> AcquisitionFunction:
    """Factory function to create acquisition functions.

    Parameters
    ----------
    config : AcquisitionConfig
        Configuration for acquisition function

    Returns
    -------
    AcquisitionFunction
        Instantiated acquisition function

    Examples
    --------
    >>> config = AcquisitionConfig(acq_type=AcquisitionType.EI)
    >>> acq_fn = create_acquisition_function(config)
    >>> isinstance(acq_fn, ExpectedImprovement)
    True
    """
    if config.acq_type == AcquisitionType.EI:
        return ExpectedImprovement(config)
    elif config.acq_type == AcquisitionType.UCB:
        return UpperConfidenceBound(config)
    elif config.acq_type == AcquisitionType.PI:
        return ProbabilityOfImprovement(config)
    elif config.acq_type == AcquisitionType.THOMPSON:
        return ThompsonSampling(config)
    elif config.acq_type == AcquisitionType.GREEDY:
        # Pure exploitation: return mean (negated for minimization)
        class GreedyAcquisition(AcquisitionFunction):
            def evaluate(self, mean, std, **kwargs):
                mean = np.asarray(mean).ravel()
                return -mean if self.config.minimize else mean
        return GreedyAcquisition(config)
    elif config.acq_type == AcquisitionType.UNCERTAINTY:
        # Pure exploration: return std
        class UncertaintyAcquisition(AcquisitionFunction):
            def evaluate(self, mean, std, **kwargs):
                return np.asarray(std).ravel()
        return UncertaintyAcquisition(config)
    else:
        raise ValueError(f"Unknown acquisition type: {config.acq_type}")


def batch_acquisition_sequential(
    acq_fn: AcquisitionFunction,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    already_selected: Optional[np.ndarray] = None
) -> np.ndarray:
    """Select batch of molecules sequentially (greedy).

    Algorithm
    ---------
    1. Select x1 with highest acquisition value
    2. Assume x1 will have outcome = μ(x1) (hallucination)
    3. Update f_best if necessary
    4. Select x2 with highest acquisition value given x1
    5. Repeat until batch_size molecules selected

    Parameters
    ----------
    acq_fn : AcquisitionFunction
        Acquisition function to use
    mean : np.ndarray, shape (n_candidates,)
        Predicted means for all candidates
    std : np.ndarray, shape (n_candidates,)
        Predicted standard deviations
    batch_size : int
        Number of molecules to select
    already_selected : Optional[np.ndarray]
        Indices of already selected molecules (to exclude)

    Returns
    -------
    np.ndarray, shape (batch_size,)
        Indices of selected molecules
    """
    n_candidates = len(mean)
    selected = []
    available = np.ones(n_candidates, dtype=bool)

    if already_selected is not None:
        available[already_selected] = False

    for _ in range(batch_size):
        # Evaluate acquisition on available candidates
        acq_vals = acq_fn.evaluate(mean, std)
        acq_vals[~available] = -np.inf

        # Select best available
        idx = np.argmax(acq_vals)
        selected.append(idx)
        available[idx] = False

        # Update f_best with hallucinated outcome
        if acq_fn.f_best is None:
            acq_fn.set_best_value(mean[idx])
        else:
            if acq_fn.config.minimize:
                acq_fn.set_best_value(min(acq_fn.f_best, mean[idx]))
            else:
                acq_fn.set_best_value(max(acq_fn.f_best, mean[idx]))

    return np.array(selected)


def batch_acquisition_local_penalization(
    acq_fn: AcquisitionFunction,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    lipschitz_constant: float = 10.0,
    already_selected: Optional[np.ndarray] = None
) -> np.ndarray:
    """Select batch using local penalization (discourage nearby selections).

    Algorithm
    ---------
    1. Select x1 with highest acquisition value
    2. Penalize acquisition near x1 by factor r = 1 - (1 - d/L)^s
       where d is distance to x1, L is Lipschitz constant, s is slope
    3. Select x2 with highest penalized acquisition
    4. Repeat until batch_size molecules selected

    Note: This implementation assumes distance is already encoded in std
    (molecules far from training data have high std). For explicit distance-based
    penalization, pass a distance matrix.

    Parameters
    ----------
    acq_fn : AcquisitionFunction
        Acquisition function to use
    mean : np.ndarray, shape (n_candidates,)
        Predicted means for all candidates
    std : np.ndarray, shape (n_candidates,)
        Predicted standard deviations
    batch_size : int
        Number of molecules to select
    lipschitz_constant : float
        Controls radius of penalization
    already_selected : Optional[np.ndarray]
        Indices of already selected molecules

    Returns
    -------
    np.ndarray, shape (batch_size,)
        Indices of selected molecules
    """
    n_candidates = len(mean)
    selected = []
    available = np.ones(n_candidates, dtype=bool)
    penalty = np.ones(n_candidates)

    if already_selected is not None:
        available[already_selected] = False
        penalty[already_selected] = 0.0

    for _ in range(batch_size):
        # Evaluate acquisition with penalty
        acq_vals = acq_fn.evaluate(mean, std) * penalty
        acq_vals[~available] = -np.inf

        # Select best available
        idx = np.argmax(acq_vals)
        selected.append(idx)
        available[idx] = False

        # Apply local penalization (simplified: penalize by uncertainty radius)
        # In practice, you'd use actual distance matrix
        distances = np.abs(mean - mean[idx]) / (std[idx] + 1e-9)
        local_penalty = np.clip(distances / lipschitz_constant, 0, 1)
        penalty *= local_penalty

    return np.array(selected)
