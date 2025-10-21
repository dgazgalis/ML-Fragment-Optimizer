"""
Bayesian Optimization for Active Learning

This module implements Bayesian optimization for molecular property optimization.
It combines Gaussian process surrogate models with acquisition functions to
intelligently select which molecules to evaluate next.

Mathematical Background
-----------------------
Bayesian optimization iteratively:
1. Fit surrogate model p(y|x, D) on observed data D
2. Compute acquisition function α(x) = f(μ(x), σ(x))
3. Select x* = argmax α(x) to evaluate next
4. Observe y* = f(x*) and update D ← D ∪ {(x*, y*)}

For molecular optimization:
- x: molecular fingerprint or descriptor
- y: property of interest (e.g., binding affinity, solubility)
- Surrogate: Gaussian Process (GP) or Random Forest
- Acquisition: EI, UCB, PI, or Thompson Sampling

Author: Claude
Date: 2025-10-20
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple, Callable, Dict, Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

from .acquisition_functions import (
    AcquisitionFunction,
    AcquisitionConfig,
    AcquisitionType,
    create_acquisition_function,
    batch_acquisition_sequential,
    batch_acquisition_local_penalization,
)


@dataclass
class BayesianOptConfig:
    """Configuration for Bayesian optimization.

    Attributes
    ----------
    acquisition : AcquisitionConfig
        Configuration for acquisition function
    kernel : str
        GP kernel type: 'rbf', 'matern', 'matern52'
    length_scale : float
        Initial length scale for kernel
    length_scale_bounds : tuple
        Bounds for length scale optimization
    noise_level : float
        Initial noise level (observation noise)
    noise_bounds : tuple
        Bounds for noise level optimization
    n_restarts : int
        Number of random restarts for GP hyperparameter optimization
    normalize_y : bool
        Whether to normalize target values
    alpha : float
        Additional nugget for numerical stability
    random_state : Optional[int]
        Random seed for reproducibility
    """
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    kernel: str = "matern52"
    length_scale: float = 1.0
    length_scale_bounds: Tuple[float, float] = (1e-2, 1e2)
    noise_level: float = 0.1
    noise_bounds: Tuple[float, float] = (1e-5, 1e1)
    n_restarts: int = 10
    normalize_y: bool = True
    alpha: float = 1e-6
    random_state: Optional[int] = None


class BayesianOptimizer:
    """Bayesian optimization for molecular property optimization.

    This class provides a complete Bayesian optimization loop:
    1. Fit GP surrogate on observed data
    2. Predict mean and uncertainty for candidates
    3. Evaluate acquisition function
    4. Select best candidate(s) to evaluate
    5. Update with new observations

    Examples
    --------
    >>> # Setup
    >>> config = BayesianOptConfig(
    ...     acquisition=AcquisitionConfig(acq_type=AcquisitionType.EI),
    ...     kernel='matern52'
    ... )
    >>> optimizer = BayesianOptimizer(config)
    >>>
    >>> # Initial observations (e.g., from previous experiments)
    >>> X_observed = np.random.randn(10, 100)  # 10 molecules, 100 features
    >>> y_observed = np.random.rand(10)  # Property values
    >>>
    >>> # Fit surrogate
    >>> optimizer.fit(X_observed, y_observed)
    >>>
    >>> # Select next molecule to evaluate
    >>> X_candidates = np.random.randn(1000, 100)  # 1000 candidates
    >>> next_idx = optimizer.select_next(X_candidates, batch_size=1)
    >>>
    >>> # Evaluate selected molecule and update
    >>> X_new = X_candidates[next_idx]
    >>> y_new = evaluate_molecule(X_new)  # Your evaluation function
    >>> optimizer.update(X_new, y_new)
    """

    def __init__(self, config: BayesianOptConfig):
        """Initialize Bayesian optimizer.

        Parameters
        ----------
        config : BayesianOptConfig
            Configuration for optimizer
        """
        self.config = config
        self.gp: Optional[GaussianProcessRegressor] = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler() if config.normalize_y else None
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.acq_fn = create_acquisition_function(config.acquisition)

        # Build kernel
        self.kernel = self._build_kernel()

    def _build_kernel(self):
        """Build GP kernel from configuration."""
        # Base kernel
        if self.config.kernel == "rbf":
            kernel = RBF(
                length_scale=self.config.length_scale,
                length_scale_bounds=self.config.length_scale_bounds
            )
        elif self.config.kernel in ["matern", "matern32"]:
            kernel = Matern(
                length_scale=self.config.length_scale,
                length_scale_bounds=self.config.length_scale_bounds,
                nu=1.5  # Matern 3/2
            )
        elif self.config.kernel == "matern52":
            kernel = Matern(
                length_scale=self.config.length_scale,
                length_scale_bounds=self.config.length_scale_bounds,
                nu=2.5  # Matern 5/2 (twice differentiable, smoother)
            )
        else:
            raise ValueError(f"Unknown kernel: {self.config.kernel}")

        # Add constant kernel for scaling
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * kernel

        # Add white noise kernel for observation noise
        kernel = kernel + WhiteKernel(
            noise_level=self.config.noise_level,
            noise_level_bounds=self.config.noise_bounds
        )

        return kernel

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimize_hyperparameters: bool = True
    ) -> BayesianOptimizer:
        """Fit Gaussian process on observed data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training features (e.g., molecular fingerprints)
        y : np.ndarray, shape (n_samples,)
            Training targets (e.g., property values)
        optimize_hyperparameters : bool
            Whether to optimize GP hyperparameters

        Returns
        -------
        self : BayesianOptimizer
            Fitted optimizer
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        # Store training data
        self.X_train = X.copy()
        self.y_train = y.copy()

        # Normalize features
        X_scaled = self.scaler_X.fit_transform(X)

        # Normalize targets if requested
        if self.scaler_y is not None:
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            y_scaled = y

        # Create and fit GP
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.config.n_restarts if optimize_hyperparameters else 0,
            alpha=self.config.alpha,
            normalize_y=False,  # We handle normalization ourselves
            random_state=self.config.random_state
        )

        self.gp.fit(X_scaled, y_scaled)

        # Update acquisition function with best observed value
        if self.config.acquisition.minimize:
            f_best = np.min(y)
        else:
            f_best = np.max(y)
        self.acq_fn.set_best_value(f_best)

        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict mean and standard deviation for candidates.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Candidate features
        return_std : bool
            Whether to return standard deviation

        Returns
        -------
        mean : np.ndarray, shape (n_samples,)
            Predicted mean values
        std : np.ndarray, shape (n_samples,)
            Predicted standard deviations (if return_std=True)
        """
        if self.gp is None:
            raise ValueError("Must call fit() before predict()")

        X = np.asarray(X)
        X_scaled = self.scaler_X.transform(X)

        if return_std:
            mean_scaled, std_scaled = self.gp.predict(X_scaled, return_std=True)

            # Denormalize
            if self.scaler_y is not None:
                mean = self.scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
                # Scale std by same factor
                std = std_scaled * self.scaler_y.scale_[0]
            else:
                mean = mean_scaled
                std = std_scaled

            return mean, std
        else:
            mean_scaled = self.gp.predict(X_scaled, return_std=False)

            if self.scaler_y is not None:
                mean = self.scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
            else:
                mean = mean_scaled

            return mean

    def evaluate_acquisition(
        self,
        X: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Evaluate acquisition function for candidates.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Candidate features
        **kwargs
            Additional arguments for acquisition function

        Returns
        -------
        acq_values : np.ndarray, shape (n_samples,)
            Acquisition values (higher is better)
        """
        mean, std = self.predict(X, return_std=True)
        return self.acq_fn.evaluate(mean, std, **kwargs)

    def select_next(
        self,
        X_candidates: np.ndarray,
        batch_size: int = 1,
        batch_strategy: str = "sequential",
        exclude_indices: Optional[np.ndarray] = None,
        constraint_probs: Optional[np.ndarray] = None,
        **acq_kwargs
    ) -> np.ndarray:
        """Select next molecule(s) to evaluate.

        Parameters
        ----------
        X_candidates : np.ndarray, shape (n_candidates, n_features)
            Candidate molecules
        batch_size : int
            Number of molecules to select
        batch_strategy : str
            Batch selection strategy: 'sequential' or 'local_penalization'
        exclude_indices : Optional[np.ndarray]
            Indices to exclude from selection (e.g., already evaluated)
        constraint_probs : Optional[np.ndarray], shape (n_candidates,)
            Constraint satisfaction probabilities (e.g., synthesis feasibility)
        **acq_kwargs
            Additional arguments for acquisition function

        Returns
        -------
        selected_indices : np.ndarray, shape (batch_size,)
            Indices of selected molecules
        """
        if self.gp is None:
            raise ValueError("Must call fit() before select_next()")

        # Predict for all candidates
        mean, std = self.predict(X_candidates, return_std=True)

        # Evaluate acquisition
        acq_values = self.acq_fn.evaluate(mean, std, **acq_kwargs)

        # Apply constraints if provided
        if constraint_probs is not None:
            acq_values = self.acq_fn.apply_constraints(acq_values, constraint_probs)

        # Exclude already evaluated molecules
        if exclude_indices is not None:
            acq_values[exclude_indices] = -np.inf

        # Single molecule selection
        if batch_size == 1:
            return np.array([np.argmax(acq_values)])

        # Batch selection
        if batch_strategy == "sequential":
            return batch_acquisition_sequential(
                self.acq_fn,
                mean,
                std,
                batch_size,
                already_selected=exclude_indices
            )
        elif batch_strategy == "local_penalization":
            return batch_acquisition_local_penalization(
                self.acq_fn,
                mean,
                std,
                batch_size,
                already_selected=exclude_indices
            )
        else:
            raise ValueError(f"Unknown batch strategy: {batch_strategy}")

    def update(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        refit: bool = True
    ) -> BayesianOptimizer:
        """Update optimizer with new observations.

        Parameters
        ----------
        X_new : np.ndarray, shape (n_new, n_features)
            New observations (features)
        y_new : np.ndarray, shape (n_new,)
            New observations (targets)
        refit : bool
            Whether to refit GP with updated data

        Returns
        -------
        self : BayesianOptimizer
            Updated optimizer
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Must call fit() before update()")

        X_new = np.asarray(X_new)
        y_new = np.asarray(y_new).ravel()

        # Ensure 2D
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        # Append to training data
        self.X_train = np.vstack([self.X_train, X_new])
        self.y_train = np.hstack([self.y_train, y_new])

        # Refit if requested
        if refit:
            self.fit(self.X_train, self.y_train, optimize_hyperparameters=False)

        return self

    def get_best_observation(self) -> Tuple[np.ndarray, float]:
        """Get best observed molecule and its value.

        Returns
        -------
        X_best : np.ndarray, shape (n_features,)
            Best observed molecule
        y_best : float
            Best observed value
        """
        if self.y_train is None:
            raise ValueError("No observations available")

        if self.config.acquisition.minimize:
            best_idx = np.argmin(self.y_train)
        else:
            best_idx = np.argmax(self.y_train)

        return self.X_train[best_idx], self.y_train[best_idx]

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get optimized GP hyperparameters.

        Returns
        -------
        hyperparameters : dict
            GP kernel hyperparameters
        """
        if self.gp is None:
            raise ValueError("Must call fit() before get_hyperparameters()")

        return {
            "kernel": str(self.gp.kernel_),
            "log_likelihood": self.gp.log_marginal_likelihood_value_,
        }


class MultiObjectiveBayesianOptimizer:
    """Multi-objective Bayesian optimization.

    For optimizing multiple objectives simultaneously (e.g., maximize binding affinity
    and drug-likeness, minimize toxicity).

    Approaches:
    1. Scalarization: α₁·f₁(x) + α₂·f₂(x) + ...
    2. Pareto front: find non-dominated solutions
    3. Expected Hypervolume Improvement (EHVI)

    Examples
    --------
    >>> # Setup for 2 objectives
    >>> configs = [
    ...     BayesianOptConfig(acquisition=AcquisitionConfig(minimize=False)),  # Maximize affinity
    ...     BayesianOptConfig(acquisition=AcquisitionConfig(minimize=True)),   # Minimize toxicity
    ... ]
    >>> optimizer = MultiObjectiveBayesianOptimizer(configs, weights=[0.7, 0.3])
    >>>
    >>> # Fit on multi-objective data
    >>> X = np.random.randn(20, 100)
    >>> y = np.random.rand(20, 2)  # 2 objectives
    >>> optimizer.fit(X, y)
    >>>
    >>> # Select next molecule
    >>> X_candidates = np.random.randn(1000, 100)
    >>> next_idx = optimizer.select_next(X_candidates)
    """

    def __init__(
        self,
        configs: List[BayesianOptConfig],
        weights: Optional[List[float]] = None,
        scalarization: str = "weighted_sum"
    ):
        """Initialize multi-objective optimizer.

        Parameters
        ----------
        configs : List[BayesianOptConfig]
            Configuration for each objective
        weights : Optional[List[float]]
            Weights for scalarization (must sum to 1)
        scalarization : str
            Scalarization method: 'weighted_sum', 'tchebycheff'
        """
        self.configs = configs
        self.n_objectives = len(configs)
        self.scalarization = scalarization

        # Default: equal weights
        if weights is None:
            weights = [1.0 / self.n_objectives] * self.n_objectives
        else:
            weights = np.asarray(weights)
            if not np.isclose(weights.sum(), 1.0):
                weights = weights / weights.sum()
        self.weights = weights

        # Create optimizer for each objective
        self.optimizers = [BayesianOptimizer(config) for config in configs]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimize_hyperparameters: bool = True
    ) -> MultiObjectiveBayesianOptimizer:
        """Fit GPs for all objectives.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training features
        y : np.ndarray, shape (n_samples, n_objectives)
            Training targets for each objective
        optimize_hyperparameters : bool
            Whether to optimize hyperparameters

        Returns
        -------
        self : MultiObjectiveBayesianOptimizer
            Fitted optimizer
        """
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if y.shape[1] != self.n_objectives:
            raise ValueError(
                f"Expected {self.n_objectives} objectives, got {y.shape[1]}"
            )

        # Fit each objective independently
        for i, optimizer in enumerate(self.optimizers):
            optimizer.fit(X, y[:, i], optimize_hyperparameters)

        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict for all objectives.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Candidates
        return_std : bool
            Whether to return standard deviations

        Returns
        -------
        mean : np.ndarray, shape (n_samples, n_objectives)
            Predicted means
        std : np.ndarray, shape (n_samples, n_objectives)
            Predicted standard deviations (if return_std=True)
        """
        if return_std:
            means = []
            stds = []
            for optimizer in self.optimizers:
                mean, std = optimizer.predict(X, return_std=True)
                means.append(mean)
                stds.append(std)
            return np.column_stack(means), np.column_stack(stds)
        else:
            means = [optimizer.predict(X, return_std=False) for optimizer in self.optimizers]
            return np.column_stack(means)

    def evaluate_acquisition(
        self,
        X: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Evaluate scalarized acquisition function.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Candidates
        **kwargs
            Additional arguments for acquisition

        Returns
        -------
        acq_values : np.ndarray, shape (n_samples,)
            Scalarized acquisition values
        """
        # Get acquisition for each objective
        acq_values_list = []
        for optimizer in self.optimizers:
            acq = optimizer.evaluate_acquisition(X, **kwargs)
            acq_values_list.append(acq)

        acq_values = np.column_stack(acq_values_list)

        # Scalarization
        if self.scalarization == "weighted_sum":
            # Normalize each objective to [0, 1]
            acq_norm = np.zeros_like(acq_values)
            for i in range(self.n_objectives):
                acq_min = acq_values[:, i].min()
                acq_max = acq_values[:, i].max()
                if acq_max > acq_min:
                    acq_norm[:, i] = (acq_values[:, i] - acq_min) / (acq_max - acq_min)
                else:
                    acq_norm[:, i] = 0.0

            # Weighted sum
            return np.dot(acq_norm, self.weights)

        elif self.scalarization == "tchebycheff":
            # Tchebycheff scalarization: minimize max_i w_i * |f_i - f_i^*|
            # For acquisition (to maximize), use negation
            return -np.max(self.weights * (-acq_values), axis=1)

        else:
            raise ValueError(f"Unknown scalarization: {self.scalarization}")

    def select_next(
        self,
        X_candidates: np.ndarray,
        batch_size: int = 1,
        **kwargs
    ) -> np.ndarray:
        """Select next molecule(s) using scalarized acquisition.

        Parameters
        ----------
        X_candidates : np.ndarray, shape (n_candidates, n_features)
            Candidates
        batch_size : int
            Number to select
        **kwargs
            Additional arguments

        Returns
        -------
        selected_indices : np.ndarray, shape (batch_size,)
            Indices of selected molecules
        """
        acq_values = self.evaluate_acquisition(X_candidates, **kwargs)

        if batch_size == 1:
            return np.array([np.argmax(acq_values)])
        else:
            # Greedy batch selection
            selected = []
            available = np.ones(len(acq_values), dtype=bool)

            for _ in range(batch_size):
                acq_values[~available] = -np.inf
                idx = np.argmax(acq_values)
                selected.append(idx)
                available[idx] = False

            return np.array(selected)

    def update(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        refit: bool = True
    ) -> MultiObjectiveBayesianOptimizer:
        """Update with new observations.

        Parameters
        ----------
        X_new : np.ndarray, shape (n_new, n_features)
            New features
        y_new : np.ndarray, shape (n_new, n_objectives)
            New targets
        refit : bool
            Whether to refit

        Returns
        -------
        self : MultiObjectiveBayesianOptimizer
            Updated optimizer
        """
        y_new = np.asarray(y_new)
        if y_new.ndim == 1:
            y_new = y_new.reshape(-1, 1)

        for i, optimizer in enumerate(self.optimizers):
            optimizer.update(X_new, y_new[:, i], refit=refit)

        return self
