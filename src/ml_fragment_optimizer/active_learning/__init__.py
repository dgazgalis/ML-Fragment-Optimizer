"""
Active Learning Module for ML-Fragment-Optimizer

This module provides a comprehensive suite of active learning strategies for
molecular optimization, including:

- Bayesian optimization with Gaussian process surrogates
- Multiple acquisition functions (EI, UCB, PI, Thompson Sampling)
- Diversity-based selection (MaxMin, clustering, sphere exclusion)
- Multi-objective portfolio selection
- Design of experiments (Latin Hypercube, adaptive design)

Typical Workflow
----------------
1. Initial Exploration:
   - Use ExperimentDesigner with LHS for space-filling initial samples
   - Or use diversity selection (MaxMin) for initial library

2. Model Training:
   - Train ADMET predictor on initial data
   - Fit Gaussian process for uncertainty quantification

3. Active Learning Loop:
   - Predict mean and uncertainty for candidate molecules
   - Select next batch using acquisition function or portfolio selector
   - Evaluate selected molecules
   - Update model and repeat

4. Multi-Objective Optimization:
   - Use PortfolioSelector to balance multiple objectives
   - Or use MultiObjectiveBayesianOptimizer for Pareto frontier

Examples
--------
>>> # Example 1: Bayesian Optimization
>>> from ml_fragment_optimizer.active_learning import (
...     BayesianOptimizer,
...     BayesianOptConfig,
...     AcquisitionConfig,
...     AcquisitionType
... )
>>>
>>> config = BayesianOptConfig(
...     acquisition=AcquisitionConfig(acq_type=AcquisitionType.EI)
... )
>>> optimizer = BayesianOptimizer(config)
>>> optimizer.fit(X_train, y_train)
>>> next_indices = optimizer.select_next(X_candidates, batch_size=10)

>>> # Example 2: Diversity Selection
>>> from ml_fragment_optimizer.active_learning import (
...     select_diverse_molecules,
...     DistanceMetric
... )
>>>
>>> selected = select_diverse_molecules(
...     fingerprints,
...     n_select=50,
...     method='maxmin',
...     metric=DistanceMetric.TANIMOTO
... )

>>> # Example 3: Portfolio Selection
>>> from ml_fragment_optimizer.active_learning import (
...     PortfolioSelector,
...     PortfolioConfig,
...     Objective,
...     ObjectiveType
... )
>>>
>>> objectives = [
...     Objective('affinity', predicted_affinity, weight=0.5),
...     Objective('uncertainty', uncertainty, weight=0.3),
... ]
>>> config = PortfolioConfig(batch_size=20)
>>> selector = PortfolioSelector(config)
>>> selected = selector.select(objectives, fingerprints=fingerprints)

>>> # Example 4: Experiment Design
>>> from ml_fragment_optimizer.active_learning import (
...     ExperimentDesigner,
...     ExperimentDesignConfig,
...     DesignStrategy
... )
>>>
>>> config = ExperimentDesignConfig(
...     strategy=DesignStrategy.HYBRID,
...     initial_samples=50,
...     samples_per_round=20
... )
>>> designer = ExperimentDesigner(config)
>>> initial_batch = designer.design_initial_experiments(X_candidates)

Author: Claude
Date: 2025-10-20
License: MIT
"""

from .acquisition_functions import (
    AcquisitionFunction,
    AcquisitionConfig,
    AcquisitionType,
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
    ThompsonSampling,
    create_acquisition_function,
    batch_acquisition_sequential,
    batch_acquisition_local_penalization,
)

from .bayesian_opt import (
    BayesianOptConfig,
    BayesianOptimizer,
    MultiObjectiveBayesianOptimizer,
)

from .diversity_sampler import (
    DistanceMetric,
    DiversityConfig,
    tanimoto_distance,
    dice_distance,
    compute_distance_matrix,
    MaxMinSelector,
    SphereExclusionSelector,
    ClusteringSelector,
    ScaffoldDiversitySelector,
    select_diverse_molecules,
)

from .portfolio_selector import (
    ObjectiveType,
    Objective,
    PortfolioConfig,
    PortfolioSelector,
    UtilityFunction,
    select_portfolio_with_utility,
)

from .experiment_designer import (
    DesignStrategy,
    ExperimentDesignConfig,
    ExperimentalRound,
    ExperimentDesigner,
)

__all__ = [
    # Acquisition functions
    "AcquisitionFunction",
    "AcquisitionConfig",
    "AcquisitionType",
    "ExpectedImprovement",
    "UpperConfidenceBound",
    "ProbabilityOfImprovement",
    "ThompsonSampling",
    "create_acquisition_function",
    "batch_acquisition_sequential",
    "batch_acquisition_local_penalization",

    # Bayesian optimization
    "BayesianOptConfig",
    "BayesianOptimizer",
    "MultiObjectiveBayesianOptimizer",

    # Diversity sampling
    "DistanceMetric",
    "DiversityConfig",
    "tanimoto_distance",
    "dice_distance",
    "compute_distance_matrix",
    "MaxMinSelector",
    "SphereExclusionSelector",
    "ClusteringSelector",
    "ScaffoldDiversitySelector",
    "select_diverse_molecules",

    # Portfolio selection
    "ObjectiveType",
    "Objective",
    "PortfolioConfig",
    "PortfolioSelector",
    "UtilityFunction",
    "select_portfolio_with_utility",

    # Experiment design
    "DesignStrategy",
    "ExperimentDesignConfig",
    "ExperimentalRound",
    "ExperimentDesigner",
]

__version__ = "0.1.0"
