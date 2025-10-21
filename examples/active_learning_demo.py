"""
Active Learning Demo: Complete Fragment Optimization Campaign

This script demonstrates all components of the active learning module:
1. Acquisition functions (EI, UCB, PI, Thompson Sampling)
2. Bayesian optimization (single and multi-objective)
3. Diversity sampling (MaxMin, clustering, sphere exclusion)
4. Portfolio selection (multi-objective batch selection)
5. Experiment design (LHS, adaptive, hybrid)

We simulate a fragment optimization campaign for a hypothetical target protein,
selecting molecules to maximize binding affinity while managing synthesis cost
and ensuring chemical diversity.

Author: Claude
Date: 2025-10-20
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from active_learning import (
    # Acquisition functions
    AcquisitionConfig,
    AcquisitionType,
    ExpectedImprovement,
    UpperConfidenceBound,
    create_acquisition_function,

    # Bayesian optimization
    BayesianOptimizer,
    BayesianOptConfig,
    MultiObjectiveBayesianOptimizer,

    # Diversity sampling
    DistanceMetric,
    select_diverse_molecules,
    MaxMinSelector,
    ClusteringSelector,

    # Portfolio selection
    PortfolioSelector,
    PortfolioConfig,
    Objective,
    ObjectiveType,

    # Experiment design
    ExperimentDesigner,
    ExperimentDesignConfig,
    DesignStrategy,
)


def simulate_molecular_library(n_molecules=5000, n_features=1024, seed=42):
    """Simulate molecular library with fingerprints and properties.

    Parameters
    ----------
    n_molecules : int
        Number of molecules in library
    n_features : int
        Fingerprint dimension
    seed : int
        Random seed

    Returns
    -------
    dict with:
        fingerprints : np.ndarray, shape (n_molecules, n_features)
        true_affinity : np.ndarray, shape (n_molecules,)
        synthesis_cost : np.ndarray, shape (n_molecules,)
        synthesis_feasibility : np.ndarray, shape (n_molecules,)
        descriptors : np.ndarray, shape (n_molecules, 10)
    """
    np.random.seed(seed)

    # Binary fingerprints
    fingerprints = np.random.randint(0, 2, (n_molecules, n_features))

    # Continuous descriptors (for visualization and GP)
    descriptors = np.random.randn(n_molecules, 10)

    # True affinity (unknown to optimizer)
    # Based on first 3 descriptor dimensions: f(x) = x1 + 2*x2 - 0.5*x3 + noise
    true_affinity = (
        descriptors[:, 0] +
        2.0 * descriptors[:, 1] -
        0.5 * descriptors[:, 2] +
        np.random.randn(n_molecules) * 0.5
    )

    # Normalize to [0, 1] for easier interpretation
    true_affinity = (true_affinity - true_affinity.min()) / (true_affinity.max() - true_affinity.min())

    # Synthesis cost (correlated with complexity)
    complexity = np.sum(fingerprints, axis=1) / n_features
    synthesis_cost = 100 + 500 * complexity + np.random.randn(n_molecules) * 50
    synthesis_cost = np.clip(synthesis_cost, 100, 1000)

    # Synthesis feasibility (correlated with cost)
    synthesis_feasibility = 1.0 - (synthesis_cost - 100) / 900
    synthesis_feasibility = np.clip(synthesis_feasibility + np.random.randn(n_molecules) * 0.1, 0, 1)

    return {
        'fingerprints': fingerprints,
        'descriptors': descriptors,
        'true_affinity': true_affinity,
        'synthesis_cost': synthesis_cost,
        'synthesis_feasibility': synthesis_feasibility,
    }


def demo_acquisition_functions(library):
    """Demonstrate different acquisition functions."""
    print("=" * 80)
    print("DEMO 1: Acquisition Functions")
    print("=" * 80)

    # Simulate GP predictions
    n_molecules = len(library['true_affinity'])
    mean = np.random.rand(n_molecules)
    std = np.random.rand(n_molecules) * 0.2
    current_best = 0.7

    # Test each acquisition function
    acq_types = [
        AcquisitionType.EI,
        AcquisitionType.UCB,
        AcquisitionType.PI,
        AcquisitionType.THOMPSON,
    ]

    results = {}

    for acq_type in acq_types:
        config = AcquisitionConfig(acq_type=acq_type, kappa=2.0, xi=0.01)
        acq_fn = create_acquisition_function(config)

        if acq_type != AcquisitionType.THOMPSON:
            acq_fn.set_best_value(current_best)

        if acq_type == AcquisitionType.THOMPSON:
            acq_values = acq_fn.evaluate(mean, std, seed=42)
        else:
            acq_values = acq_fn.evaluate(mean, std)

        best_idx = np.argmax(acq_values)
        results[acq_type.value] = {
            'values': acq_values,
            'best_idx': best_idx,
            'best_mean': mean[best_idx],
            'best_std': std[best_idx],
        }

        print(f"\n{acq_type.value.upper()}:")
        print(f"  Best candidate: index={best_idx}")
        print(f"  Predicted mean: {mean[best_idx]:.3f}")
        print(f"  Predicted std: {std[best_idx]:.3f}")
        print(f"  Acquisition value: {acq_values[best_idx]:.3f}")

    return results


def demo_bayesian_optimization(library):
    """Demonstrate Bayesian optimization."""
    print("\n" + "=" * 80)
    print("DEMO 2: Bayesian Optimization")
    print("=" * 80)

    # Initial random sample
    n_initial = 50
    initial_indices = np.random.choice(len(library['descriptors']), n_initial, replace=False)

    X_initial = library['descriptors'][initial_indices]
    y_initial = library['true_affinity'][initial_indices]

    # Setup optimizer
    config = BayesianOptConfig(
        acquisition=AcquisitionConfig(acq_type=AcquisitionType.EI, xi=0.01),
        kernel='matern52',
        n_restarts=5,
        random_state=42
    )

    optimizer = BayesianOptimizer(config)
    optimizer.fit(X_initial, y_initial)

    print(f"\nInitial observations: {n_initial}")
    print(f"Initial best: {np.max(y_initial):.3f}")

    # Active learning loop
    n_rounds = 5
    batch_size = 10

    all_selected = list(initial_indices)
    best_per_round = [np.max(y_initial)]

    for round_num in range(n_rounds):
        # Select next batch
        available_mask = np.ones(len(library['descriptors']), dtype=bool)
        available_mask[all_selected] = False
        available_indices = np.where(available_mask)[0]

        X_candidates = library['descriptors'][available_indices]

        # Predict
        mean, std = optimizer.predict(X_candidates)

        # Select using acquisition function
        acq_values = optimizer.acq_fn.evaluate(mean, std)

        # Top batch_size
        local_selected = np.argsort(acq_values)[-batch_size:]
        selected_indices = available_indices[local_selected]

        # Evaluate
        X_new = library['descriptors'][selected_indices]
        y_new = library['true_affinity'][selected_indices]

        # Update
        optimizer.update(X_new, y_new)
        all_selected.extend(selected_indices.tolist())
        best_per_round.append(np.max(y_new))

        print(f"\nRound {round_num + 1}:")
        print(f"  Selected {batch_size} molecules")
        print(f"  Best in batch: {np.max(y_new):.3f}")
        print(f"  Mean in batch: {np.mean(y_new):.3f}")
        print(f"  Mean predicted: {mean[local_selected].mean():.3f}")

    # Final results
    X_best, y_best = optimizer.get_best_observation()
    print(f"\nFinal best: {y_best:.3f}")
    print(f"Improvement: {y_best - np.max(y_initial):.3f}")

    return {
        'best_per_round': best_per_round,
        'selected_indices': all_selected,
        'final_best': y_best,
    }


def demo_diversity_sampling(library):
    """Demonstrate diversity-based selection."""
    print("\n" + "=" * 80)
    print("DEMO 3: Diversity Sampling")
    print("=" * 80)

    fingerprints = library['fingerprints']

    # MaxMin selection
    print("\nMaxMin Selection:")
    maxmin_selected = select_diverse_molecules(
        fingerprints,
        n_select=50,
        method='maxmin',
        metric=DistanceMetric.TANIMOTO,
        random_state=42
    )
    print(f"  Selected {len(maxmin_selected)} molecules")

    # Compute diversity
    from active_learning import compute_distance_matrix
    dist_matrix = compute_distance_matrix(
        fingerprints[maxmin_selected],
        metric=DistanceMetric.TANIMOTO
    )
    np.fill_diagonal(dist_matrix, np.inf)
    min_dist = np.min(dist_matrix)
    mean_dist = np.mean(dist_matrix[dist_matrix < np.inf])

    print(f"  Minimum pairwise distance: {min_dist:.3f}")
    print(f"  Mean pairwise distance: {mean_dist:.3f}")

    # K-means clustering
    print("\nK-means Clustering Selection:")
    kmeans_selected = select_diverse_molecules(
        library['descriptors'],
        n_select=50,
        method='kmeans',
        metric=DistanceMetric.EUCLIDEAN,
        n_clusters=50,
        selection_mode='medoid',
        random_state=42
    )
    print(f"  Selected {len(kmeans_selected)} molecules")

    # Random (baseline)
    print("\nRandom Selection (baseline):")
    random_selected = select_diverse_molecules(
        fingerprints,
        n_select=50,
        method='random',
        random_state=42
    )

    # Compare diversity
    random_dist = compute_distance_matrix(
        fingerprints[random_selected],
        metric=DistanceMetric.TANIMOTO
    )
    np.fill_diagonal(random_dist, np.inf)
    random_mean_dist = np.mean(random_dist[random_dist < np.inf])

    print(f"  Mean pairwise distance: {random_mean_dist:.3f}")
    print(f"\nDiversity comparison:")
    print(f"  MaxMin: {mean_dist:.3f}")
    print(f"  Random: {random_mean_dist:.3f}")
    print(f"  Improvement: {(mean_dist / random_mean_dist - 1) * 100:.1f}%")

    return {
        'maxmin': maxmin_selected,
        'kmeans': kmeans_selected,
        'random': random_selected,
    }


def demo_portfolio_selection(library):
    """Demonstrate portfolio selection."""
    print("\n" + "=" * 80)
    print("DEMO 4: Portfolio Selection")
    print("=" * 80)

    # Simulate model predictions
    predicted_affinity = library['true_affinity'] + np.random.randn(len(library['true_affinity'])) * 0.1
    uncertainty = np.random.rand(len(library['true_affinity'])) * 0.2

    # Define objectives
    objectives = [
        Objective(
            name='affinity',
            values=predicted_affinity,
            weight=0.5,
            obj_type=ObjectiveType.MAXIMIZE
        ),
        Objective(
            name='uncertainty',
            values=uncertainty,
            weight=0.3,
            obj_type=ObjectiveType.MAXIMIZE  # Explore uncertain regions
        ),
    ]

    # Configure portfolio
    config = PortfolioConfig(
        batch_size=30,
        budget=15000,  # Maximum synthesis cost
        synthesis_threshold=0.4,  # Minimum feasibility
        strategy='greedy',
        diversity_penalty=0.5,
        diversity_metric=DistanceMetric.TANIMOTO
    )

    # Select portfolio
    selector = PortfolioSelector(config)
    selected = selector.select(
        objectives=objectives,
        fingerprints=library['fingerprints'],
        costs=library['synthesis_cost'],
        synthesis_scores=library['synthesis_feasibility']
    )

    print(f"\nSelected {len(selected)} molecules")

    # Compute metrics
    metrics = selector.compute_portfolio_metrics(
        selected,
        objectives,
        library['fingerprints'],
        library['synthesis_cost']
    )

    print(f"\nPortfolio Metrics:")
    print(f"  Mean affinity: {metrics['mean_affinity']:.3f}")
    print(f"  Max affinity: {metrics['max_affinity']:.3f}")
    print(f"  Mean uncertainty: {metrics['mean_uncertainty']:.3f}")
    print(f"  Total cost: ${metrics['total_cost']:.0f}")
    print(f"  Mean cost: ${metrics['mean_cost']:.0f}")
    print(f"  Mean diversity: {metrics['mean_diversity']:.3f}")
    print(f"  Min diversity: {metrics['min_diversity']:.3f}")

    # Check constraints
    print(f"\nConstraint Satisfaction:")
    print(f"  Budget: ${metrics['total_cost']:.0f} / ${config.budget:.0f}")
    print(f"  Synthesis feasibility: all >= {config.synthesis_threshold}")

    return {
        'selected': selected,
        'metrics': metrics,
    }


def demo_experiment_design(library):
    """Demonstrate design of experiments."""
    print("\n" + "=" * 80)
    print("DEMO 5: Experiment Design")
    print("=" * 80)

    # Configure experiment design
    config = ExperimentDesignConfig(
        strategy=DesignStrategy.HYBRID,
        initial_samples=50,
        samples_per_round=20,
        max_rounds=5,
        exploration_weight=0.5,
        adaptive_weight_decay=True,
        random_state=42
    )

    designer = ExperimentDesigner(config)

    # Initial exploration (LHS)
    print("\nRound 0: Initial Exploration (LHS)")
    initial_batch = designer.design_initial_experiments(library['descriptors'])
    y_initial = library['true_affinity'][initial_batch]
    designer.update(initial_batch, y_initial)

    print(f"  Selected {len(initial_batch)} molecules")
    print(f"  Mean affinity: {np.mean(y_initial):.3f}")
    print(f"  Best affinity: {np.max(y_initial):.3f}")

    # Setup GP for subsequent rounds
    bo_config = BayesianOptConfig(
        acquisition=AcquisitionConfig(acq_type=AcquisitionType.UCB, kappa=2.0),
        kernel='matern52',
        random_state=42
    )
    optimizer = BayesianOptimizer(bo_config)

    # Adaptive rounds
    for round_num in range(1, config.max_rounds):
        print(f"\nRound {round_num}: Adaptive Design")

        # Train GP
        indices, y_obs = designer.get_observed_data()
        X_obs = library['descriptors'][indices]
        optimizer.fit(X_obs, np.array(y_obs))

        # Predict for all candidates
        mean, std = optimizer.predict(library['descriptors'])

        # Design next round
        exclude = designer.selected_indices_all
        next_batch = designer.design_next_round(
            library['descriptors'],
            mean=mean,
            std=std,
            round_number=round_num,
            exclude_indices=exclude
        )

        # Evaluate
        y_new = library['true_affinity'][next_batch]
        designer.update(next_batch, y_new, mean[next_batch], std[next_batch])

        print(f"  Selected {len(next_batch)} molecules")
        print(f"  Mean affinity: {np.mean(y_new):.3f}")
        print(f"  Best in batch: {np.max(y_new):.3f}")
        print(f"  Mean predicted: {mean[next_batch].mean():.3f}")
        print(f"  Mean uncertainty: {std[next_batch].mean():.3f}")

    # Campaign summary
    summary = designer.get_campaign_summary()
    print(f"\nCampaign Summary:")
    print(f"  Total rounds: {summary['total_rounds']}")
    print(f"  Total experiments: {summary['total_experiments']}")
    print(f"  Best observed: {summary['best_observed']:.3f}")
    print(f"  Mean observed: {summary['mean_observed']:.3f}")
    print(f"  Exploration trend: {summary['exploration_trend']}")

    return {
        'designer': designer,
        'summary': summary,
    }


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("ACTIVE LEARNING MODULE - COMPREHENSIVE DEMO")
    print("=" * 80)
    print("\nSimulating fragment optimization campaign with 5000 molecules")

    # Generate simulated library
    library = simulate_molecular_library(n_molecules=5000, n_features=1024)

    print(f"\nLibrary Statistics:")
    print(f"  Molecules: {len(library['true_affinity'])}")
    print(f"  Fingerprint dimension: {library['fingerprints'].shape[1]}")
    print(f"  Affinity range: [{library['true_affinity'].min():.3f}, {library['true_affinity'].max():.3f}]")
    print(f"  Cost range: [${library['synthesis_cost'].min():.0f}, ${library['synthesis_cost'].max():.0f}]")

    # Run demos
    acq_results = demo_acquisition_functions(library)
    bo_results = demo_bayesian_optimization(library)
    diversity_results = demo_diversity_sampling(library)
    portfolio_results = demo_portfolio_selection(library)
    design_results = demo_experiment_design(library)

    # Final summary
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Acquisition functions guide molecule selection (EI, UCB, PI, Thompson)")
    print("2. Bayesian optimization balances exploitation and exploration")
    print("3. Diversity sampling ensures broad chemical space coverage")
    print("4. Portfolio selection balances multiple objectives and constraints")
    print("5. Experiment design provides systematic multi-round campaigns")

    print("\nFor production use:")
    print("- Replace simulated library with real molecular data")
    print("- Integrate with ADMET prediction models")
    print("- Connect to synthesis planning tools")
    print("- Use RDKit for scaffold analysis")
    print("- Visualize results with molecular structures")


if __name__ == "__main__":
    main()
