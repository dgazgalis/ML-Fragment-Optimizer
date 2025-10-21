"""
Unit tests for Bayesian optimization.

Author: Claude
Date: 2025-10-20
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ml_fragment_optimizer.active_learning.bayesian_opt import (
    BayesianOptConfig,
    BayesianOptimizer,
    MultiObjectiveBayesianOptimizer,
)
from ml_fragment_optimizer.active_learning.acquisition_functions import AcquisitionConfig, AcquisitionType


class TestBayesianOptimizer:
    """Test Bayesian optimizer."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        X = np.random.randn(20, 10)
        y = np.sum(X[:, :3], axis=1) + np.random.randn(20) * 0.1
        return X, y

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        config = BayesianOptConfig(
            acquisition=AcquisitionConfig(acq_type=AcquisitionType.EI),
            kernel="matern52",
            random_state=42
        )
        return BayesianOptimizer(config)

    def test_fit(self, optimizer, sample_data):
        """Test fitting GP."""
        X, y = sample_data
        optimizer.fit(X, y)

        assert optimizer.gp is not None
        assert optimizer.X_train is not None
        assert optimizer.y_train is not None
        assert len(optimizer.y_train) == len(y)

    def test_predict(self, optimizer, sample_data):
        """Test prediction."""
        X_train, y_train = sample_data
        optimizer.fit(X_train, y_train)

        X_test = np.random.randn(10, 10)
        mean, std = optimizer.predict(X_test)

        assert mean.shape == (10,)
        assert std.shape == (10,)
        assert np.all(std >= 0)

    def test_select_next(self, optimizer, sample_data):
        """Test selecting next molecule."""
        X_train, y_train = sample_data
        optimizer.fit(X_train, y_train)

        X_candidates = np.random.randn(100, 10)
        selected = optimizer.select_next(X_candidates, batch_size=5)

        assert len(selected) == 5
        assert len(np.unique(selected)) == 5

    def test_update(self, optimizer, sample_data):
        """Test updating with new data."""
        X_train, y_train = sample_data
        optimizer.fit(X_train, y_train)

        X_new = np.random.randn(3, 10)
        y_new = np.random.randn(3)

        initial_size = len(optimizer.y_train)
        optimizer.update(X_new, y_new)

        assert len(optimizer.y_train) == initial_size + 3

    def test_get_best_observation(self, optimizer, sample_data):
        """Test getting best observation."""
        X, y = sample_data
        optimizer.fit(X, y)

        X_best, y_best = optimizer.get_best_observation()

        assert X_best.shape == (10,)
        assert y_best == np.max(y)

    def test_minimization(self, sample_data):
        """Test minimization."""
        config = BayesianOptConfig(
            acquisition=AcquisitionConfig(
                acq_type=AcquisitionType.EI,
                minimize=True
            )
        )
        optimizer = BayesianOptimizer(config)

        X, y = sample_data
        optimizer.fit(X, y)

        X_best, y_best = optimizer.get_best_observation()

        assert y_best == np.min(y)

    def test_constraint_selection(self, optimizer, sample_data):
        """Test selection with constraints."""
        X_train, y_train = sample_data
        optimizer.fit(X_train, y_train)

        X_candidates = np.random.randn(100, 10)
        constraint_probs = np.random.rand(100)

        selected = optimizer.select_next(
            X_candidates,
            batch_size=5,
            constraint_probs=constraint_probs
        )

        assert len(selected) == 5


class TestMultiObjectiveBayesianOptimizer:
    """Test multi-objective Bayesian optimizer."""

    @pytest.fixture
    def sample_data_multi(self):
        """Generate multi-objective training data."""
        np.random.seed(42)
        X = np.random.randn(20, 10)
        y1 = np.sum(X[:, :3], axis=1) + np.random.randn(20) * 0.1
        y2 = -np.sum(X[:, 3:6], axis=1) + np.random.randn(20) * 0.1
        y = np.column_stack([y1, y2])
        return X, y

    @pytest.fixture
    def multi_optimizer(self):
        """Create multi-objective optimizer."""
        configs = [
            BayesianOptConfig(
                acquisition=AcquisitionConfig(minimize=False),
                random_state=42
            ),
            BayesianOptConfig(
                acquisition=AcquisitionConfig(minimize=True),
                random_state=42
            ),
        ]
        return MultiObjectiveBayesianOptimizer(configs, weights=[0.6, 0.4])

    def test_fit_multi(self, multi_optimizer, sample_data_multi):
        """Test fitting multi-objective GP."""
        X, y = sample_data_multi
        multi_optimizer.fit(X, y)

        assert len(multi_optimizer.optimizers) == 2
        assert all(opt.gp is not None for opt in multi_optimizer.optimizers)

    def test_predict_multi(self, multi_optimizer, sample_data_multi):
        """Test multi-objective prediction."""
        X_train, y_train = sample_data_multi
        multi_optimizer.fit(X_train, y_train)

        X_test = np.random.randn(10, 10)
        mean, std = multi_optimizer.predict(X_test)

        assert mean.shape == (10, 2)
        assert std.shape == (10, 2)
        assert np.all(std >= 0)

    def test_select_next_multi(self, multi_optimizer, sample_data_multi):
        """Test multi-objective selection."""
        X_train, y_train = sample_data_multi
        multi_optimizer.fit(X_train, y_train)

        X_candidates = np.random.randn(100, 10)
        selected = multi_optimizer.select_next(X_candidates, batch_size=5)

        assert len(selected) == 5

    def test_scalarization_methods(self, sample_data_multi):
        """Test different scalarization methods."""
        X, y = sample_data_multi

        # Weighted sum
        configs = [BayesianOptConfig(random_state=42)] * 2
        opt_weighted = MultiObjectiveBayesianOptimizer(
            configs, scalarization="weighted_sum"
        )
        opt_weighted.fit(X, y)

        # Tchebycheff
        opt_tcheby = MultiObjectiveBayesianOptimizer(
            configs, scalarization="tchebycheff"
        )
        opt_tcheby.fit(X, y)

        X_candidates = np.random.randn(100, 10)

        selected_weighted = opt_weighted.select_next(X_candidates, batch_size=5)
        selected_tcheby = opt_tcheby.select_next(X_candidates, batch_size=5)

        # Both should return 5 selections
        assert len(selected_weighted) == 5
        assert len(selected_tcheby) == 5


class TestBayesianOptIntegration:
    """Integration tests for Bayesian optimization."""

    def test_full_optimization_loop(self):
        """Test complete optimization loop."""
        # Setup
        np.random.seed(42)

        # True function: maximize x1 + 2*x2 - x3
        def objective(X):
            return X[:, 0] + 2*X[:, 1] - X[:, 2]

        # Initial data
        X_initial = np.random.randn(10, 5)
        y_initial = objective(X_initial)

        # Optimizer
        config = BayesianOptConfig(
            acquisition=AcquisitionConfig(acq_type=AcquisitionType.UCB, kappa=2.0),
            random_state=42
        )
        optimizer = BayesianOptimizer(config)
        optimizer.fit(X_initial, y_initial)

        # Candidates
        X_candidates = np.random.randn(1000, 5)

        # Optimization loop
        for _ in range(5):
            # Select next
            selected = optimizer.select_next(X_candidates, batch_size=3)

            # Evaluate
            X_new = X_candidates[selected]
            y_new = objective(X_new)

            # Update
            optimizer.update(X_new, y_new)

            # Mark as evaluated
            X_candidates = np.delete(X_candidates, selected, axis=0)

        # Best should improve
        _, y_final = optimizer.get_best_observation()
        assert y_final >= np.max(y_initial)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
