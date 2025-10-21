"""
Unit tests for acquisition functions.

Author: Claude
Date: 2025-10-20
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ml_fragment_optimizer.active_learning.acquisition_functions import (
    AcquisitionConfig,
    AcquisitionType,
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
    ThompsonSampling,
    create_acquisition_function,
    batch_acquisition_sequential,
)


class TestExpectedImprovement:
    """Test Expected Improvement acquisition function."""

    def test_ei_basic(self):
        """Test basic EI calculation."""
        config = AcquisitionConfig(acq_type=AcquisitionType.EI, xi=0.01)
        ei = ExpectedImprovement(config)
        ei.set_best_value(0.8)

        mean = np.array([0.7, 0.85, 0.75])
        std = np.array([0.1, 0.05, 0.15])

        acq = ei.evaluate(mean, std)

        # All values should be non-negative
        assert np.all(acq >= 0)

        # Molecule with high mean and high std should have high EI
        assert acq[1] > 0  # mean=0.85 > f_best=0.8

    def test_ei_zero_std(self):
        """Test EI with zero standard deviation."""
        config = AcquisitionConfig(acq_type=AcquisitionType.EI)
        ei = ExpectedImprovement(config)
        ei.set_best_value(0.8)

        mean = np.array([0.85, 0.7])
        std = np.array([0.0, 0.0])  # Deterministic predictions

        acq = ei.evaluate(mean, std)

        # Zero std should give zero EI
        assert np.allclose(acq, 0.0)

    def test_ei_minimization(self):
        """Test EI for minimization."""
        config = AcquisitionConfig(
            acq_type=AcquisitionType.EI,
            minimize=True
        )
        ei = ExpectedImprovement(config)
        ei.set_best_value(0.5)

        mean = np.array([0.3, 0.7, 0.5])
        std = np.array([0.1, 0.1, 0.1])

        acq = ei.evaluate(mean, std)

        # For minimization, lower mean should give higher EI
        assert acq[0] > acq[1]

    def test_ei_with_gradients(self):
        """Test EI gradient calculation."""
        config = AcquisitionConfig(acq_type=AcquisitionType.EI)
        ei = ExpectedImprovement(config)
        ei.set_best_value(0.8)

        mean = np.array([0.85])
        std = np.array([0.1])
        mean_grad = np.array([[1.0, 0.5]])
        std_grad = np.array([[0.2, 0.3]])

        acq, grad = ei.evaluate_with_gradients(mean, std, mean_grad, std_grad)

        assert acq.shape == (1,)
        assert grad.shape == (1, 2)
        assert np.all(np.isfinite(grad))


class TestUpperConfidenceBound:
    """Test Upper Confidence Bound acquisition function."""

    def test_ucb_basic(self):
        """Test basic UCB calculation."""
        config = AcquisitionConfig(acq_type=AcquisitionType.UCB, kappa=2.0)
        ucb = UpperConfidenceBound(config)

        mean = np.array([0.7, 0.85, 0.75])
        std = np.array([0.1, 0.05, 0.15])

        acq = ucb.evaluate(mean, std)

        # UCB = mean + kappa * std
        expected = mean + 2.0 * std
        assert np.allclose(acq, expected)

    def test_ucb_minimization(self):
        """Test UCB for minimization."""
        config = AcquisitionConfig(
            acq_type=AcquisitionType.UCB,
            kappa=2.0,
            minimize=True
        )
        ucb = UpperConfidenceBound(config)

        mean = np.array([0.7, 0.85])
        std = np.array([0.1, 0.1])

        acq = ucb.evaluate(mean, std)

        # UCB = mean - kappa * std for minimization
        expected = mean - 2.0 * std
        assert np.allclose(acq, expected)

    def test_ucb_adaptive_kappa(self):
        """Test adaptive kappa adjustment."""
        config = AcquisitionConfig(acq_type=AcquisitionType.UCB, kappa=2.0)
        ucb = UpperConfidenceBound(config)

        # Early iteration: high kappa
        ucb.set_adaptive_kappa(iteration=0, total_iterations=10)
        kappa_early = ucb.config.kappa

        # Late iteration: lower kappa
        ucb.set_adaptive_kappa(iteration=9, total_iterations=10)
        kappa_late = ucb.config.kappa

        assert kappa_late < kappa_early


class TestProbabilityOfImprovement:
    """Test Probability of Improvement acquisition function."""

    def test_pi_basic(self):
        """Test basic PI calculation."""
        config = AcquisitionConfig(acq_type=AcquisitionType.PI, xi=0.01)
        pi = ProbabilityOfImprovement(config)
        pi.set_best_value(0.8)

        mean = np.array([0.85, 0.7])
        std = np.array([0.1, 0.1])

        acq = pi.evaluate(mean, std)

        # All values should be in [0, 1]
        assert np.all(acq >= 0)
        assert np.all(acq <= 1)

        # Higher mean should give higher PI
        assert acq[0] > acq[1]


class TestThompsonSampling:
    """Test Thompson Sampling acquisition function."""

    def test_thompson_basic(self):
        """Test Thompson sampling."""
        config = AcquisitionConfig(acq_type=AcquisitionType.THOMPSON)
        ts = ThompsonSampling(config)

        mean = np.array([0.7, 0.85, 0.75])
        std = np.array([0.1, 0.05, 0.15])

        acq = ts.evaluate(mean, std, seed=42)

        # Samples should be finite
        assert np.all(np.isfinite(acq))

    def test_thompson_reproducible(self):
        """Test Thompson sampling reproducibility."""
        config = AcquisitionConfig(acq_type=AcquisitionType.THOMPSON)
        ts = ThompsonSampling(config)

        mean = np.array([0.7, 0.85, 0.75])
        std = np.array([0.1, 0.05, 0.15])

        acq1 = ts.evaluate(mean, std, seed=42)
        acq2 = ts.evaluate(mean, std, seed=42)

        # Same seed should give same samples
        assert np.allclose(acq1, acq2)


class TestAcquisitionFactory:
    """Test acquisition function factory."""

    def test_create_ei(self):
        """Test creating EI."""
        config = AcquisitionConfig(acq_type=AcquisitionType.EI)
        acq = create_acquisition_function(config)
        assert isinstance(acq, ExpectedImprovement)

    def test_create_ucb(self):
        """Test creating UCB."""
        config = AcquisitionConfig(acq_type=AcquisitionType.UCB)
        acq = create_acquisition_function(config)
        assert isinstance(acq, UpperConfidenceBound)

    def test_create_greedy(self):
        """Test creating greedy (pure exploitation)."""
        config = AcquisitionConfig(acq_type=AcquisitionType.GREEDY)
        acq = create_acquisition_function(config)

        mean = np.array([0.7, 0.85, 0.75])
        std = np.array([0.1, 0.1, 0.1])

        acq_val = acq.evaluate(mean, std)

        # Greedy should just return mean
        assert np.allclose(acq_val, mean)


class TestBatchAcquisition:
    """Test batch acquisition strategies."""

    def test_batch_sequential(self):
        """Test sequential batch acquisition."""
        config = AcquisitionConfig(acq_type=AcquisitionType.EI)
        acq_fn = ExpectedImprovement(config)
        acq_fn.set_best_value(0.5)

        mean = np.random.rand(100)
        std = np.random.rand(100) * 0.1

        selected = batch_acquisition_sequential(
            acq_fn, mean, std, batch_size=10
        )

        # Should select 10 molecules
        assert len(selected) == 10

        # All should be unique
        assert len(np.unique(selected)) == 10

    def test_batch_exclude(self):
        """Test batch acquisition with exclusions."""
        config = AcquisitionConfig(acq_type=AcquisitionType.EI)
        acq_fn = ExpectedImprovement(config)
        acq_fn.set_best_value(0.5)

        mean = np.random.rand(100)
        std = np.random.rand(100) * 0.1

        exclude = np.array([0, 1, 2, 3, 4])

        selected = batch_acquisition_sequential(
            acq_fn, mean, std, batch_size=10, already_selected=exclude
        )

        # Should not select excluded indices
        assert not np.any(np.isin(selected, exclude))


class TestConstraints:
    """Test constraint handling."""

    def test_apply_constraints(self):
        """Test constraint penalties."""
        config = AcquisitionConfig(
            acq_type=AcquisitionType.EI,
            constraint_threshold=0.5,
            constraint_weight=100.0
        )
        acq_fn = ExpectedImprovement(config)
        acq_fn.set_best_value(0.8)

        mean = np.array([0.85, 0.85])
        std = np.array([0.1, 0.1])
        constraint_probs = np.array([0.9, 0.3])  # First satisfies, second violates

        # Baseline acquisition
        acq_no_constraint = acq_fn.evaluate(mean, std)

        # With constraint
        acq_with_constraint = acq_fn.apply_constraints(
            acq_no_constraint, constraint_probs
        )

        # First should be unchanged (satisfies constraint)
        assert np.isclose(acq_with_constraint[0], acq_no_constraint[0])

        # Second should be penalized (violates constraint)
        assert acq_with_constraint[1] < acq_no_constraint[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
