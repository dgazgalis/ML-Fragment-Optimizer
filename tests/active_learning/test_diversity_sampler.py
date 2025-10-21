"""
Unit tests for diversity sampling.

Author: Claude
Date: 2025-10-20
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ml_fragment_optimizer.active_learning.diversity_sampler import (
    DistanceMetric,
    tanimoto_distance,
    dice_distance,
    compute_distance_matrix,
    MaxMinSelector,
    SphereExclusionSelector,
    ClusteringSelector,
    select_diverse_molecules,
)


class TestDistanceMetrics:
    """Test distance metric calculations."""

    def test_tanimoto_distance(self):
        """Test Tanimoto distance."""
        fp1 = np.array([[1, 0, 1, 1, 0]])
        fp2 = np.array([[1, 1, 1, 0, 0]])

        dist = tanimoto_distance(fp1, fp2)

        # Tanimoto coefficient: |intersection| / |union| = 2 / 4 = 0.5
        # Tanimoto distance: 1 - 0.5 = 0.5
        assert np.isclose(dist[0, 0], 0.5, atol=0.01)

    def test_tanimoto_identical(self):
        """Test Tanimoto distance for identical fingerprints."""
        fp = np.array([[1, 0, 1, 1, 0]])

        dist = tanimoto_distance(fp, fp)

        # Identical fingerprints should have distance 0
        assert np.isclose(dist[0, 0], 0.0)

    def test_dice_distance(self):
        """Test Dice distance."""
        fp1 = np.array([[1, 0, 1, 1, 0]])
        fp2 = np.array([[1, 1, 1, 0, 0]])

        dist = dice_distance(fp1, fp2)

        # Dice coefficient: 2 * |intersection| / (|A| + |B|) = 2*2 / (3+3) = 4/6
        # Dice distance: 1 - 4/6 = 2/6 = 0.333
        assert np.isclose(dist[0, 0], 0.333, atol=0.01)

    def test_compute_distance_matrix_tanimoto(self):
        """Test distance matrix computation with Tanimoto."""
        fps = np.random.randint(0, 2, (10, 128))

        dist_matrix = compute_distance_matrix(
            fps, metric=DistanceMetric.TANIMOTO
        )

        # Should be square
        assert dist_matrix.shape == (10, 10)

        # Diagonal should be zero
        assert np.allclose(np.diag(dist_matrix), 0.0)

        # Should be symmetric
        assert np.allclose(dist_matrix, dist_matrix.T)

    def test_compute_distance_matrix_euclidean(self):
        """Test distance matrix with Euclidean."""
        descriptors = np.random.randn(10, 50)

        dist_matrix = compute_distance_matrix(
            descriptors, metric=DistanceMetric.EUCLIDEAN
        )

        assert dist_matrix.shape == (10, 10)
        assert np.allclose(np.diag(dist_matrix), 0.0)


class TestMaxMinSelector:
    """Test MaxMin diversity selection."""

    @pytest.fixture
    def fingerprints(self):
        """Generate binary fingerprints."""
        np.random.seed(42)
        return np.random.randint(0, 2, (100, 1024))

    def test_maxmin_basic(self, fingerprints):
        """Test basic MaxMin selection."""
        selector = MaxMinSelector(
            metric=DistanceMetric.TANIMOTO,
            random_state=42
        )

        selected = selector.select(fingerprints, n_select=10)

        assert len(selected) == 10
        assert len(np.unique(selected)) == 10

    def test_maxmin_initial_index(self, fingerprints):
        """Test MaxMin with specified initial index."""
        selector = MaxMinSelector(metric=DistanceMetric.TANIMOTO)

        selected = selector.select(fingerprints, n_select=10, initial_idx=0)

        # First selection should be initial index
        assert selected[0] == 0

    def test_maxmin_exclude(self, fingerprints):
        """Test MaxMin with exclusions."""
        selector = MaxMinSelector(metric=DistanceMetric.TANIMOTO)

        exclude = np.array([0, 1, 2])
        selected = selector.select(
            fingerprints, n_select=10, exclude_indices=exclude
        )

        # Should not select excluded indices
        assert not np.any(np.isin(selected, exclude))

    def test_maxmin_diversity(self, fingerprints):
        """Test that MaxMin produces diverse selection."""
        selector = MaxMinSelector(
            metric=DistanceMetric.TANIMOTO,
            random_state=42
        )

        selected = selector.select(fingerprints, n_select=10)

        # Compute pairwise distances within selected set
        selected_fps = fingerprints[selected]
        dist_matrix = compute_distance_matrix(
            selected_fps, metric=DistanceMetric.TANIMOTO
        )

        # Minimum pairwise distance should be relatively large
        # (excluding diagonal)
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix)

        # For random fingerprints, should be reasonably diverse
        assert min_dist > 0.1


class TestSphereExclusionSelector:
    """Test Sphere Exclusion diversity selection."""

    @pytest.fixture
    def fingerprints(self):
        """Generate binary fingerprints."""
        np.random.seed(42)
        return np.random.randint(0, 2, (100, 1024))

    def test_sphere_exclusion_basic(self, fingerprints):
        """Test basic sphere exclusion."""
        selector = SphereExclusionSelector(
            metric=DistanceMetric.TANIMOTO,
            radius=0.3,
            random_state=42
        )

        selected = selector.select(fingerprints, n_select=20)

        # May return fewer than requested due to exclusions
        assert len(selected) <= 20
        assert len(np.unique(selected)) == len(selected)

    def test_sphere_exclusion_diversity(self, fingerprints):
        """Test sphere exclusion produces diverse selection."""
        selector = SphereExclusionSelector(
            metric=DistanceMetric.TANIMOTO,
            radius=0.4,
            random_state=42
        )

        selected = selector.select(fingerprints, n_select=50)

        # Pairwise distances should be at least radius
        selected_fps = fingerprints[selected]
        dist_matrix = compute_distance_matrix(
            selected_fps, metric=DistanceMetric.TANIMOTO
        )

        # Check minimum pairwise distance
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix)

        # Should be >= radius (with some numerical tolerance)
        assert min_dist >= 0.4 - 0.05


class TestClusteringSelector:
    """Test Clustering-based diversity selection."""

    @pytest.fixture
    def descriptors(self):
        """Generate continuous descriptors."""
        np.random.seed(42)
        return np.random.randn(200, 50)

    def test_clustering_basic(self, descriptors):
        """Test basic clustering selection."""
        selector = ClusteringSelector(
            n_clusters=10,
            metric=DistanceMetric.EUCLIDEAN,
            random_state=42
        )

        selected = selector.select(descriptors, n_select=10)

        assert len(selected) == 10
        assert len(np.unique(selected)) == 10

    def test_clustering_with_scores(self, descriptors):
        """Test clustering with score-based selection."""
        selector = ClusteringSelector(
            n_clusters=10,
            selection_mode="best",
            random_state=42
        )

        scores = np.random.rand(200)

        selected = selector.select(
            descriptors, n_select=10, scores=scores
        )

        assert len(selected) == 10

    def test_clustering_modes(self, descriptors):
        """Test different clustering selection modes."""
        modes = ["medoid", "centroid", "random"]

        for mode in modes:
            selector = ClusteringSelector(
                n_clusters=10,
                selection_mode=mode,
                random_state=42
            )

            selected = selector.select(descriptors, n_select=10)
            assert len(selected) == 10


class TestDiverseSelection:
    """Test unified diverse selection interface."""

    @pytest.fixture
    def fingerprints(self):
        """Generate fingerprints."""
        np.random.seed(42)
        return np.random.randint(0, 2, (500, 2048))

    def test_select_diverse_maxmin(self, fingerprints):
        """Test diverse selection with MaxMin."""
        selected = select_diverse_molecules(
            fingerprints,
            n_select=50,
            method="maxmin",
            metric=DistanceMetric.TANIMOTO,
            random_state=42
        )

        assert len(selected) == 50

    def test_select_diverse_kmeans(self, fingerprints):
        """Test diverse selection with k-means."""
        selected = select_diverse_molecules(
            fingerprints,
            n_select=50,
            method="kmeans",
            metric=DistanceMetric.EUCLIDEAN,
            n_clusters=50,
            random_state=42
        )

        assert len(selected) == 50

    def test_select_diverse_random(self, fingerprints):
        """Test random selection (baseline)."""
        selected = select_diverse_molecules(
            fingerprints,
            n_select=50,
            method="random",
            random_state=42
        )

        assert len(selected) == 50

    def test_select_diverse_with_exclude(self, fingerprints):
        """Test diverse selection with exclusions."""
        exclude = np.arange(10)

        selected = select_diverse_molecules(
            fingerprints,
            n_select=50,
            method="maxmin",
            metric=DistanceMetric.TANIMOTO,
            exclude_indices=exclude,
            random_state=42
        )

        # Should not include excluded indices
        assert not np.any(np.isin(selected, exclude))


class TestScaffoldDiversitySelector:
    """Test scaffold-based diversity selection."""

    def test_scaffold_selector_no_rdkit(self):
        """Test that scaffold selector requires RDKit."""
        from active_learning.diversity_sampler import ScaffoldDiversitySelector

        selector = ScaffoldDiversitySelector()

        # Should fail without RDKit
        if not selector._rdkit_available:
            smiles = ["c1ccccc1", "c1ccccc1C", "c1ccccc1CC"]
            with pytest.raises(ImportError):
                selector.select_from_smiles(smiles, n_select=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
