"""
Diversity-Based Molecular Selection

This module implements diversity-based selection strategies for exploring chemical space.
Diversity selection complements Bayesian optimization by ensuring broad coverage of
molecular scaffolds and avoiding redundant exploration.

Mathematical Background
-----------------------
Diversity selection aims to maximize dissimilarity within a selected set:

1. MaxMin: Maximize minimum pairwise distance
   x* = argmax_x min_{x' ∈ S} d(x, x')

2. MaxSum: Maximize sum of pairwise distances
   x* = argmax_x Σ_{x' ∈ S} d(x, x')

3. K-means clustering: Select cluster centroids

4. Sphere exclusion: Select points with d(x, S) > radius

Common distance metrics:
- Tanimoto (Jaccard) for binary fingerprints
- Euclidean for continuous descriptors
- Scaffold similarity (Murcko frameworks)

Author: Claude
Date: 2025-10-20
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, List, Tuple, Callable
from enum import Enum

import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances


class DistanceMetric(Enum):
    """Supported distance metrics."""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    TANIMOTO = "tanimoto"  # For binary fingerprints
    DICE = "dice"  # For binary fingerprints


@dataclass
class DiversityConfig:
    """Configuration for diversity selection.

    Attributes
    ----------
    metric : DistanceMetric
        Distance metric to use
    n_clusters : int
        Number of clusters for clustering-based methods
    batch_size : int
        Number of molecules to select
    min_distance : float
        Minimum distance threshold for sphere exclusion
    use_minibatch : bool
        Use MiniBatchKMeans for large datasets (>10k molecules)
    random_state : Optional[int]
        Random seed for reproducibility
    """
    metric: DistanceMetric = DistanceMetric.TANIMOTO
    n_clusters: int = 10
    batch_size: int = 10
    min_distance: float = 0.3
    use_minibatch: bool = False
    random_state: Optional[int] = None


def tanimoto_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute Tanimoto (Jaccard) distance for binary fingerprints.

    Tanimoto coefficient:
        T(A, B) = |A ∩ B| / |A ∪ B| = Σ(a_i · b_i) / (Σa_i + Σb_i - Σ(a_i · b_i))

    Tanimoto distance:
        d(A, B) = 1 - T(A, B)

    Parameters
    ----------
    X : np.ndarray, shape (n_samples_X, n_features)
        Binary fingerprints (0/1 or boolean)
    Y : Optional[np.ndarray], shape (n_samples_Y, n_features)
        Second set of fingerprints. If None, compute pairwise distances within X

    Returns
    -------
    distances : np.ndarray
        If Y is None: shape (n_samples_X * (n_samples_X - 1) / 2,) condensed distance matrix
        If Y is not None: shape (n_samples_X, n_samples_Y) distance matrix

    Examples
    --------
    >>> # Binary fingerprints
    >>> fp1 = np.array([1, 0, 1, 1, 0])
    >>> fp2 = np.array([1, 1, 1, 0, 0])
    >>> fp3 = np.array([0, 1, 0, 1, 1])
    >>> X = np.vstack([fp1, fp2, fp3])
    >>>
    >>> # Pairwise distances within X
    >>> dist = tanimoto_distance(X)
    >>> dist_matrix = squareform(dist)
    >>>
    >>> # Distances from X to Y
    >>> Y = np.array([[1, 1, 0, 0, 1]])
    >>> dist_xy = tanimoto_distance(X, Y)
    """
    X = np.asarray(X, dtype=float)

    if Y is None:
        # Pairwise within X (condensed form)
        n = X.shape[0]
        intersection = X @ X.T
        cardinality = X.sum(axis=1)
        union = cardinality[:, None] + cardinality[None, :] - intersection

        # Avoid division by zero
        union = np.maximum(union, 1e-10)

        tanimoto = intersection / union
        tanimoto_dist = 1.0 - tanimoto

        # Extract upper triangle (condensed form)
        indices = np.triu_indices(n, k=1)
        return tanimoto_dist[indices]
    else:
        # Pairwise between X and Y
        Y = np.asarray(Y, dtype=float)
        intersection = X @ Y.T
        cardinality_X = X.sum(axis=1, keepdims=True)
        cardinality_Y = Y.sum(axis=1, keepdims=True)
        union = cardinality_X + cardinality_Y.T - intersection

        union = np.maximum(union, 1e-10)

        tanimoto = intersection / union
        return 1.0 - tanimoto


def dice_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute Dice distance for binary fingerprints.

    Dice coefficient:
        D(A, B) = 2·|A ∩ B| / (|A| + |B|)

    Dice distance:
        d(A, B) = 1 - D(A, B)

    Parameters
    ----------
    X : np.ndarray, shape (n_samples_X, n_features)
        Binary fingerprints
    Y : Optional[np.ndarray], shape (n_samples_Y, n_features)
        Second set of fingerprints

    Returns
    -------
    distances : np.ndarray
        Distance matrix
    """
    X = np.asarray(X, dtype=float)

    if Y is None:
        n = X.shape[0]
        intersection = X @ X.T
        cardinality = X.sum(axis=1)
        denominator = cardinality[:, None] + cardinality[None, :]

        denominator = np.maximum(denominator, 1e-10)

        dice = 2.0 * intersection / denominator
        dice_dist = 1.0 - dice

        indices = np.triu_indices(n, k=1)
        return dice_dist[indices]
    else:
        Y = np.asarray(Y, dtype=float)
        intersection = X @ Y.T
        cardinality_X = X.sum(axis=1, keepdims=True)
        cardinality_Y = Y.sum(axis=1, keepdims=True)
        denominator = cardinality_X + cardinality_Y.T

        denominator = np.maximum(denominator, 1e-10)

        dice = 2.0 * intersection / denominator
        return 1.0 - dice


def compute_distance_matrix(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    metric: Union[str, DistanceMetric] = DistanceMetric.TANIMOTO,
    condensed: bool = False
) -> np.ndarray:
    """Compute distance matrix between molecular fingerprints.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples_X, n_features)
        Molecular fingerprints or descriptors
    Y : Optional[np.ndarray], shape (n_samples_Y, n_features)
        Second set of fingerprints. If None, compute pairwise within X
    metric : Union[str, DistanceMetric]
        Distance metric to use
    condensed : bool
        If True and Y is None, return condensed distance matrix (upper triangle)

    Returns
    -------
    distances : np.ndarray
        Distance matrix
    """
    if isinstance(metric, str):
        metric = DistanceMetric(metric)

    if metric == DistanceMetric.TANIMOTO:
        dist = tanimoto_distance(X, Y)
        if Y is None and not condensed:
            dist = squareform(dist)
        return dist
    elif metric == DistanceMetric.DICE:
        dist = dice_distance(X, Y)
        if Y is None and not condensed:
            dist = squareform(dist)
        return dist
    else:
        # Use sklearn for other metrics
        metric_str = metric.value
        if Y is None:
            if condensed:
                return pdist(X, metric=metric_str)
            else:
                return pairwise_distances(X, metric=metric_str)
        else:
            return cdist(X, Y, metric=metric_str)


class MaxMinSelector:
    """MaxMin diversity selection.

    Algorithm
    ---------
    1. Select initial molecule (random or max distance to existing set)
    2. For each subsequent selection:
       - Compute distance from each candidate to all selected molecules
       - Select candidate with maximum minimum distance to selected set

    This ensures selected molecules are maximally dissimilar.

    Examples
    --------
    >>> # Generate random fingerprints
    >>> np.random.seed(42)
    >>> fingerprints = np.random.randint(0, 2, (100, 1024))
    >>>
    >>> # Select 10 diverse molecules
    >>> selector = MaxMinSelector(metric=DistanceMetric.TANIMOTO)
    >>> selected_indices = selector.select(fingerprints, n_select=10)
    >>> len(selected_indices)
    10
    """

    def __init__(
        self,
        metric: DistanceMetric = DistanceMetric.TANIMOTO,
        random_state: Optional[int] = None
    ):
        """Initialize MaxMin selector.

        Parameters
        ----------
        metric : DistanceMetric
            Distance metric to use
        random_state : Optional[int]
            Random seed for reproducibility
        """
        self.metric = metric
        self.random_state = random_state

    def select(
        self,
        X: np.ndarray,
        n_select: int,
        initial_idx: Optional[int] = None,
        exclude_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Select diverse subset using MaxMin algorithm.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Molecular fingerprints
        n_select : int
            Number of molecules to select
        initial_idx : Optional[int]
            Index of initial molecule. If None, select randomly
        exclude_indices : Optional[np.ndarray]
            Indices to exclude from selection

        Returns
        -------
        selected_indices : np.ndarray, shape (n_select,)
            Indices of selected molecules
        """
        n_samples = X.shape[0]

        if n_select > n_samples:
            raise ValueError(f"Cannot select {n_select} from {n_samples} samples")

        # Initialize
        if self.random_state is not None:
            np.random.seed(self.random_state)

        selected = []
        available = np.ones(n_samples, dtype=bool)

        if exclude_indices is not None:
            available[exclude_indices] = False

        # Select initial molecule
        if initial_idx is not None:
            if not available[initial_idx]:
                raise ValueError("Initial index is excluded")
            selected.append(initial_idx)
            available[initial_idx] = False
        else:
            available_indices = np.where(available)[0]
            initial_idx = np.random.choice(available_indices)
            selected.append(initial_idx)
            available[initial_idx] = False

        # Initialize minimum distances
        min_distances = compute_distance_matrix(
            X,
            X[initial_idx:initial_idx+1],
            metric=self.metric
        ).ravel()
        min_distances[~available] = -np.inf

        # Iteratively select molecules
        for _ in range(n_select - 1):
            # Select molecule with maximum minimum distance
            next_idx = np.argmax(min_distances)
            selected.append(next_idx)
            available[next_idx] = False

            # Update minimum distances
            new_distances = compute_distance_matrix(
                X,
                X[next_idx:next_idx+1],
                metric=self.metric
            ).ravel()

            min_distances = np.minimum(min_distances, new_distances)
            min_distances[~available] = -np.inf

        return np.array(selected)


class SphereExclusionSelector:
    """Sphere exclusion diversity selection.

    Algorithm
    ---------
    1. Select initial molecule
    2. Exclude all molecules within radius r
    3. Select next molecule from remaining pool
    4. Repeat until n_select molecules or no molecules remain

    Parameters
    ----------
    metric : DistanceMetric
        Distance metric
    radius : float
        Exclusion radius (molecules within this distance are excluded)
    random_state : Optional[int]
        Random seed

    Examples
    --------
    >>> selector = SphereExclusionSelector(
    ...     metric=DistanceMetric.TANIMOTO,
    ...     radius=0.3
    ... )
    >>> fingerprints = np.random.randint(0, 2, (100, 1024))
    >>> selected = selector.select(fingerprints, n_select=20)
    """

    def __init__(
        self,
        metric: DistanceMetric = DistanceMetric.TANIMOTO,
        radius: float = 0.3,
        random_state: Optional[int] = None
    ):
        """Initialize sphere exclusion selector."""
        self.metric = metric
        self.radius = radius
        self.random_state = random_state

    def select(
        self,
        X: np.ndarray,
        n_select: int,
        exclude_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Select diverse subset using sphere exclusion.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Molecular fingerprints
        n_select : int
            Maximum number to select (may return fewer)
        exclude_indices : Optional[np.ndarray]
            Indices to exclude

        Returns
        -------
        selected_indices : np.ndarray
            Indices of selected molecules
        """
        n_samples = X.shape[0]

        if self.random_state is not None:
            np.random.seed(self.random_state)

        selected = []
        available = np.ones(n_samples, dtype=bool)

        if exclude_indices is not None:
            available[exclude_indices] = False

        while len(selected) < n_select and available.any():
            # Select random available molecule
            available_indices = np.where(available)[0]
            idx = np.random.choice(available_indices)
            selected.append(idx)
            available[idx] = False

            # Compute distances to selected molecule
            distances = compute_distance_matrix(
                X,
                X[idx:idx+1],
                metric=self.metric
            ).ravel()

            # Exclude molecules within radius
            within_sphere = distances < self.radius
            available[within_sphere] = False

        return np.array(selected)


class ClusteringSelector:
    """Clustering-based diversity selection.

    Uses k-means (or mini-batch k-means) to cluster molecules, then selects
    representative molecules from each cluster (e.g., centroid, medoid, or
    highest-scoring molecule in cluster).

    Examples
    --------
    >>> selector = ClusteringSelector(
    ...     n_clusters=10,
    ...     metric=DistanceMetric.EUCLIDEAN,
    ...     selection_mode='medoid'
    ... )
    >>> descriptors = np.random.randn(1000, 100)
    >>> selected = selector.select(descriptors, n_select=10)
    """

    def __init__(
        self,
        n_clusters: int = 10,
        metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
        selection_mode: str = "medoid",
        use_minibatch: bool = False,
        random_state: Optional[int] = None
    ):
        """Initialize clustering selector.

        Parameters
        ----------
        n_clusters : int
            Number of clusters
        metric : DistanceMetric
            Distance metric
        selection_mode : str
            How to select from each cluster: 'medoid', 'centroid', 'random'
        use_minibatch : bool
            Use MiniBatchKMeans for large datasets
        random_state : Optional[int]
            Random seed
        """
        self.n_clusters = n_clusters
        self.metric = metric
        self.selection_mode = selection_mode
        self.use_minibatch = use_minibatch
        self.random_state = random_state

    def select(
        self,
        X: np.ndarray,
        n_select: Optional[int] = None,
        scores: Optional[np.ndarray] = None,
        exclude_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Select diverse subset using clustering.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Molecular fingerprints or descriptors
        n_select : Optional[int]
            Number to select. If None, select one per cluster
        scores : Optional[np.ndarray], shape (n_samples,)
            Scores for each molecule (e.g., predicted affinity)
            Used when selection_mode='best'
        exclude_indices : Optional[np.ndarray]
            Indices to exclude

        Returns
        -------
        selected_indices : np.ndarray
            Indices of selected molecules
        """
        if n_select is None:
            n_select = self.n_clusters

        # Clustering
        if self.use_minibatch:
            clusterer = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state
            )
        else:
            clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state
            )

        labels = clusterer.fit_predict(X)

        # Select from each cluster
        selected = []
        available = np.ones(len(X), dtype=bool)
        if exclude_indices is not None:
            available[exclude_indices] = False

        for cluster_id in range(self.n_clusters):
            cluster_mask = (labels == cluster_id) & available

            if not cluster_mask.any():
                continue

            cluster_indices = np.where(cluster_mask)[0]

            if self.selection_mode == "random":
                idx = np.random.choice(cluster_indices)
            elif self.selection_mode == "medoid":
                # Medoid: molecule closest to cluster center
                cluster_center = clusterer.cluster_centers_[cluster_id]
                distances = np.linalg.norm(X[cluster_indices] - cluster_center, axis=1)
                idx = cluster_indices[np.argmin(distances)]
            elif self.selection_mode == "centroid":
                # Closest to centroid (same as medoid for k-means)
                cluster_center = clusterer.cluster_centers_[cluster_id]
                distances = np.linalg.norm(X[cluster_indices] - cluster_center, axis=1)
                idx = cluster_indices[np.argmin(distances)]
            elif self.selection_mode == "best":
                # Highest score in cluster
                if scores is None:
                    raise ValueError("Must provide scores for selection_mode='best'")
                idx = cluster_indices[np.argmax(scores[cluster_indices])]
            else:
                raise ValueError(f"Unknown selection mode: {self.selection_mode}")

            selected.append(idx)

            if len(selected) >= n_select:
                break

        return np.array(selected[:n_select])


class ScaffoldDiversitySelector:
    """Scaffold-based diversity selection.

    Groups molecules by Murcko scaffold, then selects representatives from
    different scaffolds to maximize scaffold diversity.

    Note: This requires RDKit for scaffold decomposition. If RDKit is not
    available, falls back to fingerprint-based clustering.

    Examples
    --------
    >>> # Requires RDKit and SMILES strings
    >>> smiles_list = [...]  # List of SMILES
    >>> selector = ScaffoldDiversitySelector()
    >>> selected = selector.select_from_smiles(smiles_list, n_select=10)
    """

    def __init__(self, random_state: Optional[int] = None):
        """Initialize scaffold diversity selector."""
        self.random_state = random_state
        self._rdkit_available = False

        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            self._rdkit_available = True
            self.Chem = Chem
            self.MurckoScaffold = MurckoScaffold
        except ImportError:
            pass

    def select_from_smiles(
        self,
        smiles_list: List[str],
        n_select: int,
        scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Select diverse molecules based on Murcko scaffolds.

        Parameters
        ----------
        smiles_list : List[str]
            SMILES strings
        n_select : int
            Number to select
        scores : Optional[np.ndarray], shape (n_samples,)
            Scores for tie-breaking

        Returns
        -------
        selected_indices : np.ndarray
            Indices of selected molecules
        """
        if not self._rdkit_available:
            raise ImportError("RDKit is required for scaffold-based selection")

        # Group by scaffold
        scaffold_to_indices = {}
        for i, smiles in enumerate(smiles_list):
            mol = self.Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            try:
                scaffold = self.MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            except:
                scaffold = smiles  # Use original if scaffold extraction fails

            if scaffold not in scaffold_to_indices:
                scaffold_to_indices[scaffold] = []
            scaffold_to_indices[scaffold].append(i)

        # Select from each scaffold
        selected = []
        scaffolds = list(scaffold_to_indices.keys())

        if self.random_state is not None:
            np.random.seed(self.random_state)
            np.random.shuffle(scaffolds)

        for scaffold in scaffolds:
            indices = scaffold_to_indices[scaffold]

            if scores is not None:
                # Select highest-scoring from this scaffold
                best_in_scaffold = indices[np.argmax(scores[indices])]
            else:
                # Select random
                best_in_scaffold = np.random.choice(indices)

            selected.append(best_in_scaffold)

            if len(selected) >= n_select:
                break

        return np.array(selected[:n_select])


def select_diverse_molecules(
    X: np.ndarray,
    n_select: int,
    method: str = "maxmin",
    metric: DistanceMetric = DistanceMetric.TANIMOTO,
    scores: Optional[np.ndarray] = None,
    exclude_indices: Optional[np.ndarray] = None,
    **kwargs
) -> np.ndarray:
    """Unified interface for diversity selection.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Molecular fingerprints or descriptors
    n_select : int
        Number of molecules to select
    method : str
        Selection method: 'maxmin', 'sphere_exclusion', 'kmeans', 'random'
    metric : DistanceMetric
        Distance metric to use
    scores : Optional[np.ndarray], shape (n_samples,)
        Scores for ranking (used in some methods)
    exclude_indices : Optional[np.ndarray]
        Indices to exclude from selection
    **kwargs
        Additional method-specific parameters

    Returns
    -------
    selected_indices : np.ndarray, shape (n_select,)
        Indices of selected molecules

    Examples
    --------
    >>> fingerprints = np.random.randint(0, 2, (1000, 2048))
    >>> selected = select_diverse_molecules(
    ...     fingerprints,
    ...     n_select=50,
    ...     method='maxmin',
    ...     metric=DistanceMetric.TANIMOTO
    ... )
    """
    if method == "maxmin":
        selector = MaxMinSelector(metric=metric, **kwargs)
        return selector.select(X, n_select, exclude_indices=exclude_indices)

    elif method == "sphere_exclusion":
        radius = kwargs.get("radius", 0.3)
        selector = SphereExclusionSelector(metric=metric, radius=radius, **kwargs)
        return selector.select(X, n_select, exclude_indices=exclude_indices)

    elif method == "kmeans":
        n_clusters = kwargs.get("n_clusters", n_select)
        selection_mode = kwargs.get("selection_mode", "medoid")
        selector = ClusteringSelector(
            n_clusters=n_clusters,
            metric=metric,
            selection_mode=selection_mode,
            **kwargs
        )
        return selector.select(X, n_select, scores=scores, exclude_indices=exclude_indices)

    elif method == "random":
        # Random selection (baseline)
        available = np.ones(len(X), dtype=bool)
        if exclude_indices is not None:
            available[exclude_indices] = False
        available_indices = np.where(available)[0]
        random_state = kwargs.get("random_state", None)
        if random_state is not None:
            np.random.seed(random_state)
        return np.random.choice(available_indices, size=n_select, replace=False)

    else:
        raise ValueError(f"Unknown method: {method}")
