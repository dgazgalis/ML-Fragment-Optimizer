"""
Rigorous model benchmarking with proper train/test splitting strategies.

This module implements various splitting strategies to prevent data leakage and
provide realistic performance estimates for molecular property prediction models.

Key Features:
-------------
- Scaffold-based splitting (Bemis-Murcko scaffolds)
- Temporal splitting (by registration/publication date)
- Cluster-based splitting (molecular clustering)
- Activity cliff-aware splitting
- K-fold cross-validation with proper grouping
- Data leakage detection

Literature:
-----------
- Martin et al. (2012) "Does rational selection of training and test sets
  improve the outcome of QSAR modeling?" J Chem Inf Model 52(10):2570-2578
- Ramsundar et al. (2019) "Deep Learning for the Life Sciences" O'Reilly
- Sheridan (2013) "Time-Split Cross-Validation as a Method for Estimating the
  Goodness of Prospective Prediction" J Chem Inf Model 53(4):783-790

WARNING: Random splits severely overestimate model performance due to molecular
similarity between train and test sets. Always use scaffold or cluster splits
for realistic estimates.
"""

from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
from collections import defaultdict
import warnings
from dataclasses import dataclass

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Scaffold splitting will not work.")

from sklearn.model_selection import KFold, GroupKFold
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


@dataclass
class BenchmarkResult:
    """Results from a benchmarking run."""
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    per_scaffold_metrics: Optional[Dict[str, Dict[str, float]]] = None
    leakage_score: Optional[float] = None


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """
    Generate Bemis-Murcko scaffold from SMILES string.

    Parameters:
    -----------
    smiles : str
        SMILES string of molecule
    include_chirality : bool
        Whether to include chirality in scaffold (default: False)

    Returns:
    --------
    scaffold : str
        SMILES string of scaffold

    References:
    -----------
    Bemis & Murcko (1996) "The properties of known drugs. 1. Molecular frameworks"
    J Med Chem 39(15):2887-2893

    Example:
    --------
    >>> scaffold = generate_scaffold("CCc1ccccc1")
    >>> print(scaffold)
    'c1ccccc1'
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for scaffold generation")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        warnings.warn(f"Could not parse SMILES: {smiles}")
        return ""

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality
    )
    return scaffold


def scaffold_split(
    smiles_list: List[str],
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    balanced: bool = True,
    random_state: Optional[int] = None
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split molecules by Bemis-Murcko scaffolds to prevent data leakage.

    Molecules with the same scaffold are kept in the same split. This prevents
    the model from learning scaffold-specific patterns that inflate performance.

    Parameters:
    -----------
    smiles_list : List[str]
        List of SMILES strings
    test_size : float
        Fraction of data for test set (default: 0.2)
    val_size : Optional[float]
        Fraction of data for validation set (default: None)
    balanced : bool
        Whether to balance scaffold sizes across splits (default: True)
    random_state : Optional[int]
        Random seed for reproducibility

    Returns:
    --------
    splits : Tuple[np.ndarray, np.ndarray] or Tuple[np.ndarray, np.ndarray, np.ndarray]
        Indices for train/test or train/val/test splits

    Example:
    --------
    >>> smiles = ["CCc1ccccc1", "CCCc1ccccc1", "c1cccnc1", "c1ccncc1"]
    >>> train_idx, test_idx = scaffold_split(smiles, test_size=0.25)
    >>> print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    Train: 3, Test: 1

    Notes:
    ------
    - Scaffolds are kept together to prevent leakage of structural information
    - Balanced mode attempts to equalize split sizes by sorting scaffolds
    - Unbalanced mode can lead to very different split sizes
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for scaffold splitting")

    if random_state is not None:
        np.random.seed(random_state)

    # Generate scaffolds for all molecules
    scaffolds = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles)
        scaffolds[scaffold].append(idx)

    # Sort scaffolds by size (descending) for balanced splits
    if balanced:
        scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    else:
        scaffold_sets = list(scaffolds.values())
        np.random.shuffle(scaffold_sets)

    # Distribute scaffolds across splits
    n_total = len(smiles_list)
    n_test = int(n_total * test_size)

    if val_size is not None:
        n_val = int(n_total * val_size)
        train_idx, val_idx, test_idx = [], [], []

        # Round-robin assignment for balance
        for i, scaffold_set in enumerate(scaffold_sets):
            if len(test_idx) < n_test:
                test_idx.extend(scaffold_set)
            elif len(val_idx) < n_val:
                val_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        return np.array(train_idx), np.array(val_idx), np.array(test_idx)
    else:
        train_idx, test_idx = [], []

        for scaffold_set in scaffold_sets:
            if len(test_idx) < n_test:
                test_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        return np.array(train_idx), np.array(test_idx)


def temporal_split(
    dates: np.ndarray,
    test_size: float = 0.2,
    val_size: Optional[float] = None
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split data by temporal ordering (e.g., publication date, registration date).

    This mimics prospective validation where you train on historical data and
    predict on future data. Critical for assessing model performance in production.

    Parameters:
    -----------
    dates : np.ndarray
        Array of dates (as timestamps or sortable values)
    test_size : float
        Fraction of most recent data for test set
    val_size : Optional[float]
        Fraction of data for validation set

    Returns:
    --------
    splits : Tuple of indices
        Train/test or train/val/test splits

    Example:
    --------
    >>> dates = np.array([2018, 2019, 2019, 2020, 2021, 2021, 2022])
    >>> train_idx, test_idx = temporal_split(dates, test_size=0.3)
    >>> print(dates[train_idx])
    [2018 2019 2019 2020 2021]
    >>> print(dates[test_idx])
    [2021 2022]

    References:
    -----------
    Sheridan (2013) "Time-Split Cross-Validation as a Method for Estimating the
    Goodness of Prospective Prediction" J Chem Inf Model 53(4):783-790

    WARNING: Temporal splits often show worse performance than random splits,
    but provide realistic estimates of prospective performance.
    """
    # Sort by date
    sorted_indices = np.argsort(dates)
    n_total = len(dates)

    if val_size is not None:
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_test - n_val

        train_idx = sorted_indices[:n_train]
        val_idx = sorted_indices[n_train:n_train + n_val]
        test_idx = sorted_indices[n_train + n_val:]

        return train_idx, val_idx, test_idx
    else:
        n_train = int(n_total * (1 - test_size))
        train_idx = sorted_indices[:n_train]
        test_idx = sorted_indices[n_train:]

        return train_idx, test_idx


def cluster_split(
    X: np.ndarray,
    n_clusters: Optional[int] = None,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    method: str = "kmeans",
    random_state: Optional[int] = None
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split data by molecular clusters to prevent similarity leakage.

    Alternative to scaffold splitting that uses descriptor-based clustering.
    Useful when scaffold diversity is limited.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    n_clusters : Optional[int]
        Number of clusters (if None, estimated automatically)
    test_size : float
        Fraction for test set
    val_size : Optional[float]
        Fraction for validation set
    method : str
        Clustering method: "kmeans" or "hierarchical"
    random_state : Optional[int]
        Random seed

    Returns:
    --------
    splits : Tuple of indices

    Example:
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20)
    >>> train_idx, test_idx = cluster_split(X, n_clusters=10, test_size=0.2)
    """
    n_samples = X.shape[0]

    # Estimate number of clusters if not provided
    if n_clusters is None:
        n_clusters = max(10, int(n_samples * 0.1))

    # Perform clustering
    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = clusterer.fit_predict(X)
    elif method == "hierarchical":
        distances = pdist(X, metric="euclidean")
        Z = linkage(distances, method="ward")
        cluster_labels = fcluster(Z, n_clusters, criterion="maxclust")
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Split by clusters
    unique_clusters = np.unique(cluster_labels)
    np.random.shuffle(unique_clusters)

    n_test_clusters = int(len(unique_clusters) * test_size)

    if val_size is not None:
        n_val_clusters = int(len(unique_clusters) * val_size)

        test_clusters = unique_clusters[:n_test_clusters]
        val_clusters = unique_clusters[n_test_clusters:n_test_clusters + n_val_clusters]
        train_clusters = unique_clusters[n_test_clusters + n_val_clusters:]

        train_idx = np.where(np.isin(cluster_labels, train_clusters))[0]
        val_idx = np.where(np.isin(cluster_labels, val_clusters))[0]
        test_idx = np.where(np.isin(cluster_labels, test_clusters))[0]

        return train_idx, val_idx, test_idx
    else:
        test_clusters = unique_clusters[:n_test_clusters]
        train_clusters = unique_clusters[n_test_clusters:]

        train_idx = np.where(np.isin(cluster_labels, train_clusters))[0]
        test_idx = np.where(np.isin(cluster_labels, test_clusters))[0]

        return train_idx, test_idx


def activity_cliff_split(
    smiles_list: List[str],
    activities: np.ndarray,
    similarity_threshold: float = 0.9,
    activity_diff_threshold: float = 2.0,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data ensuring activity cliffs don't span train/test sets.

    Activity cliffs are pairs of similar molecules with large activity differences.
    Having one in train and one in test artificially inflates performance.

    Parameters:
    -----------
    smiles_list : List[str]
        List of SMILES strings
    activities : np.ndarray
        Activity values (e.g., pIC50)
    similarity_threshold : float
        Tanimoto similarity threshold for defining "similar" (default: 0.9)
    activity_diff_threshold : float
        Activity difference threshold for defining "cliff" (default: 2.0 log units)
    test_size : float
        Fraction for test set
    random_state : Optional[int]
        Random seed

    Returns:
    --------
    train_idx, test_idx : Tuple[np.ndarray, np.ndarray]
        Train and test indices

    Example:
    --------
    >>> smiles = ["CCc1ccccc1", "CCCc1ccccc1", "c1cccnc1"]
    >>> activities = np.array([5.0, 7.5, 6.0])
    >>> train_idx, test_idx = activity_cliff_split(smiles, activities)

    References:
    -----------
    Stumpfe & Bajorath (2012) "Exploring activity cliffs in medicinal chemistry"
    J Med Chem 55(7):2932-2942

    Notes:
    ------
    This is computationally expensive for large datasets due to pairwise
    similarity calculations. Consider sampling for very large datasets.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for activity cliff splitting")

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(smiles_list)

    # Generate fingerprints
    fps = []
    valid_indices = []
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
            valid_indices.append(idx)
        else:
            warnings.warn(f"Could not parse SMILES: {smiles}")

    valid_indices = np.array(valid_indices)
    activities = activities[valid_indices]

    # Find activity cliff pairs
    cliff_pairs = set()
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            # Calculate Tanimoto similarity
            similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])

            # Check if it's an activity cliff
            if similarity >= similarity_threshold:
                activity_diff = abs(activities[i] - activities[j])
                if activity_diff >= activity_diff_threshold:
                    cliff_pairs.add(frozenset([i, j]))

    # Build graph of cliff relationships
    cliff_groups = []
    remaining = set(range(len(fps)))

    for pair in cliff_pairs:
        pair_list = list(pair)
        # Check if either element is already in a group
        found_group = False
        for group in cliff_groups:
            if any(p in group for p in pair_list):
                group.update(pair_list)
                found_group = True
                break
        if not found_group:
            cliff_groups.append(set(pair_list))

        for p in pair_list:
            remaining.discard(p)

    # Add singleton molecules not in any cliff
    for idx in remaining:
        cliff_groups.append({idx})

    # Randomly assign groups to train/test
    np.random.shuffle(cliff_groups)

    n_test = int(len(fps) * test_size)
    test_idx_local = []
    train_idx_local = []

    for group in cliff_groups:
        if len(test_idx_local) < n_test:
            test_idx_local.extend(group)
        else:
            train_idx_local.extend(group)

    # Map back to original indices
    train_idx = valid_indices[train_idx_local]
    test_idx = valid_indices[test_idx_local]

    return train_idx, test_idx


def scaffold_k_fold(
    smiles_list: List[str],
    n_splits: int = 5,
    random_state: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    K-fold cross-validation with scaffold-based grouping.

    Ensures that molecules with the same scaffold are always in the same fold.

    Parameters:
    -----------
    smiles_list : List[str]
        List of SMILES strings
    n_splits : int
        Number of folds
    random_state : Optional[int]
        Random seed

    Returns:
    --------
    folds : List[Tuple[np.ndarray, np.ndarray]]
        List of (train_idx, test_idx) tuples for each fold

    Example:
    --------
    >>> smiles = ["CCc1ccccc1"] * 50 + ["c1cccnc1"] * 50
    >>> folds = scaffold_k_fold(smiles, n_splits=5)
    >>> for i, (train_idx, test_idx) in enumerate(folds):
    ...     print(f"Fold {i}: train={len(train_idx)}, test={len(test_idx)}")
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for scaffold-based k-fold")

    # Generate scaffold groups
    scaffolds = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles)
        scaffolds[scaffold].append(idx)

    # Convert to array of group labels
    groups = np.zeros(len(smiles_list), dtype=int)
    for group_id, (scaffold, indices) in enumerate(scaffolds.items()):
        for idx in indices:
            groups[idx] = group_id

    # Use GroupKFold to ensure scaffolds don't split
    gkf = GroupKFold(n_splits=n_splits)

    folds = []
    for train_idx, test_idx in gkf.split(smiles_list, groups=groups):
        folds.append((train_idx, test_idx))

    return folds


def detect_data_leakage(
    train_smiles: List[str],
    test_smiles: List[str],
    similarity_threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Detect potential data leakage between train and test sets.

    Parameters:
    -----------
    train_smiles : List[str]
        Training set SMILES
    test_smiles : List[str]
        Test set SMILES
    similarity_threshold : float
        Tanimoto similarity threshold for flagging leakage

    Returns:
    --------
    leakage_report : Dict[str, Any]
        Report containing:
        - n_similar_pairs: number of similar train/test pairs
        - leakage_score: fraction of test set with similar training example
        - examples: list of (train_idx, test_idx, similarity) tuples

    Example:
    --------
    >>> train = ["CCc1ccccc1", "CCCc1ccccc1"]
    >>> test = ["CCCCc1ccccc1"]
    >>> report = detect_data_leakage(train, test, similarity_threshold=0.8)
    >>> print(f"Leakage score: {report['leakage_score']:.2%}")

    Notes:
    ------
    High leakage scores indicate that the test set is not truly independent
    from the training set, leading to overoptimistic performance estimates.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for leakage detection")

    # Generate fingerprints
    train_fps = []
    for smiles in train_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            train_fps.append(fp)

    test_fps = []
    test_valid_idx = []
    for idx, smiles in enumerate(test_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            test_fps.append(fp)
            test_valid_idx.append(idx)

    # Find similar pairs
    from rdkit import DataStructs
    similar_pairs = []
    test_with_similar = set()

    for test_idx, test_fp in enumerate(test_fps):
        max_similarity = 0.0
        max_train_idx = -1

        for train_idx, train_fp in enumerate(train_fps):
            similarity = DataStructs.TanimotoSimilarity(train_fp, test_fp)
            if similarity > max_similarity:
                max_similarity = similarity
                max_train_idx = train_idx

        if max_similarity >= similarity_threshold:
            similar_pairs.append((max_train_idx, test_valid_idx[test_idx], max_similarity))
            test_with_similar.add(test_valid_idx[test_idx])

    leakage_score = len(test_with_similar) / len(test_smiles) if test_smiles else 0.0

    return {
        "n_similar_pairs": len(similar_pairs),
        "leakage_score": leakage_score,
        "examples": similar_pairs[:10],  # Show top 10 examples
        "warning": leakage_score > 0.1  # Flag if >10% of test set has similar train example
    }


def calculate_split_statistics(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    smiles_list: List[str],
    activities: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculate statistics about train/test split quality.

    Parameters:
    -----------
    train_idx, test_idx : np.ndarray
        Train and test indices
    smiles_list : List[str]
        Full list of SMILES
    activities : Optional[np.ndarray]
        Activity values (for distribution comparison)

    Returns:
    --------
    stats : Dict[str, Any]
        Statistics including scaffold diversity, size balance, activity distribution

    Example:
    --------
    >>> train_idx, test_idx = scaffold_split(smiles)
    >>> stats = calculate_split_statistics(train_idx, test_idx, smiles)
    >>> print(stats)
    """
    if not RDKIT_AVAILABLE:
        warnings.warn("RDKit not available, scaffold statistics will be limited")
        scaffolds_train = []
        scaffolds_test = []
    else:
        scaffolds_train = [generate_scaffold(smiles_list[i]) for i in train_idx]
        scaffolds_test = [generate_scaffold(smiles_list[i]) for i in test_idx]

    stats = {
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "train_fraction": len(train_idx) / (len(train_idx) + len(test_idx)),
        "n_unique_scaffolds_train": len(set(scaffolds_train)) if scaffolds_train else None,
        "n_unique_scaffolds_test": len(set(scaffolds_test)) if scaffolds_test else None,
        "scaffold_overlap": len(set(scaffolds_train) & set(scaffolds_test)) if scaffolds_train else None,
    }

    if activities is not None:
        from scipy import stats as scipy_stats
        train_activities = activities[train_idx]
        test_activities = activities[test_idx]

        # Kolmogorov-Smirnov test for distribution similarity
        ks_stat, ks_pval = scipy_stats.ks_2samp(train_activities, test_activities)

        stats.update({
            "train_activity_mean": float(np.mean(train_activities)),
            "train_activity_std": float(np.std(train_activities)),
            "test_activity_mean": float(np.mean(test_activities)),
            "test_activity_std": float(np.std(test_activities)),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "distributions_similar": ks_pval > 0.05
        })

    return stats


# Example usage and testing
if __name__ == "__main__":
    # Example with synthetic data
    print("Benchmarking Module - Example Usage")
    print("=" * 60)

    if RDKIT_AVAILABLE:
        # Example SMILES
        smiles_example = [
            "CCc1ccccc1",  # ethylbenzene
            "CCCc1ccccc1",  # propylbenzene
            "CCCCc1ccccc1",  # butylbenzene
            "c1cccnc1",  # pyridine
            "c1ccncc1",  # pyridine isomer
            "CCN1CCCCC1",  # N-ethylpiperidine
        ]

        activities = np.array([5.0, 5.2, 5.5, 7.0, 7.1, 6.5])

        print("\n1. Scaffold Split:")
        train_idx, test_idx = scaffold_split(smiles_example, test_size=0.33)
        print(f"   Train indices: {train_idx}")
        print(f"   Test indices: {test_idx}")

        print("\n2. Split Statistics:")
        stats = calculate_split_statistics(train_idx, test_idx, smiles_example, activities)
        for key, value in stats.items():
            print(f"   {key}: {value}")

        print("\n3. Data Leakage Detection:")
        leakage = detect_data_leakage(
            [smiles_example[i] for i in train_idx],
            [smiles_example[i] for i in test_idx]
        )
        print(f"   Leakage score: {leakage['leakage_score']:.2%}")
        print(f"   Warning: {leakage['warning']}")
    else:
        print("RDKit not available - install with: pip install rdkit")

    print("\n4. Temporal Split:")
    dates = np.array([2018, 2019, 2019, 2020, 2021, 2021])
    train_idx, test_idx = temporal_split(dates, test_size=0.33)
    print(f"   Train dates: {dates[train_idx]}")
    print(f"   Test dates: {dates[test_idx]}")

    print("\n" + "=" * 60)
    print("See docstrings for detailed usage examples")
