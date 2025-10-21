"""
Activity Cliff Detection and Analysis

Identifies and analyzes activity cliffs - pairs or groups of structurally similar
molecules with large differences in biological activity.

References:
    - Stumpfe & Bajorath (2012) "Exploring Activity Cliffs in Medicinal Chemistry"
      J. Med. Chem. 55, 2932-2942
    - Guha & Van Drie (2008) "Structure-Activity Landscape Index: Identifying and
      Quantifying Activity Cliffs" J. Chem. Inf. Model. 48, 646-658
    - Maggiora (2006) "On Outliers and Activity Cliffs - Why QSAR Often Disappoints"
      J. Chem. Inf. Model. 46, 1535
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


@dataclass
class ActivityCliff:
    """Activity cliff between two molecules."""

    mol1_idx: int
    mol2_idx: int
    mol1_smiles: str
    mol2_smiles: str
    similarity: float
    activity1: float
    activity2: float
    activity_difference: float
    sali: float  # Structure-Activity Landscape Index

    @property
    def cliff_strength(self) -> float:
        """Strength of cliff (normalized SALI)."""
        return abs(self.sali)

    def __repr__(self) -> str:
        return (
            f"ActivityCliff(sim={self.similarity:.3f}, "
            f"ΔActivity={self.activity_difference:.2f}, SALI={self.sali:.2f})"
        )


@dataclass
class CliffAnalysisResults:
    """Results from activity cliff analysis."""

    cliffs: List[ActivityCliff]
    n_molecules: int
    n_cliffs: int
    mean_sali: float
    max_sali: float
    cliff_molecules: List[int]  # Indices of molecules involved in cliffs
    cliff_pairs: List[Tuple[int, int]]

    def get_cliff_forming_molecules(self) -> List[int]:
        """Get indices of molecules that form activity cliffs."""
        return sorted(set(self.cliff_molecules))

    def get_top_cliffs(self, n: int = 10) -> List[ActivityCliff]:
        """Get top N cliffs by SALI magnitude."""
        return sorted(self.cliffs, key=lambda x: abs(x.sali), reverse=True)[:n]


class ActivityCliffAnalyzer:
    """
    Analyzer for detecting and characterizing activity cliffs.

    Activity cliffs are defined as pairs of molecules with:
    1. High structural similarity (Tanimoto > threshold)
    2. Large difference in activity (ΔActivity > threshold)

    Examples:
        >>> from rdkit import Chem
        >>> smiles = ["c1ccccc1", "c1ccc(Cl)cc1", "c1ccc([N+](=O)[O-])cc1"]
        >>> mols = [Chem.MolFromSmiles(s) for s in smiles]
        >>> activities = [5.0, 5.5, 8.5]  # nitro group causes large jump
        >>>
        >>> analyzer = ActivityCliffAnalyzer(similarity_threshold=0.7, activity_threshold=2.0)
        >>> results = analyzer.detect_cliffs(mols, activities)
        >>> print(f"Found {results.n_cliffs} activity cliffs")
        >>> for cliff in results.get_top_cliffs(5):
        ...     print(f"SALI={cliff.sali:.2f}: {cliff.mol1_smiles} vs {cliff.mol2_smiles}")
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        activity_threshold: float = 1.0,
        fingerprint_type: str = "morgan",
        radius: int = 2,
        n_bits: int = 2048,
    ):
        """
        Initialize activity cliff analyzer.

        Args:
            similarity_threshold: Minimum Tanimoto similarity to consider (0-1)
            activity_threshold: Minimum activity difference to be a cliff
            fingerprint_type: Type of fingerprint ('morgan', 'rdkit', 'topological')
            radius: Radius for Morgan fingerprint
            n_bits: Number of bits in fingerprint
        """
        self.similarity_threshold = similarity_threshold
        self.activity_threshold = activity_threshold
        self.fingerprint_type = fingerprint_type
        self.radius = radius
        self.n_bits = n_bits

    def detect_cliffs(
        self,
        mols: List[Chem.Mol],
        activities: List[float],
    ) -> CliffAnalysisResults:
        """
        Detect activity cliffs in dataset.

        Args:
            mols: List of molecules
            activities: List of activity values

        Returns:
            CliffAnalysisResults containing detected cliffs
        """
        if len(mols) != len(activities):
            raise ValueError(f"Mismatch: {len(mols)} molecules but {len(activities)} activities")

        logger.info(f"Detecting activity cliffs for {len(mols)} molecules...")

        # Calculate fingerprints
        fps = self._calculate_fingerprints(mols)

        # Calculate pairwise similarities
        n = len(mols)
        cliffs = []
        cliff_molecules = set()
        cliff_pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                # Calculate similarity
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])

                # Calculate activity difference
                act_diff = abs(activities[i] - activities[j])

                # Check if cliff
                if sim >= self.similarity_threshold and act_diff >= self.activity_threshold:
                    # Calculate SALI
                    sali = self._calculate_sali(sim, act_diff)

                    cliff = ActivityCliff(
                        mol1_idx=i,
                        mol2_idx=j,
                        mol1_smiles=Chem.MolToSmiles(mols[i]),
                        mol2_smiles=Chem.MolToSmiles(mols[j]),
                        similarity=sim,
                        activity1=activities[i],
                        activity2=activities[j],
                        activity_difference=act_diff,
                        sali=sali,
                    )
                    cliffs.append(cliff)
                    cliff_molecules.add(i)
                    cliff_molecules.add(j)
                    cliff_pairs.append((i, j))

        # Calculate statistics
        if cliffs:
            sali_values = [abs(c.sali) for c in cliffs]
            mean_sali = np.mean(sali_values)
            max_sali = np.max(sali_values)
        else:
            mean_sali = 0.0
            max_sali = 0.0

        results = CliffAnalysisResults(
            cliffs=cliffs,
            n_molecules=len(mols),
            n_cliffs=len(cliffs),
            mean_sali=mean_sali,
            max_sali=max_sali,
            cliff_molecules=list(cliff_molecules),
            cliff_pairs=cliff_pairs,
        )

        logger.info(f"Found {len(cliffs)} activity cliffs")
        logger.info(f"{len(cliff_molecules)} molecules involved in cliffs")

        return results

    def _calculate_fingerprints(self, mols: List[Chem.Mol]) -> List[DataStructs.ExplicitBitVect]:
        """Calculate molecular fingerprints."""
        fps = []

        for mol in mols:
            if mol is None:
                fps.append(None)
                continue

            if self.fingerprint_type == "morgan":
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.radius, nBits=self.n_bits
                )
            elif self.fingerprint_type == "rdkit":
                fp = Chem.RDKFingerprint(mol, fpSize=self.n_bits)
            elif self.fingerprint_type == "topological":
                fp = Chem.RDKFingerprint(mol, fpSize=self.n_bits)
            else:
                raise ValueError(f"Unknown fingerprint type: {self.fingerprint_type}")

            fps.append(fp)

        return fps

    def _calculate_sali(self, similarity: float, activity_diff: float) -> float:
        """
        Calculate Structure-Activity Landscape Index (SALI).

        SALI quantifies the "steepness" of an activity cliff:
            SALI = |ΔActivity| / (1 - Similarity)

        High SALI indicates a steep cliff (large activity change for small structural change).

        Args:
            similarity: Tanimoto similarity (0-1)
            activity_diff: Absolute activity difference

        Returns:
            SALI value
        """
        if similarity >= 1.0:
            # Identical molecules - avoid division by zero
            return 0.0

        return activity_diff / (1.0 - similarity)

    def calculate_similarity_matrix(
        self,
        mols: List[Chem.Mol],
    ) -> np.ndarray:
        """
        Calculate pairwise similarity matrix.

        Args:
            mols: List of molecules

        Returns:
            n×n similarity matrix
        """
        fps = self._calculate_fingerprints(mols)
        n = len(fps)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if fps[i] is None or fps[j] is None:
                    sim_matrix[i, j] = 0.0
                    sim_matrix[j, i] = 0.0
                else:
                    sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim

        return sim_matrix

    def identify_cliff_features(
        self,
        cliff: ActivityCliff,
        mol1: Chem.Mol,
        mol2: Chem.Mol,
    ) -> Dict[str, any]:
        """
        Identify structural features responsible for activity cliff.

        Args:
            cliff: ActivityCliff object
            mol1: First molecule
            mol2: Second molecule

        Returns:
            Dictionary with analysis of differing features
        """
        # Get Morgan fingerprints with feature info
        info1 = {}
        info2 = {}

        fp1 = AllChem.GetMorganFingerprint(mol1, self.radius, bitInfo=info1)
        fp2 = AllChem.GetMorganFingerprint(mol2, self.radius, bitInfo=info2)

        # Find unique features
        features1_only = set(info1.keys()) - set(info2.keys())
        features2_only = set(info2.keys()) - set(info1.keys())

        # Identify substructure differences using MCS
        from rdkit.Chem import rdFMCS

        mcs_result = rdFMCS.FindMCS(
            [mol1, mol2],
            timeout=10,
            bondCompare=rdFMCS.BondCompare.CompareOrderExact,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
        )

        mcs_smarts = mcs_result.smartsString if mcs_result else ""

        return {
            "unique_features_mol1": len(features1_only),
            "unique_features_mol2": len(features2_only),
            "mcs_smarts": mcs_smarts,
            "mcs_num_atoms": mcs_result.numAtoms if mcs_result else 0,
            "mcs_num_bonds": mcs_result.numBonds if mcs_result else 0,
        }

    def create_cliff_aware_splits(
        self,
        mols: List[Chem.Mol],
        activities: List[float],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[List[int], List[int]]:
        """
        Create train/test splits that avoid splitting activity cliffs.

        Ensures that cliff-forming pairs stay together in the same split
        to prevent data leakage.

        Args:
            mols: List of molecules
            activities: Activity values
            test_size: Fraction for test set
            random_state: Random seed

        Returns:
            Tuple of (train_indices, test_indices)
        """
        # Detect cliffs
        results = self.detect_cliffs(mols, activities)

        # Build cliff graph (molecules as nodes, cliffs as edges)
        from collections import defaultdict

        graph = defaultdict(list)
        for cliff in results.cliffs:
            graph[cliff.mol1_idx].append(cliff.mol2_idx)
            graph[cliff.mol2_idx].append(cliff.mol1_idx)

        # Find connected components (cliff clusters)
        visited = set()
        clusters = []

        def dfs(node, cluster):
            visited.add(node)
            cluster.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, cluster)

        for i in range(len(mols)):
            if i not in visited:
                cluster = []
                dfs(i, cluster)
                clusters.append(cluster)

        # Assign clusters to train/test
        np.random.seed(random_state)
        cluster_order = np.random.permutation(len(clusters))

        train_indices = []
        test_indices = []
        n_test_target = int(len(mols) * test_size)

        for idx in cluster_order:
            cluster = clusters[idx]
            if len(test_indices) < n_test_target:
                test_indices.extend(cluster)
            else:
                train_indices.extend(cluster)

        logger.info(f"Created cliff-aware splits: {len(train_indices)} train, {len(test_indices)} test")
        logger.info(f"Kept {len(clusters)} cliff clusters intact")

        return train_indices, test_indices


def detect_activity_cliffs(
    mols: List[Chem.Mol],
    activities: List[float],
    similarity_threshold: float = 0.7,
    activity_threshold: float = 1.0,
) -> CliffAnalysisResults:
    """
    Convenience function to detect activity cliffs.

    Args:
        mols: List of molecules
        activities: Activity values
        similarity_threshold: Minimum similarity to consider
        activity_threshold: Minimum activity difference

    Returns:
        CliffAnalysisResults

    Examples:
        >>> from rdkit import Chem
        >>> smiles = ["c1ccccc1", "c1ccc(Cl)cc1", "c1ccc([N+](=O)[O-])cc1"]
        >>> mols = [Chem.MolFromSmiles(s) for s in smiles]
        >>> activities = [5.0, 5.5, 8.5]
        >>> results = detect_activity_cliffs(mols, activities, activity_threshold=2.0)
        >>> print(f"Found {results.n_cliffs} cliffs")
    """
    analyzer = ActivityCliffAnalyzer(
        similarity_threshold=similarity_threshold,
        activity_threshold=activity_threshold,
    )
    return analyzer.detect_cliffs(mols, activities)


def calculate_sali_matrix(
    mols: List[Chem.Mol],
    activities: List[float],
) -> np.ndarray:
    """
    Calculate SALI (Structure-Activity Landscape Index) matrix.

    Args:
        mols: List of molecules
        activities: Activity values

    Returns:
        n×n SALI matrix
    """
    analyzer = ActivityCliffAnalyzer()

    # Get similarity matrix
    sim_matrix = analyzer.calculate_similarity_matrix(mols)

    # Calculate SALI matrix
    n = len(mols)
    sali_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            act_diff = abs(activities[i] - activities[j])

            if sim < 1.0:
                sali = analyzer._calculate_sali(sim, act_diff)
                sali_matrix[i, j] = sali
                sali_matrix[j, i] = sali

    return sali_matrix


if __name__ == "__main__":
    # Example usage
    from rdkit import Chem

    print("=== Activity Cliff Detection ===\n")

    # Example dataset with activity cliffs
    smiles_list = [
        "c1ccccc1",              # benzene: activity = 5.0
        "c1ccc(F)cc1",           # fluorobenzene: activity = 5.5
        "c1ccc(Cl)cc1",          # chlorobenzene: activity = 6.0
        "c1ccc([N+](=O)[O-])cc1",  # nitrobenzene: activity = 8.5 (cliff!)
        "c1ccc(N)cc1",           # aniline: activity = 4.5
        "c1ccc(O)cc1",           # phenol: activity = 5.8
    ]

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    activities = [5.0, 5.5, 6.0, 8.5, 4.5, 5.8]

    # Detect cliffs
    analyzer = ActivityCliffAnalyzer(
        similarity_threshold=0.7,
        activity_threshold=2.0,
    )
    results = analyzer.detect_cliffs(mols, activities)

    print(f"Dataset: {results.n_molecules} molecules")
    print(f"Found: {results.n_cliffs} activity cliffs")
    print(f"Mean SALI: {results.mean_sali:.2f}")
    print(f"Max SALI: {results.max_sali:.2f}")
    print()

    # Show top cliffs
    print("Top Activity Cliffs:")
    for i, cliff in enumerate(results.get_top_cliffs(5), 1):
        print(f"{i}. SALI={cliff.sali:.2f}")
        print(f"   Mol1: {cliff.mol1_smiles} (activity={cliff.activity1:.2f})")
        print(f"   Mol2: {cliff.mol2_smiles} (activity={cliff.activity2:.2f})")
        print(f"   Similarity: {cliff.similarity:.3f}")
        print(f"   ΔActivity: {cliff.activity_difference:.2f}")

        # Analyze features
        features = analyzer.identify_cliff_features(
            cliff, mols[cliff.mol1_idx], mols[cliff.mol2_idx]
        )
        print(f"   MCS: {features['mcs_smarts']} ({features['mcs_num_atoms']} atoms)")
        print()

    # Cliff-aware splitting
    print("\nCliff-Aware Train/Test Split:")
    train_idx, test_idx = analyzer.create_cliff_aware_splits(
        mols, activities, test_size=0.3
    )
    print(f"Train set: {len(train_idx)} molecules (indices: {sorted(train_idx)})")
    print(f"Test set: {len(test_idx)} molecules (indices: {sorted(test_idx)})")
