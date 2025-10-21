"""
Matched Molecular Pair Analysis (MMPA)

Implements efficient algorithms for identifying matched molecular pairs (MMPs) and
analyzing structural transformations with associated property changes.

References:
    - Hussain & Rea (2010) "Computationally Efficient Algorithm to Identify
      Matched Molecular Pairs (MMPs) in Large Data Sets" J. Chem. Inf. Model. 50, 339-348
    - Lewell et al. (1998) "RECAP-Retrosynthetic Combinatorial Analysis Procedure"
      J. Chem. Inf. Comput. Sci. 38, 511-522
    - Warner et al. (2010) "Matched Molecular Pair Analysis (MMPA) Highlights the
      Impact of Nonoptimized Substructures" J. Med. Chem. 53, 4035-4044
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMMPA
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class MolecularPair:
    """Matched molecular pair with transformation information."""

    mol1_idx: int
    mol2_idx: int
    mol1_smiles: str
    mol2_smiles: str
    core_smiles: str
    transform_smiles: str  # Format: "[*]R1>>[*]R2"
    property_change: float
    property1: float
    property2: float

    @property
    def transform_key(self) -> str:
        """Canonical transformation key."""
        parts = self.transform_smiles.split(">>")
        if len(parts) == 2:
            # Ensure consistent ordering (smaller SMILES first)
            r1, r2 = sorted([parts[0], parts[1]])
            return f"{r1}>>{r2}"
        return self.transform_smiles


@dataclass
class TransformationStatistics:
    """Statistics for a molecular transformation."""

    transformation: str  # Format: "R1>>R2"
    n_pairs: int
    mean_change: float
    std_change: float
    median_change: float
    min_change: float
    max_change: float
    p_value: float  # Statistical significance
    effect_size: float  # Cohen's d
    pairs: List[MolecularPair]

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if transformation effect is statistically significant."""
        return self.p_value < alpha and self.n_pairs >= 3

    def __repr__(self) -> str:
        return (
            f"TransformationStatistics({self.transformation}, "
            f"n={self.n_pairs}, Δ={self.mean_change:.2f}±{self.std_change:.2f}, "
            f"p={self.p_value:.4f})"
        )


class MatchedMolecularPairAnalyzer:
    """
    Analyzer for matched molecular pairs with property-based filtering.

    Examples:
        >>> from rdkit import Chem
        >>> smiles = ["c1ccccc1", "c1ccc(Cl)cc1", "c1ccc(F)cc1", "c1ccc(Br)cc1"]
        >>> mols = [Chem.MolFromSmiles(s) for s in smiles]
        >>> activities = [5.0, 6.5, 6.8, 7.2]
        >>>
        >>> analyzer = MatchedMolecularPairAnalyzer()
        >>> pairs = analyzer.find_pairs(mols, activities)
        >>> print(f"Found {len(pairs)} matched pairs")
        >>>
        >>> # Analyze transformations
        >>> stats = analyzer.analyze_transformations(pairs)
        >>> for trans, stat in sorted(stats.items(), key=lambda x: abs(x[1].mean_change), reverse=True):
        ...     if stat.is_significant():
        ...         print(f"{stat.transformation}: Δ={stat.mean_change:.2f} (p={stat.p_value:.4f})")
    """

    def __init__(
        self,
        max_variable_size: int = 13,
        min_variable_size: int = 1,
        max_cuts: int = 2,
        n_jobs: int = -1,
    ):
        """
        Initialize MMPA analyzer.

        Args:
            max_variable_size: Maximum size of variable fragment (heavy atoms)
            min_variable_size: Minimum size of variable fragment (heavy atoms)
            max_cuts: Maximum number of cuts to make (1=single cut, 2=double cut)
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        """
        self.max_variable_size = max_variable_size
        self.min_variable_size = min_variable_size
        self.max_cuts = max_cuts
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    def find_pairs(
        self,
        mols: List[Chem.Mol],
        properties: List[float],
        mol_ids: Optional[List[str]] = None,
    ) -> List[MolecularPair]:
        """
        Find all matched molecular pairs in dataset.

        Args:
            mols: List of RDKit molecules
            properties: List of property values (e.g., activities, logP)
            mol_ids: Optional molecule identifiers

        Returns:
            List of MolecularPair objects
        """
        if len(mols) != len(properties):
            raise ValueError(f"Mismatch: {len(mols)} molecules but {len(properties)} properties")

        if mol_ids is None:
            mol_ids = [str(i) for i in range(len(mols))]

        logger.info(f"Finding matched pairs for {len(mols)} molecules...")

        # Generate molecular fragments using RDKit's MMPA
        mol_fragments = self._fragment_molecules(mols)

        # Index fragments by core
        core_index = defaultdict(list)
        for idx, frags in enumerate(mol_fragments):
            for core, context in frags:
                core_index[core].append((idx, context))

        # Find pairs with same core
        pairs = []
        for core, contexts in core_index.items():
            if len(contexts) < 2:
                continue

            # All pairwise combinations
            for i in range(len(contexts)):
                for j in range(i + 1, len(contexts)):
                    idx1, ctx1 = contexts[i]
                    idx2, ctx2 = contexts[j]

                    # Skip if same molecule
                    if idx1 == idx2:
                        continue

                    # Create transformation string
                    transform = f"{ctx1}>>{ctx2}"

                    pair = MolecularPair(
                        mol1_idx=idx1,
                        mol2_idx=idx2,
                        mol1_smiles=Chem.MolToSmiles(mols[idx1]),
                        mol2_smiles=Chem.MolToSmiles(mols[idx2]),
                        core_smiles=core,
                        transform_smiles=transform,
                        property_change=properties[idx2] - properties[idx1],
                        property1=properties[idx1],
                        property2=properties[idx2],
                    )
                    pairs.append(pair)

        logger.info(f"Found {len(pairs)} matched molecular pairs")
        return pairs

    def _fragment_molecules(
        self, mols: List[Chem.Mol]
    ) -> List[List[Tuple[str, str]]]:
        """
        Fragment molecules using MMPA algorithm.

        Returns:
            List of fragment lists, where each fragment is (core_smiles, context_smiles)
        """
        all_fragments = []

        for mol in mols:
            if mol is None:
                all_fragments.append([])
                continue

            # Use RDKit's MMPA fragmentation
            try:
                frags = self._fragment_mol(mol)
                all_fragments.append(frags)
            except Exception as e:
                logger.warning(f"Failed to fragment molecule: {e}")
                all_fragments.append([])

        return all_fragments

    def _fragment_mol(self, mol: Chem.Mol) -> List[Tuple[str, str]]:
        """Fragment a single molecule."""
        fragments = []

        # Get all single cuts
        if self.max_cuts >= 1:
            frags = rdMMPA.FragmentMol(
                mol,
                maxCuts=1,
                resultsAsMols=False,
                maxCutBonds=30,
            )
            for core, chains in frags:
                # Filter by variable size
                for chain in chains.split('.'):
                    chain_mol = Chem.MolFromSmiles(chain)
                    if chain_mol is None:
                        continue
                    n_heavy = chain_mol.GetNumHeavyAtoms()
                    if self.min_variable_size <= n_heavy <= self.max_variable_size:
                        fragments.append((core, chain))

        # Get all double cuts if requested
        if self.max_cuts >= 2:
            frags = rdMMPA.FragmentMol(
                mol,
                maxCuts=2,
                resultsAsMols=False,
                maxCutBonds=30,
            )
            for core, chains in frags:
                for chain in chains.split('.'):
                    chain_mol = Chem.MolFromSmiles(chain)
                    if chain_mol is None:
                        continue
                    n_heavy = chain_mol.GetNumHeavyAtoms()
                    if self.min_variable_size <= n_heavy <= self.max_variable_size:
                        fragments.append((core, chain))

        return fragments

    def analyze_transformations(
        self,
        pairs: List[MolecularPair],
        min_pairs: int = 3,
    ) -> Dict[str, TransformationStatistics]:
        """
        Analyze statistics for each transformation.

        Args:
            pairs: List of molecular pairs
            min_pairs: Minimum number of pairs required for statistics

        Returns:
            Dictionary mapping transformation to statistics
        """
        # Group pairs by transformation
        transform_groups = defaultdict(list)
        for pair in pairs:
            transform_groups[pair.transform_key].append(pair)

        # Calculate statistics for each transformation
        stats = {}
        for transform, group_pairs in transform_groups.items():
            if len(group_pairs) < min_pairs:
                continue

            changes = [p.property_change for p in group_pairs]

            # Statistical tests
            # One-sample t-test against zero (no change)
            if len(changes) >= 3:
                t_stat, p_value = stats.ttest_1samp(changes, 0)
            else:
                t_stat, p_value = 0.0, 1.0

            # Effect size (Cohen's d)
            mean_change = np.mean(changes)
            std_change = np.std(changes, ddof=1) if len(changes) > 1 else 0.0
            effect_size = mean_change / std_change if std_change > 0 else 0.0

            stat = TransformationStatistics(
                transformation=transform,
                n_pairs=len(group_pairs),
                mean_change=mean_change,
                std_change=std_change,
                median_change=np.median(changes),
                min_change=np.min(changes),
                max_change=np.max(changes),
                p_value=p_value,
                effect_size=effect_size,
                pairs=group_pairs,
            )
            stats[transform] = stat

        return stats

    def find_activity_cliffs(
        self,
        pairs: List[MolecularPair],
        cliff_threshold: float = 2.0,
    ) -> List[MolecularPair]:
        """
        Find activity cliffs (small structural change, large property change).

        Args:
            pairs: List of molecular pairs
            cliff_threshold: Minimum property change to be considered a cliff

        Returns:
            List of pairs that form activity cliffs
        """
        cliffs = []
        for pair in pairs:
            if abs(pair.property_change) >= cliff_threshold:
                cliffs.append(pair)

        logger.info(f"Found {len(cliffs)} activity cliffs (threshold={cliff_threshold})")
        return cliffs

    def get_top_transformations(
        self,
        stats: Dict[str, TransformationStatistics],
        n: int = 10,
        sort_by: str = "mean_change",
        only_significant: bool = True,
    ) -> List[TransformationStatistics]:
        """
        Get top N transformations by specified metric.

        Args:
            stats: Transformation statistics dictionary
            n: Number of top transformations to return
            sort_by: Metric to sort by ('mean_change', 'effect_size', 'n_pairs')
            only_significant: Only include statistically significant transformations

        Returns:
            List of top TransformationStatistics
        """
        filtered = [
            s for s in stats.values()
            if not only_significant or s.is_significant()
        ]

        if sort_by == "mean_change":
            sorted_stats = sorted(filtered, key=lambda x: abs(x.mean_change), reverse=True)
        elif sort_by == "effect_size":
            sorted_stats = sorted(filtered, key=lambda x: abs(x.effect_size), reverse=True)
        elif sort_by == "n_pairs":
            sorted_stats = sorted(filtered, key=lambda x: x.n_pairs, reverse=True)
        else:
            raise ValueError(f"Unknown sort_by: {sort_by}")

        return sorted_stats[:n]


def find_matched_pairs(
    mols: List[Chem.Mol],
    properties: List[float],
    max_variable_size: int = 13,
    min_pairs: int = 3,
) -> Tuple[List[MolecularPair], Dict[str, TransformationStatistics]]:
    """
    Convenience function to find matched pairs and analyze transformations.

    Args:
        mols: List of RDKit molecules
        properties: List of property values
        max_variable_size: Maximum size of variable fragment
        min_pairs: Minimum pairs for transformation statistics

    Returns:
        Tuple of (all pairs, transformation statistics)

    Examples:
        >>> from rdkit import Chem
        >>> smiles = ["c1ccccc1", "c1ccc(Cl)cc1", "c1ccc(F)cc1"]
        >>> mols = [Chem.MolFromSmiles(s) for s in smiles]
        >>> activities = [5.0, 6.5, 6.8]
        >>> pairs, stats = find_matched_pairs(mols, activities)
        >>> print(f"Found {len(pairs)} pairs, {len(stats)} transformations")
    """
    analyzer = MatchedMolecularPairAnalyzer(max_variable_size=max_variable_size)
    pairs = analyzer.find_pairs(mols, properties)
    stats = analyzer.analyze_transformations(pairs, min_pairs=min_pairs)

    return pairs, stats


def group_transformations_by_type(
    stats: Dict[str, TransformationStatistics]
) -> Dict[str, List[TransformationStatistics]]:
    """
    Group transformations by chemical type.

    Args:
        stats: Transformation statistics dictionary

    Returns:
        Dictionary mapping transformation type to list of statistics

    Examples of types:
        - "halogenation": H→F, H→Cl, H→Br
        - "methylation": H→CH3
        - "ring_modification": phenyl→pyridyl
    """
    groups = defaultdict(list)

    for stat in stats.values():
        # Parse transformation
        parts = stat.transformation.split(">>")
        if len(parts) != 2:
            groups["other"].append(stat)
            continue

        r1, r2 = parts[0].replace("[*]", ""), parts[1].replace("[*]", "")

        # Classify transformation type
        if r1 == "[H]" and r2 in ["F", "Cl", "Br", "I"]:
            groups["halogenation"].append(stat)
        elif r1 == "[H]" and "C" in r2:
            if "O" in r2:
                groups["oxygenation"].append(stat)
            elif "N" in r2:
                groups["amination"].append(stat)
            else:
                groups["alkylation"].append(stat)
        elif "c1" in r1 and "c1" in r2:
            groups["ring_modification"].append(stat)
        else:
            groups["other"].append(stat)

    return dict(groups)


if __name__ == "__main__":
    # Example usage
    from rdkit import Chem

    # Example dataset: halogenated benzenes
    smiles_list = [
        "c1ccccc1",          # benzene
        "Fc1ccccc1",         # fluorobenzene
        "Clc1ccccc1",        # chlorobenzene
        "Brc1ccccc1",        # bromobenzene
        "Fc1ccc(F)cc1",      # 1,4-difluorobenzene
        "Clc1ccc(Cl)cc1",    # 1,4-dichlorobenzene
        "Cc1ccccc1",         # toluene
        "Cc1ccc(C)cc1",      # p-xylene
    ]

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]

    # Synthetic activities (for demonstration)
    activities = [5.0, 5.5, 6.0, 6.5, 6.0, 6.8, 5.2, 5.5]

    print("=== Matched Molecular Pair Analysis ===\n")

    # Find pairs and analyze
    analyzer = MatchedMolecularPairAnalyzer()
    pairs = analyzer.find_pairs(mols, activities)
    print(f"Found {len(pairs)} matched molecular pairs\n")

    # Analyze transformations
    stats = analyzer.analyze_transformations(pairs, min_pairs=1)
    print(f"Identified {len(stats)} unique transformations\n")

    # Show top transformations
    print("Top transformations by effect size:")
    top = analyzer.get_top_transformations(stats, n=5, sort_by="mean_change", only_significant=False)
    for i, stat in enumerate(top, 1):
        print(f"{i}. {stat.transformation}")
        print(f"   N={stat.n_pairs}, Δ={stat.mean_change:.2f}±{stat.std_change:.2f}")
        print(f"   p={stat.p_value:.4f}, d={stat.effect_size:.2f}")
        print()

    # Find activity cliffs
    cliffs = analyzer.find_activity_cliffs(pairs, cliff_threshold=1.0)
    print(f"\nActivity cliffs (ΔActivity ≥ 1.0): {len(cliffs)}")
    for cliff in cliffs[:3]:
        print(f"  {cliff.mol1_smiles} → {cliff.mol2_smiles}")
        print(f"  Transformation: {cliff.transform_smiles}")
        print(f"  ΔActivity: {cliff.property_change:.2f}")
        print()
