"""
Advanced molecular cleaning and standardization utilities.

This module provides functions for removing salts, neutralizing charges,
standardizing tautomers, removing duplicates, and filtering by properties.
Includes PAINS filters and aggregator detection.
"""

from typing import Dict, Optional, List, Tuple, Set
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, MolStandardize
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem import FilterCatalog
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    raise ImportError("RDKit is required. Install with: pip install rdkit")


# PAINS patterns (Pan-Assay Interference Compounds)
PAINS_A = [
    "[OH,NH2,SH]c1ccc([OH,NH2,SH])cc1",  # Catechol, aminophenol
    "c1ccc2c(c1)C(=O)c1ccccc1C2=O",  # Anthraquinone
    "[#6]S(=O)(=O)[OH]",  # Sulfonic acid
    "P(=O)([OH])[OH]",  # Phosphonic acid
    "N=N",  # Azo
    "[N+](=O)[O-]",  # Nitro
    "c1cc[nH+]cc1",  # Pyridinium
    "c1c([OH])c([OH])ccc1",  # Catechol
    "C=CC(=O)[OH]",  # Alpha,beta-unsaturated carbonyl
]

# Aggregator patterns
AGGREGATOR_SMARTS = [
    "c1ccc(cc1)c2ccccc2",  # Biphenyl
    "c1ccc2c(c1)cccc2c3ccccc3",  # Polyaromatic
]

# Reactive/unstable groups
REACTIVE_SMARTS = [
    "C(=O)Cl",  # Acyl chloride
    "S(=O)(=O)Cl",  # Sulfonyl chloride
    "[N+](=O)[O-]",  # Nitro
    "N=C=O",  # Isocyanate
    "N=C=S",  # Isothiocyanate
    "C=C=O",  # Ketene
    "P(=O)([OH])[OH]",  # Phosphonic acid
    "C[N+](C)(C)C",  # Quaternary ammonium
]


class MolecularStandardizer:
    """Standardize molecular structures."""

    def __init__(self):
        """Initialize standardizer."""
        self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
        self.tautomer_enumerator.SetMaxTautomers(100)

        self.uncharger = rdMolStandardize.Uncharger()
        self.fragment_remover = rdMolStandardize.LargestFragmentChooser()

    def standardize(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Standardize molecule (remove salts, neutralize, canonical tautomer).

        Args:
            mol: RDKit molecule

        Returns:
            Standardized molecule
        """
        if mol is None:
            return None

        try:
            # Remove fragments (salts)
            mol = self.fragment_remover.choose(mol)

            # Neutralize
            mol = self.uncharger.uncharge(mol)

            # Canonical tautomer
            mol = self.tautomer_enumerator.Canonicalize(mol)

            # Sanitize
            Chem.SanitizeMol(mol)

            return mol

        except Exception as e:
            warnings.warn(f"Standardization failed: {e}")
            return None

    def enumerate_tautomers(
        self,
        mol: Chem.Mol,
        max_tautomers: int = 10
    ) -> List[Chem.Mol]:
        """
        Enumerate tautomers.

        Args:
            mol: RDKit molecule
            max_tautomers: Maximum number of tautomers

        Returns:
            List of tautomer molecules
        """
        if mol is None:
            return []

        try:
            self.tautomer_enumerator.SetMaxTautomers(max_tautomers)
            tautomers = self.tautomer_enumerator.Enumerate(mol)
            return list(tautomers)
        except:
            return [mol]


class PAINSFilter:
    """Filter Pan-Assay Interference Compounds (PAINS)."""

    def __init__(self, include_default: bool = True):
        """
        Initialize PAINS filter.

        Args:
            include_default: Include default RDKit PAINS patterns
        """
        self.patterns = []

        if include_default:
            # Use RDKit's built-in PAINS catalog
            params = FilterCatalog.FilterCatalogParams()
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
            self.catalog = FilterCatalog.FilterCatalog(params)
        else:
            self.catalog = None

        # Add custom PAINS patterns
        for smarts in PAINS_A:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                self.patterns.append(pattern)

    def is_pains(self, mol: Chem.Mol) -> Tuple[bool, List[str]]:
        """
        Check if molecule is a PAINS compound.

        Args:
            mol: RDKit molecule

        Returns:
            Tuple of (is_pains, list of matched pattern names)
        """
        if mol is None:
            return False, []

        matches = []

        # Check RDKit catalog
        if self.catalog:
            entry = self.catalog.GetFirstMatch(mol)
            if entry:
                matches.append(entry.GetDescription())

        # Check custom patterns
        for pattern in self.patterns:
            if mol.HasSubstructMatch(pattern):
                matches.append(Chem.MolToSmarts(pattern))

        return len(matches) > 0, matches


class AggregatorFilter:
    """Filter potential aggregator compounds."""

    def __init__(self):
        """Initialize aggregator filter."""
        self.patterns = []

        for smarts in AGGREGATOR_SMARTS:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                self.patterns.append(pattern)

    def is_aggregator(self, mol: Chem.Mol) -> bool:
        """
        Check if molecule is likely to aggregate.

        Args:
            mol: RDKit molecule

        Returns:
            True if likely aggregator
        """
        if mol is None:
            return False

        # Check structural patterns
        for pattern in self.patterns:
            if mol.HasSubstructMatch(pattern):
                return True

        # Check properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        n_aromatic_rings = Descriptors.NumAromaticRings(mol)

        # High MW + high logP + multiple aromatic rings
        if mw > 400 and logp > 4 and n_aromatic_rings >= 3:
            return True

        return False


class ReactiveFilter:
    """Filter reactive/unstable functional groups."""

    def __init__(self):
        """Initialize reactive filter."""
        self.patterns = []

        for smarts in REACTIVE_SMARTS:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                self.patterns.append((smarts, pattern))

    def has_reactive_groups(self, mol: Chem.Mol) -> Tuple[bool, List[str]]:
        """
        Check for reactive functional groups.

        Args:
            mol: RDKit molecule

        Returns:
            Tuple of (has_reactive, list of matched groups)
        """
        if mol is None:
            return False, []

        matches = []

        for smarts, pattern in self.patterns:
            if mol.HasSubstructMatch(pattern):
                matches.append(smarts)

        return len(matches) > 0, matches


class PropertyFilter:
    """Filter molecules by physicochemical properties."""

    def __init__(
        self,
        mw_range: Tuple[float, float] = (0, 1000),
        logp_range: Tuple[float, float] = (-10, 10),
        hbd_range: Tuple[int, int] = (0, 10),
        hba_range: Tuple[int, int] = (0, 20),
        tpsa_range: Tuple[float, float] = (0, 200),
        rotatable_bonds_range: Tuple[int, int] = (0, 20),
        rings_range: Tuple[int, int] = (0, 10)
    ):
        """
        Initialize property filter.

        Args:
            mw_range: Molecular weight range
            logp_range: LogP range
            hbd_range: H-bond donors range
            hba_range: H-bond acceptors range
            tpsa_range: Topological polar surface area range
            rotatable_bonds_range: Rotatable bonds range
            rings_range: Number of rings range
        """
        self.mw_range = mw_range
        self.logp_range = logp_range
        self.hbd_range = hbd_range
        self.hba_range = hba_range
        self.tpsa_range = tpsa_range
        self.rotatable_bonds_range = rotatable_bonds_range
        self.rings_range = rings_range

    def passes_filters(self, mol: Chem.Mol) -> Tuple[bool, Dict]:
        """
        Check if molecule passes property filters.

        Args:
            mol: RDKit molecule

        Returns:
            Tuple of (passes, dict of properties)
        """
        if mol is None:
            return False, {}

        props = {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'rings': Descriptors.RingCount(mol)
        }

        passes = (
            self.mw_range[0] <= props['mw'] <= self.mw_range[1] and
            self.logp_range[0] <= props['logp'] <= self.logp_range[1] and
            self.hbd_range[0] <= props['hbd'] <= self.hbd_range[1] and
            self.hba_range[0] <= props['hba'] <= self.hba_range[1] and
            self.tpsa_range[0] <= props['tpsa'] <= self.tpsa_range[1] and
            self.rotatable_bonds_range[0] <= props['rotatable_bonds'] <= self.rotatable_bonds_range[1] and
            self.rings_range[0] <= props['rings'] <= self.rings_range[1]
        )

        return passes, props

    @staticmethod
    def rule_of_five(mol: Chem.Mol) -> Tuple[bool, int]:
        """
        Check Lipinski's Rule of Five.

        Args:
            mol: RDKit molecule

        Returns:
            Tuple of (passes, number of violations)
        """
        if mol is None:
            return False, 5

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1

        return violations <= 1, violations

    @staticmethod
    def rule_of_three(mol: Chem.Mol) -> Tuple[bool, int]:
        """
        Check Rule of Three (for fragments).

        Args:
            mol: RDKit molecule

        Returns:
            Tuple of (passes, number of violations)
        """
        if mol is None:
            return False, 4

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)

        violations = 0
        if mw > 300:
            violations += 1
        if logp > 3:
            violations += 1
        if hbd > 3:
            violations += 1
        if hba > 3:
            violations += 1
        if rotatable_bonds > 3:
            violations += 1
        if tpsa > 60:
            violations += 1

        return violations == 0, violations


class DuplicateRemover:
    """Remove duplicate molecules."""

    @staticmethod
    def remove_by_inchikey(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicates by InChIKey.

        Args:
            df: DataFrame with 'mol' column

        Returns:
            DataFrame with duplicates removed
        """
        from tqdm import tqdm

        print("Generating InChIKeys...")
        df['_inchikey'] = [Chem.MolToInchiKey(mol) if mol else None
                          for mol in tqdm(df['mol'])]

        before = len(df)
        df = df.drop_duplicates(subset=['_inchikey'], keep='first')
        after = len(df)

        print(f"Removed {before - after} duplicates by InChIKey")

        return df.drop(columns=['_inchikey'])

    @staticmethod
    def remove_by_canonical_smiles(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicates by canonical SMILES.

        Args:
            df: DataFrame with 'mol' column

        Returns:
            DataFrame with duplicates removed
        """
        from tqdm import tqdm

        print("Generating canonical SMILES...")
        df['_canonical_smiles'] = [Chem.MolToSmiles(mol) if mol else None
                                   for mol in tqdm(df['mol'])]

        before = len(df)
        df = df.drop_duplicates(subset=['_canonical_smiles'], keep='first')
        after = len(df)

        print(f"Removed {before - after} duplicates by canonical SMILES")

        return df.drop(columns=['_canonical_smiles'])


class ComprehensiveCleaner:
    """Comprehensive molecular cleaning pipeline."""

    def __init__(
        self,
        standardize: bool = True,
        remove_pains: bool = True,
        remove_aggregators: bool = True,
        remove_reactive: bool = True,
        apply_rule_of_five: bool = False,
        apply_rule_of_three: bool = False,
        custom_property_filter: Optional[PropertyFilter] = None,
        remove_duplicates: bool = True
    ):
        """
        Initialize comprehensive cleaner.

        Args:
            standardize: Standardize molecules
            remove_pains: Remove PAINS compounds
            remove_aggregators: Remove aggregators
            remove_reactive: Remove reactive compounds
            apply_rule_of_five: Apply Lipinski's Rule of Five
            apply_rule_of_three: Apply Rule of Three
            custom_property_filter: Custom property filter
            remove_duplicates: Remove duplicates
        """
        self.standardize = standardize
        self.remove_pains = remove_pains
        self.remove_aggregators = remove_aggregators
        self.remove_reactive = remove_reactive
        self.apply_rule_of_five = apply_rule_of_five
        self.apply_rule_of_three = apply_rule_of_three
        self.custom_property_filter = custom_property_filter
        self.remove_duplicates = remove_duplicates

        if standardize:
            self.standardizer = MolecularStandardizer()
        if remove_pains:
            self.pains_filter = PAINSFilter()
        if remove_aggregators:
            self.aggregator_filter = AggregatorFilter()
        if remove_reactive:
            self.reactive_filter = ReactiveFilter()

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean molecular dataset.

        Args:
            df: DataFrame with 'mol' column

        Returns:
            Tuple of (cleaned_df, statistics_dict)
        """
        from tqdm import tqdm

        initial_count = len(df)
        stats = {'initial': initial_count}

        # Remove invalid molecules
        df = df[df['mol'].notna()].copy()
        stats['after_invalid_removal'] = len(df)

        # Standardize
        if self.standardize:
            print("Standardizing molecules...")
            df['mol'] = [self.standardizer.standardize(mol)
                        for mol in tqdm(df['mol'])]
            df = df[df['mol'].notna()].copy()
            stats['after_standardization'] = len(df)

        # Remove PAINS
        if self.remove_pains:
            print("Removing PAINS compounds...")
            pains_flags = [self.pains_filter.is_pains(mol)[0]
                          for mol in tqdm(df['mol'])]
            df = df[~np.array(pains_flags)].copy()
            stats['after_pains_removal'] = len(df)

        # Remove aggregators
        if self.remove_aggregators:
            print("Removing aggregators...")
            agg_flags = [self.aggregator_filter.is_aggregator(mol)
                        for mol in tqdm(df['mol'])]
            df = df[~np.array(agg_flags)].copy()
            stats['after_aggregator_removal'] = len(df)

        # Remove reactive
        if self.remove_reactive:
            print("Removing reactive compounds...")
            reactive_flags = [self.reactive_filter.has_reactive_groups(mol)[0]
                             for mol in tqdm(df['mol'])]
            df = df[~np.array(reactive_flags)].copy()
            stats['after_reactive_removal'] = len(df)

        # Rule of Five
        if self.apply_rule_of_five:
            print("Applying Rule of Five...")
            ro5_flags = [PropertyFilter.rule_of_five(mol)[0]
                        for mol in tqdm(df['mol'])]
            df = df[ro5_flags].copy()
            stats['after_rule_of_five'] = len(df)

        # Rule of Three
        if self.apply_rule_of_three:
            print("Applying Rule of Three...")
            ro3_flags = [PropertyFilter.rule_of_three(mol)[0]
                        for mol in tqdm(df['mol'])]
            df = df[ro3_flags].copy()
            stats['after_rule_of_three'] = len(df)

        # Custom property filter
        if self.custom_property_filter:
            print("Applying custom property filter...")
            prop_flags = [self.custom_property_filter.passes_filters(mol)[0]
                         for mol in tqdm(df['mol'])]
            df = df[prop_flags].copy()
            stats['after_property_filter'] = len(df)

        # Remove duplicates
        if self.remove_duplicates:
            print("Removing duplicates...")
            df = DuplicateRemover.remove_by_inchikey(df)
            stats['after_deduplication'] = len(df)

        stats['final'] = len(df)
        stats['removed'] = initial_count - len(df)
        stats['retention_rate'] = 100 * len(df) / initial_count

        return df, stats


if __name__ == "__main__":
    print("Molecular Cleaning Utilities")
    print("=" * 50)

    # Example molecules
    smiles_list = [
        "CCO",  # Ethanol - clean
        "c1ccccc1",  # Benzene - clean
        "CC(=O)Cl",  # Acetyl chloride - reactive
        "c1ccc(O)c(O)c1",  # Catechol - PAINS
        "c1ccc(cc1)c2ccccc2c3ccccc3",  # Polyaromatic - aggregator
        "CC(C)(C)c1ccc(O)c(c1)c2ccccc2",  # Large lipophilic - aggregator
    ]

    from .data_processing import MoleculeDataLoader
    df = MoleculeDataLoader.from_smiles_list(smiles_list)

    print(f"\nInitial: {len(df)} molecules")

    # Test PAINS filter
    print("\n" + "=" * 50)
    print("PAINS Filter:")
    pains_filter = PAINSFilter()
    for idx, mol in enumerate(df['mol']):
        is_pains, matches = pains_filter.is_pains(mol)
        if is_pains:
            print(f"  Molecule {idx}: PAINS (matches: {matches})")

    # Test aggregator filter
    print("\n" + "=" * 50)
    print("Aggregator Filter:")
    agg_filter = AggregatorFilter()
    for idx, mol in enumerate(df['mol']):
        if agg_filter.is_aggregator(mol):
            print(f"  Molecule {idx}: Potential aggregator")

    # Test reactive filter
    print("\n" + "=" * 50)
    print("Reactive Filter:")
    reactive_filter = ReactiveFilter()
    for idx, mol in enumerate(df['mol']):
        has_reactive, groups = reactive_filter.has_reactive_groups(mol)
        if has_reactive:
            print(f"  Molecule {idx}: Reactive (groups: {groups})")

    # Comprehensive cleaning
    print("\n" + "=" * 50)
    print("Comprehensive Cleaning:")
    cleaner = ComprehensiveCleaner(
        remove_pains=True,
        remove_aggregators=True,
        remove_reactive=True
    )
    df_clean, stats = cleaner.clean(df)

    print(f"\nCleaning Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
