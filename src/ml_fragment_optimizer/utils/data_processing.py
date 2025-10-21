"""
Core data processing utilities for molecular datasets.

This module provides functions for loading, cleaning, standardizing, and
augmenting molecular datasets for machine learning applications.
"""

from typing import Optional, List, Tuple, Dict, Union, Callable
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from dataclasses import dataclass
from enum import Enum
import json

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, SaltRemover
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    raise ImportError("RDKit is required. Install with: pip install rdkit")

from scipy.optimize import curve_fit
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class SplitStrategy(Enum):
    """Dataset splitting strategies."""
    RANDOM = "random"
    SCAFFOLD = "scaffold"
    TEMPORAL = "temporal"
    STRATIFIED = "stratified"
    CLUSTER = "cluster"


@dataclass
class DatasetStatistics:
    """Statistics for a molecular dataset."""
    n_molecules: int
    n_valid: int
    n_invalid: int
    n_duplicates: int
    mw_mean: float
    mw_std: float
    logp_mean: float
    logp_std: float
    n_rotatable_bonds_mean: float
    n_hbd_mean: float
    n_hba_mean: float


class MoleculeDataLoader:
    """Load molecules from various file formats."""

    @staticmethod
    def from_smiles_file(
        filepath: Union[str, Path],
        smiles_col: str = "smiles",
        id_col: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load molecules from SMILES file (CSV/TSV).

        Args:
            filepath: Path to file containing SMILES
            smiles_col: Column name containing SMILES strings
            id_col: Column name for molecule IDs
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame with 'mol', 'smiles', and other columns
        """
        filepath = Path(filepath)

        # Auto-detect delimiter
        if filepath.suffix == '.tsv':
            kwargs.setdefault('sep', '\t')

        df = pd.read_csv(filepath, **kwargs)

        if smiles_col not in df.columns:
            raise ValueError(f"Column '{smiles_col}' not found in file")

        # Convert SMILES to mol objects
        print(f"Loading {len(df)} molecules from {filepath.name}...")
        df['mol'] = [Chem.MolFromSmiles(s) if pd.notna(s) else None
                     for s in tqdm(df[smiles_col], desc="Parsing SMILES")]

        # Rename smiles column if needed
        if smiles_col != 'smiles':
            df['smiles'] = df[smiles_col]

        # Add ID column if not present
        if id_col and id_col in df.columns:
            df['mol_id'] = df[id_col]
        elif 'mol_id' not in df.columns:
            df['mol_id'] = [f"mol_{i:06d}" for i in range(len(df))]

        return df

    @staticmethod
    def from_sdf(
        filepath: Union[str, Path],
        id_property: Optional[str] = None,
        sanitize: bool = True
    ) -> pd.DataFrame:
        """
        Load molecules from SDF file.

        Args:
            filepath: Path to SDF file
            id_property: Property name to use as molecule ID
            sanitize: Whether to sanitize molecules

        Returns:
            DataFrame with molecules and properties
        """
        filepath = Path(filepath)

        supplier = Chem.SDMolSupplier(str(filepath), sanitize=sanitize)

        data = []
        for idx, mol in enumerate(tqdm(supplier, desc="Loading SDF")):
            if mol is None:
                data.append({'mol': None, 'mol_id': f"mol_{idx:06d}"})
                continue

            # Extract properties
            props = mol.GetPropsAsDict()

            # Get or create ID
            if id_property and id_property in props:
                mol_id = props[id_property]
            else:
                mol_id = f"mol_{idx:06d}"

            data.append({
                'mol': mol,
                'mol_id': mol_id,
                'smiles': Chem.MolToSmiles(mol),
                **props
            })

        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} molecules from {filepath.name}")

        return df

    @staticmethod
    def from_smiles_list(
        smiles_list: List[str],
        ids: Optional[List[str]] = None,
        **properties
    ) -> pd.DataFrame:
        """
        Load molecules from list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings
            ids: Optional list of molecule IDs
            **properties: Additional properties as lists

        Returns:
            DataFrame with molecules
        """
        if ids is None:
            ids = [f"mol_{i:06d}" for i in range(len(smiles_list))]

        if len(ids) != len(smiles_list):
            raise ValueError("Length of ids must match smiles_list")

        mols = [Chem.MolFromSmiles(s) if s else None for s in smiles_list]

        data = {
            'mol_id': ids,
            'smiles': smiles_list,
            'mol': mols
        }

        # Add additional properties
        for key, values in properties.items():
            if len(values) != len(smiles_list):
                raise ValueError(f"Length of {key} must match smiles_list")
            data[key] = values

        return pd.DataFrame(data)


class MoleculeDataCleaner:
    """Clean and standardize molecular datasets."""

    def __init__(
        self,
        remove_salts: bool = True,
        neutralize: bool = True,
        remove_invalid: bool = True,
        remove_duplicates: bool = True,
        keep_largest_fragment: bool = True
    ):
        """
        Initialize data cleaner.

        Args:
            remove_salts: Remove salt fragments
            neutralize: Neutralize charges where appropriate
            remove_invalid: Remove molecules that can't be parsed
            remove_duplicates: Remove duplicate molecules
            keep_largest_fragment: Keep only largest fragment
        """
        self.remove_salts = remove_salts
        self.neutralize = neutralize
        self.remove_invalid = remove_invalid
        self.remove_duplicates = remove_duplicates
        self.keep_largest_fragment = keep_largest_fragment

        if remove_salts:
            self.salt_remover = SaltRemover.SaltRemover()

    def clean(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Clean molecular dataset.

        Args:
            df: DataFrame with 'mol' column
            inplace: Modify DataFrame in place

        Returns:
            Cleaned DataFrame
        """
        if not inplace:
            df = df.copy()

        initial_count = len(df)
        print(f"Starting with {initial_count} molecules")

        # Remove invalid molecules
        if self.remove_invalid:
            df = df[df['mol'].notna()].copy()
            print(f"After removing invalid: {len(df)} molecules "
                  f"({initial_count - len(df)} removed)")

        # Remove salts and keep largest fragment
        if self.remove_salts or self.keep_largest_fragment:
            print("Removing salts and keeping largest fragment...")
            df['mol'] = [self._remove_salts_largest_fragment(mol)
                        for mol in tqdm(df['mol'])]
            df = df[df['mol'].notna()].copy()

        # Neutralize charges
        if self.neutralize:
            print("Neutralizing charges...")
            df['mol'] = [self._neutralize_mol(mol)
                        for mol in tqdm(df['mol'])]

        # Update SMILES
        print("Updating SMILES...")
        df['smiles'] = [Chem.MolToSmiles(mol) if mol else None
                       for mol in df['mol']]

        # Remove duplicates
        if self.remove_duplicates:
            before_dedup = len(df)
            df = self._remove_duplicates(df)
            print(f"After deduplication: {len(df)} molecules "
                  f"({before_dedup - len(df)} duplicates removed)")

        print(f"Final: {len(df)} molecules ({initial_count - len(df)} total removed)")

        return df

    def _remove_salts_largest_fragment(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Remove salts and keep largest fragment."""
        if mol is None:
            return None

        try:
            if self.remove_salts:
                mol = self.salt_remover.StripMol(mol)

            if self.keep_largest_fragment:
                # Get fragments
                frags = Chem.GetMolFrags(mol, asMols=True)
                if len(frags) > 1:
                    # Keep largest by heavy atom count
                    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

            return mol
        except Exception:
            return None

    def _neutralize_mol(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Neutralize charges in molecule."""
        if mol is None:
            return None

        try:
            # Standard neutralization patterns
            pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
            at_matches = mol.GetSubstructMatches(pattern)
            at_matches_list = [y[0] for y in at_matches]

            if len(at_matches_list) > 0:
                for at_idx in at_matches_list:
                    atom = mol.GetAtomWithIdx(at_idx)
                    chg = atom.GetFormalCharge()
                    hcount = atom.GetTotalNumHs()
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(hcount - chg)
                    atom.UpdatePropertyCache()

            return mol
        except Exception:
            return None

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate molecules based on InChIKey."""
        print("Generating InChIKeys for deduplication...")
        df['inchikey'] = [Chem.MolToInchiKey(mol) if mol else None
                         for mol in tqdm(df['mol'])]

        # Remove duplicates, keeping first occurrence
        df = df.drop_duplicates(subset=['inchikey'], keep='first')
        df = df.drop(columns=['inchikey'])

        return df


class DatasetSplitter:
    """Split datasets into train/validation/test sets."""

    @staticmethod
    def random_split(
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Random split.

        Args:
            df: DataFrame to split
            train_size: Fraction for training
            val_size: Fraction for validation
            test_size: Fraction for test
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6

        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            random_state=random_state
        )

        # Second split: val vs test
        val_ratio = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio,
            random_state=random_state
        )

        print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    @staticmethod
    def scaffold_split(
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scaffold-based split (molecules with same scaffold go to same set).

        Args:
            df: DataFrame with 'mol' column
            train_size: Fraction for training
            val_size: Fraction for validation
            test_size: Fraction for test
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from rdkit.Chem.Scaffolds import MurckoScaffold

        # Generate scaffolds
        print("Generating Murcko scaffolds...")
        scaffolds = {}
        for idx, mol in enumerate(tqdm(df['mol'])):
            if mol is None:
                scaffold = "invalid"
            else:
                try:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                except:
                    scaffold = "invalid"

            if scaffold not in scaffolds:
                scaffolds[scaffold] = []
            scaffolds[scaffold].append(idx)

        # Sort scaffolds by size (largest first)
        scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)

        # Distribute scaffolds to sets
        train_indices, val_indices, test_indices = [], [], []
        train_count, val_count, test_count = 0, 0, 0

        for scaffold_set in scaffold_sets:
            # Add to smallest set
            if train_count < train_size * len(df):
                train_indices.extend(scaffold_set)
                train_count += len(scaffold_set)
            elif val_count < val_size * len(df):
                val_indices.extend(scaffold_set)
                val_count += len(scaffold_set)
            else:
                test_indices.extend(scaffold_set)
                test_count += len(scaffold_set)

        train_df = df.iloc[train_indices].copy()
        val_df = df.iloc[val_indices].copy()
        test_df = df.iloc[test_indices].copy()

        print(f"Scaffold split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        print(f"Unique scaffolds - Train: {len(set([scaffolds.keys() for idx in train_indices]))}, "
              f"Val: {len(set([scaffolds.keys() for idx in val_indices]))}, "
              f"Test: {len(set([scaffolds.keys() for idx in test_indices]))}")

        return train_df, val_df, test_df

    @staticmethod
    def temporal_split(
        df: pd.DataFrame,
        date_column: str,
        train_end_date: str,
        val_end_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Temporal split based on dates.

        Args:
            df: DataFrame with date column
            date_column: Name of column containing dates
            train_end_date: End date for training set (ISO format)
            val_end_date: End date for validation set (ISO format)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        train_end = pd.to_datetime(train_end_date)
        val_end = pd.to_datetime(val_end_date)

        train_df = df[df[date_column] <= train_end].copy()
        val_df = df[(df[date_column] > train_end) & (df[date_column] <= val_end)].copy()
        test_df = df[df[date_column] > val_end].copy()

        print(f"Temporal split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    @staticmethod
    def stratified_split(
        df: pd.DataFrame,
        stratify_column: str,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Stratified split (maintain class distribution).

        Args:
            df: DataFrame to split
            stratify_column: Column to stratify by
            train_size: Fraction for training
            val_size: Fraction for validation
            test_size: Fraction for test
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # First split
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            stratify=df[stratify_column],
            random_state=random_state
        )

        # Second split
        val_ratio = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio,
            stratify=temp_df[stratify_column],
            random_state=random_state
        )

        print(f"Stratified split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df


class DataAugmenter:
    """Augment molecular datasets."""

    @staticmethod
    def enumerate_smiles(
        mol: Chem.Mol,
        n_variants: int = 10,
        random_type: str = "restricted"
    ) -> List[str]:
        """
        Generate SMILES variants for a molecule.

        Args:
            mol: RDKit molecule
            n_variants: Number of variants to generate
            random_type: Type of randomization ('restricted', 'unrestricted')

        Returns:
            List of SMILES strings
        """
        if mol is None:
            return []

        smiles_list = []
        for _ in range(n_variants):
            try:
                if random_type == "restricted":
                    smi = Chem.MolToSmiles(mol, doRandom=True)
                else:
                    smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
                smiles_list.append(smi)
            except:
                continue

        return list(set(smiles_list))  # Remove duplicates

    @staticmethod
    def generate_conformers(
        mol: Chem.Mol,
        n_conformers: int = 10,
        random_seed: int = 42
    ) -> Optional[Chem.Mol]:
        """
        Generate 3D conformers for a molecule.

        Args:
            mol: RDKit molecule
            n_conformers: Number of conformers to generate
            random_seed: Random seed

        Returns:
            Molecule with conformers
        """
        if mol is None:
            return None

        mol = Chem.AddHs(mol)

        try:
            AllChem.EmbedMultipleConfs(
                mol,
                numConfs=n_conformers,
                randomSeed=random_seed,
                useRandomCoords=True
            )
            AllChem.MMFFOptimizeMoleculeConfs(mol)
            return mol
        except:
            return None


class DoseResponseAnalyzer:
    """Analyze dose-response data and calculate IC50 values."""

    @staticmethod
    def hill_equation(
        x: np.ndarray,
        bottom: float,
        top: float,
        ic50: float,
        hill_slope: float
    ) -> np.ndarray:
        """
        Four-parameter Hill equation.

        Args:
            x: Concentrations
            bottom: Minimum response
            top: Maximum response
            ic50: Half-maximal inhibitory concentration
            hill_slope: Hill slope

        Returns:
            Predicted responses
        """
        return bottom + (top - bottom) / (1 + (x / ic50) ** hill_slope)

    @staticmethod
    def fit_dose_response(
        concentrations: np.ndarray,
        responses: np.ndarray,
        bounds: Optional[Tuple] = None
    ) -> Dict[str, float]:
        """
        Fit dose-response curve.

        Args:
            concentrations: Array of concentrations
            responses: Array of responses
            bounds: Optional bounds for parameters (bottom, top, ic50, hill_slope)

        Returns:
            Dictionary with fitted parameters and R²
        """
        # Default bounds
        if bounds is None:
            bounds = (
                [0, 0, concentrations.min(), -10],  # Lower bounds
                [100, 100, concentrations.max(), 10]  # Upper bounds
            )

        # Initial guess
        p0 = [
            responses.min(),  # bottom
            responses.max(),  # top
            np.median(concentrations),  # ic50
            -1.0  # hill_slope
        ]

        try:
            popt, pcov = curve_fit(
                DoseResponseAnalyzer.hill_equation,
                concentrations,
                responses,
                p0=p0,
                bounds=bounds,
                maxfev=10000
            )

            # Calculate R²
            y_pred = DoseResponseAnalyzer.hill_equation(concentrations, *popt)
            ss_res = np.sum((responses - y_pred) ** 2)
            ss_tot = np.sum((responses - responses.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Calculate standard errors
            perr = np.sqrt(np.diag(pcov))

            return {
                'bottom': popt[0],
                'top': popt[1],
                'ic50': popt[2],
                'hill_slope': popt[3],
                'bottom_se': perr[0],
                'top_se': perr[1],
                'ic50_se': perr[2],
                'hill_slope_se': perr[3],
                'r_squared': r_squared
            }
        except Exception as e:
            warnings.warn(f"Curve fitting failed: {e}")
            return {
                'bottom': np.nan,
                'top': np.nan,
                'ic50': np.nan,
                'hill_slope': np.nan,
                'bottom_se': np.nan,
                'top_se': np.nan,
                'ic50_se': np.nan,
                'hill_slope_se': np.nan,
                'r_squared': np.nan
            }


def calculate_dataset_statistics(df: pd.DataFrame) -> DatasetStatistics:
    """
    Calculate statistics for a molecular dataset.

    Args:
        df: DataFrame with 'mol' column

    Returns:
        DatasetStatistics object
    """
    valid_mols = df[df['mol'].notna()]['mol']

    if len(valid_mols) == 0:
        return DatasetStatistics(
            n_molecules=len(df),
            n_valid=0,
            n_invalid=len(df),
            n_duplicates=0,
            mw_mean=0.0,
            mw_std=0.0,
            logp_mean=0.0,
            logp_std=0.0,
            n_rotatable_bonds_mean=0.0,
            n_hbd_mean=0.0,
            n_hba_mean=0.0
        )

    # Calculate properties
    mws = [Descriptors.MolWt(mol) for mol in valid_mols]
    logps = [Descriptors.MolLogP(mol) for mol in valid_mols]
    n_rot_bonds = [Descriptors.NumRotatableBonds(mol) for mol in valid_mols]
    n_hbd = [Descriptors.NumHDonors(mol) for mol in valid_mols]
    n_hba = [Descriptors.NumHAcceptors(mol) for mol in valid_mols]

    return DatasetStatistics(
        n_molecules=len(df),
        n_valid=len(valid_mols),
        n_invalid=len(df) - len(valid_mols),
        n_duplicates=0,  # This would need to be calculated separately
        mw_mean=np.mean(mws),
        mw_std=np.std(mws),
        logp_mean=np.mean(logps),
        logp_std=np.std(logps),
        n_rotatable_bonds_mean=np.mean(n_rot_bonds),
        n_hbd_mean=np.mean(n_hbd),
        n_hba_mean=np.mean(n_hba)
    )


def impute_missing_values(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str = "mean"
) -> pd.DataFrame:
    """
    Impute missing values in dataset.

    Args:
        df: DataFrame with missing values
        columns: Columns to impute
        strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')

    Returns:
        DataFrame with imputed values
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            warnings.warn(f"Column {col} not found in DataFrame")
            continue

        if strategy == "mean":
            df[col].fillna(df[col].mean(), inplace=True)
        elif strategy == "median":
            df[col].fillna(df[col].median(), inplace=True)
        elif strategy == "mode":
            df[col].fillna(df[col].mode()[0], inplace=True)
        elif strategy == "drop":
            df = df.dropna(subset=[col])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    return df


def batch_process(
    items: List,
    process_fn: Callable,
    batch_size: int = 1000,
    desc: str = "Processing"
) -> List:
    """
    Process items in batches with progress bar.

    Args:
        items: List of items to process
        process_fn: Function to apply to each item
        batch_size: Size of each batch
        desc: Description for progress bar

    Returns:
        List of processed items
    """
    results = []

    for i in tqdm(range(0, len(items), batch_size), desc=desc):
        batch = items[i:i + batch_size]
        batch_results = [process_fn(item) for item in batch]
        results.extend(batch_results)

    return results


if __name__ == "__main__":
    # Example usage
    print("Data Processing Utilities")
    print("=" * 50)

    # Example: Create sample dataset
    smiles_list = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CCO.[Na+].[Cl-]",  # Ethanol with salt
    ]

    df = MoleculeDataLoader.from_smiles_list(smiles_list)
    print(f"\nLoaded {len(df)} molecules")

    # Clean data
    cleaner = MoleculeDataCleaner()
    df_clean = cleaner.clean(df)
    print(f"After cleaning: {len(df_clean)} molecules")

    # Calculate statistics
    stats = calculate_dataset_statistics(df_clean)
    print(f"\nDataset Statistics:")
    print(f"  Valid molecules: {stats.n_valid}")
    print(f"  Mean MW: {stats.mw_mean:.2f} ± {stats.mw_std:.2f}")
    print(f"  Mean LogP: {stats.logp_mean:.2f} ± {stats.logp_std:.2f}")
