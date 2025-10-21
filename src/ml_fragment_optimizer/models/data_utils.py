"""
Data Utilities for ADMET Prediction

This module provides utilities for:
- Loading and preprocessing ADMET datasets
- Data normalization and standardization
- Train/validation/test splitting with stratification
- Handling missing values
- Data augmentation for molecular structures

Author: Claude Code
Date: 2025-10-20
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .admet_predictor import ADMET_TASKS


class ADMETDataProcessor:
    """
    Processor for ADMET datasets with normalization and splitting.
    """

    def __init__(
        self,
        task_names: Optional[List[str]] = None,
        normalize: bool = True,
        handle_missing: str = 'drop'
    ):
        """
        Initialize data processor.

        Args:
            task_names: List of tasks to process (None = all tasks)
            normalize: Whether to normalize target values
            handle_missing: How to handle missing values ('drop', 'mean', 'median', 'zero')
        """
        if task_names is None:
            self.task_names = list(ADMET_TASKS.keys())
        else:
            self.task_names = task_names

        self.normalize = normalize
        self.handle_missing = handle_missing

        # Scalers for each task (fitted on training data)
        self.scalers: Dict[str, StandardScaler] = {}

    def load_csv(
        self,
        csv_path: Path,
        smiles_column: str = 'smiles',
        validate_smiles: bool = True
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """
        Load ADMET data from CSV file.

        Expected CSV format:
        smiles,solubility,permeability,cyp3a4,herg,logd,pka
        CCO,-0.77,-4.5,0,0,0.31,15.9
        ...

        Args:
            csv_path: Path to CSV file
            smiles_column: Name of SMILES column
            validate_smiles: Whether to validate SMILES strings

        Returns:
            Tuple of (smiles_list, targets_dict)
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Extract SMILES
        if smiles_column not in df.columns:
            raise ValueError(f"SMILES column '{smiles_column}' not found in CSV")

        smiles_list = df[smiles_column].tolist()

        # Validate SMILES
        if validate_smiles:
            from rdkit import Chem
            valid_indices = []
            for i, smi in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    valid_indices.append(i)
                else:
                    warnings.warn(f"Invalid SMILES at row {i}: {smi}")

            df = df.iloc[valid_indices].reset_index(drop=True)
            smiles_list = [smiles_list[i] for i in valid_indices]

        print(f"Loaded {len(smiles_list)} valid SMILES")

        # Extract targets
        targets = {}
        for task in self.task_names:
            if task in df.columns:
                targets[task] = df[task].values.reshape(-1, 1).astype(np.float32)
            else:
                warnings.warn(f"Task '{task}' not found in CSV. Filling with NaN.")
                targets[task] = np.full((len(smiles_list), 1), np.nan, dtype=np.float32)

        # Handle missing values
        smiles_list, targets = self._handle_missing_values(smiles_list, targets)

        return smiles_list, targets

    def _handle_missing_values(
        self,
        smiles_list: List[str],
        targets: Dict[str, np.ndarray]
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """
        Handle missing values in targets.

        Args:
            smiles_list: List of SMILES
            targets: Dict of target arrays

        Returns:
            Tuple of (filtered_smiles, filtered_targets)
        """
        if self.handle_missing == 'drop':
            # Drop samples with any missing values
            n_samples = len(smiles_list)
            valid_mask = np.ones(n_samples, dtype=bool)

            for task, values in targets.items():
                valid_mask &= ~np.isnan(values).flatten()

            n_valid = valid_mask.sum()
            if n_valid < n_samples:
                print(f"Dropping {n_samples - n_valid} samples with missing values")

            smiles_list = [smi for i, smi in enumerate(smiles_list) if valid_mask[i]]
            targets = {task: values[valid_mask] for task, values in targets.items()}

        elif self.handle_missing in ['mean', 'median', 'zero']:
            # Impute missing values
            for task, values in targets.items():
                mask = np.isnan(values)
                if mask.any():
                    if self.handle_missing == 'mean':
                        fill_value = np.nanmean(values)
                    elif self.handle_missing == 'median':
                        fill_value = np.nanmedian(values)
                    else:  # zero
                        fill_value = 0.0

                    values[mask] = fill_value
                    print(f"Imputed {mask.sum()} missing values for '{task}' with {self.handle_missing}")

        return smiles_list, targets

    def normalize_targets(
        self,
        targets: Dict[str, np.ndarray],
        fit: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Normalize target values to zero mean and unit variance.

        Args:
            targets: Dict of target arrays
            fit: Whether to fit scalers (True for training, False for test)

        Returns:
            Normalized targets
        """
        if not self.normalize:
            return targets

        normalized = {}

        for task, values in targets.items():
            if fit:
                # Fit scaler on non-NaN values
                mask = ~np.isnan(values)
                if mask.any():
                    scaler = StandardScaler()
                    scaler.fit(values[mask])
                    self.scalers[task] = scaler
                else:
                    warnings.warn(f"All values are NaN for task '{task}'. Skipping normalization.")
                    normalized[task] = values
                    continue

            # Transform
            if task in self.scalers:
                normalized[task] = self.scalers[task].transform(values)
            else:
                normalized[task] = values

        return normalized

    def denormalize_predictions(
        self,
        predictions: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Convert normalized predictions back to original scale.

        Args:
            predictions: Dict of normalized predictions

        Returns:
            Denormalized predictions
        """
        if not self.normalize:
            return predictions

        denormalized = {}

        for task, values in predictions.items():
            if task in self.scalers:
                denormalized[task] = self.scalers[task].inverse_transform(values)
            else:
                denormalized[task] = values

        return denormalized

    def split_data(
        self,
        smiles_list: List[str],
        targets: Dict[str, np.ndarray],
        train_size: float = 0.8,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify_task: Optional[str] = None
    ) -> Tuple[
        List[str], Dict[str, np.ndarray],
        List[str], Dict[str, np.ndarray],
        List[str], Dict[str, np.ndarray]
    ]:
        """
        Split data into train/validation/test sets.

        Args:
            smiles_list: List of SMILES
            targets: Dict of targets
            train_size: Fraction for training
            val_size: Fraction for validation (test = 1 - train - val)
            random_state: Random seed
            stratify_task: Task name for stratification (for classification)

        Returns:
            Tuple of (train_smiles, train_targets, val_smiles, val_targets,
                     test_smiles, test_targets)
        """
        n_samples = len(smiles_list)
        indices = np.arange(n_samples)

        # Stratification for classification tasks
        stratify_values = None
        if stratify_task is not None and stratify_task in targets:
            # Bin continuous values for stratification
            values = targets[stratify_task].flatten()
            stratify_values = pd.qcut(values, q=5, labels=False, duplicates='drop')

        # First split: train + val vs test
        test_size = 1.0 - train_size - val_size
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_values
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (train_size + val_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_values[train_val_idx] if stratify_values is not None else None
        )

        # Create splits
        def create_split(indices):
            split_smiles = [smiles_list[i] for i in indices]
            split_targets = {task: values[indices] for task, values in targets.items()}
            return split_smiles, split_targets

        train_smiles, train_targets = create_split(train_idx)
        val_smiles, val_targets = create_split(val_idx)
        test_smiles, test_targets = create_split(test_idx)

        print(f"\nData split:")
        print(f"  Train: {len(train_smiles)} samples ({len(train_smiles)/n_samples*100:.1f}%)")
        print(f"  Val:   {len(val_smiles)} samples ({len(val_smiles)/n_samples*100:.1f}%)")
        print(f"  Test:  {len(test_smiles)} samples ({len(test_smiles)/n_samples*100:.1f}%)")

        # Normalize targets (fit on training data)
        train_targets = self.normalize_targets(train_targets, fit=True)
        val_targets = self.normalize_targets(val_targets, fit=False)
        test_targets = self.normalize_targets(test_targets, fit=False)

        return (train_smiles, train_targets,
                val_smiles, val_targets,
                test_smiles, test_targets)


class MolecularAugmenter:
    """
    Data augmentation for molecular structures.

    Techniques:
    1. SMILES randomization: Generate different valid SMILES for same molecule
    2. Stereoisomer enumeration: Generate stereoisomers
    3. Tautomer enumeration: Generate tautomers
    """

    @staticmethod
    def randomize_smiles(smiles: str, n_augmentations: int = 5) -> List[str]:
        """
        Generate multiple SMILES representations of the same molecule.

        This is a form of data augmentation that doesn't change the molecule
        but provides different "views" of it.

        Args:
            smiles: Input SMILES
            n_augmentations: Number of augmented SMILES to generate

        Returns:
            List of augmented SMILES (includes original)
        """
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles]

        augmented = [smiles]  # Include original

        for _ in range(n_augmentations - 1):
            # Generate random SMILES by shuffling atom order
            random_smiles = Chem.MolToSmiles(mol, doRandom=True)
            if random_smiles not in augmented:
                augmented.append(random_smiles)

        return augmented

    @staticmethod
    def enumerate_stereoisomers(smiles: str, max_isomers: int = 10) -> List[str]:
        """
        Enumerate stereoisomers of a molecule.

        Useful for exploring stereochemistry effects on ADMET properties.

        Args:
            smiles: Input SMILES
            max_isomers: Maximum number of stereoisomers

        Returns:
            List of stereoisomer SMILES
        """
        from rdkit import Chem
        from rdkit.Chem.EnumerateStereoisomers import (
            EnumerateStereoisomers,
            StereoEnumerationOptions
        )

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles]

        opts = StereoEnumerationOptions(unique=True, maxIsomers=max_isomers)

        isomers = []
        for isomer in EnumerateStereoisomers(mol, options=opts):
            isomer_smiles = Chem.MolToSmiles(isomer)
            isomers.append(isomer_smiles)

        return isomers if len(isomers) > 0 else [smiles]


def create_sample_dataset(
    output_path: Path,
    n_samples: int = 1000,
    random_state: int = 42
):
    """
    Create a sample ADMET dataset for testing.

    Args:
        output_path: Path to save CSV
        n_samples: Number of samples to generate
        random_state: Random seed
    """
    np.random.seed(random_state)

    # Generate random SMILES (simplified)
    # In practice, use real datasets like ChEMBL, PubChem, or proprietary data
    from rdkit import Chem
    from rdkit.Chem import AllChem

    print(f"Generating {n_samples} sample molecules...")

    smiles_list = []
    for _ in range(n_samples):
        # Generate random molecule (very simple approach)
        mol = Chem.MolFromSmiles('C' * np.random.randint(2, 10))
        if mol is not None:
            smi = Chem.MolToSmiles(mol)
            smiles_list.append(smi)

    # Generate synthetic target values
    # Correlated with molecular properties for realism
    data = {'smiles': smiles_list}

    for task, (min_val, max_val) in ADMET_TASKS.items():
        # Add some noise and correlations
        values = np.random.uniform(min_val, max_val, len(smiles_list))

        # Add some missing values (10%)
        missing_mask = np.random.random(len(smiles_list)) < 0.1
        values[missing_mask] = np.nan

        data[task] = values

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    print(f"Saved sample dataset to {output_path}")


if __name__ == "__main__":
    # Test data processor
    print("Testing data processor...")

    # Create sample dataset
    output_path = Path("sample_admet_data.csv")
    create_sample_dataset(output_path, n_samples=100)

    # Load and process
    processor = ADMETDataProcessor()
    smiles_list, targets = processor.load_csv(output_path)

    # Split data
    splits = processor.split_data(smiles_list, targets)
    train_smiles, train_targets, val_smiles, val_targets, test_smiles, test_targets = splits

    print(f"\nTraining set statistics:")
    for task, values in train_targets.items():
        print(f"  {task}: mean={values.mean():.4f}, std={values.std():.4f}")

    # Clean up
    output_path.unlink()
    print("\nTest completed!")
