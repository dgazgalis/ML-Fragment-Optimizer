"""
Benchmark datasets for evaluating fragment optimization models.

This module provides curated benchmark sets including activity cliffs,
scaffold splits, temporal splits, and external validation sets.
"""

from typing import Optional, Dict, Tuple
import pandas as pd
from pathlib import Path
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    raise ImportError("RDKit is required")


class ActivityCliffBenchmark:
    """Benchmark for activity cliff prediction."""

    @staticmethod
    def generate_activity_cliff_pairs(
        df: pd.DataFrame,
        activity_col: str = "activity",
        similarity_threshold: float = 0.85,
        activity_diff_threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        Generate activity cliff pairs.

        Activity cliffs = structurally similar but different activity.

        Args:
            df: DataFrame with molecules and activities
            activity_col: Column with activity values (e.g., pIC50)
            similarity_threshold: Minimum Tanimoto similarity
            activity_diff_threshold: Minimum activity difference

        Returns:
            DataFrame with activity cliff pairs
        """
        from tqdm import tqdm

        print("Generating molecular fingerprints...")
        fps = []
        for mol in tqdm(df['mol']):
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                fps.append(fp)
            else:
                fps.append(None)

        df['fp'] = fps

        # Find pairs
        pairs = []
        print("Finding activity cliff pairs...")

        for i in tqdm(range(len(df))):
            if df.iloc[i]['fp'] is None:
                continue

            for j in range(i + 1, len(df)):
                if df.iloc[j]['fp'] is None:
                    continue

                # Calculate similarity
                sim = DataStructs.TanimotoSimilarity(
                    df.iloc[i]['fp'],
                    df.iloc[j]['fp']
                )

                if sim >= similarity_threshold:
                    # Check activity difference
                    act_diff = abs(
                        df.iloc[i][activity_col] - df.iloc[j][activity_col]
                    )

                    if act_diff >= activity_diff_threshold:
                        pairs.append({
                            'mol1_id': df.iloc[i]['mol_id'],
                            'mol2_id': df.iloc[j]['mol_id'],
                            'smiles1': df.iloc[i]['smiles'],
                            'smiles2': df.iloc[j]['smiles'],
                            'activity1': df.iloc[i][activity_col],
                            'activity2': df.iloc[j][activity_col],
                            'similarity': sim,
                            'activity_diff': act_diff,
                            'is_cliff': True
                        })

        return pd.DataFrame(pairs)


class ScaffoldSplitBenchmark:
    """Benchmark with scaffold-based splits."""

    @staticmethod
    def create_scaffold_split_benchmark(
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15
    ) -> Dict[str, pd.DataFrame]:
        """
        Create scaffold split benchmark.

        Args:
            df: DataFrame with molecules
            train_size: Training set fraction
            val_size: Validation set fraction
            test_size: Test set fraction

        Returns:
            Dictionary with train/val/test DataFrames and metadata
        """
        from tqdm import tqdm

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

        # Sort by size
        scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)

        # Distribute to sets
        train_idx, val_idx, test_idx = [], [], []
        train_count, val_count, test_count = 0, 0, 0

        for scaffold_set in scaffold_sets:
            if train_count < train_size * len(df):
                train_idx.extend(scaffold_set)
                train_count += len(scaffold_set)
            elif val_count < val_size * len(df):
                val_idx.extend(scaffold_set)
                val_count += len(scaffold_set)
            else:
                test_idx.extend(scaffold_set)
                test_count += len(scaffold_set)

        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # Calculate scaffold overlap
        train_scaffolds = set()
        val_scaffolds = set()
        test_scaffolds = set()

        for idx in train_idx:
            mol = df.iloc[idx]['mol']
            if mol:
                try:
                    train_scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
                except:
                    pass

        for idx in val_idx:
            mol = df.iloc[idx]['mol']
            if mol:
                try:
                    val_scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
                except:
                    pass

        for idx in test_idx:
            mol = df.iloc[idx]['mol']
            if mol:
                try:
                    test_scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
                except:
                    pass

        overlap_train_val = len(train_scaffolds & val_scaffolds)
        overlap_train_test = len(train_scaffolds & test_scaffolds)
        overlap_val_test = len(val_scaffolds & test_scaffolds)

        metadata = {
            'n_scaffolds_train': len(train_scaffolds),
            'n_scaffolds_val': len(val_scaffolds),
            'n_scaffolds_test': len(test_scaffolds),
            'scaffold_overlap_train_val': overlap_train_val,
            'scaffold_overlap_train_test': overlap_train_test,
            'scaffold_overlap_val_test': overlap_val_test,
            'split_type': 'scaffold'
        }

        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'metadata': metadata
        }


class TemporalSplitBenchmark:
    """Benchmark with temporal splits."""

    @staticmethod
    def create_temporal_split_benchmark(
        df: pd.DataFrame,
        date_column: str,
        train_end_date: str,
        val_end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Create temporal split benchmark.

        Args:
            df: DataFrame with date column
            date_column: Column with dates
            train_end_date: End date for training
            val_end_date: End date for validation

        Returns:
            Dictionary with train/val/test splits and metadata
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        train_end = pd.to_datetime(train_end_date)
        val_end = pd.to_datetime(val_end_date)

        train_df = df[df[date_column] <= train_end].copy()
        val_df = df[(df[date_column] > train_end) & (df[date_column] <= val_end)].copy()
        test_df = df[df[date_column] > val_end].copy()

        metadata = {
            'train_date_range': (
                train_df[date_column].min().isoformat(),
                train_df[date_column].max().isoformat()
            ),
            'val_date_range': (
                val_df[date_column].min().isoformat(),
                val_df[date_column].max().isoformat()
            ),
            'test_date_range': (
                test_df[date_column].min().isoformat(),
                test_df[date_column].max().isoformat()
            ),
            'split_type': 'temporal'
        }

        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'metadata': metadata
        }


class ExternalValidationSet:
    """External validation set for final model evaluation."""

    @staticmethod
    def create_external_validation_set(
        df: pd.DataFrame,
        fraction: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create external validation set.

        Args:
            df: Full dataset
            fraction: Fraction for external validation
            random_state: Random seed

        Returns:
            Tuple of (development_set, external_validation_set)
        """
        from sklearn.model_selection import train_test_split

        dev_df, ext_df = train_test_split(
            df,
            test_size=fraction,
            random_state=random_state
        )

        print(f"Development set: {len(dev_df)} molecules")
        print(f"External validation set: {len(ext_df)} molecules")

        return dev_df, ext_df


def load_benchmark_results(benchmark_name: str) -> pd.DataFrame:
    """
    Load baseline results for a benchmark.

    Args:
        benchmark_name: Name of benchmark

    Returns:
        DataFrame with baseline results
    """
    # This would load pre-computed baseline results
    # For now, return empty DataFrame

    return pd.DataFrame({
        'model': [],
        'metric': [],
        'value': [],
        'std': []
    })


def evaluate_on_benchmark(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    task_type: str = "regression"
) -> Dict[str, float]:
    """
    Evaluate predictions on benchmark.

    Args:
        predictions: DataFrame with predictions
        ground_truth: DataFrame with ground truth
        task_type: 'regression' or 'classification'

    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        roc_auc_score, accuracy_score, f1_score
    )

    if task_type == "regression":
        rmse = np.sqrt(mean_squared_error(
            ground_truth['activity'],
            predictions['predicted_activity']
        ))
        mae = mean_absolute_error(
            ground_truth['activity'],
            predictions['predicted_activity']
        )
        r2 = r2_score(
            ground_truth['activity'],
            predictions['predicted_activity']
        )

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    elif task_type == "classification":
        accuracy = accuracy_score(
            ground_truth['label'],
            predictions['predicted_label']
        )

        # Assume binary classification
        try:
            auc = roc_auc_score(
                ground_truth['label'],
                predictions['predicted_probability']
            )
        except:
            auc = np.nan

        f1 = f1_score(
            ground_truth['label'],
            predictions['predicted_label']
        )

        return {
            'accuracy': accuracy,
            'auc_roc': auc,
            'f1': f1
        }

    else:
        raise ValueError(f"Unknown task type: {task_type}")


if __name__ == "__main__":
    print("Benchmark Data Utilities")
    print("=" * 50)

    # Example: Create sample data
    np.random.seed(42)

    smiles_list = [
        "CCO",
        "CC(C)O",
        "CCCO",
        "CC(C)CO",
        "c1ccccc1",
        "c1ccccc1C",
        "c1ccccc1CC",
        "c1ccc(O)cc1",
        "c1ccc(C)cc1",
        "c1ccc(CC)cc1",
    ]

    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from src.utils.data_processing import MoleculeDataLoader

    df = MoleculeDataLoader.from_smiles_list(smiles_list)
    df['activity'] = np.random.uniform(4, 9, len(df))  # pIC50 values

    print(f"\nSample dataset: {len(df)} molecules")

    # Test activity cliff detection
    print("\n" + "=" * 50)
    print("Activity Cliff Detection:")

    cliff_detector = ActivityCliffBenchmark()
    cliff_pairs = cliff_detector.generate_activity_cliff_pairs(
        df,
        similarity_threshold=0.7,
        activity_diff_threshold=1.0
    )

    print(f"\nFound {len(cliff_pairs)} activity cliff pairs")
    if len(cliff_pairs) > 0:
        print("\nExample pairs:")
        print(cliff_pairs[['smiles1', 'smiles2', 'similarity', 'activity_diff']].head())

    # Test scaffold split
    print("\n" + "=" * 50)
    print("Scaffold Split Benchmark:")

    scaffold_benchmark = ScaffoldSplitBenchmark()
    splits = scaffold_benchmark.create_scaffold_split_benchmark(df)

    print(f"\nTrain: {len(splits['train'])} molecules")
    print(f"Val: {len(splits['val'])} molecules")
    print(f"Test: {len(splits['test'])} molecules")
    print(f"\nMetadata:")
    for key, value in splits['metadata'].items():
        print(f"  {key}: {value}")

    print("\nBenchmark utilities test completed!")
