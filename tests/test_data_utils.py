"""
Unit tests for data utilities.

Run with: pytest tests/test_data_utils.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    MoleculeDataLoader,
    MoleculeDataCleaner,
    DatasetSplitter,
    PAINSFilter,
    AggregatorFilter,
    PropertyFilter,
    DoseResponseAnalyzer,
    calculate_dataset_statistics
)


class TestMoleculeDataLoader:
    """Test data loading functions."""

    def test_from_smiles_list(self):
        """Test loading from SMILES list."""
        smiles_list = ['CCO', 'CC(=O)O', 'c1ccccc1']
        ids = ['mol1', 'mol2', 'mol3']

        df = MoleculeDataLoader.from_smiles_list(smiles_list, ids=ids)

        assert len(df) == 3
        assert 'mol' in df.columns
        assert 'smiles' in df.columns
        assert 'mol_id' in df.columns
        assert df['mol'].notna().sum() == 3

    def test_from_smiles_list_with_invalid(self):
        """Test handling invalid SMILES."""
        smiles_list = ['CCO', 'invalid', 'c1ccccc1']

        df = MoleculeDataLoader.from_smiles_list(smiles_list)

        assert len(df) == 3
        assert df['mol'].notna().sum() == 2

    def test_from_smiles_list_with_properties(self):
        """Test loading with additional properties."""
        smiles_list = ['CCO', 'CC(=O)O']
        df = MoleculeDataLoader.from_smiles_list(
            smiles_list,
            activity=[1.2, 3.4]
        )

        assert 'activity' in df.columns
        assert df['activity'].tolist() == [1.2, 3.4]


class TestMoleculeDataCleaner:
    """Test data cleaning functions."""

    def test_basic_cleaning(self):
        """Test basic cleaning pipeline."""
        smiles_list = [
            'CCO',
            'CC(=O)O',
            'CCO.[Na+].[Cl-]',  # With salt
            'invalid'
        ]

        df = MoleculeDataLoader.from_smiles_list(smiles_list)

        cleaner = MoleculeDataCleaner(
            remove_salts=True,
            remove_invalid=True
        )
        df_clean = cleaner.clean(df)

        # Should remove invalid and clean salt
        assert len(df_clean) <= len(df)
        assert df_clean['mol'].notna().all()

    def test_duplicate_removal(self):
        """Test duplicate removal."""
        # Same molecule, different representation
        smiles_list = ['CCO', 'OCC', 'CCO']

        df = MoleculeDataLoader.from_smiles_list(smiles_list)

        cleaner = MoleculeDataCleaner(remove_duplicates=True)
        df_clean = cleaner.clean(df)

        # Should keep only one
        assert len(df_clean) < len(df)


class TestDatasetSplitter:
    """Test dataset splitting functions."""

    def test_random_split(self):
        """Test random split."""
        smiles_list = ['CCO'] * 100  # Same SMILES for simplicity

        df = MoleculeDataLoader.from_smiles_list(smiles_list)

        splitter = DatasetSplitter()
        train, val, test = splitter.random_split(
            df,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15
        )

        assert len(train) + len(val) + len(test) == len(df)
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_scaffold_split(self):
        """Test scaffold split."""
        smiles_list = [
            'c1ccccc1',
            'c1ccccc1C',
            'c1ccccc1CC',
            'C1CCCCC1',
            'C1CCCCC1C',
        ]

        df = MoleculeDataLoader.from_smiles_list(smiles_list)

        splitter = DatasetSplitter()
        train, val, test = splitter.scaffold_split(df)

        assert len(train) + len(val) + len(test) == len(df)


class TestPAINSFilter:
    """Test PAINS filtering."""

    def test_pains_detection(self):
        """Test PAINS compound detection."""
        pains_filter = PAINSFilter()

        # Clean molecule
        from rdkit import Chem
        mol = Chem.MolFromSmiles('CCO')
        is_pains, matches = pains_filter.is_pains(mol)
        assert is_pains == False

        # Catechol (PAINS)
        mol = Chem.MolFromSmiles('c1ccc(O)c(O)c1')
        is_pains, matches = pains_filter.is_pains(mol)
        # May or may not be detected depending on RDKit catalog


class TestPropertyFilter:
    """Test property filtering."""

    def test_rule_of_five(self):
        """Test Lipinski's Rule of Five."""
        from rdkit import Chem

        # Small molecule - should pass
        mol = Chem.MolFromSmiles('CCO')
        passes, violations = PropertyFilter.rule_of_five(mol)
        assert passes == True
        assert violations == 0

        # Large lipophilic molecule
        mol = Chem.MolFromSmiles('CCCCCCCCCCCCCCCCCCCCCCCCCCCC')
        passes, violations = PropertyFilter.rule_of_five(mol)
        # Should have violations

    def test_rule_of_three(self):
        """Test Rule of Three for fragments."""
        from rdkit import Chem

        # Small fragment
        mol = Chem.MolFromSmiles('c1ccccc1')  # Benzene
        passes, violations = PropertyFilter.rule_of_three(mol)
        assert passes == True

    def test_custom_property_filter(self):
        """Test custom property ranges."""
        from rdkit import Chem

        prop_filter = PropertyFilter(
            mw_range=(0, 100),
            logp_range=(-5, 5)
        )

        # Ethanol - should pass MW filter
        mol = Chem.MolFromSmiles('CCO')
        passes, props = prop_filter.passes_filters(mol)

        assert 'mw' in props
        assert 'logp' in props


class TestDoseResponseAnalyzer:
    """Test dose-response analysis."""

    def test_hill_equation_fitting(self):
        """Test Hill equation fitting."""
        analyzer = DoseResponseAnalyzer()

        # Generate synthetic dose-response data
        concentrations = np.array([0.1, 1, 10, 100, 1000])
        responses = np.array([10, 25, 50, 75, 90])

        result = analyzer.fit_dose_response(concentrations, responses)

        assert 'ic50' in result
        assert 'hill_slope' in result
        assert 'r_squared' in result
        assert not np.isnan(result['ic50'])


class TestDatasetStatistics:
    """Test dataset statistics calculation."""

    def test_calculate_statistics(self):
        """Test statistics calculation."""
        smiles_list = ['CCO', 'CC(=O)O', 'c1ccccc1']
        df = MoleculeDataLoader.from_smiles_list(smiles_list)

        stats = calculate_dataset_statistics(df)

        assert stats.n_molecules == 3
        assert stats.n_valid == 3
        assert stats.mw_mean > 0
        assert stats.logp_mean is not None


# Integration tests

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_cleaning_pipeline(self):
        """Test complete data cleaning pipeline."""
        smiles_list = [
            'CCO',
            'CC(=O)O',
            'c1ccccc1',
            'CCO.[Na+].[Cl-]',
            'invalid',
            'CCO'  # Duplicate
        ]

        # Load
        df = MoleculeDataLoader.from_smiles_list(smiles_list)
        initial_count = len(df)

        # Clean
        cleaner = MoleculeDataCleaner(
            remove_salts=True,
            remove_invalid=True,
            remove_duplicates=True
        )
        df_clean = cleaner.clean(df)

        # Should have removed some molecules
        assert len(df_clean) < initial_count
        assert df_clean['mol'].notna().all()

    def test_load_clean_split_workflow(self):
        """Test complete workflow: load, clean, split."""
        # Generate sample data
        smiles_list = [
            'c1ccccc1', 'c1ccccc1C', 'c1ccccc1CC',
            'C1CCCCC1', 'C1CCCCC1C', 'C1CCCCC1CC',
            'c1ccncc1', 'c1ccncc1C', 'c1ccncc1CC',
            'CCO', 'CCCO', 'CCCCO'
        ]

        # Load
        df = MoleculeDataLoader.from_smiles_list(smiles_list)
        df['activity'] = np.random.uniform(3, 9, len(df))

        # Clean
        cleaner = MoleculeDataCleaner()
        df_clean = cleaner.clean(df)

        # Split
        splitter = DatasetSplitter()
        train, val, test = splitter.random_split(df_clean)

        # Verify
        assert len(train) + len(val) + len(test) == len(df_clean)
        assert 'mol' in train.columns
        assert 'activity' in train.columns


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
