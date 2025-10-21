"""
Unit tests for molecular featurization utilities.
"""

import pytest
import numpy as np
from ml_fragment_optimizer.utils.featurizers import (
    MolecularFeaturizer,
    calculate_basic_properties,
    smiles_to_mol_safe,
    batch_featurize,
)


class TestMolecularFeaturizer:
    """Test MolecularFeaturizer class."""

    def test_morgan_fingerprint(self):
        """Test Morgan fingerprint generation."""
        featurizer = MolecularFeaturizer(
            fingerprint_type="morgan",
            radius=2,
            n_bits=2048
        )

        smiles = "CCO"
        features = featurizer.featurize(smiles)

        assert features.shape == (2048,)
        assert features.dtype == np.float64 or features.dtype == np.int64

    def test_batch_featurization(self):
        """Test batch featurization."""
        featurizer = MolecularFeaturizer(fingerprint_type="morgan")

        smiles_list = ["CCO", "c1ccccc1", "CC(C)C"]
        features = featurizer.featurize(smiles_list)

        assert features.shape == (3, 2048)

    def test_invalid_smiles(self):
        """Test handling of invalid SMILES."""
        featurizer = MolecularFeaturizer(fingerprint_type="morgan")

        with pytest.raises(ValueError):
            featurizer.featurize(["CCO", "INVALID_SMILES", "c1ccccc1"])

    def test_maccs_fingerprint(self):
        """Test MACCS keys fingerprint."""
        featurizer = MolecularFeaturizer(fingerprint_type="maccs")

        smiles = "CCO"
        features = featurizer.featurize(smiles)

        # MACCS keys have 167 bits (index 0 is unused)
        assert len(features) == 167

    def test_with_descriptors(self):
        """Test featurization with descriptors."""
        featurizer = MolecularFeaturizer(
            fingerprint_type="morgan",
            n_bits=1024,
            include_descriptors=True
        )

        smiles = "CCO"
        features = featurizer.featurize(smiles)

        # Should have fingerprint + descriptors
        assert len(features) > 1024
        assert featurizer.n_features > 1024

    def test_feature_names(self):
        """Test feature name generation."""
        featurizer = MolecularFeaturizer(
            fingerprint_type="morgan",
            n_bits=512,
            include_descriptors=False
        )

        feature_names = featurizer.get_feature_names()
        assert len(feature_names) == 512
        assert all("MORGAN_bit_" in name for name in feature_names)


class TestBasicProperties:
    """Test basic property calculations."""

    def test_calculate_properties(self):
        """Test property calculation for simple molecule."""
        smiles = "CCO"
        props = calculate_basic_properties(smiles)

        assert "molecular_weight" in props
        assert "logp" in props
        assert "tpsa" in props
        assert "n_hba" in props
        assert "n_hbd" in props

        # Check reasonable values for ethanol
        assert 40 < props["molecular_weight"] < 50
        assert -1 < props["logp"] < 1
        assert props["n_hba"] == 1  # One O atom
        assert props["n_hbd"] == 1  # One OH group

    def test_invalid_smiles_properties(self):
        """Test error handling for invalid SMILES."""
        with pytest.raises(ValueError):
            calculate_basic_properties("INVALID_SMILES")


class TestUtilityFunctions:
    """Test utility functions."""

    def test_smiles_to_mol_safe(self):
        """Test safe SMILES parsing."""
        # Valid SMILES
        mol = smiles_to_mol_safe("CCO")
        assert mol is not None

        # Invalid SMILES
        mol = smiles_to_mol_safe("INVALID")
        assert mol is None

    def test_batch_featurize(self):
        """Test batch featurization utility."""
        featurizer = MolecularFeaturizer(fingerprint_type="morgan", n_bits=1024)
        smiles_list = ["CCO", "c1ccccc1", "CC(C)C"]

        feature_matrix, valid_smiles = batch_featurize(
            smiles_list,
            featurizer,
            chunk_size=2,
            show_progress=False
        )

        assert feature_matrix.shape == (3, 1024)
        assert len(valid_smiles) == 3
        assert valid_smiles == smiles_list

    def test_batch_featurize_with_invalid(self):
        """Test batch featurization with invalid SMILES."""
        featurizer = MolecularFeaturizer(fingerprint_type="morgan")
        smiles_list = ["CCO", "INVALID", "c1ccccc1"]

        feature_matrix, valid_smiles = batch_featurize(
            smiles_list,
            featurizer,
            show_progress=False
        )

        # Should skip invalid SMILES
        assert feature_matrix.shape[0] == 2
        assert len(valid_smiles) == 2
        assert "INVALID" not in valid_smiles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
