"""
Pytest configuration and shared fixtures for ML-Fragment-Optimizer tests.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
import shutil


@pytest.fixture
def sample_smiles():
    """Sample valid SMILES strings for testing."""
    return [
        "CCO",  # ethanol
        "c1ccccc1",  # benzene
        "CC(=O)O",  # acetic acid
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
        "CC(C)NCC(COc1ccccc1)O",  # propranolol
        "c1ccc2c(c1)ccc3c2cccc3",  # anthracene
        "C1CCC(CC1)N",  # cyclohexylamine
        "Cc1ccccc1",  # toluene
        "CCN(CC)CC",  # triethylamine
    ]


@pytest.fixture
def invalid_smiles():
    """Sample invalid SMILES strings for testing."""
    return [
        "INVALID",
        "C1CCC",  # Incomplete ring
        "xyz123",
        "",
        "C1CC1CCC1CC",  # Invalid structure
    ]


@pytest.fixture
def sample_properties():
    """Sample property values matching sample_smiles."""
    return {
        "solubility": [-0.77, -2.13, -0.17, -0.80, -3.97, -2.53, -4.45, -1.49, -2.73, -1.18],
        "logp": [0.46, 2.13, 0.17, -0.07, 3.97, 3.48, 4.45, 1.49, 2.73, 1.45],
        "mw": [46.07, 78.11, 60.05, 194.19, 206.28, 259.34, 178.23, 99.17, 92.14, 101.19],
    }


@pytest.fixture
def sample_data_csv(tmp_path, sample_smiles, sample_properties):
    """Create a sample CSV file with SMILES and properties."""
    csv_path = tmp_path / "sample_data.csv"

    df = pd.DataFrame({
        "SMILES": sample_smiles,
        **sample_properties
    })

    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_data_csv_mixed(tmp_path, sample_smiles, invalid_smiles, sample_properties):
    """Create a CSV with both valid and invalid SMILES."""
    csv_path = tmp_path / "mixed_data.csv"

    all_smiles = sample_smiles[:5] + invalid_smiles[:3] + sample_smiles[5:]
    n = len(all_smiles)

    # Extend properties to match length
    props = {}
    for key, values in sample_properties.items():
        extended = values[:5] + [0.0] * 3 + values[5:]
        props[key] = extended[:n]

    df = pd.DataFrame({
        "SMILES": all_smiles,
        **props
    })

    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_smiles_file(tmp_path, sample_smiles):
    """Create a .smi file with SMILES strings."""
    smi_path = tmp_path / "molecules.smi"

    with open(smi_path, 'w') as f:
        for i, smi in enumerate(sample_smiles):
            f.write(f"{smi} mol_{i+1}\n")

    return smi_path


@pytest.fixture
def small_training_data(tmp_path):
    """Create a small training dataset for quick tests."""
    csv_path = tmp_path / "small_train.csv"

    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
    solubility = [-0.77, -2.13, -0.17, -0.80]

    df = pd.DataFrame({
        "SMILES": smiles,
        "solubility": solubility
    })

    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    return out_dir


@pytest.fixture
def cleanup_after_test():
    """Cleanup fixture that runs after each test."""
    yield
    # Any cleanup code here


@pytest.fixture(scope="session")
def rdkit_available():
    """Check if RDKit is available."""
    try:
        from rdkit import Chem
        return True
    except ImportError:
        return False


@pytest.fixture
def mock_model_file(tmp_path):
    """Create a mock saved model file for testing."""
    import pickle

    model_path = tmp_path / "mock_model.pkl"

    # Create a simple mock model structure
    mock_data = {
        "properties": ["solubility", "logp"],
        "model_type": "random_forest",
        "version": "0.1.0"
    }

    with open(model_path, 'wb') as f:
        pickle.dump(mock_data, f)

    return model_path
