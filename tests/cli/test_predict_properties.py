"""
Integration tests for predict_properties CLI.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch
import pandas as pd


@pytest.fixture
def trained_model(sample_data_csv, tmp_path):
    """Create a trained model for prediction tests."""
    from ml_fragment_optimizer.models.admet_predictor import ADMETPredictor
    from ml_fragment_optimizer.utils.featurizers import MolecularFeaturizer

    # Load training data
    df = pd.read_csv(sample_data_csv)
    smiles = df["SMILES"].tolist()
    properties = {"solubility": df["solubility"].tolist()}

    # Train model
    featurizer = MolecularFeaturizer()
    predictor = ADMETPredictor(
        properties=["solubility"],
        model_type="random_forest",
        featurizer=featurizer
    )
    predictor.fit(smiles, properties)

    # Save model
    model_path = tmp_path / "test_model.pkl"
    predictor.save(model_path)

    return model_path


def test_predict_basic(trained_model, sample_smiles_file, output_dir):
    """Test basic prediction from SMILES file."""
    from ml_fragment_optimizer.cli.predict_properties import main

    output_file = output_dir / "predictions.csv"

    args = [
        "predict_properties.py",
        "--model", str(trained_model),
        "--input", str(sample_smiles_file),
        "--output", str(output_file),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    # Check output exists and has predictions
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert "SMILES" in df.columns
    assert "ID" in df.columns
    assert "solubility_predicted" in df.columns
    assert len(df) > 0


def test_predict_with_uncertainty(trained_model, sample_smiles_file, output_dir):
    """Test prediction with uncertainty quantification."""
    from ml_fragment_optimizer.cli.predict_properties import main

    output_file = output_dir / "predictions_uncertainty.csv"

    args = [
        "predict_properties.py",
        "--model", str(trained_model),
        "--input", str(sample_smiles_file),
        "--output", str(output_file),
        "--uncertainty",
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    df = pd.read_csv(output_file)
    assert "solubility_predicted" in df.columns
    assert "solubility_uncertainty" in df.columns


def test_predict_csv_input(trained_model, sample_data_csv, output_dir):
    """Test prediction from CSV file."""
    from ml_fragment_optimizer.cli.predict_properties import main

    output_file = output_dir / "predictions_csv.csv"

    args = [
        "predict_properties.py",
        "--model", str(trained_model),
        "--input", str(sample_data_csv),
        "--output", str(output_file),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    df = pd.read_csv(output_file)
    assert len(df) > 0
    assert "solubility_predicted" in df.columns


def test_predict_with_outlier_detection(trained_model, sample_smiles_file, output_dir):
    """Test prediction with outlier flagging."""
    from ml_fragment_optimizer.cli.predict_properties import main

    output_file = output_dir / "predictions_outliers.csv"

    args = [
        "predict_properties.py",
        "--model", str(trained_model),
        "--input", str(sample_smiles_file),
        "--output", str(output_file),
        "--flag-outliers",
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    df = pd.read_csv(output_file)
    assert "is_outlier" in df.columns
    assert df["is_outlier"].dtype == bool


def test_predict_batch_size(trained_model, sample_smiles_file, output_dir):
    """Test prediction with custom batch size."""
    from ml_fragment_optimizer.cli.predict_properties import main

    output_file = output_dir / "predictions_batch.csv"

    args = [
        "predict_properties.py",
        "--model", str(trained_model),
        "--input", str(sample_smiles_file),
        "--output", str(output_file),
        "--batch-size", "2",
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    df = pd.read_csv(output_file)
    assert len(df) > 0


def test_predict_missing_model(sample_smiles_file, output_dir, tmp_path):
    """Test error handling for missing model file."""
    from ml_fragment_optimizer.cli.predict_properties import main

    output_file = output_dir / "predictions.csv"

    args = [
        "predict_properties.py",
        "--model", str(tmp_path / "nonexistent.pkl"),
        "--input", str(sample_smiles_file),
        "--output", str(output_file),
        "--log-level", "ERROR"
    ]

    with patch.object(sys, 'argv', args):
        with pytest.raises((SystemExit, FileNotFoundError)):
            main()


def test_predict_missing_input(trained_model, output_dir, tmp_path):
    """Test error handling for missing input file."""
    from ml_fragment_optimizer.cli.predict_properties import main

    output_file = output_dir / "predictions.csv"

    args = [
        "predict_properties.py",
        "--model", str(trained_model),
        "--input", str(tmp_path / "nonexistent.smi"),
        "--output", str(output_file),
        "--log-level", "ERROR"
    ]

    with patch.object(sys, 'argv', args):
        with pytest.raises(SystemExit):
            main()


def test_predict_invalid_smiles(trained_model, tmp_path, output_dir):
    """Test prediction with invalid SMILES."""
    from ml_fragment_optimizer.cli.predict_properties import main

    # Create file with invalid SMILES
    invalid_file = tmp_path / "invalid.smi"
    with open(invalid_file, 'w') as f:
        f.write("INVALID mol1\n")
        f.write("CCO mol2\n")
        f.write("xyz123 mol3\n")

    output_file = output_dir / "predictions_invalid.csv"

    args = [
        "predict_properties.py",
        "--model", str(trained_model),
        "--input", str(invalid_file),
        "--output", str(output_file),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    # Should complete with valid molecules only
    df = pd.read_csv(output_file)
    assert len(df) == 1  # Only CCO is valid
    assert df["SMILES"].iloc[0] == "CCO"


def test_predict_mixed_valid_invalid(trained_model, sample_data_csv_mixed, output_dir):
    """Test prediction with mixed valid/invalid SMILES."""
    from ml_fragment_optimizer.cli.predict_properties import main

    output_file = output_dir / "predictions_mixed.csv"

    args = [
        "predict_properties.py",
        "--model", str(trained_model),
        "--input", str(sample_data_csv_mixed),
        "--output", str(output_file),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    df = pd.read_csv(output_file)
    # Should have predictions for valid molecules only
    assert len(df) > 0
    assert "solubility_predicted" in df.columns


def test_predict_output_directory_creation(trained_model, sample_smiles_file, tmp_path):
    """Test automatic output directory creation."""
    from ml_fragment_optimizer.cli.predict_properties import main

    output_file = tmp_path / "new_dir" / "subdir" / "predictions.csv"

    args = [
        "predict_properties.py",
        "--model", str(trained_model),
        "--input", str(sample_smiles_file),
        "--output", str(output_file),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    assert output_file.exists()
    assert output_file.parent.exists()
