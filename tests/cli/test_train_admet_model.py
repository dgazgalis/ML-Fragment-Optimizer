"""
Integration tests for train_admet_model CLI.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch
import pandas as pd


def test_train_basic(sample_data_csv, output_dir):
    """Test basic training with minimal arguments."""
    from ml_fragment_optimizer.cli.train_admet_model import main

    args = [
        "train_admet_model.py",
        "--data", str(sample_data_csv),
        "--properties", "solubility",
        "--output-dir", str(output_dir),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    # Check outputs exist
    assert (output_dir / "admet_model.pkl").exists()
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "training.log").exists()


def test_train_multiple_properties(sample_data_csv, output_dir):
    """Test training with multiple properties."""
    from ml_fragment_optimizer.cli.train_admet_model import main

    args = [
        "train_admet_model.py",
        "--data", str(sample_data_csv),
        "--properties", "solubility,logp,mw",
        "--output-dir", str(output_dir),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    # Check model trained for all properties
    metrics = pd.read_csv(output_dir / "metrics.csv", index_col=0)
    assert "solubility" in metrics.index
    assert "logp" in metrics.index
    assert "mw" in metrics.index


def test_train_xgboost(sample_data_csv, output_dir):
    """Test training with XGBoost model."""
    pytest.importorskip("xgboost")  # Skip if XGBoost not installed

    from ml_fragment_optimizer.cli.train_admet_model import main

    args = [
        "train_admet_model.py",
        "--data", str(sample_data_csv),
        "--properties", "solubility",
        "--model-type", "xgboost",
        "--output-dir", str(output_dir),
        "--n-estimators", "50",
        "--learning-rate", "0.1",
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    assert (output_dir / "admet_model.pkl").exists()


def test_train_with_descriptors(sample_data_csv, output_dir):
    """Test training with molecular descriptors."""
    from ml_fragment_optimizer.cli.train_admet_model import main

    args = [
        "train_admet_model.py",
        "--data", str(sample_data_csv),
        "--properties", "solubility",
        "--use-descriptors",
        "--output-dir", str(output_dir),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    assert (output_dir / "admet_model.pkl").exists()


def test_train_missing_file(tmp_path, output_dir):
    """Test error handling for missing input file."""
    from ml_fragment_optimizer.cli.train_admet_model import main

    args = [
        "train_admet_model.py",
        "--data", str(tmp_path / "nonexistent.csv"),
        "--properties", "solubility",
        "--output-dir", str(output_dir),
        "--log-level", "ERROR"
    ]

    with patch.object(sys, 'argv', args):
        with pytest.raises(SystemExit):
            main()


def test_train_invalid_property(sample_data_csv, output_dir):
    """Test error handling for invalid property name."""
    from ml_fragment_optimizer.cli.train_admet_model import main

    args = [
        "train_admet_model.py",
        "--data", str(sample_data_csv),
        "--properties", "nonexistent_property",
        "--output-dir", str(output_dir),
        "--log-level", "ERROR"
    ]

    with patch.object(sys, 'argv', args):
        with pytest.raises(SystemExit):
            main()


def test_train_mixed_smiles(sample_data_csv_mixed, output_dir):
    """Test training with mixed valid/invalid SMILES."""
    from ml_fragment_optimizer.cli.train_admet_model import main

    args = [
        "train_admet_model.py",
        "--data", str(sample_data_csv_mixed),
        "--properties", "solubility",
        "--output-dir", str(output_dir),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    # Should succeed and train on valid molecules only
    assert (output_dir / "admet_model.pkl").exists()


def test_train_different_fingerprints(sample_data_csv, output_dir):
    """Test training with different fingerprint types."""
    from ml_fragment_optimizer.cli.train_admet_model import main

    for fp_type in ["morgan", "maccs", "rdkit"]:
        fp_output_dir = output_dir / fp_type
        fp_output_dir.mkdir()

        args = [
            "train_admet_model.py",
            "--data", str(sample_data_csv),
            "--properties", "solubility",
            "--fingerprint-type", fp_type,
            "--output-dir", str(fp_output_dir),
            "--log-level", "WARNING"
        ]

        with patch.object(sys, 'argv', args):
            main()

        assert (fp_output_dir / "admet_model.pkl").exists()


def test_train_small_dataset_warning(small_training_data, output_dir, caplog):
    """Test warning for small datasets."""
    from ml_fragment_optimizer.cli.train_admet_model import main

    args = [
        "train_admet_model.py",
        "--data", str(small_training_data),
        "--properties", "solubility",
        "--output-dir", str(output_dir),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    # Should complete but may warn about small dataset
    assert (output_dir / "admet_model.pkl").exists()


def test_train_custom_hyperparameters(sample_data_csv, output_dir):
    """Test training with custom hyperparameters."""
    from ml_fragment_optimizer.cli.train_admet_model import main

    args = [
        "train_admet_model.py",
        "--data", str(sample_data_csv),
        "--properties", "solubility",
        "--n-estimators", "200",
        "--max-depth", "10",
        "--fingerprint-bits", "4096",
        "--fingerprint-radius", "3",
        "--output-dir", str(output_dir),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    assert (output_dir / "admet_model.pkl").exists()

    # Check config was saved with custom values
    import yaml
    with open(output_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    assert config["n_estimators"] == 200
    assert config["max_depth"] == 10
    assert config["fingerprint_bits"] == 4096
