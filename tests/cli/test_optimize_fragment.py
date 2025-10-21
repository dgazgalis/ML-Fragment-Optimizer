"""
Integration tests for optimize_fragment CLI.
"""

import pytest
import sys
from unittest.mock import patch
import pandas as pd


@pytest.fixture
def trained_admet_model(sample_data_csv, tmp_path):
    """Create a trained ADMET model for optimization tests."""
    from ml_fragment_optimizer.models.admet_predictor import ADMETPredictor
    from ml_fragment_optimizer.utils.featurizers import MolecularFeaturizer

    df = pd.read_csv(sample_data_csv)
    smiles = df["SMILES"].tolist()
    properties = {"solubility": df["solubility"].tolist(), "logp": df["logp"].tolist()}

    featurizer = MolecularFeaturizer()
    predictor = ADMETPredictor(
        properties=["solubility", "logp"],
        model_type="random_forest",
        featurizer=featurizer
    )
    predictor.fit(smiles, properties)

    model_path = tmp_path / "admet_model.pkl"
    predictor.save(model_path)
    return model_path


def test_optimize_basic(output_dir):
    """Test basic fragment optimization without model."""
    from ml_fragment_optimizer.cli.optimize_fragment import main

    output_file = output_dir / "optimized.csv"

    args = [
        "optimize_fragment.py",
        "--fragment", "c1ccccc1",
        "--target-property", "solubility",
        "--target-value", "-1.0",
        "--num-suggestions", "5",
        "--output", str(output_file),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    assert output_file.exists()
    df = pd.read_csv(output_file)
    assert len(df) > 0
    assert "smiles" in df.columns
    assert "score" in df.columns
    assert "sa_score" in df.columns


def test_optimize_with_model(trained_admet_model, output_dir):
    """Test optimization with trained ADMET model."""
    from ml_fragment_optimizer.cli.optimize_fragment import main

    output_file = output_dir / "optimized_model.csv"

    args = [
        "optimize_fragment.py",
        "--fragment", "c1ccccc1",
        "--target-property", "solubility",
        "--target-value", "-1.0",
        "--model", str(trained_admet_model),
        "--num-suggestions", "5",
        "--output", str(output_file),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    assert output_file.exists()
    df = pd.read_csv(output_file)
    assert "solubility" in df.columns


def test_optimize_maximize(output_dir):
    """Test optimization with maximization."""
    from ml_fragment_optimizer.cli.optimize_fragment import main

    output_file = output_dir / "optimized_max.csv"

    args = [
        "optimize_fragment.py",
        "--fragment", "c1ccccc1",
        "--target-property", "solubility",
        "--maximize",
        "--num-suggestions", "5",
        "--output", str(output_file),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    assert output_file.exists()


def test_optimize_multiple_properties(trained_admet_model, output_dir):
    """Test optimization with multiple target properties."""
    from ml_fragment_optimizer.cli.optimize_fragment import main

    output_file = output_dir / "optimized_multi.csv"

    args = [
        "optimize_fragment.py",
        "--fragment", "CCO",
        "--target-property", "solubility,logp",
        "--target-value", "-1.0,2.0",
        "--model", str(trained_admet_model),
        "--num-suggestions", "5",
        "--output", str(output_file),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    assert output_file.exists()
    df = pd.read_csv(output_file)
    assert "solubility" in df.columns
    assert "logp" in df.columns


def test_optimize_sa_score_filter(output_dir):
    """Test SA score filtering."""
    from ml_fragment_optimizer.cli.optimize_fragment import main

    output_file = output_dir / "optimized_sa.csv"

    args = [
        "optimize_fragment.py",
        "--fragment", "c1ccccc1",
        "--target-property", "solubility",
        "--target-value", "-1.0",
        "--max-sa-score", "5.0",
        "--num-suggestions", "5",
        "--output", str(output_file),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    assert output_file.exists()
    df = pd.read_csv(output_file)
    # All suggestions should have SA score <= 5.0
    assert all(df["sa_score"] <= 5.0)


def test_optimize_invalid_fragment(output_dir):
    """Test error handling for invalid fragment SMILES."""
    from ml_fragment_optimizer.cli.optimize_fragment import main

    output_file = output_dir / "optimized.csv"

    args = [
        "optimize_fragment.py",
        "--fragment", "INVALID",
        "--target-property", "solubility",
        "--target-value", "-1.0",
        "--output", str(output_file),
        "--log-level", "ERROR"
    ]

    with patch.object(sys, 'argv', args):
        with pytest.raises(SystemExit):
            main()


def test_optimize_mismatched_properties(output_dir):
    """Test error for mismatched property/value counts."""
    from ml_fragment_optimizer.cli.optimize_fragment import main

    output_file = output_dir / "optimized.csv"

    args = [
        "optimize_fragment.py",
        "--fragment", "c1ccccc1",
        "--target-property", "solubility,logp",
        "--target-value", "-1.0",  # Only one value for two properties
        "--output", str(output_file),
        "--log-level", "ERROR"
    ]

    with patch.object(sys, 'argv', args):
        with pytest.raises(SystemExit):
            main()
