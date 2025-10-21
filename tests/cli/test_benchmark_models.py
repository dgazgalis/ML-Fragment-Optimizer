"""
Integration tests for benchmark_models CLI.
"""

import pytest
import sys
from unittest.mock import patch
import pandas as pd
import json


def test_benchmark_basic(sample_data_csv, output_dir):
    """Test basic benchmarking."""
    from ml_fragment_optimizer.cli.benchmark_models import main

    args = [
        "benchmark_models.py",
        "--data", str(sample_data_csv),
        "--properties", "solubility",
        "--split-type", "random",
        "--output-dir", str(output_dir),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    # Check outputs
    assert (output_dir / "benchmark_results.json").exists()
    assert (output_dir / "rmse_comparison.png").exists()
    assert (output_dir / "r2_comparison.png").exists()


def test_benchmark_multiple_models(sample_data_csv, output_dir):
    """Test benchmarking multiple model types."""
    pytest.importorskip("xgboost")

    from ml_fragment_optimizer.cli.benchmark_models import main

    args = [
        "benchmark_models.py",
        "--data", str(sample_data_csv),
        "--properties", "solubility",
        "--model-types", "random_forest,xgboost",
        "--split-type", "random",
        "--output-dir", str(output_dir),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    # Check results include both models
    with open(output_dir / "benchmark_results.json") as f:
        results = json.load(f)

    assert "random_forest" in results
    assert "xgboost" in results


def test_benchmark_scaffold_split(sample_data_csv, output_dir):
    """Test scaffold-based splitting."""
    from ml_fragment_optimizer.cli.benchmark_models import main

    args = [
        "benchmark_models.py",
        "--data", str(sample_data_csv),
        "--properties", "solubility",
        "--split-type", "scaffold",
        "--test-size", "0.3",
        "--output-dir", str(output_dir),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    assert (output_dir / "benchmark_results.json").exists()


def test_benchmark_multiple_properties(sample_data_csv, output_dir):
    """Test benchmarking with multiple properties."""
    from ml_fragment_optimizer.cli.benchmark_models import main

    args = [
        "benchmark_models.py",
        "--data", str(sample_data_csv),
        "--properties", "solubility,logp",
        "--split-type", "random",
        "--output-dir", str(output_dir),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    with open(output_dir / "benchmark_results.json") as f:
        results = json.load(f)

    # Check both properties benchmarked
    for model_name, props in results.items():
        assert "solubility" in props
        assert "logp" in props


def test_benchmark_compare_mode(sample_data_csv, tmp_path, output_dir):
    """Test compare-only mode with pre-trained models."""
    from ml_fragment_optimizer.models.admet_predictor import ADMETPredictor
    from ml_fragment_optimizer.utils.featurizers import MolecularFeaturizer
    from ml_fragment_optimizer.cli.benchmark_models import main

    # Train a model first
    df = pd.read_csv(sample_data_csv)
    smiles = df["SMILES"].tolist()
    properties = {"solubility": df["solubility"].tolist()}

    featurizer = MolecularFeaturizer()
    predictor = ADMETPredictor(
        properties=["solubility"],
        model_type="random_forest",
        featurizer=featurizer
    )
    predictor.fit(smiles, properties)

    model_path = tmp_path / "model.pkl"
    predictor.save(model_path)

    # Now benchmark it
    args = [
        "benchmark_models.py",
        "--data", str(sample_data_csv),
        "--models", str(model_path),
        "--compare-only",
        "--output-dir", str(output_dir),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    assert (output_dir / "benchmark_results.json").exists()
