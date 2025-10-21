"""
Integration tests for active_learning_loop CLI.
"""

import pytest
import sys
from unittest.mock import patch
import pandas as pd


def test_active_learning_simulate(sample_data_csv, tmp_path, output_dir):
    """Test active learning in simulation mode."""
    from ml_fragment_optimizer.cli.active_learning_loop import main

    # Create candidate pool
    candidate_file = tmp_path / "candidates.smi"
    with open(candidate_file, 'w') as f:
        for smi in ["CCO", "c1ccccc1", "CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]:
            f.write(f"{smi}\n")

    args = [
        "active_learning_loop.py",
        "--initial-data", str(sample_data_csv),
        "--candidate-pool", str(candidate_file),
        "--properties", "solubility",
        "--iterations", "2",
        "--batch-size", "2",
        "--simulate",
        "--output-dir", str(output_dir),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    # Check outputs
    assert (output_dir / "final_model.pkl").exists()
    assert (output_dir / "final_training_data.csv").exists()
    assert (output_dir / "optimization_history.csv").exists()


def test_active_learning_acquisition_functions(sample_data_csv, tmp_path, output_dir):
    """Test different acquisition functions."""
    from ml_fragment_optimizer.cli.active_learning_loop import main

    candidate_file = tmp_path / "candidates.smi"
    with open(candidate_file, 'w') as f:
        for smi in ["CCO", "c1ccccc1", "CC(=O)O"]:
            f.write(f"{smi}\n")

    for acq_func in ["ei", "ucb", "greedy"]:
        acq_output_dir = output_dir / acq_func
        acq_output_dir.mkdir()

        args = [
            "active_learning_loop.py",
            "--initial-data", str(sample_data_csv),
            "--candidate-pool", str(candidate_file),
            "--properties", "solubility",
            "--iterations", "1",
            "--batch-size", "1",
            "--acquisition", acq_func,
            "--simulate",
            "--output-dir", str(acq_output_dir),
            "--log-level", "WARNING"
        ]

        with patch.object(sys, 'argv', args):
            main()

        assert (acq_output_dir / "final_model.pkl").exists()


def test_active_learning_diversity(sample_data_csv, tmp_path, output_dir):
    """Test diversity-aware selection."""
    from ml_fragment_optimizer.cli.active_learning_loop import main

    candidate_file = tmp_path / "candidates.smi"
    with open(candidate_file, 'w') as f:
        for smi in ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCC"]:
            f.write(f"{smi}\n")

    args = [
        "active_learning_loop.py",
        "--initial-data", str(sample_data_csv),
        "--candidate-pool", str(candidate_file),
        "--properties", "solubility",
        "--iterations", "1",
        "--batch-size", "2",
        "--diversity-weight", "0.5",
        "--simulate",
        "--output-dir", str(output_dir),
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        main()

    history = pd.read_csv(output_dir / "optimization_history.csv")
    assert len(history) == 1
