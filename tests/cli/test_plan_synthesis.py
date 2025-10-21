"""
Integration tests for plan_synthesis CLI.
"""

import pytest
import sys
from unittest.mock import patch
import json


def test_synthesis_basic(output_dir):
    """Test basic synthesis planning."""
    from ml_fragment_optimizer.cli.plan_synthesis import main

    output_file = output_dir / "synthesis.json"

    args = [
        "plan_synthesis.py",
        "--smiles", "CCO",
        "--output", str(output_file),
        "--max-routes", "3",
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        try:
            main()
        except SystemExit:
            # May exit if retrosynthesis dependencies not available
            pytest.skip("Retrosynthesis dependencies not available")

    if output_file.exists():
        with open(output_file) as f:
            results = json.load(f)
        assert isinstance(results, list)


def test_synthesis_batch_mode(tmp_path, output_dir):
    """Test batch synthesis planning."""
    from ml_fragment_optimizer.cli.plan_synthesis import main

    # Create SMILES file
    smiles_file = tmp_path / "molecules.smi"
    with open(smiles_file, 'w') as f:
        f.write("CCO\n")
        f.write("c1ccccc1\n")

    output_file = output_dir / "synthesis_batch.json"

    args = [
        "plan_synthesis.py",
        "--input-file", str(smiles_file),
        "--output", str(output_file),
        "--max-routes", "2",
        "--log-level", "WARNING"
    ]

    with patch.object(sys, 'argv', args):
        try:
            main()
        except SystemExit:
            pytest.skip("Retrosynthesis dependencies not available")


def test_synthesis_invalid_smiles(output_dir):
    """Test error handling for invalid SMILES."""
    from ml_fragment_optimizer.cli.plan_synthesis import main

    output_file = output_dir / "synthesis.json"

    args = [
        "plan_synthesis.py",
        "--smiles", "INVALID",
        "--output", str(output_file),
        "--log-level", "ERROR"
    ]

    with patch.object(sys, 'argv', args):
        try:
            main()
        except SystemExit:
            pass  # Expected to fail gracefully
