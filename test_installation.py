"""
Installation Test Script

This script verifies that all components of ML-Fragment-Optimizer
are correctly installed and functional.

Run this after installation to ensure everything works.

Author: Claude Code
Date: 2025-10-20
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    print("-" * 50)

    tests = [
        ("NumPy", "import numpy as np"),
        ("Pandas", "import pandas as pd"),
        ("PyTorch", "import torch"),
        ("Scikit-learn", "import sklearn"),
        ("RDKit", "from rdkit import Chem"),
        ("Matplotlib", "import matplotlib.pyplot as plt"),
        ("Seaborn", "import seaborn as sns"),
    ]

    passed = 0
    failed = 0

    for name, import_str in tests:
        try:
            exec(import_str)
            print(f"✓ {name:20s} OK")
            passed += 1
        except ImportError as e:
            print(f"✗ {name:20s} FAILED: {e}")
            failed += 1

    # Test optional packages
    print("\nOptional packages:")
    optional_tests = [
        ("PyTorch Geometric", "from torch_geometric.data import Data"),
    ]

    for name, import_str in optional_tests:
        try:
            exec(import_str)
            print(f"✓ {name:20s} OK")
        except ImportError:
            print(f"○ {name:20s} Not installed (optional)")

    print("-" * 50)
    print(f"Core packages: {passed} passed, {failed} failed")

    return failed == 0


def test_modules():
    """Test that ML-Fragment-Optimizer modules work"""
    print("\nTesting ML-Fragment-Optimizer modules...")
    print("-" * 50)

    try:
        # Test fingerprints module
        from src.models import MolecularFeaturizer

        featurizer = MolecularFeaturizer()
        features = featurizer.featurize("CCO")

        if features is None:
            print("✗ MolecularFeaturizer: Failed to featurize ethanol")
            return False

        print("✓ MolecularFeaturizer: OK")

        # Test ADMET predictor
        from src.models import ADMETPredictor, ADMETConfig

        config = ADMETConfig(model_type='fingerprint', num_epochs=1)
        predictor = ADMETPredictor(config)
        print("✓ ADMETPredictor: OK")

        # Test uncertainty module
        from src.models import EvidentialOutput, MCDropoutWrapper
        import torch

        evid_layer = EvidentialOutput(128)
        x = torch.randn(4, 128)
        gamma, lambda_p, alpha, beta = evid_layer(x)
        print("✓ Uncertainty quantification: OK")

        # Test data utils
        from src.models.data_utils import ADMETDataProcessor

        processor = ADMETDataProcessor()
        print("✓ Data utilities: OK")

        print("-" * 50)
        print("All modules working correctly!")
        return True

    except Exception as e:
        print(f"✗ Module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rdkit():
    """Test RDKit functionality"""
    print("\nTesting RDKit functionality...")
    print("-" * 50)

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, AllChem

        # Test SMILES parsing
        mol = Chem.MolFromSmiles("CCO")
        if mol is None:
            print("✗ SMILES parsing failed")
            return False
        print("✓ SMILES parsing: OK")

        # Test descriptor calculation
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        print(f"✓ Descriptors: MW={mw:.2f}, LogP={logp:.2f}")

        # Test fingerprints
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        print(f"✓ Morgan fingerprint: {len(fp)} bits")

        print("-" * 50)
        print("RDKit working correctly!")
        return True

    except Exception as e:
        print(f"✗ RDKit test failed: {e}")
        return False


def test_pytorch():
    """Test PyTorch functionality"""
    print("\nTesting PyTorch functionality...")
    print("-" * 50)

    try:
        import torch
        import torch.nn as nn

        # Test tensor creation
        x = torch.randn(4, 10)
        print(f"✓ Tensor creation: shape {x.shape}")

        # Test GPU availability
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            device = torch.device('cuda')
        else:
            print("○ GPU not available (using CPU)")
            device = torch.device('cpu')

        # Test simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        model = model.to(device)
        x = x.to(device)

        with torch.no_grad():
            y = model(x)

        print(f"✓ Model forward pass: input {x.shape} → output {y.shape}")

        print("-" * 50)
        print("PyTorch working correctly!")
        return True

    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False


def test_example():
    """Test complete workflow with example molecules"""
    print("\nTesting complete workflow...")
    print("-" * 50)

    try:
        from src.models import MolecularFeaturizer

        # Test molecules
        test_smiles = [
            "CCO",  # Ethanol
            "c1ccccc1",  # Benzene
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        ]

        featurizer = MolecularFeaturizer()
        features, failed = featurizer.featurize_batch(test_smiles)

        print(f"✓ Featurized {len(features)}/{len(test_smiles)} molecules")

        if len(failed) > 0:
            print(f"  Failed indices: {failed}")

        for feat in features[:2]:  # Show first 2
            print(f"\n  SMILES: {feat.smiles}")
            if feat.morgan_fp is not None:
                print(f"    Morgan FP: {feat.morgan_fp.shape}")
            if feat.maccs_keys is not None:
                print(f"    MACCS keys: {feat.maccs_keys.shape}")
            if feat.rdkit_descriptors is not None:
                print(f"    RDKit descriptors: {feat.rdkit_descriptors.shape}")
            if feat.graph_data is not None:
                print(f"    Graph: {feat.graph_data.x.shape[0]} nodes, "
                      f"{feat.graph_data.edge_index.shape[1]} edges")

        print("-" * 50)
        print("Complete workflow test passed!")
        return True

    except Exception as e:
        print(f"✗ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("ML-Fragment-Optimizer Installation Test")
    print("=" * 70)

    results = {
        "Imports": test_imports(),
        "Modules": test_modules(),
        "RDKit": test_rdkit(),
        "PyTorch": test_pytorch(),
        "Example": test_example(),
    }

    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s} {status}")
        all_passed = all_passed and passed

    print("=" * 70)

    if all_passed:
        print("\n🎉 All tests passed! Installation successful.")
        print("\nYou can now:")
        print("  1. Run example training: python examples/example_training.py")
        print("  2. Run example inference: python examples/example_inference.py")
        print("  3. Read documentation: README.md and ARCHITECTURE.md")
    else:
        print("\n⚠ Some tests failed. Please check the error messages above.")
        print("\nCommon issues:")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - PyTorch Geometric: conda install pyg -c pyg")
        print("  - CUDA issues: Check PyTorch CUDA installation")

    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
