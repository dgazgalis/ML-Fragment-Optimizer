#!/usr/bin/env python3
"""
Unit Tests for QSAR/SAR Analysis Module

Tests all components of the QSAR package.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import unittest
import numpy as np
from rdkit import Chem

from qsar import mmpa, activity_cliffs, free_wilson, sar_visualization, feature_importance, bioisostere_suggester


class TestMMPA(unittest.TestCase):
    """Test Matched Molecular Pair Analysis."""

    def setUp(self):
        """Set up test data."""
        self.smiles = [
            "c1ccccc1",       # benzene
            "Fc1ccccc1",      # fluorobenzene
            "Clc1ccccc1",     # chlorobenzene
            "Brc1ccccc1",     # bromobenzene
        ]
        self.mols = [Chem.MolFromSmiles(s) for s in self.smiles]
        self.activities = [5.0, 5.5, 6.0, 6.5]

    def test_find_pairs(self):
        """Test finding matched pairs."""
        analyzer = mmpa.MatchedMolecularPairAnalyzer()
        pairs = analyzer.find_pairs(self.mols, self.activities)

        self.assertIsInstance(pairs, list)
        self.assertGreater(len(pairs), 0)

        # Check pair structure
        pair = pairs[0]
        self.assertIsInstance(pair, mmpa.MolecularPair)
        self.assertIn("mol1_idx", dir(pair))
        self.assertIn("mol2_idx", dir(pair))

    def test_analyze_transformations(self):
        """Test transformation analysis."""
        analyzer = mmpa.MatchedMolecularPairAnalyzer()
        pairs = analyzer.find_pairs(self.mols, self.activities)
        stats = analyzer.analyze_transformations(pairs, min_pairs=1)

        self.assertIsInstance(stats, dict)
        if stats:
            # Check statistics structure
            first_stat = list(stats.values())[0]
            self.assertIsInstance(first_stat, mmpa.TransformationStatistics)
            self.assertIn("transformation", dir(first_stat))
            self.assertIn("mean_change", dir(first_stat))

    def test_find_activity_cliffs(self):
        """Test activity cliff finding from pairs."""
        analyzer = mmpa.MatchedMolecularPairAnalyzer()
        pairs = analyzer.find_pairs(self.mols, self.activities)
        cliffs = analyzer.find_activity_cliffs(pairs, cliff_threshold=0.5)

        self.assertIsInstance(cliffs, list)

    def test_convenience_function(self):
        """Test convenience function."""
        pairs, stats = mmpa.find_matched_pairs(self.mols, self.activities)

        self.assertIsInstance(pairs, list)
        self.assertIsInstance(stats, dict)


class TestActivityCliffs(unittest.TestCase):
    """Test Activity Cliff Detection."""

    def setUp(self):
        """Set up test data with known cliff."""
        self.smiles = [
            "c1ccccc1",                   # benzene: 5.0
            "c1ccc(F)cc1",                # fluorobenzene: 5.5
            "c1ccc([N+](=O)[O-])cc1",    # nitrobenzene: 8.5 (cliff!)
        ]
        self.mols = [Chem.MolFromSmiles(s) for s in self.smiles]
        self.activities = [5.0, 5.5, 8.5]

    def test_detect_cliffs(self):
        """Test cliff detection."""
        analyzer = activity_cliffs.ActivityCliffAnalyzer(
            similarity_threshold=0.6,
            activity_threshold=2.0,
        )
        results = analyzer.detect_cliffs(self.mols, self.activities)

        self.assertIsInstance(results, activity_cliffs.CliffAnalysisResults)
        self.assertGreaterEqual(results.n_cliffs, 0)

    def test_calculate_similarity_matrix(self):
        """Test similarity matrix calculation."""
        analyzer = activity_cliffs.ActivityCliffAnalyzer()
        sim_matrix = analyzer.calculate_similarity_matrix(self.mols)

        self.assertEqual(sim_matrix.shape, (len(self.mols), len(self.mols)))
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(sim_matrix), np.ones(len(self.mols)))

    def test_sali_calculation(self):
        """Test SALI calculation."""
        analyzer = activity_cliffs.ActivityCliffAnalyzer()

        # Test normal case
        sali = analyzer._calculate_sali(similarity=0.8, activity_diff=2.0)
        expected = 2.0 / (1.0 - 0.8)  # = 10.0
        self.assertAlmostEqual(sali, expected)

        # Test edge case (identical molecules)
        sali = analyzer._calculate_sali(similarity=1.0, activity_diff=2.0)
        self.assertEqual(sali, 0.0)

    def test_cliff_aware_splits(self):
        """Test cliff-aware train/test splitting."""
        analyzer = activity_cliffs.ActivityCliffAnalyzer()
        train_idx, test_idx = analyzer.create_cliff_aware_splits(
            self.mols, self.activities, test_size=0.3, random_state=42
        )

        self.assertEqual(len(train_idx) + len(test_idx), len(self.mols))
        self.assertEqual(len(set(train_idx) & set(test_idx)), 0)  # No overlap

    def test_convenience_function(self):
        """Test convenience function."""
        results = activity_cliffs.detect_activity_cliffs(
            self.mols, self.activities, activity_threshold=2.0
        )

        self.assertIsInstance(results, activity_cliffs.CliffAnalysisResults)


class TestFreeWilson(unittest.TestCase):
    """Test Free-Wilson Analysis."""

    def setUp(self):
        """Set up test data."""
        # Para-substituted benzenes
        self.smiles = [
            "c1ccc(F)cc1",
            "c1ccc(Cl)cc1",
            "c1ccc(Br)cc1",
            "c1ccc(C)cc1",
            "c1ccc(O)cc1",
        ]
        self.mols = [Chem.MolFromSmiles(s) for s in self.smiles]
        self.activities = [5.5, 6.0, 6.5, 5.2, 5.8]

    def test_fit(self):
        """Test model fitting."""
        analyzer = free_wilson.FreeWilsonAnalyzer()
        model = analyzer.fit(self.mols, self.activities, position_atoms=[3])

        self.assertIsInstance(model, free_wilson.FreeWilsonModel)
        self.assertIsNotNone(model.baseline_activity)
        self.assertGreater(len(model.contributions), 0)

    def test_contributions(self):
        """Test substituent contributions."""
        analyzer = free_wilson.FreeWilsonAnalyzer()
        model = analyzer.fit(self.mols, self.activities, position_atoms=[3])

        for contrib in model.contributions:
            self.assertIsInstance(contrib, free_wilson.SubstituentContribution)
            self.assertIn("position", dir(contrib))
            self.assertIn("substituent", dir(contrib))
            self.assertIn("contribution", dir(contrib))

    def test_get_top_contributors(self):
        """Test getting top contributors."""
        analyzer = free_wilson.FreeWilsonAnalyzer()
        model = analyzer.fit(self.mols, self.activities, position_atoms=[3])

        top = analyzer.get_top_contributors(model, n=3)
        self.assertLessEqual(len(top), 3)


class TestSARVisualization(unittest.TestCase):
    """Test SAR Visualization."""

    def setUp(self):
        """Set up test data."""
        self.smiles = [
            "c1ccccc1",
            "c1ccc(Cl)cc1",
            "c1ccc(F)cc1",
            "c1ccc(Br)cc1",
        ]
        self.mols = [Chem.MolFromSmiles(s) for s in self.smiles]
        self.activities = [5.0, 6.5, 6.0, 7.0]

    def test_plot_activity_landscape_pca(self):
        """Test PCA activity landscape."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        viz = sar_visualization.SARVisualizer()
        ax = viz.plot_activity_landscape(
            self.mols, self.activities, method='pca'
        )

        self.assertIsNotNone(ax)

    def test_plot_similarity_heatmap(self):
        """Test similarity heatmap."""
        import matplotlib
        matplotlib.use('Agg')

        viz = sar_visualization.SARVisualizer()
        ax = viz.plot_similarity_heatmap(self.mols, self.activities)

        self.assertIsNotNone(ax)

    def test_calculate_fingerprints(self):
        """Test fingerprint calculation."""
        viz = sar_visualization.SARVisualizer()
        fps = viz._calculate_fingerprints(self.mols, as_bitvect=False)

        self.assertEqual(len(fps), len(self.mols))
        for fp in fps:
            self.assertIsNotNone(fp)


class TestFeatureImportance(unittest.TestCase):
    """Test Feature Importance."""

    def setUp(self):
        """Set up test data."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        self.X_train, self.X_test = X[:80], X[80:]
        self.y_train, self.y_test = y[:80], y[80:]

        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def test_permutation_importance(self):
        """Test permutation importance."""
        from sklearn.metrics import r2_score

        pi = feature_importance.PermutationImportance(
            self.model, r2_score, n_repeats=3
        )
        importances = pi.calculate(self.X_test, self.y_test)

        self.assertIsInstance(importances, dict)
        self.assertEqual(len(importances), self.X_test.shape[1])

    def test_shap_interpreter_init(self):
        """Test SHAP interpreter initialization."""
        try:
            interpreter = feature_importance.SHAPInterpreter(
                self.model, self.X_train[:10]
            )
            self.assertIsNotNone(interpreter)
        except ImportError:
            self.skipTest("SHAP not installed")


class TestBioisostereSuggester(unittest.TestCase):
    """Test Bioisosteric Replacement Suggester."""

    def setUp(self):
        """Set up test data."""
        self.mols = [
            Chem.MolFromSmiles("c1ccc(Cl)cc1"),  # chlorobenzene
            Chem.MolFromSmiles("c1ccc(C(=O)O)cc1"),  # benzoic acid
        ]

    def test_suggest_replacements(self):
        """Test replacement suggestions."""
        suggester = bioisostere_suggester.BioisostereSuggester()

        for mol in self.mols:
            suggestions = suggester.suggest_replacements(
                mol, max_suggestions=5, filter_drug_like=False
            )

            self.assertIsInstance(suggestions, list)
            # May or may not find suggestions depending on patterns

    def test_drug_likeness(self):
        """Test drug-likeness calculation."""
        suggester = bioisostere_suggester.BioisostereSuggester()

        for mol in self.mols:
            is_drug_like = suggester._is_drug_like(mol)
            self.assertIsInstance(is_drug_like, bool)

            score = suggester._calculate_drug_likeness(mol)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_sa_score(self):
        """Test synthetic accessibility estimation."""
        suggester = bioisostere_suggester.BioisostereSuggester()

        for mol in self.mols:
            sa_score = suggester._estimate_sa_score(mol)
            self.assertGreaterEqual(sa_score, 1.0)
            self.assertLessEqual(sa_score, 10.0)

    def test_convenience_function(self):
        """Test convenience function."""
        suggestions = bioisostere_suggester.suggest_bioisosteres(
            self.mols[0], max_suggestions=5
        )

        self.assertIsInstance(suggestions, list)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""

    def test_complete_workflow(self):
        """Test complete QSAR workflow."""
        # Dataset
        smiles = [
            "c1ccccc1",
            "c1ccc(Cl)cc1",
            "c1ccc(F)cc1",
            "c1ccc(Br)cc1",
            "c1ccc([N+](=O)[O-])cc1",
        ]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        activities = [5.0, 6.0, 5.5, 6.5, 8.5]

        # MMPA
        pairs, stats = mmpa.find_matched_pairs(mols, activities)
        self.assertIsInstance(pairs, list)

        # Activity cliffs
        cliff_results = activity_cliffs.detect_activity_cliffs(
            mols, activities, activity_threshold=2.0
        )
        self.assertIsInstance(cliff_results, activity_cliffs.CliffAnalysisResults)

        # Should detect cliff between benzene and nitrobenzene
        self.assertGreater(cliff_results.n_cliffs, 0)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
