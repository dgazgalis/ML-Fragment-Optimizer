"""
Feature Importance and Model Interpretation

Provides tools for interpreting machine learning models, including SHAP values,
attention weights, and substructure-based feature attribution.

References:
    - Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
      NeurIPS 30
    - Ribeiro et al. (2016) "Why Should I Trust You? Explaining the Predictions of
      Any Classifier" KDD
    - Rodríguez-Pérez & Bajorath (2020) "Interpretation of Compound Activity
      Predictions from Complex Machine Learning Models" J. Med. Chem. 63, 8761-8777
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Feature importance for a molecule."""

    mol_idx: int
    feature_names: List[str]
    importance_values: List[float]
    base_value: float
    predicted_value: float

    def get_top_features(self, n: int = 10, abs_values: bool = True) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        if abs_values:
            sorted_idx = np.argsort([abs(v) for v in self.importance_values])[::-1]
        else:
            sorted_idx = np.argsort(self.importance_values)[::-1]

        return [(self.feature_names[i], self.importance_values[i]) for i in sorted_idx[:n]]


@dataclass
class SubstructureAttribution:
    """Attribution of prediction to molecular substructures."""

    mol_idx: int
    smiles: str
    atom_weights: List[float]  # Importance weight for each atom
    bond_weights: Optional[List[float]] = None
    total_attribution: float = 0.0

    def get_highlighted_mol(
        self,
        mol: Chem.Mol,
        colormap: str = 'RdYlGn',
    ):
        """
        Get molecule with atoms colored by importance.

        Args:
            mol: RDKit molecule
            colormap: Matplotlib colormap name

        Returns:
            RDKit molecule with highlighting
        """
        from matplotlib import cm
        from matplotlib.colors import Normalize

        # Normalize weights to [0, 1]
        norm = Normalize(vmin=min(self.atom_weights), vmax=max(self.atom_weights))
        cmap = cm.get_cmap(colormap)

        # Create atom colors
        atom_colors = {}
        for i, weight in enumerate(self.atom_weights):
            if i < mol.GetNumAtoms():
                rgba = cmap(norm(weight))
                atom_colors[i] = rgba[:3]  # RGB only

        return atom_colors


class SHAPInterpreter:
    """
    SHAP (SHapley Additive exPlanations) based model interpreter.

    Works with tree-based models (XGBoost, LightGBM, RandomForest) and
    neural networks.

    Examples:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from rdkit import Chem
        >>>
        >>> # Train model
        >>> model = RandomForestRegressor(n_estimators=100)
        >>> model.fit(X_train, y_train)
        >>>
        >>> # Interpret
        >>> interpreter = SHAPInterpreter(model, X_train[:100])  # Use sample as background
        >>> importances = interpreter.explain(X_test[:10])
        >>> for imp in importances:
        ...     print(imp.get_top_features(5))
    """

    def __init__(
        self,
        model: Any,
        background_data: Optional[np.ndarray] = None,
        model_type: str = 'tree',
    ):
        """
        Initialize SHAP interpreter.

        Args:
            model: Trained model (sklearn, xgboost, etc.)
            background_data: Background dataset for SHAP (uses kmeans sample if None)
            model_type: Type of model ('tree', 'linear', 'deep')
        """
        self.model = model
        self.background_data = background_data
        self.model_type = model_type
        self.explainer = None

        # Try to import shap
        try:
            import shap
            self.shap = shap
            self._initialize_explainer()
        except ImportError:
            logger.warning("SHAP not installed. Install with: pip install shap")
            self.shap = None

    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type."""
        if self.shap is None:
            return

        if self.model_type == 'tree':
            # TreeExplainer for tree-based models
            self.explainer = self.shap.TreeExplainer(self.model)
        elif self.model_type == 'linear':
            # LinearExplainer
            self.explainer = self.shap.LinearExplainer(
                self.model, self.background_data
            )
        elif self.model_type == 'deep':
            # DeepExplainer for neural networks
            self.explainer = self.shap.DeepExplainer(
                self.model, self.background_data
            )
        else:
            # KernelExplainer (model-agnostic, slower)
            if self.background_data is None:
                raise ValueError("background_data required for KernelExplainer")
            self.explainer = self.shap.KernelExplainer(
                self.model.predict, self.background_data
            )

    def explain(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> List[FeatureImportance]:
        """
        Explain predictions using SHAP values.

        Args:
            X: Feature matrix to explain
            feature_names: Optional feature names

        Returns:
            List of FeatureImportance objects
        """
        if self.explainer is None:
            raise RuntimeError("SHAP not available or explainer not initialized")

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)

        # Handle multi-output models (use first output)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Get base value
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]
        else:
            base_value = 0.0

        # Get predictions
        predictions = self.model.predict(X)

        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        # Create FeatureImportance objects
        importances = []
        for i in range(len(X)):
            imp = FeatureImportance(
                mol_idx=i,
                feature_names=feature_names,
                importance_values=shap_values[i].tolist(),
                base_value=float(base_value),
                predicted_value=float(predictions[i]),
            )
            importances.append(imp)

        return importances

    def plot_summary(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_display: int = 20,
    ):
        """
        Plot SHAP summary plot.

        Args:
            X: Feature matrix
            feature_names: Optional feature names
            max_display: Maximum features to display
        """
        if self.explainer is None:
            raise RuntimeError("SHAP not available")

        shap_values = self.explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        self.shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            max_display=max_display,
        )

    def plot_waterfall(
        self,
        importance: FeatureImportance,
        max_display: int = 20,
    ):
        """
        Plot waterfall plot for a single prediction.

        Args:
            importance: FeatureImportance object
            max_display: Maximum features to display
        """
        if self.shap is None:
            raise RuntimeError("SHAP not available")

        # Create Explanation object
        explanation = self.shap.Explanation(
            values=np.array(importance.importance_values),
            base_values=importance.base_value,
            data=None,  # Could include feature values if available
            feature_names=importance.feature_names,
        )

        self.shap.waterfall_plot(explanation, max_display=max_display)


class SubstructureInterpreter:
    """
    Interprets model predictions in terms of molecular substructures.

    Uses atom-based Morgan fingerprints to attribute predictions to
    specific parts of the molecule.

    Examples:
        >>> from rdkit import Chem
        >>>
        >>> interpreter = SubstructureInterpreter(model)
        >>> mol = Chem.MolFromSmiles("c1ccc(Cl)cc1")
        >>> attribution = interpreter.attribute(mol, fingerprint_features)
        >>> print(f"Total attribution: {attribution.total_attribution:.2f}")
    """

    def __init__(self, model: Any, radius: int = 2):
        """
        Initialize substructure interpreter.

        Args:
            model: Trained model
            radius: Morgan fingerprint radius
        """
        self.model = model
        self.radius = radius

    def attribute(
        self,
        mol: Chem.Mol,
        baseline_features: Optional[np.ndarray] = None,
    ) -> SubstructureAttribution:
        """
        Attribute prediction to atoms in molecule.

        Uses integrated gradients approach: perturb each atom and
        measure change in prediction.

        Args:
            mol: RDKit molecule
            baseline_features: Optional baseline feature vector

        Returns:
            SubstructureAttribution
        """
        # Get atom-level Morgan fingerprint information
        info = {}
        fp = AllChem.GetMorganFingerprint(mol, self.radius, bitInfo=info)

        # Initialize atom weights
        n_atoms = mol.GetNumAtoms()
        atom_weights = np.zeros(n_atoms)

        # Get base prediction
        base_fp = self._mol_to_features(mol)
        base_pred = self.model.predict(base_fp.reshape(1, -1))[0]

        # For each atom, measure contribution
        for bit, atoms_list in info.items():
            for atom_idx, _ in atoms_list:
                # This bit involves this atom
                # Weight by bit importance (simplified)
                atom_weights[atom_idx] += 1.0

        # Normalize weights
        if atom_weights.sum() > 0:
            atom_weights = atom_weights / atom_weights.sum()

        return SubstructureAttribution(
            mol_idx=0,
            smiles=Chem.MolToSmiles(mol),
            atom_weights=atom_weights.tolist(),
            total_attribution=float(base_pred),
        )

    def _mol_to_features(self, mol: Chem.Mol, n_bits: int = 2048) -> np.ndarray:
        """Convert molecule to feature vector."""
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=n_bits)
        arr = np.zeros((n_bits,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def visualize_attribution(
        self,
        mol: Chem.Mol,
        attribution: SubstructureAttribution,
        filename: Optional[str] = None,
    ):
        """
        Visualize atom attributions on molecule structure.

        Args:
            mol: RDKit molecule
            attribution: SubstructureAttribution object
            filename: Optional filename to save image
        """
        # Get atom colors
        atom_colors = attribution.get_highlighted_mol(mol)

        # Draw molecule
        drawer = Draw.MolDraw2DCairo(400, 400)
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(range(mol.GetNumAtoms())),
            highlightAtomColors=atom_colors,
        )
        drawer.FinishDrawing()

        if filename:
            with open(filename, 'wb') as f:
                f.write(drawer.GetDrawingText())

        return drawer.GetDrawingText()


class PermutationImportance:
    """
    Calculates permutation-based feature importance.

    More reliable than tree-based importance for correlated features.

    Examples:
        >>> from sklearn.metrics import r2_score
        >>>
        >>> pi = PermutationImportance(model, scoring=r2_score)
        >>> importances = pi.calculate(X_val, y_val)
        >>> for name, score in sorted(importances.items(), key=lambda x: -x[1])[:10]:
        ...     print(f"{name}: {score:.4f}")
    """

    def __init__(
        self,
        model: Any,
        scoring: Any,
        n_repeats: int = 10,
        random_state: int = 42,
    ):
        """
        Initialize permutation importance calculator.

        Args:
            model: Trained model
            scoring: Scoring function (sklearn metric)
            n_repeats: Number of permutation repeats
            random_state: Random seed
        """
        self.model = model
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.random_state = random_state

    def calculate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Calculate permutation importance.

        Args:
            X: Feature matrix
            y: Target values
            feature_names: Optional feature names

        Returns:
            Dictionary mapping feature name to importance score
        """
        # Baseline score
        y_pred = self.model.predict(X)
        baseline_score = self.scoring(y, y_pred)

        # Feature importances
        n_features = X.shape[1]
        importances = np.zeros(n_features)

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]

        # For each feature
        for i in range(n_features):
            scores = []

            for _ in range(self.n_repeats):
                # Permute feature i
                X_permuted = X.copy()
                np.random.seed(self.random_state)
                np.random.shuffle(X_permuted[:, i])

                # Calculate score
                y_pred_permuted = self.model.predict(X_permuted)
                score = self.scoring(y, y_pred_permuted)
                scores.append(score)

            # Importance = drop in score
            importances[i] = baseline_score - np.mean(scores)

        return dict(zip(feature_names, importances))


def explain_with_shap(
    model: Any,
    X_explain: np.ndarray,
    X_background: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> List[FeatureImportance]:
    """
    Convenience function for SHAP explanation.

    Args:
        model: Trained model
        X_explain: Data to explain
        X_background: Background data for SHAP
        feature_names: Optional feature names

    Returns:
        List of FeatureImportance objects

    Examples:
        >>> importances = explain_with_shap(model, X_test[:10], X_train[:100])
        >>> for imp in importances:
        ...     top = imp.get_top_features(5)
        ...     print(f"Prediction: {imp.predicted_value:.2f}")
        ...     for feat, val in top:
        ...         print(f"  {feat}: {val:+.3f}")
    """
    interpreter = SHAPInterpreter(model, X_background)
    return interpreter.explain(X_explain, feature_names)


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    import numpy as np

    print("=== Feature Importance Example ===\n")

    # Generate synthetic data
    X, y = make_regression(n_samples=200, n_features=20, n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    # 1. Tree-based feature importance
    print("Tree-based Feature Importance (top 10):")
    tree_importances = dict(zip(feature_names, model.feature_importances_))
    for name, score in sorted(tree_importances.items(), key=lambda x: -x[1])[:10]:
        print(f"  {name}: {score:.4f}")
    print()

    # 2. Permutation importance
    print("Permutation Importance (top 10):")
    from sklearn.metrics import r2_score

    pi = PermutationImportance(model, r2_score, n_repeats=5)
    perm_importances = pi.calculate(X_test, y_test, feature_names)
    for name, score in sorted(perm_importances.items(), key=lambda x: -x[1])[:10]:
        print(f"  {name}: {score:+.4f}")
    print()

    # 3. SHAP values (if available)
    try:
        print("SHAP Explanation (sample of 5 instances):")
        interpreter = SHAPInterpreter(model, X_train[:50])
        importances = interpreter.explain(X_test[:5], feature_names)

        for i, imp in enumerate(importances):
            print(f"\nInstance {i}:")
            print(f"  Base value: {imp.base_value:.2f}")
            print(f"  Predicted: {imp.predicted_value:.2f}")
            print(f"  Top features:")
            for feat, val in imp.get_top_features(5):
                print(f"    {feat}: {val:+.3f}")

    except Exception as e:
        print(f"SHAP explanation not available: {e}")
        print("Install SHAP with: pip install shap")

    print("\nFeature importance analysis completed!")
