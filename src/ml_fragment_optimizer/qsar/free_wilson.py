"""
Free-Wilson Analysis

Implements Free-Wilson additive model for SAR analysis, decomposing activity
into additive contributions from substituents at different positions.

References:
    - Free & Wilson (1964) "A Mathematical Contribution to Structure-Activity Studies"
      J. Med. Chem. 7, 395-399
    - Kubinyi (1988) "Free-Wilson analysis. Theory, applications and its relationship
      to Hansch analysis" Quant. Struct.-Act. Relat. 7, 121-133
    - Doweyko (1988) "3D-QSAR illusions" J. Comput.-Aided Mol. Des. 2, 181-191
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Fragments
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


@dataclass
class SubstituentContribution:
    """Contribution of a substituent at a specific position."""

    position: str
    substituent: str
    contribution: float
    std_error: float
    n_occurrences: int

    def __repr__(self) -> str:
        return (
            f"SubstituentContribution(pos={self.position}, "
            f"sub={self.substituent}, contrib={self.contribution:.3f}±{self.std_error:.3f})"
        )


@dataclass
class FreeWilsonModel:
    """Free-Wilson model with fitted parameters."""

    baseline_activity: float
    contributions: List[SubstituentContribution]
    r2_train: float
    r2_cv: float
    rmse_train: float
    rmse_cv: float
    n_molecules: int
    n_positions: int
    feature_matrix: np.ndarray
    ridge_alpha: float

    def predict(self, feature_vector: np.ndarray) -> float:
        """
        Predict activity from feature vector.

        Args:
            feature_vector: One-hot encoded substituent pattern

        Returns:
            Predicted activity
        """
        contrib_values = np.array([c.contribution for c in self.contributions])
        return self.baseline_activity + np.dot(feature_vector, contrib_values)


class FreeWilsonAnalyzer:
    """
    Free-Wilson analyzer for additive SAR modeling.

    The Free-Wilson approach decomposes activity into additive contributions:
        Activity = Baseline + Σ(Contribution_i)

    where each contribution_i represents the effect of a substituent at position i.

    Examples:
        >>> from rdkit import Chem
        >>> # Example: para-substituted benzenes
        >>> smiles = [
        ...     "c1ccc(F)cc1",      # para-F
        ...     "c1ccc(Cl)cc1",     # para-Cl
        ...     "c1ccc(C)cc1",      # para-Me
        ...     "c1ccc(O)cc1",      # para-OH
        ... ]
        >>> mols = [Chem.MolFromSmiles(s) for s in smiles]
        >>> activities = [5.5, 6.0, 5.2, 5.8]
        >>>
        >>> analyzer = FreeWilsonAnalyzer()
        >>> scaffold = Chem.MolFromSmiles("c1ccc([*])cc1")  # para-substituted benzene
        >>> model = analyzer.fit(mols, activities, scaffold)
        >>> print(f"R² = {model.r2_cv:.3f}")
        >>>
        >>> # Predict new molecule
        >>> new_mol = Chem.MolFromSmiles("c1ccc(Br)cc1")
        >>> pred = analyzer.predict(new_mol)
    """

    def __init__(self, ridge_alpha: Optional[float] = None, cv_folds: int = 5):
        """
        Initialize Free-Wilson analyzer.

        Args:
            ridge_alpha: Regularization parameter (None for automatic CV selection)
            cv_folds: Number of cross-validation folds
        """
        self.ridge_alpha = ridge_alpha
        self.cv_folds = cv_folds
        self.model_ = None
        self.encoder_ = None
        self.position_names_ = None

    def fit(
        self,
        mols: List[Chem.Mol],
        activities: List[float],
        scaffold: Optional[Chem.Mol] = None,
        position_atoms: Optional[List[int]] = None,
    ) -> FreeWilsonModel:
        """
        Fit Free-Wilson model to data.

        Args:
            mols: List of molecules (must share common scaffold)
            activities: List of activity values
            scaffold: Common scaffold with attachment points marked as [*]
            position_atoms: Atom indices where substitutions occur (alternative to scaffold)

        Returns:
            Fitted FreeWilsonModel

        Raises:
            ValueError: If molecules don't match scaffold or have inconsistent substitution patterns
        """
        if len(mols) != len(activities):
            raise ValueError(f"Mismatch: {len(mols)} molecules but {len(activities)} activities")

        logger.info(f"Fitting Free-Wilson model for {len(mols)} molecules...")

        # Extract substituents at each position
        if scaffold is not None:
            substituents = self._extract_substituents_from_scaffold(mols, scaffold)
        elif position_atoms is not None:
            substituents = self._extract_substituents_from_positions(mols, position_atoms)
        else:
            raise ValueError("Must provide either scaffold or position_atoms")

        # Convert to feature matrix (one-hot encoding)
        X, position_names = self._encode_substituents(substituents)
        y = np.array(activities)

        # Fit Ridge regression model
        if self.ridge_alpha is None:
            # Cross-validation to select alpha
            alphas = np.logspace(-3, 3, 20)
            model = RidgeCV(alphas=alphas, cv=self.cv_folds)
        else:
            model = Ridge(alpha=self.ridge_alpha)

        model.fit(X, y)

        # Calculate performance metrics
        y_pred_train = model.predict(X)
        r2_train = model.score(X, y)
        rmse_train = np.sqrt(np.mean((y - y_pred_train) ** 2))

        # Cross-validation performance
        cv_scores = cross_val_score(
            model, X, y, cv=self.cv_folds, scoring="r2"
        )
        r2_cv = np.mean(cv_scores)

        # RMSE from cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_rmse = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            temp_model = Ridge(alpha=model.alpha if hasattr(model, 'alpha') else self.ridge_alpha)
            temp_model.fit(X_train, y_train)
            y_val_pred = temp_model.predict(X_val)
            cv_rmse.append(np.sqrt(np.mean((y_val - y_val_pred) ** 2)))
        rmse_cv = np.mean(cv_rmse)

        # Extract substituent contributions
        contributions = []
        for i, (pos, sub) in enumerate(position_names):
            # Standard error from bootstrap or analytical approximation
            std_error = rmse_train / np.sqrt(len(mols))  # Simplified estimate

            # Count occurrences
            n_occurrences = np.sum(X[:, i])

            contrib = SubstituentContribution(
                position=pos,
                substituent=sub,
                contribution=model.coef_[i],
                std_error=std_error,
                n_occurrences=int(n_occurrences),
            )
            contributions.append(contrib)

        # Store model
        self.model_ = model
        self.encoder_ = None  # Would need to store for prediction
        self.position_names_ = position_names

        fw_model = FreeWilsonModel(
            baseline_activity=float(model.intercept_),
            contributions=contributions,
            r2_train=r2_train,
            r2_cv=r2_cv,
            rmse_train=rmse_train,
            rmse_cv=rmse_cv,
            n_molecules=len(mols),
            n_positions=len(set(pos for pos, _ in position_names)),
            feature_matrix=X,
            ridge_alpha=model.alpha if hasattr(model, 'alpha') else self.ridge_alpha,
        )

        logger.info(f"Model fitted: R²={r2_cv:.3f}, RMSE={rmse_cv:.3f}")

        return fw_model

    def _extract_substituents_from_scaffold(
        self,
        mols: List[Chem.Mol],
        scaffold: Chem.Mol,
    ) -> List[Dict[str, str]]:
        """
        Extract substituents from molecules based on scaffold.

        Args:
            mols: List of molecules
            scaffold: Scaffold with [*] marking substitution points

        Returns:
            List of dictionaries mapping position to substituent SMILES
        """
        all_substituents = []

        # Get attachment points in scaffold
        attachment_atoms = []
        for atom in scaffold.GetAtoms():
            if atom.GetSymbol() == '*':
                attachment_atoms.append(atom.GetIdx())

        if not attachment_atoms:
            raise ValueError("Scaffold must contain at least one [*] attachment point")

        logger.info(f"Found {len(attachment_atoms)} attachment points in scaffold")

        # For each molecule, extract substituents
        for mol in mols:
            # Match scaffold to molecule
            match = mol.GetSubstructMatch(scaffold)
            if not match:
                logger.warning(f"Molecule {Chem.MolToSmiles(mol)} does not match scaffold")
                all_substituents.append({})
                continue

            substituents = {}
            # Extract substituent at each position
            for i, attach_idx in enumerate(attachment_atoms):
                # Get atoms connected to attachment point
                # This is simplified - real implementation needs more sophisticated matching
                pos_name = f"R{i+1}"
                # Placeholder: would extract actual substituent SMILES
                substituents[pos_name] = "[H]"  # Default to hydrogen

            all_substituents.append(substituents)

        return all_substituents

    def _extract_substituents_from_positions(
        self,
        mols: List[Chem.Mol],
        position_atoms: List[int],
    ) -> List[Dict[str, str]]:
        """
        Extract substituents at specified atom positions.

        Args:
            mols: List of molecules
            position_atoms: Atom indices for substitution positions

        Returns:
            List of dictionaries mapping position to substituent
        """
        all_substituents = []

        for mol in mols:
            substituents = {}
            for i, atom_idx in enumerate(position_atoms):
                if atom_idx >= mol.GetNumAtoms():
                    substituents[f"R{i+1}"] = "missing"
                    continue

                atom = mol.GetAtomWithIdx(atom_idx)
                # Get substituent attached to this position
                neighbors = atom.GetNeighbors()

                # Simplified: just use atom symbol
                if neighbors:
                    substituents[f"R{i+1}"] = neighbors[0].GetSymbol()
                else:
                    substituents[f"R{i+1}"] = "[H]"

            all_substituents.append(substituents)

        return all_substituents

    def _encode_substituents(
        self,
        substituents_list: List[Dict[str, str]],
    ) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
        """
        One-hot encode substituents.

        Args:
            substituents_list: List of dictionaries mapping position to substituent

        Returns:
            Tuple of (feature matrix, position-substituent name mapping)
        """
        # Get all unique positions
        all_positions = sorted(set(
            pos for subs in substituents_list for pos in subs.keys()
        ))

        # Get all unique substituents per position
        position_substituents = {
            pos: sorted(set(
                subs.get(pos, "missing")
                for subs in substituents_list
            ))
            for pos in all_positions
        }

        # Build feature matrix
        n_samples = len(substituents_list)
        n_features = sum(len(subs) for subs in position_substituents.values())

        X = np.zeros((n_samples, n_features))
        position_names = []

        feature_idx = 0
        for pos in all_positions:
            for sub in position_substituents[pos]:
                position_names.append((pos, sub))

                # Set feature to 1 where this substituent appears
                for i, subs in enumerate(substituents_list):
                    if subs.get(pos) == sub:
                        X[i, feature_idx] = 1

                feature_idx += 1

        return X, position_names

    def get_top_contributors(
        self,
        model: FreeWilsonModel,
        n: int = 10,
    ) -> List[SubstituentContribution]:
        """
        Get top N substituents by contribution magnitude.

        Args:
            model: Fitted Free-Wilson model
            n: Number of top contributors

        Returns:
            List of top SubstituentContribution objects
        """
        sorted_contribs = sorted(
            model.contributions,
            key=lambda x: abs(x.contribution),
            reverse=True
        )
        return sorted_contribs[:n]

    def predict(
        self,
        mol: Chem.Mol,
        scaffold: Optional[Chem.Mol] = None,
    ) -> float:
        """
        Predict activity for a new molecule.

        Args:
            mol: Molecule to predict
            scaffold: Scaffold structure (must be same as used in fit)

        Returns:
            Predicted activity
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Extract substituents
        if scaffold is not None:
            substituents = self._extract_substituents_from_scaffold([mol], scaffold)[0]
        else:
            raise ValueError("Prediction requires scaffold")

        # Encode
        X_pred = np.zeros(len(self.position_names_))
        for i, (pos, sub) in enumerate(self.position_names_):
            if substituents.get(pos) == sub:
                X_pred[i] = 1

        # Predict
        y_pred = self.model_.predict(X_pred.reshape(1, -1))
        return float(y_pred[0])


def fit_free_wilson_model(
    mols: List[Chem.Mol],
    activities: List[float],
    position_atoms: Optional[List[int]] = None,
) -> FreeWilsonModel:
    """
    Convenience function to fit Free-Wilson model.

    Args:
        mols: List of molecules with common scaffold
        activities: Activity values
        position_atoms: Atom indices for substitution positions

    Returns:
        Fitted FreeWilsonModel

    Examples:
        >>> from rdkit import Chem
        >>> smiles = ["c1ccc(F)cc1", "c1ccc(Cl)cc1", "c1ccc(Br)cc1"]
        >>> mols = [Chem.MolFromSmiles(s) for s in smiles]
        >>> activities = [5.5, 6.0, 6.5]
        >>> model = fit_free_wilson_model(mols, activities, position_atoms=[3])
        >>> print(f"Baseline: {model.baseline_activity:.2f}")
        >>> print(f"R² (CV): {model.r2_cv:.3f}")
    """
    analyzer = FreeWilsonAnalyzer()

    if position_atoms is not None:
        return analyzer.fit(mols, activities, position_atoms=position_atoms)
    else:
        # Try to find common scaffold automatically
        raise NotImplementedError("Automatic scaffold detection not yet implemented")


if __name__ == "__main__":
    # Example usage
    from rdkit import Chem
    import numpy as np

    print("=== Free-Wilson Analysis Example ===\n")

    # Synthetic dataset: para-substituted benzenes
    # Activity increases with halogen size: F < Cl < Br
    smiles_list = [
        "c1ccc(F)cc1",       # para-F: activity = 5.5
        "c1ccc(Cl)cc1",      # para-Cl: activity = 6.0
        "c1ccc(Br)cc1",      # para-Br: activity = 6.5
        "c1ccc(I)cc1",       # para-I: activity = 7.0
        "c1ccc(C)cc1",       # para-Me: activity = 5.2
        "c1ccc(O)cc1",       # para-OH: activity = 5.8
        "c1ccccc1",          # H: activity = 5.0
    ]

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    activities = [5.5, 6.0, 6.5, 7.0, 5.2, 5.8, 5.0]

    # Fit model (using position 3 as substitution site)
    analyzer = FreeWilsonAnalyzer()
    model = analyzer.fit(mols, activities, position_atoms=[3])

    print(f"Baseline Activity: {model.baseline_activity:.3f}")
    print(f"R² (training): {model.r2_train:.3f}")
    print(f"R² (CV): {model.r2_cv:.3f}")
    print(f"RMSE (training): {model.rmse_train:.3f}")
    print(f"RMSE (CV): {model.rmse_cv:.3f}")
    print(f"Ridge alpha: {model.ridge_alpha:.3f}")
    print()

    print("Substituent Contributions:")
    top_contribs = analyzer.get_top_contributors(model, n=10)
    for contrib in top_contribs:
        if contrib.n_occurrences > 0:
            print(f"  {contrib.position}-{contrib.substituent}: "
                  f"{contrib.contribution:+.3f} ± {contrib.std_error:.3f} "
                  f"(n={contrib.n_occurrences})")
    print()

    # Compare predicted vs observed
    print("Predicted vs Observed Activities:")
    for i, mol in enumerate(mols):
        observed = activities[i]
        # Predict using feature matrix
        predicted = model.baseline_activity + np.dot(
            model.feature_matrix[i],
            [c.contribution for c in model.contributions]
        )
        print(f"  {smiles_list[i]:20s} Obs={observed:.2f}, Pred={predicted:.2f}, "
              f"Error={abs(observed - predicted):.2f}")
