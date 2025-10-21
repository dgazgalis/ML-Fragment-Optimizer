"""
QSAR model building with automated feature selection and validation.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from loguru import logger

from ml_fragment_optimizer.utils.featurizers import MolecularFeaturizer
from ml_fragment_optimizer.models.admet_predictor import ADMETPredictor


class QSARModelBuilder:
    """
    Automated QSAR model builder with feature selection and validation.
    """

    def __init__(
        self,
        property_name: str,
        model_type: str = "random_forest",
        feature_selection: str = "mutual_info",
        n_features: Optional[int] = None,
        cv_folds: int = 5,
    ):
        """
        Initialize QSAR model builder.

        Args:
            property_name: Name of property to model
            model_type: Type of ML model
            feature_selection: Method for feature selection ("mutual_info", "f_test", None)
            n_features: Number of features to select (None = all)
            cv_folds: Number of cross-validation folds
        """
        self.property_name = property_name
        self.model_type = model_type
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.cv_folds = cv_folds

        self.featurizer = MolecularFeaturizer(
            fingerprint_type="morgan",
            radius=2,
            n_bits=2048,
            include_descriptors=True
        )
        self.scaler = StandardScaler()
        self.selector = None
        self.model = None

    def build_and_validate(
        self,
        smiles: List[str],
        values: np.ndarray,
    ) -> Dict[str, float]:
        """
        Build QSAR model with cross-validation.

        Args:
            smiles: List of SMILES strings
            values: Property values

        Returns:
            Dictionary of CV metrics
        """
        logger.info(f"Building QSAR model for {self.property_name}")
        logger.info(f"Dataset size: {len(smiles)} molecules")

        # Featurize
        X = self.featurizer.featurize(smiles)
        logger.info(f"Initial features: {X.shape[1]}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Feature selection
        if self.feature_selection is not None:
            X_scaled = self._select_features(X_scaled, values)
            logger.info(f"Selected features: {X_scaled.shape[1]}")

        # Cross-validation
        metrics = self._cross_validate(X_scaled, values)

        # Train final model on all data
        self.model = ADMETPredictor(
            properties=[self.property_name],
            model_type=self.model_type,
        )
        self.model.fit(smiles, {self.property_name: values}, validation_split=0.0)

        return metrics

    def _select_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Select most informative features."""
        if self.n_features is None:
            self.n_features = min(500, X.shape[1])

        if self.feature_selection == "mutual_info":
            self.selector = SelectKBest(mutual_info_regression, k=self.n_features)
        elif self.feature_selection == "f_test":
            self.selector = SelectKBest(f_regression, k=self.n_features)
        else:
            raise ValueError(f"Unknown feature selection: {self.feature_selection}")

        X_selected = self.selector.fit_transform(X, y)
        return X_selected

    def _cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform cross-validation."""
        # Create model instance
        predictor = ADMETPredictor(
            properties=[self.property_name],
            model_type=self.model_type,
        )
        base_model = predictor._create_model()

        # Cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        cv_scores = {
            "r2": [],
            "neg_mae": [],
            "neg_rmse": [],
        }

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            base_model.fit(X_train, y_train)
            y_pred = base_model.predict(X_val)

            # Calculate metrics
            r2 = 1 - np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
            mae = np.mean(np.abs(y_val - y_pred))
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))

            cv_scores["r2"].append(r2)
            cv_scores["neg_mae"].append(-mae)
            cv_scores["neg_rmse"].append(-rmse)

        # Average metrics
        metrics = {
            "mean_r2": np.mean(cv_scores["r2"]),
            "std_r2": np.std(cv_scores["r2"]),
            "mean_mae": -np.mean(cv_scores["neg_mae"]),
            "std_mae": np.std(cv_scores["neg_mae"]),
            "mean_rmse": -np.mean(cv_scores["neg_rmse"]),
            "std_rmse": np.std(cv_scores["neg_rmse"]),
        }

        logger.info(f"CV R²: {metrics['mean_r2']:.3f} ± {metrics['std_r2']:.3f}")
        logger.info(f"CV MAE: {metrics['mean_mae']:.3f} ± {metrics['std_mae']:.3f}")
        logger.info(f"CV RMSE: {metrics['mean_rmse']:.3f} ± {metrics['std_rmse']:.3f}")

        return metrics

    def predict(self, smiles: List[str]) -> np.ndarray:
        """Predict property values."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")

        predictions = self.model.predict(smiles)
        return predictions[self.property_name]

    def save(self, path: Path) -> None:
        """Save QSAR model."""
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "featurizer": self.featurizer,
                "scaler": self.scaler,
                "selector": self.selector,
                "property_name": self.property_name,
            },
            path,
        )
        logger.info(f"Saved QSAR model to {path}")

    @classmethod
    def load(cls, path: Path) -> "QSARModelBuilder":
        """Load QSAR model."""
        import joblib
        data = joblib.load(path)
        builder = cls(property_name=data["property_name"])
        builder.model = data["model"]
        builder.featurizer = data["featurizer"]
        builder.scaler = data["scaler"]
        builder.selector = data["selector"]
        logger.info(f"Loaded QSAR model from {path}")
        return builder
