"""
Prospective validation framework for molecular property prediction.

Prospective validation mimics real-world deployment by testing models on
future/external data. This is the gold standard for assessing model utility.

Key Features:
-------------
- Temporal validation (train on past, test on future)
- External validation set handling
- Experimental validation tracking
- Hit rate analysis (top-K enrichment)
- Prospective vs retrospective performance gap analysis
- Model performance tracking over time

Literature:
-----------
- Sheridan (2013) "Time-Split Cross-Validation as a Method for Estimating the
  Goodness of Prospective Prediction" J Chem Inf Model 53(4):783-790
- Wallach & Heifets (2018) "Most Ligand-Based Classification Benchmarks Reward
  Memorization Rather than Generalization" J Chem Inf Model 58(5):916-932
- Krstajic et al. (2014) "Cross-validation pitfalls when selecting and assessing
  regression and classification models" J Cheminform 6:10

WARNING: Retrospective validation (random splits, cross-validation) often
severely overestimates model performance. Always perform prospective validation
before deployment.
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import warnings
import json


@dataclass
class ProspectiveResult:
    """Results from prospective validation."""
    test_date: Optional[str] = None
    n_predictions: int = 0
    n_experimental: int = 0
    hit_rate: Optional[float] = None
    enrichment: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    experimental_values: Optional[np.ndarray] = None


@dataclass
class ValidationCampaign:
    """Track a prospective validation campaign over time."""
    name: str
    model_version: str
    start_date: str
    results: List[ProspectiveResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemporalValidator:
    """
    Temporal validation for time-series molecular data.

    Simulates prospective validation by training on historical data and
    testing on future data.

    Example:
    --------
    >>> validator = TemporalValidator()
    >>> results = validator.validate(X, y, dates, train_until="2020-01-01")
    >>> print(f"Prospective R²: {results['r2']:.4f}")

    References:
    -----------
    Sheridan (2013) "Time-Split Cross-Validation as a Method for Estimating the
    Goodness of Prospective Prediction" J Chem Inf Model 53(4):783-790
    """

    def __init__(self, metric: str = "r2"):
        """
        Parameters:
        -----------
        metric : str
            Evaluation metric
        """
        self.metric = metric
        self.history = []

    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: np.ndarray,
        model: Any,
        train_until: str,
        test_until: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform temporal validation.

        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            Targets
        dates : np.ndarray
            Dates (as timestamps or strings)
        model : Any
            ML model with fit/predict interface
        train_until : str
            Train on data up to this date
        test_until : Optional[str]
            Test on data up to this date (if None, use all future data)

        Returns:
        --------
        results : Dict[str, Any]
            Validation results including metrics and predictions

        Example:
        --------
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor()
        >>> dates = np.array(["2018-01-01", "2019-01-01", "2020-01-01"])
        >>> results = validator.validate(X, y, dates, model, "2019-06-01")
        """
        # Convert dates to comparable format
        if isinstance(dates[0], str):
            dates_parsed = np.array([datetime.fromisoformat(d) for d in dates])
        else:
            dates_parsed = dates

        train_cutoff = datetime.fromisoformat(train_until) if isinstance(train_until, str) else train_until

        # Split data
        train_mask = dates_parsed <= train_cutoff
        test_mask = dates_parsed > train_cutoff

        if test_until is not None:
            test_cutoff = datetime.fromisoformat(test_until) if isinstance(test_until, str) else test_until
            test_mask = test_mask & (dates_parsed <= test_cutoff)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(X_test) == 0:
            warnings.warn("No test data found after train_until date")
            return {"error": "No test data"}

        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        from . import metrics
        results = {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_until": train_until,
            "test_until": test_until,
            "rmse": metrics.rmse(y_test, y_pred),
            "mae": metrics.mae(y_test, y_pred),
            "r2": metrics.r2_score(y_test, y_pred),
            "spearman": metrics.spearman_correlation(y_test, y_pred),
            "predictions": y_pred,
            "ground_truth": y_test,
            "test_dates": dates[test_mask]
        }

        self.history.append(results)
        return results

    def rolling_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: np.ndarray,
        model: Any,
        window_size: int = 365,
        step_size: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Perform rolling temporal validation.

        Train on sliding window of historical data, test on future data.

        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            Targets
        dates : np.ndarray
            Dates
        model : Any
            ML model
        window_size : int
            Training window size in days
        step_size : int
            Step size between validations in days

        Returns:
        --------
        results : List[Dict[str, Any]]
            Results for each validation window

        Example:
        --------
        >>> results = validator.rolling_validation(X, y, dates, model)
        >>> for r in results:
        ...     print(f"{r['train_until']}: R²={r['r2']:.4f}")
        """
        # Convert dates
        if isinstance(dates[0], str):
            dates_parsed = np.array([datetime.fromisoformat(d) for d in dates])
        else:
            dates_parsed = dates

        min_date = np.min(dates_parsed)
        max_date = np.max(dates_parsed)

        results = []
        current_date = min_date + timedelta(days=window_size)

        while current_date < max_date:
            train_start = current_date - timedelta(days=window_size)
            test_end = current_date + timedelta(days=step_size)

            # Train on window, test on next period
            train_mask = (dates_parsed >= train_start) & (dates_parsed < current_date)
            test_mask = (dates_parsed >= current_date) & (dates_parsed < test_end)

            if np.sum(train_mask) > 0 and np.sum(test_mask) > 0:
                X_train, y_train = X[train_mask], y[train_mask]
                X_test, y_test = X[test_mask], y[test_mask]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                from . import metrics
                result = {
                    "train_start": train_start.isoformat(),
                    "train_until": current_date.isoformat(),
                    "test_until": test_end.isoformat(),
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "r2": metrics.r2_score(y_test, y_pred),
                    "rmse": metrics.rmse(y_test, y_pred)
                }
                results.append(result)

            current_date += timedelta(days=step_size)

        return results


class ExperimentalValidator:
    """
    Track experimental validation of model predictions.

    Use this to compare model predictions against wet-lab results.

    Example:
    --------
    >>> validator = ExperimentalValidator()
    >>> validator.add_prediction(smiles="CCO", predicted_ic50=5.2, confidence=0.8)
    >>> validator.add_experimental(smiles="CCO", experimental_ic50=4.9)
    >>> summary = validator.get_summary()
    """

    def __init__(self):
        self.predictions = {}
        self.experimental = {}

    def add_prediction(
        self,
        compound_id: str,
        predicted_value: float,
        confidence: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add a model prediction.

        Parameters:
        -----------
        compound_id : str
            Unique compound identifier
        predicted_value : float
            Predicted property value
        confidence : Optional[float]
            Model confidence (0-1)
        metadata : Optional[Dict]
            Additional metadata
        """
        self.predictions[compound_id] = {
            "predicted_value": predicted_value,
            "confidence": confidence,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }

    def add_experimental(
        self,
        compound_id: str,
        experimental_value: float,
        metadata: Optional[Dict] = None
    ):
        """
        Add experimental validation result.

        Parameters:
        -----------
        compound_id : str
            Unique compound identifier
        experimental_value : float
            Experimentally measured value
        metadata : Optional[Dict]
            Additional metadata (e.g., assay conditions)
        """
        self.experimental[compound_id] = {
            "experimental_value": experimental_value,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of experimental validation.

        Returns:
        --------
        summary : Dict[str, Any]
            Summary statistics including metrics and hit rates

        Example:
        --------
        >>> summary = validator.get_summary()
        >>> print(f"Validated: {summary['n_validated']}/{summary['n_predictions']}")
        >>> print(f"RMSE: {summary['rmse']:.4f}")
        """
        # Find compounds with both predictions and experimental values
        validated_ids = set(self.predictions.keys()) & set(self.experimental.keys())

        if len(validated_ids) == 0:
            return {
                "n_predictions": len(self.predictions),
                "n_experimental": len(self.experimental),
                "n_validated": 0,
                "validation_rate": 0.0
            }

        # Extract matched values
        predicted = np.array([self.predictions[cid]["predicted_value"] for cid in validated_ids])
        experimental = np.array([self.experimental[cid]["experimental_value"] for cid in validated_ids])
        confidences = np.array([
            self.predictions[cid]["confidence"] if self.predictions[cid]["confidence"] is not None else 1.0
            for cid in validated_ids
        ])

        # Calculate metrics
        from . import metrics
        summary = {
            "n_predictions": len(self.predictions),
            "n_experimental": len(self.experimental),
            "n_validated": len(validated_ids),
            "validation_rate": len(validated_ids) / len(self.predictions),
            "rmse": metrics.rmse(experimental, predicted),
            "mae": metrics.mae(experimental, predicted),
            "r2": metrics.r2_score(experimental, predicted),
            "spearman": metrics.spearman_correlation(experimental, predicted),
            "mean_confidence": np.mean(confidences),
            "confidence_correlation": np.corrcoef(confidences, np.abs(experimental - predicted))[0, 1]
        }

        return summary

    def export_to_json(self, filepath: str):
        """Export validation data to JSON file."""
        data = {
            "predictions": self.predictions,
            "experimental": self.experimental,
            "summary": self.get_summary()
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def import_from_json(self, filepath: str):
        """Import validation data from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.predictions = data["predictions"]
        self.experimental = data["experimental"]


def calculate_hit_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k: int,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate hit rate for top-K predictions.

    Hit rate = fraction of top-K predictions that are true hits.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    top_k : int
        Number of top predictions to test
    threshold : Optional[float]
        Threshold for defining a "hit" (if None, use binary labels)

    Returns:
    --------
    stats : Dict[str, float]
        Hit rate statistics

    Example:
    --------
    >>> y_true = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    >>> y_pred = np.array([0.9, 0.1, 0.8, 0.3, 0.7, 0.2, 0.4, 0.6])
    >>> stats = calculate_hit_rate(y_true, y_pred, top_k=3)
    >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
    """
    # Get indices of top-K predictions
    top_k_indices = np.argsort(y_pred)[-top_k:]

    if threshold is not None:
        # Define hits based on threshold
        is_hit = y_true >= threshold
    else:
        # Assume binary labels
        is_hit = y_true == 1

    # Count hits in top-K
    n_hits_in_top_k = np.sum(is_hit[top_k_indices])
    n_total_hits = np.sum(is_hit)

    # Random baseline
    random_hit_rate = n_total_hits / len(y_true)
    expected_hits = random_hit_rate * top_k

    return {
        "hit_rate": n_hits_in_top_k / top_k,
        "n_hits": n_hits_in_top_k,
        "n_tested": top_k,
        "enrichment": (n_hits_in_top_k / top_k) / random_hit_rate if random_hit_rate > 0 else 0,
        "expected_hits_random": expected_hits,
        "improvement_over_random": n_hits_in_top_k - expected_hits
    }


def compare_retrospective_prospective(
    retrospective_metrics: Dict[str, float],
    prospective_metrics: Dict[str, float]
) -> Dict[str, Any]:
    """
    Compare retrospective and prospective validation results.

    Quantify the performance gap between internal validation (CV, random splits)
    and prospective validation.

    Parameters:
    -----------
    retrospective_metrics : Dict[str, float]
        Metrics from retrospective validation
    prospective_metrics : Dict[str, float]
        Metrics from prospective validation

    Returns:
    --------
    comparison : Dict[str, Any]
        Comparison statistics

    Example:
    --------
    >>> retro = {"r2": 0.85, "rmse": 0.5}
    >>> prosp = {"r2": 0.72, "rmse": 0.7}
    >>> comp = compare_retrospective_prospective(retro, prosp)
    >>> print(f"R² drop: {comp['r2']['absolute_drop']:.3f}")

    Notes:
    ------
    Large gaps indicate overfitting or data leakage. Common causes:
    - Molecular similarity between train/test
    - Temporal trends not captured in cross-validation
    - Assay differences between datasets
    """
    comparison = {}

    for metric in retrospective_metrics:
        if metric in prospective_metrics:
            retro_val = retrospective_metrics[metric]
            prosp_val = prospective_metrics[metric]

            # For metrics where higher is better (R², accuracy)
            if metric in ["r2", "accuracy", "f1", "roc_auc", "spearman", "kendall"]:
                drop = retro_val - prosp_val
                relative_drop = drop / retro_val if retro_val != 0 else 0
            # For metrics where lower is better (RMSE, MAE)
            else:
                drop = prosp_val - retro_val
                relative_drop = drop / retro_val if retro_val != 0 else 0

            comparison[metric] = {
                "retrospective": retro_val,
                "prospective": prosp_val,
                "absolute_drop": drop,
                "relative_drop": relative_drop,
                "percentage_drop": relative_drop * 100
            }

    return comparison


def track_model_performance(
    validation_results: List[Dict[str, Any]],
    metric: str = "r2"
) -> Dict[str, Any]:
    """
    Track model performance over time.

    Detect performance degradation, concept drift, or improvement.

    Parameters:
    -----------
    validation_results : List[Dict[str, Any]]
        List of validation results over time
    metric : str
        Metric to track

    Returns:
    --------
    analysis : Dict[str, Any]
        Performance tracking analysis

    Example:
    --------
    >>> results = [
    ...     {"date": "2020-01", "r2": 0.85},
    ...     {"date": "2020-02", "r2": 0.83},
    ...     {"date": "2020-03", "r2": 0.80}
    ... ]
    >>> analysis = track_model_performance(results)
    >>> print(f"Trend: {analysis['trend']}")
    """
    if len(validation_results) < 2:
        return {"error": "Need at least 2 validation results"}

    # Extract metric values
    values = np.array([r[metric] for r in validation_results])
    dates = [r.get("date", i) for i, r in enumerate(validation_results)]

    # Calculate trend
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        range(len(values)), values
    )

    # Detect significant changes
    mean_value = np.mean(values)
    std_value = np.std(values)

    analysis = {
        "metric": metric,
        "n_timepoints": len(values),
        "mean": mean_value,
        "std": std_value,
        "min": np.min(values),
        "max": np.max(values),
        "trend_slope": slope,
        "trend_pvalue": p_value,
        "trend_significant": p_value < 0.05,
        "trend_direction": "improving" if slope > 0 else "degrading" if slope < 0 else "stable",
        "values": values.tolist(),
        "dates": dates
    }

    # Detect outliers (>2 std from mean)
    outliers = np.abs(values - mean_value) > 2 * std_value
    if np.any(outliers):
        analysis["outliers"] = [
            {"date": dates[i], "value": values[i]}
            for i in np.where(outliers)[0]
        ]

    return analysis


class CampaignManager:
    """
    Manage multiple prospective validation campaigns.

    Example:
    --------
    >>> manager = CampaignManager()
    >>> campaign = manager.create_campaign("Campaign 1", "model_v1")
    >>> campaign.add_result(result)
    >>> manager.save("campaigns.json")
    """

    def __init__(self):
        self.campaigns: Dict[str, ValidationCampaign] = {}

    def create_campaign(
        self,
        name: str,
        model_version: str,
        metadata: Optional[Dict] = None
    ) -> ValidationCampaign:
        """Create a new validation campaign."""
        campaign = ValidationCampaign(
            name=name,
            model_version=model_version,
            start_date=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        self.campaigns[name] = campaign
        return campaign

    def add_result(
        self,
        campaign_name: str,
        result: ProspectiveResult
    ):
        """Add a result to a campaign."""
        if campaign_name not in self.campaigns:
            raise ValueError(f"Campaign {campaign_name} not found")
        self.campaigns[campaign_name].results.append(result)

    def get_campaign_summary(self, campaign_name: str) -> Dict[str, Any]:
        """Get summary statistics for a campaign."""
        if campaign_name not in self.campaigns:
            raise ValueError(f"Campaign {campaign_name} not found")

        campaign = self.campaigns[campaign_name]
        results = campaign.results

        if len(results) == 0:
            return {"error": "No results yet"}

        return {
            "name": campaign.name,
            "model_version": campaign.model_version,
            "start_date": campaign.start_date,
            "n_results": len(results),
            "total_predictions": sum(r.n_predictions for r in results),
            "total_experimental": sum(r.n_experimental for r in results),
            "average_hit_rate": np.mean([r.hit_rate for r in results if r.hit_rate is not None]),
            "metadata": campaign.metadata
        }

    def save(self, filepath: str):
        """Save campaigns to JSON file."""
        data = {
            name: {
                "name": campaign.name,
                "model_version": campaign.model_version,
                "start_date": campaign.start_date,
                "metadata": campaign.metadata,
                "results": [
                    {
                        "test_date": r.test_date,
                        "n_predictions": r.n_predictions,
                        "n_experimental": r.n_experimental,
                        "hit_rate": r.hit_rate,
                        "enrichment": r.enrichment,
                        "metrics": r.metrics
                    }
                    for r in campaign.results
                ]
            }
            for name, campaign in self.campaigns.items()
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


# Example usage
if __name__ == "__main__":
    print("Prospective Validation Module - Example Usage")
    print("=" * 60)

    # Generate synthetic temporal data
    np.random.seed(42)
    n_samples = 1000

    dates = [
        (datetime(2018, 1, 1) + timedelta(days=i)).isoformat()
        for i in range(n_samples)
    ]
    X = np.random.randn(n_samples, 10)
    y = np.sum(X[:, :3], axis=1) + np.random.randn(n_samples) * 0.5

    print("\n1. Temporal Validation:")
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    validator = TemporalValidator()
    result = validator.validate(X, y, np.array(dates), model, train_until="2020-07-01")
    print(f"   Train size: {result['train_size']}")
    print(f"   Test size: {result['test_size']}")
    print(f"   Prospective R²: {result['r2']:.4f}")
    print(f"   Prospective RMSE: {result['rmse']:.4f}")

    print("\n2. Rolling Validation:")
    results = validator.rolling_validation(X, y, np.array(dates), model, window_size=300, step_size=60)
    print(f"   Number of windows: {len(results)}")
    r2_values = [r['r2'] for r in results]
    print(f"   Average R²: {np.mean(r2_values):.4f}")
    print(f"   R² std: {np.std(r2_values):.4f}")

    print("\n3. Experimental Validation Tracking:")
    exp_validator = ExperimentalValidator()
    exp_validator.add_prediction("compound_1", predicted_value=5.2, confidence=0.85)
    exp_validator.add_prediction("compound_2", predicted_value=6.1, confidence=0.92)
    exp_validator.add_experimental("compound_1", experimental_value=4.9)
    summary = exp_validator.get_summary()
    print(f"   Predictions: {summary['n_predictions']}")
    print(f"   Validated: {summary['n_validated']}")
    print(f"   Validation rate: {summary['validation_rate']:.2%}")

    print("\n4. Hit Rate Analysis:")
    y_true = np.random.binomial(1, 0.2, 100)
    y_pred = np.random.rand(100)
    # Make top predictions more likely to be hits (simulate good model)
    y_pred[y_true == 1] += 0.3
    stats = calculate_hit_rate(y_true, y_pred, top_k=10)
    print(f"   Hit rate (top 10): {stats['hit_rate']:.2%}")
    print(f"   Enrichment: {stats['enrichment']:.2f}x")

    print("\n5. Retrospective vs Prospective Comparison:")
    retro = {"r2": 0.85, "rmse": 0.45}
    prosp = {"r2": 0.72, "rmse": 0.63}
    comparison = compare_retrospective_prospective(retro, prosp)
    print(f"   R² drop: {comparison['r2']['absolute_drop']:.3f} ({comparison['r2']['percentage_drop']:.1f}%)")
    print(f"   RMSE increase: {comparison['rmse']['absolute_drop']:.3f}")

    print("\n" + "=" * 60)
