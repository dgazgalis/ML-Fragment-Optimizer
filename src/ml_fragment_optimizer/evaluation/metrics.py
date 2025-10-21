"""
Comprehensive metrics library for molecular property prediction.

Provides regression, classification, ranking, and uncertainty metrics for
evaluating ML models in drug discovery.

Key Features:
-------------
- Regression metrics: RMSE, MAE, R², MAPE, Huber loss
- Classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Ranking metrics: Spearman's ρ, Kendall's τ, Top-K accuracy
- Multi-task metrics: average, weighted, per-task
- Uncertainty metrics: negative log-likelihood, calibration error
- Chemical space coverage metrics

Literature:
-----------
- Tropsha (2010) "Best Practices for QSAR Model Development, Validation, and
  Exploitation" Mol Inf 29(6-7):476-488
- Sheridan et al. (2004) "Protocols for bridging the peptide to nonpeptide
  gap in topological similarity searches" J Chem Inf Comput Sci 44(4):1294-1300
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from scipy import stats
from dataclasses import dataclass
import warnings


@dataclass
class RegressionMetrics:
    """Container for regression metrics."""
    rmse: float
    mae: float
    r2: float
    spearman: float
    kendall: float
    mape: Optional[float] = None
    huber: Optional[float] = None


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None


# ============================================================================
# Regression Metrics
# ============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns:
    --------
    rmse : float
        Root mean squared error (same units as y)

    Example:
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.1, 2.9])
    >>> print(f"RMSE: {rmse(y_true, y_pred):.4f}")
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    More robust to outliers than RMSE.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns:
    --------
    mae : float
        Mean absolute error

    Example:
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.1, 2.9])
    >>> print(f"MAE: {mae(y_true, y_pred):.4f}")
    """
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² (coefficient of determination).

    Measures fraction of variance explained by model.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns:
    --------
    r2 : float
        R² score (-∞ to 1, where 1 is perfect)

    Example:
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.1, 2.9])
    >>> print(f"R²: {r2_score(y_true, y_pred):.4f}")

    Notes:
    ------
    - R² = 1: perfect predictions
    - R² = 0: predictions as good as mean baseline
    - R² < 0: predictions worse than mean baseline
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Mean Absolute Percentage Error.

    Scale-independent metric, but sensitive to values near zero.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    epsilon : float
        Small constant to avoid division by zero

    Returns:
    --------
    mape : float
        Mean absolute percentage error (0 to 100+)

    Example:
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 2.1, 2.9])
    >>> print(f"MAPE: {mape(y_true, y_pred):.2f}%")

    WARNING: MAPE is undefined when y_true contains zeros and is biased when
    y_true contains small values.
    """
    if np.any(np.abs(y_true) < epsilon):
        warnings.warn("y_true contains values near zero, MAPE may be unreliable")
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def huber_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta: float = 1.0
) -> float:
    """
    Huber loss (robust to outliers).

    Behaves like MSE for small errors, MAE for large errors.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    delta : float
        Threshold for switching from quadratic to linear

    Returns:
    --------
    loss : float
        Huber loss

    Example:
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0, 10.0])
    >>> y_pred = np.array([1.1, 2.1, 2.9, 5.0])
    >>> print(f"Huber: {huber_loss(y_true, y_pred):.4f}")

    References:
    -----------
    Huber (1964) "Robust Estimation of a Location Parameter"
    Ann Math Statist 35(1):73-101
    """
    error = y_true - y_pred
    abs_error = np.abs(error)

    quadratic = 0.5 * error ** 2
    linear = delta * abs_error - 0.5 * delta ** 2

    return np.mean(np.where(abs_error <= delta, quadratic, linear))


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman's rank correlation coefficient.

    Measures monotonic relationship (non-linear correlations).

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns:
    --------
    rho : float
        Spearman correlation (-1 to 1)

    Example:
    --------
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1.1, 1.9, 3.1, 3.8, 5.2])
    >>> print(f"Spearman ρ: {spearman_correlation(y_true, y_pred):.4f}")

    Notes:
    ------
    Spearman correlation is useful when:
    - Relationship is monotonic but non-linear
    - Data contains outliers
    - Only rank ordering matters (e.g., virtual screening)
    """
    return stats.spearmanr(y_true, y_pred)[0]


def kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Kendall's tau rank correlation coefficient.

    Measures ordinal association (concordant vs discordant pairs).

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns:
    --------
    tau : float
        Kendall's tau (-1 to 1)

    Example:
    --------
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1.1, 1.9, 3.1, 3.8, 5.2])
    >>> print(f"Kendall τ: {kendall_tau(y_true, y_pred):.4f}")

    Notes:
    ------
    Kendall's tau is more robust to ties than Spearman's rho and has a
    clearer interpretation (fraction of concordant pairs minus discordant).
    """
    return stats.kendalltau(y_true, y_pred)[0]


# ============================================================================
# Classification Metrics
# ============================================================================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Classification accuracy (fraction correct).

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels

    Returns:
    --------
    acc : float
        Accuracy (0 to 1)

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> print(f"Accuracy: {accuracy(y_true, y_pred):.2%}")
    """
    return np.mean(y_true == y_pred)


def precision(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Precision (positive predictive value).

    Of all predicted positives, what fraction are truly positive?

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    pos_label : int
        Label to consider as positive class

    Returns:
    --------
    prec : float
        Precision (0 to 1)

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 1, 1])
    >>> print(f"Precision: {precision(y_true, y_pred):.2%}")
    """
    true_pos = np.sum((y_true == pos_label) & (y_pred == pos_label))
    pred_pos = np.sum(y_pred == pos_label)
    return true_pos / pred_pos if pred_pos > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Recall (sensitivity, true positive rate).

    Of all true positives, what fraction did we predict correctly?

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    pos_label : int
        Label to consider as positive class

    Returns:
    --------
    rec : float
        Recall (0 to 1)

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> print(f"Recall: {recall(y_true, y_pred):.2%}")
    """
    true_pos = np.sum((y_true == pos_label) & (y_pred == pos_label))
    actual_pos = np.sum(y_true == pos_label)
    return true_pos / actual_pos if actual_pos > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    F1 score (harmonic mean of precision and recall).

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    pos_label : int
        Label to consider as positive class

    Returns:
    --------
    f1 : float
        F1 score (0 to 1)

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 1, 1])
    >>> print(f"F1: {f1_score(y_true, y_pred):.4f}")
    """
    prec = precision(y_true, y_pred, pos_label)
    rec = recall(y_true, y_pred, pos_label)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Area Under the ROC Curve.

    Measures classifier's ability to discriminate between classes.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels
    y_score : np.ndarray
        Predicted scores or probabilities

    Returns:
    --------
    auc : float
        ROC-AUC (0.5 to 1.0, where 0.5 is random)

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_score = np.array([0.1, 0.9, 0.8, 0.2])
    >>> print(f"ROC-AUC: {roc_auc(y_true, y_score):.4f}")

    Notes:
    ------
    ROC-AUC is equivalent to the probability that a randomly chosen positive
    instance is ranked higher than a randomly chosen negative instance.
    """
    # Simple implementation using Mann-Whitney U statistic
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5

    # Count pairs where positive score > negative score
    n_correct = np.sum(pos_scores[:, None] > neg_scores[None, :])
    n_ties = np.sum(pos_scores[:, None] == neg_scores[None, :])

    return (n_correct + 0.5 * n_ties) / (len(pos_scores) * len(neg_scores))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Area Under the Precision-Recall Curve.

    More informative than ROC-AUC for imbalanced datasets.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels
    y_score : np.ndarray
        Predicted scores or probabilities

    Returns:
    --------
    auc : float
        PR-AUC (0 to 1)

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_score = np.array([0.1, 0.9, 0.8, 0.2])
    >>> print(f"PR-AUC: {pr_auc(y_true, y_score):.4f}")

    References:
    -----------
    Davis & Goadrich (2006) "The relationship between Precision-Recall and
    ROC curves" ICML
    """
    # Sort by score (descending)
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # Calculate precision at each threshold
    true_pos_cumsum = np.cumsum(y_true_sorted)
    total_pos = np.sum(y_true)

    precisions = true_pos_cumsum / np.arange(1, len(y_true) + 1)
    recalls = true_pos_cumsum / total_pos

    # Trapezoidal rule for AUC
    return np.trapz(precisions, recalls)


def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Matthews Correlation Coefficient.

    Balanced metric for binary classification, even with imbalanced classes.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels

    Returns:
    --------
    mcc : float
        Matthews correlation coefficient (-1 to 1, where 0 is random)

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> print(f"MCC: {matthews_corrcoef(y_true, y_pred):.4f}")

    References:
    -----------
    Matthews (1975) "Comparison of the predicted and observed secondary
    structure of T4 phage lysozyme" Biochim Biophys Acta 405(2):442-451
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / denominator if denominator > 0 else 0.0


# ============================================================================
# Ranking Metrics
# ============================================================================

def top_k_accuracy(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int,
    threshold: Optional[float] = None
) -> float:
    """
    Top-K accuracy for ranking tasks.

    What fraction of top-K predictions are true positives?

    Parameters:
    -----------
    y_true : np.ndarray
        True labels (1 for active, 0 for inactive)
    y_score : np.ndarray
        Predicted scores
    k : int
        Number of top predictions to consider
    threshold : Optional[float]
        Activity threshold for defining "active"

    Returns:
    --------
    acc : float
        Fraction of top-K that are active (0 to 1)

    Example:
    --------
    >>> y_true = np.array([0, 1, 0, 1, 1])
    >>> y_score = np.array([0.1, 0.9, 0.3, 0.8, 0.7])
    >>> print(f"Top-3 accuracy: {top_k_accuracy(y_true, y_score, k=3):.2%}")

    Notes:
    ------
    Useful for virtual screening where you only test top predictions.
    """
    # Get indices of top k scores
    top_k_indices = np.argsort(y_score)[-k:]

    if threshold is not None:
        # Count how many top-k have true values above threshold
        return np.mean(y_true[top_k_indices] >= threshold)
    else:
        # Count how many top-k are labeled as positive
        return np.mean(y_true[top_k_indices] == 1)


def enrichment_factor(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fraction: float = 0.01
) -> float:
    """
    Enrichment factor at given fraction of dataset.

    How much better is the model than random selection?

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels
    y_score : np.ndarray
        Predicted scores
    fraction : float
        Fraction of dataset to consider (e.g., 0.01 = top 1%)

    Returns:
    --------
    ef : float
        Enrichment factor (1.0 = random, >1.0 = better than random)

    Example:
    --------
    >>> y_true = np.array([0, 0, 0, 1, 1])
    >>> y_score = np.array([0.1, 0.2, 0.3, 0.9, 0.8])
    >>> print(f"EF(1%): {enrichment_factor(y_true, y_score, 0.2):.2f}")

    References:
    -----------
    Sheridan et al. (2004) "Protocols for bridging the peptide to nonpeptide
    gap in topological similarity searches" J Chem Inf Comput Sci 44(4):1294-1300
    """
    n_total = len(y_true)
    n_actives = np.sum(y_true)
    n_selected = max(1, int(n_total * fraction))

    # Get top fraction
    top_indices = np.argsort(y_score)[-n_selected:]
    n_actives_found = np.sum(y_true[top_indices])

    # Calculate enrichment
    hit_rate = n_actives_found / n_selected
    random_hit_rate = n_actives / n_total

    return hit_rate / random_hit_rate if random_hit_rate > 0 else 0.0


def bedroc_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    alpha: float = 20.0
) -> float:
    """
    Boltzmann-Enhanced Discrimination of ROC (BEDROC).

    Emphasizes early retrieval (finding actives in top ranks).

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels
    y_score : np.ndarray
        Predicted scores
    alpha : float
        Early retrieval emphasis parameter (default: 20.0)

    Returns:
    --------
    bedroc : float
        BEDROC score (0 to 1)

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_score = np.array([0.1, 0.9, 0.8, 0.2])
    >>> print(f"BEDROC: {bedroc_score(y_true, y_score):.4f}")

    References:
    -----------
    Truchon & Bayly (2007) "Evaluating virtual screening methods: good and
    bad metrics for the early recognition problem" J Chem Inf Model 47(2):488-508
    """
    # Sort by score (descending)
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]

    n = len(y_true)
    n_actives = np.sum(y_true)

    if n_actives == 0 or n_actives == n:
        return 0.0

    # Calculate exponentially weighted sum
    exp_weights = np.exp(-alpha * np.arange(n) / n)
    weighted_sum = np.sum(y_true_sorted * exp_weights)

    # Normalization factors
    max_sum = np.sum(exp_weights[:n_actives])
    min_sum = np.sum(exp_weights[-n_actives:])
    random_sum = n_actives * np.sum(exp_weights) / n

    # BEDROC formula
    if max_sum == min_sum:
        return 0.0

    bedroc = (weighted_sum - min_sum) / (max_sum - min_sum)
    rie = (random_sum - min_sum) / (max_sum - min_sum)  # Random performance

    # Normalize to [0, 1]
    return (bedroc - rie) / (1 - rie) if rie < 1 else 0.0


# ============================================================================
# Uncertainty Metrics
# ============================================================================

def negative_log_likelihood(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray
) -> float:
    """
    Negative log-likelihood for regression with uncertainty.

    Penalizes both prediction error and poor uncertainty estimates.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred_mean : np.ndarray
        Predicted means
    y_pred_std : np.ndarray
        Predicted standard deviations

    Returns:
    --------
    nll : float
        Negative log-likelihood (lower is better)

    Example:
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred_mean = np.array([1.1, 2.1, 2.9])
    >>> y_pred_std = np.array([0.5, 0.5, 0.5])
    >>> print(f"NLL: {negative_log_likelihood(y_true, y_pred_mean, y_pred_std):.4f}")

    Notes:
    ------
    NLL rewards both accurate predictions and calibrated uncertainties.
    Overconfident predictions (small std) are heavily penalized if wrong.
    """
    from scipy.stats import norm

    # Avoid numerical issues
    y_pred_std = np.maximum(y_pred_std, 1e-6)

    # Negative log-likelihood under Gaussian assumption
    nll = -np.mean(norm.logpdf(y_true, loc=y_pred_mean, scale=y_pred_std))

    return nll


def uncertainty_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray
) -> float:
    """
    Correlation between prediction error and uncertainty.

    Well-calibrated models should have high uncertainty for large errors.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    y_std : np.ndarray
        Predicted uncertainties

    Returns:
    --------
    corr : float
        Spearman correlation between |error| and uncertainty

    Example:
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y_pred = np.array([1.1, 2.5, 2.9, 3.8])
    >>> y_std = np.array([0.1, 0.8, 0.2, 0.3])
    >>> print(f"Error-uncertainty correlation: {uncertainty_correlation(y_true, y_pred, y_std):.4f}")
    """
    errors = np.abs(y_true - y_pred)
    return stats.spearmanr(errors, y_std)[0]


# ============================================================================
# Multi-task Metrics
# ============================================================================

def multitask_metrics(
    y_true_list: List[np.ndarray],
    y_pred_list: List[np.ndarray],
    task_names: Optional[List[str]] = None,
    aggregation: str = "mean"
) -> Dict[str, Any]:
    """
    Calculate metrics for multi-task learning.

    Parameters:
    -----------
    y_true_list : List[np.ndarray]
        True values for each task
    y_pred_list : List[np.ndarray]
        Predicted values for each task
    task_names : Optional[List[str]]
        Names of tasks
    aggregation : str
        How to aggregate across tasks: "mean", "median", "weighted"

    Returns:
    --------
    metrics : Dict[str, Any]
        Per-task and aggregated metrics

    Example:
    --------
    >>> y_true_list = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    >>> y_pred_list = [np.array([1.1, 2.1, 2.9]), np.array([4.1, 5.2, 5.8])]
    >>> metrics = multitask_metrics(y_true_list, y_pred_list)
    >>> print(metrics)
    """
    n_tasks = len(y_true_list)

    if task_names is None:
        task_names = [f"task_{i}" for i in range(n_tasks)]

    per_task_metrics = {}
    rmse_values = []
    mae_values = []
    r2_values = []

    for i, (y_true, y_pred, name) in enumerate(zip(y_true_list, y_pred_list, task_names)):
        task_rmse = rmse(y_true, y_pred)
        task_mae = mae(y_true, y_pred)
        task_r2 = r2_score(y_true, y_pred)

        per_task_metrics[name] = {
            "rmse": task_rmse,
            "mae": task_mae,
            "r2": task_r2
        }

        rmse_values.append(task_rmse)
        mae_values.append(task_mae)
        r2_values.append(task_r2)

    # Aggregate
    if aggregation == "mean":
        agg_rmse = np.mean(rmse_values)
        agg_mae = np.mean(mae_values)
        agg_r2 = np.mean(r2_values)
    elif aggregation == "median":
        agg_rmse = np.median(rmse_values)
        agg_mae = np.median(mae_values)
        agg_r2 = np.median(r2_values)
    elif aggregation == "weighted":
        # Weight by number of samples
        weights = [len(y) for y in y_true_list]
        total = sum(weights)
        agg_rmse = sum(r * w for r, w in zip(rmse_values, weights)) / total
        agg_mae = sum(m * w for m, w in zip(mae_values, weights)) / total
        agg_r2 = sum(r * w for r, w in zip(r2_values, weights)) / total
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    return {
        "per_task": per_task_metrics,
        "aggregated": {
            "rmse": agg_rmse,
            "mae": agg_mae,
            "r2": agg_r2
        },
        "aggregation_method": aggregation
    }


def calculate_all_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> RegressionMetrics:
    """
    Calculate all regression metrics at once.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns:
    --------
    metrics : RegressionMetrics
        All regression metrics

    Example:
    --------
    >>> y_true = np.random.randn(100)
    >>> y_pred = y_true + np.random.randn(100) * 0.1
    >>> metrics = calculate_all_regression_metrics(y_true, y_pred)
    >>> print(f"R²: {metrics.r2:.4f}, RMSE: {metrics.rmse:.4f}")
    """
    return RegressionMetrics(
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        r2=r2_score(y_true, y_pred),
        spearman=spearman_correlation(y_true, y_pred),
        kendall=kendall_tau(y_true, y_pred),
        mape=mape(y_true, y_pred),
        huber=huber_loss(y_true, y_pred)
    )


def calculate_all_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None
) -> ClassificationMetrics:
    """
    Calculate all classification metrics at once.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_score : Optional[np.ndarray]
        Predicted scores/probabilities (for AUC)

    Returns:
    --------
    metrics : ClassificationMetrics
        All classification metrics

    Example:
    --------
    >>> y_true = np.random.binomial(1, 0.5, 100)
    >>> y_pred = np.random.binomial(1, 0.5, 100)
    >>> y_score = np.random.rand(100)
    >>> metrics = calculate_all_classification_metrics(y_true, y_pred, y_score)
    >>> print(f"F1: {metrics.f1:.4f}, ROC-AUC: {metrics.roc_auc:.4f}")
    """
    return ClassificationMetrics(
        accuracy=accuracy(y_true, y_pred),
        precision=precision(y_true, y_pred),
        recall=recall(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        roc_auc=roc_auc(y_true, y_score) if y_score is not None else None,
        pr_auc=pr_auc(y_true, y_score) if y_score is not None else None
    )


# Example usage
if __name__ == "__main__":
    print("Metrics Module - Example Usage")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100

    # Regression example
    y_true_reg = np.random.randn(n_samples)
    y_pred_reg = y_true_reg + np.random.randn(n_samples) * 0.3

    print("\n1. Regression Metrics:")
    reg_metrics = calculate_all_regression_metrics(y_true_reg, y_pred_reg)
    print(f"   RMSE: {reg_metrics.rmse:.4f}")
    print(f"   MAE: {reg_metrics.mae:.4f}")
    print(f"   R²: {reg_metrics.r2:.4f}")
    print(f"   Spearman ρ: {reg_metrics.spearman:.4f}")

    # Classification example
    y_true_clf = np.random.binomial(1, 0.3, n_samples)
    y_score_clf = np.random.rand(n_samples)
    y_pred_clf = (y_score_clf > 0.5).astype(int)

    print("\n2. Classification Metrics:")
    clf_metrics = calculate_all_classification_metrics(y_true_clf, y_pred_clf, y_score_clf)
    print(f"   Accuracy: {clf_metrics.accuracy:.4f}")
    print(f"   Precision: {clf_metrics.precision:.4f}")
    print(f"   Recall: {clf_metrics.recall:.4f}")
    print(f"   F1: {clf_metrics.f1:.4f}")
    print(f"   ROC-AUC: {clf_metrics.roc_auc:.4f}")

    # Ranking metrics
    print("\n3. Ranking Metrics:")
    ef = enrichment_factor(y_true_clf, y_score_clf, fraction=0.1)
    print(f"   Enrichment Factor (10%): {ef:.2f}")

    top3_acc = top_k_accuracy(y_true_clf, y_score_clf, k=10)
    print(f"   Top-10 Accuracy: {top3_acc:.2%}")

    # Multi-task example
    print("\n4. Multi-task Metrics:")
    y_true_list = [np.random.randn(50), np.random.randn(50)]
    y_pred_list = [y_true_list[0] + np.random.randn(50) * 0.2,
                   y_true_list[1] + np.random.randn(50) * 0.3]
    mt_metrics = multitask_metrics(y_true_list, y_pred_list, ["Task A", "Task B"])
    print(f"   Aggregated RMSE: {mt_metrics['aggregated']['rmse']:.4f}")
    print(f"   Task A R²: {mt_metrics['per_task']['Task A']['r2']:.4f}")

    print("\n" + "=" * 60)
