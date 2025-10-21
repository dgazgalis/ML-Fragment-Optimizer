"""
Model calibration analysis and calibration methods.

Calibrated models produce probability estimates that match observed frequencies.
For example, if a model predicts 70% probability, approximately 70% of such
predictions should be correct.

Key Features:
-------------
- Reliability diagrams (calibration curves)
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier score for classification
- Calibration methods: Platt scaling, isotonic regression, temperature scaling
- Confidence interval coverage analysis

Literature:
-----------
- Guo et al. (2017) "On Calibration of Modern Neural Networks" ICML
- Niculescu-Mizil & Caruana (2005) "Predicting good probabilities with
  supervised learning" ICML
- Zadrozny & Elkan (2002) "Transforming classifier scores into accurate
  multiclass probability estimates" KDD

Example:
--------
>>> from calibration import plot_calibration_curve, expected_calibration_error
>>> y_true = np.array([0, 0, 1, 1, 1])
>>> y_prob = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
>>> ece = expected_calibration_error(y_true, y_prob, n_bins=5)
>>> plot_calibration_curve(y_true, y_prob)
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import warnings
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    warnings.warn("Matplotlib not available. Plotting functions will not work.")

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax


@dataclass
class CalibrationResult:
    """Results from calibration analysis."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    bin_accuracies: np.ndarray
    bin_confidences: np.ndarray
    bin_counts: np.ndarray


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform"
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    ECE measures the difference between predicted probabilities and actual
    frequencies, averaged over bins.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_prob : np.ndarray
        Predicted probabilities for positive class
    n_bins : int
        Number of bins for discretization (default: 10)
    strategy : str
        Binning strategy: "uniform" or "quantile" (default: "uniform")

    Returns:
    --------
    ece : float
        Expected Calibration Error (0 to 1, lower is better)

    Example:
    --------
    >>> y_true = np.array([0, 0, 1, 1, 1, 1])
    >>> y_prob = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9])
    >>> ece = expected_calibration_error(y_true, y_prob)
    >>> print(f"ECE: {ece:.4f}")

    References:
    -----------
    Naeini et al. (2015) "Obtaining Well Calibrated Probabilities Using
    Bayesian Binning" AAAI

    Notes:
    ------
    - ECE = 0 indicates perfect calibration
    - ECE < 0.05 is generally considered well-calibrated
    - Use more bins for larger datasets
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have same length")

    # Create bins
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    # Assign samples to bins
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    # Calculate ECE
    ece = 0.0
    n_samples = len(y_true)

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_size = np.sum(mask)

            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return ece


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform"
) -> float:
    """
    Calculate Maximum Calibration Error (MCE).

    MCE is the maximum difference between predicted probability and actual
    frequency across all bins.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities
    n_bins : int
        Number of bins
    strategy : str
        Binning strategy

    Returns:
    --------
    mce : float
        Maximum Calibration Error

    Example:
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_prob = np.array([0.1, 0.3, 0.6, 0.9])
    >>> mce = maximum_calibration_error(y_true, y_prob)

    Notes:
    ------
    MCE is more sensitive to outliers than ECE, highlighting worst-case
    miscalibration.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Create bins
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    # Calculate MCE
    mce = 0.0

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            calibration_error = np.abs(bin_acc - bin_conf)

            mce = max(mce, calibration_error)

    return mce


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate Brier score (mean squared error for probabilities).

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities

    Returns:
    --------
    brier : float
        Brier score (0 to 1, lower is better)

    Example:
    --------
    >>> y_true = np.array([0, 1, 1])
    >>> y_prob = np.array([0.2, 0.8, 0.9])
    >>> bs = brier_score(y_true, y_prob)

    References:
    -----------
    Brier (1950) "Verification of forecasts expressed in terms of probability"
    Mon Weather Rev 78(1):1-3

    Notes:
    ------
    Brier score combines calibration and refinement (discrimination).
    A well-calibrated but uninformative model can still have high Brier score.
    """
    return np.mean((y_prob - y_true) ** 2)


def calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve data for plotting.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities
    n_bins : int
        Number of bins
    strategy : str
        Binning strategy

    Returns:
    --------
    bin_accuracies : np.ndarray
        Observed frequency in each bin
    bin_confidences : np.ndarray
        Mean predicted probability in each bin
    bin_counts : np.ndarray
        Number of samples in each bin

    Example:
    --------
    >>> y_true = np.random.binomial(1, 0.7, 100)
    >>> y_prob = np.random.beta(7, 3, 100)
    >>> acc, conf, counts = calibration_curve(y_true, y_prob)
    """
    # Create bins
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if np.sum(mask) > 0:
            bin_accuracies.append(np.mean(y_true[mask]))
            bin_confidences.append(np.mean(y_prob[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)
            bin_counts.append(0)

    return (
        np.array(bin_accuracies),
        np.array(bin_confidences),
        np.array(bin_counts)
    )


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
    ax: Optional[Any] = None,
    show_histogram: bool = True
) -> Any:
    """
    Plot reliability diagram (calibration curve).

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities
    n_bins : int
        Number of bins
    strategy : str
        Binning strategy
    ax : Optional[matplotlib.axes.Axes]
        Axes object (if None, creates new figure)
    show_histogram : bool
        Whether to show histogram of predictions

    Returns:
    --------
    ax : matplotlib.axes.Axes
        Axes object with plot

    Example:
    --------
    >>> y_true = np.random.binomial(1, 0.7, 1000)
    >>> y_prob = np.random.beta(7, 3, 1000)
    >>> plot_calibration_curve(y_true, y_prob)
    >>> plt.show()

    Notes:
    ------
    A perfectly calibrated model produces a diagonal line. Deviations indicate
    miscalibration.
    """
    if not MPL_AVAILABLE:
        raise ImportError("Matplotlib is required for plotting")

    bin_acc, bin_conf, bin_counts = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )

    # Remove empty bins
    valid = ~np.isnan(bin_acc)
    bin_acc = bin_acc[valid]
    bin_conf = bin_conf[valid]
    bin_counts = bin_counts[valid]

    if ax is None:
        if show_histogram:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                           gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(figsize=(8, 6))
    else:
        ax1 = ax

    # Plot calibration curve
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)
    ax1.plot(bin_conf, bin_acc, 'o-', label='Model calibration', linewidth=2)

    # Add error bars based on bin size
    errors = 1.96 * np.sqrt(bin_acc * (1 - bin_acc) / bin_counts)  # 95% CI
    ax1.errorbar(bin_conf, bin_acc, yerr=errors, fmt='none', alpha=0.3)

    ax1.set_xlabel('Mean predicted probability', fontsize=12)
    ax1.set_ylabel('Observed frequency', fontsize=12)
    ax1.set_title('Calibration Curve (Reliability Diagram)', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Calculate and display metrics
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    mce = maximum_calibration_error(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    bs = brier_score(y_true, y_prob)

    textstr = f'ECE: {ece:.4f}\nMCE: {mce:.4f}\nBrier: {bs:.4f}'
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

    # Histogram of predictions
    if show_histogram and ax is None:
        ax2.hist(y_prob, bins=n_bins, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Predicted probability', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Distribution of predictions', fontsize=12)
        ax2.grid(alpha=0.3)

    return ax1 if not show_histogram else (ax1, ax2)


class PlattScaling:
    """
    Platt scaling for probability calibration.

    Fits a logistic regression model on validation set to calibrate probabilities.

    References:
    -----------
    Platt (1999) "Probabilistic outputs for support vector machines and
    comparisons to regularized likelihood methods"

    Example:
    --------
    >>> platt = PlattScaling()
    >>> platt.fit(val_scores, val_labels)
    >>> calibrated_probs = platt.transform(test_scores)
    """

    def __init__(self):
        self.calibrator = LogisticRegression()

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> "PlattScaling":
        """
        Fit Platt scaling on validation set.

        Parameters:
        -----------
        scores : np.ndarray
            Uncalibrated scores or logits
        y_true : np.ndarray
            True labels

        Returns:
        --------
        self : PlattScaling
        """
        scores = scores.reshape(-1, 1)
        self.calibrator.fit(scores, y_true)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to get calibrated probabilities.

        Parameters:
        -----------
        scores : np.ndarray
            Uncalibrated scores

        Returns:
        --------
        calibrated_probs : np.ndarray
            Calibrated probabilities
        """
        scores = scores.reshape(-1, 1)
        return self.calibrator.predict_proba(scores)[:, 1]

    def fit_transform(self, scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(scores, y_true).transform(scores)


class IsotonicCalibration:
    """
    Isotonic regression for probability calibration.

    Fits a non-decreasing step function on validation set. More flexible than
    Platt scaling but requires more data.

    References:
    -----------
    Zadrozny & Elkan (2002) "Transforming classifier scores into accurate
    multiclass probability estimates" KDD

    Example:
    --------
    >>> iso = IsotonicCalibration()
    >>> iso.fit(val_probs, val_labels)
    >>> calibrated_probs = iso.transform(test_probs)
    """

    def __init__(self, out_of_bounds: str = "clip"):
        """
        Parameters:
        -----------
        out_of_bounds : str
            How to handle predictions outside [0, 1]: "clip" or "nan"
        """
        self.calibrator = IsotonicRegression(out_of_bounds=out_of_bounds)

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibration":
        """Fit isotonic regression on validation set."""
        self.calibrator.fit(probs, y_true)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""
        return self.calibrator.predict(probs)

    def fit_transform(self, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(probs, y_true).transform(probs)


class TemperatureScaling:
    """
    Temperature scaling for neural network calibration.

    Divides logits by a learned temperature parameter before softmax.
    Simple but effective for deep learning models.

    References:
    -----------
    Guo et al. (2017) "On Calibration of Modern Neural Networks" ICML

    Example:
    --------
    >>> temp = TemperatureScaling()
    >>> temp.fit(val_logits, val_labels)
    >>> calibrated_probs = temp.transform(test_logits)

    Notes:
    ------
    - Temperature scaling preserves model accuracy (argmax unchanged)
    - Typically improves calibration without requiring large validation set
    - Temperature > 1 "softens" probabilities (reduces overconfidence)
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        max_iter: int = 100,
        lr: float = 0.01
    ) -> "TemperatureScaling":
        """
        Fit temperature parameter using gradient descent.

        Parameters:
        -----------
        logits : np.ndarray
            Model logits (before softmax)
        y_true : np.ndarray
            True labels
        max_iter : int
            Maximum optimization iterations
        lr : float
            Learning rate

        Returns:
        --------
        self : TemperatureScaling
        """
        # Simple gradient descent to minimize negative log likelihood
        temperature = 1.0

        for _ in range(max_iter):
            scaled_logits = logits / temperature
            probs = softmax(scaled_logits, axis=-1)

            # Negative log likelihood
            nll = -np.mean(np.log(probs[np.arange(len(y_true)), y_true] + 1e-10))

            # Gradient w.r.t. temperature (approximate)
            grad = np.sum((probs - np.eye(probs.shape[1])[y_true]) * scaled_logits) / len(y_true)

            # Update temperature
            temperature -= lr * grad / temperature

            # Ensure temperature stays positive
            temperature = max(0.01, temperature)

        self.temperature = temperature
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        scaled_logits = logits / self.temperature
        return softmax(scaled_logits, axis=-1)

    def fit_transform(self, logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(logits, y_true).transform(logits)


def confidence_interval_coverage(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    confidence_level: float = 0.9
) -> Dict[str, float]:
    """
    Calculate confidence interval coverage for regression.

    Check if predicted confidence intervals contain the true values at the
    expected rate.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred_mean : np.ndarray
        Predicted means
    y_pred_std : np.ndarray
        Predicted standard deviations
    confidence_level : float
        Confidence level (e.g., 0.9 for 90%)

    Returns:
    --------
    coverage_stats : Dict[str, float]
        Coverage statistics including empirical coverage and calibration error

    Example:
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y_pred_mean = np.array([1.1, 2.1, 2.9, 3.8])
    >>> y_pred_std = np.array([0.5, 0.5, 0.5, 0.5])
    >>> stats = confidence_interval_coverage(y_true, y_pred_mean, y_pred_std)
    >>> print(f"Coverage: {stats['coverage']:.2%}")

    Notes:
    ------
    For well-calibrated regression models:
    - 90% confidence intervals should contain 90% of true values
    - 95% confidence intervals should contain 95% of true values
    """
    from scipy import stats

    # Calculate z-score for confidence level
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    # Calculate confidence intervals
    lower = y_pred_mean - z_score * y_pred_std
    upper = y_pred_mean + z_score * y_pred_std

    # Check coverage
    within_interval = (y_true >= lower) & (y_true <= upper)
    empirical_coverage = np.mean(within_interval)

    # Calibration error
    calibration_error = abs(empirical_coverage - confidence_level)

    # Sharpness (average interval width)
    sharpness = np.mean(upper - lower)

    return {
        "confidence_level": confidence_level,
        "empirical_coverage": empirical_coverage,
        "calibration_error": calibration_error,
        "sharpness": sharpness,
        "well_calibrated": calibration_error < 0.05
    }


def analyze_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> CalibrationResult:
    """
    Comprehensive calibration analysis.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities
    n_bins : int
        Number of bins

    Returns:
    --------
    result : CalibrationResult
        Comprehensive calibration metrics

    Example:
    --------
    >>> y_true = np.random.binomial(1, 0.7, 1000)
    >>> y_prob = np.random.beta(7, 3, 1000)
    >>> result = analyze_calibration(y_true, y_prob)
    >>> print(f"ECE: {result.ece:.4f}")
    """
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    mce = maximum_calibration_error(y_true, y_prob, n_bins=n_bins)
    bs = brier_score(y_true, y_prob)
    bin_acc, bin_conf, bin_counts = calibration_curve(y_true, y_prob, n_bins=n_bins)

    return CalibrationResult(
        ece=ece,
        mce=mce,
        brier_score=bs,
        bin_accuracies=bin_acc,
        bin_confidences=bin_conf,
        bin_counts=bin_counts
    )


# Example usage
if __name__ == "__main__":
    print("Calibration Module - Example Usage")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000

    # Well-calibrated model
    y_true_calibrated = np.random.binomial(1, 0.7, n_samples)
    y_prob_calibrated = np.random.beta(7, 3, n_samples)

    # Overconfident model
    y_true_overconfident = np.random.binomial(1, 0.5, n_samples)
    y_prob_overconfident = np.random.beta(9, 1, n_samples)

    print("\n1. Calibrated Model:")
    result1 = analyze_calibration(y_true_calibrated, y_prob_calibrated)
    print(f"   ECE: {result1.ece:.4f}")
    print(f"   MCE: {result1.mce:.4f}")
    print(f"   Brier Score: {result1.brier_score:.4f}")

    print("\n2. Overconfident Model:")
    result2 = analyze_calibration(y_true_overconfident, y_prob_overconfident)
    print(f"   ECE: {result2.ece:.4f}")
    print(f"   MCE: {result2.mce:.4f}")
    print(f"   Brier Score: {result2.brier_score:.4f}")

    print("\n3. Platt Scaling Calibration:")
    platt = PlattScaling()
    # Use half for calibration, half for evaluation
    mid = n_samples // 2
    platt.fit(y_prob_overconfident[:mid], y_true_overconfident[:mid])
    y_prob_calibrated_platt = platt.transform(y_prob_overconfident[mid:])
    result3 = analyze_calibration(y_true_overconfident[mid:], y_prob_calibrated_platt)
    print(f"   ECE after calibration: {result3.ece:.4f}")

    print("\n4. Confidence Interval Coverage (Regression):")
    y_true_reg = np.random.normal(0, 1, n_samples)
    y_pred_mean = y_true_reg + np.random.normal(0, 0.1, n_samples)
    y_pred_std = np.ones(n_samples) * 0.5
    coverage = confidence_interval_coverage(y_true_reg, y_pred_mean, y_pred_std, 0.9)
    print(f"   Expected coverage: 90%")
    print(f"   Empirical coverage: {coverage['empirical_coverage']:.2%}")
    print(f"   Well calibrated: {coverage['well_calibrated']}")

    if MPL_AVAILABLE:
        print("\n5. Plotting calibration curves...")
        plot_calibration_curve(y_true_overconfident, y_prob_overconfident)
        plt.savefig("calibration_example.png")
        print("   Saved to calibration_example.png")

    print("\n" + "=" * 60)
