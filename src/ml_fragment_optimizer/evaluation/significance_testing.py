"""
Statistical significance testing for model comparison and feature importance.

Rigorous statistical testing is essential to determine whether observed
performance differences are real or due to chance.

Key Features:
-------------
- Paired t-test for model comparison
- Wilcoxon signed-rank test (non-parametric)
- Permutation tests for feature importance
- Confidence intervals via bootstrapping
- Bonferroni correction for multiple hypothesis testing
- Effect size calculation (Cohen's d)

Literature:
-----------
- Demšar (2006) "Statistical Comparisons of Classifiers over Multiple Data Sets"
  JMLR 7:1-30
- Dietterich (1998) "Approximate Statistical Tests for Comparing Supervised
  Classification Learning Algorithms" Neural Computation 10(7):1895-1923
- Nadeau & Bengio (2003) "Inference for the Generalization Error"
  Machine Learning 52(3):239-281
- Efron & Tibshirani (1993) "An Introduction to the Bootstrap" Chapman & Hall

WARNING: Multiple comparisons without correction lead to false discoveries
(p-hacking). Always apply appropriate corrections when testing multiple
hypotheses.
"""

from typing import Optional, List, Dict, Any, Tuple, Callable
import numpy as np
from scipy import stats
from dataclasses import dataclass
import warnings


@dataclass
class TestResult:
    """Results from a statistical test."""
    statistic: float
    pvalue: float
    significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    method: str = "unknown"


def paired_ttest(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    y_true: np.ndarray,
    metric: str = "squared_error",
    alternative: str = "two-sided"
) -> TestResult:
    """
    Paired t-test for comparing two models.

    Tests whether two models have significantly different performance on the
    same test set.

    Parameters:
    -----------
    predictions_a : np.ndarray
        Predictions from model A
    predictions_b : np.ndarray
        Predictions from model B
    y_true : np.ndarray
        True labels
    metric : str
        Metric for comparison: "squared_error", "absolute_error", "accuracy"
    alternative : str
        Alternative hypothesis: "two-sided", "less", "greater"

    Returns:
    --------
    result : TestResult
        Test results including p-value and effect size

    Example:
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> pred_a = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    >>> pred_b = np.array([1.2, 2.3, 2.8, 4.1, 5.1])
    >>> result = paired_ttest(pred_a, pred_b, y_true)
    >>> print(f"p-value: {result.pvalue:.4f}")
    >>> if result.significant:
    ...     print("Model A is significantly different from Model B")

    References:
    -----------
    Dietterich (1998) "Approximate Statistical Tests for Comparing Supervised
    Classification Learning Algorithms" Neural Computation 10(7):1895-1923

    Notes:
    ------
    - Paired tests are more powerful than unpaired tests
    - Assumes normally distributed differences (check with Q-Q plot)
    - Use Wilcoxon test if normality assumption is violated
    """
    if len(predictions_a) != len(predictions_b) or len(predictions_a) != len(y_true):
        raise ValueError("All arrays must have the same length")

    # Calculate per-sample errors
    if metric == "squared_error":
        errors_a = (y_true - predictions_a) ** 2
        errors_b = (y_true - predictions_b) ** 2
    elif metric == "absolute_error":
        errors_a = np.abs(y_true - predictions_a)
        errors_b = np.abs(y_true - predictions_b)
    elif metric == "accuracy":
        errors_a = (predictions_a == y_true).astype(float)
        errors_b = (predictions_b == y_true).astype(float)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Paired t-test on differences
    differences = errors_a - errors_b
    t_stat, p_value = stats.ttest_rel(errors_a, errors_b, alternative=alternative)

    # Calculate effect size (Cohen's d)
    cohen_d = np.mean(differences) / (np.std(differences) + 1e-10)

    # Confidence interval for mean difference
    ci = stats.t.interval(
        0.95,
        len(differences) - 1,
        loc=np.mean(differences),
        scale=stats.sem(differences)
    )

    return TestResult(
        statistic=t_stat,
        pvalue=p_value,
        significant=p_value < 0.05,
        effect_size=cohen_d,
        confidence_interval=ci,
        method="paired_ttest"
    )


def wilcoxon_test(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    y_true: np.ndarray,
    metric: str = "squared_error",
    alternative: str = "two-sided"
) -> TestResult:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Use when differences are not normally distributed or data contains outliers.

    Parameters:
    -----------
    predictions_a : np.ndarray
        Predictions from model A
    predictions_b : np.ndarray
        Predictions from model B
    y_true : np.ndarray
        True labels
    metric : str
        Metric for comparison
    alternative : str
        Alternative hypothesis

    Returns:
    --------
    result : TestResult
        Test results

    Example:
    --------
    >>> result = wilcoxon_test(pred_a, pred_b, y_true)
    >>> print(f"Wilcoxon p-value: {result.pvalue:.4f}")

    References:
    -----------
    Demšar (2006) "Statistical Comparisons of Classifiers over Multiple Data Sets"
    JMLR 7:1-30

    Notes:
    ------
    - More robust to outliers than t-test
    - Slightly less powerful if data is truly normal
    - Good default choice for small sample sizes
    """
    if len(predictions_a) != len(predictions_b) or len(predictions_a) != len(y_true):
        raise ValueError("All arrays must have the same length")

    # Calculate per-sample errors
    if metric == "squared_error":
        errors_a = (y_true - predictions_a) ** 2
        errors_b = (y_true - predictions_b) ** 2
    elif metric == "absolute_error":
        errors_a = np.abs(y_true - predictions_a)
        errors_b = np.abs(y_true - predictions_b)
    elif metric == "accuracy":
        errors_a = (predictions_a == y_true).astype(float)
        errors_b = (predictions_b == y_true).astype(float)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Wilcoxon signed-rank test
    stat, p_value = stats.wilcoxon(
        errors_a, errors_b,
        alternative=alternative,
        zero_method="wilcox"
    )

    return TestResult(
        statistic=stat,
        pvalue=p_value,
        significant=p_value < 0.05,
        method="wilcoxon"
    )


def permutation_test(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    y_true: np.ndarray,
    metric: str = "squared_error",
    n_permutations: int = 10000,
    random_state: Optional[int] = None
) -> TestResult:
    """
    Permutation test for model comparison.

    Non-parametric test that makes no distributional assumptions.

    Parameters:
    -----------
    predictions_a : np.ndarray
        Predictions from model A
    predictions_b : np.ndarray
        Predictions from model B
    y_true : np.ndarray
        True labels
    metric : str
        Metric for comparison
    n_permutations : int
        Number of permutations (default: 10000)
    random_state : Optional[int]
        Random seed

    Returns:
    --------
    result : TestResult
        Test results

    Example:
    --------
    >>> result = permutation_test(pred_a, pred_b, y_true, n_permutations=10000)
    >>> print(f"Permutation p-value: {result.pvalue:.4f}")

    References:
    -----------
    Ojala & Garriga (2010) "Permutation Tests for Studying Classifier
    Performance" JMLR 11:1833-1863

    Notes:
    ------
    - Exact test (no distributional assumptions)
    - Computationally expensive for large n_permutations
    - p-value precision limited by n_permutations
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Calculate errors
    if metric == "squared_error":
        errors_a = (y_true - predictions_a) ** 2
        errors_b = (y_true - predictions_b) ** 2
    elif metric == "absolute_error":
        errors_a = np.abs(y_true - predictions_a)
        errors_b = np.abs(y_true - predictions_b)
    elif metric == "accuracy":
        errors_a = (predictions_a == y_true).astype(float)
        errors_b = (predictions_b == y_true).astype(float)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Observed test statistic (mean difference)
    observed_stat = np.mean(errors_a) - np.mean(errors_b)

    # Permutation test
    perm_stats = []
    for _ in range(n_permutations):
        # Randomly swap errors between models
        mask = np.random.rand(len(errors_a)) < 0.5
        perm_a = np.where(mask, errors_a, errors_b)
        perm_b = np.where(mask, errors_b, errors_a)

        perm_stat = np.mean(perm_a) - np.mean(perm_b)
        perm_stats.append(perm_stat)

    perm_stats = np.array(perm_stats)

    # Two-sided p-value
    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))

    return TestResult(
        statistic=observed_stat,
        pvalue=p_value,
        significant=p_value < 0.05,
        method="permutation"
    )


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_fn: Callable,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate bootstrap confidence interval for any statistic.

    Parameters:
    -----------
    data : np.ndarray
        Data to bootstrap
    statistic_fn : Callable
        Function that computes statistic from data
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%)
    random_state : Optional[int]
        Random seed

    Returns:
    --------
    estimate : float
        Point estimate
    ci : Tuple[float, float]
        Confidence interval (lower, upper)

    Example:
    --------
    >>> data = np.random.randn(100)
    >>> estimate, ci = bootstrap_confidence_interval(data, np.mean, n_bootstrap=10000)
    >>> print(f"Mean: {estimate:.4f}, 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    References:
    -----------
    Efron & Tibshirani (1993) "An Introduction to the Bootstrap" Chapman & Hall

    Notes:
    ------
    - Bootstrap provides confidence intervals without parametric assumptions
    - Works for complex statistics (median, variance, etc.)
    - Requires sufficiently large original sample size
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(data)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.randint(0, n, size=n)
        bootstrap_sample = data[indices]
        stat = statistic_fn(bootstrap_sample)
        bootstrap_stats.append(stat)

    bootstrap_stats = np.array(bootstrap_stats)

    # Point estimate
    estimate = statistic_fn(data)

    # Percentile confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return estimate, (lower, upper)


def bootstrap_model_comparison(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    y_true: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Bootstrap-based model comparison with confidence intervals.

    Parameters:
    -----------
    predictions_a : np.ndarray
        Predictions from model A
    predictions_b : np.ndarray
        Predictions from model B
    y_true : np.ndarray
        True labels
    metric_fn : Callable
        Metric function (e.g., lambda y_true, y_pred: np.mean((y_true - y_pred)**2))
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level
    random_state : Optional[int]
        Random seed

    Returns:
    --------
    results : Dict[str, Any]
        Comparison results with confidence intervals

    Example:
    --------
    >>> def mse(y_true, y_pred):
    ...     return np.mean((y_true - y_pred) ** 2)
    >>> results = bootstrap_model_comparison(pred_a, pred_b, y_true, mse)
    >>> print(f"Difference: {results['difference']:.4f}")
    >>> print(f"95% CI: {results['ci_difference']}")
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(y_true)
    diffs = []
    metric_a_values = []
    metric_b_values = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.randint(0, n, size=n)
        y_true_boot = y_true[indices]
        pred_a_boot = predictions_a[indices]
        pred_b_boot = predictions_b[indices]

        # Calculate metrics
        metric_a = metric_fn(y_true_boot, pred_a_boot)
        metric_b = metric_fn(y_true_boot, pred_b_boot)

        metric_a_values.append(metric_a)
        metric_b_values.append(metric_b)
        diffs.append(metric_a - metric_b)

    metric_a_values = np.array(metric_a_values)
    metric_b_values = np.array(metric_b_values)
    diffs = np.array(diffs)

    # Confidence intervals
    alpha = 1 - confidence_level
    ci_a = (np.percentile(metric_a_values, 100 * alpha / 2),
            np.percentile(metric_a_values, 100 * (1 - alpha / 2)))
    ci_b = (np.percentile(metric_b_values, 100 * alpha / 2),
            np.percentile(metric_b_values, 100 * (1 - alpha / 2)))
    ci_diff = (np.percentile(diffs, 100 * alpha / 2),
               np.percentile(diffs, 100 * (1 - alpha / 2)))

    # p-value: fraction of bootstrap samples where difference crosses zero
    p_value = np.mean(diffs * np.sign(np.mean(diffs)) <= 0)

    return {
        "metric_a": metric_fn(y_true, predictions_a),
        "metric_b": metric_fn(y_true, predictions_b),
        "difference": np.mean(diffs),
        "ci_a": ci_a,
        "ci_b": ci_b,
        "ci_difference": ci_diff,
        "pvalue": p_value,
        "significant": ci_diff[0] * ci_diff[1] > 0  # CI doesn't contain 0
    }


def bonferroni_correction(
    pvalues: List[float],
    alpha: float = 0.05
) -> Tuple[List[bool], float]:
    """
    Bonferroni correction for multiple hypothesis testing.

    Parameters:
    -----------
    pvalues : List[float]
        List of p-values
    alpha : float
        Family-wise error rate (default: 0.05)

    Returns:
    --------
    significant : List[bool]
        Whether each test is significant after correction
    corrected_alpha : float
        Corrected significance threshold

    Example:
    --------
    >>> pvalues = [0.01, 0.03, 0.05, 0.10]
    >>> significant, threshold = bonferroni_correction(pvalues)
    >>> print(f"Corrected threshold: {threshold:.4f}")
    >>> print(f"Significant tests: {sum(significant)}/{len(pvalues)}")

    References:
    -----------
    Bonferroni (1936) "Teoria statistica delle classi e calcolo delle
    probabilità" Pubblicazioni del R Istituto Superiore di Scienze Economiche
    e Commerciali di Firenze 8:3-62

    Notes:
    ------
    - Very conservative (high false negative rate)
    - Use Holm-Bonferroni or FDR for more power
    - Controls family-wise error rate (FWER)
    """
    n_tests = len(pvalues)
    corrected_alpha = alpha / n_tests
    significant = [p < corrected_alpha for p in pvalues]

    return significant, corrected_alpha


def holm_bonferroni_correction(
    pvalues: List[float],
    alpha: float = 0.05
) -> List[bool]:
    """
    Holm-Bonferroni correction (less conservative than Bonferroni).

    Parameters:
    -----------
    pvalues : List[float]
        List of p-values
    alpha : float
        Family-wise error rate

    Returns:
    --------
    significant : List[bool]
        Whether each test is significant after correction

    Example:
    --------
    >>> pvalues = [0.01, 0.03, 0.05, 0.10]
    >>> significant = holm_bonferroni_correction(pvalues)
    >>> print(f"Significant tests: {sum(significant)}/{len(pvalues)}")

    References:
    -----------
    Holm (1979) "A simple sequentially rejective multiple test procedure"
    Scand J Statist 6(2):65-70

    Notes:
    ------
    - More powerful than Bonferroni
    - Still controls FWER
    - Step-down procedure
    """
    n_tests = len(pvalues)
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = np.array(pvalues)[sorted_indices]

    significant_sorted = []
    for i, p in enumerate(sorted_pvalues):
        # Adjusted alpha for this step
        adjusted_alpha = alpha / (n_tests - i)
        if p < adjusted_alpha:
            significant_sorted.append(True)
        else:
            # Once we fail to reject, reject all subsequent
            significant_sorted.extend([False] * (n_tests - i))
            break

    # Unsort results
    significant = [False] * n_tests
    for i, sig in zip(sorted_indices, significant_sorted):
        significant[i] = sig

    return significant


def benjamini_hochberg_fdr(
    pvalues: List[float],
    fdr: float = 0.05
) -> List[bool]:
    """
    Benjamini-Hochberg procedure for controlling False Discovery Rate.

    Parameters:
    -----------
    pvalues : List[float]
        List of p-values
    fdr : float
        False discovery rate (default: 0.05)

    Returns:
    --------
    significant : List[bool]
        Whether each test is significant

    Example:
    --------
    >>> pvalues = [0.001, 0.01, 0.03, 0.05, 0.10]
    >>> significant = benjamini_hochberg_fdr(pvalues, fdr=0.05)
    >>> print(f"Significant tests: {sum(significant)}/{len(pvalues)}")

    References:
    -----------
    Benjamini & Hochberg (1995) "Controlling the False Discovery Rate: A
    Practical and Powerful Approach to Multiple Testing" J R Stat Soc B 57(1):289-300

    Notes:
    ------
    - More powerful than FWER methods for large number of tests
    - Controls expected proportion of false positives among discoveries
    - Widely used in genomics and high-dimensional testing
    """
    n_tests = len(pvalues)
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = np.array(pvalues)[sorted_indices]

    # Find largest i where p(i) <= (i/m) * FDR
    threshold_indices = []
    for i in range(n_tests):
        threshold = (i + 1) / n_tests * fdr
        if sorted_pvalues[i] <= threshold:
            threshold_indices.append(i)

    if len(threshold_indices) == 0:
        return [False] * n_tests

    # Reject all up to and including max i
    max_i = max(threshold_indices)
    significant_sorted = [i <= max_i for i in range(n_tests)]

    # Unsort results
    significant = [False] * n_tests
    for i, sig in zip(sorted_indices, significant_sorted):
        significant[i] = sig

    return significant


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.

    Measures the standardized difference between two means.

    Parameters:
    -----------
    group_a : np.ndarray
        First group
    group_b : np.ndarray
        Second group

    Returns:
    --------
    d : float
        Cohen's d effect size

    Example:
    --------
    >>> group_a = np.random.randn(100)
    >>> group_b = np.random.randn(100) + 0.5
    >>> d = cohens_d(group_a, group_b)
    >>> print(f"Cohen's d: {d:.4f}")

    Interpretation:
    ---------------
    - d < 0.2: negligible effect
    - 0.2 ≤ d < 0.5: small effect
    - 0.5 ≤ d < 0.8: medium effect
    - d ≥ 0.8: large effect

    References:
    -----------
    Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences" 2nd ed.
    """
    mean_diff = np.mean(group_a) - np.mean(group_b)
    pooled_std = np.sqrt((np.var(group_a) + np.var(group_b)) / 2)
    return mean_diff / (pooled_std + 1e-10)


def feature_importance_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    feature_idx: int,
    metric_fn: Callable,
    n_permutations: int = 1000,
    random_state: Optional[int] = None
) -> TestResult:
    """
    Permutation test for feature importance.

    Tests whether a feature significantly contributes to model performance.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    model : Any
        Trained model
    feature_idx : int
        Index of feature to test
    metric_fn : Callable
        Metric function
    n_permutations : int
        Number of permutations
    random_state : Optional[int]
        Random seed

    Returns:
    --------
    result : TestResult
        Test results

    Example:
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> model = RandomForestRegressor()
    >>> model.fit(X_train, y_train)
    >>> def mse(y_true, y_pred): return np.mean((y_true - y_pred)**2)
    >>> result = feature_importance_permutation_test(X_test, y_test, model, 0, mse)
    >>> print(f"Feature 0 importance p-value: {result.pvalue:.4f}")

    References:
    -----------
    Breiman (2001) "Random Forests" Machine Learning 45(1):5-32

    Notes:
    ------
    - Permuting a feature destroys its relationship with target
    - If performance drops significantly, feature is important
    - Model-agnostic method
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Baseline performance
    y_pred = model.predict(X)
    baseline_score = metric_fn(y, y_pred)

    # Permutation scores
    perm_scores = []
    for _ in range(n_permutations):
        X_perm = X.copy()
        X_perm[:, feature_idx] = np.random.permutation(X_perm[:, feature_idx])
        y_pred_perm = model.predict(X_perm)
        perm_score = metric_fn(y, y_pred_perm)
        perm_scores.append(perm_score)

    perm_scores = np.array(perm_scores)

    # Feature importance = performance drop
    importance = np.mean(perm_scores) - baseline_score

    # p-value: how often permutation is as good as baseline
    # (for loss metrics, higher is worse)
    p_value = np.mean(perm_scores <= baseline_score)

    return TestResult(
        statistic=importance,
        pvalue=p_value,
        significant=p_value < 0.05,
        method="permutation_importance"
    )


def mcnemar_test(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    y_true: np.ndarray
) -> TestResult:
    """
    McNemar's test for comparing binary classifiers.

    Tests whether two classifiers have significantly different error rates.

    Parameters:
    -----------
    predictions_a : np.ndarray
        Binary predictions from model A
    predictions_b : np.ndarray
        Binary predictions from model B
    y_true : np.ndarray
        True binary labels

    Returns:
    --------
    result : TestResult
        Test results

    Example:
    --------
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> pred_a = np.array([0, 1, 0, 0, 1])
    >>> pred_b = np.array([0, 1, 1, 1, 1])
    >>> result = mcnemar_test(pred_a, pred_b, y_true)
    >>> print(f"McNemar p-value: {result.pvalue:.4f}")

    References:
    -----------
    McNemar (1947) "Note on the sampling error of the difference between
    correlated proportions or percentages" Psychometrika 12(2):153-157

    Notes:
    ------
    - Specifically for binary classification
    - More powerful than comparing accuracies
    - Accounts for correlation between models
    """
    # Check predictions are correct
    correct_a = (predictions_a == y_true)
    correct_b = (predictions_b == y_true)

    # Contingency table
    # n01: A wrong, B correct
    # n10: A correct, B wrong
    n01 = np.sum(~correct_a & correct_b)
    n10 = np.sum(correct_a & ~correct_b)

    # McNemar's test statistic (with continuity correction)
    if n01 + n10 == 0:
        return TestResult(statistic=0, pvalue=1.0, significant=False, method="mcnemar")

    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)

    # Chi-squared test with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(stat, df=1)

    return TestResult(
        statistic=stat,
        pvalue=p_value,
        significant=p_value < 0.05,
        method="mcnemar"
    )


# Example usage
if __name__ == "__main__":
    print("Significance Testing Module - Example Usage")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100

    y_true = np.random.randn(n_samples)
    pred_a = y_true + np.random.randn(n_samples) * 0.5
    pred_b = y_true + np.random.randn(n_samples) * 0.6

    print("\n1. Paired t-test:")
    result = paired_ttest(pred_a, pred_b, y_true, metric="squared_error")
    print(f"   t-statistic: {result.statistic:.4f}")
    print(f"   p-value: {result.pvalue:.4f}")
    print(f"   Cohen's d: {result.effect_size:.4f}")
    print(f"   Significant: {result.significant}")

    print("\n2. Wilcoxon test:")
    result = wilcoxon_test(pred_a, pred_b, y_true)
    print(f"   W-statistic: {result.statistic:.4f}")
    print(f"   p-value: {result.pvalue:.4f}")
    print(f"   Significant: {result.significant}")

    print("\n3. Permutation test:")
    result = permutation_test(pred_a, pred_b, y_true, n_permutations=1000)
    print(f"   Test statistic: {result.statistic:.4f}")
    print(f"   p-value: {result.pvalue:.4f}")
    print(f"   Significant: {result.significant}")

    print("\n4. Bootstrap confidence interval:")
    def rmse(data):
        return np.sqrt(np.mean(data ** 2))
    errors = y_true - pred_a
    estimate, ci = bootstrap_confidence_interval(errors, rmse, n_bootstrap=1000)
    print(f"   RMSE estimate: {estimate:.4f}")
    print(f"   95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    print("\n5. Multiple testing correction:")
    pvalues = [0.001, 0.01, 0.03, 0.05, 0.10]
    print(f"   Original p-values: {pvalues}")

    sig_bonf, threshold = bonferroni_correction(pvalues)
    print(f"   Bonferroni: {sum(sig_bonf)}/{len(pvalues)} significant (α={threshold:.4f})")

    sig_holm = holm_bonferroni_correction(pvalues)
    print(f"   Holm-Bonferroni: {sum(sig_holm)}/{len(pvalues)} significant")

    sig_bh = benjamini_hochberg_fdr(pvalues)
    print(f"   Benjamini-Hochberg: {sum(sig_bh)}/{len(pvalues)} significant")

    print("\n6. Effect size (Cohen's d):")
    group_a = np.random.randn(50)
    group_b = np.random.randn(50) + 0.5
    d = cohens_d(group_a, group_b)
    print(f"   Cohen's d: {d:.4f}")
    if abs(d) < 0.2:
        print("   Interpretation: negligible effect")
    elif abs(d) < 0.5:
        print("   Interpretation: small effect")
    elif abs(d) < 0.8:
        print("   Interpretation: medium effect")
    else:
        print("   Interpretation: large effect")

    print("\n7. McNemar's test (binary classification):")
    y_true_bin = np.random.binomial(1, 0.5, 100)
    pred_a_bin = np.random.binomial(1, 0.7, 100)
    pred_b_bin = np.random.binomial(1, 0.6, 100)
    result = mcnemar_test(pred_a_bin, pred_b_bin, y_true_bin)
    print(f"   χ² statistic: {result.statistic:.4f}")
    print(f"   p-value: {result.pvalue:.4f}")
    print(f"   Significant: {result.significant}")

    print("\n" + "=" * 60)
    print("\nWARNING: Always correct for multiple comparisons!")
    print("Uncorrected testing leads to false discoveries (p-hacking).")
