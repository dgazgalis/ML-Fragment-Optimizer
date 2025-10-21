"""
Comprehensive examples demonstrating the evaluation module.

This script shows how to use all components of the evaluation module in
realistic scenarios for molecular property prediction.

Run this script to see examples of:
1. Proper train/test splitting strategies
2. Model calibration assessment and correction
3. Applicability domain definition
4. Temporal/prospective validation
5. Statistical model comparison
6. Complete validation pipeline
"""

import numpy as np
import warnings
from typing import Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("=" * 80)
print("ML-Fragment-Optimizer Evaluation Module - Comprehensive Examples")
print("=" * 80)


# =============================================================================
# Example 1: Proper Train/Test Splitting
# =============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 1: Proper Train/Test Splitting")
print("=" * 80)

try:
    from evaluation import benchmarks

    # Simulate molecular dataset
    print("\n1.1 Scaffold-Based Splitting (Prevents Similarity Leakage)")
    print("-" * 80)

    # Example SMILES (benzene derivatives and pyridines)
    smiles_example = [
        "CCc1ccccc1",      # ethylbenzene
        "CCCc1ccccc1",     # propylbenzene
        "CCCCc1ccccc1",    # butylbenzene
        "c1ccccc1",        # benzene
        "Cc1ccccc1",       # toluene
        "c1cccnc1",        # pyridine
        "Cc1cccnc1",       # methylpyridine
        "CCc1cccnc1",      # ethylpyridine
        "c1ccncc1",        # pyridine isomer
        "CCN1CCCCC1",      # N-ethylpiperidine
    ]
    activities = np.array([5.0, 5.2, 5.5, 4.8, 4.9, 7.0, 7.1, 7.2, 6.9, 6.5])

    # Scaffold split
    train_idx, test_idx = benchmarks.scaffold_split(
        smiles_example,
        test_size=0.3,
        balanced=True
    )

    print(f"Total molecules: {len(smiles_example)}")
    print(f"Training set size: {len(train_idx)}")
    print(f"Test set size: {len(test_idx)}")
    print(f"\nTraining SMILES:")
    for idx in train_idx:
        print(f"  {smiles_example[idx]}")
    print(f"\nTest SMILES:")
    for idx in test_idx:
        print(f"  {smiles_example[idx]}")

    # Check for leakage
    print("\n1.2 Data Leakage Detection")
    print("-" * 80)
    leakage = benchmarks.detect_data_leakage(
        [smiles_example[i] for i in train_idx],
        [smiles_example[i] for i in test_idx],
        similarity_threshold=0.8
    )
    print(f"Leakage score: {leakage['leakage_score']:.2%}")
    print(f"Similar pairs found: {leakage['n_similar_pairs']}")
    print(f"Warning flag: {leakage['warning']}")

    if leakage['examples']:
        print("\nMost similar train/test pairs:")
        for train_i, test_i, sim in leakage['examples'][:3]:
            print(f"  Train[{train_i}] vs Test[{test_i}]: Similarity={sim:.3f}")

except ImportError:
    print("RDKit not available - skipping scaffold splitting examples")


# =============================================================================
# Example 2: Temporal Splitting
# =============================================================================
print("\n1.3 Temporal Splitting (Prospective Validation)")
print("-" * 80)

dates = np.array([2018, 2018, 2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022])
train_idx_temp, test_idx_temp = benchmarks.temporal_split(dates, test_size=0.3)

print(f"Training dates: {dates[train_idx_temp]}")
print(f"Test dates: {dates[test_idx_temp]}")
print(f"Note: Test set contains only FUTURE data (no temporal leakage)")


# =============================================================================
# Example 2: Model Calibration
# =============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 2: Model Calibration Assessment")
print("=" * 80)

from evaluation import calibration

# Simulate binary classification
print("\n2.1 Calibration Metrics for Binary Classification")
print("-" * 80)

np.random.seed(42)
n_samples = 1000

# Well-calibrated model
y_true_calib = np.random.binomial(1, 0.7, n_samples)
y_prob_calib = np.random.beta(7, 3, n_samples)

# Overconfident model
y_true_overconf = np.random.binomial(1, 0.5, n_samples)
y_prob_overconf = np.random.beta(9, 1, n_samples)

print("Well-calibrated model:")
result1 = calibration.analyze_calibration(y_true_calib, y_prob_calib)
print(f"  ECE: {result1.ece:.4f}")
print(f"  MCE: {result1.mce:.4f}")
print(f"  Brier Score: {result1.brier_score:.4f}")

print("\nOverconfident model:")
result2 = calibration.analyze_calibration(y_true_overconf, y_prob_overconf)
print(f"  ECE: {result2.ece:.4f}")
print(f"  MCE: {result2.mce:.4f}")
print(f"  Brier Score: {result2.brier_score:.4f}")

print("\n2.2 Calibration Correction with Platt Scaling")
print("-" * 80)

# Split for calibration
mid = n_samples // 2
platt = calibration.PlattScaling()
platt.fit(y_prob_overconf[:mid], y_true_overconf[:mid])
y_prob_corrected = platt.transform(y_prob_overconf[mid:])

result3 = calibration.analyze_calibration(
    y_true_overconf[mid:],
    y_prob_corrected
)
print(f"ECE before calibration: {result2.ece:.4f}")
print(f"ECE after calibration: {result3.ece:.4f}")
print(f"Improvement: {(result2.ece - result3.ece):.4f}")


# =============================================================================
# Example 3: Applicability Domain
# =============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 3: Applicability Domain Assessment")
print("=" * 80)

from evaluation import applicability_domain as ad

# Simulate molecular descriptors
np.random.seed(42)
n_train = 500
n_test = 100
n_features = 10

X_train = np.random.randn(n_train, n_features)
X_test = np.random.randn(n_test, n_features)

# Add some outliers to test set
X_test[-10:] += 4.0

print("\n3.1 k-NN Applicability Domain")
print("-" * 80)

ad_knn = ad.KNNApplicabilityDomain(k=5)
ad_knn.fit(X_train)
result_knn = ad_knn.predict(X_test)

print(f"Total test samples: {len(X_test)}")
print(f"In-domain samples: {np.sum(result_knn.in_domain)} ({np.mean(result_knn.in_domain):.1%})")
print(f"Out-of-domain samples: {np.sum(~result_knn.in_domain)}")
print(f"Mean reliability score: {np.mean(result_knn.reliability_scores):.4f}")

# Show examples
print("\nMost reliable predictions:")
sorted_idx = np.argsort(result_knn.reliability_scores)[::-1]
for i in sorted_idx[:3]:
    print(f"  Sample {i}: Reliability={result_knn.reliability_scores[i]:.4f}")

print("\nLeast reliable predictions:")
for i in sorted_idx[-3:]:
    print(f"  Sample {i}: Reliability={result_knn.reliability_scores[i]:.4f}")

print("\n3.2 Ensemble Applicability Domain")
print("-" * 80)

results = ad.ensemble_applicability_domain(X_train, X_test)
print("Consensus across methods:")
for method, result in results.items():
    in_domain_pct = np.mean(result.in_domain) * 100
    print(f"  {method:20s}: {in_domain_pct:5.1f}% in-domain")

# Consensus: sample is in-domain if majority of methods agree
consensus_in_domain = np.mean([r.in_domain for r in results.values()], axis=0) > 0.5
print(f"\nConsensus in-domain: {np.sum(consensus_in_domain)} samples")


# =============================================================================
# Example 4: Conformal Prediction
# =============================================================================
print("\n3.3 Conformal Prediction (Distribution-Free Uncertainty)")
print("-" * 80)

from sklearn.ensemble import RandomForestRegressor

y_train = np.random.randn(n_train)
y_calib = np.random.randn(100)
X_calib = np.random.randn(100, n_features)

model = RandomForestRegressor(n_estimators=10, random_state=42)
cp = ad.ConformalPredictor(model, confidence=0.9)
cp.fit(X_train, y_train, X_calib, y_calib)

y_pred, intervals = cp.predict(X_test[:5], return_intervals=True)

print("Sample predictions with 90% confidence intervals:")
for i in range(5):
    print(f"  Sample {i}: {y_pred[i]:6.2f}  "
          f"[{intervals[i,0]:6.2f}, {intervals[i,1]:6.2f}]  "
          f"Width: {intervals[i,1] - intervals[i,0]:.2f}")


# =============================================================================
# Example 5: Prospective Validation
# =============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 4: Prospective Validation")
print("=" * 80)

from evaluation import prospective_validation as pv
from datetime import datetime, timedelta

print("\n4.1 Temporal Validation")
print("-" * 80)

# Simulate time-series data
n_samples = 1000
dates = np.array([
    (datetime(2018, 1, 1) + timedelta(days=i)).isoformat()
    for i in range(n_samples)
])
X = np.random.randn(n_samples, 10)
y = np.sum(X[:, :3], axis=1) + np.random.randn(n_samples) * 0.5

model = RandomForestRegressor(n_estimators=10, random_state=42)
validator = pv.TemporalValidator()

result = validator.validate(
    X, y, dates, model,
    train_until="2020-07-01"
)

print(f"Training set: {result['train_size']} samples")
print(f"Test set: {result['test_size']} samples")
print(f"Training period: up to {result['train_until']}")
print(f"\nProspective performance:")
print(f"  R²: {result['r2']:.4f}")
print(f"  RMSE: {result['rmse']:.4f}")
print(f"  MAE: {result['mae']:.4f}")
print(f"  Spearman ρ: {result['spearman']:.4f}")

print("\n4.2 Hit Rate Analysis")
print("-" * 80)

y_true = np.random.binomial(1, 0.2, 100)
y_pred = np.random.rand(100)
# Simulate good model: actives score higher
y_pred[y_true == 1] += 0.4

stats = pv.calculate_hit_rate(y_true, y_pred, top_k=10)

print(f"Top-10 predictions:")
print(f"  Hit rate: {stats['hit_rate']:.1%}")
print(f"  Hits found: {stats['n_hits']}/{stats['n_tested']}")
print(f"  Enrichment: {stats['enrichment']:.2f}x over random")
print(f"  Expected hits (random): {stats['expected_hits_random']:.1f}")
print(f"  Improvement: +{stats['improvement_over_random']:.1f} hits")


# =============================================================================
# Example 6: Statistical Model Comparison
# =============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 5: Statistical Model Comparison")
print("=" * 80)

from evaluation import significance_testing as st

np.random.seed(42)
n_samples = 100

y_true = np.random.randn(n_samples)
pred_a = y_true + np.random.randn(n_samples) * 0.5  # Model A
pred_b = y_true + np.random.randn(n_samples) * 0.6  # Model B (slightly worse)

print("\n5.1 Paired t-test")
print("-" * 80)

result_ttest = st.paired_ttest(pred_a, pred_b, y_true, metric="squared_error")

print(f"Model A vs Model B:")
print(f"  t-statistic: {result_ttest.statistic:.4f}")
print(f"  p-value: {result_ttest.pvalue:.4f}")
print(f"  Cohen's d (effect size): {result_ttest.effect_size:.4f}")
print(f"  95% CI of difference: [{result_ttest.confidence_interval[0]:.4f}, "
      f"{result_ttest.confidence_interval[1]:.4f}]")
print(f"  Significant: {result_ttest.significant}")

if result_ttest.effect_size is not None:
    d = abs(result_ttest.effect_size)
    if d < 0.2:
        interpretation = "negligible"
    elif d < 0.5:
        interpretation = "small"
    elif d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    print(f"  Effect size interpretation: {interpretation}")

print("\n5.2 Permutation Test (Non-parametric)")
print("-" * 80)

result_perm = st.permutation_test(pred_a, pred_b, y_true, n_permutations=1000)

print(f"Permutation test:")
print(f"  Test statistic: {result_perm.statistic:.4f}")
print(f"  p-value: {result_perm.pvalue:.4f}")
print(f"  Significant: {result_perm.significant}")

print("\n5.3 Multiple Testing Correction")
print("-" * 80)

# Simulate testing multiple features
pvalues = [0.001, 0.01, 0.03, 0.05, 0.10, 0.20, 0.50]

print(f"Original p-values: {pvalues}")

sig_bonf, threshold = st.bonferroni_correction(pvalues, alpha=0.05)
print(f"\nBonferroni correction:")
print(f"  Corrected α: {threshold:.4f}")
print(f"  Significant: {sum(sig_bonf)}/{len(pvalues)}")

sig_holm = st.holm_bonferroni_correction(pvalues, alpha=0.05)
print(f"\nHolm-Bonferroni correction:")
print(f"  Significant: {sum(sig_holm)}/{len(pvalues)}")

sig_bh = st.benjamini_hochberg_fdr(pvalues, fdr=0.05)
print(f"\nBenjamini-Hochberg FDR control:")
print(f"  Significant: {sum(sig_bh)}/{len(pvalues)}")

print("\nComparison:")
for i, p in enumerate(pvalues):
    print(f"  p={p:.3f}: Bonf={sig_bonf[i]}, Holm={sig_holm[i]}, BH={sig_bh[i]}")


# =============================================================================
# Example 7: Complete Validation Pipeline
# =============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 6: Complete Validation Pipeline")
print("=" * 80)

from evaluation import metrics

print("\n6.1 Setup: Generate synthetic molecular dataset")
print("-" * 80)

np.random.seed(42)
n_molecules = 500
n_features = 20

# Simulate molecular descriptors
X_all = np.random.randn(n_molecules, n_features)
# Activity depends on first 3 features + noise
y_all = 2.0 * X_all[:, 0] + 1.5 * X_all[:, 1] - 1.0 * X_all[:, 2]
y_all += np.random.randn(n_molecules) * 0.5

print(f"Dataset size: {n_molecules} molecules")
print(f"Features: {n_features}")
print(f"Target: continuous (e.g., pIC50)")

print("\n6.2 Split Data (Cluster-based)")
print("-" * 80)

train_idx, val_idx, test_idx = benchmarks.cluster_split(
    X_all,
    n_clusters=50,
    test_size=0.2,
    val_size=0.1,
    random_state=42
)

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]

print(f"Training: {len(train_idx)} samples")
print(f"Validation: {len(val_idx)} samples")
print(f"Test: {len(test_idx)} samples")

print("\n6.3 Train Two Models")
print("-" * 80)

from sklearn.linear_model import Ridge

model_rf = RandomForestRegressor(n_estimators=50, random_state=42)
model_ridge = Ridge(alpha=1.0)

model_rf.fit(X_train, y_train)
model_ridge.fit(X_train, y_train)

print("Trained: Random Forest and Ridge Regression")

print("\n6.4 Evaluate on Test Set")
print("-" * 80)

y_pred_rf = model_rf.predict(X_test)
y_pred_ridge = model_ridge.predict(X_test)

metrics_rf = metrics.calculate_all_regression_metrics(y_test, y_pred_rf)
metrics_ridge = metrics.calculate_all_regression_metrics(y_test, y_pred_ridge)

print("Random Forest:")
print(f"  RMSE: {metrics_rf.rmse:.4f}")
print(f"  MAE: {metrics_rf.mae:.4f}")
print(f"  R²: {metrics_rf.r2:.4f}")
print(f"  Spearman ρ: {metrics_rf.spearman:.4f}")

print("\nRidge Regression:")
print(f"  RMSE: {metrics_ridge.rmse:.4f}")
print(f"  MAE: {metrics_ridge.mae:.4f}")
print(f"  R²: {metrics_ridge.r2:.4f}")
print(f"  Spearman ρ: {metrics_ridge.spearman:.4f}")

print("\n6.5 Statistical Comparison")
print("-" * 80)

comparison = st.paired_ttest(y_pred_rf, y_pred_ridge, y_test, metric="squared_error")

print(f"Random Forest vs Ridge:")
print(f"  p-value: {comparison.pvalue:.4f}")
print(f"  Significant difference: {comparison.significant}")

if comparison.significant:
    if np.mean((y_test - y_pred_rf)**2) < np.mean((y_test - y_pred_ridge)**2):
        print("  Winner: Random Forest")
    else:
        print("  Winner: Ridge Regression")
else:
    print("  No significant difference")

print("\n6.6 Applicability Domain")
print("-" * 80)

ad_model = ad.KNNApplicabilityDomain(k=5)
ad_model.fit(X_train)
ad_result = ad_model.predict(X_test)

reliable_mask = ad_result.in_domain
n_reliable = np.sum(reliable_mask)

print(f"Test samples in-domain: {n_reliable}/{len(X_test)} ({n_reliable/len(X_test):.1%})")

# Re-evaluate on reliable predictions only
if n_reliable > 0:
    metrics_rf_reliable = metrics.calculate_all_regression_metrics(
        y_test[reliable_mask],
        y_pred_rf[reliable_mask]
    )

    print(f"\nRandom Forest (reliable predictions only):")
    print(f"  RMSE: {metrics_rf_reliable.rmse:.4f} (vs {metrics_rf.rmse:.4f} all)")
    print(f"  R²: {metrics_rf_reliable.r2:.4f} (vs {metrics_rf.r2:.4f} all)")

print("\n6.7 Summary & Recommendations")
print("-" * 80)

print("\nValidation Results:")
print(f"  ✓ Used cluster-based splitting (no similarity leakage)")
print(f"  ✓ Evaluated two models on independent test set")
print(f"  ✓ Performed statistical comparison")
print(f"  ✓ Defined applicability domain")
print(f"  ✓ Analyzed performance on reliable predictions")

print(f"\nBest Model: Random Forest")
print(f"  Test R²: {metrics_rf.r2:.4f}")
print(f"  Reliable R²: {metrics_rf_reliable.r2:.4f}")
print(f"  Fraction in-domain: {n_reliable/len(X_test):.1%}")

print(f"\nRecommendations:")
print(f"  • Deploy Random Forest for predictions")
print(f"  • Flag predictions outside applicability domain")
print(f"  • Monitor performance on prospective data")
print(f"  • Retrain when performance degrades")


# =============================================================================
# Final Summary
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Key Takeaways")
print("=" * 80)

print("""
1. ALWAYS use scaffold/cluster splits (never random for molecules)
2. ALWAYS check for data leakage
3. ALWAYS assess calibration for probabilistic models
4. ALWAYS define applicability domain
5. ALWAYS perform statistical testing when comparing models
6. ALWAYS correct for multiple comparisons
7. ALWAYS validate prospectively if possible
8. NEVER trust a single random split

Following these practices ensures trustworthy ML models for drug discovery.

For more details, see:
  - README.md for documentation
  - Individual module docstrings for API reference
  - Published literature for theoretical background
""")

print("=" * 80)
print("Examples completed successfully!")
print("=" * 80)
