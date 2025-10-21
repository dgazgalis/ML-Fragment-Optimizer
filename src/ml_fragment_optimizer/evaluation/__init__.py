"""
ML-Fragment-Optimizer Evaluation Module

Comprehensive model evaluation, benchmarking, and validation toolkit for
fragment-based drug discovery machine learning models.

This module provides rigorous validation strategies to prevent data leakage,
assess model calibration, define applicability domains, and perform statistical
significance testing.

Modules:
--------
- benchmarks: Scaffold/temporal/cluster-based splits, k-fold CV
- calibration: Reliability diagrams, ECE, calibration methods
- applicability_domain: Conformal prediction, outlier detection
- prospective_validation: Temporal validation, external validation
- metrics: Comprehensive metrics library
- significance_testing: Statistical tests, permutation tests, bootstrapping

Key Features:
-------------
- Scaffold-based splitting to prevent molecular similarity leakage
- Temporal validation for prospective performance estimation
- Conformal prediction for distribution-free uncertainty quantification
- Calibration analysis for trustworthy probability estimates
- Statistical significance testing with multiple comparison correction

Literature:
-----------
- Martin et al. (2012) "Does rational selection of training and test sets
  improve the outcome of QSAR modeling?" J Chem Inf Model
- Chen et al. (2019) "The rise of deep learning in drug discovery" Drug Discov Today
- Sheridan (2013) "Time-Split Cross-Validation as a Method for Estimating the
  Goodness of Prospective Prediction" J Chem Inf Model
- Vovk et al. (2005) "Algorithmic Learning in a Random World" Springer

Example:
--------
>>> from evaluation import benchmarks, metrics, calibration
>>>
>>> # Scaffold-based train/test split
>>> train_idx, test_idx = benchmarks.scaffold_split(smiles_list, test_size=0.2)
>>>
>>> # Train model and evaluate
>>> model.fit(X_train, y_train)
>>> y_pred = model.predict(X_test)
>>>
>>> # Calculate metrics
>>> rmse = metrics.rmse(y_test, y_pred)
>>> r2 = metrics.r2_score(y_test, y_pred)
>>>
>>> # Check calibration
>>> calibration.plot_calibration_curve(y_test, y_pred_proba)
>>> ece = calibration.expected_calibration_error(y_test, y_pred_proba)
"""

from . import benchmarks
from . import calibration
from . import applicability_domain
from . import prospective_validation
from . import metrics
from . import significance_testing

__version__ = "0.1.0"
__all__ = [
    "benchmarks",
    "calibration",
    "applicability_domain",
    "prospective_validation",
    "metrics",
    "significance_testing",
]
