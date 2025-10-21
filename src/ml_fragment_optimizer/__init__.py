"""
ML-Fragment-Optimizer: Machine Learning-Driven Fragment Optimization for Drug Discovery

A comprehensive toolkit for:
- Multi-task ADMET property prediction
- Retrosynthesis planning and synthesis route scoring
- Active learning for fragment optimization
- QSAR model building and interpretation
- Integration with GCNCMC simulation workflows
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Make imports optional to avoid breaking if dependencies are missing
__all__ = ["__version__"]

try:
    from ml_fragment_optimizer.models.admet_predictor import ADMETPredictor
    __all__.append("ADMETPredictor")
except ImportError:
    pass

try:
    from ml_fragment_optimizer.qsar.model_builder import QSARModelBuilder
    __all__.append("QSARModelBuilder")
except ImportError:
    pass

try:
    from ml_fragment_optimizer.synthesis.retrosynthesis import RetrosynthesisAnalyzer
    __all__.append("RetrosynthesisAnalyzer")
except ImportError:
    pass

try:
    from ml_fragment_optimizer.active_learning.optimizer import ActiveLearningOptimizer
    __all__.append("ActiveLearningOptimizer")
except ImportError:
    pass

try:
    from ml_fragment_optimizer.utils.featurizers import MolecularFeaturizer
    __all__.append("MolecularFeaturizer")
except ImportError:
    pass
