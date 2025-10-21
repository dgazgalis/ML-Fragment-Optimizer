"""
QSAR/SAR Analysis Module for ML-Fragment-Optimizer

This module provides comprehensive tools for Structure-Activity Relationship (SAR)
analysis, Matched Molecular Pair Analysis (MMPA), and model interpretation.

Components:
- mmpa: Matched Molecular Pair Analysis for identifying transformations
- free_wilson: Free-Wilson additive model for SAR decomposition
- activity_cliffs: Activity cliff detection and analysis
- sar_visualization: Publication-quality SAR visualizations
- feature_importance: Model interpretation and explainability
- bioisostere_suggester: Context-aware bioisosteric replacement suggestions

Example:
    >>> from qsar import mmpa, activity_cliffs, sar_visualization
    >>> from rdkit import Chem
    >>>
    >>> # Analyze matched pairs
    >>> mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    >>> pairs = mmpa.find_matched_pairs(mols, activities)
    >>>
    >>> # Detect activity cliffs
    >>> cliffs = activity_cliffs.detect_cliffs(mols, activities)
    >>>
    >>> # Visualize SAR landscape
    >>> sar_visualization.plot_activity_landscape(mols, activities)

References:
    - Hussain & Rea (2010) "Computationally Efficient Algorithm to Identify
      Matched Molecular Pairs (MMPs) in Large Data Sets" J. Chem. Inf. Model.
    - Grieco et al. (2017) "Matched Molecular Pairs and Activity Cliffs"
      Methods Mol. Biol.
    - Free & Wilson (1964) "A Mathematical Contribution to Structure-Activity Studies"
      J. Med. Chem.
"""

__version__ = "1.0.0"
__author__ = "ML-Fragment-Optimizer Development Team"

from . import mmpa
from . import free_wilson
from . import activity_cliffs
from . import sar_visualization
from . import feature_importance
from . import bioisostere_suggester

__all__ = [
    "mmpa",
    "free_wilson",
    "activity_cliffs",
    "sar_visualization",
    "feature_importance",
    "bioisostere_suggester",
]
