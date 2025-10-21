"""
Synthesis Module for ML-Fragment-Optimizer

This module provides retrosynthesis analysis, synthetic accessibility scoring,
building block availability checking, and comprehensive route scoring for
fragment-based drug design.

Components:
    - retrosynthesis: Retrosynthetic route generation and analysis
    - sa_score: Synthetic accessibility scoring (SAScore algorithm)
    - building_blocks: Commercial availability checking
    - route_scoring: Comprehensive synthesis route evaluation

Author: ML-Fragment-Optimizer Team
"""

from .retrosynthesis import RetrosynthesisAnalyzer, RetrosynthesisRoute
from .sa_score import SAScoreCalculator, calculate_sa_score
from .building_blocks import BuildingBlockChecker, AvailabilityResult
from .route_scoring import RouteScorer, MakeabilityScore

__all__ = [
    'RetrosynthesisAnalyzer',
    'RetrosynthesisRoute',
    'SAScoreCalculator',
    'calculate_sa_score',
    'BuildingBlockChecker',
    'AvailabilityResult',
    'RouteScorer',
    'MakeabilityScore',
]

__version__ = '0.1.0'
