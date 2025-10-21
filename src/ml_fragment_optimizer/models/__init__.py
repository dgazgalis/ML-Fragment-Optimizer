"""
ML-Fragment-Optimizer Models Module

This module provides state-of-the-art deep learning models for ADMET prediction
with uncertainty quantification.

Main Components:
- ADMETPredictor: Main prediction interface
- MolecularFeaturizer: Convert SMILES to features
- DMPNNModel: Message passing neural network
- Uncertainty quantification methods

Author: Claude Code
Date: 2025-10-20
"""

from .admet_predictor import (
    ADMETPredictor,
    ADMETConfig,
    ADMETDataset,
    FingerprintModel,
    collate_admet_batch,
    ADMET_TASKS
)

from .fingerprints import (
    MolecularFeaturizer,
    MoleculeFeatures
)

from .chemprop_wrapper import (
    DMPNNModel,
    DMPNNEncoder,
    DirectedMessagePassing,
    create_dmpnn_model
)

from .uncertainty import (
    EvidentialOutput,
    evidential_loss,
    nig_uncertainty,
    MCDropoutWrapper,
    DeepEnsemble,
    CalibrationMetrics,
    compute_uncertainty_metrics
)

__all__ = [
    # Main predictor
    'ADMETPredictor',
    'ADMETConfig',
    'ADMETDataset',
    'FingerprintModel',
    'collate_admet_batch',
    'ADMET_TASKS',

    # Featurization
    'MolecularFeaturizer',
    'MoleculeFeatures',

    # D-MPNN
    'DMPNNModel',
    'DMPNNEncoder',
    'DirectedMessagePassing',
    'create_dmpnn_model',

    # Uncertainty
    'EvidentialOutput',
    'evidential_loss',
    'nig_uncertainty',
    'MCDropoutWrapper',
    'DeepEnsemble',
    'CalibrationMetrics',
    'compute_uncertainty_metrics',
]

__version__ = '0.1.0'
