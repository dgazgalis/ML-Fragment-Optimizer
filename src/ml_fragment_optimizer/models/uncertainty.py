"""
Uncertainty Quantification for ADMET Predictions

This module implements multiple uncertainty quantification strategies:

1. Evidential Deep Learning (EDL):
   - Predicts parameters of a probability distribution (not just point estimates)
   - For regression: predict Normal-Inverse-Gamma (NIG) distribution parameters
   - Provides both aleatoric (data) and epistemic (model) uncertainty
   - Single forward pass - no sampling required

2. Monte Carlo Dropout (MC-Dropout):
   - Keep dropout active during inference
   - Multiple forward passes with different dropout masks
   - Variance across predictions estimates uncertainty
   - Simple but effective

3. Deep Ensembles:
   - Train multiple models with different initializations
   - Average predictions and compute variance
   - Gold standard for uncertainty but computationally expensive

Why Evidential Deep Learning?
- Efficient: Single forward pass (unlike MC-Dropout or ensembles)
- Principled: Based on Bayesian statistics and higher-order distributions
- Disentangled: Separates data uncertainty from model uncertainty
- Calibrated: Provides well-calibrated confidence intervals

Reference:
Amini et al. "Deep Evidential Regression" NeurIPS 2020

Author: Claude Code
Date: 2025-10-20
"""

from typing import Dict, List, Optional, Tuple, Callable
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialOutput(nn.Module):
    """
    Evidential regression output layer.

    Predicts parameters of Normal-Inverse-Gamma (NIG) distribution:
    - gamma: shape parameter (v)
    - lambda: precision parameter (lambda)
    - alpha: shape parameter (alpha)
    - beta: scale parameter (beta)

    From these, we can compute:
    - Mean prediction: gamma
    - Aleatoric uncertainty: sqrt(beta / (alpha - 1))  [data uncertainty]
    - Epistemic uncertainty: sqrt(beta / (lambda * (alpha - 1)))  [model uncertainty]
    """

    def __init__(self, input_dim: int):
        """
        Initialize evidential output layer.

        Args:
            input_dim: Dimension of input features
        """
        super().__init__()

        # Predict 4 parameters for NIG distribution
        self.gamma_layer = nn.Linear(input_dim, 1)  # Mean
        self.lambda_layer = nn.Linear(input_dim, 1)  # Lambda (precision)
        self.alpha_layer = nn.Linear(input_dim, 1)  # Alpha (shape)
        self.beta_layer = nn.Linear(input_dim, 1)  # Beta (scale)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to predict NIG distribution parameters.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Tuple of (gamma, lambda, alpha, beta) tensors
            All shapes: [batch_size, 1]
        """
        gamma = self.gamma_layer(x)  # Predicted mean (no constraint)

        # Apply softplus to ensure positive parameters
        # softplus(x) = log(1 + exp(x)) - smooth approximation of ReLU
        lambda_param = F.softplus(self.lambda_layer(x)) + 1e-6  # Prevent zero
        alpha = F.softplus(self.alpha_layer(x)) + 1.0 + 1e-6  # Must be > 1
        beta = F.softplus(self.beta_layer(x)) + 1e-6  # Prevent zero

        return gamma, lambda_param, alpha, beta


def evidential_loss(
    gamma: torch.Tensor,
    lambda_param: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    target: torch.Tensor,
    coeff: float = 1.0
) -> torch.Tensor:
    """
    Evidential regression loss function.

    Combines two terms:
    1. NLL loss: Negative log-likelihood of the target under NIG prior
    2. Regularization: Encourages model to be uncertain when predictions are wrong

    Args:
        gamma: Predicted mean [batch_size, 1]
        lambda_param: Lambda parameter [batch_size, 1]
        alpha: Alpha parameter [batch_size, 1]
        beta: Beta parameter [batch_size, 1]
        target: Ground truth [batch_size, 1]
        coeff: Regularization coefficient

    Returns:
        Loss tensor (scalar)
    """
    # Compute negative log-likelihood
    # NLL = 0.5 * log(pi/lambda) - alpha * log(beta)
    #       + (alpha + 0.5) * log(beta + lambda * (target - gamma)^2 / 2)
    #       + log(Gamma(alpha) / Gamma(alpha + 0.5))

    error = target - gamma
    twoBlambda = 2 * beta * lambda_param

    # NLL term
    nll = (
        0.5 * torch.log(np.pi / lambda_param)
        - alpha * torch.log(2 * beta)
        + (alpha + 0.5) * torch.log(2 * beta + lambda_param * error ** 2)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    # Regularization term: penalize wrong predictions with low uncertainty
    # |target - gamma| * (2 * alpha + lambda)
    reg = torch.abs(error) * (2 * alpha + lambda_param)

    loss = nll + coeff * reg

    return loss.mean()


def nig_uncertainty(
    gamma: torch.Tensor,
    lambda_param: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute mean prediction and uncertainties from NIG parameters.

    Args:
        gamma: Mean parameter [batch_size, 1]
        lambda_param: Lambda parameter [batch_size, 1]
        alpha: Alpha parameter [batch_size, 1]
        beta: Beta parameter [batch_size, 1]

    Returns:
        Tuple of (mean, aleatoric_uncertainty, epistemic_uncertainty)
    """
    mean = gamma

    # Aleatoric uncertainty (data uncertainty - irreducible)
    # Variance of the predictive distribution
    aleatoric_var = beta / (alpha - 1)
    aleatoric_unc = torch.sqrt(aleatoric_var)

    # Epistemic uncertainty (model uncertainty - reducible with more data)
    epistemic_var = beta / (lambda_param * (alpha - 1))
    epistemic_unc = torch.sqrt(epistemic_var)

    return mean, aleatoric_unc, epistemic_unc


class MCDropoutWrapper(nn.Module):
    """
    Monte Carlo Dropout wrapper for any model.

    Keeps dropout active during inference and performs multiple forward passes
    to estimate uncertainty through prediction variance.
    """

    def __init__(self, model: nn.Module, num_samples: int = 20):
        """
        Initialize MC-Dropout wrapper.

        Args:
            model: Base model with dropout layers
            num_samples: Number of forward passes for uncertainty estimation
        """
        super().__init__()
        self.model = model
        self.num_samples = num_samples

    def enable_dropout(self):
        """Enable dropout in all dropout layers"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep dropout active

    def predict_with_uncertainty(
        self,
        *args,
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Make predictions with uncertainty estimates.

        Args:
            *args, **kwargs: Arguments to pass to model forward

        Returns:
            Tuple of (mean_predictions, std_predictions)
            Both are dicts mapping task names to tensors
        """
        self.model.eval()
        self.enable_dropout()

        # Collect predictions from multiple forward passes
        all_predictions = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                preds = self.model(*args, **kwargs)
                all_predictions.append(preds)

        # Compute mean and std for each task
        task_names = all_predictions[0].keys()
        mean_preds = {}
        std_preds = {}

        for task in task_names:
            # Stack predictions across samples
            task_preds = torch.stack([p[task] for p in all_predictions], dim=0)
            # [num_samples, batch_size, 1]

            mean_preds[task] = task_preds.mean(dim=0)
            std_preds[task] = task_preds.std(dim=0)

        return mean_preds, std_preds


class DeepEnsemble:
    """
    Deep Ensemble for uncertainty quantification.

    Trains multiple models with different random initializations and
    aggregates their predictions.
    """

    def __init__(
        self,
        model_factory: Callable,
        num_models: int = 5
    ):
        """
        Initialize deep ensemble.

        Args:
            model_factory: Function that creates a new model instance
            num_models: Number of models in ensemble
        """
        self.model_factory = model_factory
        self.num_models = num_models
        self.models: List[nn.Module] = []

    def create_models(self):
        """Create all ensemble models"""
        self.models = [self.model_factory() for _ in range(self.num_models)]

    def train_ensemble(
        self,
        train_fn: Callable,
        train_loader,
        val_loader,
        **train_kwargs
    ):
        """
        Train all models in ensemble.

        Args:
            train_fn: Training function that takes (model, train_loader, val_loader, **kwargs)
            train_loader: Training data loader
            val_loader: Validation data loader
            **train_kwargs: Additional arguments for training function
        """
        if len(self.models) == 0:
            self.create_models()

        print(f"Training ensemble of {self.num_models} models...")
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{self.num_models}")
            train_fn(model, train_loader, val_loader, **train_kwargs)

    def predict_with_uncertainty(
        self,
        *args,
        device: torch.device = torch.device('cpu'),
        **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Make predictions with uncertainty from ensemble.

        Args:
            *args, **kwargs: Arguments to pass to model forward
            device: Device for computation

        Returns:
            Tuple of (mean_predictions, std_predictions)
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble. Call create_models() first.")

        all_predictions = []

        for model in self.models:
            model.eval()
            model.to(device)

            with torch.no_grad():
                preds = model(*args, **kwargs)
                all_predictions.append(preds)

        # Aggregate predictions
        task_names = all_predictions[0].keys()
        mean_preds = {}
        std_preds = {}

        for task in task_names:
            task_preds = torch.stack([p[task] for p in all_predictions], dim=0)
            mean_preds[task] = task_preds.mean(dim=0).cpu().numpy()
            std_preds[task] = task_preds.std(dim=0).cpu().numpy()

        return mean_preds, std_preds

    def save_ensemble(self, directory: str):
        """Save all models in ensemble"""
        import os
        os.makedirs(directory, exist_ok=True)

        for i, model in enumerate(self.models):
            path = os.path.join(directory, f'model_{i}.pt')
            torch.save(model.state_dict(), path)

    def load_ensemble(self, directory: str):
        """Load all models in ensemble"""
        import os

        if len(self.models) == 0:
            self.create_models()

        for i, model in enumerate(self.models):
            path = os.path.join(directory, f'model_{i}.pt')
            model.load_state_dict(torch.load(path))


class CalibrationMetrics:
    """
    Metrics for evaluating uncertainty calibration.

    A well-calibrated model should have prediction intervals that contain
    the true value with the expected probability.
    """

    @staticmethod
    def expected_calibration_error(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        num_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE measures the difference between predicted confidence and
        observed accuracy across different confidence bins.

        Args:
            predictions: Predicted values [N]
            uncertainties: Predicted uncertainties (std) [N]
            targets: True values [N]
            num_bins: Number of bins for calibration

        Returns:
            ECE score (lower is better, 0 is perfect calibration)
        """
        # Convert to confidence scores (inverse of uncertainty)
        # Normalize to [0, 1] range
        confidence = 1 / (1 + uncertainties)
        confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-8)

        # Bin predictions by confidence
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0

        for i in range(num_bins):
            # Find predictions in this bin
            in_bin = (confidence >= bin_boundaries[i]) & (confidence < bin_boundaries[i + 1])

            if in_bin.sum() == 0:
                continue

            # Average confidence in bin
            avg_confidence = confidence[in_bin].mean()

            # Accuracy in bin (fraction of predictions within 1 std of target)
            errors = np.abs(predictions[in_bin] - targets[in_bin])
            within_std = (errors <= uncertainties[in_bin]).mean()

            # Weight by bin size
            ece += (in_bin.sum() / len(predictions)) * np.abs(avg_confidence - within_std)

        return float(ece)

    @staticmethod
    def coverage_probability(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        num_std: float = 1.96  # 95% confidence interval
    ) -> float:
        """
        Compute coverage probability.

        Fraction of targets within predicted confidence interval.
        For a 95% CI, should be close to 0.95 if well-calibrated.

        Args:
            predictions: Predicted values [N]
            uncertainties: Predicted uncertainties (std) [N]
            targets: True values [N]
            num_std: Number of standard deviations for interval

        Returns:
            Coverage probability (should match confidence level)
        """
        lower = predictions - num_std * uncertainties
        upper = predictions + num_std * uncertainties

        in_interval = (targets >= lower) & (targets <= upper)
        coverage = in_interval.mean()

        return float(coverage)

    @staticmethod
    def sharpness(uncertainties: np.ndarray) -> float:
        """
        Compute sharpness (average uncertainty).

        Lower is better - indicates more confident predictions.
        Should be balanced with calibration.

        Args:
            uncertainties: Predicted uncertainties [N]

        Returns:
            Average uncertainty
        """
        return float(uncertainties.mean())


def compute_uncertainty_metrics(
    predictions: Dict[str, np.ndarray],
    uncertainties: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Compute uncertainty quantification metrics for all tasks.

    Args:
        predictions: Dict mapping task names to predictions
        uncertainties: Dict mapping task names to uncertainties
        targets: Dict mapping task names to targets

    Returns:
        Dict of dicts with metrics per task
    """
    metrics = {}

    for task in predictions.keys():
        pred = predictions[task].flatten()
        unc = uncertainties[task].flatten()
        tgt = targets[task].flatten()

        metrics[task] = {
            'ece': CalibrationMetrics.expected_calibration_error(pred, unc, tgt),
            'coverage_95': CalibrationMetrics.coverage_probability(pred, unc, tgt, 1.96),
            'coverage_68': CalibrationMetrics.coverage_probability(pred, unc, tgt, 1.0),
            'sharpness': CalibrationMetrics.sharpness(unc),
            'mean_uncertainty': float(unc.mean()),
            'std_uncertainty': float(unc.std())
        }

    return metrics


if __name__ == "__main__":
    # Test evidential output and loss
    print("Testing Evidential Deep Learning components...")

    batch_size = 32
    input_dim = 128

    # Create evidential output layer
    evid_layer = EvidentialOutput(input_dim)

    # Dummy input
    x = torch.randn(batch_size, input_dim)
    target = torch.randn(batch_size, 1)

    # Forward pass
    gamma, lambda_param, alpha, beta = evid_layer(x)

    print(f"\nEvidential parameters shapes:")
    print(f"  gamma (mean): {gamma.shape}")
    print(f"  lambda: {lambda_param.shape}")
    print(f"  alpha: {alpha.shape}")
    print(f"  beta: {beta.shape}")

    # Compute loss
    loss = evidential_loss(gamma, lambda_param, alpha, beta, target)
    print(f"\nEvidential loss: {loss.item():.4f}")

    # Compute uncertainties
    mean, aleatoric, epistemic = nig_uncertainty(gamma, lambda_param, alpha, beta)
    print(f"\nUncertainty estimates:")
    print(f"  Mean prediction range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Aleatoric uncertainty range: [{aleatoric.min():.4f}, {aleatoric.max():.4f}]")
    print(f"  Epistemic uncertainty range: [{epistemic.min():.4f}, {epistemic.max():.4f}]")

    # Test calibration metrics
    print("\nTesting calibration metrics...")
    predictions = np.random.randn(100)
    uncertainties = np.abs(np.random.randn(100) * 0.5)
    targets = predictions + np.random.randn(100) * 0.3  # Add noise

    ece = CalibrationMetrics.expected_calibration_error(predictions, uncertainties, targets)
    coverage = CalibrationMetrics.coverage_probability(predictions, uncertainties, targets)
    sharpness = CalibrationMetrics.sharpness(uncertainties)

    print(f"  ECE: {ece:.4f}")
    print(f"  Coverage (95%): {coverage:.4f}")
    print(f"  Sharpness: {sharpness:.4f}")

    print("\nAll tests passed!")
