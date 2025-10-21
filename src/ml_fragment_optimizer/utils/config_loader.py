"""
Configuration file loading and validation utilities.
"""

from typing import Any, Dict, Optional
from pathlib import Path
import yaml
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved configuration to {output_path}")


@dataclass
class ADMETModelConfig:
    """Configuration for ADMET model training."""

    # Model architecture
    model_type: str = "random_forest"  # random_forest, xgboost, neural_net, chemprop
    fingerprint_type: str = "morgan"
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    use_descriptors: bool = False

    # Training parameters
    n_estimators: int = 100
    max_depth: Optional[int] = None
    learning_rate: float = 0.1
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1

    # Multi-task settings
    properties: list = field(default_factory=lambda: ["solubility", "logp", "clearance"])
    task_weights: Optional[Dict[str, float]] = None

    # Validation
    cv_folds: int = 5
    early_stopping_rounds: int = 50

    # Output
    save_checkpoints: bool = True
    checkpoint_dir: str = "models/checkpoints"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ADMETModelConfig":
        """Create from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ADMETModelConfig":
        """Load from YAML file."""
        config_dict = load_config(yaml_path)
        return cls.from_dict(config_dict)


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning loop."""

    # Acquisition function
    acquisition_function: str = "uncertainty"  # uncertainty, expected_improvement, thompson
    batch_size: int = 10
    n_iterations: int = 20

    # Model retraining
    retrain_frequency: int = 1  # Retrain every N iterations
    initial_training_size: int = 100

    # Stopping criteria
    max_budget: Optional[int] = None  # Max number of experiments
    convergence_threshold: float = 0.01

    # Exploration-exploitation
    epsilon: float = 0.1  # For epsilon-greedy
    temperature: float = 1.0  # For softmax sampling

    # Integration
    fragment_database_path: Optional[str] = None
    gcmc_output_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ActiveLearningConfig":
        """Create from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ActiveLearningConfig":
        """Load from YAML file."""
        config_dict = load_config(yaml_path)
        return cls.from_dict(config_dict)


@dataclass
class SynthesisConfig:
    """Configuration for synthesis planning."""

    # Retrosynthesis
    max_depth: int = 5
    max_routes: int = 10
    reaction_rules_file: Optional[str] = None

    # Scoring
    route_score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "n_steps": 0.3,
            "building_block_availability": 0.4,
            "reaction_feasibility": 0.3,
        }
    )

    # Building blocks
    building_blocks_database: Optional[str] = None
    commercial_availability_threshold: float = 0.7

    # Filters
    min_synthetic_accessibility: float = 0.3
    max_steps: int = 8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SynthesisConfig":
        """Create from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "SynthesisConfig":
        """Load from YAML file."""
        config_dict = load_config(yaml_path)
        return cls.from_dict(config_dict)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Metrics
    regression_metrics: list = field(
        default_factory=lambda: ["rmse", "mae", "r2", "spearman"]
    )
    classification_metrics: list = field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1", "auroc"]
    )

    # Cross-validation
    cv_strategy: str = "kfold"  # kfold, stratified_kfold, leave_one_out
    n_splits: int = 5

    # Visualization
    generate_plots: bool = True
    plot_format: str = "png"  # png, pdf, svg

    # Output
    results_dir: str = "results/evaluation"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EvaluationConfig":
        """Create from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "EvaluationConfig":
        """Load from YAML file."""
        config_dict = load_config(yaml_path)
        return cls.from_dict(config_dict)


def merge_configs(
    default_config: Dict[str, Any],
    user_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge user config with default config.

    Args:
        default_config: Default configuration
        user_config: User-provided configuration

    Returns:
        Merged configuration (user overrides defaults)
    """
    merged = default_config.copy()

    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged
