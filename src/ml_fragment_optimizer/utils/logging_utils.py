"""
Structured logging utilities.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
    format_string: Optional[str] = None,
) -> None:
    """
    Configure loguru logger with console and file handlers.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (None = console only)
        rotation: Log rotation policy (e.g., "10 MB", "1 day")
        retention: Log retention policy (e.g., "30 days", "1 week")
        format_string: Custom format string
    """
    # Remove default handler
    logger.remove()

    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Console handler with colors
    logger.add(
        sys.stderr,
        format=format_string,
        level=log_level,
        colorize=True,
    )

    # File handler if requested
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format=format_string,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )
        logger.info(f"Logging to file: {log_file}")


def log_experiment_start(experiment_name: str, config: dict) -> None:
    """
    Log experiment start with configuration.

    Args:
        experiment_name: Name of experiment
        config: Configuration dictionary
    """
    logger.info("=" * 80)
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info("=" * 80)
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)


def log_experiment_end(experiment_name: str, results: dict) -> None:
    """
    Log experiment end with results.

    Args:
        experiment_name: Name of experiment
        results: Results dictionary
    """
    logger.info("=" * 80)
    logger.info(f"Experiment completed: {experiment_name}")
    logger.info("=" * 80)
    logger.info("Results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)


class LoggerContext:
    """Context manager for temporary logger configuration."""

    def __init__(self, log_level: str = "INFO", log_file: Optional[Path] = None):
        self.log_level = log_level
        self.log_file = log_file
        self.handler_id = None

    def __enter__(self):
        if self.log_file is not None:
            self.handler_id = logger.add(
                self.log_file,
                level=self.log_level,
                rotation="10 MB",
                retention="30 days",
            )
        return logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler_id is not None:
            logger.remove(self.handler_id)
