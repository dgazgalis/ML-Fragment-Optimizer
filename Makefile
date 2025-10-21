.PHONY: help install install-dev test lint format clean docs

help:
	@echo "ML-Fragment-Optimizer - Makefile commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install       Install package"
	@echo "  make install-dev   Install with development dependencies"
	@echo "  make install-all   Install with all optional dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make format        Format code with black and isort"
	@echo "  make lint          Run linters (ruff, mypy)"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean         Remove build artifacts and cache"
	@echo "  make clean-data    Remove data files"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

format:
	@echo "Formatting code with black..."
	black src/ apps/ tests/
	@echo "Sorting imports with isort..."
	isort src/ apps/ tests/

lint:
	@echo "Linting with ruff..."
	ruff check src/ apps/ tests/
	@echo "Type checking with mypy..."
	mypy src/

test:
	@echo "Running tests..."
	pytest -v

test-cov:
	@echo "Running tests with coverage..."
	pytest --cov=ml_fragment_optimizer --cov-report=html --cov-report=term-missing

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info
	rm -rf src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage

clean-data:
	@echo "Cleaning data files..."
	rm -rf data/datasets/*.csv data/datasets/*.sdf
	rm -rf models/pretrained/*.pkl models/pretrained/*.pt
	rm -rf logs/*.log

docs:
	@echo "Building documentation..."
	cd docs && make html
	@echo "Documentation built in docs/_build/html/"

# Training shortcuts
train-example:
	@echo "Training example ADMET model..."
	mlfrag-train --config configs/admet_model.yaml --data data/datasets/example.csv

predict-example:
	@echo "Running example prediction..."
	mlfrag-predict --model models/admet/admet_model.pkl --input data/datasets/test.smi --output predictions.csv

# Development workflow
dev-setup: install-dev
	@echo "Setting up pre-commit hooks..."
	pre-commit install
	@echo "Development environment ready!"

check: format lint test
	@echo "All checks passed!"
