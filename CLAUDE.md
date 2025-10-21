# CLAUDE.md - ML-Fragment-Optimizer

This file provides guidance to Claude Code when working with the ML-Fragment-Optimizer codebase.

## Project Overview

**ML-Fragment-Optimizer** is a machine learning toolkit for fragment-based drug discovery. It provides multi-task ADMET prediction, QSAR model building, and integration with GCNCMC simulation workflows.

## Technology Stack

- **Language**: Python 3.10+
- **ML Frameworks**: scikit-learn, XGBoost, CatBoost, PyTorch (optional)
- **Chemistry**: RDKit
- **Development**: pytest, black, ruff, mypy
- **Package Management**: pip, conda

## Project Structure

```
ML-Fragment-Optimizer/
├── src/ml_fragment_optimizer/     # Main package
│   ├── models/                    # ADMET prediction models
│   │   ├── admet_predictor.py     # Main predictor class
│   │   └── __init__.py
│   ├── synthesis/                 # Retrosynthesis (placeholder)
│   │   ├── retrosynthesis.py
│   │   └── __init__.py
│   ├── active_learning/           # Active learning (placeholder)
│   │   ├── optimizer.py
│   │   └── __init__.py
│   ├── qsar/                      # QSAR model building
│   │   ├── model_builder.py
│   │   └── __init__.py
│   ├── utils/                     # Utilities
│   │   ├── featurizers.py         # Molecular featurization
│   │   ├── config_loader.py       # Configuration management
│   │   ├── logging_utils.py       # Logging setup
│   │   └── __init__.py
│   └── __init__.py
├── apps/                          # CLI applications
│   ├── train_admet_model.py       # Train models
│   ├── predict_properties.py      # Batch prediction
│   ├── optimize_fragment.py       # Fragment optimization (stub)
│   ├── plan_synthesis.py          # Synthesis planning (stub)
│   ├── active_learning_loop.py    # Active learning (stub)
│   ├── benchmark_models.py        # Model benchmarking (stub)
│   └── __init__.py
├── configs/                       # Configuration files
│   ├── admet_model.yaml
│   ├── active_learning.yaml
│   ├── synthesis_constraints.yaml
│   └── evaluation.yaml
├── data/                          # Data directories
│   ├── datasets/
│   ├── building_blocks/
│   └── benchmarks/
├── models/                        # Trained models
│   └── pretrained/
├── notebooks/                     # Jupyter notebooks
│   └── 01_quick_start.ipynb
├── tests/                         # Unit tests
│   ├── test_featurizers.py
│   └── __init__.py
├── docs/                          # Documentation
├── pyproject.toml                 # Modern packaging config
├── setup.py                       # Backward compatibility
├── requirements.txt               # Core dependencies
├── requirements-dev.txt           # Dev dependencies
├── environment.yml                # Conda environment
├── Makefile                       # Common tasks
├── .gitignore
├── LICENSE
└── README.md
```

## Key Components

### 1. Molecular Featurization (`src/ml_fragment_optimizer/utils/featurizers.py`)

**Purpose**: Convert SMILES to numerical features for ML models

**Key Classes**:
- `MolecularFeaturizer`: Main featurization class
  - Supports: Morgan, MACCS, RDKit, Avalon, AtomPair fingerprints
  - Optional: RDKit molecular descriptors
  - Configurable: radius, n_bits, chirality

**Example Usage**:
```python
featurizer = MolecularFeaturizer(
    fingerprint_type="morgan",
    radius=2,
    n_bits=2048,
    include_descriptors=True
)
features = featurizer.featurize(["CCO", "c1ccccc1"])
```

### 2. ADMET Predictor (`src/ml_fragment_optimizer/models/admet_predictor.py`)

**Purpose**: Multi-task ADMET property prediction

**Key Features**:
- Multi-task learning (predict multiple properties simultaneously)
- Model types: Random Forest, XGBoost, Gradient Boosting
- Uncertainty quantification (for tree-based models)
- Save/load functionality

**Example Usage**:
```python
predictor = ADMETPredictor(
    properties=["solubility", "logp"],
    model_type="xgboost"
)
metrics = predictor.fit(smiles, properties_dict)
predictions = predictor.predict(new_smiles)
```

### 3. QSAR Model Builder (`src/ml_fragment_optimizer/qsar/model_builder.py`)

**Purpose**: Automated QSAR model building with feature selection

**Key Features**:
- Automated feature selection (mutual information, f-test)
- Cross-validation
- Model evaluation metrics

### 4. CLI Applications (`apps/`)

**Fully Implemented**:
- `train_admet_model.py`: Train multi-task ADMET models
- `predict_properties.py`: Batch prediction with uncertainty

**Stubs (for future implementation)**:
- `optimize_fragment.py`: Fragment optimization
- `plan_synthesis.py`: Retrosynthesis planning
- `active_learning_loop.py`: Active learning
- `benchmark_models.py`: Model benchmarking

## CLI Usage

### Training

```bash
# Basic training
mlfrag-train --data admet_data.csv \
             --properties solubility,logp \
             --model-type xgboost \
             --output-dir models/admet_v1

# With configuration file
mlfrag-train --config configs/admet_model.yaml \
             --data admet_data.csv
```

**Input CSV Format**:
```csv
SMILES,solubility,logp,clearance
CCO,-0.77,0.46,12.3
c1ccccc1,-2.13,2.13,8.7
```

### Prediction

```bash
# Basic prediction
mlfrag-predict --model models/admet_v1/admet_model.pkl \
               --input molecules.smi \
               --output predictions.csv

# With uncertainty and outlier detection
mlfrag-predict --model models/admet_v1/admet_model.pkl \
               --input molecules.sdf \
               --output predictions.csv \
               --uncertainty \
               --flag-outliers
```

## Development Workflow

### Setup Development Environment

```bash
# Clone and install
git clone <repo>
cd ML-Fragment-Optimizer

# Create conda environment
conda env create -f environment.yml
conda activate mlfrag

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

### Code Style

- **Black**: Code formatting (line length: 100)
- **Ruff**: Fast linting
- **isort**: Import sorting
- **mypy**: Type checking

```bash
# Format code
make format

# Lint
make lint

# Run tests
make test
```

### Adding New Features

1. **New ADMET Property**:
   - Add to training data CSV
   - Update model configuration
   - Retrain model

2. **New Featurization Method**:
   - Add to `MolecularFeaturizer.AVAILABLE_FINGERPRINTS`
   - Implement in `_get_fingerprint()` method
   - Add tests

3. **New Model Type**:
   - Add to `ADMETPredictor._create_model()`
   - Update CLI argument choices
   - Document in README

## Integration with GCMC Workflow

### Fragment Database Integration

```python
import sqlite3
from ml_fragment_optimizer import ADMETPredictor

# Connect to fragment-manager database
conn = sqlite3.connect("../fragment-manager/fragments.db")

# Load fragments
cursor = conn.execute("SELECT smiles FROM fragments")
smiles = [row[0] for row in cursor.fetchall()]

# Predict properties
predictor = ADMETPredictor.load("models/admet_model.pkl")
predictions = predictor.predict(smiles)

# Update database
for smi, sol in zip(smiles, predictions["solubility"]):
    conn.execute(
        "UPDATE fragments SET solubility=? WHERE smiles=?",
        (float(sol), smi)
    )
conn.commit()
```

### GCNCMC Cluster Analysis

```bash
# 1. Run GCNCMC
cd ../Grand_SACP
./scripts/openmm_gcncmc_production.py ...

# 2. Cluster results
cd ../GCNCMC-Analyzer
./gcncmc-analyzer cluster trajectory.dcd --num-clusters 10

# 3. Predict properties for clusters
cd ../ML-Fragment-Optimizer
mlfrag-predict --model models/admet_model.pkl \
               --input ../GCNCMC-Analyzer/clusters/cluster_centers.smi \
               --output cluster_predictions.csv
```

## Configuration Files

All config files use YAML format. Example structure:

```yaml
# configs/admet_model.yaml
model_type: xgboost
fingerprint_type: morgan
fingerprint_radius: 2
fingerprint_bits: 2048

n_estimators: 100
learning_rate: 0.1

properties:
  - solubility
  - logp
  - clearance

cv_folds: 5
```

## Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_featurizers.py

# With coverage
pytest --cov=ml_fragment_optimizer --cov-report=html

# Parallel execution
pytest -n auto
```

## Common Pitfalls

1. **Invalid SMILES**: Always validate SMILES before featurization
2. **Missing Properties**: Check CSV columns match `--properties` argument
3. **Memory Issues**: Use batch processing for large datasets
4. **Model Loading**: Ensure featurizer configuration matches training
5. **Python Path**: CLI apps add `src/` to path for development use

## Future Development Priorities

### Phase 1 (Current - v0.1.0)
- [x] Core ADMET prediction
- [x] CLI applications (train, predict)
- [x] Basic documentation
- [x] Unit tests for featurizers

### Phase 2 (v0.2.0)
- [ ] Implement fragment optimization logic
- [ ] Retrosynthesis planning
- [ ] Active learning loop
- [ ] Graph neural network models
- [ ] Advanced uncertainty quantification

### Phase 3 (v0.3.0+)
- [ ] Multi-objective optimization
- [ ] Transfer learning
- [ ] Web interface
- [ ] Integration with commercial databases

## Dependencies

### Core
- numpy, pandas, scipy
- scikit-learn, xgboost, catboost
- rdkit (chemistry)
- torch (optional, for future GNNs)

### Development
- pytest, pytest-cov
- black, ruff, isort, mypy
- pre-commit

### Optional
- chemprop (D-MPNN models)
- botorch, gpytorch (Bayesian optimization)
- shap (model interpretability)

## Entry Points

Defined in `pyproject.toml`:

```toml
[project.scripts]
mlfrag-train = "ml_fragment_optimizer.apps.train_admet_model:main"
mlfrag-predict = "ml_fragment_optimizer.apps.predict_properties:main"
mlfrag-optimize = "ml_fragment_optimizer.apps.optimize_fragment:main"
mlfrag-synthesis = "ml_fragment_optimizer.apps.plan_synthesis:main"
mlfrag-active-learning = "ml_fragment_optimizer.apps.active_learning_loop:main"
mlfrag-benchmark = "ml_fragment_optimizer.apps.benchmark_models:main"
```

## Error Handling

- **Invalid SMILES**: Log warning, skip molecule
- **Missing data**: Raise ValueError with clear message
- **Model not fitted**: Raise RuntimeError
- **File not found**: Clear error message with expected path

## Logging

Uses `loguru` for structured logging:

```python
from ml_fragment_optimizer.utils.logging_utils import setup_logger

setup_logger(
    log_level="INFO",
    log_file=Path("training.log")
)
```

## Performance Tips

1. **Batch Processing**: Use `batch_featurize()` for large datasets
2. **Parallel Jobs**: Set `n_jobs=-1` for tree models
3. **Memory**: Process in chunks for very large files
4. **GPU**: PyTorch models will use GPU if available

## Versioning

- Version: 0.1.0 (initial release)
- Semantic versioning: MAJOR.MINOR.PATCH
- Tagged releases on GitHub

## License

MIT License - permissive, allows commercial use

## Contact

- Issues: GitHub Issues
- Email: your.email@example.com
- Documentation: README.md, inline docstrings

---

**When working on this project**:
- Follow PEP 8 and project style guide
- Add type hints to all functions
- Write tests for new features
- Update documentation
- Use configuration files for experiments
- Log important events and errors
