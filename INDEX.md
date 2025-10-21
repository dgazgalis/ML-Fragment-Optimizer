# ML-Fragment-Optimizer - Complete File Index

## Core Package Files (src/ml_fragment_optimizer/)

### Main Package
- `src/ml_fragment_optimizer/__init__.py` - Package initialization, exports main classes

### Models Module (src/ml_fragment_optimizer/models/)
- `__init__.py` - Models module init
- `admet_predictor.py` - **COMPLETE** Multi-task ADMET predictor (200 lines)
  - Classes: `ADMETPredictor`
  - Features: Train, predict, uncertainty, save/load
  - Models: RandomForest, XGBoost, GradientBoosting

### QSAR Module (src/ml_fragment_optimizer/qsar/)
- `__init__.py` - QSAR module init
- `model_builder.py` - **COMPLETE** Automated QSAR builder (150 lines)
  - Classes: `QSARModelBuilder`
  - Features: Feature selection, cross-validation, automated training

### Utils Module (src/ml_fragment_optimizer/utils/)
- `__init__.py` - Utils module init
- `featurizers.py` - **COMPLETE** Molecular featurization (250 lines)
  - Classes: `MolecularFeaturizer`
  - Functions: `calculate_basic_properties()`, `batch_featurize()`, etc.
  - Fingerprints: Morgan, MACCS, RDKit, Avalon, AtomPair
- `config_loader.py` - **COMPLETE** Configuration management (200 lines)
  - Classes: `ADMETModelConfig`, `ActiveLearningConfig`, `SynthesisConfig`, `EvaluationConfig`
  - Functions: `load_config()`, `save_config()`, `merge_configs()`
- `logging_utils.py` - **COMPLETE** Structured logging (90 lines)
  - Functions: `setup_logger()`, `log_experiment_start()`, `log_experiment_end()`
  - Classes: `LoggerContext`

### Active Learning Module (src/ml_fragment_optimizer/active_learning/)
- `__init__.py` - Active learning module init
- `optimizer.py` - **STUB** Active learning optimizer (placeholder)

### Synthesis Module (src/ml_fragment_optimizer/synthesis/)
- `__init__.py` - Synthesis module init
- `retrosynthesis.py` - **STUB** Retrosynthesis planner (placeholder)

### Integration Module (src/ml_fragment_optimizer/integration/)
- `__init__.py` - Integration module init (empty, for future GCMC integration)

### Evaluation Module (src/ml_fragment_optimizer/evaluation/)
- `__init__.py` - Evaluation module init (empty, for future benchmarking)

## CLI Applications (apps/)

### Fully Implemented
- `apps/__init__.py` - Apps package init
- `apps/train_admet_model.py` - **COMPLETE** Train ADMET models CLI (250 lines)
  - Entry point: `mlfrag-train`
  - Features: Full training pipeline, config support, validation
- `apps/predict_properties.py` - **COMPLETE** Batch prediction CLI (200 lines)
  - Entry point: `mlfrag-predict`
  - Features: Multiple input formats, uncertainty, outlier detection

### Stubs (Future Implementation)
- `apps/optimize_fragment.py` - Fragment optimization (stub)
  - Entry point: `mlfrag-optimize`
- `apps/plan_synthesis.py` - Synthesis planning (stub)
  - Entry point: `mlfrag-synthesis`
- `apps/active_learning_loop.py` - Active learning loop (stub)
  - Entry point: `mlfrag-active-learning`
- `apps/benchmark_models.py` - Model benchmarking (stub)
  - Entry point: `mlfrag-benchmark`

## Configuration Files (configs/)

- `configs/admet_model.yaml` - ADMET model training configuration
  - Model type, fingerprints, hyperparameters, properties
- `configs/active_learning.yaml` - Active learning configuration
  - Acquisition function, iterations, integration paths
- `configs/synthesis_constraints.yaml` - Synthesis planning configuration
  - Retrosynthesis settings, route scoring, filters
- `configs/evaluation.yaml` - Model evaluation configuration
  - Metrics, cross-validation, visualization

## Tests (tests/)

- `tests/__init__.py` - Tests package init
- `tests/test_featurizers.py` - **COMPLETE** Featurizer tests (150 lines)
  - Test classes: `TestMolecularFeaturizer`, `TestBasicProperties`, `TestUtilityFunctions`
  - Coverage: All fingerprint types, descriptors, batch processing, error handling

## Notebooks (notebooks/)

- `notebooks/01_quick_start.ipynb` - Quick start tutorial notebook
  - Sections: Featurization, training, prediction, uncertainty, save/load

## Setup and Configuration

### Packaging
- `pyproject.toml` - **Modern Python packaging** (PEP 621)
  - Metadata, dependencies, optional features, scripts, tools config
- `setup.py` - Backward compatibility wrapper
- `MANIFEST.in` - Package data inclusion rules

### Dependencies
- `requirements.txt` - Core dependencies
- `requirements-dev.txt` - Development dependencies
- `environment.yml` - Conda environment specification

### Development Tools
- `.gitignore` - Git ignore rules (comprehensive)
- `Makefile` - Development task automation
  - Targets: install, test, format, lint, clean, docs
- `LICENSE` - MIT License

## Documentation

### User Documentation
- `README.md` - Existing (D-MPNN focused, comprehensive)
- `README_NEW.md` - New (GCMC workflow focused, comprehensive)
  - **Note**: Decision needed on which to use as primary

### Developer Documentation
- `CLAUDE.md` - **Developer guidance** for Claude Code
  - Project structure, components, workflows, integration points
- `PROJECT_SUMMARY.md` - **Complete project summary**
  - What was created, features, statistics, next steps
- `INDEX.md` - **This file** - Complete file index

## Data Directories (data/)

- `data/datasets/.gitkeep` - Training datasets (empty, tracked)
- `data/building_blocks/.gitkeep` - Building blocks library (empty, tracked)
- `data/benchmarks/.gitkeep` - Benchmark datasets (empty, tracked)

## Model Storage (models/)

- `models/pretrained/.gitkeep` - Trained models storage (empty, tracked)

## Documentation (docs/)

- Currently empty, ready for Sphinx documentation

## File Statistics

### Python Source Files
```
src/ml_fragment_optimizer/
├── models/admet_predictor.py         200 lines (COMPLETE)
├── qsar/model_builder.py             150 lines (COMPLETE)
├── utils/featurizers.py              250 lines (COMPLETE)
├── utils/config_loader.py            200 lines (COMPLETE)
├── utils/logging_utils.py             90 lines (COMPLETE)
├── active_learning/optimizer.py       40 lines (STUB)
└── synthesis/retrosynthesis.py        40 lines (STUB)

apps/
├── train_admet_model.py              250 lines (COMPLETE)
├── predict_properties.py             200 lines (COMPLETE)
├── optimize_fragment.py               70 lines (STUB)
├── plan_synthesis.py                  60 lines (STUB)
├── active_learning_loop.py            70 lines (STUB)
└── benchmark_models.py                50 lines (STUB)

tests/
└── test_featurizers.py               150 lines (COMPLETE)

Total: ~1,820 lines of Python code
```

### Configuration Files
```
pyproject.toml                        230 lines (COMPLETE)
configs/*.yaml                        120 lines (4 files, COMPLETE)
environment.yml                        80 lines (COMPLETE)
requirements*.txt                     150 lines (3 files, COMPLETE)
```

### Documentation Files
```
README.md                             450 lines (existing)
README_NEW.md                         400 lines (new)
CLAUDE.md                             400 lines (COMPLETE)
PROJECT_SUMMARY.md                    450 lines (COMPLETE)
INDEX.md                              250 lines (this file)
```

## Entry Points (Command-Line Tools)

Installed after `pip install -e .`:

1. `mlfrag-train` → `apps/train_admet_model.py:main()`
2. `mlfrag-predict` → `apps/predict_properties.py:main()`
3. `mlfrag-optimize` → `apps/optimize_fragment.py:main()` (stub)
4. `mlfrag-synthesis` → `apps/plan_synthesis.py:main()` (stub)
5. `mlfrag-active-learning` → `apps/active_learning_loop.py:main()` (stub)
6. `mlfrag-benchmark` → `apps/benchmark_models.py:main()` (stub)

## Import Structure

### Main Package Imports
```python
from ml_fragment_optimizer import (
    ADMETPredictor,           # models/admet_predictor.py
    QSARModelBuilder,         # qsar/model_builder.py
    RetrosynthesisPlanner,    # synthesis/retrosynthesis.py (stub)
    ActiveLearningOptimizer,  # active_learning/optimizer.py (stub)
    MolecularFeaturizer,      # utils/featurizers.py
)
```

### Utility Imports
```python
from ml_fragment_optimizer.utils.featurizers import (
    MolecularFeaturizer,
    calculate_basic_properties,
    smiles_to_mol_safe,
    batch_featurize,
)

from ml_fragment_optimizer.utils.config_loader import (
    ADMETModelConfig,
    ActiveLearningConfig,
    SynthesisConfig,
    EvaluationConfig,
    load_config,
    save_config,
)

from ml_fragment_optimizer.utils.logging_utils import (
    setup_logger,
    log_experiment_start,
    log_experiment_end,
    LoggerContext,
)
```

## Development Status

### ✅ Complete and Working
- Project structure
- Package configuration (pyproject.toml)
- Molecular featurization (all fingerprint types)
- ADMET predictor (multi-task, uncertainty)
- QSAR model builder
- Configuration management
- Logging utilities
- CLI training application
- CLI prediction application
- Unit tests for featurizers
- Example configurations
- Documentation

### 🚧 Stub/Placeholder
- Fragment optimization logic
- Retrosynthesis planning
- Active learning optimizer
- Model benchmarking
- Synthesis integration
- GCMC integration utilities

### 📋 Planned (Not Started)
- Graph neural networks
- Bayesian optimization
- Transfer learning
- Multi-objective optimization
- Web interface
- Advanced uncertainty methods
- Model interpretability (SHAP)
- Comprehensive test suite
- API documentation (Sphinx)

## Quick Reference

### Installation
```bash
cd ML-Fragment-Optimizer
pip install -e .                    # Basic
pip install -e ".[dev]"             # Development
pip install -e ".[all]"             # All features
```

### Basic Usage
```bash
# Train model
mlfrag-train --data data.csv --properties solubility,logp --output-dir models/

# Predict
mlfrag-predict --model models/admet_model.pkl --input molecules.smi --output predictions.csv
```

### Python API
```python
from ml_fragment_optimizer import ADMETPredictor

predictor = ADMETPredictor(properties=["solubility", "logp"])
predictor.fit(smiles, properties_dict)
predictions = predictor.predict(new_smiles)
```

### Testing
```bash
pytest                              # Run all tests
pytest tests/test_featurizers.py   # Specific test
pytest --cov=ml_fragment_optimizer # With coverage
```

### Development
```bash
make format                         # Format code
make lint                           # Run linters
make test                           # Run tests
make clean                          # Clean artifacts
```

## Integration Points

### With fragment-manager
- Load fragments from SQLite database
- Predict properties for fragment library
- Update database with predictions

### With GCNCMC-Analyzer
- Predict properties for cluster centers
- Analyze fragment binding preferences
- Prioritize fragments for synthesis

### With Grand_SACP
- Pre-screen fragments by predicted properties
- Post-analysis of GCNCMC results
- Iterative fragment optimization

## Notes

1. **README Decision**: Two README files exist - choose one as primary
2. **Stub Modules**: Several modules are placeholders for future work
3. **Type Hints**: All code includes comprehensive type hints
4. **Modern Packaging**: Uses pyproject.toml (PEP 621)
5. **Code Quality**: Configured for black, ruff, isort, mypy

## Contact

For questions about specific files or components:
- General: See `README_NEW.md`
- Development: See `CLAUDE.md`
- Overview: See `PROJECT_SUMMARY.md`
- This index: See `INDEX.md`
