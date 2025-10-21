# ML-Fragment-Optimizer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Machine Learning-Driven Fragment Optimization for Drug Discovery**

A comprehensive toolkit for optimizing molecular fragments using machine learning, with integration into GCNCMC simulation workflows. Accelerate fragment-based drug discovery through ADMET prediction, retrosynthesis planning, and active learning.

---

## Features

- **Multi-Task ADMET Prediction**: Train models to predict absorption, distribution, metabolism, excretion, and toxicity properties
- **QSAR Model Building**: Automated feature selection and model validation
- **Retrosynthesis Planning**: Score synthesis routes and evaluate fragment accessibility (coming soon)
- **Active Learning**: Iterative optimization loop for efficient experimental design (coming soon)
- **GCMC Integration**: Direct integration with fragment-manager and Grand_SACP/STORMM-GCMC workflows
- **Uncertainty Quantification**: Estimate prediction confidence for reliable decision-making
- **Batch Processing**: Efficient prediction for large molecular libraries
- **Flexible Featurization**: Multiple fingerprint types and molecular descriptors

---

## Installation

### Quick Start (pip)

```bash
# Clone repository
git clone https://github.com/yourusername/ML-Fragment-Optimizer.git
cd ML-Fragment-Optimizer

# Install in editable mode
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Conda Installation (Recommended)

```bash
# Create conda environment
conda env create -f environment.yml
conda activate mlfrag

# Install package
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Optional Dependencies

```bash
# For advanced ML models (GNNs, Chemprop)
pip install -e ".[ml-advanced]"

# For Bayesian optimization
pip install -e ".[optimization]"

# For model interpretability (SHAP)
pip install -e ".[shap-analysis]"

# For GPU acceleration
pip install -e ".[gpu]"
```

---

## Quick Start

### 1. Train an ADMET Model

```bash
# Train model on your data
mlfrag-train --data admet_data.csv \
             --properties solubility,logp,clearance \
             --model-type xgboost \
             --output-dir models/admet_v1

# Input CSV format:
# SMILES,solubility,logp,clearance
# CCO,-0.77,0.46,12.3
# c1ccccc1,-2.13,2.13,8.7
```

### 2. Predict Properties

```bash
# Predict ADMET properties for new molecules
mlfrag-predict --model models/admet_v1/admet_model.pkl \
               --input new_fragments.smi \
               --output predictions.csv \
               --uncertainty
```

### 3. Optimize Fragments (Coming Soon)

```bash
# Suggest modifications to improve properties
mlfrag-optimize --fragment "c1ccccc1" \
                --target-property solubility \
                --target-value -2.0 \
                --num-suggestions 10
```

### 4. Python API

```python
from ml_fragment_optimizer import ADMETPredictor, MolecularFeaturizer

# Create and train predictor
predictor = ADMETPredictor(
    properties=["solubility", "logp"],
    model_type="random_forest"
)

predictor.fit(smiles_list, properties_dict)

# Make predictions
predictions = predictor.predict(["CCO", "c1ccccc1"])
print(predictions)
# {'solubility': [-0.77, -2.13], 'logp': [0.46, 2.13]}

# Get uncertainty estimates
predictions, uncertainties = predictor.predict(
    ["CCO", "c1ccccc1"],
    return_uncertainty=True
)
```

---

## Integration with GCMC Workflow

ML-Fragment-Optimizer seamlessly integrates with the existing fragment-based drug discovery pipeline:

### Complete Workflow

```bash
# 1. Filter and prepare fragment library
cd ../Fragment-Filter
./fragfilter diversity library.sdf --num-select 100 --output diverse.sdf

# 2. Add fragments to database with parameterization
cd ../fragment-manager
./fragmanager.py add ../Fragment-Filter/diverse.sdf --auto-param

# 3. Train ADMET model on known fragments
cd ../ML-Fragment-Optimizer
mlfrag-train --data fragment_properties.csv \
             --properties binding_affinity,solubility \
             --output-dir models/fragment_admet

# 4. Run GCNCMC simulations
cd ../Grand_SACP
./scripts/openmm_gcncmc_production.py \
    ../protein/protein.pdb \
    ../fragments/fragment.pdb \
    ../templates/fragment.xml

# 5. Analyze GCNCMC results
cd ../GCNCMC-Analyzer
./gcncmc-analyzer analyze ../Grand_SACP/output/trajectory.dcd \
    --topology ../Grand_SACP/protein/protein.pdb

# 6. Predict properties for promising clusters
cd ../ML-Fragment-Optimizer
mlfrag-predict --model models/fragment_admet/admet_model.pkl \
               --input ../GCNCMC-Analyzer/clusters/cluster_centers.smi \
               --output cluster_predictions.csv

# 7. Optimize top-scoring fragments
mlfrag-optimize --fragment "best_fragment_smiles" \
                --target-property binding_affinity \
                --num-suggestions 20

# 8. Validate with new GCNCMC simulations
# (iterate steps 4-7)
```

---

## Command-Line Interface

### Available Commands

| Command | Description |
|---------|-------------|
| `mlfrag-train` | Train multi-task ADMET models |
| `mlfrag-predict` | Batch prediction of properties |
| `mlfrag-optimize` | Suggest fragment modifications |
| `mlfrag-synthesis` | Plan synthesis routes |
| `mlfrag-active-learning` | Run active learning loop |
| `mlfrag-benchmark` | Benchmark model performance |

### Example Usage

```bash
# Training with configuration file
mlfrag-train --config configs/admet_model.yaml --data admet_data.csv

# Prediction with uncertainty and outlier detection
mlfrag-predict --model models/admet_model.pkl \
               --input molecules.sdf \
               --output predictions.csv \
               --uncertainty \
               --flag-outliers
```

---

## Python API Documentation

### Core Classes

#### `MolecularFeaturizer`

```python
from ml_fragment_optimizer.utils.featurizers import MolecularFeaturizer

# Create featurizer
featurizer = MolecularFeaturizer(
    fingerprint_type="morgan",
    radius=2,
    n_bits=2048,
    use_chirality=True,
    include_descriptors=True
)

# Featurize molecules
features = featurizer.featurize(["CCO", "c1ccccc1"])
```

#### `ADMETPredictor`

```python
from ml_fragment_optimizer import ADMETPredictor

# Initialize and train
predictor = ADMETPredictor(
    properties=["solubility", "logp"],
    model_type="xgboost"
)

metrics = predictor.fit(smiles, properties_dict)

# Predict with uncertainty
preds, uncs = predictor.predict(["CCCO"], return_uncertainty=True)

# Save/load
predictor.save("model.pkl")
predictor = ADMETPredictor.load("model.pkl")
```

---

## Configuration Files

Example `configs/admet_model.yaml`:

```yaml
# Model architecture
model_type: xgboost
fingerprint_type: morgan
fingerprint_radius: 2
fingerprint_bits: 2048

# Training parameters
n_estimators: 100
learning_rate: 0.1
test_size: 0.2

# Properties to predict
properties:
  - solubility
  - logp
  - clearance

# Validation
cv_folds: 5
```

---

## Project Structure

```
ML-Fragment-Optimizer/
├── src/ml_fragment_optimizer/
│   ├── models/           # ADMET prediction models
│   ├── synthesis/        # Retrosynthesis planning
│   ├── active_learning/  # Active learning loop
│   ├── qsar/             # QSAR model building
│   └── utils/            # Utilities
├── apps/                 # CLI applications
├── configs/              # Configuration files
├── data/                 # Datasets
├── models/               # Trained models
└── tests/                # Unit tests
```

---

## Development

### Code Style

```bash
# Format code
black src/ apps/ tests/

# Lint
ruff check src/ apps/ tests/

# Type check
mypy src/
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ml_fragment_optimizer --cov-report=html
```

---

## Roadmap

### Current Status (v0.1.0)
- [x] ADMET prediction models (RF, XGBoost, GBM)
- [x] Multiple featurization options
- [x] Batch prediction with uncertainty
- [x] CLI applications
- [x] Python API

### Planned Features (v0.2.0)
- [ ] Retrosynthesis planning
- [ ] Active learning loop
- [ ] Graph neural networks
- [ ] Bayesian optimization
- [ ] SHAP interpretability

---

## Citation

```bibtex
@software{ml_fragment_optimizer,
  title = {ML-Fragment-Optimizer: Machine Learning-Driven Fragment Optimization},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/ML-Fragment-Optimizer}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Related Projects

- **fragment-manager**: Fragment database management
- **Grand_SACP**: GCNCMC simulations with OpenMM
- **GCNCMC-Analyzer**: Post-processing for GCNCMC trajectories
- **Fragment-Filter**: Library filtering and diversity selection

See [CLAUDE.md](../CLAUDE.md) for complete workflow integration.
