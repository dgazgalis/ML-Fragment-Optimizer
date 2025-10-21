# ML-Fragment-Optimizer

A comprehensive machine learning framework for fragment-based drug design, combining ADMET property prediction with retrosynthesis and synthetic feasibility assessment.

## Features

### Synthesis Planning Module (NEW)

Complete retrosynthesis and synthetic feasibility assessment:
- **Retrosynthesis Analysis** - Generate synthetic routes using ML-based (AiZynthFinder) or template-based methods
- **SAScore Calculation** - Estimate synthetic complexity using the Ertl & Schuffenhauer algorithm
- **Building Block Availability** - Check commercial availability from catalogs and databases
- **Route Scoring** - Comprehensive makeability assessment combining multiple factors

See [src/synthesis/README.md](src/synthesis/README.md) for detailed synthesis module documentation.

### Multi-Task ADMET Prediction
Predict six critical drug properties simultaneously:
- **Solubility (LogS)**: Aqueous solubility at pH 7.4
- **Permeability (Caco-2)**: Intestinal permeability
- **CYP3A4 Inhibition**: Drug-drug interaction potential
- **hERG Liability**: Cardiotoxicity risk
- **LogD**: Lipophilicity at pH 7.4
- **pKa**: Acid dissociation constant

### State-of-the-Art Architecture
- **D-MPNN (Directed Message Passing Neural Network)**: Bond-level message passing for graph-based learning
- **Fingerprint Models**: Traditional molecular fingerprints with deep neural networks
- **Multi-task Learning**: Shared representations across related properties

### Uncertainty Quantification
Three methods for estimating prediction uncertainty:
1. **Evidential Deep Learning**: Single-pass uncertainty via Normal-Inverse-Gamma distribution
2. **Monte Carlo Dropout**: Variance estimation through multiple stochastic forward passes
3. **Deep Ensembles**: Model averaging for robust uncertainty

### Comprehensive Featurization
- **Morgan/ECFP Fingerprints**: Circular substructure fingerprints
- **MACCS Keys**: 166-bit pharmacophore descriptors
- **RDKit 2D Descriptors**: Physicochemical properties
- **Graph Representations**: Molecular graphs for message passing

## Installation

### Requirements
- Python 3.10+
- PyTorch 2.0+
- RDKit 2023.9+
- PyTorch Geometric 2.4+ (optional, for graph models)
- NumPy, Pandas, Scikit-learn

### Setup

```bash
# Create conda environment
conda create -n ml-fragment-optimizer python=3.10
conda activate ml-fragment-optimizer

# Install PyTorch (adjust for your CUDA version)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Install RDKit
conda install -c conda-forge rdkit

# Install PyTorch Geometric (for graph models)
conda install pyg -c pyg

# Install additional dependencies
pip install pandas scikit-learn matplotlib seaborn
```

## Quick Start

### Basic Usage

```python
from src.models import ADMETPredictor, ADMETConfig, MolecularFeaturizer

# Configure predictor
config = ADMETConfig(
    model_type='dmpnn',        # or 'fingerprint'
    hidden_dim=256,
    num_epochs=100,
    batch_size=32,
    use_evidential=True        # Enable uncertainty quantification
)

# Initialize predictor
predictor = ADMETPredictor(config)

# Make predictions
smiles = [
    'CCO',                              # Ethanol
    'c1ccccc1',                         # Benzene
    'CC(=O)Oc1ccccc1C(=O)O'            # Aspirin
]

predictions, uncertainties = predictor.predict(smiles, return_uncertainty=True)

# Access predictions
print(f"Solubility: {predictions['solubility']}")
print(f"Permeability: {predictions['permeability']}")
```

### Training a Model

```python
from pathlib import Path
from torch.utils.data import DataLoader
from src.models import ADMETDataset, collate_admet_batch
from src.models.data_utils import ADMETDataProcessor

# Load and process data
processor = ADMETDataProcessor(normalize=True, handle_missing='drop')
smiles_list, targets = processor.load_csv('data/admet_training.csv')

# Split data
train_smiles, train_targets, val_smiles, val_targets, test_smiles, test_targets = \
    processor.split_data(smiles_list, targets, train_size=0.8, val_size=0.1)

# Create datasets
featurizer = MolecularFeaturizer()

train_dataset = ADMETDataset(train_smiles, train_targets, featurizer, config.model_type)
val_dataset = ADMETDataset(val_smiles, val_targets, featurizer, config.model_type)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_admet_batch(batch, config.model_type)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=lambda batch: collate_admet_batch(batch, config.model_type)
)

# Train model
history = predictor.train(
    train_loader,
    val_loader,
    save_path=Path('models/best_model.pt')
)

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_history.png')
```

## Architecture Details

### Directed Message Passing Neural Network (D-MPNN)

The D-MPNN architecture implements bond-level message passing:

```
Input Molecule
      ↓
Convert to Graph (atoms=nodes, bonds=edges)
      ↓
Initialize Bond Features
      ↓
Message Passing (T steps):
  - Aggregate messages from neighboring bonds
  - Update bond hidden states
      ↓
Aggregate to Atom Representations
      ↓
Pool to Graph Representation
      ↓
Task-Specific Prediction Heads
      ↓
Multi-Task Predictions
```

**Key advantages:**
- **Bond-level messaging**: Captures richer structural information than atom-level
- **Directed edges**: Each bond becomes two directed edges for asymmetric information flow
- **Stereochemistry**: Can encode stereochemical information naturally

### Evidential Deep Learning

Instead of predicting a single value, evidential deep learning predicts parameters of a probability distribution:

```
Neural Network Output
       ↓
Four Parameters: γ, λ, α, β
       ↓
Normal-Inverse-Gamma Distribution
       ↓
Extract: Mean, Aleatoric Uncertainty, Epistemic Uncertainty
```

**Benefits:**
- **Single forward pass**: No sampling required (unlike MC-Dropout)
- **Disentangled uncertainty**: Separates data noise from model uncertainty
- **Calibrated**: Provides well-calibrated confidence intervals

## Module Structure

```
ML-Fragment-Optimizer/
├── src/
│   ├── models/                      # ADMET prediction models
│   │   ├── __init__.py              # Package initialization
│   │   ├── fingerprints.py          # Molecular featurization
│   │   ├── chemprop_wrapper.py      # D-MPNN implementation
│   │   ├── uncertainty.py           # Uncertainty quantification
│   │   ├── admet_predictor.py       # Main predictor interface
│   │   └── data_utils.py            # Data loading and processing
│   └── synthesis/                   # Synthesis planning module (NEW)
│       ├── __init__.py              # Package initialization
│       ├── retrosynthesis.py        # Retrosynthesis analysis
│       ├── sa_score.py              # SAScore calculation
│       ├── building_blocks.py       # Building block availability
│       ├── route_scoring.py         # Route scoring and ranking
│       ├── example_simple.py        # Simple usage examples
│       ├── example_workflow.py      # Complete workflow example
│       ├── test_synthesis.py        # Test suite
│       └── README.md                # Detailed documentation
├── README.md                        # This file
└── requirements.txt                 # Dependencies
```

## Advanced Usage

### Using Monte Carlo Dropout

```python
from src.models import MCDropoutWrapper

# Wrap trained model
mc_wrapper = MCDropoutWrapper(predictor.model, num_samples=20)

# Get predictions with uncertainty
mean_preds, std_preds = mc_wrapper.predict_with_uncertainty(
    features.x,
    features.edge_index,
    features.edge_attr,
    features.batch
)
```

### Using Deep Ensembles

```python
from src.models import DeepEnsemble

def model_factory():
    """Create a new model instance"""
    return predictor.create_model()

# Create ensemble
ensemble = DeepEnsemble(model_factory, num_models=5)
ensemble.create_models()

# Train ensemble
def train_fn(model, train_loader, val_loader):
    # Your training logic here
    pass

ensemble.train_ensemble(train_fn, train_loader, val_loader)

# Predict with uncertainty
mean_preds, std_preds = ensemble.predict_with_uncertainty(
    x, edge_index, edge_attr, batch
)
```

### Data Augmentation

```python
from src.models.data_utils import MolecularAugmenter

augmenter = MolecularAugmenter()

# SMILES randomization
augmented_smiles = augmenter.randomize_smiles('CCO', n_augmentations=5)
print(augmented_smiles)
# ['CCO', 'OCC', 'C(O)C', ...]

# Stereoisomer enumeration
stereoisomers = augmenter.enumerate_stereoisomers('C[C@H](O)CC', max_isomers=10)
```

## Uncertainty Calibration

Evaluate how well-calibrated your uncertainty estimates are:

```python
from src.models import compute_uncertainty_metrics

# Make predictions on test set
predictions, uncertainties = predictor.predict(test_smiles, return_uncertainty=True)

# Compute calibration metrics
metrics = compute_uncertainty_metrics(predictions, uncertainties, test_targets)

for task, task_metrics in metrics.items():
    print(f"\n{task}:")
    print(f"  Expected Calibration Error: {task_metrics['ece']:.4f}")
    print(f"  Coverage (95% CI): {task_metrics['coverage_95']:.4f}")
    print(f"  Sharpness: {task_metrics['sharpness']:.4f}")
```

## Customization

### Adding New ADMET Tasks

Edit `src/models/admet_predictor.py`:

```python
ADMET_TASKS = {
    'solubility': (-10.0, 2.0),
    'permeability': (-8.0, -4.0),
    'cyp3a4': (0.0, 1.0),
    'herg': (0.0, 1.0),
    'logd': (-2.0, 6.0),
    'pka': (0.0, 14.0),
    'your_new_task': (min_value, max_value),  # Add here
}
```

### Customizing Model Architecture

```python
# Custom D-MPNN configuration
config = ADMETConfig(
    model_type='dmpnn',
    hidden_dim=512,                    # Larger hidden dimension
    num_message_passing_steps=5,       # More message passing steps
    num_ffn_layers=3,                  # Deeper feed-forward network
    task_head_hidden_dim=256,          # Larger task heads
    dropout=0.2,                       # More dropout
    pooling='mean'                     # Different pooling method
)
```

## Performance Tips

### GPU Acceleration

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    config.batch_size = 64  # Can use larger batches on GPU
else:
    print("Using CPU")
    config.batch_size = 16  # Smaller batches for CPU
```

### Memory Optimization

For large datasets:
```python
# Use gradient accumulation for large effective batch size
config.batch_size = 16           # Smaller physical batch size
gradient_accumulation_steps = 4  # Effective batch size = 64

# In training loop:
loss.backward()
if (batch_idx + 1) % gradient_accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### Parallel Data Loading

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_admet_batch(batch, config.model_type),
    num_workers=4,          # Parallel data loading
    pin_memory=True,        # Faster GPU transfer
    persistent_workers=True # Reuse workers
)
```

## Testing

Test individual modules:

```bash
# Test featurization
cd src/models
python fingerprints.py

# Test D-MPNN
python chemprop_wrapper.py

# Test uncertainty quantification
python uncertainty.py

# Test data utilities
python data_utils.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ml_fragment_optimizer,
  title = {ML-Fragment-Optimizer: Deep Learning for ADMET Prediction},
  author = {Claude Code},
  year = {2025},
  url = {https://github.com/yourusername/ML-Fragment-Optimizer}
}
```

## References

1. **D-MPNN**: Yang et al. "Analyzing Learned Molecular Representations for Property Prediction" *J. Chem. Inf. Model.* 2019
2. **Evidential Deep Learning**: Amini et al. "Deep Evidential Regression" *NeurIPS* 2020
3. **RDKit**: https://www.rdkit.org/
4. **PyTorch Geometric**: Fey & Lenssen "Fast Graph Representation Learning with PyTorch Geometric" *ICLR Workshop* 2019

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For questions or issues:
- Open an issue on GitHub

## Acknowledgments

This project builds on:
- Chemprop (MIT License)
- RDKit (BSD License)
- PyTorch Geometric (MIT License)
