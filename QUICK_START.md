# Quick Start Guide

Get up and running with ML-Fragment-Optimizer in 5 minutes.

## Installation

```bash
# Create conda environment
conda create -n ml-fragment-opt python=3.10
conda activate ml-fragment-opt

# Install PyTorch (adjust for your CUDA version)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Install RDKit
conda install -c conda-forge rdkit

# Install PyTorch Geometric (optional, for graph models)
conda install pyg -c pyg

# Install other dependencies
pip install pandas scikit-learn matplotlib seaborn

# Verify installation
python test_installation.py
```

## 1-Minute Example: Predict ADMET Properties

```python
from src.models import MolecularFeaturizer

# Featurize molecules
featurizer = MolecularFeaturizer()
smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
features = [featurizer.featurize(s) for s in smiles]

print(f"Featurized {len(features)} molecules")
for feat in features:
    print(f"  {feat.smiles}: {feat.morgan_fp.shape[0]} fingerprint bits")
```

## 5-Minute Example: Train a Model

```bash
# Run example training script
python examples/example_training.py
```

This will:
1. Create sample ADMET dataset (500 molecules)
2. Split into train/val/test sets
3. Train D-MPNN model for 20 epochs
4. Save model and generate plots
5. Evaluate on test set

**Expected output:**
```
Training completed in ~2-5 minutes (CPU) or ~30 seconds (GPU)
Model saved to: outputs/best_model.pt
Plots saved to: outputs/training_history.png, outputs/test_predictions.png
```

## 10-Minute Example: Make Predictions

```bash
# Run example inference script
python examples/example_inference.py
```

This will:
1. Load trained model
2. Predict ADMET properties for 5 drug molecules
3. Interpret predictions (flag potential issues)
4. Save results to CSV

**Expected output:**
```
Molecule: Aspirin (CC(=O)Oc1ccccc1C(=O)O)
  Solubility (LogS):     -2.45  ✓ GOOD
  Permeability (Caco-2): -5.80  ○ MODERATE
  CYP3A4 Inhibition:      0.35  ✓ NON-INHIBITOR
  hERG Liability:         0.12  ✓ SAFE
  LogD (pH 7.4):          1.19  ✓ OPTIMAL
```

## Common Use Cases

### Use Case 1: Quick Property Prediction

```python
from pathlib import Path
from src.models import ADMETPredictor, ADMETConfig

# Load pre-trained model
predictor = ADMETPredictor(ADMETConfig())
predictor.load_model(Path('outputs/best_model.pt'))

# Predict
smiles = ['CCO', 'c1ccccc1']
predictions, _ = predictor.predict(smiles)

print(predictions['solubility'])  # [[-0.77], [-2.45]]
```

### Use Case 2: Custom Training Data

```python
from src.models.data_utils import ADMETDataProcessor

# Prepare your CSV file with columns: smiles, solubility, permeability, etc.
processor = ADMETDataProcessor(normalize=True)
smiles_list, targets = processor.load_csv('my_data.csv')

# Split and train (see example_training.py for complete code)
train_smiles, train_targets, val_smiles, val_targets, test_smiles, test_targets = \
    processor.split_data(smiles_list, targets)
```

### Use Case 3: Batch Prediction (High-Throughput)

```python
# Process large libraries efficiently
import pandas as pd
from src.models import ADMETPredictor, ADMETConfig

# Load large library
df = pd.read_csv('large_library.csv')
smiles_list = df['smiles'].tolist()

# Predict in batches
predictor = ADMETPredictor(ADMETConfig(batch_size=128))
predictor.load_model('outputs/best_model.pt')

predictions, _ = predictor.predict(smiles_list)

# Save results
df['predicted_solubility'] = predictions['solubility'].flatten()
df['predicted_permeability'] = predictions['permeability'].flatten()
df.to_csv('predictions.csv', index=False)
```

### Use Case 4: Uncertainty Quantification

```python
from src.models import ADMETPredictor, ADMETConfig

# Enable evidential deep learning
config = ADMETConfig(use_evidential=True)
predictor = ADMETPredictor(config)

# Train model (see example_training.py)
# ...

# Predict with uncertainty
predictions, uncertainties = predictor.predict(smiles, return_uncertainty=True)

# High uncertainty = low confidence = needs more data or experiments
for i, smi in enumerate(smiles):
    pred = predictions['solubility'][i][0]
    unc = uncertainties['solubility'][i][0]
    print(f"{smi}: {pred:.2f} ± {unc:.2f}")
```

## Model Types

### D-MPNN (Graph-based) - Recommended

**Best for:** Most tasks, especially with sufficient data (>1000 samples)

```python
config = ADMETConfig(
    model_type='dmpnn',
    hidden_dim=256,
    num_message_passing_steps=3
)
```

**Pros:**
- State-of-the-art performance
- Captures full molecular structure
- No information loss

**Cons:**
- Requires PyTorch Geometric
- Slower than fingerprint models
- More parameters (needs more data)

### Fingerprint Model (Traditional)

**Best for:** Small datasets, fast inference, production deployment

```python
config = ADMETConfig(
    model_type='fingerprint',
    hidden_dim=512
)
```

**Pros:**
- Faster training and inference
- Works with fewer samples
- No additional dependencies

**Cons:**
- Information loss from fingerprinting
- May not capture subtle structural features

## Performance Tuning

### For Small Datasets (<1000 samples)

```python
config = ADMETConfig(
    model_type='fingerprint',  # Simpler model
    hidden_dim=128,            # Fewer parameters
    dropout=0.3,               # More regularization
    batch_size=16,
    learning_rate=1e-4
)
```

### For Large Datasets (>10,000 samples)

```python
config = ADMETConfig(
    model_type='dmpnn',
    hidden_dim=512,            # More capacity
    num_message_passing_steps=5,
    batch_size=128,            # Larger batches
    learning_rate=1e-3         # Faster learning
)
```

### For GPU Acceleration

```python
import torch

# Check GPU
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Larger batch size on GPU
config = ADMETConfig(
    batch_size=128,  # vs 32 on CPU
    num_workers=4    # Parallel data loading
)
```

## Troubleshooting

### Import Errors

```bash
# Missing RDKit
conda install -c conda-forge rdkit

# Missing PyTorch Geometric
conda install pyg -c pyg

# CUDA errors
# Check PyTorch CUDA version matches your system
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory

```python
# Reduce batch size
config.batch_size = 16  # or 8

# Reduce model size
config.hidden_dim = 128
config.num_message_passing_steps = 2

# Use gradient accumulation
gradient_accumulation_steps = 4
```

### Slow Training

```python
# Use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reduce message passing steps
config.num_message_passing_steps = 2

# Switch to fingerprint model
config.model_type = 'fingerprint'

# Increase batch size (if memory allows)
config.batch_size = 64
```

### Poor Performance

```python
# More data
# - ADMET models need 1000+ samples for good performance
# - Consider data augmentation

# Hyperparameter tuning
# - Try different learning rates: [1e-5, 1e-4, 1e-3]
# - Try different architectures: [128, 256, 512] hidden_dim
# - More epochs with patience

# Check data quality
# - Remove outliers
# - Check for duplicates
# - Validate SMILES strings
```

## Next Steps

1. **Read Documentation**
   - [README.md](README.md) - Overview and features
   - [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed technical documentation

2. **Explore Examples**
   - `examples/example_training.py` - Complete training pipeline
   - `examples/example_inference.py` - Making predictions

3. **Customize for Your Data**
   - Prepare CSV with your ADMET data
   - Adjust `ADMET_TASKS` in `src/models/admet_predictor.py`
   - Train custom model

4. **Advanced Features**
   - Uncertainty quantification with evidential deep learning
   - Monte Carlo dropout for confidence estimates
   - Deep ensembles for robust predictions

## Support

- **Documentation**: See README.md and ARCHITECTURE.md
- **Issues**: Open an issue on GitHub
- **Examples**: Check `examples/` directory

## Quick Command Reference

```bash
# Installation
python test_installation.py

# Training
python examples/example_training.py

# Inference
python examples/example_inference.py

# Test individual modules
cd src/models
python fingerprints.py
python chemprop_wrapper.py
python uncertainty.py
python data_utils.py
```

Happy predicting! 🎉
