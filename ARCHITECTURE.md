# ML-Fragment-Optimizer Architecture Documentation

## Overview

This document provides detailed architecture documentation for the ML-Fragment-Optimizer ADMET prediction system.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│  (ADMETPredictor class - High-level API for training/inference)│
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Processing Layer                      │
│  ┌─────────────────┐  ┌──────────────────┐                 │
│  │ MolecularFeaturizer │  │ ADMETDataProcessor│               │
│  │ - SMILES to features│  │ - Normalization  │               │
│  │ - Fingerprints      │  │ - Train/val split│               │
│  │ - Graph conversion  │  │ - Missing values │               │
│  └─────────────────┘  └──────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Architecture Layer                  │
│  ┌──────────────────────┐  ┌─────────────────────────┐     │
│  │   DMPNNModel         │  │  FingerprintModel       │     │
│  │   (Graph-based)      │  │  (Traditional ML)       │     │
│  │                      │  │                         │     │
│  │  ┌────────────────┐ │  │  ┌───────────────────┐  │     │
│  │  │ Message Passing│ │  │  │ Feed-forward NN   │  │     │
│  │  │ (Bond-level)   │ │  │  │ (Multi-layer)     │  │     │
│  │  └────────────────┘ │  │  └───────────────────┘  │     │
│  │         │            │  │           │             │     │
│  │         ▼            │  │           ▼             │     │
│  │  ┌────────────────┐ │  │  ┌───────────────────┐  │     │
│  │  │ Graph Pooling  │ │  │  │ Multi-task Heads  │  │     │
│  │  └────────────────┘ │  │  └───────────────────┘  │     │
│  │         │            │  │           │             │     │
│  │         ▼            │  │           ▼             │     │
│  │  ┌────────────────┐ │  │  ┌───────────────────┐  │     │
│  │  │ Multi-task     │ │  │  │ Task Predictions  │  │     │
│  │  │ Prediction     │ │  │  └───────────────────┘  │     │
│  │  │ Heads          │ │  │                         │     │
│  │  └────────────────┘ │  │                         │     │
│  └──────────────────────┘  └─────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Uncertainty Quantification Layer                │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Evidential DL  │  │ MC-Dropout   │  │ Deep Ensemble  │  │
│  │ (Single pass)  │  │ (Sampling)   │  │ (Multiple      │  │
│  │                │  │              │  │  models)       │  │
│  └────────────────┘  └──────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Output Layer                            │
│  Predictions + Uncertainties for 6 ADMET properties         │
└─────────────────────────────────────────────────────────────┘
```

## Module Details

### 1. Molecular Featurization (`fingerprints.py`)

#### Purpose
Convert SMILES strings into machine learning-ready representations.

#### Key Components

**MolecularFeaturizer Class**
- Handles all featurization methods
- Validates SMILES strings
- Batch processing with error handling

**Feature Types**

1. **Morgan Fingerprints (ECFP)**
   - Circular fingerprints encoding substructures
   - Default: radius=2 (ECFP4), 2048 bits
   - Captures local chemical environment
   ```python
   # Example: C-C-O encodes:
   # - Atom C with neighbor C
   # - Atom C with neighbors C and O
   # - Atom O with neighbor C
   ```

2. **MACCS Keys**
   - 166 predefined structural patterns
   - Pharmacophore-like descriptors
   - Fast computation

3. **RDKit 2D Descriptors**
   - 20 physicochemical properties
   - Molecular weight, logP, TPSA, etc.
   - Drug-like properties

4. **Graph Representation**
   - Nodes: Atoms with feature vectors
   - Edges: Bonds (bidirectional)
   - Node features (25-dim):
     * Atom type one-hot (13 types)
     * Atomic number (normalized)
     * Degree, formal charge
     * Hybridization (sp, sp2, sp3)
     * Aromaticity, in-ring
   - Edge features (6-dim):
     * Bond type (single, double, triple, aromatic)
     * Conjugation, in-ring

#### Design Decisions

**Why Morgan fingerprints?**
- Fast computation
- Good performance on diverse tasks
- Interpretable bits map to substructures

**Why graphs?**
- Preserve full molecular structure
- No information loss from hashing
- Enable sophisticated architectures (message passing)

**Why multiple feature types?**
- Ensemble of representations
- Different features capture different aspects
- User can choose based on task/resources

### 2. D-MPNN Architecture (`chemprop_wrapper.py`)

#### Purpose
Implement directed message passing neural networks for molecular property prediction.

#### Architecture Flow

```
Input: Molecular Graph
  ├─ Nodes (atoms): [N, node_features]
  ├─ Edges (bonds): [E, edge_features]
  └─ Edge Index: [2, E]

Step 1: Bond Message Initialization
  bonds_hidden = MLP(edge_features)  → [E, hidden_dim]

Step 2: Message Passing (T iterations)
  For each directed edge e_ij:
    neighbors = {e_kj | k ≠ i}  # Edges ending at source of e_ij
    message = Aggregate(bonds_hidden[neighbors])
    bonds_hidden[e_ij] = MLP([bonds_hidden[e_ij], message])

Step 3: Atom Aggregation
  For each atom i:
    incoming_bonds = {e_ji | for all j}
    atom_repr[i] = MLP([atom_features[i], Sum(bonds_hidden[incoming_bonds])])

Step 4: Graph Pooling
  graph_repr = Sum(atom_repr)  # or Mean, Max

Step 5: Multi-Task Prediction
  For each task t:
    prediction[t] = MLP_t(graph_repr)
```

#### Key Design Choices

**Bond-level vs Atom-level Messaging**
- Bond-level captures richer information
- Example: C-C single bond vs C=C double bond have different "messages"
- Enables stereochemistry encoding

**Directed Edges**
- Each undirected bond → 2 directed edges
- Bond A→B is different from bond B→A
- Context-dependent representations

**Message Aggregation**
- Sum aggregation: permutation invariant
- Preserves magnitude information
- Alternative: Mean (normalizes by degree)

**Multi-Task Learning**
- Shared encoder for all tasks
- Task-specific heads (2-layer MLP)
- Benefits:
  * Transfer learning between related properties
  * More efficient than separate models
  * Regularization effect

### 3. Uncertainty Quantification (`uncertainty.py`)

#### Why Uncertainty Matters

In drug discovery:
- **Safety**: Need confidence in toxicity predictions
- **Resource allocation**: Focus experiments on uncertain molecules
- **Active learning**: Query most informative samples

#### Method 1: Evidential Deep Learning

**Concept**
Instead of predicting a single value, predict parameters of a probability distribution.

**Normal-Inverse-Gamma (NIG) Distribution**
```
Target ~ Normal(μ, σ²)
μ ~ Normal(γ, σ²/λ)
σ² ~ Inverse-Gamma(α, β)
```

**Network Output**
Four parameters: (γ, λ, α, β)

**Uncertainty Decomposition**
```python
# Aleatoric (data uncertainty - irreducible)
aleatoric = sqrt(β / (α - 1))

# Epistemic (model uncertainty - reducible with more data)
epistemic = sqrt(β / (λ * (α - 1)))

# Total uncertainty
total = sqrt(aleatoric² + epistemic²)
```

**Loss Function**
```python
loss = NLL(target | γ, λ, α, β) + coeff * |error| * (2α + λ)
     └─ Likelihood term          └─ Regularization term
```

**Advantages**
- ✓ Single forward pass (fast)
- ✓ Disentangled uncertainties
- ✓ Principled (Bayesian)
- ✓ Well-calibrated

**Disadvantages**
- ✗ More complex loss
- ✗ Requires careful tuning
- ✗ Four outputs per task

#### Method 2: Monte Carlo Dropout

**Concept**
Keep dropout active during inference and sample multiple predictions.

**Algorithm**
```python
predictions = []
for _ in range(N):
    pred = model(x)  # With dropout active
    predictions.append(pred)

mean = mean(predictions)
std = std(predictions)
```

**Advantages**
- ✓ Simple to implement
- ✓ Works with any model with dropout
- ✓ Approximates Bayesian posterior

**Disadvantages**
- ✗ Requires N forward passes (slow)
- ✗ Uncertainty depends on dropout rate
- ✗ Can underestimate uncertainty

#### Method 3: Deep Ensembles

**Concept**
Train multiple models with different initializations and aggregate predictions.

**Algorithm**
```python
models = [train_model() for _ in range(M)]

predictions = [model(x) for model in models]
mean = mean(predictions)
std = std(predictions)
```

**Advantages**
- ✓ Gold standard for uncertainty
- ✓ Captures model uncertainty well
- ✓ Ensemble improves performance

**Disadvantages**
- ✗ Expensive: M × training cost
- ✗ M × inference cost
- ✗ M × storage cost

#### Calibration Metrics

**Expected Calibration Error (ECE)**
Measures alignment between confidence and accuracy.
```
ECE = Σ |confidence_bin - accuracy_bin| * proportion_bin
```
Lower is better. 0 = perfect calibration.

**Coverage Probability**
Fraction of targets within predicted confidence interval.
```
coverage = mean(target ∈ [μ - k*σ, μ + k*σ])
```
For 95% CI, should be ~0.95 if well-calibrated.

**Sharpness**
Average uncertainty (lower = more confident).
```
sharpness = mean(uncertainties)
```
Trade-off: Want low sharpness AND good calibration.

### 4. ADMET Predictor (`admet_predictor.py`)

#### Purpose
High-level interface for training and inference.

#### Training Pipeline

```python
1. Data Loading
   ├─ Load SMILES and targets from CSV
   ├─ Validate SMILES
   └─ Handle missing values

2. Featurization
   ├─ Convert SMILES to features
   ├─ Create PyTorch dataset
   └─ Batch with custom collate function

3. Model Creation
   ├─ Infer input dimensions from data
   ├─ Initialize model architecture
   └─ Move to GPU if available

4. Training Loop
   ├─ Forward pass → predictions
   ├─ Compute multi-task loss
   ├─ Backward pass → gradients
   ├─ Gradient clipping
   ├─ Optimizer step
   └─ Validation check

5. Early Stopping
   ├─ Monitor validation loss
   ├─ Save best model
   └─ Stop if no improvement

6. Model Saving
   └─ Save state_dict + config
```

#### Inference Pipeline

```python
1. Model Loading
   └─ Load checkpoint with config

2. Featurization
   ├─ Convert input SMILES to features
   └─ Handle invalid SMILES gracefully

3. Batch Prediction
   ├─ Create data loader
   ├─ Forward pass (no gradients)
   └─ Collect predictions

4. Post-processing
   ├─ Denormalize if needed
   ├─ Compute uncertainties
   └─ Return predictions + uncertainties
```

### 5. Data Utilities (`data_utils.py`)

#### ADMETDataProcessor

**Responsibilities**
- Load data from CSV
- Handle missing values (drop, impute)
- Normalize targets (fit on train, apply to val/test)
- Train/validation/test splitting with stratification

**Normalization Strategy**
```python
# Fit on training data
scaler = StandardScaler()
scaler.fit(train_targets)

# Transform all splits
train_normalized = scaler.transform(train_targets)
val_normalized = scaler.transform(val_targets)
test_normalized = scaler.transform(test_targets)

# Predictions are in normalized space
predictions_normalized = model(features)

# Denormalize for interpretation
predictions = scaler.inverse_transform(predictions_normalized)
```

**Why normalize?**
- Tasks have different scales (solubility: -10 to 2, hERG: 0 to 1)
- Neural networks train better with normalized inputs
- Prevents one task from dominating loss

#### MolecularAugmenter

**SMILES Randomization**
- Same molecule, different SMILES representation
- Data augmentation without changing chemistry
- Example: CCO = OCC = C(O)C
- Helps model generalize

**Stereoisomer Enumeration**
- Generate different stereoisomers
- Explores stereochemistry effects
- Use cautiously: different stereoisomers may have very different properties

## Training Best Practices

### Hyperparameter Tuning

**Model Architecture**
```python
# Start with these defaults
hidden_dim = 256
num_message_passing_steps = 3
num_ffn_layers = 2
dropout = 0.1

# Increase for complex datasets
hidden_dim = 512
num_message_passing_steps = 5

# Decrease for small datasets
hidden_dim = 128
dropout = 0.2
```

**Training**
```python
# Learning rate
lr = 1e-4  # Good default
lr = 1e-3  # If training slowly
lr = 1e-5  # If loss oscillating

# Batch size
batch_size = 32   # Default
batch_size = 64   # If GPU memory allows
batch_size = 16   # For limited memory

# Early stopping
patience = 10     # Standard
patience = 20     # For noisy data
```

### Avoiding Overfitting

1. **Regularization**
   - Dropout (0.1-0.3)
   - Weight decay (1e-5 to 1e-4)
   - Early stopping

2. **Data augmentation**
   - SMILES randomization
   - Increase training set size

3. **Model size**
   - Don't overparameterize
   - Rule of thumb: ~10-100 samples per parameter

4. **Cross-validation**
   - K-fold CV for small datasets
   - Ensures model generalizes

### Handling Imbalanced Data

For classification tasks (CYP3A4, hERG):

```python
# Class weights
class_weights = compute_class_weights(train_targets['cyp3a4'])
loss = weighted_bce_loss(predictions, targets, class_weights)

# Stratified splitting
processor.split_data(..., stratify_task='cyp3a4')

# Oversampling minority class
# Use scikit-learn's SMOTE or random oversampling
```

## Deployment Considerations

### Model Serving

**Option 1: REST API (FastAPI)**
```python
from fastapi import FastAPI
app = FastAPI()

predictor = ADMETPredictor(config)
predictor.load_model('model.pt')

@app.post("/predict")
def predict(smiles: List[str]):
    predictions, _ = predictor.predict(smiles)
    return predictions
```

**Option 2: Batch Processing**
```python
# For high-throughput screening
predictor = ADMETPredictor(config)
predictor.load_model('model.pt')

# Process in large batches
for batch in smiles_batches:
    predictions, _ = predictor.predict(batch)
    save_predictions(predictions)
```

### Performance Optimization

**GPU Utilization**
```python
# Larger batches on GPU
config.batch_size = 128  # vs 32 on CPU

# Mixed precision training (PyTorch 1.6+)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    predictions = model(features)
    loss = compute_loss(predictions, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Model Quantization**
```python
# Reduce model size for deployment
import torch.quantization as quantization

model_quantized = quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)
```

## Future Extensions

### Planned Features

1. **Hybrid Architecture**
   - Combine graph and fingerprint features
   - Dual encoder: D-MPNN + FFN
   - Late fusion before prediction heads

2. **Attention Mechanisms**
   - Atom-level attention
   - Identify important substructures
   - Interpretability

3. **Transfer Learning**
   - Pre-train on large datasets (ChEMBL)
   - Fine-tune on specific tasks
   - Few-shot learning

4. **Active Learning**
   - Query most uncertain molecules
   - Iterative model improvement
   - Efficient experimental design

5. **Multi-Fidelity Learning**
   - Combine experimental + computed data
   - Transfer from cheap predictions to expensive experiments

6. **Explainability**
   - Attention maps
   - Substructure attribution
   - SHAP values for molecular features

## References

1. Yang et al. "Analyzing Learned Molecular Representations for Property Prediction" *J. Chem. Inf. Model.* 2019
2. Amini et al. "Deep Evidential Regression" *NeurIPS* 2020
3. Lakshminarayanan et al. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" *NeurIPS* 2017
4. Gal & Ghahramani "Dropout as a Bayesian Approximation" *ICML* 2016

## Contact

For questions or contributions, please open an issue on GitHub.
