# Getting Started with QSAR/SAR Analysis Module

## 5-Minute Quick Start

### Installation

```bash
cd ML-Fragment-Optimizer
pip install -r requirements_qsar.txt
```

### First Analysis

```python
from rdkit import Chem
from qsar import mmpa, activity_cliffs, sar_visualization
import matplotlib.pyplot as plt

# Your data
smiles = ["c1ccccc1", "c1ccc(Cl)cc1", "c1ccc(F)cc1", "c1ccc(Br)cc1"]
mols = [Chem.MolFromSmiles(s) for s in smiles]
activities = [5.0, 6.5, 6.0, 7.0]

# Find matched pairs
pairs, stats = mmpa.find_matched_pairs(mols, activities)
print(f"Found {len(pairs)} pairs, {len(stats)} transformations")

# Detect activity cliffs
results = activity_cliffs.detect_activity_cliffs(mols, activities)
print(f"Found {results.n_cliffs} activity cliffs")

# Visualize
viz = sar_visualization.SARVisualizer()
viz.plot_activity_landscape(mols, activities, method='pca')
plt.savefig('my_first_sar_landscape.png', dpi=300)
print("Saved: my_first_sar_landscape.png")
```

**Output**: Console summary + high-resolution figure

---

## Complete Workflow (30 minutes)

### Step 1: Prepare Your Data

```python
from rdkit import Chem
import pandas as pd

# From CSV
df = pd.read_csv('my_compounds.csv')
mols = [Chem.MolFromSmiles(s) for s in df['smiles']]
activities = df['activity'].values

# Or from SDF
from rdkit.Chem import SDMolSupplier
suppl = SDMolSupplier('my_compounds.sdf')
mols = [mol for mol in suppl if mol is not None]
activities = [float(mol.GetProp('Activity')) for mol in mols]
```

### Step 2: Exploratory Analysis

```python
from qsar import activity_cliffs, sar_visualization
import matplotlib.pyplot as plt

# Detect activity cliffs
analyzer = activity_cliffs.ActivityCliffAnalyzer(
    similarity_threshold=0.7,
    activity_threshold=2.0
)
cliff_results = analyzer.detect_cliffs(mols, activities)

print(f"Dataset: {len(mols)} molecules")
print(f"Activity range: {min(activities):.2f} - {max(activities):.2f}")
print(f"Activity cliffs: {cliff_results.n_cliffs}")
print(f"Molecules in cliffs: {len(cliff_results.cliff_molecules)}")

# Visualize activity landscape
viz = sar_visualization.SARVisualizer(figsize=(12, 10))

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

viz.plot_activity_landscape(mols, activities, method='pca', ax=axes[0, 0])
viz.plot_activity_landscape(mols, activities, method='tsne', ax=axes[0, 1])
viz.plot_similarity_heatmap(mols, activities, ax=axes[1, 0])

# Highlight cliffs
cliff_pairs = [(c.mol1_idx, c.mol2_idx) for c in cliff_results.cliffs[:10]]
viz.plot_activity_cliffs(mols, activities, cliff_pairs, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('exploratory_analysis.png', dpi=300)
```

### Step 3: Matched Molecular Pair Analysis

```python
from qsar import mmpa

# Find pairs
analyzer = mmpa.MatchedMolecularPairAnalyzer(
    max_variable_size=13,
    min_variable_size=1,
    max_cuts=2  # Single and double cuts
)

pairs = analyzer.find_pairs(mols, activities)
print(f"Found {len(pairs)} matched molecular pairs")

# Analyze transformations
stats = analyzer.analyze_transformations(pairs, min_pairs=3)
print(f"Identified {len(stats)} unique transformations")

# Show significant transformations
print("\nSignificant transformations (p < 0.05):")
for transform, stat in sorted(stats.items(), key=lambda x: abs(x[1].mean_change), reverse=True):
    if stat.is_significant(alpha=0.05):
        print(f"{stat.transformation}")
        print(f"  N = {stat.n_pairs}")
        print(f"  Δ = {stat.mean_change:.2f} ± {stat.std_change:.2f}")
        print(f"  p = {stat.p_value:.4f}")
        print(f"  Effect size (d) = {stat.effect_size:.2f}")
        print()

# Group by type
grouped = mmpa.group_transformations_by_type(stats)
for group_name, group_stats in grouped.items():
    if group_stats:
        print(f"\n{group_name.upper()} ({len(group_stats)} transformations)")
        for stat in group_stats[:3]:
            print(f"  {stat.transformation}: Δ={stat.mean_change:.2f}")
```

### Step 4: Free-Wilson Analysis (If Applicable)

```python
from qsar import free_wilson

# Select congeneric series (e.g., para-substituted benzenes)
# You need to identify these from your dataset
series_indices = [0, 2, 5, 8, 12]  # Indices of congeneric molecules
series_mols = [mols[i] for i in series_indices]
series_activities = [activities[i] for i in series_indices]

# Fit Free-Wilson model
fw_analyzer = free_wilson.FreeWilsonAnalyzer()
fw_model = fw_analyzer.fit(
    series_mols,
    series_activities,
    position_atoms=[3, 7]  # Substitution positions
)

print(f"\nFree-Wilson Analysis ({len(series_mols)} molecules)")
print(f"Baseline activity: {fw_model.baseline_activity:.2f}")
print(f"R² (training): {fw_model.r2_train:.3f}")
print(f"R² (CV): {fw_model.r2_cv:.3f}")
print(f"RMSE (CV): {fw_model.rmse_cv:.3f}")

# Top contributors
print("\nTop substituent contributions:")
for contrib in fw_analyzer.get_top_contributors(fw_model, n=10):
    if contrib.n_occurrences > 0:
        print(f"{contrib.position}-{contrib.substituent}: {contrib.contribution:+.3f}")

# Visualize contributions
contributions_dict = {}
for contrib in fw_model.contributions:
    if contrib.position not in contributions_dict:
        contributions_dict[contrib.position] = {}
    contributions_dict[contrib.position][contrib.substituent] = contrib.contribution

viz.plot_free_wilson_contributions(contributions_dict)
plt.savefig('free_wilson_contributions.png', dpi=300)
```

### Step 5: Train ML Model

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from rdkit.Chem import AllChem, DataStructs
import numpy as np

# Calculate fingerprints
def mol_to_fp(mol, n_bits=2048):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

X = np.array([mol_to_fp(mol) for mol in mols])
y = np.array(activities)

# Cliff-aware splitting
train_idx, test_idx = analyzer.create_cliff_aware_splits(
    mols, activities, test_size=0.2, random_state=42
)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred_test = model.predict(X_test)
r2 = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\nModel Performance:")
print(f"R² (test): {r2:.3f}")
print(f"RMSE (test): {rmse:.3f}")
```

### Step 6: Model Interpretation

```python
from qsar import feature_importance
from sklearn.metrics import r2_score

# Feature importance (tree-based)
feature_names = [f"Bit_{i}" for i in range(X.shape[1])]
tree_importances = dict(zip(feature_names, model.feature_importances_))
top_features = sorted(tree_importances.items(), key=lambda x: -x[1])[:10]

print("\nTop 10 features (tree-based):")
for feat, imp in top_features:
    print(f"  {feat}: {imp:.4f}")

# Permutation importance (more reliable)
pi = feature_importance.PermutationImportance(model, r2_score, n_repeats=5)
perm_imp = pi.calculate(X_test, y_test, feature_names)
top_perm = sorted(perm_imp.items(), key=lambda x: -x[1])[:10]

print("\nTop 10 features (permutation):")
for feat, imp in top_perm:
    print(f"  {feat}: {imp:+.4f}")

# SHAP values (if installed)
try:
    shap_interpreter = feature_importance.SHAPInterpreter(model, X_train[:100])
    importances = shap_interpreter.explain(X_test[:5], feature_names)

    print("\nSHAP explanation (first 5 test samples):")
    for i, imp in enumerate(importances):
        print(f"\nSample {i}: True={y_test[i]:.2f}, Pred={imp.predicted_value:.2f}")
        for feat, val in imp.get_top_features(3):
            print(f"  {feat}: {val:+.3f}")
except Exception as e:
    print(f"SHAP not available: {e}")
```

### Step 7: Bioisosteric Replacements

```python
from qsar import bioisostere_suggester

# Select molecules for optimization (e.g., top performers)
top_idx = np.argsort(activities)[-3:]  # Top 3 molecules

for idx in top_idx:
    mol = mols[idx]
    activity = activities[idx]

    print(f"\nMolecule {idx}: Activity = {activity:.2f}")
    print(f"SMILES: {Chem.MolToSmiles(mol)}")

    # Get suggestions
    suggester = bioisostere_suggester.BioisostereSuggester(model=model)
    suggestions = suggester.suggest_replacements(
        mol,
        max_suggestions=5,
        filter_drug_like=True
    )

    if suggestions:
        print(f"Found {len(suggestions)} bioisosteric replacements:")
        for i, sug in enumerate(suggestions, 1):
            print(f"\n{i}. {sug.replacement_name} ({sug.category})")
            print(f"   Rationale: {sug.rationale}")

            if sug.predicted_activity_change is not None:
                new_activity = activity + sug.predicted_activity_change
                print(f"   Predicted activity: {new_activity:.2f} (Δ={sug.predicted_activity_change:+.2f})")

            if sug.drug_likeness_score is not None:
                print(f"   Drug-likeness: {sug.drug_likeness_score:.2f}/1.00")

            if sug.synthetic_accessibility is not None:
                print(f"   Synthetic accessibility: {sug.synthetic_accessibility:.1f}/10.0")
    else:
        print("No bioisosteric replacements found.")
```

### Step 8: Generate Report

```python
# Create comprehensive figure
fig = plt.figure(figsize=(18, 12))

# 1. Activity landscape (PCA)
ax1 = plt.subplot(2, 3, 1)
viz.plot_activity_landscape(mols, activities, method='pca', ax=ax1)

# 2. Activity landscape (t-SNE)
ax2 = plt.subplot(2, 3, 2)
viz.plot_activity_landscape(mols, activities, method='tsne', ax=ax2)

# 3. Similarity heatmap
ax3 = plt.subplot(2, 3, 3)
viz.plot_similarity_heatmap(mols, activities, ax=ax3)

# 4. Activity cliffs
ax4 = plt.subplot(2, 3, 4)
cliff_pairs = [(c.mol1_idx, c.mol2_idx) for c in cliff_results.cliffs[:15]]
viz.plot_activity_cliffs(mols, activities, cliff_pairs, method='pca', ax=ax4)

# 5. SALI heatmap
ax5 = plt.subplot(2, 3, 5)
sali_matrix = activity_cliffs.calculate_sali_matrix(mols, activities)
viz.plot_sali_heatmap(sali_matrix, ax=ax5)

# 6. Matched pair network (if networkx available)
try:
    ax6 = plt.subplot(2, 3, 6)
    pair_tuples = [(p.mol1_idx, p.mol2_idx, abs(p.property_change)) for p in pairs[:20]]
    viz.plot_matched_pair_network(pair_tuples, mols, activities, ax=ax6)
except ImportError:
    pass

plt.suptitle('QSAR/SAR Analysis Report', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('qsar_report.png', dpi=300, bbox_inches='tight')
print("\nSaved comprehensive report: qsar_report.png")
```

---

## Real-World Example: Kinase Inhibitor Series

```python
# Load kinase inhibitor data
smiles_list = [
    "c1ccc2c(c1)nccc2",              # quinoline (IC50 = 5.2)
    "c1ccc2c(c1)ncc(Cl)c2",          # 3-chloroquinoline (IC50 = 4.8)
    "c1ccc2c(c1)ncc(F)c2",           # 3-fluoroquinoline (IC50 = 4.5)
    "c1ccc2c(c1)ncc(C)c2",           # 3-methylquinoline (IC50 = 5.5)
    "c1ccc2c(c1)ncc(OC)c2",          # 3-methoxyquinoline (IC50 = 6.0)
    "c1ccc2c(c1)ncc([N+](=O)[O-])c2", # 3-nitroquinoline (IC50 = 3.5) - cliff!
]

mols = [Chem.MolFromSmiles(s) for s in smiles_list]
ic50_values = [5.2, 4.8, 4.5, 5.5, 6.0, 3.5]  # Lower = more potent

# Convert IC50 to pIC50 (higher = more potent)
activities = [-np.log10(ic50 * 1e-6) for ic50 in ic50_values]

# Full analysis
pairs, stats = mmpa.find_matched_pairs(mols, activities)
cliff_results = activity_cliffs.detect_activity_cliffs(mols, activities, activity_threshold=1.0)

print(f"Kinase Inhibitor Series Analysis:")
print(f"  Molecules: {len(mols)}")
print(f"  Matched pairs: {len(pairs)}")
print(f"  Transformations: {len(stats)}")
print(f"  Activity cliffs: {cliff_results.n_cliffs}")

# Identify best transformation
for transform, stat in sorted(stats.items(), key=lambda x: x[1].mean_change, reverse=True)[:3]:
    print(f"\nTransformation: {stat.transformation}")
    print(f"  Mean change: {stat.mean_change:.2f} (increase potency)")
    print(f"  p-value: {stat.p_value:.4f}")
```

---

## Tips for Your Data

### Data Quality
- **Minimum size**: 50-100 molecules for meaningful statistics
- **Activity range**: At least 2-3 log units for good models
- **Diversity**: Balance between diversity (for model) and similarity (for MMPA)

### Activity Cliffs
- **Detection**: Adjust thresholds based on assay reproducibility
- **Splitting**: Always use cliff-aware splits for validation
- **Interpretation**: Cliffs often indicate important SAR features

### MMPA
- **Congeneric series**: Works best with systematic exploration
- **Fragment size**: Default (1-13 atoms) works for most cases
- **Statistical power**: Need 3+ pairs per transformation

### Model Building
- **Features**: Morgan fingerprints (radius=2, 2048 bits) good default
- **Algorithm**: RandomForest good starting point
- **Validation**: Use cliff-aware splits + cross-validation

---

## Next Steps

1. **Run example workflow**: `python examples/qsar_complete_workflow.py`
2. **Read full documentation**: `src/qsar/README.md`
3. **Check quick reference**: `QSAR_QUICK_REFERENCE.md`
4. **Explore tests**: `tests/test_qsar.py`

---

## Getting Help

- **Documentation**: See `src/qsar/README.md`
- **Examples**: See `examples/qsar_complete_workflow.py`
- **Tests**: See `tests/test_qsar.py` for usage patterns
- **Issues**: Open GitHub issue with example data

---

## Common Questions

**Q: How many molecules do I need?**
A: Minimum 50-100 for MMPA, 20+ for Free-Wilson, any number for cliffs/visualization.

**Q: My MMPA found no pairs. Why?**
A: Dataset too diverse. Try increasing `max_variable_size` or use more similar molecules.

**Q: Should I use cliff-aware splitting?**
A: Yes! Always use it to avoid data leakage and overly optimistic performance estimates.

**Q: Which dimensionality reduction for visualization?**
A: PCA (fast, linear), t-SNE (slow, good clusters), UMAP (fast, good clusters, requires install).

**Q: Do I need SHAP?**
A: No, but it's very useful for model interpretation. Install with `pip install shap`.

**Q: Can I use my own fingerprints?**
A: Yes! The module accepts any feature matrix (numpy array).
