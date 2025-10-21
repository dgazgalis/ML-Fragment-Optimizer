# QSAR Module Quick Reference Card

## Installation

```bash
pip install -r requirements_qsar.txt
```

## Import

```python
from qsar import (
    mmpa,
    activity_cliffs,
    free_wilson,
    sar_visualization,
    feature_importance,
    bioisostere_suggester
)
```

## Matched Molecular Pair Analysis (MMPA)

```python
# Find pairs and analyze
pairs, stats = mmpa.find_matched_pairs(mols, activities)

# Show significant transformations
for transform, stat in stats.items():
    if stat.is_significant(alpha=0.05):
        print(f"{stat.transformation}: Δ={stat.mean_change:.2f} (p={stat.p_value:.4f})")

# Find activity cliffs from pairs
analyzer = mmpa.MatchedMolecularPairAnalyzer()
cliffs = analyzer.find_activity_cliffs(pairs, cliff_threshold=2.0)
```

## Activity Cliff Detection

```python
# Detect cliffs
results = activity_cliffs.detect_activity_cliffs(
    mols, activities,
    similarity_threshold=0.7,
    activity_threshold=2.0
)

# Get top cliffs
for cliff in results.get_top_cliffs(5):
    print(f"SALI={cliff.sali:.2f}: {cliff.activity_difference:.2f}")

# Cliff-aware splitting
analyzer = activity_cliffs.ActivityCliffAnalyzer()
train_idx, test_idx = analyzer.create_cliff_aware_splits(mols, activities)
```

## Free-Wilson Analysis

```python
# Fit model
analyzer = free_wilson.FreeWilsonAnalyzer()
model = analyzer.fit(mols, activities, position_atoms=[3])

print(f"Baseline: {model.baseline_activity:.2f}")
print(f"R² (CV): {model.r2_cv:.3f}")

# Top contributors
for contrib in analyzer.get_top_contributors(model, n=10):
    print(f"{contrib.position}-{contrib.substituent}: {contrib.contribution:+.3f}")
```

## SAR Visualization

```python
viz = sar_visualization.SARVisualizer()

# Activity landscape
viz.plot_activity_landscape(mols, activities, method='tsne')
plt.savefig('landscape.png', dpi=300)

# Similarity heatmap
viz.plot_similarity_heatmap(mols, activities)

# Activity cliffs
cliff_pairs = [(0, 5), (2, 8)]  # Cliff-forming indices
viz.plot_activity_cliffs(mols, activities, cliff_pairs)

# Molecule grid
viz.plot_molecule_grid(mols, activities, mols_per_row=4)
```

## Model Interpretation

```python
# SHAP values
from qsar.feature_importance import SHAPInterpreter

interpreter = SHAPInterpreter(model, X_train[:100])
importances = interpreter.explain(X_test)

for imp in importances:
    print(f"Prediction: {imp.predicted_value:.2f}")
    for feat, val in imp.get_top_features(5):
        print(f"  {feat}: {val:+.3f}")

# Permutation importance
from sklearn.metrics import r2_score

pi = feature_importance.PermutationImportance(model, r2_score)
perm_imp = pi.calculate(X_val, y_val, feature_names)
```

## Bioisosteric Replacements

```python
# Get suggestions
suggestions = bioisostere_suggester.suggest_bioisosteres(
    mol, max_suggestions=10
)

for sug in suggestions:
    print(f"{sug.replacement_name}: {sug.rationale}")
    print(f"  Drug-likeness: {sug.drug_likeness_score:.2f}")
    print(f"  Synthetic accessibility: {sug.synthetic_accessibility:.1f}/10")

# With trained model for activity prediction
suggester = bioisostere_suggester.BioisostereSuggester(model=model)
suggestions = suggester.suggest_replacements(mol)
```

## Common Patterns

### Complete Analysis Workflow

```python
from rdkit import Chem
import numpy as np

# Load data
mols = [Chem.MolFromSmiles(s) for s in smiles_list]
activities = np.array(activity_values)

# 1. Detect cliffs
cliff_results = activity_cliffs.detect_activity_cliffs(mols, activities)

# 2. Cliff-aware split
analyzer = activity_cliffs.ActivityCliffAnalyzer()
train_idx, test_idx = analyzer.create_cliff_aware_splits(mols, activities)

# 3. MMPA
pairs, stats = mmpa.find_matched_pairs(mols, activities)

# 4. Train model
X = calculate_features(mols)
model.fit(X[train_idx], activities[train_idx])

# 5. Interpret
interpreter = feature_importance.SHAPInterpreter(model, X[train_idx][:100])
importances = interpreter.explain(X[test_idx])

# 6. Visualize
viz = sar_visualization.SARVisualizer()
viz.plot_activity_landscape(mols, activities, method='tsne')
plt.savefig('sar_landscape.png', dpi=300)
```

### Fragment Optimization

```python
# Analyze current fragments
pairs, stats = mmpa.find_matched_pairs(fragment_mols, fragment_activities)

# Find best transformations
top_transforms = analyzer.get_top_transformations(stats, n=5, sort_by="mean_change")

# Get bioisostere suggestions for best fragments
best_fragment = fragment_mols[np.argmax(fragment_activities)]
suggestions = bioisostere_suggester.suggest_bioisosteres(best_fragment)

# Apply top suggestions and predict
for sug in suggestions[:5]:
    new_mol = apply_replacement(best_fragment, sug)
    predicted_activity = model.predict(mol_to_features(new_mol))
    print(f"{sug.replacement_name}: predicted={predicted_activity:.2f}")
```

## Configuration Tips

### MMPA
- `max_variable_size=13`: Default, increase for larger fragments
- `max_cuts=2`: Double cuts for more pairs but slower
- `min_pairs=3`: Minimum pairs for statistics

### Activity Cliffs
- `similarity_threshold=0.7`: Typical, 0.6-0.85 range
- `activity_threshold=2.0`: Adjust based on assay (1-2 log units)

### Visualizations
- `method='pca'`: Fast, linear
- `method='tsne'`: Slower, non-linear, good clusters
- `method='umap'`: Fast, non-linear (requires umap-learn)

### SHAP
- Use TreeExplainer for tree models (fast)
- Use KernelExplainer for black-box (slow, needs background)
- Background size: 50-100 samples typical

## Performance Optimization

```python
# MMPA: Use parallelization
analyzer = mmpa.MatchedMolecularPairAnalyzer(n_jobs=-1)

# Cliffs: Pre-calculate similarity matrix
sim_matrix = analyzer.calculate_similarity_matrix(mols)

# Visualization: Subsample for large datasets
if len(mols) > 5000:
    indices = np.random.choice(len(mols), 5000, replace=False)
    viz.plot_activity_landscape(mols[indices], activities[indices])

# SHAP: Use smaller background
interpreter = SHAPInterpreter(model, X_train[:50])
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| MMPA finds no pairs | Increase `max_variable_size`, check dataset diversity |
| Free-Wilson R² low | Check additivity assumption, need more molecules |
| SHAP errors | Install shap: `pip install shap` |
| t-SNE slow | Use PCA or UMAP, or subsample dataset |
| Cliffs in test set | Use `create_cliff_aware_splits()` |

## Example Datasets

### Simple Test
```python
smiles = ["c1ccccc1", "c1ccc(Cl)cc1", "c1ccc(F)cc1"]
mols = [Chem.MolFromSmiles(s) for s in smiles]
activities = [5.0, 6.5, 6.0]
```

### With Activity Cliff
```python
smiles = [
    "c1ccccc1",                    # 5.0
    "c1ccc(Cl)cc1",                # 6.0
    "c1ccc([N+](=O)[O-])cc1",     # 8.5 (cliff!)
]
activities = [5.0, 6.0, 8.5]
```

## Citation

```bibtex
@software{ml_fragment_optimizer_qsar,
  title = {ML-Fragment-Optimizer QSAR/SAR Analysis Module},
  year = {2024},
  url = {https://github.com/yourusername/ML-Fragment-Optimizer}
}
```

## Resources

- Full Documentation: `src/qsar/README.md`
- Complete Workflow: `examples/qsar_complete_workflow.py`
- Unit Tests: `tests/test_qsar.py`
- Module Summary: `QSAR_MODULE_SUMMARY.md`
