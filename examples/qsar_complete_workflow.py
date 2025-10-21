#!/usr/bin/env python3
"""
Complete QSAR/SAR Analysis Workflow Example

Demonstrates a comprehensive workflow using all modules in the QSAR package:
1. Matched Molecular Pair Analysis (MMPA)
2. Activity Cliff Detection
3. Free-Wilson Analysis
4. SAR Visualization
5. Model Training and Interpretation
6. Bioisosteric Replacement Suggestions

This example uses a synthetic dataset of para-substituted benzenes.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Import QSAR modules
from qsar import mmpa, activity_cliffs, free_wilson, sar_visualization, feature_importance, bioisostere_suggester


def generate_synthetic_dataset():
    """Generate synthetic dataset of substituted benzenes."""
    # Para-substituted benzenes with various substituents
    dataset = [
        # Base structures
        ("c1ccccc1", 5.0, "benzene"),

        # Halogens (increasing activity with size)
        ("Fc1ccccc1", 5.5, "fluorobenzene"),
        ("Clc1ccccc1", 6.0, "chlorobenzene"),
        ("Brc1ccccc1", 6.5, "bromobenzene"),
        ("Ic1ccccc1", 7.0, "iodobenzene"),

        # Para-substituted halogens
        ("Fc1ccc(F)cc1", 6.0, "1,4-difluorobenzene"),
        ("Clc1ccc(Cl)cc1", 6.8, "1,4-dichlorobenzene"),
        ("Brc1ccc(Br)cc1", 7.5, "1,4-dibromobenzene"),

        # Alkyl groups
        ("Cc1ccccc1", 5.2, "toluene"),
        ("CCc1ccccc1", 5.8, "ethylbenzene"),
        ("Cc1ccc(C)cc1", 5.5, "p-xylene"),

        # Oxygen-containing
        ("COc1ccccc1", 5.9, "anisole"),
        ("Oc1ccccc1", 5.8, "phenol"),
        ("COc1ccc(OC)cc1", 6.2, "1,4-dimethoxybenzene"),

        # Nitrogen-containing (activity cliff!)
        ("Nc1ccccc1", 4.5, "aniline"),
        ("[N+](=O)([O-])c1ccccc1", 8.5, "nitrobenzene"),  # Big jump!
        ("c1ccc(N)cc1", 4.8, "para-aniline"),
        ("[N+](=O)([O-])c1ccc([N+](=O)[O-])cc1", 9.2, "1,4-dinitrobenzene"),  # Very active!

        # Carboxylic acids and derivatives
        ("c1ccc(C(=O)O)cc1", 6.5, "benzoic acid"),
        ("c1ccc(C(=O)N)cc1", 5.8, "benzamide"),
        ("c1ccc(C(=O)OC)cc1", 6.2, "methyl benzoate"),

        # Heterocycles
        ("c1ccncc1", 6.8, "pyridine"),
        ("c1cnccc1", 6.5, "pyridine (meta)"),
        ("c1ncccc1", 6.3, "pyridine (ortho)"),

        # Mixed substituents
        ("Cc1ccc(Cl)cc1", 6.3, "4-chlorotoluene"),
        ("COc1ccc(Cl)cc1", 6.7, "4-chloroanisole"),
        ("Nc1ccc(Cl)cc1", 5.2, "4-chloroaniline"),
    ]

    smiles_list = [s for s, _, _ in dataset]
    activities = [a for _, a, _ in dataset]
    names = [n for _, _, n in dataset]
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]

    return mols, activities, names, smiles_list


def calculate_molecular_features(mols):
    """Calculate molecular fingerprints for ML."""
    features = []

    for mol in mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        arr = np.zeros((2048,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        features.append(arr)

    return np.array(features)


def main():
    """Run complete QSAR/SAR analysis workflow."""

    print("=" * 80)
    print("QSAR/SAR ANALYSIS COMPLETE WORKFLOW")
    print("=" * 80)
    print()

    # ========================================================================
    # 1. Load Dataset
    # ========================================================================
    print("Step 1: Loading Dataset")
    print("-" * 80)

    mols, activities, names, smiles_list = generate_synthetic_dataset()
    print(f"Loaded {len(mols)} molecules")
    print(f"Activity range: {min(activities):.2f} - {max(activities):.2f}")
    print()

    # ========================================================================
    # 2. Matched Molecular Pair Analysis
    # ========================================================================
    print("Step 2: Matched Molecular Pair Analysis (MMPA)")
    print("-" * 80)

    analyzer = mmpa.MatchedMolecularPairAnalyzer(max_variable_size=13)
    pairs = analyzer.find_pairs(mols, activities)
    stats = analyzer.analyze_transformations(pairs, min_pairs=2)

    print(f"Found {len(pairs)} matched molecular pairs")
    print(f"Identified {len(stats)} unique transformations")
    print()

    # Show top transformations
    print("Top 5 transformations by effect size:")
    top_transforms = analyzer.get_top_transformations(
        stats, n=5, sort_by="mean_change", only_significant=False
    )
    for i, stat in enumerate(top_transforms, 1):
        print(f"{i}. {stat.transformation}")
        print(f"   N={stat.n_pairs}, Δ={stat.mean_change:.2f}±{stat.std_change:.2f}")
        print(f"   p={stat.p_value:.4f}, d={stat.effect_size:.2f}")
    print()

    # Find activity cliffs from pairs
    cliff_pairs = analyzer.find_activity_cliffs(pairs, cliff_threshold=2.0)
    print(f"Found {len(cliff_pairs)} activity cliffs (ΔActivity ≥ 2.0)")
    print()

    # ========================================================================
    # 3. Activity Cliff Detection
    # ========================================================================
    print("Step 3: Activity Cliff Detection")
    print("-" * 80)

    cliff_analyzer = activity_cliffs.ActivityCliffAnalyzer(
        similarity_threshold=0.7,
        activity_threshold=2.0,
    )
    cliff_results = cliff_analyzer.detect_cliffs(mols, activities)

    print(f"Detected {cliff_results.n_cliffs} activity cliffs")
    print(f"Mean SALI: {cliff_results.mean_sali:.2f}")
    print(f"Max SALI: {cliff_results.max_sali:.2f}")
    print(f"{len(cliff_results.cliff_molecules)} molecules involved in cliffs")
    print()

    # Show top cliffs
    print("Top 3 activity cliffs by SALI:")
    for i, cliff in enumerate(cliff_results.get_top_cliffs(3), 1):
        print(f"{i}. SALI={cliff.sali:.2f}")
        print(f"   Mol1: {names[cliff.mol1_idx]} (activity={cliff.activity1:.2f})")
        print(f"   Mol2: {names[cliff.mol2_idx]} (activity={cliff.activity2:.2f})")
        print(f"   Similarity: {cliff.similarity:.3f}")
        print(f"   ΔActivity: {cliff.activity_difference:.2f}")
    print()

    # ========================================================================
    # 4. Free-Wilson Analysis
    # ========================================================================
    print("Step 4: Free-Wilson Analysis")
    print("-" * 80)

    # Select para-substituted benzenes for Free-Wilson
    para_indices = [i for i, s in enumerate(smiles_list) if "c1ccc(" in s]
    para_mols = [mols[i] for i in para_indices]
    para_activities = [activities[i] for i in para_indices]
    para_names = [names[i] for i in para_indices]

    if len(para_mols) >= 5:
        fw_analyzer = free_wilson.FreeWilsonAnalyzer()
        try:
            fw_model = fw_analyzer.fit(para_mols, para_activities, position_atoms=[3])

            print(f"Free-Wilson Model for {len(para_mols)} para-substituted benzenes:")
            print(f"Baseline Activity: {fw_model.baseline_activity:.3f}")
            print(f"R² (training): {fw_model.r2_train:.3f}")
            print(f"R² (CV): {fw_model.r2_cv:.3f}")
            print(f"RMSE (CV): {fw_model.rmse_cv:.3f}")
            print()

            # Top contributors
            print("Top 5 substituent contributions:")
            top_contribs = fw_analyzer.get_top_contributors(fw_model, n=5)
            for contrib in top_contribs:
                if contrib.n_occurrences > 0:
                    print(f"  {contrib.position}-{contrib.substituent}: "
                          f"{contrib.contribution:+.3f} ± {contrib.std_error:.3f}")
        except Exception as e:
            print(f"Free-Wilson analysis failed: {e}")
    else:
        print(f"Insufficient para-substituted benzenes for Free-Wilson analysis")
    print()

    # ========================================================================
    # 5. Train ML Model
    # ========================================================================
    print("Step 5: Training Random Forest Model")
    print("-" * 80)

    # Calculate features
    X = calculate_molecular_features(mols)
    y = np.array(activities)

    # Train/test split (cliff-aware)
    train_idx, test_idx = cliff_analyzer.create_cliff_aware_splits(
        mols, activities, test_size=0.2, random_state=42
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"Model Performance:")
    print(f"  Training:   R²={r2_train:.3f}, RMSE={rmse_train:.3f}")
    print(f"  Test:       R²={r2_test:.3f}, RMSE={rmse_test:.3f}")
    print()

    # ========================================================================
    # 6. Model Interpretation
    # ========================================================================
    print("Step 6: Model Interpretation")
    print("-" * 80)

    # Feature importance (tree-based)
    feature_names = [f"Bit_{i}" for i in range(X.shape[1])]
    tree_importances = dict(zip(feature_names, model.feature_importances_))
    top_features = sorted(tree_importances.items(), key=lambda x: -x[1])[:10]

    print("Top 10 features by tree-based importance:")
    for feat, imp in top_features:
        print(f"  {feat}: {imp:.4f}")
    print()

    # Permutation importance
    print("Calculating permutation importance...")
    pi = feature_importance.PermutationImportance(model, r2_score, n_repeats=5)
    perm_imp = pi.calculate(X_test, y_test, feature_names)
    top_perm = sorted(perm_imp.items(), key=lambda x: -x[1])[:10]

    print("Top 10 features by permutation importance:")
    for feat, imp in top_perm:
        print(f"  {feat}: {imp:+.4f}")
    print()

    # SHAP (if available)
    try:
        print("Calculating SHAP values...")
        shap_interpreter = feature_importance.SHAPInterpreter(model, X_train[:50])
        importances = shap_interpreter.explain(X_test[:3], feature_names)

        print("SHAP explanation for first 3 test samples:")
        for i, imp in enumerate(importances):
            print(f"\nSample {i} (True={y_test[i]:.2f}, Pred={imp.predicted_value:.2f}):")
            for feat, val in imp.get_top_features(5):
                print(f"  {feat}: {val:+.3f}")
    except Exception as e:
        print(f"SHAP analysis not available: {e}")
        print("Install SHAP with: pip install shap")
    print()

    # ========================================================================
    # 7. Bioisosteric Replacements
    # ========================================================================
    print("Step 7: Bioisosteric Replacement Suggestions")
    print("-" * 80)

    # Pick a few molecules for suggestions
    test_mols = [
        (mols[1], names[1]),  # fluorobenzene
        (mols[18], names[18]),  # benzoic acid
    ]

    suggester = bioisostere_suggester.BioisostereSuggester(model=model)

    for mol, name in test_mols:
        print(f"\nMolecule: {name}")
        suggestions = suggester.suggest_replacements(
            mol, max_suggestions=5, filter_drug_like=False
        )

        if suggestions:
            for i, sug in enumerate(suggestions, 1):
                print(f"\n{i}. {sug.replacement_name} ({sug.category})")
                print(f"   {sug.rationale}")
                if sug.predicted_activity_change is not None:
                    print(f"   Predicted ΔActivity: {sug.predicted_activity_change:+.2f}")
                if sug.drug_likeness_score is not None:
                    print(f"   Drug-likeness: {sug.drug_likeness_score:.2f}")
        else:
            print("  No suggestions found")
    print()

    # ========================================================================
    # 8. Visualizations
    # ========================================================================
    print("Step 8: Generating Visualizations")
    print("-" * 80)

    viz = sar_visualization.SARVisualizer(figsize=(10, 8))

    # Create multi-panel figure
    fig = plt.figure(figsize=(16, 12))

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
    cliff_pairs_viz = [(c.mol1_idx, c.mol2_idx) for c in cliff_results.cliffs[:10]]
    viz.plot_activity_cliffs(mols, activities, cliff_pairs_viz, ax=ax4)

    # 5. SALI heatmap
    ax5 = plt.subplot(2, 3, 5)
    sali_matrix = activity_cliffs.calculate_sali_matrix(mols, activities)
    viz.plot_sali_heatmap(sali_matrix, ax=ax5)

    # 6. Matched pair network (if networkx available)
    try:
        ax6 = plt.subplot(2, 3, 6)
        pair_tuples = [
            (p.mol1_idx, p.mol2_idx, abs(p.property_change))
            for p in pairs[:20]  # Limit to avoid clutter
        ]
        viz.plot_matched_pair_network(pair_tuples, mols, activities, ax=ax6)
    except ImportError:
        print("  Skipping network plot (networkx not installed)")

    plt.tight_layout()
    output_file = Path(__file__).parent / "qsar_workflow_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {output_file}")
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nDataset: {len(mols)} molecules")
    print(f"Matched Pairs: {len(pairs)}")
    print(f"Transformations: {len(stats)}")
    print(f"Activity Cliffs: {cliff_results.n_cliffs}")
    print(f"Model R² (test): {r2_test:.3f}")
    print(f"\nVisualization saved to: {output_file}")
    print("\nThis workflow demonstrated:")
    print("  ✓ Matched Molecular Pair Analysis")
    print("  ✓ Activity Cliff Detection")
    print("  ✓ Free-Wilson Analysis")
    print("  ✓ ML Model Training & Evaluation")
    print("  ✓ Model Interpretation (SHAP, Permutation)")
    print("  ✓ Bioisosteric Replacement Suggestions")
    print("  ✓ Publication-Quality Visualizations")
    print()


if __name__ == "__main__":
    main()
