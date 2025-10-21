"""
Example usage of data processing utilities.

This script demonstrates how to:
1. Load molecular datasets from various sources
2. Clean and standardize molecules
3. Split datasets for ML training
4. Process assay data
5. Generate benchmarks
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    MoleculeDataLoader,
    MoleculeDataCleaner,
    DatasetSplitter,
    MoleculeNetLoader,
    ComprehensiveCleaner,
    calculate_dataset_statistics,
    FileConverter,
    IdentifierConverter
)


def example_1_load_and_clean():
    """Example 1: Load and clean molecular data."""
    print("=" * 70)
    print("Example 1: Load and Clean Molecular Data")
    print("=" * 70)

    # Create sample SMILES data
    smiles_data = {
        'compound_id': ['cpd1', 'cpd2', 'cpd3', 'cpd4', 'cpd5'],
        'smiles': [
            'CCO',  # Ethanol
            'CC(=O)O',  # Acetic acid
            'c1ccccc1',  # Benzene
            'CCO.[Na+].[Cl-]',  # Ethanol + salt
            'invalid_smiles'  # Invalid
        ],
        'activity': [1.2, 3.4, 5.6, 7.8, 9.0]
    }

    df = pd.DataFrame(smiles_data)
    print(f"\nInitial dataset: {len(df)} compounds")
    print(df)

    # Load molecules
    loader = MoleculeDataLoader()
    df = loader.from_smiles_list(
        df['smiles'].tolist(),
        ids=df['compound_id'].tolist(),
        activity=df['activity'].tolist()
    )

    print(f"\nAfter loading: {len(df)} compounds with mol objects")

    # Clean data
    cleaner = MoleculeDataCleaner(
        remove_salts=True,
        neutralize=True,
        remove_invalid=True,
        remove_duplicates=True
    )

    df_clean = cleaner.clean(df)
    print(f"\nAfter cleaning: {len(df_clean)} compounds")
    print(df_clean[['mol_id', 'smiles', 'activity']])

    # Calculate statistics
    stats = calculate_dataset_statistics(df_clean)
    print(f"\nDataset Statistics:")
    print(f"  Valid molecules: {stats.n_valid}")
    print(f"  Mean MW: {stats.mw_mean:.2f} ± {stats.mw_std:.2f}")
    print(f"  Mean LogP: {stats.logp_mean:.2f} ± {stats.logp_std:.2f}")

    return df_clean


def example_2_comprehensive_cleaning():
    """Example 2: Comprehensive molecular cleaning."""
    print("\n" + "=" * 70)
    print("Example 2: Comprehensive Molecular Cleaning")
    print("=" * 70)

    # Sample molecules including problematic ones
    smiles_data = {
        'smiles': [
            'CCO',  # Clean
            'c1ccccc1',  # Clean
            'c1ccc(O)c(O)c1',  # Catechol (PAINS)
            'CC(=O)Cl',  # Acyl chloride (reactive)
            'c1ccc(cc1)c2ccccc2c3ccccc3',  # Polyaromatic (aggregator)
            'CC(C)C',  # Clean
            'CCCCCCCCCCCCCCCCCC',  # Long chain
        ]
    }

    df = MoleculeDataLoader.from_smiles_list(smiles_data['smiles'])

    print(f"\nInitial: {len(df)} molecules")

    # Comprehensive cleaning
    cleaner = ComprehensiveCleaner(
        standardize=True,
        remove_pains=True,
        remove_aggregators=True,
        remove_reactive=True,
        apply_rule_of_five=False,  # Don't apply for this example
        remove_duplicates=True
    )

    df_clean, stats = cleaner.clean(df)

    print(f"\nCleaning Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nRemaining molecules:")
    print(df_clean[['mol_id', 'smiles']])

    return df_clean


def example_3_dataset_splitting():
    """Example 3: Dataset splitting strategies."""
    print("\n" + "=" * 70)
    print("Example 3: Dataset Splitting Strategies")
    print("=" * 70)

    # Create larger sample dataset
    np.random.seed(42)
    smiles_list = [
        'CCO', 'CC(C)O', 'CCCO', 'CC(C)CO', 'CCCCO',
        'c1ccccc1', 'c1ccccc1C', 'c1ccccc1CC', 'c1ccccc1CCC',
        'c1ccc(O)cc1', 'c1ccc(C)cc1', 'c1ccc(CC)cc1',
        'C1CCCCC1', 'C1CCCCC1C', 'C1CCCCC1CC',
        'c1ccncc1', 'c1cccnc1', 'c1ccncc1C',
        'CC(=O)O', 'CCC(=O)O', 'CCCC(=O)O'
    ]

    df = MoleculeDataLoader.from_smiles_list(smiles_list)
    df['activity'] = np.random.uniform(3, 9, len(df))

    print(f"\nDataset size: {len(df)} molecules")

    # Random split
    print("\n1. Random Split:")
    splitter = DatasetSplitter()
    train, val, test = splitter.random_split(df, train_size=0.7, val_size=0.15, test_size=0.15)

    # Scaffold split
    print("\n2. Scaffold Split:")
    train, val, test = splitter.scaffold_split(df, train_size=0.7, val_size=0.15, test_size=0.15)

    # Stratified split (using binned activity)
    print("\n3. Stratified Split:")
    df['activity_bin'] = pd.cut(df['activity'], bins=3, labels=['low', 'medium', 'high'])
    train, val, test = splitter.stratified_split(
        df,
        stratify_column='activity_bin',
        train_size=0.7,
        val_size=0.15,
        test_size=0.15
    )

    return train, val, test


def example_4_load_public_dataset():
    """Example 4: Load MoleculeNet dataset."""
    print("\n" + "=" * 70)
    print("Example 4: Load MoleculeNet Dataset")
    print("=" * 70)

    loader = MoleculeNetLoader()

    # List available datasets
    print("\nAvailable MoleculeNet datasets:")
    datasets = loader.list_datasets()
    print(datasets.to_string(index=False))

    # Load BACE dataset (small, fast to download)
    print("\n" + "-" * 70)
    print("Loading BACE dataset...")

    try:
        df = loader.load_dataset('bace', parse_molecules=True)

        print(f"\nLoaded {len(df)} molecules")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df[['smiles', 'Class']].head())

        # Calculate statistics
        stats = calculate_dataset_statistics(df)
        print(f"\nDataset Statistics:")
        print(f"  Valid molecules: {stats.n_valid}/{stats.n_molecules}")
        print(f"  Mean MW: {stats.mw_mean:.2f} ± {stats.mw_std:.2f}")
        print(f"  Mean LogP: {stats.logp_mean:.2f} ± {stats.logp_std:.2f}")

        return df

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Note: This requires internet connection")
        return None


def example_5_format_conversion():
    """Example 5: Format conversion."""
    print("\n" + "=" * 70)
    print("Example 5: Format Conversion")
    print("=" * 70)

    # Create sample data
    smiles_list = ['CCO', 'CC(=O)O', 'c1ccccc1']
    df = MoleculeDataLoader.from_smiles_list(smiles_list)
    df['activity'] = [1.2, 3.4, 5.6]

    print(f"\nOriginal data: {len(df)} molecules")

    # Add identifiers
    print("\nAdding molecular identifiers...")
    converter = IdentifierConverter()
    df = converter.add_identifiers(df)

    print("\nMolecular identifiers:")
    print(df[['mol_id', 'smiles', 'inchi', 'inchikey', 'formula']])

    # Save to CSV
    output_csv = Path('example_molecules.csv')
    df[['mol_id', 'smiles', 'activity']].to_csv(output_csv, index=False)
    print(f"\nSaved to CSV: {output_csv}")

    # Convert to SDF
    output_sdf = Path('example_molecules.sdf')
    n_written = FileConverter.smiles_to_sdf(
        output_csv,
        output_sdf,
        generate_3d=False
    )
    print(f"\nConverted to SDF: {n_written} molecules")

    # Convert back to CSV
    output_csv2 = Path('example_molecules_converted.csv')
    n_written = FileConverter.sdf_to_smiles(output_sdf, output_csv2)
    print(f"\nConverted back to CSV: {n_written} molecules")

    # Clean up
    output_csv.unlink()
    output_sdf.unlink()
    output_csv2.unlink()

    print("\nFormat conversion completed!")


def example_6_assay_data_processing():
    """Example 6: Process assay data."""
    print("\n" + "=" * 70)
    print("Example 6: Process Assay Data")
    print("=" * 70)

    from src.utils import (
        QualityController,
        AssayNormalizer,
        HitCaller
    )

    # Generate synthetic plate data (96-well)
    np.random.seed(42)
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cols = list(range(1, 13))

    data = []
    for row in rows:
        for col in cols:
            well = f"{row}{col:02d}"

            # Positive controls (column 1)
            if col == 1:
                value = np.random.normal(10000, 500)
                well_type = "positive"
            # Negative controls (column 12)
            elif col == 12:
                value = np.random.normal(1000, 100)
                well_type = "negative"
            # Samples
            else:
                value = np.random.normal(5000, 1000)
                well_type = "sample"

            data.append({
                'well': well,
                'row': row,
                'column': col,
                'value': value,
                'well_type': well_type
            })

    plate_df = pd.DataFrame(data)

    print(f"\nPlate data: {len(plate_df)} wells")

    # Quality control
    positive_wells = [f"{row}01" for row in rows]
    negative_wells = [f"{row}12" for row in rows]

    qc_metrics = QualityController.assess_plate_quality(
        plate_df,
        positive_wells,
        negative_wells
    )

    print(f"\nQuality Control Metrics:")
    print(f"  Z'-factor: {qc_metrics.z_prime_factor:.3f}")
    print(f"  S/N: {qc_metrics.signal_to_noise:.2f}")
    print(f"  CV (Pos): {qc_metrics.cv_positive:.2f}%")
    print(f"  CV (Neg): {qc_metrics.cv_negative:.2f}%")
    print(f"  Passed: {qc_metrics.passed}")

    # Normalize data
    pos_mean = plate_df[plate_df['well'].isin(positive_wells)]['value'].mean()
    neg_mean = plate_df[plate_df['well'].isin(negative_wells)]['value'].mean()

    normalizer = AssayNormalizer()
    plate_df['percent_inhibition'] = normalizer.percent_inhibition(
        plate_df['value'].values,
        pos_mean,
        neg_mean
    )

    # Call hits
    sample_data = plate_df[plate_df['well_type'] == 'sample']
    hits = HitCaller.call_hits_by_threshold(
        sample_data['percent_inhibition'].values,
        threshold=50,
        direction="greater"
    )

    print(f"\nHits identified: {np.sum(hits)}/{len(hits)} samples")

    return plate_df


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("ML-Fragment-Optimizer: Data Processing Examples")
    print("=" * 70)

    # Example 1
    df_clean = example_1_load_and_clean()

    # Example 2
    df_filtered = example_2_comprehensive_cleaning()

    # Example 3
    train, val, test = example_3_dataset_splitting()

    # Example 4
    df_public = example_4_load_public_dataset()

    # Example 5
    example_5_format_conversion()

    # Example 6
    plate_df = example_6_assay_data_processing()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
