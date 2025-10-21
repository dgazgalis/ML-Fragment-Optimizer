# Data Utilities Guide

Complete guide to data processing, cleaning, and management utilities in ML-Fragment-Optimizer.

## Table of Contents

1. [Overview](#overview)
2. [Data Loading](#data-loading)
3. [Data Cleaning](#data-cleaning)
4. [Dataset Splitting](#dataset-splitting)
5. [Public Datasets](#public-datasets)
6. [Assay Processing](#assay-processing)
7. [Molecular Cleaning](#molecular-cleaning)
8. [Format Conversion](#format-conversion)
9. [Building Block Catalogs](#building-block-catalogs)
10. [Benchmarks](#benchmarks)
11. [Best Practices](#best-practices)

## Overview

The data utilities provide comprehensive tools for:
- Loading molecules from various sources
- Cleaning and standardizing structures
- Processing biological assay data
- Managing building block catalogs
- Creating benchmark datasets

## Data Loading

### Load from SMILES File

```python
from src.utils import MoleculeDataLoader

loader = MoleculeDataLoader()

# Load from CSV
df = loader.from_smiles_file(
    'molecules.csv',
    smiles_col='SMILES',
    id_col='ID'
)
```

### Load from SDF File

```python
df = loader.from_sdf(
    'molecules.sdf',
    id_property='MOL_ID',
    sanitize=True
)
```

### Load from SMILES List

```python
smiles_list = ['CCO', 'CC(=O)O', 'c1ccccc1']
ids = ['mol1', 'mol2', 'mol3']

df = loader.from_smiles_list(
    smiles_list,
    ids=ids,
    activity=[1.2, 3.4, 5.6]
)
```

## Data Cleaning

### Basic Cleaning

```python
from src.utils import MoleculeDataCleaner

cleaner = MoleculeDataCleaner(
    remove_salts=True,
    neutralize=True,
    remove_invalid=True,
    remove_duplicates=True,
    keep_largest_fragment=True
)

df_clean = cleaner.clean(df)
```

### What Gets Removed

1. **Invalid molecules**: Can't be parsed by RDKit
2. **Salts**: Counter-ions and solvents
3. **Charged species**: Neutralized where appropriate
4. **Duplicates**: Based on InChIKey

### Comprehensive Cleaning

```python
from src.utils import ComprehensiveCleaner

cleaner = ComprehensiveCleaner(
    standardize=True,              # Canonical forms
    remove_pains=True,             # PAINS compounds
    remove_aggregators=True,       # Potential aggregators
    remove_reactive=True,          # Reactive groups
    apply_rule_of_five=True,       # Lipinski's rules
    apply_rule_of_three=False,     # Fragment rules
    remove_duplicates=True
)

df_clean, stats = cleaner.clean(df)

print(stats)
# {'initial': 1000, 'after_pains_removal': 950, ..., 'final': 850}
```

## Dataset Splitting

### Random Split

```python
from src.utils import DatasetSplitter

splitter = DatasetSplitter()

train, val, test = splitter.random_split(
    df,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42
)
```

### Scaffold Split (Recommended)

Ensures molecules with same scaffold are in same set.

```python
train, val, test = splitter.scaffold_split(
    df,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42
)
```

### Temporal Split

For time-series data.

```python
train, val, test = splitter.temporal_split(
    df,
    date_column='date',
    train_end_date='2020-01-01',
    val_end_date='2021-01-01'
)
```

### Stratified Split

Maintains class distribution.

```python
train, val, test = splitter.stratified_split(
    df,
    stratify_column='activity_class',
    train_size=0.7,
    val_size=0.15,
    test_size=0.15
)
```

## Public Datasets

### MoleculeNet

```python
from src.utils import MoleculeNetLoader

loader = MoleculeNetLoader()

# List available datasets
datasets = loader.list_datasets()
print(datasets)

# Load dataset
df = loader.load_dataset('bace')  # Beta-secretase 1
```

**Available datasets:**
- `bace`: β-secretase 1 inhibitors (1,513)
- `bbbp`: Blood-brain barrier penetration (2,039)
- `clintox`: Clinical trial toxicity (1,478)
- `esol`: Aqueous solubility (1,128)
- `freesolv`: Hydration free energy (642)
- `lipophilicity`: Octanol/water distribution (4,200)
- `hiv`: HIV replication inhibition (41,127)
- `tox21`: Multi-task toxicity (7,831)
- `sider`: Adverse drug reactions (1,427)

### ChEMBL

```python
from src.utils import ChEMBLLoader

loader = ChEMBLLoader()

# Fetch IC50 data for specific target
df = loader.fetch_target_activities(
    target_chembl_id='CHEMBL1862',  # BACE1
    activity_type='IC50',
    max_results=5000
)
```

### ADMET Datasets

```python
from src.utils import ADMETLoader

loader = ADMETLoader()

# List available
datasets = loader.list_datasets()

# Load specific dataset
df = loader.load_dataset('caco2_wang')  # Permeability
df = loader.load_dataset('herg')        # Cardiotoxicity
df = loader.load_dataset('ames')        # Mutagenicity
```

## Assay Processing

### Plate Data Processing

```python
from src.utils import (
    PlateReaderParser,
    QualityController,
    AssayNormalizer,
    HitCaller
)

# Parse plate reader output
parser = PlateReaderParser()
plate_df = parser.parse_generic_matrix('plate_data.csv')

# Quality control
positive_wells = ['A01', 'A02', 'A03']
negative_wells = ['H10', 'H11', 'H12']

qc_metrics = QualityController.assess_plate_quality(
    plate_df,
    positive_wells,
    negative_wells
)

print(f"Z'-factor: {qc_metrics.z_prime_factor:.3f}")
print(f"Passed: {qc_metrics.passed}")
```

### Normalization

```python
normalizer = AssayNormalizer()

# Percent inhibition
pos_mean = plate_df[plate_df['well'].isin(positive_wells)]['value'].mean()
neg_mean = plate_df[plate_df['well'].isin(negative_wells)]['value'].mean()

plate_df['percent_inhibition'] = normalizer.percent_inhibition(
    plate_df['value'].values,
    pos_mean,
    neg_mean
)

# Z-scores
plate_df['z_score'] = normalizer.z_score(plate_df['value'].values)

# B-scores (corrects row/column effects)
plate_df['b_score'] = normalizer.b_score(plate_df)
```

### Hit Calling

```python
caller = HitCaller()

# Threshold-based
hits = caller.call_hits_by_threshold(
    plate_df['percent_inhibition'].values,
    threshold=50,
    direction='greater'
)

# Z-score based
hits = caller.call_hits_by_zscore(
    plate_df['value'].values,
    threshold=3.0,
    direction='greater'
)

# Activity classification
categories = caller.classify_activity(
    plate_df['percent_inhibition'].values,
    labels=['inactive', 'weak', 'moderate', 'strong'],
    thresholds=[25, 50, 75]
)
```

### Dose-Response Analysis

```python
from src.utils import DoseResponseAnalyzer

analyzer = DoseResponseAnalyzer()

concentrations = np.array([0.1, 1, 10, 100, 1000])  # nM
responses = np.array([5, 15, 50, 85, 95])  # % inhibition

fit = analyzer.fit_dose_response(concentrations, responses)

print(f"IC50: {fit['ic50']:.2f} nM")
print(f"Hill slope: {fit['hill_slope']:.2f}")
print(f"R²: {fit['r_squared']:.3f}")
```

## Molecular Cleaning

### PAINS Filtering

```python
from src.utils import PAINSFilter

pains_filter = PAINSFilter()

for mol in df['mol']:
    is_pains, matches = pains_filter.is_pains(mol)
    if is_pains:
        print(f"PAINS compound: {matches}")
```

### Aggregator Detection

```python
from src.utils import AggregatorFilter

agg_filter = AggregatorFilter()

for mol in df['mol']:
    if agg_filter.is_aggregator(mol):
        print("Potential aggregator detected")
```

### Reactive Groups

```python
from src.utils import ReactiveFilter

reactive_filter = ReactiveFilter()

for mol in df['mol']:
    has_reactive, groups = reactive_filter.has_reactive_groups(mol)
    if has_reactive:
        print(f"Reactive groups: {groups}")
```

### Property Filtering

```python
from src.utils import PropertyFilter

# Rule of Five
ro5_filter = PropertyFilter()
passes, props = ro5_filter.passes_filters(mol)

# Or use static method
passes, n_violations = PropertyFilter.rule_of_five(mol)

# Rule of Three (fragments)
passes, n_violations = PropertyFilter.rule_of_three(mol)

# Custom ranges
custom_filter = PropertyFilter(
    mw_range=(100, 500),
    logp_range=(-2, 5),
    hbd_range=(0, 5),
    hba_range=(0, 10)
)
```

## Format Conversion

### SMILES to SDF

```python
from src.utils import FileConverter

FileConverter.smiles_to_sdf(
    'molecules.csv',
    'molecules.sdf',
    smiles_col='SMILES',
    generate_3d=True  # Generate 3D coordinates
)
```

### SDF to SMILES

```python
FileConverter.sdf_to_smiles(
    'molecules.sdf',
    'molecules.csv',
    include_properties=True
)
```

### Add Identifiers

```python
from src.utils import IdentifierConverter

converter = IdentifierConverter()
df = converter.add_identifiers(df)

# Adds: canonical_smiles, inchi, inchikey, formula
```

### Large File Processing

```python
from src.utils import BatchProcessor

processor = BatchProcessor()

for batch_result in processor.process_large_sdf(
    'large_file.sdf',
    process_fn=my_processing_function,
    batch_size=1000
):
    # Process each batch
    pass
```

## Building Block Catalogs

### Initialize Catalog

```python
from data.building_blocks.catalog_manager import CatalogManager

manager = CatalogManager()  # Uses default SQLite DB

# Or specify custom location
manager = CatalogManager(db_path='my_catalog.db')
```

### Add Building Blocks

```python
# From CSV
manager.add_from_csv(
    'building_blocks.csv',
    smiles_col='SMILES',
    source='vendor_a'
)

# From SMILES list
smiles_list = ['c1ccccc1', 'C1CCCCC1', 'c1ccncc1']
manager.add_from_smiles_list(smiles_list, source='fragments')
```

### Search Catalog

```python
# By properties
results = manager.search_by_properties(
    mw_range=(100, 300),
    logp_range=(-1, 3),
    source='fragments',
    limit=1000
)

# By substructure
results = manager.search_by_substructure(
    'c1ccccc1',  # SMARTS pattern
    limit=100
)
```

### Catalog Statistics

```python
stats = manager.get_statistics()

print(f"Total: {stats['total']}")
print(f"By source: {stats['by_source']}")
print(f"MW range: {stats['mw_range']}")
```

## Benchmarks

### Activity Cliffs

```python
from data.benchmarks.benchmark_data import ActivityCliffBenchmark

detector = ActivityCliffBenchmark()
cliff_pairs = detector.generate_activity_cliff_pairs(
    df,
    activity_col='pIC50',
    similarity_threshold=0.85,
    activity_diff_threshold=2.0
)
```

### Scaffold Split Benchmark

```python
from data.benchmarks.benchmark_data import ScaffoldSplitBenchmark

benchmark = ScaffoldSplitBenchmark()
splits = benchmark.create_scaffold_split_benchmark(df)

train = splits['train']
val = splits['val']
test = splits['test']
metadata = splits['metadata']
```

### External Validation

```python
from data.benchmarks.benchmark_data import ExternalValidationSet

dev_set, ext_val_set = ExternalValidationSet.create_external_validation_set(
    df,
    fraction=0.2
)
```

## Best Practices

### 1. Data Cleaning Pipeline

```python
# Recommended cleaning pipeline
from src.utils import (
    MoleculeDataLoader,
    MoleculeDataCleaner,
    ComprehensiveCleaner,
    calculate_dataset_statistics
)

# 1. Load data
loader = MoleculeDataLoader()
df = loader.from_smiles_file('molecules.csv')

# 2. Basic cleaning
basic_cleaner = MoleculeDataCleaner()
df = basic_cleaner.clean(df)

# 3. Advanced filtering
advanced_cleaner = ComprehensiveCleaner(
    remove_pains=True,
    remove_aggregators=True,
    apply_rule_of_five=True
)
df, stats = advanced_cleaner.clean(df)

# 4. Check statistics
stats = calculate_dataset_statistics(df)
print(stats)
```

### 2. Dataset Splitting

```python
# Use scaffold split for drug-like molecules
splitter = DatasetSplitter()
train, val, test = splitter.scaffold_split(df)

# Verify no scaffold overlap
# This is important for realistic evaluation
```

### 3. Quality Control

```python
# Always check data quality
print(f"Valid molecules: {df['mol'].notna().sum()}/{len(df)}")
print(f"Unique SMILES: {df['smiles'].nunique()}")

# Check for leakage between sets
train_smiles = set(train['smiles'])
test_smiles = set(test['smiles'])
overlap = train_smiles & test_smiles
print(f"Train/test overlap: {len(overlap)}")  # Should be 0
```

### 4. Handling Missing Data

```python
from src.utils import impute_missing_values

# Impute missing activity values
df = impute_missing_values(
    df,
    columns=['activity'],
    strategy='median'  # or 'mean', 'mode', 'drop'
)
```

### 5. Memory Management

```python
# For large datasets, use batch processing
from src.utils import batch_process

def process_molecule(mol):
    # Your processing function
    return result

results = batch_process(
    df['mol'].tolist(),
    process_fn=process_molecule,
    batch_size=1000,
    desc="Processing molecules"
)
```

## Common Pitfalls

### 1. Forgetting to Remove Duplicates

```python
# Always check and remove duplicates
from src.utils import DuplicateRemover

df = DuplicateRemover.remove_by_inchikey(df)
```

### 2. Not Standardizing Molecules

```python
# Standardize before training
from src.utils import MolecularStandardizer

standardizer = MolecularStandardizer()
df['mol'] = [standardizer.standardize(mol) for mol in df['mol']]
```

### 3. Ignoring PAINS in Virtual Screening

```python
# Remove PAINS for HTS/virtual screening
pains_filter = PAINSFilter()
df['is_pains'] = [pains_filter.is_pains(mol)[0] for mol in df['mol']]
df = df[~df['is_pains']]
```

### 4. Using Wrong Split Strategy

```python
# ❌ Don't use random split for drug discovery
train, val, test = splitter.random_split(df)  # Can leak similar molecules

# ✅ Use scaffold split instead
train, val, test = splitter.scaffold_split(df)  # More realistic
```

## Performance Tips

### 1. Vectorize Operations

```python
# Use numpy/pandas vectorized operations
df['mw'] = df['mol'].apply(lambda m: Descriptors.MolWt(m) if m else None)
```

### 2. Use Caching

```python
# MoleculeNet datasets are automatically cached
loader = MoleculeNetLoader()
df = loader.load_dataset('bace')  # Cached after first download
```

### 3. Parallel Processing

```python
# For CPU-intensive tasks
from multiprocessing import Pool

def process_mol(mol):
    # Heavy computation
    return result

with Pool() as pool:
    results = pool.map(process_mol, df['mol'])
```

## Troubleshooting

### Issue: Invalid SMILES

```python
# Check for invalid SMILES
invalid = df[df['mol'].isna()]
print(f"Invalid SMILES: {len(invalid)}")
print(invalid['smiles'])
```

### Issue: Memory Error

```python
# Use batch processing
from src.utils import BatchProcessor

processor = BatchProcessor()
# Process in chunks
```

### Issue: Download Fails

```python
# Check network connection
# Clear cache if corrupted
import shutil
from pathlib import Path

cache_dir = Path.home() / '.ml_fragment_optimizer' / 'datasets'
shutil.rmtree(cache_dir)  # Remove cache
```

## API Reference

See module docstrings for complete API documentation:

```python
from src.utils import MoleculeDataLoader
help(MoleculeDataLoader)
```

## Support

For issues or questions:
- Check this guide first
- Search GitHub issues
- Create new issue with reproducible example
