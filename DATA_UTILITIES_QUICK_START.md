# Data Utilities Quick Start

Fast reference for common data processing tasks in ML-Fragment-Optimizer.

## Installation

```bash
# Install dependencies
conda install -c conda-forge rdkit pandas numpy scipy scikit-learn requests tqdm

# Or use pip
pip install -r requirements_data.txt
```

## Basic Workflow

### 1. Load Data

```python
from src.utils import MoleculeDataLoader

loader = MoleculeDataLoader()

# From CSV
df = loader.from_smiles_file('molecules.csv', smiles_col='SMILES')

# From SDF
df = loader.from_sdf('molecules.sdf')

# From list
df = loader.from_smiles_list(['CCO', 'CC(=O)O', 'c1ccccc1'])
```

### 2. Clean Data

```python
from src.utils import MoleculeDataCleaner

cleaner = MoleculeDataCleaner(
    remove_salts=True,
    neutralize=True,
    remove_invalid=True,
    remove_duplicates=True
)

df_clean = cleaner.clean(df)
```

### 3. Advanced Cleaning

```python
from src.utils import ComprehensiveCleaner

cleaner = ComprehensiveCleaner(
    standardize=True,
    remove_pains=True,
    remove_aggregators=True,
    remove_reactive=True,
    apply_rule_of_five=True
)

df_clean, stats = cleaner.clean(df)
print(stats)
```

### 4. Split Dataset

```python
from src.utils import DatasetSplitter

splitter = DatasetSplitter()

# Scaffold split (recommended)
train, val, test = splitter.scaffold_split(df_clean)

# Or random split
train, val, test = splitter.random_split(df_clean)
```

## Load Public Datasets

```python
from src.utils import MoleculeNetLoader

loader = MoleculeNetLoader()

# Load BACE dataset
df = loader.load_dataset('bace')

# List available datasets
datasets = loader.list_datasets()
print(datasets)
```

## Process Assay Data

```python
from src.utils import (
    QualityController,
    AssayNormalizer,
    HitCaller
)

# Quality control
qc_metrics = QualityController.assess_plate_quality(
    plate_df,
    positive_wells=['A01', 'A02'],
    negative_wells=['H11', 'H12']
)

print(f"Z'-factor: {qc_metrics.z_prime_factor:.3f}")

# Normalize
normalizer = AssayNormalizer()
plate_df['percent_inhibition'] = normalizer.percent_inhibition(
    plate_df['value'].values,
    positive_control_mean,
    negative_control_mean
)

# Call hits
hits = HitCaller.call_hits_by_threshold(
    plate_df['percent_inhibition'].values,
    threshold=50,
    direction='greater'
)
```

## Filter by Properties

```python
from src.utils import PropertyFilter

# Rule of Five
prop_filter = PropertyFilter()
passes, props = prop_filter.passes_filters(mol)

# Or static method
from rdkit import Chem
mol = Chem.MolFromSmiles('CCO')
passes, n_violations = PropertyFilter.rule_of_five(mol)

# Rule of Three (fragments)
passes, n_violations = PropertyFilter.rule_of_three(mol)
```

## Remove PAINS

```python
from src.utils import PAINSFilter

pains_filter = PAINSFilter()

for mol in df['mol']:
    is_pains, matches = pains_filter.is_pains(mol)
    if is_pains:
        print(f"PAINS: {matches}")
```

## Convert Formats

```python
from src.utils import FileConverter

# SMILES to SDF
FileConverter.smiles_to_sdf(
    'molecules.csv',
    'molecules.sdf',
    generate_3d=False
)

# SDF to SMILES
FileConverter.sdf_to_smiles(
    'molecules.sdf',
    'molecules.csv',
    include_properties=True
)
```

## Manage Building Blocks

```python
from data.building_blocks.catalog_manager import CatalogManager

manager = CatalogManager()

# Add from CSV
manager.add_from_csv('building_blocks.csv', source='vendor')

# Search by properties
results = manager.search_by_properties(
    mw_range=(100, 300),
    logp_range=(-1, 3)
)

# Substructure search
results = manager.search_by_substructure('c1ccccc1')  # Benzene
```

## Calculate Statistics

```python
from src.utils import calculate_dataset_statistics

stats = calculate_dataset_statistics(df)

print(f"Valid: {stats.n_valid}/{stats.n_molecules}")
print(f"MW: {stats.mw_mean:.1f} ± {stats.mw_std:.1f}")
print(f"LogP: {stats.logp_mean:.2f} ± {stats.logp_std:.2f}")
```

## Dose-Response Analysis

```python
from src.utils import DoseResponseAnalyzer
import numpy as np

analyzer = DoseResponseAnalyzer()

concentrations = np.array([0.1, 1, 10, 100, 1000])
responses = np.array([10, 25, 50, 75, 90])

fit = analyzer.fit_dose_response(concentrations, responses)

print(f"IC50: {fit['ic50']:.2f}")
print(f"Hill slope: {fit['hill_slope']:.2f}")
print(f"R²: {fit['r_squared']:.3f}")
```

## Common Workflows

### Complete Pipeline

```python
from src.utils import (
    MoleculeDataLoader,
    ComprehensiveCleaner,
    DatasetSplitter,
    calculate_dataset_statistics
)

# 1. Load
loader = MoleculeDataLoader()
df = loader.from_smiles_file('molecules.csv')

# 2. Clean
cleaner = ComprehensiveCleaner(remove_pains=True)
df_clean, stats = cleaner.clean(df)
print(f"Cleaned: {stats['initial']} → {stats['final']}")

# 3. Split
splitter = DatasetSplitter()
train, val, test = splitter.scaffold_split(df_clean)

# 4. Statistics
train_stats = calculate_dataset_statistics(train)
print(f"Train: {train_stats.n_valid} molecules")
```

### Assay Processing Pipeline

```python
from src.utils import (
    PlateReaderParser,
    QualityController,
    AssayNormalizer,
    ReplicateHandler,
    HitCaller
)

# 1. Parse
parser = PlateReaderParser()
plate_df = parser.parse_generic_matrix('plate.csv')

# 2. QC
qc = QualityController.assess_plate_quality(
    plate_df, positive_wells, negative_wells
)

if qc.passed:
    # 3. Normalize
    normalizer = AssayNormalizer()
    plate_df['normalized'] = normalizer.percent_inhibition(...)

    # 4. Handle replicates
    handler = ReplicateHandler()
    agg_df = handler.aggregate_replicates(
        plate_df,
        group_cols=['compound_id'],
        value_col='normalized'
    )

    # 5. Call hits
    hits = HitCaller.call_hits_by_threshold(
        agg_df['normalized_mean'].values,
        threshold=50
    )
```

## Tips

### Memory Efficiency

```python
# For large files, use batch processing
from src.utils import batch_process

results = batch_process(
    items,
    process_fn=my_function,
    batch_size=1000
)
```

### Check Data Quality

```python
# Always check validity
print(f"Valid: {df['mol'].notna().sum()}/{len(df)}")

# Check duplicates
print(f"Unique SMILES: {df['smiles'].nunique()}")

# Check for train/test leakage
train_set = set(train['smiles'])
test_set = set(test['smiles'])
overlap = train_set & test_set
print(f"Overlap: {len(overlap)}")  # Should be 0
```

### Impute Missing Values

```python
from src.utils import impute_missing_values

df = impute_missing_values(
    df,
    columns=['activity'],
    strategy='median'
)
```

## Available Datasets

### MoleculeNet
- `bace` - β-secretase 1 (1,513)
- `bbbp` - BBB penetration (2,039)
- `clintox` - Toxicity (1,478)
- `esol` - Solubility (1,128)
- `freesolv` - Free energy (642)
- `lipophilicity` - LogD (4,200)
- `hiv` - HIV inhibition (41,127)
- `tox21` - Multi-task tox (7,831)
- `sider` - Side effects (1,427)

### ADMET (TDC)
- Absorption: `caco2_wang`, `bioavailability_ma`, `hia_hou`
- Distribution: `bbb_martins`, `ppbr_az`, `vdss_lombardo`
- Metabolism: `cyp2c9_veith`, `cyp2d6_veith`, `cyp3a4_veith`
- Toxicity: `herg`, `ames`, `dili`, `ld50_zhu`

## Documentation

- **Complete Guide**: `docs/DATA_UTILITIES_GUIDE.md`
- **Summary**: `DATA_UTILITIES_SUMMARY.md`
- **Examples**: `examples/data_processing_example.py`
- **Tests**: `tests/test_data_utils.py`

## Run Examples

```bash
# Run complete examples
python examples/data_processing_example.py

# Run tests
pytest tests/test_data_utils.py -v
```

## Common Issues

**Invalid SMILES**
```python
# Check and report
invalid = df[df['mol'].isna()]
print(f"Invalid: {len(invalid)}")
print(invalid['smiles'])
```

**Memory Error**
```python
# Use batch processing
from src.utils import BatchProcessor
processor = BatchProcessor()
# Process in chunks
```

**Download Fails**
```bash
# Clear cache
rm -rf ~/.ml_fragment_optimizer/datasets/
```

## Get Help

```python
# Module documentation
from src.utils import MoleculeDataLoader
help(MoleculeDataLoader)

# Function documentation
help(MoleculeDataLoader.from_smiles_file)
```

## Next Steps

1. **Load your data** using appropriate loader
2. **Clean thoroughly** with ComprehensiveCleaner
3. **Split properly** using scaffold split
4. **Calculate statistics** to understand your dataset
5. **Train models** using clean, split data

See `docs/DATA_UTILITIES_GUIDE.md` for complete documentation.
