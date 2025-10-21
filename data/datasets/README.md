# Datasets for ML-Fragment-Optimizer

This directory contains or references molecular datasets used for training and benchmarking machine learning models.

## Dataset Categories

### 1. MoleculeNet Benchmarks

Public benchmark datasets for molecular property prediction. Automatically downloaded via `dataset_loaders.py`.

**Classification Tasks:**
- **BACE**: β-secretase 1 inhibitors (1,513 compounds)
- **BBBP**: Blood-brain barrier penetration (2,039 compounds)
- **ClinTox**: Clinical trial toxicity (1,478 compounds)
- **HIV**: HIV replication inhibition (41,127 compounds)
- **Tox21**: Toxicity in 12 assays (7,831 compounds)
- **SIDER**: Adverse drug reactions (1,427 compounds)

**Regression Tasks:**
- **ESOL**: Aqueous solubility (1,128 compounds)
- **FreeSolv**: Hydration free energy (642 compounds)
- **Lipophilicity**: Octanol/water distribution (4,200 compounds)

### 2. ADMET Datasets (Therapeutics Data Commons)

ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) property datasets.

**Absorption:**
- Caco-2 permeability
- Bioavailability
- HIA (Human Intestinal Absorption)
- PAMPA permeability

**Distribution:**
- BBB (Blood-Brain Barrier) penetration
- PPB (Plasma Protein Binding)
- VDss (Volume of Distribution)

**Metabolism:**
- CYP450 inhibition (1A2, 2C9, 2C19, 2D6, 3A4)
- CYP450 substrate
- Half-life

**Excretion:**
- Clearance (microsomal, hepatocyte)

**Toxicity:**
- hERG cardiotoxicity
- AMES mutagenicity
- DILI (Drug-Induced Liver Injury)
- LD50 acute toxicity
- Skin reaction

### 3. ChEMBL Data

Bioactivity data extracted from ChEMBL database via web services API.

**Usage:**
```python
from src.utils.dataset_loaders import ChEMBLLoader

loader = ChEMBLLoader()

# Fetch IC50 data for a specific target
df = loader.fetch_target_activities(
    target_chembl_id='CHEMBL1862',  # Beta-secretase 1
    activity_type='IC50',
    max_results=5000
)
```

**Popular Targets:**
- CHEMBL1862: Beta-secretase 1 (BACE1)
- CHEMBL220: Acetylcholinesterase
- CHEMBL1827: Dopamine D2 receptor
- CHEMBL204: Thrombin
- CHEMBL1824: Carbonic anhydrase II

## Data Format

All datasets follow a standard format:

### CSV Format
```
mol_id,smiles,property1,property2,...
mol_001,CCO,1.23,active
mol_002,c1ccccc1,4.56,inactive
```

### Required Columns
- `mol_id`: Unique molecule identifier
- `smiles`: SMILES representation
- `mol`: RDKit mol object (added during loading)

### Optional Columns
- Activity/property values
- Metadata (source, date, etc.)
- Quality flags

## Download Instructions

### Automatic Download

Most datasets are automatically downloaded and cached:

```python
from src.utils.dataset_loaders import MoleculeNetLoader

loader = MoleculeNetLoader()

# Automatically downloads and caches
df = loader.load_dataset('bace')
```

### Cache Location

Datasets are cached in: `~/.ml_fragment_optimizer/datasets/`

### Manual Download

For large datasets not automatically downloaded:

1. **PubChem Bioassays**: Visit https://pubchem.ncbi.nlm.nih.gov/
2. **BindingDB**: Visit https://www.bindingdb.org/
3. **ZINC Fragments**: Visit https://zinc15.docking.org/

## Train/Test Splits

### Literature Splits

Use pre-defined splits from literature:

```python
from src.utils.dataset_loaders import load_benchmark_splits

train_df, val_df, test_df = load_benchmark_splits('bace')
```

### Custom Splits

```python
from src.utils.data_processing import DatasetSplitter

splitter = DatasetSplitter()

# Random split
train, val, test = splitter.random_split(df)

# Scaffold split (molecules with same scaffold stay together)
train, val, test = splitter.scaffold_split(df)

# Temporal split
train, val, test = splitter.temporal_split(
    df,
    date_column='date',
    train_end_date='2020-01-01',
    val_end_date='2021-01-01'
)
```

## Data Quality

### Data Cleaning Pipeline

```python
from src.utils.data_processing import MoleculeDataCleaner

cleaner = MoleculeDataCleaner(
    remove_salts=True,
    neutralize=True,
    remove_invalid=True,
    remove_duplicates=True
)

df_clean = cleaner.clean(df)
```

### Advanced Cleaning

```python
from src.utils.molecular_cleaning import ComprehensiveCleaner

cleaner = ComprehensiveCleaner(
    standardize=True,
    remove_pains=True,
    remove_aggregators=True,
    remove_reactive=True,
    apply_rule_of_five=True
)

df_clean, stats = cleaner.clean(df)
```

## Known Issues

### MoleculeNet Datasets

- **HIV**: Highly imbalanced (95% inactive)
- **Tox21**: Many missing values (sparse multi-task)
- **SIDER**: Small dataset, may overfit

### ChEMBL Data

- Activity values may have different units
- Multiple measurements per compound (replicates)
- Data quality varies by source

### ADMET Datasets

- Different assay protocols across datasets
- Species differences (human vs animal)
- Limited to specific chemical space

## Citations

### MoleculeNet
```
Wu et al. (2018) MoleculeNet: a benchmark for molecular machine learning.
Chemical Science, 9(2), 513-530.
```

### Therapeutics Data Commons
```
Huang et al. (2021) Therapeutics Data Commons: Machine Learning Datasets
and Tasks for Drug Discovery and Development. arXiv:2102.09548
```

### ChEMBL
```
Gaulton et al. (2017) The ChEMBL database in 2017.
Nucleic Acids Research, 45(D1), D945-D954.
```

## Adding Custom Datasets

### From CSV/SMILES File

```python
from src.utils.data_processing import MoleculeDataLoader

df = MoleculeDataLoader.from_smiles_file(
    'my_dataset.csv',
    smiles_col='SMILES',
    id_col='ID'
)
```

### From SDF File

```python
df = MoleculeDataLoader.from_sdf(
    'my_molecules.sdf',
    id_property='MOL_ID'
)
```

### Save Processed Dataset

```python
# Save to CSV (without mol objects)
df[['mol_id', 'smiles', 'activity']].to_csv('processed.csv', index=False)

# Save to SDF (with 3D coords)
from src.utils.format_converters import FileConverter

FileConverter.smiles_to_sdf(
    'processed.csv',
    'processed.sdf',
    generate_3d=True
)
```

## Best Practices

1. **Always clean data** before training
2. **Use scaffold splits** for drug-like molecules
3. **Check for duplicates** between train/test
4. **Document preprocessing** steps
5. **Version control** dataset versions
6. **Cite data sources** properly

## Support

For issues with datasets:
- Check dataset loader logs
- Verify network connectivity
- Clear cache if corrupted: `rm -rf ~/.ml_fragment_optimizer/datasets/`
- Report issues on GitHub
