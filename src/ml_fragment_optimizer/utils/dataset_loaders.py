"""
Public dataset loaders for common molecular machine learning benchmarks.

This module provides functions to download and load popular datasets
including MoleculeNet, ChEMBL, PubChem, and ADMET datasets.
"""

from typing import Optional, Dict, List, Tuple
import pandas as pd
from pathlib import Path
import requests
from io import StringIO
import json
import warnings
from tqdm import tqdm
import hashlib

try:
    from rdkit import Chem
except ImportError:
    raise ImportError("RDKit is required. Install with: pip install rdkit")


# Dataset metadata
DATASET_REGISTRY = {
    "bace": {
        "name": "BACE",
        "description": "Binding to human β-secretase 1 (binary classification)",
        "task": "classification",
        "n_samples": 1513,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        "smiles_col": "mol",
        "target_col": "Class",
        "citation": "Subramanian et al. (2016)"
    },
    "bbbp": {
        "name": "BBBP",
        "description": "Blood-brain barrier penetration (binary classification)",
        "task": "classification",
        "n_samples": 2039,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "smiles_col": "smiles",
        "target_col": "p_np",
        "citation": "Martins et al. (2012)"
    },
    "clintox": {
        "name": "ClinTox",
        "description": "Clinical trial toxicity (multi-task classification)",
        "task": "classification",
        "n_samples": 1478,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv",
        "smiles_col": "smiles",
        "target_col": ["FDA_APPROVED", "CT_TOX"],
        "citation": "Gayvert et al. (2016)"
    },
    "esol": {
        "name": "ESOL",
        "description": "Aqueous solubility (regression)",
        "task": "regression",
        "n_samples": 1128,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "smiles_col": "smiles",
        "target_col": "measured log solubility in mols per litre",
        "citation": "Delaney (2004)"
    },
    "freesolv": {
        "name": "FreeSolv",
        "description": "Hydration free energy (regression)",
        "task": "regression",
        "n_samples": 642,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
        "smiles_col": "smiles",
        "target_col": "expt",
        "citation": "Mobley & Guthrie (2014)"
    },
    "lipophilicity": {
        "name": "Lipophilicity",
        "description": "Octanol/water distribution coefficient (regression)",
        "task": "regression",
        "n_samples": 4200,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        "smiles_col": "smiles",
        "target_col": "exp",
        "citation": "Gaulton et al. (2012)"
    },
    "hiv": {
        "name": "HIV",
        "description": "HIV replication inhibition (binary classification)",
        "task": "classification",
        "n_samples": 41127,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        "smiles_col": "smiles",
        "target_col": "HIV_active",
        "citation": "AIDS Antiviral Screen"
    },
    "tox21": {
        "name": "Tox21",
        "description": "Toxicity prediction (multi-task classification)",
        "task": "classification",
        "n_samples": 7831,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv",
        "smiles_col": "smiles",
        "target_col": [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
            "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE",
            "SR-MMP", "SR-p53"
        ],
        "citation": "Tox21 Challenge 2014"
    },
    "sider": {
        "name": "SIDER",
        "description": "Adverse drug reactions (multi-task classification)",
        "task": "classification",
        "n_samples": 1427,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv",
        "smiles_col": "smiles",
        "target_col": None,  # Multiple columns
        "citation": "Kuhn et al. (2016)"
    }
}


class DatasetDownloader:
    """Download and cache datasets."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize dataset downloader.

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".ml_fragment_optimizer" / "datasets"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_file(
        self,
        url: str,
        filename: Optional[str] = None,
        force_download: bool = False
    ) -> Path:
        """
        Download file with caching.

        Args:
            url: URL to download
            filename: Optional filename for cached file
            force_download: Force re-download even if cached

        Returns:
            Path to downloaded file
        """
        if filename is None:
            filename = url.split("/")[-1]

        filepath = self.cache_dir / filename

        if filepath.exists() and not force_download:
            print(f"Using cached file: {filepath}")
            return filepath

        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)

        print(f"Downloaded to: {filepath}")
        return filepath

    def verify_checksum(self, filepath: Path, expected_md5: str) -> bool:
        """
        Verify file checksum.

        Args:
            filepath: Path to file
            expected_md5: Expected MD5 hash

        Returns:
            True if checksum matches
        """
        md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)

        actual_md5 = md5.hexdigest()
        return actual_md5 == expected_md5


class MoleculeNetLoader:
    """Load MoleculeNet benchmark datasets."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize MoleculeNet loader.

        Args:
            cache_dir: Directory to cache datasets
        """
        self.downloader = DatasetDownloader(cache_dir)

    def load_dataset(
        self,
        name: str,
        force_download: bool = False,
        parse_molecules: bool = True
    ) -> pd.DataFrame:
        """
        Load MoleculeNet dataset.

        Args:
            name: Dataset name (e.g., 'bace', 'bbbp')
            force_download: Force re-download
            parse_molecules: Parse SMILES to mol objects

        Returns:
            DataFrame with dataset
        """
        name = name.lower()

        if name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {name}. "
                           f"Available: {list(DATASET_REGISTRY.keys())}")

        metadata = DATASET_REGISTRY[name]
        print(f"\nLoading {metadata['name']} dataset")
        print(f"Description: {metadata['description']}")
        print(f"Citation: {metadata['citation']}")

        # Download dataset
        filepath = self.downloader.download_file(
            metadata['url'],
            filename=f"{name}.csv",
            force_download=force_download
        )

        # Load CSV
        df = pd.read_csv(filepath)

        # Rename SMILES column
        smiles_col = metadata['smiles_col']
        if smiles_col != 'smiles':
            df['smiles'] = df[smiles_col]

        # Parse molecules
        if parse_molecules:
            print(f"Parsing {len(df)} SMILES strings...")
            df['mol'] = [Chem.MolFromSmiles(s) if pd.notna(s) else None
                        for s in tqdm(df['smiles'])]

            n_valid = df['mol'].notna().sum()
            print(f"Valid molecules: {n_valid}/{len(df)}")

        # Add metadata
        df.attrs['dataset_name'] = metadata['name']
        df.attrs['task'] = metadata['task']
        df.attrs['target_columns'] = metadata['target_col']

        return df

    def list_datasets(self) -> pd.DataFrame:
        """
        List available datasets.

        Returns:
            DataFrame with dataset information
        """
        data = []
        for key, meta in DATASET_REGISTRY.items():
            data.append({
                'name': key,
                'full_name': meta['name'],
                'task': meta['task'],
                'n_samples': meta['n_samples'],
                'description': meta['description']
            })

        return pd.DataFrame(data)


class ChEMBLLoader:
    """Load ChEMBL data via web services."""

    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize ChEMBL loader.

        Args:
            cache_dir: Directory to cache queries
        """
        self.downloader = DatasetDownloader(cache_dir)

    def fetch_target_activities(
        self,
        target_chembl_id: str,
        activity_type: str = "IC50",
        max_results: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch activities for a specific target.

        Args:
            target_chembl_id: ChEMBL target ID (e.g., 'CHEMBL1862')
            activity_type: Type of activity (IC50, Ki, etc.)
            max_results: Maximum number of results

        Returns:
            DataFrame with activities
        """
        print(f"Fetching {activity_type} data for {target_chembl_id}...")

        url = f"{self.BASE_URL}/activity.json"
        params = {
            'target_chembl_id': target_chembl_id,
            'standard_type': activity_type,
            'limit': max_results,
            'format': 'json'
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        if 'activities' not in data:
            warnings.warn(f"No activities found for {target_chembl_id}")
            return pd.DataFrame()

        activities = data['activities']

        # Parse data
        records = []
        for act in activities:
            records.append({
                'molecule_chembl_id': act.get('molecule_chembl_id'),
                'canonical_smiles': act.get('canonical_smiles'),
                'standard_type': act.get('standard_type'),
                'standard_value': act.get('standard_value'),
                'standard_units': act.get('standard_units'),
                'pchembl_value': act.get('pchembl_value'),
                'activity_comment': act.get('activity_comment'),
                'assay_chembl_id': act.get('assay_chembl_id')
            })

        df = pd.DataFrame(records)

        # Parse molecules
        print(f"Parsing {len(df)} molecules...")
        df['mol'] = [Chem.MolFromSmiles(s) if pd.notna(s) else None
                    for s in tqdm(df['canonical_smiles'])]

        print(f"Retrieved {len(df)} activities")

        return df

    def fetch_compound(self, chembl_id: str) -> Dict:
        """
        Fetch compound information.

        Args:
            chembl_id: ChEMBL compound ID

        Returns:
            Dictionary with compound data
        """
        url = f"{self.BASE_URL}/molecule/{chembl_id}.json"
        response = requests.get(url)
        response.raise_for_status()

        return response.json()


class ADMETLoader:
    """Load ADMET datasets from Therapeutics Data Commons."""

    TDC_BASE_URL = "https://raw.githubusercontent.com/mims-harvard/TDC/master/tdc_dataset"

    ADMET_DATASETS = {
        "caco2_wang": "adme/caco2_wang.csv",
        "bioavailability_ma": "adme/bioavailability_ma.csv",
        "lipophilicity_astrazeneca": "adme/lipophilicity_astrazeneca.csv",
        "solubility_aqsoldb": "adme/solubility_aqsoldb.csv",
        "hia_hou": "adme/hia_hou.csv",
        "pgp_broccatelli": "adme/pgp_broccatelli.csv",
        "bbb_martins": "adme/bbb_martins.csv",
        "ppbr_az": "adme/ppbr_az.csv",
        "vdss_lombardo": "adme/vdss_lombardo.csv",
        "cyp2c9_veith": "adme/cyp2c9_veith.csv",
        "cyp2d6_veith": "adme/cyp2d6_veith.csv",
        "cyp3a4_veith": "adme/cyp3a4_veith.csv",
        "cyp2c9_substrate_carbonmangels": "adme/cyp2c9_substrate_carbonmangels.csv",
        "cyp2d6_substrate_carbonmangels": "adme/cyp2d6_substrate_carbonmangels.csv",
        "cyp3a4_substrate_carbonmangels": "adme/cyp3a4_substrate_carbonmangels.csv",
        "half_life_obach": "adme/half_life_obach.csv",
        "clearance_microsome_az": "adme/clearance_microsome_az.csv",
        "clearance_hepatocyte_az": "adme/clearance_hepatocyte_az.csv",
        "ld50_zhu": "tox/ld50_zhu.csv",
        "herg": "tox/herg.csv",
        "ames": "tox/ames.csv",
        "dili": "tox/dili.csv",
        "skin_reaction": "tox/skin_reaction.csv"
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize ADMET loader.

        Args:
            cache_dir: Directory to cache datasets
        """
        self.downloader = DatasetDownloader(cache_dir)

    def load_dataset(
        self,
        name: str,
        force_download: bool = False,
        parse_molecules: bool = True
    ) -> pd.DataFrame:
        """
        Load ADMET dataset.

        Args:
            name: Dataset name
            force_download: Force re-download
            parse_molecules: Parse SMILES to mol objects

        Returns:
            DataFrame with dataset
        """
        if name not in self.ADMET_DATASETS:
            raise ValueError(f"Unknown ADMET dataset: {name}. "
                           f"Available: {list(self.ADMET_DATASETS.keys())}")

        url = f"{self.TDC_BASE_URL}/{self.ADMET_DATASETS[name]}"

        print(f"Loading {name} dataset...")

        # Download
        filepath = self.downloader.download_file(
            url,
            filename=f"admet_{name}.csv",
            force_download=force_download
        )

        # Load
        df = pd.read_csv(filepath)

        # Standardize column names
        if 'Drug' in df.columns:
            df['smiles'] = df['Drug']
        elif 'SMILES' in df.columns:
            df['smiles'] = df['SMILES']

        # Parse molecules
        if parse_molecules and 'smiles' in df.columns:
            print(f"Parsing {len(df)} molecules...")
            df['mol'] = [Chem.MolFromSmiles(s) if pd.notna(s) else None
                        for s in tqdm(df['smiles'])]

        return df

    def list_datasets(self) -> List[str]:
        """
        List available ADMET datasets.

        Returns:
            List of dataset names
        """
        return list(self.ADMET_DATASETS.keys())


def load_benchmark_splits(
    dataset_name: str,
    cache_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load benchmark dataset with literature splits.

    Args:
        dataset_name: Name of dataset
        cache_dir: Cache directory

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # This would load pre-defined train/val/test splits
    # For now, we'll just load the full dataset
    loader = MoleculeNetLoader(cache_dir)
    df = loader.load_dataset(dataset_name)

    # For demonstration, do a simple 70/15/15 split
    from .data_processing import DatasetSplitter

    splitter = DatasetSplitter()
    train_df, val_df, test_df = splitter.random_split(df)

    return train_df, val_df, test_df


def get_dataset_info(dataset_name: str) -> Dict:
    """
    Get information about a dataset.

    Args:
        dataset_name: Name of dataset

    Returns:
        Dictionary with dataset metadata
    """
    if dataset_name in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_name]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    print("Dataset Loaders")
    print("=" * 50)

    # List available MoleculeNet datasets
    print("\nMoleculeNet Datasets:")
    loader = MoleculeNetLoader()
    datasets_df = loader.list_datasets()
    print(datasets_df.to_string(index=False))

    # Example: Load BACE dataset
    print("\n" + "=" * 50)
    print("Loading BACE dataset...")
    bace_df = loader.load_dataset('bace')
    print(f"\nShape: {bace_df.shape}")
    print(f"Columns: {bace_df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(bace_df.head())

    # List ADMET datasets
    print("\n" + "=" * 50)
    print("\nADMET Datasets (TDC):")
    admet_loader = ADMETLoader()
    admet_datasets = admet_loader.list_datasets()
    print(f"Available: {len(admet_datasets)} datasets")
    for ds in admet_datasets[:10]:
        print(f"  - {ds}")
    print("  ...")
