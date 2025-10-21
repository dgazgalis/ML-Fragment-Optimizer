"""
Building block catalog management for fragment growing and optimization.

This module provides functions to load, search, and manage chemical building
block catalogs from commercial vendors (ZINC, Enamine REAL, eMolecules).
"""

from typing import Optional, List, Dict, Union, Tuple
import pandas as pd
from pathlib import Path
import requests
import json
import warnings
from dataclasses import dataclass
from enum import Enum
import sqlite3
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    raise ImportError("RDKit is required. Install with: pip install rdkit")


class CatalogSource(Enum):
    """Building block catalog sources."""
    ZINC = "zinc"
    ENAMINE_REAL = "enamine_real"
    EMOLECULES = "emolecules"
    MCULE = "mcule"
    CUSTOM = "custom"


@dataclass
class BuildingBlock:
    """Building block information."""
    catalog_id: str
    smiles: str
    source: str
    mw: float
    logp: float
    n_rotatable: int
    n_hbd: int
    n_hba: int
    price: Optional[float] = None
    availability: Optional[str] = None
    supplier: Optional[str] = None
    catalog_number: Optional[str] = None


class CatalogManager:
    """Manage building block catalogs."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize catalog manager.

        Args:
            db_path: Path to SQLite database for catalog
        """
        if db_path is None:
            db_path = Path.home() / ".ml_fragment_optimizer" / "building_blocks.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS building_blocks (
                catalog_id TEXT PRIMARY KEY,
                smiles TEXT NOT NULL,
                source TEXT NOT NULL,
                mw REAL,
                logp REAL,
                n_rotatable INTEGER,
                n_hbd INTEGER,
                n_hba INTEGER,
                price REAL,
                availability TEXT,
                supplier TEXT,
                catalog_number TEXT,
                date_added TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source ON building_blocks(source)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mw ON building_blocks(mw)
        """)

        conn.commit()
        conn.close()

    def add_building_block(self, bb: BuildingBlock):
        """
        Add building block to catalog.

        Args:
            bb: BuildingBlock object
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        from datetime import datetime
        date_added = datetime.now().isoformat()

        cursor.execute("""
            INSERT OR REPLACE INTO building_blocks
            (catalog_id, smiles, source, mw, logp, n_rotatable, n_hbd, n_hba,
             price, availability, supplier, catalog_number, date_added)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            bb.catalog_id, bb.smiles, bb.source, bb.mw, bb.logp,
            bb.n_rotatable, bb.n_hbd, bb.n_hba, bb.price,
            bb.availability, bb.supplier, bb.catalog_number, date_added
        ))

        conn.commit()
        conn.close()

    def add_from_smiles_list(
        self,
        smiles_list: List[str],
        source: str = "custom",
        batch_size: int = 1000
    ):
        """
        Add building blocks from SMILES list.

        Args:
            smiles_list: List of SMILES strings
            source: Catalog source name
            batch_size: Batch size for processing
        """
        print(f"Adding {len(smiles_list)} building blocks...")

        for i in tqdm(range(0, len(smiles_list), batch_size)):
            batch = smiles_list[i:i + batch_size]

            for idx, smiles in enumerate(batch):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                catalog_id = f"{source}_{i + idx:08d}"

                bb = BuildingBlock(
                    catalog_id=catalog_id,
                    smiles=smiles,
                    source=source,
                    mw=Descriptors.MolWt(mol),
                    logp=Descriptors.MolLogP(mol),
                    n_rotatable=Descriptors.NumRotatableBonds(mol),
                    n_hbd=Descriptors.NumHDonors(mol),
                    n_hba=Descriptors.NumHAcceptors(mol)
                )

                self.add_building_block(bb)

    def add_from_csv(
        self,
        filepath: Union[str, Path],
        smiles_col: str = "smiles",
        source: str = "custom",
        **kwargs
    ):
        """
        Add building blocks from CSV file.

        Args:
            filepath: Path to CSV file
            smiles_col: Column name for SMILES
            source: Catalog source name
            **kwargs: Additional column mappings
        """
        df = pd.read_csv(filepath)

        if smiles_col not in df.columns:
            raise ValueError(f"Column '{smiles_col}' not found")

        print(f"Loading {len(df)} building blocks from CSV...")

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row[smiles_col]
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                continue

            catalog_id = kwargs.get('id_col')
            if catalog_id and catalog_id in row:
                catalog_id = str(row[catalog_id])
            else:
                catalog_id = f"{source}_{idx:08d}"

            bb = BuildingBlock(
                catalog_id=catalog_id,
                smiles=smiles,
                source=source,
                mw=Descriptors.MolWt(mol),
                logp=Descriptors.MolLogP(mol),
                n_rotatable=Descriptors.NumRotatableBonds(mol),
                n_hbd=Descriptors.NumHDonors(mol),
                n_hba=Descriptors.NumHAcceptors(mol),
                price=row.get('price'),
                availability=row.get('availability'),
                supplier=row.get('supplier'),
                catalog_number=row.get('catalog_number')
            )

            self.add_building_block(bb)

    def search_by_properties(
        self,
        mw_range: Optional[Tuple[float, float]] = None,
        logp_range: Optional[Tuple[float, float]] = None,
        source: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Search building blocks by properties.

        Args:
            mw_range: Molecular weight range
            logp_range: LogP range
            source: Catalog source
            limit: Maximum results

        Returns:
            DataFrame with matching building blocks
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM building_blocks WHERE 1=1"
        params = []

        if mw_range:
            query += " AND mw BETWEEN ? AND ?"
            params.extend(mw_range)

        if logp_range:
            query += " AND logp BETWEEN ? AND ?"
            params.extend(logp_range)

        if source:
            query += " AND source = ?"
            params.append(source)

        query += f" LIMIT {limit}"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    def search_by_substructure(
        self,
        smarts: str,
        source: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Search by substructure (slow for large catalogs).

        Args:
            smarts: SMARTS pattern
            source: Catalog source
            limit: Maximum results

        Returns:
            DataFrame with matching building blocks
        """
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise ValueError("Invalid SMARTS pattern")

        # Get candidates from database
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM building_blocks"
        params = []

        if source:
            query += " WHERE source = ?"
            params.append(source)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Filter by substructure match
        matches = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Searching"):
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol and mol.HasSubstructMatch(pattern):
                matches.append(row)

                if len(matches) >= limit:
                    break

        return pd.DataFrame(matches)

    def get_statistics(self) -> Dict:
        """
        Get catalog statistics.

        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total count
        cursor.execute("SELECT COUNT(*) FROM building_blocks")
        total = cursor.fetchone()[0]

        # By source
        cursor.execute("""
            SELECT source, COUNT(*) as count
            FROM building_blocks
            GROUP BY source
        """)
        by_source = dict(cursor.fetchall())

        # Property ranges
        cursor.execute("""
            SELECT
                MIN(mw) as min_mw, MAX(mw) as max_mw, AVG(mw) as avg_mw,
                MIN(logp) as min_logp, MAX(logp) as max_logp, AVG(logp) as avg_logp
            FROM building_blocks
        """)
        props = cursor.fetchone()

        conn.close()

        return {
            'total': total,
            'by_source': by_source,
            'mw_range': (props[0], props[1]),
            'avg_mw': props[2],
            'logp_range': (props[3], props[4]),
            'avg_logp': props[5]
        }


class ZINCDownloader:
    """Download fragments from ZINC database."""

    BASE_URL = "https://zinc15.docking.org"

    @staticmethod
    def download_fragments(
        output_file: Union[str, Path],
        tranches: Optional[List[str]] = None,
        max_compounds: int = 10000
    ):
        """
        Download ZINC fragments.

        Args:
            output_file: Output CSV file
            tranches: ZINC tranches to download
            max_compounds: Maximum compounds to download
        """
        # This is a placeholder - actual ZINC API access requires registration
        warnings.warn("ZINC download requires API access. This is a placeholder.")

        # Example tranches for fragments (MW 150-250, LogP -1 to 3)
        if tranches is None:
            tranches = ["ABCD", "EFGH"]  # Example tranche codes

        print(f"Downloading ZINC fragments...")
        print(f"Note: Actual ZINC download requires API key and proper setup")

        # Placeholder implementation
        data = {
            'zinc_id': [],
            'smiles': [],
            'mw': [],
            'logp': []
        }

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

        print(f"Downloaded {len(df)} fragments (placeholder)")


class EnamineDownloader:
    """Download from Enamine REAL database."""

    BASE_URL = "https://enamine.net"

    @staticmethod
    def download_real_fragments(
        output_file: Union[str, Path],
        max_compounds: int = 10000
    ):
        """
        Download Enamine REAL fragments.

        Args:
            output_file: Output file
            max_compounds: Maximum compounds
        """
        warnings.warn("Enamine REAL requires commercial access. This is a placeholder.")

        print("Enamine REAL contains 38+ billion compounds")
        print("Direct download not supported - use Enamine's tools")


def create_sample_catalog(output_file: Union[str, Path], n_samples: int = 1000):
    """
    Create sample building block catalog for testing.

    Args:
        output_file: Output CSV file
        n_samples: Number of sample building blocks
    """
    # Common fragment SMILES
    fragment_smiles = [
        "c1ccccc1",  # Benzene
        "C1CCCCC1",  # Cyclohexane
        "c1ccncc1",  # Pyridine
        "c1cccnc1",  # Pyridine
        "C1CCNCC1",  # Piperidine
        "C1CCOCC1",  # Tetrahydropyran
        "c1ccc2ccccc2c1",  # Naphthalene
        "CC(C)C",  # Isopropyl
        "CC(C)(C)C",  # Tert-butyl
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "CN",  # Methylamine
        "CO",  # Methanol
    ]

    data = []
    for i in range(n_samples):
        base_smiles = fragment_smiles[i % len(fragment_smiles)]
        mol = Chem.MolFromSmiles(base_smiles)

        if mol is None:
            continue

        data.append({
            'catalog_id': f"SAMPLE_{i:06d}",
            'smiles': base_smiles,
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'n_rotatable': Descriptors.NumRotatableBonds(mol),
            'n_hbd': Descriptors.NumHDonors(mol),
            'n_hba': Descriptors.NumHAcceptors(mol),
            'price': 10.0 + (i % 100),
            'availability': 'in_stock',
            'supplier': 'Sample Vendor'
        })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

    print(f"Created sample catalog with {len(df)} building blocks")
    return df


if __name__ == "__main__":
    print("Building Block Catalog Manager")
    print("=" * 50)

    # Create sample catalog
    sample_file = Path("sample_building_blocks.csv")
    create_sample_catalog(sample_file, n_samples=100)

    # Initialize catalog manager
    db_path = Path("test_building_blocks.db")
    manager = CatalogManager(db_path)

    # Add building blocks
    print("\n" + "=" * 50)
    print("Adding building blocks to catalog...")
    manager.add_from_csv(sample_file, source="sample")

    # Get statistics
    print("\n" + "=" * 50)
    print("Catalog Statistics:")
    stats = manager.get_statistics()
    print(f"Total building blocks: {stats['total']}")
    print(f"By source: {stats['by_source']}")
    print(f"MW range: {stats['mw_range'][0]:.1f} - {stats['mw_range'][1]:.1f}")
    print(f"LogP range: {stats['logp_range'][0]:.1f} - {stats['logp_range'][1]:.1f}")

    # Search by properties
    print("\n" + "=" * 50)
    print("Searching by properties (MW 50-150)...")
    results = manager.search_by_properties(
        mw_range=(50, 150),
        limit=10
    )
    print(f"\nFound {len(results)} matching building blocks:")
    print(results[['catalog_id', 'smiles', 'mw', 'logp']].head())

    # Substructure search
    print("\n" + "=" * 50)
    print("Searching for benzene rings...")
    results = manager.search_by_substructure(
        smarts="c1ccccc1",
        limit=10
    )
    print(f"\nFound {len(results)} building blocks with benzene rings:")
    print(results[['catalog_id', 'smiles']].head())

    # Clean up
    sample_file.unlink()
    db_path.unlink()

    print("\nCatalog management test completed!")
