"""
Building Block Availability Checker

Check commercial availability of molecules and fragments from various sources:
1. Local catalog (CSV file with SMILES and supplier info)
2. ZINC database API (if available)
3. Enamine REAL database (if available)
4. PubChem similarity search (fallback)

Author: ML-Fragment-Optimizer Team
"""

import logging
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from functools import lru_cache
import urllib.request
import urllib.parse
import urllib.error
import time

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. Building block checking will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class SupplierInfo:
    """Information about chemical supplier"""
    name: str
    catalog_id: str
    price_usd: Optional[float] = None
    lead_time_days: Optional[int] = None
    purity: Optional[str] = None
    quantity_mg: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'supplier': self.name,
            'catalog_id': self.catalog_id,
            'price_usd': self.price_usd,
            'lead_time_days': self.lead_time_days,
            'purity': self.purity,
            'quantity_mg': self.quantity_mg
        }


@dataclass
class AvailabilityResult:
    """Result of availability check"""
    smiles: str
    is_available: bool
    exact_match: bool
    suppliers: List[SupplierInfo] = field(default_factory=list)
    similar_compounds: List[Tuple[str, float]] = field(default_factory=list)  # (SMILES, similarity)
    source: str = "unknown"  # Which database found it

    def to_dict(self) -> Dict:
        return {
            'smiles': self.smiles,
            'is_available': self.is_available,
            'exact_match': self.exact_match,
            'num_suppliers': len(self.suppliers),
            'suppliers': [s.to_dict() for s in self.suppliers],
            'similar_compounds': [
                {'smiles': s, 'similarity': round(sim, 3)}
                for s, sim in self.similar_compounds[:5]  # Top 5
            ],
            'source': self.source
        }

    def get_best_price(self) -> Optional[float]:
        """Get lowest price across suppliers"""
        prices = [s.price_usd for s in self.suppliers if s.price_usd is not None]
        return min(prices) if prices else None

    def get_fastest_delivery(self) -> Optional[int]:
        """Get shortest lead time across suppliers"""
        lead_times = [s.lead_time_days for s in self.suppliers if s.lead_time_days is not None]
        return min(lead_times) if lead_times else None


class BuildingBlockChecker:
    """
    Check commercial availability of building blocks

    Supports multiple data sources with automatic fallback:
    1. Local catalog (fastest, most reliable)
    2. ZINC15 API
    3. Enamine REAL (if configured)
    4. PubChem similarity (fallback)

    Setup:
        1. Create local catalog CSV with columns: smiles, supplier, catalog_id, price, lead_time
        2. Configure API keys if using commercial databases
        3. Set similarity threshold for approximate matches

    Example:
        >>> checker = BuildingBlockChecker('catalogs/building_blocks.csv')
        >>> result = checker.check('CCO')  # Check ethanol
        >>> if result.is_available:
        ...     print(f"Available from {len(result.suppliers)} suppliers")
    """

    def __init__(
        self,
        local_catalog_path: Optional[Path] = None,
        use_zinc: bool = True,
        use_enamine: bool = False,
        similarity_threshold: float = 0.85,
        cache_size: int = 10000
    ):
        """
        Initialize building block checker

        Args:
            local_catalog_path: Path to CSV file with local building block catalog
            use_zinc: Enable ZINC15 API queries
            use_enamine: Enable Enamine REAL database queries
            similarity_threshold: Minimum Tanimoto similarity for similar compounds (0-1)
            cache_size: Number of results to cache in memory
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for building block checking")

        self.local_catalog: Dict[str, List[SupplierInfo]] = {}
        self.local_fingerprints: Dict[str, object] = {}  # For similarity search
        self.use_zinc = use_zinc
        self.use_enamine = use_enamine
        self.similarity_threshold = similarity_threshold

        # Load local catalog
        if local_catalog_path and Path(local_catalog_path).exists():
            self._load_local_catalog(local_catalog_path)
            logger.info(f"Loaded {len(self.local_catalog)} compounds from local catalog")
        else:
            logger.warning("No local catalog provided. Availability checking will rely on APIs.")

        # Configure cache
        self.check = lru_cache(maxsize=cache_size)(self._check_impl)

    def _load_local_catalog(self, catalog_path: Path) -> None:
        """Load local building block catalog from CSV"""
        try:
            with open(catalog_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    smiles = row.get('smiles', '').strip()
                    if not smiles:
                        continue

                    # Canonicalize SMILES
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        logger.warning(f"Invalid SMILES in catalog: {smiles}")
                        continue

                    canonical_smiles = Chem.MolToSmiles(mol)

                    # Create supplier info
                    supplier = SupplierInfo(
                        name=row.get('supplier', 'Unknown'),
                        catalog_id=row.get('catalog_id', ''),
                        price_usd=float(row['price']) if row.get('price') else None,
                        lead_time_days=int(row['lead_time']) if row.get('lead_time') else None,
                        purity=row.get('purity'),
                        quantity_mg=float(row['quantity']) if row.get('quantity') else None
                    )

                    # Add to catalog
                    if canonical_smiles not in self.local_catalog:
                        self.local_catalog[canonical_smiles] = []
                        # Generate fingerprint for similarity search
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        self.local_fingerprints[canonical_smiles] = fp

                    self.local_catalog[canonical_smiles].append(supplier)

        except Exception as e:
            logger.error(f"Failed to load local catalog: {e}")

    def _check_impl(self, smiles: str) -> AvailabilityResult:
        """
        Implementation of availability check (cached)

        Args:
            smiles: SMILES string to check

        Returns:
            AvailabilityResult with availability information
        """
        # Canonicalize input SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return AvailabilityResult(
                smiles=smiles,
                is_available=False,
                exact_match=False,
                source="invalid_smiles"
            )

        canonical_smiles = Chem.MolToSmiles(mol)

        # 1. Check local catalog (exact match)
        if canonical_smiles in self.local_catalog:
            return AvailabilityResult(
                smiles=canonical_smiles,
                is_available=True,
                exact_match=True,
                suppliers=self.local_catalog[canonical_smiles],
                source="local_catalog"
            )

        # 2. Check ZINC database
        if self.use_zinc:
            zinc_result = self._check_zinc(canonical_smiles)
            if zinc_result and zinc_result.is_available:
                return zinc_result

        # 3. Check Enamine REAL
        if self.use_enamine:
            enamine_result = self._check_enamine(canonical_smiles)
            if enamine_result and enamine_result.is_available:
                return enamine_result

        # 4. Find similar compounds in local catalog
        similar = self._find_similar_in_catalog(mol)

        if similar:
            # Found similar compounds, but not exact match
            return AvailabilityResult(
                smiles=canonical_smiles,
                is_available=False,
                exact_match=False,
                similar_compounds=similar,
                source="local_catalog_similar"
            )

        # Not found anywhere
        return AvailabilityResult(
            smiles=canonical_smiles,
            is_available=False,
            exact_match=False,
            source="not_found"
        )

    def _check_zinc(self, smiles: str) -> Optional[AvailabilityResult]:
        """
        Query ZINC15 database via API

        ZINC15 API endpoint: https://zinc15.docking.org/substances/search/
        """
        try:
            # ZINC15 substance search
            base_url = "https://zinc15.docking.org/substances/search/"
            params = {
                'q': smiles,
                'representation': 'smiles',
                'output_format': 'json'
            }

            url = f"{base_url}?{urllib.parse.urlencode(params)}"

            # Add timeout and user agent
            req = urllib.request.Request(url, headers={'User-Agent': 'ML-Fragment-Optimizer/0.1'})

            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

            if data and len(data) > 0:
                # Found in ZINC
                suppliers = []
                for item in data[:3]:  # Top 3 hits
                    suppliers.append(SupplierInfo(
                        name="ZINC15",
                        catalog_id=item.get('zinc_id', ''),
                        price_usd=None,  # ZINC doesn't provide prices
                        lead_time_days=None
                    ))

                return AvailabilityResult(
                    smiles=smiles,
                    is_available=True,
                    exact_match=True,
                    suppliers=suppliers,
                    source="zinc15"
                )

        except urllib.error.URLError as e:
            logger.warning(f"ZINC API request failed: {e}")
        except Exception as e:
            logger.error(f"Error querying ZINC: {e}")

        return None

    def _check_enamine(self, smiles: str) -> Optional[AvailabilityResult]:
        """
        Query Enamine REAL database

        Note: Enamine REAL is a very large database (>30B compounds)
        Typically requires local installation or commercial API access
        This is a placeholder implementation
        """
        # Placeholder - requires Enamine REAL setup
        logger.debug("Enamine REAL checking not implemented (requires API key)")
        return None

    def _find_similar_in_catalog(
        self,
        query_mol: 'Chem.Mol',
        max_results: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find similar compounds in local catalog using fingerprint similarity

        Args:
            query_mol: RDKit molecule object
            max_results: Maximum number of similar compounds to return

        Returns:
            List of (SMILES, similarity) tuples sorted by similarity
        """
        if not self.local_fingerprints:
            return []

        # Generate fingerprint for query
        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)

        # Calculate similarities
        similarities = []
        for smiles, fp in self.local_fingerprints.items():
            similarity = DataStructs.TanimotoSimilarity(query_fp, fp)
            if similarity >= self.similarity_threshold:
                similarities.append((smiles, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:max_results]

    def check_batch(self, smiles_list: List[str]) -> List[AvailabilityResult]:
        """
        Check availability for multiple molecules

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of AvailabilityResult objects
        """
        results = []
        for smiles in smiles_list:
            try:
                result = self.check(smiles)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to check availability for {smiles}: {e}")
                results.append(AvailabilityResult(
                    smiles=smiles,
                    is_available=False,
                    exact_match=False,
                    source="error"
                ))

        return results

    def get_statistics(self) -> Dict:
        """Get statistics about the local catalog"""
        total_compounds = len(self.local_catalog)
        total_suppliers = sum(len(suppliers) for suppliers in self.local_catalog.values())

        supplier_counts = {}
        for suppliers in self.local_catalog.values():
            for supplier in suppliers:
                supplier_counts[supplier.name] = supplier_counts.get(supplier.name, 0) + 1

        return {
            'total_compounds': total_compounds,
            'total_supplier_entries': total_suppliers,
            'unique_suppliers': len(supplier_counts),
            'compounds_per_supplier': supplier_counts
        }


def create_example_catalog(output_path: Path) -> None:
    """
    Create an example building block catalog CSV

    Args:
        output_path: Path to save example catalog
    """
    # Example catalog with common building blocks
    example_data = [
        # SMILES, supplier, catalog_id, price, lead_time, purity, quantity
        ('CCO', 'Sigma-Aldrich', 'E7023', 25.50, 2, '>=99.5%', 1000),
        ('CC(C)O', 'Sigma-Aldrich', 'I9516', 32.00, 2, '>=99.5%', 1000),
        ('c1ccccc1', 'Sigma-Aldrich', 'B1334', 45.00, 2, '>=99%', 1000),
        ('CC(=O)O', 'Sigma-Aldrich', 'A6283', 28.50, 2, '>=99.7%', 1000),
        ('C1CCCCC1', 'TCI', 'C0301', 35.00, 5, '>98.0%', 500),
        ('Cc1ccccc1', 'TCI', 'T0283', 42.00, 5, '>99.0%', 500),
        ('CC(C)(C)O', 'Alfa Aesar', 'A10917', 55.00, 7, '99%', 500),
        ('c1ccc2ccccc2c1', 'Alfa Aesar', 'A10523', 68.00, 7, '99%', 250),
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['smiles', 'supplier', 'catalog_id', 'price', 'lead_time', 'purity', 'quantity'])
        writer.writerows(example_data)

    logger.info(f"Created example catalog at {output_path}")


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    if not RDKIT_AVAILABLE:
        print("RDKit not available. Please install: pip install rdkit")
        exit(1)

    # Create example catalog
    example_catalog_path = Path('example_building_blocks.csv')
    create_example_catalog(example_catalog_path)

    # Initialize checker
    checker = BuildingBlockChecker(
        local_catalog_path=example_catalog_path,
        use_zinc=False,  # Disable for example (to avoid API calls)
        similarity_threshold=0.80
    )

    # Print catalog statistics
    stats = checker.get_statistics()
    print("\nBuilding Block Catalog Statistics:")
    print("=" * 80)
    print(f"Total compounds: {stats['total_compounds']}")
    print(f"Total supplier entries: {stats['total_supplier_entries']}")
    print(f"Unique suppliers: {stats['unique_suppliers']}")
    print("\nCompounds per supplier:")
    for supplier, count in stats['compounds_per_supplier'].items():
        print(f"  {supplier}: {count}")

    # Test molecules
    test_molecules = [
        ('CCO', 'Ethanol - should be available'),
        ('CC(C)O', 'Isopropanol - should be available'),
        ('CCCO', 'Propanol - similar to ethanol'),
        ('CC(C)(C)C(C)(C)C', 'Complex hydrocarbon - not available'),
    ]

    print("\n\nAvailability Check Results:")
    print("=" * 80)

    for smiles, description in test_molecules:
        result = checker.check(smiles)
        print(f"\n{description}")
        print(f"SMILES: {smiles}")
        print(f"Available: {result.is_available}")
        print(f"Exact match: {result.exact_match}")
        print(f"Source: {result.source}")

        if result.suppliers:
            print(f"Suppliers ({len(result.suppliers)}):")
            for supplier in result.suppliers:
                price = f"${supplier.price_usd:.2f}" if supplier.price_usd else "N/A"
                lead = f"{supplier.lead_time_days}d" if supplier.lead_time_days else "N/A"
                print(f"  - {supplier.name} ({supplier.catalog_id}): {price}, Lead: {lead}")

        if result.similar_compounds:
            print(f"Similar compounds ({len(result.similar_compounds)}):")
            for sim_smiles, similarity in result.similar_compounds[:3]:
                print(f"  - {sim_smiles} (similarity: {similarity:.2f})")

    # Clean up example file
    # example_catalog_path.unlink()
    print(f"\n\nExample catalog saved to: {example_catalog_path}")
