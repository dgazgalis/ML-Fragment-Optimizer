"""
Molecular featurization utilities for ML models.

Provides various molecular fingerprints and descriptors:
- Morgan/ECFP fingerprints
- MACCS keys
- RDKit descriptors
- Physicochemical properties
"""

from typing import List, Optional, Union, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
import logging

logger = logging.getLogger(__name__)


class MolecularFeaturizer:
    """
    Unified interface for molecular featurization.

    Supports multiple fingerprint types and descriptor sets.
    """

    AVAILABLE_FINGERPRINTS = ["morgan", "maccs", "rdkit", "avalon", "atompair"]

    def __init__(
        self,
        fingerprint_type: str = "morgan",
        radius: int = 2,
        n_bits: int = 2048,
        use_features: bool = False,
        use_chirality: bool = True,
        include_descriptors: bool = False,
        descriptor_names: Optional[List[str]] = None,
    ):
        """
        Initialize molecular featurizer.

        Args:
            fingerprint_type: Type of fingerprint ("morgan", "maccs", "rdkit", etc.)
            radius: Radius for Morgan fingerprints
            n_bits: Number of bits for fingerprints
            use_features: Use feature-based Morgan fingerprints
            use_chirality: Include chirality information
            include_descriptors: Add RDKit molecular descriptors
            descriptor_names: Specific descriptors to compute (None = all)
        """
        if fingerprint_type not in self.AVAILABLE_FINGERPRINTS:
            raise ValueError(
                f"Unknown fingerprint type: {fingerprint_type}. "
                f"Available: {self.AVAILABLE_FINGERPRINTS}"
            )

        self.fingerprint_type = fingerprint_type
        self.radius = radius
        self.n_bits = n_bits
        self.use_features = use_features
        self.use_chirality = use_chirality
        self.include_descriptors = include_descriptors

        # Set up descriptor calculator
        if include_descriptors:
            if descriptor_names is None:
                # Use all available descriptors
                descriptor_names = [desc[0] for desc in Descriptors._descList]
            self.descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
                descriptor_names
            )
            self.descriptor_names = descriptor_names
        else:
            self.descriptor_calculator = None
            self.descriptor_names = []

    def featurize(
        self,
        smiles: Union[str, List[str]],
        return_array: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Featurize molecule(s) from SMILES string(s).

        Args:
            smiles: SMILES string or list of SMILES strings
            return_array: Return numpy array (True) or list (False)

        Returns:
            Feature array(s) for the molecule(s)
        """
        is_single = isinstance(smiles, str)
        if is_single:
            smiles = [smiles]

        features_list = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning(f"Could not parse SMILES: {smi}")
                features_list.append(None)
                continue

            features = self._featurize_mol(mol)
            features_list.append(features)

        # Filter out None values
        valid_features = [f for f in features_list if f is not None]

        if not valid_features:
            raise ValueError("No valid molecules could be featurized")

        if return_array:
            result = np.array(valid_features)
            return result[0] if is_single else result
        else:
            return features_list[0] if is_single else features_list

    def _featurize_mol(self, mol: Chem.Mol) -> np.ndarray:
        """Featurize a single RDKit molecule."""
        # Generate fingerprint
        fp = self._get_fingerprint(mol)

        # Add descriptors if requested
        if self.include_descriptors:
            descriptors = self._get_descriptors(mol)
            features = np.concatenate([fp, descriptors])
        else:
            features = fp

        return features

    def _get_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """Generate molecular fingerprint."""
        if self.fingerprint_type == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                self.radius,
                nBits=self.n_bits,
                useFeatures=self.use_features,
                useChirality=self.use_chirality,
            )
        elif self.fingerprint_type == "maccs":
            fp = MACCSkeys.GenMACCSKeys(mol)
        elif self.fingerprint_type == "rdkit":
            fp = Chem.RDKFingerprint(mol, fpSize=self.n_bits)
        elif self.fingerprint_type == "avalon":
            from rdkit.Avalon import pyAvalonTools
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=self.n_bits)
        elif self.fingerprint_type == "atompair":
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                mol, nBits=self.n_bits
            )
        else:
            raise ValueError(f"Unsupported fingerprint type: {self.fingerprint_type}")

        return np.array(fp)

    def _get_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        """Calculate molecular descriptors."""
        if self.descriptor_calculator is None:
            return np.array([])

        try:
            descriptors = self.descriptor_calculator.CalcDescriptors(mol)
            # Handle NaN values
            descriptors = np.array(descriptors)
            descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
            return descriptors
        except Exception as e:
            logger.warning(f"Error calculating descriptors: {e}")
            return np.zeros(len(self.descriptor_names))

    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        feature_names = []

        # Fingerprint bits
        fp_prefix = self.fingerprint_type.upper()
        feature_names.extend([f"{fp_prefix}_bit_{i}" for i in range(self.n_bits)])

        # Descriptors
        if self.include_descriptors:
            feature_names.extend(self.descriptor_names)

        return feature_names

    @property
    def n_features(self) -> int:
        """Total number of features."""
        n_fp = self.n_bits
        n_desc = len(self.descriptor_names) if self.include_descriptors else 0
        return n_fp + n_desc


def calculate_basic_properties(smiles: str) -> dict:
    """
    Calculate basic physicochemical properties.

    Args:
        smiles: SMILES string

    Returns:
        Dictionary of properties
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "n_hba": Descriptors.NumHAcceptors(mol),
        "n_hbd": Descriptors.NumHDonors(mol),
        "n_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "n_aromatic_rings": Descriptors.NumAromaticRings(mol),
        "n_heavy_atoms": Descriptors.HeavyAtomCount(mol),
        "fraction_csp3": Descriptors.FractionCSP3(mol),
        "molar_refractivity": Descriptors.MolMR(mol),
    }


def smiles_to_mol_safe(smiles: str) -> Optional[Chem.Mol]:
    """
    Safely convert SMILES to RDKit mol object.

    Args:
        smiles: SMILES string

    Returns:
        RDKit Mol object or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Sanitize molecule
            Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        logger.warning(f"Error parsing SMILES '{smiles}': {e}")
        return None


def batch_featurize(
    smiles_list: List[str],
    featurizer: MolecularFeaturizer,
    chunk_size: int = 1000,
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Featurize large list of SMILES with progress tracking.

    Args:
        smiles_list: List of SMILES strings
        featurizer: Configured MolecularFeaturizer
        chunk_size: Process in chunks of this size
        show_progress: Show progress bar

    Returns:
        Tuple of (feature_matrix, valid_smiles)
    """
    from tqdm import tqdm

    features_list = []
    valid_smiles = []

    iterator = range(0, len(smiles_list), chunk_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Featurizing molecules")

    for i in iterator:
        chunk = smiles_list[i:i + chunk_size]

        for smi in chunk:
            try:
                feat = featurizer.featurize(smi, return_array=True)
                features_list.append(feat)
                valid_smiles.append(smi)
            except Exception as e:
                logger.debug(f"Skipping invalid SMILES '{smi}': {e}")
                continue

    if not features_list:
        raise ValueError("No valid molecules could be featurized")

    feature_matrix = np.vstack(features_list)
    return feature_matrix, valid_smiles
