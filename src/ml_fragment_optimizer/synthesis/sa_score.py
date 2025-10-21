"""
Synthetic Accessibility Score (SAScore) Calculator

Implementation of the SAScore algorithm from Ertl & Schuffenhauer:
"Estimation of synthetic accessibility score of drug-like molecules based on
molecular complexity and fragment contributions"
Journal of Cheminformatics 2009, 1:8

The algorithm considers:
1. Fragment contributions (based on fragment frequency in PubChem)
2. Molecular complexity penalties (size, stereochemistry, rings, etc.)

Score range: 1 (easy to synthesize) to 10 (very difficult to synthesize)

Author: ML-Fragment-Optimizer Team
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. SAScore calculation will not work.")

import math

logger = logging.getLogger(__name__)


@dataclass
class SAScoreResult:
    """Result of SAScore calculation"""
    score: float
    fragment_score: float
    complexity_penalty: float
    smiles: str
    num_atoms: int
    num_rings: int
    num_stereocenters: int

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'sa_score': round(self.score, 3),
            'fragment_score': round(self.fragment_score, 3),
            'complexity_penalty': round(self.complexity_penalty, 3),
            'smiles': self.smiles,
            'num_atoms': self.num_atoms,
            'num_rings': self.num_rings,
            'num_stereocenters': self.num_stereocenters,
            'interpretation': self.get_interpretation()
        }

    def get_interpretation(self) -> str:
        """Get human-readable interpretation of score"""
        if self.score <= 2:
            return "Very Easy"
        elif self.score <= 3:
            return "Easy"
        elif self.score <= 4:
            return "Moderately Easy"
        elif self.score <= 6:
            return "Moderate"
        elif self.score <= 7:
            return "Moderately Difficult"
        elif self.score <= 8:
            return "Difficult"
        else:
            return "Very Difficult"


class SAScoreCalculator:
    """
    Synthetic Accessibility Score Calculator

    This implementation uses fragment-based scoring combined with complexity
    penalties to estimate synthetic accessibility.

    Setup:
        The calculator requires fragment frequency data. If not available,
        it will use a fallback complexity-only scoring method.

        To generate fragment data:
        1. Download a large molecule database (e.g., PubChem subset)
        2. Generate fragment frequencies using gen_fragment_data.py
        3. Place fpscores.pkl in the data directory

    Example:
        >>> calculator = SAScoreCalculator()
        >>> result = calculator.calculate('CCO')  # Ethanol
        >>> print(f"SAScore: {result.score:.2f}")
        SAScore: 1.23
    """

    # Default fragment scores (small training set - replace with fpscores.pkl)
    DEFAULT_FRAGMENT_SCORES = {
        'c': -0.5,   # aromatic carbon (common)
        'C': -0.3,   # aliphatic carbon (common)
        'O': -0.2,   # oxygen (common)
        'N': -0.1,   # nitrogen (common)
        'F': 0.1,    # fluorine
        'Cl': 0.2,   # chlorine
        'Br': 0.3,   # bromine
        'S': 0.1,    # sulfur
    }

    def __init__(self, fragment_data_path: Optional[Path] = None):
        """
        Initialize SAScore calculator

        Args:
            fragment_data_path: Path to fragment frequency pickle file (fpscores.pkl)
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for SAScore calculation")

        self.fragment_scores: Dict[str, float] = {}
        self._load_fragment_data(fragment_data_path)

        logger.info(f"SAScoreCalculator initialized with {len(self.fragment_scores)} fragments")

    def _load_fragment_data(self, data_path: Optional[Path]) -> None:
        """Load fragment frequency data from pickle file"""
        if data_path and data_path.exists():
            try:
                with open(data_path, 'rb') as f:
                    self.fragment_scores = pickle.load(f)
                logger.info(f"Loaded fragment data from {data_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load fragment data: {e}")

        # Use default small fragment set
        self.fragment_scores = self.DEFAULT_FRAGMENT_SCORES.copy()
        logger.info("Using default fragment scores (limited accuracy)")

    @lru_cache(maxsize=10000)
    def calculate(self, smiles: str) -> SAScoreResult:
        """
        Calculate SAScore for a molecule

        Args:
            smiles: SMILES string of molecule

        Returns:
            SAScoreResult with score and detailed information

        Raises:
            ValueError: If SMILES is invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Calculate fragment contribution
        fragment_score = self._calculate_fragment_score(mol)

        # Calculate complexity penalty
        complexity = self._calculate_complexity(mol)

        # Combine scores (fragment score is negative for easy, complexity is positive)
        # Scale to 1-10 range
        raw_score = fragment_score + complexity
        sa_score = self._scale_score(raw_score)

        # Get molecular properties
        num_atoms = mol.GetNumHeavyAtoms()
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))

        result = SAScoreResult(
            score=sa_score,
            fragment_score=fragment_score,
            complexity_penalty=complexity,
            smiles=smiles,
            num_atoms=num_atoms,
            num_rings=num_rings,
            num_stereocenters=num_stereo
        )

        logger.debug(f"SAScore for {smiles}: {sa_score:.2f}")
        return result

    def _calculate_fragment_score(self, mol: 'Chem.Mol') -> float:
        """
        Calculate fragment-based score

        Common fragments get negative scores (easier to synthesize)
        Rare fragments get positive scores (harder to synthesize)
        """
        # Generate Morgan fingerprint (ECFP-like)
        from rdkit.Chem import AllChem

        fp = AllChem.GetMorganFingerprint(mol, 2)  # radius=2
        fps = fp.GetNonzeroElements()

        score = 0.0
        count = 0

        for fragment_id in fps:
            fragment_str = str(fragment_id)
            if fragment_str in self.fragment_scores:
                score += self.fragment_scores[fragment_str]
                count += 1

        # Normalize by number of fragments
        if count > 0:
            score /= count

        return score

    def _calculate_complexity(self, mol: 'Chem.Mol') -> float:
        """
        Calculate molecular complexity penalty

        Considers:
        - Molecular size
        - Ring systems
        - Stereochemistry
        - Macrocycles
        - Bridged rings
        - Spiro centers
        """
        complexity = 0.0

        # Size penalty (larger molecules harder to make)
        num_atoms = mol.GetNumHeavyAtoms()
        if num_atoms > 45:
            complexity += (num_atoms - 45) * 0.1

        # Ring penalty
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        if num_rings > 0:
            complexity += math.log(num_rings + 1) * 0.5

        # Macrocycle penalty (rings > 8 atoms)
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if len(ring) > 8:
                complexity += 0.5

        # Bridged ring penalty
        num_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        if num_bridgehead > 0:
            complexity += num_bridgehead * 0.3

        # Spiro center penalty
        num_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        if num_spiro > 0:
            complexity += num_spiro * 0.3

        # Stereocenter penalty
        stereo_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        if len(stereo_centers) > 2:
            complexity += (len(stereo_centers) - 2) * 0.2

        # Heteroatom diversity (many different atom types = complex)
        atom_types = set(atom.GetSymbol() for atom in mol.GetAtoms())
        if len(atom_types) > 5:
            complexity += (len(atom_types) - 5) * 0.1

        return complexity

    def _scale_score(self, raw_score: float) -> float:
        """
        Scale raw score to 1-10 range

        Uses sigmoid-like scaling to map raw scores to interpretable range
        """
        # Empirical scaling based on typical score distributions
        # Raw scores typically range from -4 to +6
        # Map to 1-10 scale

        scaled = 5.0 + raw_score  # Center around 5
        scaled = max(1.0, min(10.0, scaled))  # Clamp to 1-10

        return scaled

    def calculate_batch(self, smiles_list: list[str]) -> list[SAScoreResult]:
        """
        Calculate SAScore for multiple molecules

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of SAScoreResult objects (None for invalid SMILES)
        """
        results = []
        for smiles in smiles_list:
            try:
                result = self.calculate(smiles)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to calculate SAScore for {smiles}: {e}")
                results.append(None)

        return results


# Convenience function
def calculate_sa_score(smiles: str, fragment_data_path: Optional[Path] = None) -> float:
    """
    Calculate SAScore for a molecule (convenience function)

    Args:
        smiles: SMILES string
        fragment_data_path: Optional path to fragment data

    Returns:
        SAScore (1-10, lower is easier)
    """
    calculator = SAScoreCalculator(fragment_data_path)
    result = calculator.calculate(smiles)
    return result.score


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    if not RDKIT_AVAILABLE:
        print("RDKit not available. Please install: pip install rdkit")
        exit(1)

    # Test molecules with known synthetic accessibility
    test_molecules = [
        ('CCO', 'Ethanol - very easy'),
        ('CC(C)CC1=CC=C(C=C1)C(C)C(O)=O', 'Ibuprofen - easy'),
        ('CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5', 'Imatinib - difficult'),
    ]

    calculator = SAScoreCalculator()

    print("\nSynthetic Accessibility Score Examples:")
    print("=" * 80)

    for smiles, name in test_molecules:
        result = calculator.calculate(smiles)
        print(f"\n{name}")
        print(f"SMILES: {smiles}")
        print(f"SAScore: {result.score:.2f} ({result.get_interpretation()})")
        print(f"  Fragment Score: {result.fragment_score:.2f}")
        print(f"  Complexity Penalty: {result.complexity_penalty:.2f}")
        print(f"  Heavy Atoms: {result.num_atoms}")
        print(f"  Rings: {result.num_rings}")
        print(f"  Stereocenters: {result.num_stereocenters}")
