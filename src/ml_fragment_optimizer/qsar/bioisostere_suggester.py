"""
Bioisosteric Replacement Suggester

Suggests bioisosteric replacements based on SAR analysis, medicinal chemistry
knowledge, and predicted property improvements.

References:
    - Langmuir (1919) "Isomorphism, Isosterism and Covalence" J. Am. Chem. Soc. 41, 1543
    - Burger (1991) "Isosterism and Bioisosterism in Drug Design" Prog. Drug Res. 37, 287
    - Meanwell (2011) "Synopsis of Some Recent Tactical Application of Bioisosteres
      in Drug Design" J. Med. Chem. 54, 2529-2591
    - Wermuth (2008) "The Practice of Medicinal Chemistry" 3rd Ed., Ch. 14
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski

logger = logging.getLogger(__name__)


@dataclass
class BioisostericReplacement:
    """Bioisosteric replacement suggestion."""

    original_smarts: str
    replacement_smarts: str
    replacement_name: str
    category: str  # 'classic', 'ring', 'nonclassical'
    rationale: str
    predicted_activity_change: Optional[float] = None
    predicted_lipophilicity_change: Optional[float] = None
    predicted_solubility_change: Optional[float] = None
    drug_likeness_score: Optional[float] = None
    synthetic_accessibility: Optional[float] = None
    literature_precedent: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"BioisostericReplacement({self.original_smarts} → "
            f"{self.replacement_smarts}: {self.replacement_name})"
        )


@dataclass
class BioisostereLibrary:
    """Library of known bioisosteric replacements."""

    # Classic bioisosteres
    CLASSIC = {
        # Halogens
        ("[Cl]", "[Br]", "Cl→Br", "Often increases lipophilicity and activity"),
        ("[Br]", "[Cl]", "Br→Cl", "Often decreases lipophilicity, may improve metabolic stability"),
        ("[F]", "[Cl]", "F→Cl", "Increases size, lipophilicity"),
        ("[Cl]", "[CF3]", "Cl→CF3", "Increases lipophilicity, metabolic stability"),

        # Hydrogen bond donors/acceptors
        ("[OH]", "[NH2]", "OH→NH2", "Maintains H-bond capability, changes pKa"),
        ("[NH2]", "[OH]", "NH2→OH", "Maintains H-bond capability, changes pKa"),
        ("[C](=O)[OH]", "[C](=O)[NH2]", "COOH→CONH2", "Removes charge, may improve permeability"),
        ("[C](=O)[NH2]", "[C](=O)[OH]", "CONH2→COOH", "Adds charge, may improve solubility"),

        # Carbonyls
        ("[C](=O)", "[C](=S)", "C=O→C=S", "Thiocarbonyl bioisostere, softer electrophile"),
        ("[C](=O)", "[S](=O)(=O)", "C=O→SO2", "Stronger H-bond acceptor, increased polarity"),

        # Methylenes
        ("[CH2]", "[O]", "CH2→O", "Decreases lipophilicity, adds polarity"),
        ("[CH2]", "[S]", "CH2→S", "Similar size, different electronics"),
        ("[CH2]", "[NH]", "CH2→NH", "Adds H-bond capability"),

        # Thiophene/furan
        ("c1ccsc1", "c1ccoc1", "thiophene→furan", "Decreases size slightly, increases polarity"),
        ("c1ccoc1", "c1ccsc1", "furan→thiophene", "Increases size slightly, metabolically more stable"),
    }

    # Ring bioisosteres
    RING = {
        # Benzene replacements
        ("c1ccccc1", "c1ccncc1", "benzene→pyridine", "Adds polarity, H-bond acceptor, may improve solubility"),
        ("c1ccccc1", "c1cnccc1", "benzene→pyridine(meta)", "Adds polarity, different electronics"),
        ("c1ccccc1", "c1ncccc1", "benzene→pyridine(ortho)", "Adds polarity, ortho effects"),
        ("c1ccccc1", "c1cccs1", "benzene→thiophene", "Reduces aromatic rings, maintains planarity"),
        ("c1ccccc1", "c1cnc(N)nc1", "benzene→pyrimidine", "Increases polarity significantly"),

        # Pyridine variations
        ("c1ccncc1", "c1cncnc1", "pyridine→pyrimidine", "Increases polarity and H-bonding"),
        ("c1ccncc1", "c1cnccc1", "pyridine→pyrimidine(iso)", "Different N positioning"),

        # Five-membered rings
        ("c1ccccc1", "c1ccc[nH]1", "benzene→pyrrole", "Adds NH, H-bond donor"),
        ("c1ccc[nH]1", "c1ccco1", "pyrrole→furan", "Removes H-bond donor, adds O"),
        ("c1ccco1", "c1cccs1", "furan→thiophene", "Better metabolic stability"),
        ("c1cccs1", "c1ccc[nH]1", "thiophene→pyrrole", "Adds H-bond donor"),

        # Saturated ring modifications
        ("C1CCCCC1", "C1CCNCC1", "cyclohexane→piperidine", "Adds basicity, H-bond acceptor"),
        ("C1CCCCC1", "C1CCOCC1", "cyclohexane→tetrahydropyran", "Adds polarity, H-bond acceptor"),
    }

    # Non-classical bioisosteres
    NONCLASSICAL = {
        # Carboxylic acid bioisosteres
        ("[C](=O)[OH]", "c1n[nH]nn1", "COOH→tetrazole", "pKa similar, metabolically stable"),
        ("[C](=O)[OH]", "[S](=O)(=O)[NH2]", "COOH→sulfonamide", "Acidic, different H-bonding"),
        ("[C](=O)[OH]", "[P](=O)([OH])[OH]", "COOH→phosphonic acid", "Stronger acid, different geometry"),

        # Amide bioisosteres
        ("[C](=O)[NH]", "c1cn[nH]c1", "amide→1,2,4-triazole", "Metabolically stable, retains H-bonding"),
        ("[C](=O)[NH]", "[S](=O)(=O)[NH]", "amide→sulfonamide", "More acidic NH, different geometry"),
        ("[C](=O)[NH]", "[CH2][NH]", "amide→amine", "Removes planarity, adds flexibility"),

        # Ester bioisosteres
        ("[C](=O)[O]", "[S](=O)(=O)[O]", "ester→sulfonate", "More polar, metabolically stable"),
        ("[C](=O)[O]", "[C](=O)[NH]", "ester→amide", "Removes hydrolysis liability"),
    }


class BioisostereSuggester:
    """
    Suggests bioisosteric replacements based on SAR data and medicinal chemistry knowledge.

    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("c1ccc(Cl)cc1")  # chlorobenzene
        >>>
        >>> suggester = BioisostereSuggester()
        >>> suggestions = suggester.suggest_replacements(mol)
        >>> for sug in suggestions[:5]:
        ...     print(f"{sug.replacement_name}: {sug.rationale}")
        ...     print(f"  Drug-likeness: {sug.drug_likeness_score:.2f}")
    """

    def __init__(
        self,
        model: Optional[any] = None,
        sar_data: Optional[Dict] = None,
    ):
        """
        Initialize bioisostere suggester.

        Args:
            model: Optional trained model for activity prediction
            sar_data: Optional SAR data for context-aware suggestions
        """
        self.model = model
        self.sar_data = sar_data or {}
        self.library = BioisostereLibrary()

    def suggest_replacements(
        self,
        mol: Chem.Mol,
        max_suggestions: int = 10,
        include_classic: bool = True,
        include_ring: bool = True,
        include_nonclassical: bool = True,
        filter_drug_like: bool = True,
    ) -> List[BioisostericReplacement]:
        """
        Suggest bioisosteric replacements for molecule.

        Args:
            mol: RDKit molecule
            max_suggestions: Maximum number of suggestions
            include_classic: Include classic bioisosteres
            include_ring: Include ring bioisosteres
            include_nonclassical: Include non-classical bioisosteres
            filter_drug_like: Filter to drug-like molecules

        Returns:
            List of BioisostericReplacement objects, sorted by predicted improvement
        """
        suggestions = []

        # Try classic bioisosteres
        if include_classic:
            for orig, repl, name, rationale in self.library.CLASSIC:
                sugs = self._apply_replacement(
                    mol, orig, repl, name, "classic", rationale
                )
                suggestions.extend(sugs)

        # Try ring bioisosteres
        if include_ring:
            for orig, repl, name, rationale in self.library.RING:
                sugs = self._apply_replacement(
                    mol, orig, repl, name, "ring", rationale
                )
                suggestions.extend(sugs)

        # Try non-classical bioisosteres
        if include_nonclassical:
            for orig, repl, name, rationale in self.library.NONCLASSICAL:
                sugs = self._apply_replacement(
                    mol, orig, repl, name, "nonclassical", rationale
                )
                suggestions.extend(sugs)

        # Score and filter
        scored_suggestions = []
        for sug in suggestions:
            # Calculate drug-likeness
            try:
                new_mol = self._apply_smarts_replacement(
                    mol, sug.original_smarts, sug.replacement_smarts
                )
                if new_mol is None:
                    continue

                # Drug-likeness filters
                if filter_drug_like and not self._is_drug_like(new_mol):
                    continue

                # Calculate properties
                sug.drug_likeness_score = self._calculate_drug_likeness(new_mol)
                sug.synthetic_accessibility = self._estimate_sa_score(new_mol)

                # Predict property changes if model available
                if self.model is not None:
                    sug.predicted_activity_change = self._predict_activity_change(
                        mol, new_mol
                    )

                scored_suggestions.append(sug)

            except Exception as e:
                logger.debug(f"Failed to score suggestion: {e}")
                continue

        # Sort by composite score
        scored_suggestions.sort(
            key=lambda x: self._composite_score(x),
            reverse=True
        )

        return scored_suggestions[:max_suggestions]

    def _apply_replacement(
        self,
        mol: Chem.Mol,
        original_smarts: str,
        replacement_smarts: str,
        name: str,
        category: str,
        rationale: str,
    ) -> List[BioisostericReplacement]:
        """Apply a bioisosteric replacement pattern."""
        suggestions = []

        # Check if pattern exists in molecule
        pattern = Chem.MolFromSmarts(original_smarts)
        if pattern is None:
            return suggestions

        matches = mol.GetSubstructMatches(pattern)
        if not matches:
            return suggestions

        # Create suggestion for each match
        for match in matches:
            sug = BioisostericReplacement(
                original_smarts=original_smarts,
                replacement_smarts=replacement_smarts,
                replacement_name=name,
                category=category,
                rationale=rationale,
            )
            suggestions.append(sug)

        return suggestions

    def _apply_smarts_replacement(
        self,
        mol: Chem.Mol,
        original_smarts: str,
        replacement_smarts: str,
    ) -> Optional[Chem.Mol]:
        """Apply SMARTS-based replacement to molecule."""
        # This is simplified - real implementation would use RDKit's
        # ReplaceSubstructs or similar functionality

        try:
            # Use AllChem.ReplaceSubstructs
            pattern = Chem.MolFromSmarts(original_smarts)
            replacement = Chem.MolFromSmarts(replacement_smarts)

            if pattern is None or replacement is None:
                return None

            # Get first match
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None

            # Replace (simplified)
            new_mol = AllChem.ReplaceSubstructs(
                mol, pattern, replacement, replaceAll=False
            )

            if new_mol and len(new_mol) > 0:
                return new_mol[0]

            return None

        except Exception as e:
            logger.debug(f"Replacement failed: {e}")
            return None

    def _is_drug_like(self, mol: Chem.Mol) -> bool:
        """Check if molecule passes drug-likeness filters (Lipinski's rule)."""
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        # Lipinski's Rule of 5
        if mw > 500:
            return False
        if logp > 5:
            return False
        if hbd > 5:
            return False
        if hba > 10:
            return False

        return True

    def _calculate_drug_likeness(self, mol: Chem.Mol) -> float:
        """Calculate drug-likeness score (0-1)."""
        # QED (Quantitative Estimate of Drug-likeness)
        try:
            from rdkit.Chem import QED
            return QED.qed(mol)
        except ImportError:
            # Fallback: simple scoring based on Lipinski
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)

            # Normalize to 0-1
            mw_score = 1.0 - min(abs(mw - 350) / 350, 1.0)
            logp_score = 1.0 - min(abs(logp - 2.5) / 2.5, 1.0)
            hbd_score = 1.0 - min(hbd / 5.0, 1.0)
            hba_score = 1.0 - min(hba / 10.0, 1.0)

            return (mw_score + logp_score + hbd_score + hba_score) / 4.0

    def _estimate_sa_score(self, mol: Chem.Mol) -> float:
        """Estimate synthetic accessibility (1=easy, 10=difficult)."""
        # Simplified SA score based on complexity
        # Real implementation would use SAScore algorithm

        # Count rings, branches, complexity
        n_rings = Lipinski.RingCount(mol)
        n_hetero = Lipinski.NumHeteroatoms(mol)
        n_atoms = mol.GetNumHeavyAtoms()
        n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))

        # Simple heuristic (1-10 scale)
        complexity = (n_rings * 0.5 + n_hetero * 0.3 +
                     n_chiral * 1.0 + n_atoms * 0.05)

        sa_score = min(max(1.0 + complexity * 0.3, 1.0), 10.0)

        return sa_score

    def _predict_activity_change(
        self,
        mol_original: Chem.Mol,
        mol_new: Chem.Mol,
    ) -> float:
        """Predict change in activity from bioisosteric replacement."""
        if self.model is None:
            return 0.0

        try:
            # Get features
            fp_orig = self._mol_to_features(mol_original)
            fp_new = self._mol_to_features(mol_new)

            # Predict activities
            act_orig = self.model.predict(fp_orig.reshape(1, -1))[0]
            act_new = self.model.predict(fp_new.reshape(1, -1))[0]

            return act_new - act_orig

        except Exception as e:
            logger.debug(f"Activity prediction failed: {e}")
            return 0.0

    def _mol_to_features(self, mol: Chem.Mol, n_bits: int = 2048) -> np.ndarray:
        """Convert molecule to feature vector."""
        from rdkit import DataStructs

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros((n_bits,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _composite_score(self, suggestion: BioisostericReplacement) -> float:
        """Calculate composite score for ranking suggestions."""
        score = 0.0

        # Drug-likeness (0-1, higher better)
        if suggestion.drug_likeness_score is not None:
            score += suggestion.drug_likeness_score * 2.0

        # Synthetic accessibility (1-10, lower better)
        if suggestion.synthetic_accessibility is not None:
            score += (10.0 - suggestion.synthetic_accessibility) / 10.0

        # Predicted activity improvement
        if suggestion.predicted_activity_change is not None:
            score += suggestion.predicted_activity_change * 0.5

        return score

    def explain_suggestion(
        self,
        suggestion: BioisostericReplacement,
    ) -> str:
        """Generate human-readable explanation for suggestion."""
        explanation = [
            f"Bioisosteric Replacement: {suggestion.replacement_name}",
            f"Category: {suggestion.category}",
            f"",
            f"Rationale: {suggestion.rationale}",
            f"",
        ]

        if suggestion.predicted_activity_change is not None:
            change = suggestion.predicted_activity_change
            direction = "increase" if change > 0 else "decrease"
            explanation.append(
                f"Predicted activity change: {abs(change):.2f} log units ({direction})"
            )

        if suggestion.drug_likeness_score is not None:
            explanation.append(
                f"Drug-likeness score: {suggestion.drug_likeness_score:.2f}/1.00"
            )

        if suggestion.synthetic_accessibility is not None:
            explanation.append(
                f"Synthetic accessibility: {suggestion.synthetic_accessibility:.1f}/10.0 "
                f"({'easy' if suggestion.synthetic_accessibility < 5 else 'moderate' if suggestion.synthetic_accessibility < 7 else 'difficult'})"
            )

        if suggestion.literature_precedent:
            explanation.append(f"")
            explanation.append(f"Literature: {suggestion.literature_precedent}")

        return "\n".join(explanation)


def suggest_bioisosteres(
    mol: Chem.Mol,
    max_suggestions: int = 10,
) -> List[BioisostericReplacement]:
    """
    Convenience function to suggest bioisosteric replacements.

    Args:
        mol: RDKit molecule
        max_suggestions: Maximum number of suggestions

    Returns:
        List of BioisostericReplacement objects

    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("c1ccc(Cl)cc1")
        >>> suggestions = suggest_bioisosteres(mol, max_suggestions=5)
        >>> for sug in suggestions:
        ...     print(f"{sug.replacement_name}: {sug.rationale}")
    """
    suggester = BioisostereSuggester()
    return suggester.suggest_replacements(mol, max_suggestions=max_suggestions)


if __name__ == "__main__":
    # Example usage
    from rdkit import Chem

    print("=== Bioisosteric Replacement Suggester ===\n")

    # Example molecules
    test_molecules = [
        ("c1ccc(Cl)cc1", "chlorobenzene"),
        ("c1ccc(C(=O)O)cc1", "benzoic acid"),
        ("c1ccc(C(=O)N)cc1", "benzamide"),
        ("c1ccncc1", "pyridine"),
    ]

    suggester = BioisostereSuggester()

    for smiles, name in test_molecules:
        mol = Chem.MolFromSmiles(smiles)
        print(f"Molecule: {name} ({smiles})")
        print("=" * 60)

        suggestions = suggester.suggest_replacements(
            mol,
            max_suggestions=5,
            filter_drug_like=False,  # Show all for demonstration
        )

        if not suggestions:
            print("No bioisosteric replacements found.\n")
            continue

        for i, sug in enumerate(suggestions, 1):
            print(f"\n{i}. {sug.replacement_name} ({sug.category})")
            print(f"   Pattern: {sug.original_smarts} → {sug.replacement_smarts}")
            print(f"   {sug.rationale}")

            if sug.drug_likeness_score is not None:
                print(f"   Drug-likeness: {sug.drug_likeness_score:.2f}")

            if sug.synthetic_accessibility is not None:
                print(f"   Synthetic accessibility: {sug.synthetic_accessibility:.1f}/10")

        print("\n" + "=" * 60 + "\n")

    print("Bioisosteric replacement suggestions completed!")
