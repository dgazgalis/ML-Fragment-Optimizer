"""
Molecular Featurization Module

This module provides comprehensive molecular featurization methods including:
- ECFP/Morgan fingerprints for substructure encoding
- MACCS keys for pharmacophore patterns
- RDKit 2D descriptors for physicochemical properties
- Graph representations for message passing neural networks

Author: Claude Code
Date: 2025-10-20
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint

try:
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not available. Graph-based features disabled.")


@dataclass
class MoleculeFeatures:
    """Container for molecular features"""
    smiles: str
    morgan_fp: Optional[np.ndarray] = None
    maccs_keys: Optional[np.ndarray] = None
    rdkit_descriptors: Optional[np.ndarray] = None
    graph_data: Optional['Data'] = None

    def to_dict(self) -> dict:
        """Convert features to dictionary"""
        return {
            'smiles': self.smiles,
            'morgan_fp': self.morgan_fp,
            'maccs_keys': self.maccs_keys,
            'rdkit_descriptors': self.rdkit_descriptors,
            'graph_data': self.graph_data
        }


class MolecularFeaturizer:
    """
    Comprehensive molecular featurization class.

    Supports multiple featurization strategies:
    1. Morgan/ECFP fingerprints: Circular fingerprints encoding substructures
    2. MACCS keys: 166-bit structural key descriptors
    3. RDKit 2D descriptors: Physicochemical properties
    4. Graph representation: For message passing neural networks
    """

    def __init__(
        self,
        morgan_radius: int = 2,
        morgan_nbits: int = 2048,
        use_features: bool = True,
        use_chirality: bool = True
    ):
        """
        Initialize molecular featurizer.

        Args:
            morgan_radius: Radius for Morgan fingerprints (default: 2 for ECFP4)
            morgan_nbits: Number of bits in Morgan fingerprint
            use_features: Whether to use pharmacophore features in fingerprints
            use_chirality: Whether to include chirality information
        """
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits
        self.use_features = use_features
        self.use_chirality = use_chirality

        # Atom type encoding for graph representation
        self.atom_types = [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I',
            'B', 'Se', 'other'
        ]
        self.atom_type_to_idx = {a: i for i, a in enumerate(self.atom_types)}

    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES to RDKit molecule object with error handling.

        Args:
            smiles: SMILES string

        Returns:
            RDKit molecule object or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            # Add explicit hydrogens for accurate descriptor calculation
            mol = Chem.AddHs(mol)
            return mol
        except Exception as e:
            print(f"Error parsing SMILES '{smiles}': {e}")
            return None

    def compute_morgan_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """
        Compute Morgan (ECFP) fingerprint.

        Morgan fingerprints encode molecular substructures using circular
        neighborhood around each atom. Radius 2 corresponds to ECFP4.

        Args:
            mol: RDKit molecule

        Returns:
            Binary fingerprint as numpy array
        """
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.morgan_radius,
            nBits=self.morgan_nbits,
            useFeatures=self.use_features,
            useChirality=self.use_chirality
        )
        return np.array(fp, dtype=np.float32)

    def compute_maccs_keys(self, mol: Chem.Mol) -> np.ndarray:
        """
        Compute MACCS keys fingerprint.

        MACCS keys are 166 predefined structural key descriptors encoding
        common pharmacophore patterns and functional groups.

        Args:
            mol: RDKit molecule

        Returns:
            Binary fingerprint as numpy array (166 bits)
        """
        fp = GetMACCSKeysFingerprint(mol)
        return np.array(fp, dtype=np.float32)

    def compute_rdkit_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        """
        Compute comprehensive RDKit 2D molecular descriptors.

        Includes physicochemical properties relevant for ADMET:
        - Molecular weight and heavy atom count
        - Lipophilicity (logP)
        - Hydrogen bond donors/acceptors
        - Topological polar surface area (TPSA)
        - Rotatable bonds (flexibility)
        - Aromatic rings
        - Formal charge
        - Number of stereocenters

        Args:
            mol: RDKit molecule

        Returns:
            Descriptor vector as numpy array
        """
        try:
            descriptors = [
                Descriptors.MolWt(mol),
                Descriptors.HeavyAtomMolWt(mol),
                Descriptors.NumHeavyAtoms(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumAliphaticRings(mol),
                rdMolDescriptors.CalcFractionCSP3(mol),
                Descriptors.RingCount(mol),
                Descriptors.NumSaturatedRings(mol),
                rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                rdMolDescriptors.CalcNumSpiroAtoms(mol),
                Descriptors.FractionCsp3(mol),
                len(Chem.GetMolFrags(mol)),  # Number of fragments
                rdMolDescriptors.CalcNumAtomStereoCenters(mol),
                rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol),
                Chem.GetFormalCharge(mol),
            ]
            return np.array(descriptors, dtype=np.float32)
        except Exception as e:
            print(f"Error computing descriptors: {e}")
            # Return zeros if calculation fails
            return np.zeros(20, dtype=np.float32)

    def mol_to_graph(self, mol: Chem.Mol, smiles: str = "") -> Optional['Data']:
        """
        Convert molecule to PyTorch Geometric graph representation.

        Graph structure:
        - Nodes: Atoms with feature vectors
        - Edges: Bonds (bidirectional)

        Node features (per atom):
        - One-hot encoded atom type
        - Atomic number
        - Degree (number of bonds)
        - Formal charge
        - Hybridization (sp, sp2, sp3)
        - Aromaticity
        - Number of hydrogens
        - Is in ring

        Edge features (per bond):
        - Bond type (single, double, triple, aromatic)
        - Is conjugated
        - Is in ring

        Args:
            mol: RDKit molecule
            smiles: SMILES string for reference

        Returns:
            PyTorch Geometric Data object or None
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            return None

        try:
            # Remove explicit hydrogens for graph (they add noise)
            mol = Chem.RemoveHs(mol)

            # Build node features
            node_features = []
            for atom in mol.GetAtoms():
                # One-hot encode atom type
                atom_type = atom.GetSymbol()
                if atom_type not in self.atom_type_to_idx:
                    atom_type = 'other'
                atom_type_onehot = [0] * len(self.atom_types)
                atom_type_onehot[self.atom_type_to_idx[atom_type]] = 1

                # Hybridization one-hot
                hybridization = [0, 0, 0, 0, 0]  # sp, sp2, sp3, sp3d, sp3d2
                hyb = atom.GetHybridization()
                if hyb == Chem.HybridizationType.SP:
                    hybridization[0] = 1
                elif hyb == Chem.HybridizationType.SP2:
                    hybridization[1] = 1
                elif hyb == Chem.HybridizationType.SP3:
                    hybridization[2] = 1
                elif hyb == Chem.HybridizationType.SP3D:
                    hybridization[3] = 1
                elif hyb == Chem.HybridizationType.SP3D2:
                    hybridization[4] = 1

                features = (
                    atom_type_onehot +
                    [
                        atom.GetAtomicNum() / 100.0,  # Normalized atomic number
                        atom.GetTotalDegree() / 10.0,  # Normalized degree
                        atom.GetFormalCharge(),
                        atom.GetTotalNumHs() / 8.0,  # Normalized H count
                        int(atom.GetIsAromatic()),
                        int(atom.IsInRing()),
                    ] +
                    hybridization
                )
                node_features.append(features)

            x = torch.tensor(node_features, dtype=torch.float)

            # Build edge index and edge features
            edge_indices = []
            edge_features = []

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                # Bond type one-hot
                bond_type = [0, 0, 0, 0]  # single, double, triple, aromatic
                bt = bond.GetBondType()
                if bt == Chem.BondType.SINGLE:
                    bond_type[0] = 1
                elif bt == Chem.BondType.DOUBLE:
                    bond_type[1] = 1
                elif bt == Chem.BondType.TRIPLE:
                    bond_type[2] = 1
                elif bt == Chem.BondType.AROMATIC:
                    bond_type[3] = 1

                bond_features = bond_type + [
                    int(bond.GetIsConjugated()),
                    int(bond.IsInRing()),
                ]

                # Add both directions (undirected graph)
                edge_indices.extend([[i, j], [j, i]])
                edge_features.extend([bond_features, bond_features])

            if len(edge_indices) == 0:
                # Single atom molecule - create self-loop
                edge_indices = [[0, 0]]
                edge_features = [[0, 0, 0, 0, 0, 0]]

            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

            return Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                smiles=smiles
            )

        except Exception as e:
            print(f"Error converting molecule to graph: {e}")
            return None

    def featurize(
        self,
        smiles: str,
        include_morgan: bool = True,
        include_maccs: bool = True,
        include_descriptors: bool = True,
        include_graph: bool = True
    ) -> Optional[MoleculeFeatures]:
        """
        Compute all requested features for a molecule.

        Args:
            smiles: SMILES string
            include_morgan: Whether to compute Morgan fingerprints
            include_maccs: Whether to compute MACCS keys
            include_descriptors: Whether to compute RDKit descriptors
            include_graph: Whether to create graph representation

        Returns:
            MoleculeFeatures object or None if molecule invalid
        """
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None

        features = MoleculeFeatures(smiles=smiles)

        if include_morgan:
            features.morgan_fp = self.compute_morgan_fingerprint(mol)

        if include_maccs:
            features.maccs_keys = self.compute_maccs_keys(mol)

        if include_descriptors:
            features.rdkit_descriptors = self.compute_rdkit_descriptors(mol)

        if include_graph:
            features.graph_data = self.mol_to_graph(mol, smiles)

        return features

    def featurize_batch(
        self,
        smiles_list: List[str],
        include_morgan: bool = True,
        include_maccs: bool = True,
        include_descriptors: bool = True,
        include_graph: bool = True
    ) -> Tuple[List[MoleculeFeatures], List[int]]:
        """
        Featurize a batch of molecules with error handling.

        Args:
            smiles_list: List of SMILES strings
            include_morgan: Whether to compute Morgan fingerprints
            include_maccs: Whether to compute MACCS keys
            include_descriptors: Whether to compute RDKit descriptors
            include_graph: Whether to create graph representation

        Returns:
            Tuple of (feature list, list of failed indices)
        """
        features = []
        failed_indices = []

        for idx, smiles in enumerate(smiles_list):
            feat = self.featurize(
                smiles,
                include_morgan=include_morgan,
                include_maccs=include_maccs,
                include_descriptors=include_descriptors,
                include_graph=include_graph
            )

            if feat is None:
                failed_indices.append(idx)
            else:
                features.append(feat)

        return features, failed_indices

    def collate_graphs(self, features: List[MoleculeFeatures]) -> Optional['Batch']:
        """
        Collate graph representations into a batch for PyTorch Geometric.

        Args:
            features: List of MoleculeFeatures with graph_data

        Returns:
            PyTorch Geometric Batch object or None
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            return None

        graphs = [f.graph_data for f in features if f.graph_data is not None]
        if len(graphs) == 0:
            return None

        return Batch.from_data_list(graphs)

    def get_combined_fingerprint(
        self,
        features: MoleculeFeatures,
        include_morgan: bool = True,
        include_maccs: bool = True,
        include_descriptors: bool = True
    ) -> np.ndarray:
        """
        Concatenate multiple fingerprint types into single feature vector.

        Useful for non-graph models that require fixed-size input.

        Args:
            features: MoleculeFeatures object
            include_morgan: Include Morgan fingerprint
            include_maccs: Include MACCS keys
            include_descriptors: Include RDKit descriptors

        Returns:
            Concatenated feature vector
        """
        components = []

        if include_morgan and features.morgan_fp is not None:
            components.append(features.morgan_fp)

        if include_maccs and features.maccs_keys is not None:
            components.append(features.maccs_keys)

        if include_descriptors and features.rdkit_descriptors is not None:
            components.append(features.rdkit_descriptors)

        if len(components) == 0:
            raise ValueError("No features available to concatenate")

        return np.concatenate(components)


def test_featurizer():
    """Test featurizer with example molecules"""
    featurizer = MolecularFeaturizer()

    test_smiles = [
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "invalid_smiles",  # Should fail gracefully
    ]

    print("Testing molecular featurization...")
    features, failed = featurizer.featurize_batch(test_smiles)

    print(f"\nSuccessfully featurized: {len(features)}/{len(test_smiles)}")
    print(f"Failed indices: {failed}")

    if len(features) > 0:
        feat = features[0]
        print(f"\nExample features for '{feat.smiles}':")
        if feat.morgan_fp is not None:
            print(f"  Morgan FP shape: {feat.morgan_fp.shape}")
        if feat.maccs_keys is not None:
            print(f"  MACCS keys shape: {feat.maccs_keys.shape}")
        if feat.rdkit_descriptors is not None:
            print(f"  RDKit descriptors shape: {feat.rdkit_descriptors.shape}")
        if feat.graph_data is not None:
            print(f"  Graph nodes: {feat.graph_data.x.shape[0]}")
            print(f"  Graph edges: {feat.graph_data.edge_index.shape[1]}")

        combined = featurizer.get_combined_fingerprint(feat)
        print(f"  Combined fingerprint shape: {combined.shape}")


if __name__ == "__main__":
    test_featurizer()
