"""
File format conversion utilities for molecular structures.

This module provides functions for converting between various molecular
file formats: SMILES, SDF, MOL2, PDB, InChI, etc.
"""

from typing import Optional, List, Union
import pandas as pd
from pathlib import Path
import warnings
from io import StringIO

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolAlign
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    raise ImportError("RDKit is required. Install with: pip install rdkit")

from tqdm import tqdm


class MoleculeConverter:
    """Convert molecules between different formats."""

    @staticmethod
    def smiles_to_mol(smiles: str, sanitize: bool = True) -> Optional[Chem.Mol]:
        """
        Convert SMILES to RDKit mol object.

        Args:
            smiles: SMILES string
            sanitize: Sanitize molecule

        Returns:
            RDKit molecule
        """
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
            return mol
        except:
            return None

    @staticmethod
    def mol_to_smiles(
        mol: Chem.Mol,
        canonical: bool = True,
        isomeric: bool = True
    ) -> Optional[str]:
        """
        Convert RDKit mol to SMILES.

        Args:
            mol: RDKit molecule
            canonical: Generate canonical SMILES
            isomeric: Include stereochemistry

        Returns:
            SMILES string
        """
        if mol is None:
            return None

        try:
            return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric)
        except:
            return None

    @staticmethod
    def mol_to_inchi(mol: Chem.Mol) -> Optional[str]:
        """
        Convert mol to InChI.

        Args:
            mol: RDKit molecule

        Returns:
            InChI string
        """
        if mol is None:
            return None

        try:
            return Chem.MolToInchi(mol)
        except:
            return None

    @staticmethod
    def mol_to_inchikey(mol: Chem.Mol) -> Optional[str]:
        """
        Convert mol to InChIKey.

        Args:
            mol: RDKit molecule

        Returns:
            InChIKey string
        """
        if mol is None:
            return None

        try:
            return Chem.MolToInchiKey(mol)
        except:
            return None

    @staticmethod
    def inchi_to_mol(inchi: str) -> Optional[Chem.Mol]:
        """
        Convert InChI to mol.

        Args:
            inchi: InChI string

        Returns:
            RDKit molecule
        """
        try:
            return Chem.MolFromInchi(inchi)
        except:
            return None

    @staticmethod
    def generate_2d_coords(mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Generate 2D coordinates.

        Args:
            mol: RDKit molecule

        Returns:
            Molecule with 2D coordinates
        """
        if mol is None:
            return None

        try:
            mol = Chem.Mol(mol)  # Copy
            AllChem.Compute2DCoords(mol)
            return mol
        except:
            return None

    @staticmethod
    def generate_3d_coords(
        mol: Chem.Mol,
        optimize: bool = True,
        max_iterations: int = 500
    ) -> Optional[Chem.Mol]:
        """
        Generate 3D coordinates.

        Args:
            mol: RDKit molecule
            optimize: Optimize with force field
            max_iterations: Maximum optimization iterations

        Returns:
            Molecule with 3D coordinates
        """
        if mol is None:
            return None

        try:
            mol = Chem.AddHs(mol)
            result = AllChem.EmbedMolecule(mol, randomSeed=42)

            if result != 0:
                warnings.warn("3D embedding failed")
                return None

            if optimize:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iterations)

            return mol
        except:
            return None


class FileConverter:
    """Convert between molecular file formats."""

    @staticmethod
    def smiles_to_sdf(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        smiles_col: str = "smiles",
        id_col: Optional[str] = None,
        generate_3d: bool = False,
        **kwargs
    ) -> int:
        """
        Convert SMILES file to SDF.

        Args:
            input_file: Input CSV/TSV with SMILES
            output_file: Output SDF file
            smiles_col: Column with SMILES
            id_col: Column with IDs
            generate_3d: Generate 3D coordinates
            **kwargs: Additional pd.read_csv arguments

        Returns:
            Number of molecules written
        """
        df = pd.read_csv(input_file, **kwargs)

        if smiles_col not in df.columns:
            raise ValueError(f"Column '{smiles_col}' not found")

        writer = Chem.SDWriter(str(output_file))
        n_written = 0

        print(f"Converting {len(df)} molecules to SDF...")

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row[smiles_col]

            if pd.isna(smiles):
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Generate coordinates
            if generate_3d:
                mol = MoleculeConverter.generate_3d_coords(mol)
            else:
                mol = MoleculeConverter.generate_2d_coords(mol)

            if mol is None:
                continue

            # Set properties
            if id_col and id_col in row:
                mol.SetProp("_Name", str(row[id_col]))

            for col in df.columns:
                if col not in [smiles_col, id_col]:
                    value = row[col]
                    if not pd.isna(value):
                        mol.SetProp(col, str(value))

            writer.write(mol)
            n_written += 1

        writer.close()
        print(f"Wrote {n_written} molecules to {output_file}")

        return n_written

    @staticmethod
    def sdf_to_smiles(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        include_properties: bool = True
    ) -> int:
        """
        Convert SDF to SMILES file.

        Args:
            input_file: Input SDF file
            output_file: Output CSV file
            include_properties: Include molecule properties

        Returns:
            Number of molecules written
        """
        supplier = Chem.SDMolSupplier(str(input_file))

        data = []
        print(f"Reading molecules from {input_file}...")

        for idx, mol in enumerate(tqdm(supplier)):
            if mol is None:
                continue

            smiles = Chem.MolToSmiles(mol)

            record = {
                'mol_id': mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{idx:06d}",
                'smiles': smiles
            }

            if include_properties:
                for prop in mol.GetPropNames():
                    if not prop.startswith("_"):
                        record[prop] = mol.GetProp(prop)

            data.append(record)

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

        print(f"Wrote {len(df)} molecules to {output_file}")

        return len(df)

    @staticmethod
    def sdf_to_mol2(
        input_file: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> int:
        """
        Convert SDF to MOL2 files (one per molecule).

        Args:
            input_file: Input SDF file
            output_dir: Output directory for MOL2 files

        Returns:
            Number of molecules written
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        supplier = Chem.SDMolSupplier(str(input_file))
        n_written = 0

        print(f"Converting molecules to MOL2...")

        for idx, mol in enumerate(tqdm(supplier)):
            if mol is None:
                continue

            mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{idx:06d}"
            output_file = output_dir / f"{mol_name}.mol2"

            try:
                Chem.MolToMol2File(mol, str(output_file))
                n_written += 1
            except Exception as e:
                warnings.warn(f"Failed to write {mol_name}: {e}")

        print(f"Wrote {n_written} MOL2 files to {output_dir}")

        return n_written

    @staticmethod
    def mol2_to_sdf(
        input_dir: Union[str, Path],
        output_file: Union[str, Path]
    ) -> int:
        """
        Convert MOL2 files to SDF.

        Args:
            input_dir: Directory with MOL2 files
            output_file: Output SDF file

        Returns:
            Number of molecules written
        """
        input_dir = Path(input_dir)
        mol2_files = list(input_dir.glob("*.mol2"))

        writer = Chem.SDWriter(str(output_file))
        n_written = 0

        print(f"Converting {len(mol2_files)} MOL2 files to SDF...")

        for mol2_file in tqdm(mol2_files):
            try:
                mol = Chem.MolFromMol2File(str(mol2_file))
                if mol:
                    mol.SetProp("_Name", mol2_file.stem)
                    writer.write(mol)
                    n_written += 1
            except Exception as e:
                warnings.warn(f"Failed to read {mol2_file.name}: {e}")

        writer.close()
        print(f"Wrote {n_written} molecules to {output_file}")

        return n_written

    @staticmethod
    def stream_convert_large_file(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        input_format: str,
        output_format: str,
        chunk_size: int = 1000
    ) -> int:
        """
        Convert large files in chunks.

        Args:
            input_file: Input file
            output_file: Output file
            input_format: Input format (smiles, sdf)
            output_format: Output format (smiles, sdf)
            chunk_size: Number of molecules per chunk

        Returns:
            Number of molecules converted
        """
        if input_format == "smiles" and output_format == "sdf":
            # Read in chunks
            chunks = pd.read_csv(input_file, chunksize=chunk_size)
            writer = Chem.SDWriter(str(output_file))

            n_written = 0
            for chunk_idx, chunk in enumerate(chunks):
                print(f"Processing chunk {chunk_idx + 1}...")

                for idx, row in chunk.iterrows():
                    smiles = row.get('smiles', None)
                    if pd.isna(smiles):
                        continue

                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        mol = MoleculeConverter.generate_2d_coords(mol)
                        if mol:
                            writer.write(mol)
                            n_written += 1

            writer.close()
            return n_written

        else:
            raise NotImplementedError(f"Conversion from {input_format} to {output_format} not implemented")


class IdentifierConverter:
    """Convert between molecular identifiers."""

    @staticmethod
    def add_identifiers(df: pd.DataFrame, mol_col: str = "mol") -> pd.DataFrame:
        """
        Add molecular identifiers to DataFrame.

        Args:
            df: DataFrame with mol column
            mol_col: Name of mol column

        Returns:
            DataFrame with added identifier columns
        """
        print("Generating molecular identifiers...")

        # Canonical SMILES
        df['canonical_smiles'] = [
            Chem.MolToSmiles(mol) if mol else None
            for mol in tqdm(df[mol_col], desc="SMILES")
        ]

        # InChI
        df['inchi'] = [
            Chem.MolToInchi(mol) if mol else None
            for mol in tqdm(df[mol_col], desc="InChI")
        ]

        # InChIKey
        df['inchikey'] = [
            Chem.MolToInchiKey(mol) if mol else None
            for mol in tqdm(df[mol_col], desc="InChIKey")
        ]

        # Molecular formula
        df['formula'] = [
            Chem.rdMolDescriptors.CalcMolFormula(mol) if mol else None
            for mol in tqdm(df[mol_col], desc="Formula")
        ]

        return df


class BatchProcessor:
    """Process molecules in batches for memory efficiency."""

    @staticmethod
    def process_large_sdf(
        filepath: Union[str, Path],
        process_fn,
        batch_size: int = 1000,
        **kwargs
    ):
        """
        Process large SDF file in batches.

        Args:
            filepath: Path to SDF file
            process_fn: Function to apply to each batch
            batch_size: Number of molecules per batch
            **kwargs: Additional arguments for process_fn

        Yields:
            Results from process_fn
        """
        supplier = Chem.SDMolSupplier(str(filepath))

        batch = []
        for idx, mol in enumerate(supplier):
            if mol is not None:
                batch.append(mol)

            if len(batch) >= batch_size:
                result = process_fn(batch, **kwargs)
                yield result
                batch = []

        # Process remaining
        if batch:
            result = process_fn(batch, **kwargs)
            yield result


if __name__ == "__main__":
    print("Format Conversion Utilities")
    print("=" * 50)

    # Example: Convert between formats
    smiles_list = [
        "CCO",
        "CC(=O)O",
        "c1ccccc1",
    ]

    # Create test DataFrame
    df = pd.DataFrame({
        'mol_id': ['mol1', 'mol2', 'mol3'],
        'smiles': smiles_list,
        'activity': [1.2, 3.4, 5.6]
    })

    # Add identifiers
    df['mol'] = [Chem.MolFromSmiles(s) for s in df['smiles']]
    converter = IdentifierConverter()
    df = converter.add_identifiers(df)

    print("\nDataFrame with identifiers:")
    print(df[['mol_id', 'smiles', 'inchi', 'inchikey', 'formula']])

    # Test SMILES to SDF conversion
    print("\n" + "=" * 50)
    print("Testing SMILES to SDF conversion...")

    # Save test SMILES file
    test_smiles = Path("test_molecules.csv")
    df[['mol_id', 'smiles', 'activity']].to_csv(test_smiles, index=False)

    # Convert to SDF
    test_sdf = Path("test_molecules.sdf")
    n_written = FileConverter.smiles_to_sdf(
        test_smiles,
        test_sdf,
        generate_3d=False
    )

    print(f"Wrote {n_written} molecules to SDF")

    # Convert back to SMILES
    print("\n" + "=" * 50)
    print("Testing SDF to SMILES conversion...")

    test_output = Path("test_molecules_output.csv")
    n_written = FileConverter.sdf_to_smiles(test_sdf, test_output)

    print(f"Wrote {n_written} molecules to CSV")

    # Clean up
    test_smiles.unlink()
    test_sdf.unlink()
    test_output.unlink()

    print("\nConversion test completed successfully!")
