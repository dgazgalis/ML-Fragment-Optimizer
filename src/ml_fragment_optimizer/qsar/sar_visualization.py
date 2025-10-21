"""
SAR Visualization Tools

Publication-quality visualizations for Structure-Activity Relationship analysis.

References:
    - Bajorath (2012) "SAR Visualization Concepts" Methods Mol. Biol. 819, 549-578
    - Wawer & Bajorath (2011) "Local Structural Changes, Global Data Views"
      J. Med. Chem. 54, 2944-2951
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 10


class SARVisualizer:
    """
    Visualizer for Structure-Activity Relationship data.

    Examples:
        >>> from rdkit import Chem
        >>> smiles = ["c1ccccc1", "c1ccc(Cl)cc1", "c1ccc(F)cc1"]
        >>> mols = [Chem.MolFromSmiles(s) for s in smiles]
        >>> activities = [5.0, 6.5, 6.8]
        >>>
        >>> viz = SARVisualizer()
        >>> viz.plot_activity_landscape(mols, activities, method='pca')
        >>> plt.savefig('activity_landscape.png')
    """

    def __init__(self, figsize: Tuple[int, int] = (8, 6)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size (width, height) in inches
        """
        self.figsize = figsize

    def plot_activity_landscape(
        self,
        mols: List[Chem.Mol],
        activities: List[float],
        method: str = 'pca',
        n_components: int = 2,
        colormap: str = 'viridis',
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot 2D activity landscape with molecules projected into chemical space.

        Args:
            mols: List of molecules
            activities: Activity values
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            n_components: Number of dimensions (2 or 3)
            colormap: Matplotlib colormap name
            title: Plot title
            ax: Matplotlib axes (creates new if None)

        Returns:
            Matplotlib axes
        """
        if n_components not in [2, 3]:
            raise ValueError("n_components must be 2 or 3")

        # Calculate fingerprints
        fps = self._calculate_fingerprints(mols)
        X = np.array(fps)

        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=n_components)
            coords = reducer.fit_transform(X)
            xlabel = f"PC1 ({reducer.explained_variance_ratio_[0]:.1%})"
            ylabel = f"PC2 ({reducer.explained_variance_ratio_[1]:.1%})"
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
            coords = reducer.fit_transform(X)
            xlabel = "t-SNE 1"
            ylabel = "t-SNE 2"
        elif method == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not installed. Install with: pip install umap-learn")
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            coords = reducer.fit_transform(X)
            xlabel = "UMAP 1"
            ylabel = "UMAP 2"
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create plot
        if ax is None:
            if n_components == 2:
                fig, ax = plt.subplots(figsize=self.figsize)
            else:
                fig = plt.figure(figsize=self.figsize)
                ax = fig.add_subplot(111, projection='3d')

        # Plot
        if n_components == 2:
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=activities,
                cmap=colormap,
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2],
                c=activities,
                cmap=colormap,
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel("Component 3")

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Activity')

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Activity Landscape ({method.upper()})')

        ax.grid(True, alpha=0.3)

        return ax

    def plot_similarity_heatmap(
        self,
        mols: List[Chem.Mol],
        activities: Optional[List[float]] = None,
        mol_labels: Optional[List[str]] = None,
        title: str = "Molecular Similarity Matrix",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot heatmap of pairwise molecular similarities.

        Args:
            mols: List of molecules
            activities: Optional activity values for ordering
            mol_labels: Optional labels for molecules
            title: Plot title
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        from rdkit import DataStructs

        # Calculate similarity matrix
        fps = self._calculate_fingerprints(mols, as_bitvect=True)
        n = len(fps)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                sim_matrix[i, j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])

        # Order by activity if provided
        if activities is not None:
            order = np.argsort(activities)
            sim_matrix = sim_matrix[order, :][:, order]
            if mol_labels:
                mol_labels = [mol_labels[i] for i in order]

        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        # Heatmap
        sns.heatmap(
            sim_matrix,
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Tanimoto Similarity'},
            ax=ax,
            xticklabels=mol_labels if mol_labels else False,
            yticklabels=mol_labels if mol_labels else False,
        )

        ax.set_title(title)

        return ax

    def plot_activity_cliffs(
        self,
        mols: List[Chem.Mol],
        activities: List[float],
        cliffs: List[Tuple[int, int]],
        method: str = 'pca',
        title: str = "Activity Cliffs",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot activity landscape with cliff pairs highlighted.

        Args:
            mols: List of molecules
            activities: Activity values
            cliffs: List of (mol1_idx, mol2_idx) tuples forming cliffs
            method: Dimensionality reduction method
            title: Plot title
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        # Plot landscape
        ax = self.plot_activity_landscape(
            mols, activities, method=method, title=title, ax=ax
        )

        # Get coordinates
        fps = self._calculate_fingerprints(mols)
        X = np.array(fps)

        if method == 'pca':
            reducer = PCA(n_components=2)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not installed")
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

        coords = reducer.fit_transform(X)

        # Draw lines connecting cliff pairs
        for i, j in cliffs:
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                'r-',
                alpha=0.5,
                linewidth=1.5,
                zorder=1,
            )

        return ax

    def plot_free_wilson_contributions(
        self,
        contributions: Dict[str, Dict[str, float]],
        title: str = "Free-Wilson Substituent Contributions",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot Free-Wilson substituent contributions as grouped bar chart.

        Args:
            contributions: Dict mapping position to {substituent: contribution}
            title: Plot title
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        positions = list(contributions.keys())
        n_positions = len(positions)

        # Prepare data
        all_substituents = sorted(set(
            sub for subs in contributions.values() for sub in subs.keys()
        ))

        x = np.arange(len(all_substituents))
        width = 0.8 / n_positions

        # Plot bars for each position
        for i, pos in enumerate(positions):
            values = [contributions[pos].get(sub, 0) for sub in all_substituents]
            offset = (i - n_positions / 2) * width + width / 2
            ax.bar(x + offset, values, width, label=pos)

        ax.set_xlabel('Substituent')
        ax.set_ylabel('Contribution to Activity')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(all_substituents, rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return ax

    def plot_matched_pair_network(
        self,
        pairs: List[Tuple[int, int, float]],
        mols: List[Chem.Mol],
        activities: List[float],
        title: str = "Matched Molecular Pair Network",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot network graph of matched molecular pairs.

        Args:
            pairs: List of (mol1_idx, mol2_idx, property_change) tuples
            mols: List of molecules
            activities: Activity values
            title: Plot title
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx required for network plots. Install with: pip install networkx")

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        # Create graph
        G = nx.Graph()
        for i in range(len(mols)):
            G.add_node(i, activity=activities[i])

        for i, j, change in pairs:
            G.add_edge(i, j, weight=abs(change))

        # Layout
        pos = nx.spring_layout(G, seed=42)

        # Node colors by activity
        node_colors = [activities[i] for i in G.nodes()]

        # Edge widths by property change
        edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
        max_width = max(edge_widths) if edge_widths else 1
        edge_widths = [w / max_width * 3 for w in edge_widths]

        # Draw
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=300,
            cmap='viridis',
            ax=ax,
            edgecolors='black',
            linewidths=1,
        )

        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.5,
            ax=ax,
        )

        ax.set_title(title)
        ax.axis('off')

        # Colorbar for activities
        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_array(activities)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Activity')

        return ax

    def plot_sali_heatmap(
        self,
        sali_matrix: np.ndarray,
        mol_labels: Optional[List[str]] = None,
        title: str = "SALI Matrix (Activity Cliff Strength)",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot SALI (Structure-Activity Landscape Index) heatmap.

        Args:
            sali_matrix: n×n SALI matrix
            mol_labels: Optional labels for molecules
            title: Plot title
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        # Use log scale for better visualization
        sali_log = np.log10(sali_matrix + 1)

        sns.heatmap(
            sali_log,
            cmap='YlOrRd',
            square=True,
            cbar_kws={'label': 'log₁₀(SALI + 1)'},
            ax=ax,
            xticklabels=mol_labels if mol_labels else False,
            yticklabels=mol_labels if mol_labels else False,
        )

        ax.set_title(title)

        return ax

    def plot_molecule_grid(
        self,
        mols: List[Chem.Mol],
        activities: List[float],
        legends: Optional[List[str]] = None,
        mols_per_row: int = 4,
        mol_size: Tuple[int, int] = (200, 200),
        filename: Optional[str] = None,
    ):
        """
        Draw grid of molecules with activity labels.

        Args:
            mols: List of molecules
            activities: Activity values
            legends: Optional custom legends (defaults to activity values)
            mols_per_row: Number of molecules per row
            mol_size: Size of each molecule image (width, height)
            filename: Optional filename to save image
        """
        if legends is None:
            legends = [f"Activity: {act:.2f}" for act in activities]

        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_row,
            subImgSize=mol_size,
            legends=legends,
        )

        if filename:
            img.save(filename)

        return img

    def _calculate_fingerprints(
        self,
        mols: List[Chem.Mol],
        as_bitvect: bool = False,
    ):
        """Calculate Morgan fingerprints."""
        fps = []

        for mol in mols:
            if mol is None:
                fps.append(None)
                continue

            if as_bitvect:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fp_array = np.zeros((2048,))
                DataStructs.ConvertToNumpyArray(fp, fp_array)
                fp = fp_array

            fps.append(fp)

        return fps


def plot_activity_landscape(
    mols: List[Chem.Mol],
    activities: List[float],
    method: str = 'pca',
    filename: Optional[str] = None,
):
    """
    Convenience function to plot activity landscape.

    Args:
        mols: List of molecules
        activities: Activity values
        method: Dimensionality reduction method
        filename: Optional filename to save plot

    Examples:
        >>> from rdkit import Chem
        >>> smiles = ["c1ccccc1", "c1ccc(Cl)cc1", "c1ccc(F)cc1"]
        >>> mols = [Chem.MolFromSmiles(s) for s in smiles]
        >>> activities = [5.0, 6.5, 6.8]
        >>> plot_activity_landscape(mols, activities, method='tsne')
    """
    viz = SARVisualizer()
    viz.plot_activity_landscape(mols, activities, method=method)

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    from rdkit import Chem
    import matplotlib.pyplot as plt

    print("=== SAR Visualization Examples ===\n")

    # Create example dataset
    smiles_list = [
        "c1ccccc1",          # benzene
        "Fc1ccccc1",         # fluorobenzene
        "Clc1ccccc1",        # chlorobenzene
        "Brc1ccccc1",        # bromobenzene
        "Cc1ccccc1",         # toluene
        "c1ccc(F)cc1",       # para-fluorobenzene
        "c1ccc(Cl)cc1",      # para-chlorobenzene
        "c1ccc([N+](=O)[O-])cc1",  # para-nitrobenzene
    ]

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    activities = [5.0, 5.5, 6.0, 6.5, 5.2, 5.8, 6.2, 8.5]

    viz = SARVisualizer(figsize=(10, 8))

    # 1. Activity landscape
    print("Creating activity landscape...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    viz.plot_activity_landscape(mols, activities, method='pca', ax=axes[0, 0])
    viz.plot_activity_landscape(mols, activities, method='tsne', ax=axes[0, 1])

    # 2. Similarity heatmap
    print("Creating similarity heatmap...")
    viz.plot_similarity_heatmap(mols, activities, ax=axes[1, 0])

    # 3. Activity cliffs
    print("Highlighting activity cliffs...")
    cliffs = [(0, 7), (6, 7)]  # Cliff pairs
    viz.plot_activity_cliffs(mols, activities, cliffs, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig('sar_visualization_examples.png', dpi=300, bbox_inches='tight')
    print("Saved: sar_visualization_examples.png")

    # 4. Molecule grid
    print("\nCreating molecule grid...")
    img = viz.plot_molecule_grid(
        mols,
        activities,
        mols_per_row=4,
        filename='molecule_grid.png'
    )
    print("Saved: molecule_grid.png")

    print("\nVisualization examples completed!")
