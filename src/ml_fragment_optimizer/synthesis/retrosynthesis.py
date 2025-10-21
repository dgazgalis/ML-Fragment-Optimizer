"""
Retrosynthesis Analysis Module

Provides retrosynthetic route generation and analysis using:
1. AiZynthFinder (if available) - ML-based retrosynthesis
2. Template-based retrosynthesis (fallback) - reaction template matching
3. Simple disconnection analysis (basic fallback)

Author: ML-Fragment-Optimizer Team
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import tempfile

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. Retrosynthesis will not work.")

logger = logging.getLogger(__name__)


class RouteType(Enum):
    """Type of synthetic route"""
    LINEAR = "linear"           # Sequential steps
    CONVERGENT = "convergent"   # Multiple branches that merge
    DIVERGENT = "divergent"     # Single starting material, multiple intermediates


@dataclass
class ReactionStep:
    """Single reaction step in a synthesis route"""
    step_number: int
    reactants: List[str]  # SMILES of reactants
    product: str          # SMILES of product
    reaction_type: Optional[str] = None
    template_id: Optional[str] = None
    confidence: float = 0.0  # 0-1 score for this step

    def to_dict(self) -> Dict:
        return {
            'step': self.step_number,
            'reactants': self.reactants,
            'product': self.product,
            'reaction_type': self.reaction_type,
            'template_id': self.template_id,
            'confidence': round(self.confidence, 3)
        }


@dataclass
class RetrosynthesisRoute:
    """Complete retrosynthetic route"""
    target_smiles: str
    route_id: int
    steps: List[ReactionStep] = field(default_factory=list)
    building_blocks: List[str] = field(default_factory=list)  # Final starting materials
    route_type: RouteType = RouteType.LINEAR
    overall_score: float = 0.0
    method: str = "unknown"  # aizynthfinder, template, simple

    def to_dict(self) -> Dict:
        return {
            'target': self.target_smiles,
            'route_id': self.route_id,
            'num_steps': len(self.steps),
            'steps': [step.to_dict() for step in self.steps],
            'building_blocks': self.building_blocks,
            'route_type': self.route_type.value,
            'overall_score': round(self.overall_score, 3),
            'method': self.method,
            'metrics': self.get_metrics()
        }

    def get_metrics(self) -> Dict:
        """Calculate route metrics"""
        return {
            'num_steps': len(self.steps),
            'num_building_blocks': len(self.building_blocks),
            'convergence_degree': self._calculate_convergence(),
            'avg_step_confidence': sum(s.confidence for s in self.steps) / len(self.steps) if self.steps else 0.0
        }

    def _calculate_convergence(self) -> float:
        """
        Calculate convergence degree

        0 = fully linear
        1 = fully convergent (binary tree)
        """
        if len(self.steps) <= 1:
            return 0.0

        # Count branching points (steps with >1 reactant from previous steps)
        branches = 0
        for step in self.steps:
            if len(step.reactants) > 1:
                branches += 1

        # Normalize
        max_branches = len(self.steps) - 1
        return branches / max_branches if max_branches > 0 else 0.0


class RetrosynthesisAnalyzer:
    """
    Retrosynthesis route generation and analysis

    Supports multiple backends:
    1. AiZynthFinder (requires installation and models)
    2. Template-based retrosynthesis (built-in, limited)
    3. Simple disconnection (very basic fallback)

    AiZynthFinder Setup:
        pip install aizynthfinder
        Download models from: https://github.com/MolecularAI/aizynthfinder
        Configure model paths in config file

    Example:
        >>> analyzer = RetrosynthesisAnalyzer()
        >>> routes = analyzer.analyze('CC(=O)Oc1ccccc1C(=O)O')  # Aspirin
        >>> for route in routes[:3]:
        ...     print(f"Route {route.route_id}: {route.num_steps} steps")
    """

    def __init__(
        self,
        aizynthfinder_config: Optional[Path] = None,
        use_aizynthfinder: bool = True,
        max_routes: int = 5,
        max_steps: int = 10
    ):
        """
        Initialize retrosynthesis analyzer

        Args:
            aizynthfinder_config: Path to AiZynthFinder config YAML
            use_aizynthfinder: Try to use AiZynthFinder if available
            max_routes: Maximum number of routes to generate
            max_steps: Maximum steps per route
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for retrosynthesis analysis")

        self.max_routes = max_routes
        self.max_steps = max_steps

        # Try to import AiZynthFinder
        self.aizynthfinder_available = False
        if use_aizynthfinder:
            try:
                import aizynthfinder
                self.aizynthfinder_available = True
                self.aizynthfinder_config = aizynthfinder_config
                logger.info("AiZynthFinder available")
            except ImportError:
                logger.warning("AiZynthFinder not available. Using fallback methods.")

        # Load reaction templates for fallback method
        self.reaction_templates = self._load_default_templates()

    def _load_default_templates(self) -> List[Dict]:
        """
        Load default reaction templates for template-based retrosynthesis

        These are simplified SMARTS patterns for common reactions
        """
        # Common reaction templates (name, forward SMARTS, retro SMARTS)
        templates = [
            {
                'name': 'ester_formation',
                'retro_smarts': '[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])O.[O:3][C:4]',
                'type': 'condensation'
            },
            {
                'name': 'amide_formation',
                'retro_smarts': '[C:1](=[O:2])[N:3][C:4]>>[C:1](=[O:2])O.[N:3][C:4]',
                'type': 'condensation'
            },
            {
                'name': 'reductive_amination',
                'retro_smarts': '[C:1][N:2][C:3]>>[C:1]=O.[N:2][C:3]',
                'type': 'reduction'
            },
            {
                'name': 'suzuki_coupling',
                'retro_smarts': '[c:1][c:2]>>[c:1]Br.[c:2]B(O)O',
                'type': 'coupling'
            },
            {
                'name': 'williamson_ether',
                'retro_smarts': '[C:1][O:2][C:3]>>[C:1]O.[C:3]Br',
                'type': 'substitution'
            },
            {
                'name': 'alcohol_oxidation',
                'retro_smarts': '[C:1]=[O:2]>>[C:1][O:2]',
                'type': 'oxidation'
            },
        ]

        return templates

    def analyze(
        self,
        smiles: str,
        method: str = "auto"  # auto, aizynthfinder, template, simple
    ) -> List[RetrosynthesisRoute]:
        """
        Generate retrosynthetic routes for a target molecule

        Args:
            smiles: Target molecule SMILES
            method: Method to use (auto selects best available)

        Returns:
            List of RetrosynthesisRoute objects, sorted by score
        """
        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        canonical_smiles = Chem.MolToSmiles(mol)

        # Select method
        if method == "auto":
            if self.aizynthfinder_available:
                method = "aizynthfinder"
            else:
                method = "template"

        # Generate routes
        if method == "aizynthfinder" and self.aizynthfinder_available:
            routes = self._analyze_aizynthfinder(canonical_smiles)
        elif method == "template":
            routes = self._analyze_template_based(canonical_smiles)
        else:
            routes = self._analyze_simple(canonical_smiles)

        # Sort by score
        routes.sort(key=lambda r: r.overall_score, reverse=True)

        return routes[:self.max_routes]

    def _analyze_aizynthfinder(self, smiles: str) -> List[RetrosynthesisRoute]:
        """
        Generate routes using AiZynthFinder

        This requires AiZynthFinder to be installed and configured
        """
        try:
            from aizynthfinder.aizynthfinder import AiZynthFinder
            from aizynthfinder.utils.paths import data_path

            # Initialize AiZynthFinder
            finder = AiZynthFinder()

            # Load configuration
            if self.aizynthfinder_config and self.aizynthfinder_config.exists():
                finder.config.load(str(self.aizynthfinder_config))
            else:
                # Use default configuration (if available)
                logger.warning("No AiZynthFinder config provided, using defaults")

            # Set target molecule
            finder.target_smiles = smiles

            # Run tree search
            finder.tree_search()

            # Extract routes
            routes = []
            for i, route_node in enumerate(finder.routes[:self.max_routes]):
                route = self._convert_aizynthfinder_route(route_node, i, smiles)
                routes.append(route)

            logger.info(f"AiZynthFinder generated {len(routes)} routes")
            return routes

        except Exception as e:
            logger.error(f"AiZynthFinder analysis failed: {e}")
            logger.info("Falling back to template-based method")
            return self._analyze_template_based(smiles)

    def _convert_aizynthfinder_route(
        self,
        route_node,
        route_id: int,
        target_smiles: str
    ) -> RetrosynthesisRoute:
        """Convert AiZynthFinder route to RetrosynthesisRoute"""
        steps = []
        building_blocks = []

        # Extract steps from AiZynthFinder route tree
        # (This is simplified - actual implementation depends on AiZynthFinder API)

        try:
            # Get reactions from route
            reactions = route_node.reactions()

            for i, reaction in enumerate(reactions):
                step = ReactionStep(
                    step_number=i + 1,
                    reactants=[Chem.MolToSmiles(m) for m in reaction.reactants],
                    product=Chem.MolToSmiles(reaction.product),
                    reaction_type=reaction.metadata.get('classification', 'unknown'),
                    confidence=reaction.metadata.get('probability', 0.0)
                )
                steps.append(step)

            # Get building blocks (leaf nodes)
            building_blocks = [Chem.MolToSmiles(m) for m in route_node.leafs()]

            # Calculate overall score
            score = route_node.score if hasattr(route_node, 'score') else 0.5

        except Exception as e:
            logger.warning(f"Error parsing AiZynthFinder route: {e}")
            # Create minimal route
            step = ReactionStep(
                step_number=1,
                reactants=['C', 'C'],  # Placeholder
                product=target_smiles,
                confidence=0.5
            )
            steps = [step]
            building_blocks = ['C', 'C']
            score = 0.5

        route = RetrosynthesisRoute(
            target_smiles=target_smiles,
            route_id=route_id,
            steps=steps,
            building_blocks=building_blocks,
            overall_score=score,
            method="aizynthfinder"
        )

        return route

    def _analyze_template_based(self, smiles: str) -> List[RetrosynthesisRoute]:
        """
        Generate routes using reaction template matching

        This is a simplified approach using predefined SMARTS templates
        """
        mol = Chem.MolFromSmiles(smiles)
        routes = []

        logger.info(f"Running template-based retrosynthesis for {smiles}")

        # Try each template
        for template_idx, template in enumerate(self.reaction_templates):
            try:
                rxn = AllChem.ReactionFromSmarts(template['retro_smarts'])
                if rxn is None:
                    continue

                # Run reaction backwards
                products = rxn.RunReactants((mol,))

                if not products:
                    continue

                # Create route for each product set
                for prod_idx, product_set in enumerate(products[:2]):  # Top 2 per template
                    reactants = [Chem.MolToSmiles(m) for m in product_set]

                    # Create simple one-step route
                    step = ReactionStep(
                        step_number=1,
                        reactants=reactants,
                        product=smiles,
                        reaction_type=template['type'],
                        template_id=template['name'],
                        confidence=0.6  # Fixed confidence for template matching
                    )

                    route = RetrosynthesisRoute(
                        target_smiles=smiles,
                        route_id=len(routes),
                        steps=[step],
                        building_blocks=reactants,
                        overall_score=0.6,
                        method="template"
                    )

                    routes.append(route)

            except Exception as e:
                logger.debug(f"Template {template['name']} failed: {e}")
                continue

        if not routes:
            logger.warning("No template matches found, using simple disconnection")
            return self._analyze_simple(smiles)

        logger.info(f"Template-based method generated {len(routes)} routes")
        return routes

    def _analyze_simple(self, smiles: str) -> List[RetrosynthesisRoute]:
        """
        Simple disconnection analysis (very basic fallback)

        Disconnects at strategic bonds (e.g., C-N, C-O in functional groups)
        """
        mol = Chem.MolFromSmiles(smiles)
        routes = []

        logger.info(f"Running simple disconnection for {smiles}")

        # Look for strategic bonds to break
        # Priority: amide > ester > ether > C-C

        disconnect_patterns = [
            ('[C:1](=O)[N:2]', 'amide'),      # Amide bond
            ('[C:1](=O)[O:2]', 'ester'),      # Ester bond
            ('[C:1][O:2][C:3]', 'ether'),     # Ether bond
            ('[C:1][C:2]', 'c-c'),            # C-C bond
        ]

        for pattern_smarts, bond_type in disconnect_patterns:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern is None:
                continue

            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                continue

            # Create route by disconnecting first match
            match = matches[0]
            if len(match) >= 2:
                # Simple disconnection (not chemically accurate, just conceptual)
                reactant1 = 'Fragment1'  # Placeholder
                reactant2 = 'Fragment2'  # Placeholder

                step = ReactionStep(
                    step_number=1,
                    reactants=[reactant1, reactant2],
                    product=smiles,
                    reaction_type=f'{bond_type}_disconnection',
                    confidence=0.3  # Low confidence for simple method
                )

                route = RetrosynthesisRoute(
                    target_smiles=smiles,
                    route_id=len(routes),
                    steps=[step],
                    building_blocks=[reactant1, reactant2],
                    overall_score=0.3,
                    method="simple"
                )

                routes.append(route)

                if len(routes) >= 3:  # Limit simple routes
                    break

        if not routes:
            # Create single-step "no disconnection" route
            logger.warning("No disconnection points found")
            route = RetrosynthesisRoute(
                target_smiles=smiles,
                route_id=0,
                steps=[],
                building_blocks=[smiles],  # Molecule is already a building block
                overall_score=0.0,
                method="simple"
            )
            routes.append(route)

        return routes

    def analyze_batch(self, smiles_list: List[str]) -> Dict[str, List[RetrosynthesisRoute]]:
        """
        Analyze multiple molecules

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dictionary mapping SMILES to list of routes
        """
        results = {}
        for smiles in smiles_list:
            try:
                routes = self.analyze(smiles)
                results[smiles] = routes
            except Exception as e:
                logger.error(f"Failed to analyze {smiles}: {e}")
                results[smiles] = []

        return results


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    if not RDKIT_AVAILABLE:
        print("RDKit not available. Please install: pip install rdkit")
        exit(1)

    # Test molecules
    test_molecules = [
        ('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin'),
        ('CC(C)Cc1ccc(cc1)C(C)C(O)=O', 'Ibuprofen'),
        ('CCO', 'Ethanol'),
    ]

    analyzer = RetrosynthesisAnalyzer(
        use_aizynthfinder=False,  # Disable for example (may not be installed)
        max_routes=3
    )

    print("\nRetrosynthesis Analysis Examples:")
    print("=" * 80)

    for smiles, name in test_molecules:
        print(f"\n{name}")
        print(f"Target: {smiles}")
        print("-" * 80)

        routes = analyzer.analyze(smiles, method="template")

        if routes:
            print(f"Found {len(routes)} route(s):\n")

            for route in routes:
                metrics = route.get_metrics()
                print(f"Route {route.route_id + 1}:")
                print(f"  Method: {route.method}")
                print(f"  Steps: {metrics['num_steps']}")
                print(f"  Building blocks: {metrics['num_building_blocks']}")
                print(f"  Score: {route.overall_score:.2f}")

                for step in route.steps:
                    print(f"\n  Step {step.step_number} ({step.reaction_type}):")
                    print(f"    Reactants: {', '.join(step.reactants)}")
                    print(f"    Product: {step.product}")
                    print(f"    Confidence: {step.confidence:.2f}")

                print(f"\n  Final building blocks:")
                for bb in route.building_blocks:
                    print(f"    - {bb}")
                print()
        else:
            print("No routes found")
