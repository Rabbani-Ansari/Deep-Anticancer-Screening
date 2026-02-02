"""
Molecular Graph Representation Module
======================================

This module converts SMILES strings into molecular graphs for Graph Neural Networks.
It extracts atom-level and bond-level features using RDKit and creates PyTorch 
Geometric Data objects.

Author: [Your Name]
Project: GNN-Based Drug Candidate Shortlisting for Cancer Treatment
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# PyTorch imports
import torch
from torch_geometric.data import Data


class MolecularGraphConverter:
    """
    Converts SMILES strings to molecular graphs for GNN input.
    
    Key Concepts:
    - Atoms become nodes in the graph
    - Bonds become edges in the graph
    - Node features: atom properties (atomic number, degree, etc.)
    - Edge features: bond properties (bond type, stereochemistry)
    """
    
    def __init__(self, use_edge_features: bool = True):
        """
        Initialize the molecular graph converter.
        
        Args:
            use_edge_features: Whether to include bond features as edge attributes
        """
        self.use_edge_features = use_edge_features
        
        # Define feature vocabularies for one-hot encoding
        self.atom_features_dim = 0  # Will be calculated
        self.edge_features_dim = 0  # Will be calculated
        
    @staticmethod
    def get_atom_features(atom) -> np.ndarray:
        """
        Extract features for a single atom.
        
        Features extracted:
        1. Atomic number (one-hot encoded for common elements)
        2. Degree (number of bonded neighbors)
        3. Formal charge
        4. Hybridization (sp, sp2, sp3, etc.)
        5. Aromaticity (is atom part of aromatic ring?)
        6. Number of hydrogens
        
        Args:
            atom: RDKit atom object
            
        Returns:
            numpy array of atom features
        """
        # 1. Atomic number (one-hot for common elements in organic chemistry)
        # Common elements: C, N, O, S, F, Cl, Br, I, P, and 'Other'
        atomic_num_list = [6, 7, 8, 16, 9, 17, 35, 53, 15]  # C, N, O, S, F, Cl, Br, I, P
        atomic_num = atom.GetAtomicNum()
        
        # One-hot encoding for atomic number
        atom_type = [0] * (len(atomic_num_list) + 1)  # +1 for 'Other'
        if atomic_num in atomic_num_list:
            atom_type[atomic_num_list.index(atomic_num)] = 1
        else:
            atom_type[-1] = 1  # Other
        
        # 2. Degree (number of bonded neighbors) - one-hot encoded (0-5, 6+)
        degree = atom.GetDegree()
        degree_onehot = [0] * 7
        if degree < 6:
            degree_onehot[degree] = 1
        else:
            degree_onehot[6] = 1  # 6 or more
        
        # 3. Formal charge (one-hot: -2, -1, 0, +1, +2, other)
        charge = atom.GetFormalCharge()
        charge_onehot = [0] * 6
        if charge in [-2, -1, 0, 1, 2]:
            charge_onehot[charge + 2] = 1  # Shift to 0-index
        else:
            charge_onehot[5] = 1  # Other
        
        # 4. Hybridization (SP, SP2, SP3, SP3D, SP3D2, Other)
        hybridization_list = [
            Chem.HybridizationType.SP,
            Chem.HybridizationType.SP2,
            Chem.HybridizationType.SP3,
            Chem.HybridizationType.SP3D,
            Chem.HybridizationType.SP3D2
        ]
        hybridization = atom.GetHybridization()
        hybrid_onehot = [0] * (len(hybridization_list) + 1)
        if hybridization in hybridization_list:
            hybrid_onehot[hybridization_list.index(hybridization)] = 1
        else:
            hybrid_onehot[-1] = 1  # Other
        
        # 5. Aromaticity (binary)
        is_aromatic = [1 if atom.GetIsAromatic() else 0]
        
        # 6. Number of hydrogen atoms (one-hot: 0-4, 5+)
        num_h = atom.GetTotalNumHs()
        h_onehot = [0] * 6
        if num_h < 5:
            h_onehot[num_h] = 1
        else:
            h_onehot[5] = 1  # 5 or more
        
        # Concatenate all features
        atom_features = np.array(
            atom_type + degree_onehot + charge_onehot + 
            hybrid_onehot + is_aromatic + h_onehot
        )
        
        return atom_features
    
    @staticmethod
    def get_bond_features(bond) -> np.ndarray:
        """
        Extract features for a single bond.
        
        Features extracted:
        1. Bond type (single, double, triple, aromatic)
        2. Conjugation (is bond conjugated?)
        3. Ring membership (is bond in a ring?)
        4. Stereochemistry
        
        Args:
            bond: RDKit bond object
            
        Returns:
            numpy array of bond features
        """
        # 1. Bond type (one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC)
        bond_type_list = [
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
            Chem.BondType.AROMATIC
        ]
        bond_type = bond.GetBondType()
        bond_type_onehot = [0] * (len(bond_type_list) + 1)
        if bond_type in bond_type_list:
            bond_type_onehot[bond_type_list.index(bond_type)] = 1
        else:
            bond_type_onehot[-1] = 1  # Other
        
        # 2. Conjugation (binary)
        is_conjugated = [1 if bond.GetIsConjugated() else 0]
        
        # 3. Ring membership (binary)
        in_ring = [1 if bond.IsInRing() else 0]
        
        # 4. Stereochemistry (one-hot: STEREONONE, STEREOANY, STEREOZ, STEREOE, STEREOCIS, STEREOTRANS)
        stereo_list = [
            Chem.BondStereo.STEREONONE,
            Chem.BondStereo.STEREOANY,
            Chem.BondStereo.STEREOZ,
            Chem.BondStereo.STEREOE,
            Chem.BondStereo.STEREOCIS,
            Chem.BondStereo.STEREOTRANS
        ]
        stereo = bond.GetStereo()
        stereo_onehot = [0] * len(stereo_list)
        if stereo in stereo_list:
            stereo_onehot[stereo_list.index(stereo)] = 1
        
        # Concatenate all features
        bond_features = np.array(
            bond_type_onehot + is_conjugated + in_ring + stereo_onehot
        )
        
        return bond_features
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """
        Convert a SMILES string to a PyTorch Geometric Data object.
        
        Args:
            smiles: SMILES string representation of molecule
            
        Returns:
            PyTorch Geometric Data object with:
            - x: Node feature matrix [num_nodes, num_node_features]
            - edge_index: Edge indices [2, num_edges]
            - edge_attr: Edge feature matrix [num_edges, num_edge_features] (optional)
            Returns None if SMILES is invalid
        """
        # Parse SMILES string
        mol = Chem.MolFromSmiles(smiles)
        
        # Handle invalid SMILES
        if mol is None:
            return None
        
        # Add explicit hydrogens (optional, but can be useful)
        # mol = Chem.AddHs(mol)  # Uncomment if you want explicit H atoms
        
        # === Extract Node Features ===
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(self.get_atom_features(atom))
        
        # Convert to tensor: [num_nodes, num_node_features]
        x = torch.tensor(np.array(atom_features_list), dtype=torch.float)
        
        # === Extract Edge Information ===
        edge_indices = []
        edge_features_list = []
        
        for bond in mol.GetBonds():
            # Get atom indices
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions (undirected graph)
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            
            # Extract bond features
            if self.use_edge_features:
                bond_feat = self.get_bond_features(bond)
                edge_features_list.append(bond_feat)
                edge_features_list.append(bond_feat)  # Same features for both directions
        
        # Convert to tensor: [2, num_edges]
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            # Molecule with no bonds (single atom)
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Edge features: [num_edges, num_edge_features]
        if self.use_edge_features and len(edge_features_list) > 0:
            edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.float)
        else:
            edge_attr = None
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data
    
    def convert_dataset(
        self, 
        df: pd.DataFrame, 
        smiles_column: str = 'SMILES',
        label_column: str = 'label'
    ) -> Tuple[List[Data], List[int], List[str]]:
        """
        Convert entire dataset of SMILES to graph objects.
        
        Args:
            df: DataFrame containing SMILES and labels
            smiles_column: Name of column containing SMILES strings
            label_column: Name of column containing labels
            
        Returns:
            Tuple of:
            - List of PyTorch Geometric Data objects
            - List of labels
            - List of invalid SMILES that were skipped
        """
        graph_list = []
        label_list = []
        invalid_smiles = []
        
        print(f"Converting {len(df)} SMILES to molecular graphs...")
        
        for idx, row in df.iterrows():
            smiles = row[smiles_column]
            label = row[label_column]
            
            # Convert to graph
            graph = self.smiles_to_graph(smiles)
            
            if graph is not None:
                # Add label to graph object
                graph.y = torch.tensor([label], dtype=torch.long)
                graph_list.append(graph)
                label_list.append(label)
            else:
                invalid_smiles.append(smiles)
        
        print(f"✅ Successfully converted: {len(graph_list)} molecules")
        print(f"❌ Invalid/Skipped SMILES: {len(invalid_smiles)}")
        
        if len(invalid_smiles) > 0:
            print(f"\nFirst few invalid SMILES:")
            for smiles in invalid_smiles[:5]:
                print(f"  - {smiles}")
        
        return graph_list, label_list, invalid_smiles
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get the dimensions of node and edge features.
        
        Returns:
            Dictionary with 'node_features' and 'edge_features' dimensions
        """
        # Create a simple molecule to get feature dimensions
        mol = Chem.MolFromSmiles('CCO')  # Ethanol
        
        # Get atom feature dimension
        atom = mol.GetAtomWithIdx(0)
        atom_features = self.get_atom_features(atom)
        node_dim = len(atom_features)
        
        # Get bond feature dimension
        if self.use_edge_features and mol.GetNumBonds() > 0:
            bond = mol.GetBondWithIdx(0)
            bond_features = self.get_bond_features(bond)
            edge_dim = len(bond_features)
        else:
            edge_dim = 0
        
        return {
            'node_features': node_dim,
            'edge_features': edge_dim
        }


def visualize_molecular_graph(smiles: str, save_path: Optional[str] = None):
    """
    Visualize a molecule and its graph representation.
    
    Args:
        smiles: SMILES string
        save_path: Optional path to save the visualization
    """
    import matplotlib.pyplot as plt
    from rdkit.Chem import Draw
    
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Molecule structure
    img = Draw.MolToImage(mol, size=(400, 400))
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title(f'Molecular Structure\n{smiles}', fontweight='bold', fontsize=12)
    
    # 2. Graph representation info
    converter = MolecularGraphConverter()
    graph = converter.smiles_to_graph(smiles)
    
    info_text = f"""
    Graph Representation:
    
    Nodes (Atoms): {graph.x.shape[0]}
    Edges (Bonds): {graph.edge_index.shape[1]}
    
    Node Features: {graph.x.shape[1]}
    Edge Features: {graph.edge_attr.shape[1] if graph.edge_attr is not None else 0}
    
    Node Feature Vector Example (first atom):
    Shape: {graph.x[0].shape}
    
    Breakdown:
    - Atomic type (one-hot): 10 features
    - Degree (one-hot): 7 features  
    - Formal charge (one-hot): 6 features
    - Hybridization (one-hot): 6 features
    - Aromaticity: 1 feature
    - Num Hydrogens (one-hot): 6 features
    
    Total: {graph.x.shape[1]} features per atom
    """
    
    axes[1].text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                 verticalalignment='center', wrap=True)
    axes[1].axis('off')
    axes[1].set_title('Graph Statistics', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("MOLECULAR GRAPH CONVERTER - DEMONSTRATION")
    print("=" * 70)
    
    # Example SMILES strings
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene
    ]
    
    # Initialize converter
    converter = MolecularGraphConverter(use_edge_features=True)
    
    # Get feature dimensions
    dims = converter.get_feature_dimensions()
    print(f"\n📊 Feature Dimensions:")
    print(f"   Node features: {dims['node_features']}")
    print(f"   Edge features: {dims['edge_features']}\n")
    
    # Convert each SMILES
    for smiles in test_smiles:
        print(f"\n🧪 Converting: {smiles}")
        graph = converter.smiles_to_graph(smiles)
        
        if graph is not None:
            print(f"   ✅ Success!")
            print(f"   Nodes: {graph.x.shape[0]}, Edges: {graph.edge_index.shape[1]}")
            print(f"   Node features shape: {graph.x.shape}")
            if graph.edge_attr is not None:
                print(f"   Edge features shape: {graph.edge_attr.shape}")
        else:
            print(f"   ❌ Invalid SMILES")
    
    print("\n" + "=" * 70)
    print("To use with dataset:")
    print("graph_list, labels, invalid = converter.convert_dataset(df)")
    print("=" * 70)
