"""
Molecular Representation Exploration
=====================================

This notebook demonstrates the conversion of SMILES strings to molecular graphs
and visualizes the graph representations.

Run after completing dataset preprocessing (Step 1).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_and_preprocess_data
from src.molecular_graph import MolecularGraphConverter, visualize_molecular_graph
import pickle
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw


def explore_molecular_representation(data_path: str):
    """
    Explore molecular graph representation using the preprocessed dataset.
    
    Args:
        data_path: Path to the preprocessed CSV dataset
    """
    print("=" * 70)
    print("MOLECULAR REPRESENTATION EXPLORATION")
    print("=" * 70)
    
    # Load preprocessed data
    print("\n📂 Loading dataset...")
    df, stats = load_and_preprocess_data(data_path, threshold=6.0)
    
    # Initialize converter
    converter = MolecularGraphConverter(use_edge_features=True)
    
    # Get feature dimensions
    dims = converter.get_feature_dimensions()
    print(f"\n📊 Feature Dimensions:")
    print(f"   Node (atom) features: {dims['node_features']}")
    print(f"   Edge (bond) features: {dims['edge_features']}")
    
    # Convert dataset to graphs
    print(f"\n🔄 Converting {len(df)} molecules to graphs...")
    graph_list, label_list, invalid_smiles = converter.convert_dataset(
        df, 
        smiles_column='SMILES',
        label_column='label'
    )
    
    # Statistics
    print(f"\n📈 Conversion Statistics:")
    print(f"   Total molecules: {len(df)}")
    print(f"   Successfully converted: {len(graph_list)}")
    print(f"   Invalid SMILES: {len(invalid_smiles)}")
    print(f"   Success rate: {len(graph_list)/len(df)*100:.2f}%")
    
    # Analyze graph sizes
    num_nodes = [g.x.shape[0] for g in graph_list]
    num_edges = [g.edge_index.shape[1] for g in graph_list]
    
    print(f"\n📊 Graph Size Statistics:")
    print(f"   Avg nodes per graph: {sum(num_nodes)/len(num_nodes):.2f}")
    print(f"   Avg edges per graph: {sum(num_edges)/len(num_edges):.2f}")
    print(f"   Min nodes: {min(num_nodes)}, Max nodes: {max(num_nodes)}")
    print(f"   Min edges: {min(num_edges)}, Max edges: {max(num_edges)}")
    
    # Save processed graphs
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'molecular_graphs.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump({
            'graphs': graph_list,
            'labels': label_list,
            'feature_dims': dims,
            'invalid_smiles': invalid_smiles
        }, f)
    
    print(f"\n💾 Saved processed graphs to: {output_file}")
    
    # Visualize examples
    visualize_examples(df, graph_list)
    
    # Create distribution plots
    create_graph_distributions(num_nodes, num_edges)
    
    print("\n✅ Molecular representation exploration complete!")
    
    return graph_list, label_list, dims


def visualize_examples(df, graph_list, n_examples: int = 6):
    """
    Visualize example molecules (active and inactive).
    
    Args:
        df: DataFrame with SMILES and labels
        graph_list: List of graph objects
        n_examples: Number of examples to show (3 active, 3 inactive)
    """
    print(f"\n🖼️ Visualizing example molecules...")
    
    # Get active and inactive examples
    active_df = df[df['label'] == 1].head(3)
    inactive_df = df[df['label'] == 0].head(3)
    
    examples = []
    for _, row in active_df.iterrows():
        examples.append((row['SMILES'], 'Active', 1))
    for _, row in inactive_df.iterrows():
        examples.append((row['SMILES'], 'Inactive', 0))
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Example Molecules: Active vs Inactive Cancer Drugs', 
                 fontsize=16, fontweight='bold')
    
    for idx, (smiles, label_text, label) in enumerate(examples):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            img = Draw.MolToImage(mol, size=(300, 300))
            ax.imshow(img)
            
            # Color based on label
            color = '#2ecc71' if label == 1 else '#e74c3c'
            ax.set_title(f'{label_text}\n{smiles[:40]}...', 
                        fontweight='bold', color=color, fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center')
        
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'example_molecules.png'), 
                dpi=300, bbox_inches='tight')
    print(f"   Saved to: results/example_molecules.png")
    plt.show()


def create_graph_distributions(num_nodes, num_edges):
    """
    Create distribution plots for graph sizes.
    
    Args:
        num_nodes: List of node counts per graph
        num_edges: List of edge counts per graph
    """
    print(f"\n📊 Creating graph size distributions...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Molecular Graph Size Distributions', fontsize=16, fontweight='bold')
    
    # Nodes distribution
    axes[0].hist(num_nodes, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Number of Atoms (Nodes)', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('Atom Count Distribution', fontweight='bold')
    axes[0].axvline(x=sum(num_nodes)/len(num_nodes), color='red', 
                    linestyle='--', linewidth=2, label=f'Mean: {sum(num_nodes)/len(num_nodes):.1f}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Edges distribution
    axes[1].hist(num_edges, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Number of Bonds (Edges)', fontweight='bold')
    axes[1].set_ylabel('Frequency', fontweight='bold')
    axes[1].set_title('Bond Count Distribution', fontweight='bold')
    axes[1].axvline(x=sum(num_edges)/len(num_edges), color='blue', 
                    linestyle='--', linewidth=2, label=f'Mean: {sum(num_edges)/len(num_edges):.1f}')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'graph_distributions.png'), 
                dpi=300, bbox_inches='tight')
    print(f"   Saved to: results/graph_distributions.png")
    plt.show()


def create_feature_explanation_visual():
    """
    Create a visual explanation of molecular graph features.
    """
    print(f"\n🎨 Creating feature explanation visual...")
    
    # Example molecule: Aspirin
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    
    visualize_molecular_graph(
        smiles, 
        save_path='results/molecular_graph_explanation.png'
    )


if __name__ == "__main__":
    # Path to dataset
    data_path = "data/cancer_drugs.csv"
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found at {data_path}")
        print("\nPlease run:")
        print("  python create_sample_data.py")
        print("OR place your actual dataset in data/cancer_drugs.csv")
    else:
        # Run exploration
        graph_list, label_list, dims = explore_molecular_representation(data_path)
        
        # Create feature explanation
        create_feature_explanation_visual()
        
        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print("✅ Molecular graphs are saved in: data/processed/molecular_graphs.pkl")
        print("📊 Ready for Step 3: Building Baseline ANN Model")
        print("=" * 70)
