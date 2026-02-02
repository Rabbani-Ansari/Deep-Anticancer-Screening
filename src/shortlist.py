"""
Drug Shortlisting Mechanism
============================

This module implements the drug candidate shortlisting and ranking system
using the trained GNN model to predict cancer drug effectiveness.

Author: [Your Name]
Project: GNN-Based Drug Candidate Shortlisting for Cancer Treatment
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from src.gnn_model import MolecularGNN
from src.molecular_graph import MolecularGraphConverter


class DrugShortlister:
    """
    Drug candidate shortlisting system using trained GNN model.
    
    This system:
    1. Accepts unseen drug molecules (SMILES)
    2. Converts them to molecular graphs
    3. Predicts anticancer effectiveness using trained GNN
    4. Ranks candidates by predicted probability
    5. Returns top-K most promising candidates
    """
    
    def __init__(
        self, 
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the drug shortlisting system.
        
        Args:
            model_path: Path to trained GNN model (.pth file)
            device: Device to run inference on
        """
        self.device = device
        
        # Load trained model
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint['model_config']
        
        # Initialize model architecture
        self.model = MolecularGNN(
            node_feature_dim=model_config['node_feature_dim'],
            edge_feature_dim=model_config['edge_feature_dim'],
            hidden_dims=model_config['hidden_dims'],
            fc_dims=model_config['fc_dims'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout'],
            pooling=model_config['pooling']
        ).to(device)
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize graph converter
        self.graph_converter = MolecularGraphConverter(use_edge_features=True)
        
        print(f"✅ Loaded trained GNN model from: {model_path}")
        print(f"   Running on: {device}")
    
    def predict_single(self, smiles: str) -> Optional[Tuple[float, int]]:
        """
        Predict cancer drug effectiveness for a single molecule.
        
        Args:
            smiles: SMILES string of the drug molecule
            
        Returns:
            Tuple of (probability_active, predicted_class)
            Returns None if SMILES is invalid
        """
        # Convert SMILES to graph
        graph = self.graph_converter.smiles_to_graph(smiles)
        
        if graph is None:
            return None
        
        # Create batch
        batch = Batch.from_data_list([graph]).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(batch)
            probs = F.softmax(output, dim=1)
            prob_active = probs[0, 1].item()  # Probability of being active
            pred_class = output.argmax(dim=1).item()
        
        return prob_active, pred_class
    
    def predict_batch(
        self, 
        smiles_list: List[str]
    ) -> Tuple[List[float], List[int], List[str]]:
        """
        Predict for multiple molecules.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Tuple of:
            - List of probabilities (active class)
            - List of predicted classes (0 or 1)
            - List of invalid SMILES
        """
        probabilities = []
        predictions = []
        invalid_smiles = []
        
        # Convert all SMILES to graphs
        graphs = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            graph = self.graph_converter.smiles_to_graph(smiles)
            if graph is not None:
                graphs.append(graph)
                valid_indices.append(i)
            else:
                invalid_smiles.append(smiles)
        
        if len(graphs) == 0:
            return [], [], invalid_smiles
        
        # Batch prediction
        batch = Batch.from_data_list(graphs).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch)
            probs = F.softmax(outputs, dim=1)
            prob_active = probs[:, 1].cpu().numpy()
            pred_classes = outputs.argmax(dim=1).cpu().numpy()
        
        probabilities = prob_active.tolist()
        predictions = pred_classes.tolist()
        
        return probabilities, predictions, invalid_smiles
    
    def shortlist_drugs(
        self,
        smiles_list: List[str],
        drug_names: Optional[List[str]] = None,
        top_k: int = 10,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Shortlist and rank drug candidates.
        
        Args:
            smiles_list: List of SMILES strings to evaluate
            drug_names: Optional list of drug names (for display)
            top_k: Number of top candidates to return
            threshold: Probability threshold for considering a drug "active"
            
        Returns:
            DataFrame with ranked drug candidates
        """
        print(f"\n{'='*70}")
        print(f"DRUG CANDIDATE SHORTLISTING")
        print(f"{'='*70}")
        print(f"Evaluating {len(smiles_list)} drug molecules...")
        
        # If no names provided, use indices
        if drug_names is None:
            drug_names = [f"Drug_{i+1}" for i in range(len(smiles_list))]
        
        # Predict for all molecules
        probabilities, predictions, invalid = self.predict_batch(smiles_list)
        
        # Create results dataframe
        results = []
        for i, (smiles, name) in enumerate(zip(smiles_list, drug_names)):
            if smiles not in invalid:
                idx = smiles_list.index(smiles) - len([s for s in invalid if smiles_list.index(s) < smiles_list.index(smiles)])
                if idx < len(probabilities):
                    results.append({
                        'Rank': 0,  # Will be filled later
                        'Drug_Name': name,
                        'SMILES': smiles,
                        'Predicted_Probability': probabilities[idx],
                        'Predicted_Class': 'Active' if predictions[idx] == 1 else 'Inactive',
                        'Confidence': 'High' if abs(probabilities[idx] - 0.5) > 0.3 else 'Medium' if abs(probabilities[idx] - 0.5) > 0.15 else 'Low'
                    })
        
        # Create dataframe and sort by probability
        df = pd.DataFrame(results)
        df = df.sort_values('Predicted_Probability', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
        
        # Filter by threshold and top_k
        df_shortlisted = df[df['Predicted_Probability'] >= threshold].head(top_k)
        
        print(f"\n✅ Successfully evaluated: {len(df)} molecules")
        print(f"❌ Invalid SMILES: {len(invalid)}")
        print(f"📊 Active predictions (prob >= {threshold}): {len(df[df['Predicted_Probability'] >= threshold])}")
        print(f"🎯 Top-{top_k} shortlisted candidates: {len(df_shortlisted)}")
        
        print(f"\n{'='*70}")
        print(f"TOP {min(top_k, len(df_shortlisted))} SHORTLISTED DRUG CANDIDATES")
        print(f"{'='*70}")
        
        # Display top candidates
        for _, row in df_shortlisted.iterrows():
            print(f"\n🏆 Rank {row['Rank']}: {row['Drug_Name']}")
            print(f"   Probability: {row['Predicted_Probability']:.4f} ({row['Predicted_Probability']*100:.2f}%)")
            print(f"   Prediction:  {row['Predicted_Class']}")
            print(f"   Confidence:  {row['Confidence']}")
            print(f"   SMILES:      {row['SMILES'][:60]}{'...' if len(row['SMILES']) > 60 else ''}")
        
        print(f"\n{'='*70}\n")
        
        return df_shortlisted
    
    def save_shortlist(
        self, 
        df: pd.DataFrame, 
        output_path: str,
        format: str = 'csv'
    ):
        """
        Save shortlisted candidates to file.
        
        Args:
            df: DataFrame with shortlisted drugs
            output_path: Path to save file
            format: Output format ('csv', 'excel', 'json')
        """
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'excel':
            df.to_excel(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        
        print(f"💾 Shortlisted drugs saved to: {output_path}")


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("DRUG SHORTLISTING SYSTEM - DEMONSTRATION")
    print("=" * 70)
    
    # Example: Test molecules (mix of known active and inactive)
    test_molecules = {
        # Known anticancer drugs (should have high probability)
        'Doxorubicin': 'CC1(C2C(C3C(C(O2)OC4C(CC(CC4O)N)O)OC5CC(C(C(O5)C)O)(C)O)C(=O)C6=C(C=CC=C6C(=O)C1O)O)O',
        'Cisplatin': '[Pt](N)(N)(Cl)Cl',
        'Methotrexate': 'CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O',
        
        # Common drugs (likely inactive against cancer)
        'Aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
        'Caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'Ibuprofen': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
        
        # Test molecules
        'Ethanol': 'CCO',
        'Benzene': 'c1ccccc1',
    }
    
    smiles_list = list(test_molecules.values())
    drug_names = list(test_molecules.keys())
    
    print("\n📋 Test molecules:")
    for name in drug_names:
        print(f"   • {name}")
    
    # Check if model exists
    model_path = "results/models/gnn_model.pth"
    
    print(f"\n🔍 Looking for trained model at: {model_path}")
    
    import os
    if os.path.exists(model_path):
        # Initialize shortlister
        shortlister = DrugShortlister(model_path)
        
        # Shortlist drugs
        results = shortlister.shortlist_drugs(
            smiles_list=smiles_list,
            drug_names=drug_names,
            top_k=5,
            threshold=0.5
        )
        
        # Save results
        output_dir = 'results/shortlisting'
        os.makedirs(output_dir, exist_ok=True)
        shortlister.save_shortlist(
            results, 
            os.path.join(output_dir, 'top_candidates.csv')
        )
    else:
        print("❌ Model not found. Please train the GNN model first:")
        print("   python notebooks/04_gnn_training.py")
