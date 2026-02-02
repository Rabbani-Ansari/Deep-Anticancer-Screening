"""
Baseline ANN Model Module
==========================

This module implements a traditional Artificial Neural Network (ANN) baseline
using molecular fingerprints. This serves as a comparison baseline for the
Graph Neural Network (GNN) model.

The ANN uses Morgan (ECFP) fingerprints which are fixed-length bit vectors
that encode molecular substructures.

Author: [Your Name]
Project: GNN-Based Drug Candidate Shortlisting for Cancer Treatment
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MolecularFingerprintGenerator:
    """
    Generates molecular fingerprints from SMILES strings.
    
    Morgan Fingerprints (also known as ECFP - Extended Connectivity Fingerprints):
    - Fixed-length bit vectors
    - Encode presence/absence of molecular substructures
    - Radius parameter controls substructure size
    """
    
    def __init__(self, radius: int = 2, n_bits: int = 2048):
        """
        Initialize fingerprint generator.
        
        Args:
            radius: Morgan fingerprint radius (typically 2 or 3)
                   - radius=2 is equivalent to ECFP4
                   - radius=3 is equivalent to ECFP6
            n_bits: Length of fingerprint bit vector (typically 1024, 2048, or 4096)
        """
        self.radius = radius
        self.n_bits = n_bits
    
    def smiles_to_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """
        Convert a SMILES string to a Morgan fingerprint.
        
        Args:
            smiles: SMILES string representation of molecule
            
        Returns:
            numpy array of fingerprint bits (1s and 0s)
            Returns None if SMILES is invalid
        """
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return None
        
        # Generate Morgan fingerprint
        # useFeatures=False uses atom connectivity (standard ECFP)
        # useFeatures=True uses atom features (FCFP - Functional Class Fingerprints)
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 
            radius=self.radius, 
            nBits=self.n_bits,
            useFeatures=False
        )
        
        # Convert to numpy array
        fp_array = np.zeros((self.n_bits,), dtype=np.float32)
        for i in range(self.n_bits):
            fp_array[i] = fp[i]
        
        return fp_array
    
    def convert_dataset(
        self, 
        df: pd.DataFrame, 
        smiles_column: str = 'SMILES'
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert entire dataset to fingerprints.
        
        Args:
            df: DataFrame containing SMILES strings
            smiles_column: Name of column with SMILES
            
        Returns:
            Tuple of:
            - numpy array of fingerprints [n_samples, n_bits]
            - List of invalid SMILES that were skipped
        """
        fingerprints = []
        invalid_smiles = []
        
        print(f"Generating Morgan fingerprints (radius={self.radius}, bits={self.n_bits})...")
        
        for idx, row in df.iterrows():
            smiles = row[smiles_column]
            fp = self.smiles_to_fingerprint(smiles)
            
            if fp is not None:
                fingerprints.append(fp)
            else:
                invalid_smiles.append(smiles)
        
        fingerprints_array = np.array(fingerprints, dtype=np.float32)
        
        print(f"✅ Generated fingerprints: {len(fingerprints)}")
        print(f"❌ Invalid SMILES: {len(invalid_smiles)}")
        print(f"Fingerprint shape: {fingerprints_array.shape}")
        
        return fingerprints_array, invalid_smiles


class FingerprintDataset(Dataset):
    """
    PyTorch Dataset for molecular fingerprints.
    """
    
    def __init__(self, fingerprints: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            fingerprints: Array of fingerprints [n_samples, n_bits]
            labels: Array of labels [n_samples]
        """
        self.fingerprints = torch.FloatTensor(fingerprints)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fingerprints[idx], self.labels[idx]


class BaselineANN(nn.Module):
    """
    Baseline Artificial Neural Network for molecular classification.
    
    Architecture:
    - Input: Molecular fingerprint (2048 bits)
    - Hidden layers: 2-3 fully connected layers with ReLU activation
    - Dropout: For regularization
    - Output: 2 classes (active/inactive)
    
    This is a simple feedforward neural network that treats fingerprints
    as fixed-length feature vectors.
    """
    
    def __init__(
        self, 
        input_dim: int = 2048,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        """
        Initialize the ANN model.
        
        Args:
            input_dim: Dimension of input fingerprint
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability for regularization
            num_classes: Number of output classes (2 for binary classification)
        """
        super(BaselineANN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.num_classes = num_classes
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input fingerprint tensor [batch_size, input_dim]
            
        Returns:
            Logits for each class [batch_size, num_classes]
        """
        return self.model(x)
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ANNTrainer:
    """
    Trainer class for the baseline ANN model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer.
        
        Args:
            model: ANN model to train
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization parameter
        """
        self.model = model.to(device)
        self.device = device
        
        # Optimizer (Adam is standard for neural networks)
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function (CrossEntropyLoss for classification)
        # Note: Includes softmax, so model outputs raw logits
        self.criterion = nn.CrossEntropyLoss()
        
        # Track training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for fingerprints, labels in train_loader:
            fingerprints = fingerprints.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(fingerprints)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate on validation set.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for fingerprints, labels in val_loader:
            fingerprints = fingerprints.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(fingerprints)
            loss = self.criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        epochs: int = 50,
        verbose: bool = True
    ):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            verbose: Whether to print progress
        """
        if verbose:
            print("=" * 70)
            print(f"Training Baseline ANN on {self.device}")
            print(f"Model parameters: {self.model.get_num_params():,}")
            print("=" * 70)
        
        best_val_acc = 0
        best_epoch = 0
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
            
            # Print progress
            if verbose and epoch % 5 == 0:
                print(f"Epoch [{epoch:3d}/{epochs}] | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if verbose:
            print("=" * 70)
            print(f"✅ Training complete!")
            print(f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
            print("=" * 70)
    
    @torch.no_grad()
    def predict_proba(self, fingerprints: torch.Tensor) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            fingerprints: Input fingerprints [n_samples, input_dim]
            
        Returns:
            Class probabilities [n_samples, num_classes]
        """
        self.model.eval()
        fingerprints = fingerprints.to(self.device)
        outputs = self.model(fingerprints)
        probs = F.softmax(outputs, dim=1)
        return probs.cpu().numpy()


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("BASELINE ANN MODEL - DEMONSTRATION")
    print("=" * 70)
    
    # Example: Generate fingerprints for test molecules
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CCO",  # Ethanol
    ]
    
    # Initialize fingerprint generator
    fp_gen = MolecularFingerprintGenerator(radius=2, n_bits=2048)
    
    print("\n🔍 Generating fingerprints for test molecules:")
    for smiles in test_smiles:
        fp = fp_gen.smiles_to_fingerprint(smiles)
        if fp is not None:
            n_bits_set = np.sum(fp > 0)
            print(f"  {smiles[:30]:30s} -> {n_bits_set:4d} bits set (out of 2048)")
    
    # Create dummy model
    print("\n🧠 Creating baseline ANN model:")
    model = BaselineANN(
        input_dim=2048,
        hidden_dims=[512, 256, 128],
        dropout=0.3,
        num_classes=2
    )
    
    print(f"  Architecture: {model.input_dim} -> {' -> '.join(map(str, model.hidden_dims))} -> {model.num_classes}")
    print(f"  Total parameters: {model.get_num_params():,}")
    
    print("\n" + "=" * 70)
    print("To train on actual data:")
    print("  python notebooks/03_baseline_training.py")
    print("=" * 70)
