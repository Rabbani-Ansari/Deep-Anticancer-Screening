"""
Graph Neural Network (GNN) Model
=================================

This module implements the proposed Graph Neural Network for cancer drug
classification. The GNN operates directly on molecular graphs and learns
representations through message passing between atoms.

This is the MAIN CONTRIBUTION of the project and should outperform the
baseline ANN model.

Author: [Your Name]
Project: GNN-Based Drug Candidate Shortlisting for Cancer Treatment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Optional
import numpy as np


class MolecularGNN(nn.Module):
    """
    Graph Neural Network for molecular property prediction.
    
    Architecture:
    1. Graph Convolution Layers (GCNConv) - Learn node representations
    2. Global Pooling - Aggregate node features to graph-level
    3. Fully Connected Layers - Final classification
    
    Key Concepts:
    - Message Passing: Atoms exchange information with bonded neighbors
    - Node Embeddings: Each atom gets a learned representation
    - Graph Pooling: Combine all atom features into molecule representation
    """
    
    def __init__(
        self,
        node_feature_dim: int = 36,
        edge_feature_dim: int = 13,
        hidden_dims: List[int] = [128, 128, 64],
        fc_dims: List[int] = [128, 64],
        num_classes: int = 2,
        dropout: float = 0.3,
        pooling: str = 'mean'
    ):
        """
        Initialize the GNN model.
        
        Args:
            node_feature_dim: Dimension of input node (atom) features
            edge_feature_dim: Dimension of edge (bond) features
            hidden_dims: Hidden dimensions for GCN layers
            fc_dims: Dimensions for fully connected layers after pooling
            num_classes: Number of output classes
            dropout: Dropout probability
            pooling: Graph pooling method ('mean', 'max', or 'mean+max')
        """
        super(MolecularGNN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dims = hidden_dims
        self.fc_dims = fc_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.pooling = pooling
        
        # ============================================================
        # Graph Convolution Layers
        # ============================================================
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First GCN layer: node_features -> hidden_dim
        prev_dim = node_feature_dim
        for hidden_dim in hidden_dims:
            self.conv_layers.append(GCNConv(prev_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # ============================================================
        # Pooling
        # ============================================================
        # Determine pooled feature dimension
        if pooling == 'mean+max':
            pooled_dim = hidden_dims[-1] * 2  # Concatenate mean and max
        else:
            pooled_dim = hidden_dims[-1]
        
        # ============================================================
        # Fully Connected Layers (after pooling)
        # ============================================================
        self.fc_layers = nn.ModuleList()
        prev_dim = pooled_dim
        
        for fc_dim in fc_dims:
            self.fc_layers.append(nn.Linear(prev_dim, fc_dim))
            prev_dim = fc_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, num_classes)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            data: PyTorch Geometric Batch object containing:
                - x: Node features [num_nodes, node_feature_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - batch: Batch assignment [num_nodes]
                
        Returns:
            Logits for each class [batch_size, num_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # ============================================================
        # Message Passing: Graph Convolution Layers
        # ============================================================
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            # Graph convolution
            x = conv(x, edge_index)
            
            # Batch normalization
            x = bn(x)
            
            # ReLU activation
            x = F.relu(x)
            
            # Dropout (except last layer)
            if i < len(self.conv_layers) - 1:
                x = self.dropout_layer(x)
        
        # At this point: x has shape [num_nodes, hidden_dims[-1]]
        # Each node (atom) has a learned embedding
        
        # ============================================================
        # Graph-Level Pooling
        # ============================================================
        # Aggregate node features to get graph-level representation
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'mean+max':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        
        # Now: x has shape [batch_size, pooled_dim]
        # Each graph (molecule) has a single representation
        
        # ============================================================
        # Fully Connected Layers
        # ============================================================
        for fc in self.fc_layers:
            x = fc(x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # ============================================================
        # Output Layer
        # ============================================================
        x = self.output_layer(x)
        
        # Return raw logits (CrossEntropyLoss expects logits, not probabilities)
        return x
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_graph_embedding(self, data: Batch) -> torch.Tensor:
        """
        Get graph-level embeddings before final classification.
        Useful for visualization and analysis.
        
        Args:
            data: PyTorch Geometric Batch object
            
        Returns:
            Graph embeddings [batch_size, embedding_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Pass through GCN layers
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < len(self.conv_layers) - 1:
                x = self.dropout_layer(x)
        
        # Pool to graph level
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'mean+max':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        
        # Pass through FC layers (but not output layer)
        for fc in self.fc_layers:
            x = fc(x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        return x


class GNNTrainer:
    """
    Trainer class for the GNN model.
    Similar to ANNTrainer but works with graph data.
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
            model: GNN model to train
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training graphs
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch.y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch.y).sum().item()
            total += batch.y.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, val_loader) -> Tuple[float, float]:
        """
        Evaluate on validation set.
        
        Args:
            val_loader: DataLoader for validation graphs
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in val_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch.y)
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch.y).sum().item()
            total += batch.y.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        verbose: bool = True
    ):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs
            verbose: Print progress
        """
        if verbose:
            print("=" * 70)
            print(f"Training Proposed GNN on {self.device}")
            print(f"Model parameters: {self.model.get_num_params():,}")
            print("=" * 70)
        
        best_val_acc = 0
        best_epoch = 0
        patience = 20
        patience_counter = 0
        
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
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if verbose and epoch % 5 == 0:
                print(f"Epoch [{epoch:3d}/{epochs}] | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if patience_counter >= patience and epoch > 30:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                break
        
        if verbose:
            print("=" * 70)
            print(f"✅ Training complete!")
            print(f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
            print("=" * 70)
    
    @torch.no_grad()
    def predict_proba(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class probabilities.
        
        Args:
            data_loader: DataLoader containing graphs
            
        Returns:
            Tuple of (probabilities, predictions)
        """
        self.model.eval()
        all_probs = []
        all_preds = []
        
        for batch in data_loader:
            batch = batch.to(self.device)
            outputs = self.model(batch)
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of active class
            all_preds.extend(preds.cpu().numpy())
        
        return np.array(all_probs), np.array(all_preds)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("GRAPH NEURAL NETWORK MODEL - DEMONSTRATION")
    print("=" * 70)
    
    # Create dummy model
    print("\n🧠 Creating GNN model:")
    model = MolecularGNN(
        node_feature_dim=36,
        edge_feature_dim=13,
        hidden_dims=[128, 128, 64],
        fc_dims=[128, 64],
        num_classes=2,
        dropout=0.3,
        pooling='mean'
    )
    
    print(f"\nArchitecture:")
    print(f"  Graph Convolution Layers: {model.hidden_dims}")
    print(f"  Pooling: {model.pooling}")
    print(f"  FC Layers: {model.fc_dims}")
    print(f"  Output Classes: {model.num_classes}")
    print(f"  Total parameters: {model.get_num_params():,}")
    
    # Test with dummy data
    print("\n🧪 Testing with dummy molecular graph:")
    from torch_geometric.data import Data
    
    # Create a simple molecule (e.g., 5 atoms)
    x = torch.randn(5, 36)  # 5 atoms, 36 features each
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],  # Source nodes
        [1, 0, 2, 1, 3, 2, 4, 3]   # Target nodes
    ], dtype=torch.long)
    
    dummy_graph = Data(x=x, edge_index=edge_index)
    dummy_batch = Batch.from_data_list([dummy_graph])
    
    # Forward pass
    output = model(dummy_batch)
    print(f"  Input: 5 atoms with 36 features each")
    print(f"  Output logits: {output.shape} (batch_size=1, num_classes=2)")
    print(f"  Predicted probabilities: {F.softmax(output, dim=1)[0]}")
    
    print("\n" + "=" * 70)
    print("To train on actual data:")
    print("  python notebooks/04_gnn_training.py")
    print("=" * 70)
