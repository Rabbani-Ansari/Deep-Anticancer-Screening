"""
GNN Model Training Script
==========================

This notebook trains the proposed Graph Neural Network model using
molecular graphs and compares its performance against the baseline ANN.

This is the MAIN CONTRIBUTION of the project.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.gnn_model import MolecularGNN, GNNTrainer
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


def train_gnn_model(graph_data_path: str, save_dir: str = 'results/models'):
    """
    Train and evaluate the proposed GNN model.
    
    Args:
        graph_data_path: Path to processed molecular graphs pickle file
        save_dir: Directory to save trained model
    """
    print("=" * 70)
    print("PROPOSED GNN MODEL TRAINING")
    print("=" * 70)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # ============================================================
    # STEP 1: Load molecular graphs
    # ============================================================
    print("\n📂 STEP 1: Loading molecular graphs...")
    
    with open(graph_data_path, 'rb') as f:
        graph_data = pickle.load(f)
    
    graphs = graph_data['graphs']
    labels = graph_data['labels']
    feature_dims = graph_data['feature_dims']
    
    print(f"  Loaded {len(graphs)} molecular graphs")
    print(f"  Node feature dimension: {feature_dims['node_features']}")
    print(f"  Edge feature dimension: {feature_dims['edge_features']}")
    
    # Graph statistics
    num_nodes = [g.x.shape[0] for g in graphs]
    num_edges = [g.edge_index.shape[1] for g in graphs]
    
    print(f"\n  Graph statistics:")
    print(f"    Avg nodes: {np.mean(num_nodes):.1f} (min: {min(num_nodes)}, max: {max(num_nodes)})")
    print(f"    Avg edges: {np.mean(num_edges):.1f} (min: {min(num_edges)}, max: {max(num_edges)})")
    
    # ============================================================
    # STEP 2: Train-validation-test split
    # ============================================================
    print("\n📊 STEP 2: Splitting data...")
    
    # Create indices
    indices = list(range(len(graphs)))
    labels_array = np.array(labels)
    
    # First split: train+val vs test (80-20)
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=labels_array
    )
    
    # Second split: train vs val (80-20)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.2,
        random_state=42,
        stratify=labels_array[train_val_idx]
    )
    
    # Create datasets
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]
    
    train_labels = labels_array[train_idx]
    val_labels = labels_array[val_idx]
    test_labels = labels_array[test_idx]
    
    print(f"  Training set:   {len(train_graphs):4d} graphs ({(train_labels==1).sum()} active, {(train_labels==0).sum()} inactive)")
    print(f"  Validation set: {len(val_graphs):4d} graphs ({(val_labels==1).sum()} active, {(val_labels==0).sum()} inactive)")
    print(f"  Test set:       {len(test_graphs):4d} graphs ({(test_labels==1).sum()} active, {(test_labels==0).sum()} inactive)")
    
    # ============================================================
    # STEP 3: Create DataLoaders
    # ============================================================
    print("\n🔄 STEP 3: Creating graph dataloaders...")
    
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    # ============================================================
    # STEP 4: Initialize GNN model
    # ============================================================
    print("\n🧠 STEP 4: Initializing GNN model...")
    
    model = MolecularGNN(
        node_feature_dim=feature_dims['node_features'],
        edge_feature_dim=feature_dims['edge_features'],
        hidden_dims=[128, 128, 64],
        fc_dims=[128, 64],
        num_classes=2,
        dropout=0.3,
        pooling='mean'
    )
    
    print(f"  Model architecture:")
    print(f"    Node input:  {model.node_feature_dim}")
    print(f"    GCN layers:  {' -> '.join(map(str, model.hidden_dims))}")
    print(f"    Pooling:     {model.pooling}")
    print(f"    FC layers:   {' -> '.join(map(str, model.fc_dims))}")
    print(f"    Output:      {model.num_classes} classes")
    print(f"  Total parameters: {model.get_num_params():,}")
    
    trainer = GNNTrainer(
        model=model,
        learning_rate=0.001,
        weight_decay=1e-5
    )
    
    # ============================================================
    # STEP 5: Train the GNN
    # ============================================================
    print("\n🚀 STEP 5: Training GNN model...\n")
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        verbose=True
    )
    
    # ============================================================
    # STEP 6: Evaluate on test set
    # ============================================================
    print("\n📈 STEP 6: Evaluating on test set...")
    
    # Get predictions
    test_probs, test_preds = trainer.predict_proba(test_loader)
    
    # Calculate metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    test_auc = roc_auc_score(test_labels, test_probs)
    
    print("\n" + "=" * 70)
    print("PROPOSED GNN - TEST SET RESULTS")
    print("=" * 70)
    print(f"Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"AUC-ROC:   {test_auc:.4f}")
    print("=" * 70)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"               Inactive  Active")
    print(f"Actual Inactive   {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Active     {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds,
                                target_names=['Inactive', 'Active']))
    
    # ============================================================
    # STEP 7: Load baseline results for comparison
    # ============================================================
    print("\n📊 STEP 7: Comparing with baseline ANN...")
    
    baseline_path = os.path.join(save_dir, 'baseline_ann.pth')
    if os.path.exists(baseline_path):
        baseline_checkpoint = torch.load(baseline_path, map_location='cpu')
        baseline_metrics = baseline_checkpoint['test_metrics']
        
        print("\n" + "=" * 70)
        print("MODEL COMPARISON: GNN vs Baseline ANN")
        print("=" * 70)
        print(f"{'Metric':<15} {'Baseline ANN':<15} {'Proposed GNN':<15} {'Improvement':<15}")
        print("-" * 70)
        
        metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        gnn_metrics = [test_acc, test_precision, test_recall, test_f1, test_auc]
        
        for name, gnn_val in zip(metrics_names, gnn_metrics):
            baseline_val = baseline_metrics[name]
            if baseline_val == 0:
                improvement = 0.0 if gnn_val == 0 else 100.0
            else:
                improvement = ((gnn_val - baseline_val) / baseline_val) * 100
            
            print(f"{name.capitalize():<15} {baseline_val:<15.4f} {gnn_val:<15.4f} "
                  f"{'+' if improvement > 0 else ''}{improvement:<14.2f}%")
        
        print("=" * 70)
    else:
        print("⚠️  Baseline results not found. Train baseline first to compare.")
    
    # ============================================================
    # STEP 8: Save results
    # ============================================================
    print("\n💾 STEP 8: Saving model and results...")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'node_feature_dim': model.node_feature_dim,
            'edge_feature_dim': model.edge_feature_dim,
            'hidden_dims': model.hidden_dims,
            'fc_dims': model.fc_dims,
            'num_classes': model.num_classes,
            'dropout': model.dropout,
            'pooling': model.pooling
        },
        'training_history': {
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'train_accs': trainer.train_accs,
            'val_accs': trainer.val_accs
        },
        'test_metrics': {
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'auc': test_auc
        }
    }, os.path.join(save_dir, 'gnn_model.pth'))
    
    print(f"  Model saved to: {save_dir}/gnn_model.pth")
    
    # Save predictions
    results_df = pd.DataFrame({
        'true_label': test_labels,
        'predicted_label': test_preds,
        'predicted_probability': test_probs
    })
    results_df.to_csv(os.path.join(save_dir, 'gnn_predictions.csv'), index=False)
    print(f"  Predictions saved to: {save_dir}/gnn_predictions.csv")
    
    # ============================================================
    # STEP 9: Create visualizations
    # ============================================================
    print("\n📊 STEP 9: Creating visualizations...")
    
    create_training_plots(trainer, save_dir)
    create_evaluation_plots(test_labels, test_probs, test_preds, cm, save_dir)
    
    # Compare with baseline
    if os.path.exists(baseline_path):
        create_comparison_plots(baseline_checkpoint, trainer, save_dir)
    
    print("\n✅ GNN training complete!")
    
    return model, trainer, {
        'test_acc': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_auc': test_auc
    }


def create_training_plots(trainer, save_dir: str):
    """Create training/validation curves for GNN."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Proposed GNN Training Progress', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(trainer.train_losses) + 1)
    
    # Loss curves
    axes[0].plot(epochs, trainer.train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, trainer.val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('Loss Curves', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(epochs, trainer.train_accs, 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, trainer.val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[1].set_title('Accuracy Curves', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gnn_training.png'), dpi=300, bbox_inches='tight')
    print(f"  Training curves saved")
    plt.close()


def create_evaluation_plots(labels, probs, preds, cm, save_dir: str):
    """Create evaluation visualizations for GNN."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Proposed GNN Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0].set_xlabel('False Positive Rate', fontweight='bold')
    axes[0].set_ylabel('True Positive Rate', fontweight='bold')
    axes[0].set_title('ROC Curve', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    axes[1].plot(recall, precision, 'g-', linewidth=2)
    axes[1].set_xlabel('Recall', fontweight='bold')
    axes[1].set_ylabel('Precision', fontweight='bold')
    axes[1].set_title('Precision-Recall Curve', fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    # Confusion Matrix
    im = axes[2].imshow(cm, cmap='Blues', aspect='auto')
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(['Inactive', 'Active'])
    axes[2].set_yticklabels(['Inactive', 'Active'])
    axes[2].set_xlabel('Predicted', fontweight='bold')
    axes[2].set_ylabel('Actual', fontweight='bold')
    axes[2].set_title('Confusion Matrix', fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            text = axes[2].text(j, i, cm[i, j], ha="center", va="center",
                              color="white" if cm[i, j] > cm.max()/2 else "black",
                              fontsize=20, fontweight='bold')
    
    plt.colorbar(im, ax=axes[2])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gnn_evaluation.png'), dpi=300, bbox_inches='tight')
    print(f"  Evaluation plots saved")
    plt.close()


def create_comparison_plots(baseline_checkpoint, gnn_trainer, save_dir: str):
    """Create side-by-side comparison of baseline vs GNN."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison: Baseline ANN vs Proposed GNN', 
                 fontsize=16, fontweight='bold')
    
    baseline_history = baseline_checkpoint['training_history']
    baseline_metrics = baseline_checkpoint['test_metrics']
    
    # Training loss comparison
    ax1 = axes[0, 0]
    epochs_baseline = range(1, len(baseline_history['train_losses']) + 1)
    epochs_gnn = range(1, len(gnn_trainer.train_losses) + 1)
    
    ax1.plot(epochs_baseline, baseline_history['train_losses'], 'r-', 
             label='ANN Train', linewidth=2, alpha=0.7)
    ax1.plot(epochs_gnn, gnn_trainer.train_losses, 'b-', 
             label='GNN Train', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Loss', fontweight='bold')
    ax1.set_title('Training Loss Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Validation accuracy comparison
    ax2 = axes[0, 1]
    ax2.plot(epochs_baseline, baseline_history['val_accs'], 'r-', 
             label='ANN Val', linewidth=2, alpha=0.7)
    ax2.plot(epochs_gnn, gnn_trainer.val_accs, 'b-', 
             label='GNN Val', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Validation Accuracy (%)', fontweight='bold')
    ax2.set_title('Validation Accuracy Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Test metrics comparison (bar chart)
    ax3 = axes[1, 0]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    baseline_vals = [
        baseline_metrics['accuracy'],
        baseline_metrics['precision'],
        baseline_metrics['recall'],
        baseline_metrics['f1'],
        baseline_metrics['auc']
    ]
    
    gnn_checkpoint = torch.load(os.path.join(save_dir, 'gnn_model.pth'), map_location='cpu')
    gnn_vals = [
        gnn_checkpoint['test_metrics']['accuracy'],
        gnn_checkpoint['test_metrics']['precision'],
        gnn_checkpoint['test_metrics']['recall'],
        gnn_checkpoint['test_metrics']['f1'],
        gnn_checkpoint['test_metrics']['auc']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x - width/2, baseline_vals, width, label='Baseline ANN', 
            color='#e74c3c', alpha=0.7)
    ax3.bar(x + width/2, gnn_vals, width, label='Proposed GNN', 
            color='#2ecc71', alpha=0.7)
    
    ax3.set_ylabel('Score', fontweight='bold')
    ax3.set_title('Test Set Performance Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Improvement percentages
    ax4 = axes[1, 1]
    improvements = []
    for gnn, baseline in zip(gnn_vals, baseline_vals):
        if baseline == 0:
            imp = 0.0 if gnn == 0 else 100.0
        else:
            imp = (gnn - baseline) / baseline * 100
        improvements.append(imp)
    
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax4.barh(metrics, improvements, color=colors, alpha=0.7)
    
    ax4.set_xlabel('Improvement (%)', fontweight='bold')
    ax4.set_title('GNN Improvement over Baseline', fontweight='bold')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax4.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2,
                f'{imp:+.1f}%',
                ha='left' if imp > 0 else 'right',
                va='center',
                fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"  Comparison plots saved")
    plt.close()


if __name__ == "__main__":
    # Path to processed graphs
    graph_data_path = "data/processed/molecular_graphs.pkl"
    
    # Check if file exists
    if not os.path.exists(graph_data_path):
        print(f"❌ Error: Processed graphs not found at {graph_data_path}")
        print("\nPlease run:")
        print("  python notebooks/02_molecular_representation.py")
    else:
        # Train GNN
        model, trainer, metrics = train_gnn_model(graph_data_path)
        
        print("\n" + "=" * 70)
        print("GNN TRAINING COMPLETE")
        print("=" * 70)
        print("Next: Step 5 - Training & Evaluation (consolidated results)")
        print("=" * 70)
