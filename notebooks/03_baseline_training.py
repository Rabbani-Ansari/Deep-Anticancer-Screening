"""
Baseline ANN Training Script
=============================

This notebook trains the baseline ANN model using molecular fingerprints
and evaluates its performance on cancer drug classification.

The ANN serves as a traditional machine learning baseline to compare
against the proposed GNN model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_and_preprocess_data
from src.baseline_ann import (
    MolecularFingerprintGenerator,
    FingerprintDataset,
    BaselineANN,
    ANNTrainer
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import torch
from torch.utils.data import DataLoader
import pickle


def train_baseline_ann(data_path: str, save_dir: str = 'results/models'):
    """
    Train and evaluate the baseline ANN model.
    
    Args:
        data_path: Path to the dataset CSV
        save_dir: Directory to save trained model
    """
    print("=" * 70)
    print("BASELINE ANN MODEL TRAINING")
    print("=" * 70)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # ============================================================
    # STEP 1: Load and preprocess data
    # ============================================================
    print("\n📂 STEP 1: Loading dataset...")
    df, stats = load_and_preprocess_data(data_path, threshold=6.0)
    
    # ============================================================
    # STEP 2: Generate fingerprints
    # ============================================================
    print("\n🔍 STEP 2: Generating molecular fingerprints...")
    fp_generator = MolecularFingerprintGenerator(radius=2, n_bits=2048)
    fingerprints, invalid_smiles = fp_generator.convert_dataset(df, smiles_column='SMILES')
    
    # Remove invalid SMILES from labels
    labels = df['label'].values
    if len(invalid_smiles) > 0:
        valid_indices = [i for i, smiles in enumerate(df['SMILES']) 
                        if smiles not in invalid_smiles]
        labels = labels[valid_indices]
    
    print(f"\nFingerprint statistics:")
    print(f"  Shape: {fingerprints.shape}")
    print(f"  Avg bits set per molecule: {fingerprints.sum(axis=1).mean():.1f}")
    print(f"  Sparsity: {(1 - fingerprints.mean()) * 100:.2f}%")
    
    # ============================================================
    # STEP 3: Train-validation-test split
    # ============================================================
    print("\n📊 STEP 3: Splitting data...")
    
    # First split: train+val vs test (80-20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        fingerprints, labels, 
        test_size=0.2, 
        random_state=42,
        stratify=labels
    )
    
    # Second split: train vs val (80-20 of remaining)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.2,
        random_state=42,
        stratify=y_temp
    )
    
    print(f"  Training set:   {len(y_train):4d} samples ({(y_train==1).sum()} active, {(y_train==0).sum()} inactive)")
    print(f"  Validation set: {len(y_val):4d} samples ({(y_val==1).sum()} active, {(y_val==0).sum()} inactive)")
    print(f"  Test set:       {len(y_test):4d} samples ({(y_test==1).sum()} active, {(y_test==0).sum()} inactive)")
    
    # ============================================================
    # STEP 4: Create PyTorch datasets and dataloaders
    # ============================================================
    print("\n🔄 STEP 4: Creating dataloaders...")
    
    train_dataset = FingerprintDataset(X_train, y_train)
    val_dataset = FingerprintDataset(X_val, y_val)
    test_dataset = FingerprintDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # ============================================================
    # STEP 5: Initialize model and trainer
    # ============================================================
    print("\n🧠 STEP 5: Initializing ANN model...")
    
    model = BaselineANN(
        input_dim=2048,
        hidden_dims=[512, 256, 128],
        dropout=0.3,
        num_classes=2
    )
    
    print(f"  Model architecture:")
    print(f"    Input:  {model.input_dim}")
    for i, dim in enumerate(model.hidden_dims):
        print(f"    Hidden {i+1}: {dim}")
    print(f"    Output: {model.num_classes}")
    print(f"  Total parameters: {model.get_num_params():,}")
    
    trainer = ANNTrainer(
        model=model,
        learning_rate=0.001,
        weight_decay=1e-5
    )
    
    # ============================================================
    # STEP 6: Train the model
    # ============================================================
    print("\n🚀 STEP 6: Training ANN model...\n")
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        verbose=True
    )
    
    # ============================================================
    # STEP 7: Evaluate on test set
    # ============================================================
    print("\n📈 STEP 7: Evaluating on test set...")
    
    # Get predictions
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for fingerprints_batch, labels_batch in test_loader:
            fingerprints_batch = fingerprints_batch.to(trainer.device)
            outputs = model(fingerprints_batch)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (active)
            all_labels.extend(labels_batch.numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, zero_division=0)
    test_recall = recall_score(all_labels, all_preds, zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, zero_division=0)
    test_auc = roc_auc_score(all_labels, all_probs)
    
    print("\n" + "=" * 70)
    print("BASELINE ANN - TEST SET RESULTS")
    print("=" * 70)
    print(f"Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"AUC-ROC:   {test_auc:.4f}")
    print("=" * 70)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"               Inactive  Active")
    print(f"Actual Inactive   {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Active     {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Inactive', 'Active']))
    
    # ============================================================
    # STEP 8: Save results
    # ============================================================
    print("\n💾 STEP 8: Saving model and results...")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.input_dim,
            'hidden_dims': model.hidden_dims,
            'dropout': model.dropout,
            'num_classes': model.num_classes
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
    }, os.path.join(save_dir, 'baseline_ann.pth'))
    
    print(f"  Model saved to: {save_dir}/baseline_ann.pth")
    
    # Save predictions
    results_df = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_preds,
        'predicted_probability': all_probs
    })
    results_df.to_csv(os.path.join(save_dir, 'baseline_ann_predictions.csv'), index=False)
    print(f"  Predictions saved to: {save_dir}/baseline_ann_predictions.csv")
    
    # ============================================================
    # STEP 9: Create visualizations
    # ============================================================
    print("\n📊 STEP 9: Creating visualizations...")
    
    create_training_plots(trainer, save_dir)
    create_evaluation_plots(all_labels, all_probs, all_preds, cm, save_dir)
    
    print("\n✅ Training complete!")
    
    return model, trainer, {
        'test_acc': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_auc': test_auc
    }


def create_training_plots(trainer, save_dir: str):
    """Create training/validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Baseline ANN Training Progress', fontsize=16, fontweight='bold')
    
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
    plt.savefig(os.path.join(save_dir, 'baseline_ann_training.png'), dpi=300, bbox_inches='tight')
    print(f"  Training curves saved")
    plt.close()


def create_evaluation_plots(labels, probs, preds, cm, save_dir: str):
    """Create evaluation visualizations."""
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Baseline ANN Evaluation Metrics', fontsize=16, fontweight='bold')
    
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
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = axes[2].text(j, i, cm[i, j], ha="center", va="center", 
                              color="white" if cm[i, j] > cm.max()/2 else "black",
                              fontsize=20, fontweight='bold')
    
    plt.colorbar(im, ax=axes[2])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'baseline_ann_evaluation.png'), dpi=300, bbox_inches='tight')
    print(f"  Evaluation plots saved")
    plt.close()


if __name__ == "__main__":
    # Path to dataset
    data_path = "data/cancer_drugs.csv"
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found at {data_path}")
        print("\nPlease run:")
        print("  python create_sample_data.py")
    else:
        # Train baseline ANN
        model, trainer, metrics = train_baseline_ann(data_path)
        
        print("\n" + "=" * 70)
        print("BASELINE ANN TRAINING COMPLETE")
        print("=" * 70)
        print("Next: Step 4 - Build and train GNN model for comparison")
        print("=" * 70)
