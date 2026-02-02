"""
Dataset Exploration Script
===========================

This script demonstrates how to load and explore the cancer drug screening dataset.
It generates visualizations and statistical summaries.

Run this after downloading the NCI-60 or similar dataset.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_and_preprocess_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def explore_dataset(data_path: str):
    """
    Explore and visualize the cancer drug screening dataset.
    
    Args:
        data_path: Path to the raw CSV dataset
    """
    print("=" * 70)
    print("CANCER DRUG SCREENING DATASET EXPLORATION")
    print("=" * 70 + "\n")
    
    # Load and preprocess data
    df, stats = load_and_preprocess_data(data_path, threshold=6.0)
    
    # Display first few rows
    print("\n📋 Sample Data (first 5 rows):")
    print(df.head())
    
    # Visualizations
    create_visualizations(df)
    
    print("\n✅ Dataset exploration complete!")
    print("Processed data is ready for molecular representation (Step 2)")


def create_visualizations(df: pd.DataFrame):
    """
    Create visualization plots for the dataset.
    
    Args:
        df: Processed DataFrame
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cancer Drug Screening Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Class Distribution
    ax1 = axes[0, 0]
    class_counts = df['label'].value_counts()
    colors = ['#e74c3c', '#2ecc71']  # Red for inactive, green for active
    ax1.bar(['Inactive (0)', 'Active (1)'], class_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Compounds', fontweight='bold')
    ax1.set_title('Class Distribution', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(class_counts.values):
        ax1.text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # 2. pGI50 Distribution
    ax2 = axes[0, 1]
    ax2.hist(df['pGI50'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.axvline(x=6.0, color='red', linestyle='--', linewidth=2, label='Threshold (6.0)')
    ax2.set_xlabel('pGI50 Value', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('pGI50 Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. pGI50 by Class (Box Plot)
    ax3 = axes[1, 0]
    active_pgi50 = df[df['label'] == 1]['pGI50']
    inactive_pgi50 = df[df['label'] == 0]['pGI50']
    bp = ax3.boxplot([inactive_pgi50, active_pgi50], 
                      labels=['Inactive', 'Active'],
                      patch_artist=True,
                      notch=True,
                      showmeans=True)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][1].set_facecolor('#2ecc71')
    ax3.set_ylabel('pGI50 Value', fontweight='bold')
    ax3.set_title('pGI50 Distribution by Class', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. SMILES Length Distribution
    ax4 = axes[1, 1]
    smiles_lengths = df['SMILES'].str.len()
    ax4.hist(smiles_lengths, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('SMILES String Length', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Molecular Complexity (SMILES Length)', fontweight='bold')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'dataset_exploration.png'), dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualizations saved to: {output_dir}/dataset_exploration.png")
    plt.show()


if __name__ == "__main__":
    # Example usage - Replace with your actual data path
    data_path = "data/cancer_drugs.csv"
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found at {data_path}")
        print("\n📥 To get started:")
        print("1. Download NCI-60 dataset from: https://dtp.cancer.gov/")
        print("   OR use ChEMBL cancer drug data")
        print("2. Place CSV file in 'data/' folder")
        print("3. Ensure it has columns: SMILES, GI50 (or pGI50)")
        print("\nFor demonstration, you can create a sample dataset:")
        print("  python create_sample_data.py")
    else:
        explore_dataset(data_path)
