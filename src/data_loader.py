"""
Data Loader Module for Cancer Drug Screening Dataset
=====================================================

This module handles loading, cleaning, and preprocessing the NCI-60 
cancer drug screening dataset for Graph Neural Network training.

Author: [Your Name]
Project: GNN-Based Drug Candidate Shortlisting for Cancer Treatment
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CancerDrugDataLoader:
    """
    Handles loading and preprocessing of cancer drug screening data.
    
    Attributes:
        data_path (str): Path to the dataset CSV file
        threshold (float): pGI50 threshold for binary classification (default: 6.0)
        df (pd.DataFrame): Loaded and processed dataframe
    """
    
    def __init__(self, data_path: str, threshold: float = 6.0):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV file containing drug screening data
            threshold: pGI50 threshold for active/inactive classification
        """
        self.data_path = data_path
        self.threshold = threshold
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from CSV file.
        
        Returns:
            DataFrame containing the raw dataset
        """
        print(f"Loading dataset from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"Initial dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and invalid entries.
        
        Steps:
        1. Remove rows with missing SMILES strings
        2. Remove rows with missing activity values
        3. Remove duplicate SMILES entries
        4. Filter out invalid SMILES (handled in molecular representation step)
        
        Returns:
            Cleaned DataFrame
        """
        print("\n=== Data Cleaning ===")
        initial_count = len(self.df)
        
        # Remove missing SMILES
        self.df = self.df.dropna(subset=['SMILES'])
        print(f"After removing missing SMILES: {len(self.df)} rows")
        
        # Find activity column (support multiple naming conventions)
        activity_col = None
        for col in ['pGI50', 'GI50', 'GI50_nMol', 'Activity', 'IC50']:
            if col in self.df.columns:
                activity_col = col
                break
        
        if activity_col:
            self.df = self.df.dropna(subset=[activity_col])
            print(f"After removing missing {activity_col}: {len(self.df)} rows")
        
        # Remove duplicates based on SMILES
        before_dedup = len(self.df)
        self.df = self.df.drop_duplicates(subset=['SMILES'], keep='first')
        print(f"Removed {before_dedup - len(self.df)} duplicate SMILES")
        
        print(f"Total rows removed: {initial_count - len(self.df)}")
        print(f"Final cleaned dataset: {len(self.df)} rows\n")
        
        return self.df
    
    def normalize_labels(self) -> pd.DataFrame:
        """
        Normalize activity labels for classification.
        
        Steps:
        1. Convert GI50 to pGI50 if needed: pGI50 = -log10(GI50)
        2. Create binary labels: active (1) if pGI50 > threshold, else inactive (0)
        
        Returns:
            DataFrame with normalized labels
        """
        print("=== Label Normalization ===")
        
        # If labels already exist, use them directly
        if 'label' in self.df.columns:
            print("Using pre-existing labels from dataset.")
            # Create a dummy pGI50 for stats if not present
            if 'pGI50' not in self.df.columns:
                if 'GI50_nMol' in self.df.columns:
                    self.df['GI50'] = self.df['GI50_nMol']
                    self.df['GI50'] = self.df['GI50'].replace(0, 1e-10)
                    self.df['pGI50'] = -np.log10(self.df['GI50'] * 1e-9)  # Convert nM to M
                elif 'GI50' in self.df.columns:
                    self.df['GI50'] = self.df['GI50'].replace(0, 1e-10)
                    self.df['pGI50'] = -np.log10(self.df['GI50'])
                else:
                    # Create placeholder pGI50 based on labels
                    self.df['pGI50'] = self.df['label'].apply(lambda x: 7.0 if x == 1 else 4.0)
        else:
            # Convert GI50 to pGI50 if needed
            if 'pGI50' not in self.df.columns:
                if 'GI50_nMol' in self.df.columns:
                    print("Converting GI50_nMol to pGI50...")
                    self.df['GI50'] = self.df['GI50_nMol']
                    self.df['GI50'] = self.df['GI50'].replace(0, 1e-10)
                    self.df['pGI50'] = -np.log10(self.df['GI50'] * 1e-9)  # Convert nM to M
                elif 'GI50' in self.df.columns:
                    print("Converting GI50 to pGI50...")
                    self.df['GI50'] = self.df['GI50'].replace(0, 1e-10)
                    self.df['pGI50'] = -np.log10(self.df['GI50'])
                print("Conversion complete.")
            
            # Create binary labels
            self.df['label'] = (self.df['pGI50'] > self.threshold).astype(int)
        
        # Calculate class distribution
        n_active = (self.df['label'] == 1).sum()
        n_inactive = (self.df['label'] == 0).sum()
        total = len(self.df)
        
        print(f"\nBinary Classification Threshold: pGI50 > {self.threshold}")
        print(f"Active compounds (label=1): {n_active} ({n_active/total*100:.2f}%)")
        print(f"Inactive compounds (label=0): {n_inactive} ({n_inactive/total*100:.2f}%)")
        
        # Check for class imbalance
        if min(n_active, n_inactive) > 0:
            imbalance_ratio = max(n_active, n_inactive) / min(n_active, n_inactive)
            if imbalance_ratio > 3:
                print(f"Warning: Class imbalance detected (ratio: {imbalance_ratio:.2f}:1)")
                print("Consider using weighted loss or resampling techniques during training.\n")
        
        return self.df
    
    def get_statistics(self) -> dict:
        """
        Generate dataset statistics for reporting.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'total_compounds': len(self.df),
            'unique_smiles': self.df['SMILES'].nunique(),
            'active_compounds': (self.df['label'] == 1).sum(),
            'inactive_compounds': (self.df['label'] == 0).sum(),
            'pGI50_mean': self.df['pGI50'].mean(),
            'pGI50_std': self.df['pGI50'].std(),
            'pGI50_min': self.df['pGI50'].min(),
            'pGI50_max': self.df['pGI50'].max()
        }
        return stats
    
    def display_statistics(self):
        """Print formatted dataset statistics."""
        stats = self.get_statistics()
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Total compounds:        {stats['total_compounds']:,}")
        print(f"Unique SMILES:          {stats['unique_smiles']:,}")
        print(f"Active compounds:       {stats['active_compounds']:,}")
        print(f"Inactive compounds:     {stats['inactive_compounds']:,}")
        print(f"\npGI50 Statistics:")
        print(f"  Mean:                 {stats['pGI50_mean']:.3f}")
        print(f"  Std Dev:              {stats['pGI50_std']:.3f}")
        print(f"  Min:                  {stats['pGI50_min']:.3f}")
        print(f"  Max:                  {stats['pGI50_max']:.3f}")
        print("="*50 + "\n")
    
    def save_processed_data(self, output_path: str):
        """
        Save the processed dataset to CSV.
        
        Args:
            output_path: Path to save the processed CSV file
        """
        self.df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return the processed DataFrame."""
        return self.df


def load_and_preprocess_data(
    data_path: str, 
    threshold: float = 6.0,
    save_path: Optional[str] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Main function to load and preprocess cancer drug screening data.
    
    Args:
        data_path: Path to the raw dataset CSV
        threshold: pGI50 threshold for binary classification
        save_path: Optional path to save processed data
        
    Returns:
        Tuple of (processed_dataframe, statistics_dict)
        
    Example:
        >>> df, stats = load_and_preprocess_data('data/nci60_raw.csv')
        >>> print(f"Loaded {stats['total_compounds']} compounds")
    """
    # Initialize loader
    loader = CancerDrugDataLoader(data_path, threshold)
    
    # Load data
    loader.load_data()
    
    # Clean data
    loader.clean_data()
    
    # Normalize labels
    loader.normalize_labels()
    
    # Display statistics
    loader.display_statistics()
    
    # Save if requested
    if save_path:
        loader.save_processed_data(save_path)
    
    return loader.get_dataframe(), loader.get_statistics()


# Example usage
if __name__ == "__main__":
    # Example: Load sample data
    print("Cancer Drug Dataset Loader - Example Usage\n")
    
    # This is a demonstration - replace with actual data path
    print("To use this module:")
    print("1. Download NCI-60 dataset (or similar cancer drug screening data)")
    print("2. Ensure CSV has columns: SMILES, GI50 (or pGI50)")
    print("3. Run: df, stats = load_and_preprocess_data('path/to/data.csv')")
    print("\nExpected CSV format:")
    print("SMILES,GI50,Cell_Line")
    print("CC(=O)Oc1ccccc1C(=O)O,1.5e-06,MCF7")
    print("CCO,5.0e-05,A549")
