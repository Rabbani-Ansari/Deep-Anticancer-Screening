"""
Create Sample Cancer Drug Dataset
==================================

This script creates a sample dataset for demonstration purposes.
For real research, download actual NCI-60 or ChEMBL data.
"""

import pandas as pd
import numpy as np
import os


def create_sample_dataset(output_path: str = 'data/cancer_drugs.csv', n_samples: int = 1000):
    """
    Create a sample cancer drug screening dataset.
    
    Args:
        output_path: Path to save the sample CSV
        n_samples: Number of sample compounds to generate
    """
    np.random.seed(42)
    
    # Diverse set of REAL drug molecules (valid SMILES)
    # These are actual or realistic drug-like molecules
    all_smiles = [
        # Simple aromatic
        'c1ccccc1',  # Benzene
        'Cc1ccccc1',  # Toluene
        'c1ccc(O)cc1',  # Phenol
        'c1ccc(N)cc1',  # Aniline
        'c1ccc(C(=O)O)cc1',  # Benzoic acid
        'c1ccc(C(=O)N)cc1',  # Benzamide
        'c1ccc(Cl)cc1',  # Chlorobenzene
        'c1ccc(F)cc1',  # Fluorobenzene
        
        # Naphthalene derivatives
        'c1ccc2ccccc2c1',  # Naphthalene
        'c1ccc2c(c1)ccc(O)c2',  # Naphthol
        
        # Het erosycles
        'c1cccnc1',  # Pyridine
        'c1cnccn1',  # Pyrimidine
        'c1ccoc1',  # Furan
        'c1ccsc1',  # Thiophene
        'c1c[nH]cc1',  # Pyrrole
        'C1=CC=C2C(=C1)C=CN2',  # Indole
        'c1coc2ccccc12',  # Benzofuran
        'c1csc2ccccc12',  # Benzothiophene
        
        # Known anticancer-like drugs
        'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC(C)Cc1ccc(C(C)C(=O)O)cc1',  # Ibuprofen
        'COc1ccc(CCN)cc1OC',  # MDMA base (for diversity)
        'CC(C)NCC(O)c1ccc(O)c(O)c1',  # Isoproterenol
        
        # More complex scaffolds
        'c1ccc2c(c1)CCC2',  # Tetralin
        'c1ccc(cc1)C(c2ccccc2)O',  # Benzhydrol
        'c1ccc2c(c1)c(c3ccccc23)O',  # Anthrol
        'c1ccc(cc1)CCN',  # Phenethylamine
        'c1ccc(cc1)CCC(=O)O',  # Hydrocinnamic acid
        'c1cc(c(cc1)O)O',  # Catechol
        'c1ccc(c(c1)O)O',  # Resorcinol
        'c1cc(O)ccc1O',  # Hydroquinone
        
        # Fluorinated aromatics
        'c1cc(F)c(F)cc1',  # Difluorobenzene
        'c1cc(F)c(F)c(F)c1',  # Trifluorobenzene
        'FC(F)(F)c1ccccc1',  # Trifluorotoluene
        
        # Nitrogenous compounds
        'c1ccc(cc1)N(C)C',  # Dimethylaniline
        'c1ccc(cc1)NC(=O)C',  # Acetanilide
        'c1ccc2c(c1)nccc2',  # Quinoline
        'c1ccc2c(c1)ncc(c2)C',  # Methylquinoline
        
        # Amines and alcohols
        'CCO',  # Ethanol
        'CCCO',  # Propanol
        'CCCCO',  # Butanol
        'CC(C)O',  # Isopropanol
        'CCN',  # Ethylamine
        'CCCN',  # Propylamine
        
        # Carboxylic acids
        'CC(=O)O',  # Acetic acid
        'CCC(=O)O',  # Propionic acid
        'CCCC(=O)O',  # Butyric acid
        
        # Esters
        'CC(=O)OC',  # Methyl acetate
        'CCOC(=O)C',  # Ethyl acetate
        
        # More drug-like molecules
        'c1ccc(cc1)C(c2ccccc2)(c3ccccc3)O',  # Triphenylmethanol
        'c1ccc2c(c1)cc(cc2)O',  # 2-Naphthol
        'c1ccc(cc1)c2ccccc2',  # Biphenyl
        'c1ccc(cc1)Oc2ccccc2',  # Diphenyl ether
        'c1cnc2c(c1)cccn2',  # Imidazopyridine
        
        # Add more variations
        'Cc1ccc(C)cc1',  # Xylene
        'Cc1ccc(C(=O)O)cc1',  # Toluic acid
        'Nc1ccc(N)cc1',  # Diaminobenzene
        'Clc1ccc(Cl)cc1',  # Dichlorobenzene
        'COc1ccccc1',  # Anisole
        'CCOc1ccccc1',  # Phenetole
        
        # Complex scaffolds
        'C1CCC2C(C1)CCC3C2CCC3',  # Perhydroanthracene
        'c1ccc2c(c1)Cc3ccccc3C2',  # 9,10-Dihydroanthracene
        'c1ccc2c(c1)CC(Cc2)O',  # Tetralin-ol
        
        # Pyridine derivatives
        'Cc1ccncc1',  # Methylpyridine
        'c1cc(C)ncc1',  # Another methylpyridine
        'c1cc(N)ncc1',  # Aminopyridine
        'c1cc(O)ncc1',  # Hydroxypyridine
        
        # Indole derivatives
        'Cc1c[nH]c2ccccc12',  # Methylindole
        'c1cc2c(c[nH1]c2cc1)CC',  # Ethylindole
        
        # Benzimidazole derivatives
        'c1ccc2c(c1)nc[nH]2',  # Benzimidazole
        'Cc1nc2ccccc2[nH]1',  # Methylbenzimidazole
        
        # Benzothiazole derivatives
        'c1ccc2c(c1)ncs2',  # Benzothiazole
        'Cc1nc2ccccc2s1',  # Methylbenzothiazole
        
        # Quinazoline derivatives
        'c1ccc2c(c1)ncnc2',  # Quinazoline
        'Cc1cnc2ccccc2n1',  # Methylquinazoline
        
        # Isoquinoline derivatives
        'c1ccc2c(c1)cncc2',  # Isoquinoline
        'Cc1cnc2ccccc2c1',  # Methylisoquinoline
    ]
    
    # Make sure we have enough molecules
    if len(all_smiles) < n_samples:
        # Simple variations by adding methyl groups
        base_molecules = all_smiles.copy()
        while len(all_smiles) < n_samples:
            base = np.random.choice(base_molecules)
            # Try to add a methyl group
            if 'c1ccccc1' in base and np.random.random() > 0.5:
                variant = base.replace('c1ccccc1', 'Cc1ccccc1', 1)
                if variant not in all_smiles:
                    all_smiles.append(variant)
            else:
                all_smiles.append(base)  # Duplicate if modification fails
    
    # Select exactly n_samples molecules
    if len(all_smiles) > n_samples:
        smiles_list = np.random.choice(all_smiles, size=n_samples, replace=False).tolist()
    else:
        smiles_list = all_smiles[:n_samples]
    
    # Generate GI50 values (in Molar concentration)
    # Active compounds: GI50 ~ 1e-8 to 1e-6 M (pGI50 ~ 6-8)
    # Inactive compounds: GI50 ~ 1e-5 to 1e-3 M (pGI50 ~ 3-5)
    
    n_active = int(n_samples * 0.4)  # 40% active
    n_inactive = n_samples - n_active
    
    gi50_values = np.concatenate([
        10 ** np.random.uniform(-8, -6, n_active),  # Active
        10 ** np.random.uniform(-5, -3, n_inactive)  # Inactive
    ])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    gi50_values = gi50_values[indices]
    smiles_list = [smiles_list[i] for i in indices]
    
    # Calculate pGI50
    pgi50_values = -np.log10(gi50_values)
    
    # Synthetic compound IDs
    nsc_ids = [f'NSC_{100000 + i}' for i in range(n_samples)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'NSC': nsc_ids,
        'SMILES': smiles_list,
        'GI50': gi50_values,
        'pGI50': pgi50_values,
        'Cell_Line': np.random.choice(['MCF7', 'A549', 'HCT-116', 'PC-3', 'U-87'], n_samples),
        'Tissue_Type': np.random.choice(['Breast', 'Lung', 'Colon', 'Prostate', 'Brain'], n_samples)
    })
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("=" * 60)
    print("SAMPLE DATASET CREATED")
    print("=" * 60)
    print(f"Output path: {output_path}")
    print(f"Number of compounds: {n_samples}")
    print(f"Unique SMILES: {df['SMILES'].nunique()}")
    print(f"Columns: {list(df.columns)}")
    print(f"Active compounds (pGI50 > 6): {(df['pGI50'] > 6).sum()}")
    print(f"Inactive compounds (pGI50 <= 6): {(df['pGI50'] <= 6).sum()}")
    print("\n⚠️ NOTE: This is SYNTHETIC DATA for demonstration only!")
    print("For real research, use actual datasets like NCI-60 or ChEMBL.")
    print("=" * 60)
    
    # Display sample
    print("\nSample data (first 5 rows):") 
    print(df.head())
    
    return df


if __name__ == "__main__":
    create_sample_dataset()
