# Data directory - place your dataset CSV files here

## Recommended Datasets:

1. **NCI-60**: Download from https://dtp.cancer.gov/
2. **ChEMBL**: Download from https://www.ebi.ac.uk/chembl/
3. **GDSC**: Download from https://www.cancerrxgene.org/

## Expected Format:

Your CSV should have at minimum:
- `SMILES`: Molecular structure string
- `GI50` or `pGI50`: Activity measurement

## Quick Start:

To generate sample data for testing:
```bash
python create_sample_data.py
```
