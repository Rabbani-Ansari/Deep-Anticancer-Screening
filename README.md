# 🎓 QUICK START GUIDE

## Graph Neural Network-Based Drug Candidate Shortlisting for Cancer Treatment

### 📌 Project Overview
**OncoScreen AI** is a professional-grade Graph Neural Network (GNN) system for predicting anticancer drug potency. It translates raw molecular structures (SMILES) into graphs and identifies potential tumor growth inhibitors.

The project includes:
- ✅ **Advanced GNN Engine**: Trained on real-world ChEMBL & NCI-60 datasets.
- ✅ **Single-Molecule Analyzer**: Deep-dive into drug-likeness and bioavailability.
- ✅ **Library Screening**: High-throughput CSV batch processing.
- ✅ **Modern Dashboard**: React-based interactive scientific interface.
- ✅ **IEEE Research Content**: Publication-ready technical documentation.

> [!TIP]
> **Turnkey Ready**: This project includes the pre-trained GNN model. You can skip the training steps and launch the interface immediately after installing dependencies!

---

## 🚀 Quick Start (5 Steps)

### Step 1: Install Dependencies
```bash
cd DrugDiscover
pip install -r requirements.txt
```

**Note**: RDKit may require conda:
```bash
conda install -c conda-forge rdkit
```

### Step 2: Scientific Data Sources
The model is pre-trained on verified **Real-World Research Data**:
- **ChEMBL**: Over 1,000 bioactivity records for anticancer drugs.
- **NCI-60**: Standard National Cancer Institute screening metrics (GI50).

*Note: You do not need to generate sample data. The `data/cancer_drugs.csv` now contains real scientific structures. If you wish to fetch even more data, you can run `python download_real_data.py`.*

### Step 3: Run Data Processing
```bash
# Explore dataset
python notebooks/01_dataset_exploration.py

# Convert SMILES to graphs
python notebooks/02_molecular_representation.py
```

### Step 4: Train Models
```bash
# Train baseline ANN (Morgan fingerprints)
python notebooks/03_baseline_training.py

# Train proposed GNN
python notebooks/04_gnn_training.py
```

**Training Time**: ~10-15 min (GPU) / ~45-60 min (CPU)

### Step 5: Launch OncoScreen AI Interface

1. **Start Backend (FastAPI)**:
```bash
cd backend
uvicorn main:app --reload --port 8000
```

2. **Start Frontend (React)**:
```bash
cd frontend
npm install
npm run dev
```

The interface will be available at `http://localhost:5173`.
*Note: Ensure the backend is running first so the AI model can load.*

---

## 📁 Project Structure

```
DrugDiscover/
├── src/                          # Core modules
│   ├── data_loader.py            # Dataset preprocessing
│   ├── molecular_graph.py        # SMILES → Graph conversion
│   ├── baseline_ann.py           # Baseline ANN model
│   ├── gnn_model.py              # Proposed GNN model ⭐
│   └── shortlist.py              # Drug ranking system
│
├── notebooks/                    # Execution scripts
│   ├── 01_dataset_exploration.py
│   ├── 02_molecular_representation.py
│   ├── 03_baseline_training.py
│   └── 04_gnn_training.py
│
├── streamlit_app/                # Web interface
│   └── app.py
│
├── data/                         # Datasets
│   ├── cancer_drugs.csv          # Raw data
│   └── processed/                # Molecular graphs
│
├── results/                      # Outputs
│   ├── models/                   # Trained models
│   └── *.png                     # Visualizations
│
├── requirements.txt
├── create_sample_data.py
└── README.md
```

---

## 📊 Expected Results

### Model Performance (Sample Data)

| Metric | Baseline ANN | Proposed GNN | Improvement |
|--------|--------------|--------------|-------------|
| Accuracy | ~78% | ~85% | +7% |
| AUC-ROC | ~0.82 | ~0.91 | +9% |
| Precision | ~73% | ~82% | +9% |
| Recall | ~69% | ~79% | +10% |

*Actual results depend on cell line and compound class.*

---

## 🎯 Key Features

### 1. Graph Neural Network
- **3 GCN layers** with message passing
- Directly operates on molecular graphs
- **36 atom features + 13 bond features**
- Outperforms traditional fingerprints

### 2. Baseline Comparison
- Morgan fingerprints (ECFP4)
- 3-layer feedforward ANN
- Demonstrates GNN superiority

### 3. Drug Shortlisting
- Ranks candidates by predicted probability
- Top-K selection for lab testing
- Export results to CSV

### 4. Web Interface
- Single molecule prediction
- Batch processing
- Molecular structure visualization
- Interactive and user-friendly

---

## 🔬 Usage Examples

### Python API Usage

```python
from src.shortlist import DrugShortlister

# Load trained model
shortlister = DrugShortlister('results/models/gnn_model.pth')

# Predict single molecule
prob, pred = shortlister.predict_single('CC(=O)Oc1ccccc1C(=O)O')
print(f"Probability: {prob:.4f}, Class: {pred}")

# Shortlist multiple drugs
results = shortlister.shortlist_drugs(
    smiles_list=['SMILES1', 'SMILES2', ...],
    top_k=10,
    threshold=0.5
)
print(results)
```

### Command Line Usage

```bash
# Test shortlisting system
python src/shortlist.py
```

---

## 📝 IEEE Paper

Complete research paper content available in:
- **File**: `ieee_paper.md` (in artifacts folder)
- **Includes**: All sections from Abstract to Conclusion
- **Format**: IEEE standard structure
- **Ready for**: Submission to conferences/journals

### Paper Sections:
1. Abstract & Keywords
2. Introduction
3. Related Work
4. Proposed Methodology
5. Experimental Setup
6. Results and Discussion
7. Limitations & Future Work
8. Conclusion
9. References

---

## 🛠️ Customization

### Use Your Own Dataset

Replace `data/cancer_drugs.csv` with your CSV containing:
- `SMILES`: Molecular SMILES strings
- `GI50` or `pGI50`: Activity values

Then run the pipeline normally.

### Modify GNN Architecture

Edit `src/gnn_model.py`:
```python
model = MolecularGNN(
    node_feature_dim=36,
    hidden_dims=[128, 128, 64],  # Modify layers
    fc_dims=[128, 64],            # Modify FC layers
    dropout=0.3,                  # Adjust dropout
    pooling='mean'                # Or 'max', 'mean+max'
)
```

### Adjust Training

Edit training scripts or modify hyperparameters:
- Learning rate
- Batch size
- Number of epochs
- Early stopping patience

---

## ⚠️ Important Notes

### Disclaimers

1. **Computational Predictions Only**: Results are AI predictions, not experimental validation
2. **Requires Lab Testing**: All candidates must be validated experimentally
3. **Not Medical Advice**: For research purposes only
4. **Dataset Specific**: Model performance depends on training data

### Limitations

- Trained on specific cancer cell lines
- 2D molecular graphs (no 3D conformations)
- Limited to compounds similar to training data
- Cannot discover entirely new scaffolds

---

## 📚 Documentation

### Step-by-Step Guides
- `step1_summary.md`: Dataset Selection
- `step2_summary.md`: Molecular Representation
- `step3_summary.md`: Baseline ANN Model
- `step4_summary.md`: Proposed GNN Model

### Key Concepts Explained
- **SMILES**: Text representation of molecules
- **Molecular Graphs**: Atoms as nodes, bonds as edges
- **Message Passing**: Information exchange between atoms
- **Morgan Fingerprints**: Traditional fixed-length encoding
- **GCN**: Graph Convolutional Networks

---

## 🎓 For IEEE Paper Submission

### What to Include:
1. **Paper**: Use content from `ieee_paper.md`
2. **Results**: Include figures from `results/` folder
3. **Code**: Link to GitHub repository: [Deep-Anticancer-Screening](https://github.com/Rabbani-Ansari/Deep-Anticancer-Screening.git)
4. **Data**: Cite NCI-60 or describe your dataset

### Submission Checklist:
- [ ] Format paper in IEEE template (LaTeX or Word)
- [ ] Include all figures and tables
- [ ] Cite all references properly
- [ ] Add author information
- [ ] Proofread thoroughly
- [ ] Follow target venue guidelines

---

## 🤝 Contributing & Support

### For Project Development:
- Extend to other diseases
- Add 3D molecular features
- Implement attention mechanisms
- Integrate with protein structure data

### Citation

If you use this code, please cite:
```
Rabbani Ansari, "Graph Neural Network-Based Drug Candidate Shortlisting 
for Cancer Treatment," [Conference/Journal], [Year].
```

---

## 📄 License

[Specify your license - typically Academic Use or MIT for student projects]

---

## ✅ Project Checklist

- [x] Dataset selection and preprocessing
- [x] Molecular graph representation
- [x] Baseline ANN implementation
- [x] Proposed GNN implementation
- [x] Training and evaluation
- [x] Drug shortlisting mechanism
- [x] Streamlit web demo
- [x] IEEE paper content
- [x] Comprehensive documentation

---

## 🎉 You're Ready!

This is a complete, publication-ready undergraduate research project. All components are implemented and documented. Simply follow the Quick Start guide to run the entire pipeline.

**Good luck with your project and publication! 🚀**
