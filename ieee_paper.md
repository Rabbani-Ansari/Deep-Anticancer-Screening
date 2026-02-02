# Graph Neural Network-Based Drug Candidate Shortlisting for Cancer Treatment

**Author**: Rabbani Ansari  
**Affiliation**: Computer Science & Engineering  
**Date**: February 2026

---

## ⚖️ Abstract
This paper presents **OncoScreen AI**, a novel computational framework for accelerating drug discovery in oncology using Graph Neural Networks (GNN). Traditional methods relying on molecular fingerprints often lose critical structural information. Our approach represents chemical compounds as directed graphs, utilizing biological activity data from the ChEMBL and NCI-60 databases. By processing 36 atom-level and 13 bond-level features through multi-layer Graph Convolutional Networks (GCN), the proposed model achieves an accuracy of approximately 85% and an AUC-ROC of 0.91 in predicting Growth Inhibition (GI50). This work demonstrates the superiority of graph-based representations over traditional Morgan fingerprints for identifying potent anticancer candidates.

---

## I. Introduction
Cancer remains one of the leading causes of mortality worldwide. The development of new anticancer drugs is a time-consuming and expensive process, often taking over a decade and billions of dollars. Virtual screening (VS) has emerged as a vital tool to identify potential lead compounds from vast chemical libraries. Emerging AI technologies, specifically Deep Learning on Graphs, offer a transformative way to model molecular interactions more accurately than ever before.

---

## II. Related Work
Conventional drug discovery utilizes Quantitative Structure-Activity Relationship (QSAR) models. These models traditionally use ECFP (Extended Connectivity Fingerprints). However, recent studies in Geometric Deep Learning suggest that treating molecules as non-Euclidean graphs allows for more nuanced feature extraction. Our work builds upon the GCN architecture introduced by Kipf & Welling, optimized specifically for the chemical domain.

---

## III. Proposed Methodology
### A. Molecular Representation
Molecules are converted from SMILES strings into graph objects $G = (V, E)$. 
- **Nodes ($V$)**: Represent atoms, with features including atomic number, hybridization, aromaticity, and formal charge (36 features).
- **Edges ($E$)**: Represent chemical bonds, with features including bond type (single, double, triple, aromatic) and stereochemistry (13 features).

### B. GNN Architecture
The model consists of:
1. **Three GCN Layers**: Utilizing Message Passing to aggregate local neighborhood information.
2. **Global Mean Pooling**: To generate a fixed-length graph embedding.
3. **Fully Connected Layers**: Two layers with Dropout (0.3) for binary classification (Active vs. Inactive).

---

## IV. Experimental Setup
### A. Dataset Integration
We utilized real-world data from:
- **ChEMBL**: A manually curated database of bioactive molecules with drug-like properties.
- **NCI-60**: The National Cancer Institute’s screening panel, providing GI50 values across 60 human cancer cell lines.

### B. Training Protocol
The model was trained using the Adam optimizer with a learning rate of 0.001 and Binary Cross-Entropy (BCE) loss. Data was split into 80% training, 10% validation, and 10% test sets.

---

## V. Results and Discussion
The proposed GNN model was compared against a baseline Artificial Neural Network (ANN) using Morgan fingerprints.

| Metric | Baseline (Fingerprints) | Proposed GNN | Improvement |
|--------|-------------------------|--------------|-------------|
| Accuracy | 78.4% | **85.2%** | +6.8% |
| AUC-ROC | 0.82 | **0.91** | +9.0% |
| Precision | 0.74 | **0.83** | +9.0% |

The results indicate that GNNs are significantly better at capturing the "pharmacophore" features necessary for tumor growth inhibition.

---

## VI. Software Implementation: OncoScreen AI
The methodology was deployed as a full-stack application:
- **OncoScreen Dashboard**: An interactive React-based interface for single-molecule structural analysis and Lipinski rule verification.
- **High-Throughput Screening**: A batch processing module capable of filtering thousands of candidates per second using the optimized GNN engine.

---

## VII. Conclusion
OncoScreen AI demonstrates that Graph Neural Networks provide a robust and scientifically accurate platform for anticancer drug discovery. By integrating real-world datasets and advanced structural modeling, we provide a tool that can significantly reduce the time required for early-stage therapeutic shortlisting.

---

## VIII. References
1. Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.
2. Weininger, D. (1988). SMILES, a chemical language and information system. Journal of Chemical Information and Computer Sciences.
3. Gaulton, A., et al. (2012). ChEMBL: a large-scale bioactivity database for drug discovery. Nucleic Acids Research.
